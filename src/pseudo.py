"""Pseudo-ID mining via mutual k-NN graph for self-supervised face recognition."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.data import ParquetTwoViewDataset
    from src.model import MoCo

logger = logging.getLogger(__name__)


@dataclass
class PseudoIDState:
    """State of pseudo-ID assignments."""

    image_to_cluster: np.ndarray  # [N] int32, -1 = unclustered
    cluster_to_images: dict[int, np.ndarray]  # cluster_id -> image indices
    num_images: int = 0
    num_clusters: int = 0
    num_clustered: int = 0
    last_refresh_epoch: int = -1
    sim_threshold_used: float = 0.0

    def get_cluster(self, image_idx: int) -> int:
        """Get cluster ID for image, or -1 if unclustered."""
        if image_idx < 0 or image_idx >= len(self.image_to_cluster):
            return -1
        return int(self.image_to_cluster[image_idx])

    def sample_partner(self, image_idx: int, rng: np.random.Generator) -> int | None:
        """Sample a different image from the same cluster."""
        cluster_id = self.get_cluster(image_idx)
        if cluster_id < 0:
            return None
        images = self.cluster_to_images.get(cluster_id)
        if images is None or len(images) < 2:
            return None
        # Rejection sampling (fast for small clusters)
        for _ in range(10):
            sampled = rng.choice(images)
            if sampled != image_idx:
                return int(sampled)
        # Fallback: filter and sample
        candidates = images[images != image_idx]
        if len(candidates) == 0:
            return None
        return int(rng.choice(candidates))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for checkpoint."""
        return {
            "image_to_cluster": self.image_to_cluster,
            "cluster_sizes": {k: len(v) for k, v in self.cluster_to_images.items()},
            "num_images": self.num_images,
            "num_clusters": self.num_clusters,
            "num_clustered": self.num_clustered,
            "last_refresh_epoch": self.last_refresh_epoch,
            "sim_threshold_used": self.sim_threshold_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PseudoIDState:
        """Deserialize from checkpoint."""
        image_to_cluster = data["image_to_cluster"]
        # Rebuild cluster_to_images from image_to_cluster
        cluster_to_images: dict[int, list[int]] = {}
        for idx, cid in enumerate(image_to_cluster):
            if cid >= 0:
                cluster_to_images.setdefault(int(cid), []).append(idx)
        return cls(
            image_to_cluster=image_to_cluster,
            cluster_to_images={k: np.array(v, dtype=np.int32) for k, v in cluster_to_images.items()},
            num_images=data["num_images"],
            num_clusters=data["num_clusters"],
            num_clustered=data["num_clustered"],
            last_refresh_epoch=data["last_refresh_epoch"],
            sim_threshold_used=data["sim_threshold_used"],
        )


@dataclass
class PseudoIDManager:
    """Manages pseudo-ID mining and state."""

    knn_k: int = 20
    mutual_topk: int = 5
    min_cluster_size: int = 2
    max_cluster_size: int = 50
    state: PseudoIDState | None = None
    _image_bytes_cache: dict[int, bytes] = field(default_factory=dict)

    @property
    def num_clusters(self) -> int:
        return self.state.num_clusters if self.state else 0

    @property
    def last_refresh_epoch(self) -> int:
        return self.state.last_refresh_epoch if self.state else -1

    def clear(self) -> None:
        """Clear pseudo-ID state."""
        self.state = None
        self._image_bytes_cache.clear()

    def refresh(
        self,
        model: MoCo,
        datasets: list[ParquetTwoViewDataset],
        epoch: int,
        sim_threshold: float,
        device: torch.device,
        batch_size: int = 256,
        num_workers: int = 2,
    ) -> dict[str, Any]:
        """Run pseudo-ID mining and update state."""
        start_time = time.perf_counter()

        # Step 1: Embed all images
        logger.info("Pseudo-ID mining: embedding all training images...")
        embeddings, image_bytes_list = self._embed_all(
            model, datasets, device, batch_size, num_workers
        )
        n_images = len(embeddings)
        logger.info(f"  Embedded {n_images:,} images")

        # Step 2: Build kNN graph with FAISS
        logger.info(f"Pseudo-ID mining: building kNN graph (k={self.knn_k})...")
        knn_sims, knn_indices = self._build_knn_graph(embeddings)

        # Step 3: Filter to mutual edges
        logger.info(
            f"Pseudo-ID mining: filtering to mutual top-{self.mutual_topk} "
            f"(threshold={sim_threshold:.3f})..."
        )
        edges = self._filter_mutual_edges(knn_indices, knn_sims, sim_threshold)
        logger.info(f"  Found {len(edges):,} mutual edges")

        # Step 4: Find connected components
        logger.info("Pseudo-ID mining: finding connected components...")
        cluster_ids = self._find_components(edges, n_images)

        # Step 5: Filter by cluster size
        logger.info(
            f"Pseudo-ID mining: filtering clusters "
            f"(size {self.min_cluster_size}-{self.max_cluster_size})..."
        )
        cluster_ids, cluster_to_images = self._filter_clusters(cluster_ids)

        # Build state
        num_clustered = int((cluster_ids >= 0).sum())
        self.state = PseudoIDState(
            image_to_cluster=cluster_ids,
            cluster_to_images=cluster_to_images,
            num_images=n_images,
            num_clusters=len(cluster_to_images),
            num_clustered=num_clustered,
            last_refresh_epoch=epoch,
            sim_threshold_used=sim_threshold,
        )

        # Cache image bytes for cross-image sampling
        self._image_bytes_cache = {i: b for i, b in enumerate(image_bytes_list)}

        elapsed = time.perf_counter() - start_time
        accept_rate = num_clustered / n_images if n_images > 0 else 0.0
        avg_size = num_clustered / len(cluster_to_images) if cluster_to_images else 0.0
        max_size = max((len(v) for v in cluster_to_images.values()), default=0)

        stats = {
            "pseudo/cluster_count": len(cluster_to_images),
            "pseudo/images_clustered": num_clustered,
            "pseudo/accept_rate": accept_rate,
            "pseudo/avg_cluster_size": avg_size,
            "pseudo/max_cluster_size": max_size,
            "pseudo/sim_threshold": sim_threshold,
            "pseudo/refresh_time_sec": elapsed,
            "pseudo/refresh_epoch": epoch,
        }

        logger.info(
            f"Pseudo-ID mining complete: {len(cluster_to_images):,} clusters, "
            f"{num_clustered:,} images ({accept_rate:.1%}), "
            f"avg size {avg_size:.1f}, took {elapsed:.1f}s"
        )

        return stats

    def get_image_bytes(self, image_idx: int) -> bytes | None:
        """Get cached image bytes for cross-image pair sampling."""
        return self._image_bytes_cache.get(image_idx)

    def _embed_all(
        self,
        model: MoCo,
        datasets: list[ParquetTwoViewDataset],
        device: torch.device,
        batch_size: int,
        num_workers: int,
    ) -> tuple[np.ndarray, list[bytes]]:
        """Embed all images using momentum encoder (no augmentation)."""
        from src.model import l2_normalize

        model.eval()
        all_embeddings: list[torch.Tensor] = []
        all_image_bytes: list[bytes] = []

        for ds in datasets:
            # Create simple loader without transforms
            simple_ds = _SimpleImageDataset(ds)
            loader = DataLoader(
                simple_ds,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )

            with torch.no_grad():
                for images, image_bytes_batch in loader:
                    images = images.to(device, non_blocking=True)
                    # Use momentum encoder (more stable)
                    emb = model.backbone_k(images)
                    emb = l2_normalize(emb, dim=1)
                    all_embeddings.append(emb.cpu())
                    all_image_bytes.extend(image_bytes_batch)

        model.train()
        embeddings = torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)
        return embeddings, all_image_bytes

    def _build_knn_graph(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build kNN graph using FAISS."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not available, using sklearn (slower)")
            return self._build_knn_graph_sklearn(embeddings)

        n, d = embeddings.shape
        # IndexFlatIP on L2-normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(d)

        # Try GPU if available
        try:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 1024)  # 1GB temp
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("  Using FAISS GPU index")
        except Exception:
            logger.info("  Using FAISS CPU index")

        index.add(embeddings)

        # Search k+1 (first is self)
        k = self.knn_k + 1
        sims, indices = index.search(embeddings, k)

        # Remove self-matches
        mask = indices != np.arange(n)[:, None]
        # Take first k neighbors after removing self
        knn_sims = np.zeros((n, self.knn_k), dtype=np.float32)
        knn_indices = np.zeros((n, self.knn_k), dtype=np.int32)
        for i in range(n):
            valid = np.where(mask[i])[0][: self.knn_k]
            knn_sims[i, : len(valid)] = sims[i, valid]
            knn_indices[i, : len(valid)] = indices[i, valid]

        return knn_sims, knn_indices

    def _build_knn_graph_sklearn(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback kNN using sklearn."""
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=self.knn_k + 1, metric="cosine")
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)

        # Convert distance to similarity (cosine distance = 1 - similarity)
        sims = 1.0 - distances

        # Remove self (first column)
        return sims[:, 1:].astype(np.float32), indices[:, 1:].astype(np.int32)

    def _filter_mutual_edges(
        self,
        knn_indices: np.ndarray,
        knn_sims: np.ndarray,
        sim_threshold: float,
    ) -> list[tuple[int, int]]:
        """Keep only mutual top-k edges above threshold."""
        from scipy.sparse import csr_matrix

        n = knn_indices.shape[0]
        top_k = self.mutual_topk

        # Build sparse adjacency for "j in i's top-mutual_topk AND sim >= threshold"
        top_neighbors = knn_indices[:, :top_k]
        top_sims = knn_sims[:, :top_k]
        mask = top_sims >= sim_threshold

        rows = np.repeat(np.arange(n), top_k)[mask.ravel()]
        cols = top_neighbors.ravel()[mask.ravel()]
        data = top_sims.ravel()[mask.ravel()]

        if len(rows) == 0:
            return []

        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        adj_t = adj.T.tocsr()

        # Mutual: both A[i,j] > 0 and A[j,i] > 0
        mutual = adj.multiply(adj_t.astype(bool))
        mutual_coo = mutual.tocoo()

        edges = []
        for i, j in zip(mutual_coo.row, mutual_coo.col, strict=True):
            if i < j:  # Avoid duplicates
                edges.append((int(i), int(j)))

        return edges

    def _find_components(
        self, edges: list[tuple[int, int]], n: int
    ) -> np.ndarray:
        """Find connected components using scipy."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        if not edges:
            return np.full(n, -1, dtype=np.int32)

        # FIX: Simplified symmetric edge construction
        edge_array = np.array(edges, dtype=np.int32)  # Shape: (E, 2)
        rows = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
        cols = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
        data = np.ones(len(rows), dtype=np.int8)

        adj = csr_matrix((data, (rows, cols)), shape=(n, n))
        _, labels = connected_components(adj, directed=False)

        return labels.astype(np.int32)

    def _filter_clusters(
        self, cluster_ids: np.ndarray
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Filter clusters by size, return remapped IDs and cluster->images."""
        from collections import Counter

        counts = Counter(cluster_ids[cluster_ids >= 0])
        valid_clusters = {
            cid
            for cid, count in counts.items()
            if self.min_cluster_size <= count <= self.max_cluster_size
        }

        # Remap to consecutive IDs
        new_ids = np.full_like(cluster_ids, -1)
        cluster_to_images: dict[int, list[int]] = {}
        new_cid = 0

        for old_cid in sorted(valid_clusters):
            indices = np.where(cluster_ids == old_cid)[0]
            new_ids[indices] = new_cid
            cluster_to_images[new_cid] = indices.astype(np.int32)
            new_cid += 1

        # Convert lists to arrays
        cluster_to_images_arr = {k: np.array(v, dtype=np.int32) for k, v in cluster_to_images.items()}

        rejected_small = sum(1 for c, cnt in counts.items() if cnt < self.min_cluster_size)
        rejected_large = sum(1 for c, cnt in counts.items() if cnt > self.max_cluster_size)
        logger.info(
            f"  Valid clusters: {len(cluster_to_images)}, "
            f"rejected (small): {rejected_small}, rejected (large): {rejected_large}"
        )

        return new_ids, cluster_to_images_arr


class _SimpleImageDataset(torch.utils.data.Dataset):
    """Simple dataset that returns decoded images without augmentation."""

    def __init__(
        self, parquet_dataset: ParquetTwoViewDataset, input_size: tuple[int, int] = (112, 112)
    ):
        from torchvision import transforms

        self.ds = parquet_dataset
        self.input_size = input_size
        # Pre-load all rows for random access
        self._rows: list[tuple[bytes, str]] = []

        # Create transform once in __init__ (not per-sample)
        self._transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self._load_all()

    def _load_all(self) -> None:
        """Load all image bytes from parquet files."""
        import pyarrow.parquet as pq

        from src.data.file_utils import list_parquet_files

        # FIX: Use parquet_glob not glob_pattern
        for path in list_parquet_files(self.ds.parquet_glob):
            table = pq.read_table(path, columns=["image_bytes", "identity_id"])
            for i in range(table.num_rows):
                identity_id = table["identity_id"][i].as_py()
                if self.ds.allowed_identities is None or identity_id in self.ds.allowed_identities:
                    image_bytes = table["image_bytes"][i].as_py()
                    self._rows.append((image_bytes, identity_id))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, bytes]:
        import io

        from PIL import Image

        image_bytes, _ = self._rows[idx]

        # Decode and resize (no augmentation)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._transform(img)

        return tensor, image_bytes
