"""Pseudo-ID mining via mutual k-NN graph for self-supervised face recognition."""

from __future__ import annotations

import io
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

if TYPE_CHECKING:
    from src.data import ParquetTwoViewDataset
    from src.model import MoCo

from src.data.binary_dataset import BinaryImageDataset

logger = logging.getLogger(__name__)


@dataclass
class PseudoIDState:
    """Pseudo-ID cluster assignments for all training images."""
    image_to_cluster: np.ndarray  # [N] int32, -1 = unclustered
    cluster_to_images: dict[int, np.ndarray]  # cluster_id -> array of image indices
    num_images: int = 0
    num_clusters: int = 0
    num_clustered: int = 0
    last_refresh_epoch: int = -1
    sim_threshold_used: float = 0.0

    def get_cluster(self, image_idx: int) -> int:
        """Get cluster ID for an image, or -1 if unclustered."""
        if image_idx < 0 or image_idx >= len(self.image_to_cluster):
            return -1
        return int(self.image_to_cluster[image_idx])

    def sample_partner(self, image_idx: int, rng: np.random.Generator) -> int | None:
        """Sample a different image from the same cluster."""
        cluster_id = self.get_cluster(image_idx)
        if cluster_id < 0:
            return None
        cluster_images = self.cluster_to_images.get(cluster_id)
        if cluster_images is None or len(cluster_images) < 2:
            return None
        # Rejection sampling (fast for small clusters)
        for _ in range(10):
            sampled = rng.choice(cluster_images)
            if sampled != image_idx:
                return int(sampled)
        # Fallback: filter and sample
        candidates = cluster_images[cluster_images != image_idx]
        return int(rng.choice(candidates)) if len(candidates) else None

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for checkpoint saving."""
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
        """Deserialize state from checkpoint."""
        image_to_cluster = data["image_to_cluster"]
        cluster_to_images: dict[int, list[int]] = {}
        for idx, cluster_id in enumerate(image_to_cluster):
            if cluster_id >= 0:
                cluster_to_images.setdefault(int(cluster_id), []).append(idx)
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
    """Manages pseudo-ID mining via mutual k-NN graph construction.

    All parameters are required - no defaults. Config is the single source of truth.
    """
    knn_k: int
    mutual_topk: int
    min_cluster_size: int
    max_cluster_size: int
    threshold_start: float
    threshold_end: float
    target_coverage: float
    adaptation_rate: float
    adaptive_mode: bool
    # State (initialized at runtime)
    state: PseudoIDState | None = field(default=None, init=False)
    current_threshold: float = field(default=0.0, init=False)
    _image_bytes_cache: dict[int, bytes] = field(default_factory=dict, init=False)
    _binary_images: np.ndarray | None = field(default=None, init=False)
    _embed_start_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        """Initialize current threshold to start value."""
        self.current_threshold = self.threshold_start

    @property
    def num_clusters(self) -> int:
        return self.state.num_clusters if self.state else 0

    @property
    def last_refresh_epoch(self) -> int:
        return self.state.last_refresh_epoch if self.state else -1

    def clear(self) -> None:
        """Clear all pseudo-ID state and cached images."""
        self.state = None
        self._image_bytes_cache.clear()
        self._binary_images = None
        self.current_threshold = self.threshold_start

    def get_threshold(self) -> float:
        """Get current similarity threshold for pseudo-ID mining."""
        return self.current_threshold

    def get_image(self, idx: int) -> np.ndarray | bytes | None:
        """Get image by index - returns numpy array (binary) or bytes (parquet)."""
        if self._binary_images is not None:
            return self._binary_images[idx] if 0 <= idx < len(self._binary_images) else None
        return self._image_bytes_cache.get(idx)

    def adapt_threshold(self, coverage: float) -> None:
        """Adapt threshold based on coverage (called after each refresh).

        If coverage < target: lower threshold to include more pairs
        If coverage > target: raise threshold for higher precision
        """
        if not self.adaptive_mode:
            return

        if coverage < self.target_coverage:
            # Under target: lower threshold to get more clusters
            new_threshold = self.current_threshold - self.adaptation_rate
        else:
            # At or above target: raise threshold for precision
            new_threshold = self.current_threshold + self.adaptation_rate * 0.5

        # Clamp to [end, start] range
        self.current_threshold = max(self.threshold_end, min(self.threshold_start, new_threshold))
        logger.info(f"  -> Adaptive threshold: {self.current_threshold:.3f} "
                    f"(coverage={coverage:.1%}, target={self.target_coverage:.1%})")

    def refresh(self, model: MoCo, datasets: list[ParquetTwoViewDataset], epoch: int,
                sim_threshold: float, device: torch.device, batch_size: int = 256,
                num_workers: int = 2) -> dict[str, Any]:
        """Run pseudo-ID mining: embed -> kNN -> mutual filter -> cluster."""
        start_time = time.perf_counter()
        logger.info(f"Pseudo-ID mining (epoch {epoch}, threshold={sim_threshold:.2f})...")

        # Step 1: Embed all training images
        logger.info("  [1/4] Embedding all images...")
        t0 = time.perf_counter()
        self._embed_start_time = t0  # For speed calculation
        embeddings, image_bytes_list = self._embed_all(model, datasets, device, batch_size, num_workers)
        num_images = len(embeddings)
        embed_time = time.perf_counter() - t0
        logger.info(f"  [1/4] Embedded {num_images:,} images in {embed_time:.1f}s ({num_images/embed_time:,.0f} img/s)")

        # Step 2: Build kNN graph
        logger.info(f"  [2/4] Building k-NN graph (k={self.knn_k})...")
        t0 = time.perf_counter()
        knn_similarities, knn_indices = self._build_knn_graph(embeddings)
        logger.info(f"  [2/4] Built k-NN graph in {time.perf_counter() - t0:.1f}s")

        # Step 3: Filter to mutual edges
        logger.info(f"  [3/4] Filtering to mutual edges (threshold={sim_threshold:.2f})...")
        t0 = time.perf_counter()
        mutual_edges = self._filter_to_mutual_edges(knn_indices, knn_similarities, sim_threshold)
        logger.info(f"  [3/4] Found {len(mutual_edges):,} mutual edges in {time.perf_counter() - t0:.1f}s")

        # Step 4: Find connected components and filter by size
        logger.info("  [4/4] Finding clusters...")
        t0 = time.perf_counter()
        cluster_labels = self._find_connected_components(mutual_edges, num_images)
        cluster_labels, cluster_to_images = self._filter_clusters_by_size(cluster_labels)
        logger.info(f"  [4/4] Clustered in {time.perf_counter() - t0:.1f}s")

        # Build state
        num_clustered = int((cluster_labels >= 0).sum())
        self.state = PseudoIDState(
            image_to_cluster=cluster_labels,
            cluster_to_images=cluster_to_images,
            num_images=num_images,
            num_clusters=len(cluster_to_images),
            num_clustered=num_clustered,
            last_refresh_epoch=epoch,
            sim_threshold_used=sim_threshold,
        )
        # Store image references - for binary mode, store dataset reference; for parquet, store bytes
        if hasattr(datasets[0], 'images'):
            # Binary mode: store reference to binary dataset's images array
            self._binary_images = datasets[0].images
            self._image_bytes_cache.clear()
        else:
            # Parquet mode: store actual bytes
            self._binary_images = None
            self._image_bytes_cache = dict(enumerate(image_bytes_list))

        elapsed = time.perf_counter() - start_time
        accept_rate = num_clustered / num_images if num_images > 0 else 0.0
        avg_cluster_size = num_clustered / len(cluster_to_images) if cluster_to_images else 0.0

        logger.info(f"  -> {len(cluster_to_images):,} clusters, {num_clustered:,} images "
                    f"({accept_rate:.1%}), avg={avg_cluster_size:.1f}, {elapsed:.1f}s")

        # Adapt threshold for next refresh based on coverage
        self.adapt_threshold(accept_rate)

        return {
            "pseudo/cluster_count": len(cluster_to_images),
            "pseudo/images_clustered": num_clustered,
            "pseudo/accept_rate": accept_rate,
            "pseudo/avg_cluster_size": avg_cluster_size,
            "pseudo/max_cluster_size": max((len(v) for v in cluster_to_images.values()), default=0),
            "pseudo/sim_threshold": sim_threshold,
            "pseudo/refresh_time_sec": elapsed,
        }

    def _embed_all(self, model: MoCo, datasets: list[ParquetTwoViewDataset], device: torch.device,
                   batch_size: int, num_workers: int) -> tuple[np.ndarray, list[bytes]]:
        """Embed all images using momentum encoder - optimized for speed."""
        from src.model import l2_normalize
        model.eval()

        # Check if we have a binary dataset (fast path)
        if hasattr(datasets[0], 'images'):
            return self._embed_binary_fast(model, datasets[0].images, device)

        # Fallback for parquet datasets
        all_embeddings, all_image_bytes = [], []
        embed_batch_size = batch_size * 4
        embed_workers = max(8, num_workers * 2)

        for dataset in datasets:
            simple_ds = _SimpleImageDataset(dataset)
            total_batches = (len(simple_ds) + embed_batch_size - 1) // embed_batch_size
            loader = DataLoader(simple_ds, batch_size=embed_batch_size,
                                num_workers=embed_workers, pin_memory=True)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch_idx, (images, image_bytes_batch) in enumerate(loader):
                    embeddings = l2_normalize(model.backbone_k(images.to(device, non_blocking=True)), dim=1)
                    all_embeddings.append(embeddings.float().cpu())
                    all_image_bytes.extend(image_bytes_batch)
                    if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                        logger.info(f"        Embedding: {(batch_idx + 1) / total_batches * 100:.0f}%")

        model.train()
        return torch.cat(all_embeddings).numpy().astype(np.float32), all_image_bytes

    def _embed_binary_fast(self, model: MoCo, images: np.ndarray, device: torch.device) -> tuple[np.ndarray, list[bytes]]:
        """Ultra-fast embedding for binary datasets - skip DataLoader entirely."""
        import sys

        from src.model import l2_normalize
        num_images = len(images)
        embed_dim = model.cfg.embedding_dim
        # Smaller batch on Windows to reduce memory pressure (mmap + file cache)
        batch_size = 2048 if sys.platform == "win32" else 8192

        # Pre-allocate output (no pin_memory to avoid locking RAM)
        all_embeddings = torch.empty((num_images, embed_dim), dtype=torch.float32)

        total_batches = (num_images + batch_size - 1) // batch_size
        log_interval = max(1, total_batches // 10)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_images)

                # Direct numpy -> GPU tensor (writable copy for torch compatibility)
                batch_np = np.array(images[start_idx:end_idx], copy=True, order="C")
                batch_gpu = torch.from_numpy(batch_np).to(device)
                del batch_np  # Free numpy copy immediately
                # HWC uint8 -> CHW float32 normalized
                batch_gpu = batch_gpu.permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5).div_(0.5)

                # Forward pass
                embeddings = l2_normalize(model.backbone_k(batch_gpu), dim=1)
                all_embeddings[start_idx:end_idx] = embeddings.float().cpu()
                del batch_gpu, embeddings  # Free GPU memory

                if (batch_idx + 1) % log_interval == 0 or batch_idx == total_batches - 1:
                    pct = (batch_idx + 1) / total_batches * 100
                    elapsed = time.perf_counter() - self._embed_start_time if hasattr(self, '_embed_start_time') else 1
                    ips = end_idx / elapsed
                    logger.info(f"        Embedding: {pct:.0f}% ({end_idx:,}/{num_images:,}) @ {ips:,.0f} img/s")

        model.train()
        # Return indices as bytes for binary mode
        image_bytes = [i.to_bytes(4, "little") for i in range(num_images)]
        return all_embeddings.numpy(), image_bytes

    def _build_knn_graph(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build kNN graph using FAISS GPU with IVF index for speed."""
        num_images, embed_dim = embeddings.shape
        search_k = self.knn_k + 1  # +1 because first result is self

        try:
            import faiss

            # Try GPU first
            use_gpu = False
            try:
                gpu_resources = faiss.StandardGpuResources()
                gpu_resources.setTempMemory(2 << 30)  # 2GB temp memory
                use_gpu = True
            except Exception as e:
                logger.warning(f"FAISS GPU unavailable, falling back to CPU (slower): {e}")

            if use_gpu:
                # GPU: use flat index (fast enough with GPU)
                index = faiss.IndexFlatIP(embed_dim)
                index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
                logger.info("        k-NN backend: FAISS-GPU")
            else:
                # CPU: use IVF index for speed (approximate but much faster)
                nlist = min(4096, num_images // 100)  # Number of clusters
                quantizer = faiss.IndexFlatIP(embed_dim)
                index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.nprobe = 64  # Search this many clusters
                logger.info(f"        k-NN backend: FAISS-CPU (IVF, nlist={nlist})")
                index.train(embeddings)

            index.add(embeddings)
            similarities, indices = index.search(embeddings, search_k)

        except ImportError:
            logger.info("        k-NN backend: sklearn (slow)")
            from sklearn.neighbors import NearestNeighbors
            nn_model = NearestNeighbors(n_neighbors=search_k, metric="cosine").fit(embeddings)
            distances, indices = nn_model.kneighbors(embeddings)
            similarities = 1.0 - distances

        # Remove self-matches (vectorized, no loop)
        # First neighbor is usually self, so take neighbors 1 to k+1
        knn_indices = indices[:, 1:search_k].astype(np.int32)
        knn_similarities = similarities[:, 1:search_k].astype(np.float32)

        return knn_similarities, knn_indices

    def _filter_to_mutual_edges(self, knn_indices: np.ndarray, knn_similarities: np.ndarray,
                                 threshold: float) -> list[tuple[int, int]]:
        """Keep only edges where both nodes have each other in top-k AND similarity >= threshold."""
        from scipy.sparse import csr_matrix

        num_images = knn_indices.shape[0]
        top_neighbors = knn_indices[:, :self.mutual_topk]
        top_similarities = knn_similarities[:, :self.mutual_topk]
        above_threshold = top_similarities >= threshold

        rows = np.repeat(np.arange(num_images), self.mutual_topk)[above_threshold.ravel()]
        cols = top_neighbors.ravel()[above_threshold.ravel()]
        if len(rows) == 0:
            return []

        # Build adjacency and find mutual edges
        adjacency = csr_matrix((top_similarities.ravel()[above_threshold.ravel()], (rows, cols)),
                                shape=(num_images, num_images))
        mutual = adjacency.multiply(adjacency.T.tocsr().astype(bool)).tocoo()
        return [(int(i), int(j)) for i, j in zip(mutual.row, mutual.col, strict=True) if i < j]

    def _find_connected_components(self, edges: list[tuple[int, int]], num_images: int) -> np.ndarray:
        """Find connected components in the mutual edge graph."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        if not edges:
            return np.full(num_images, -1, dtype=np.int32)

        edge_array = np.array(edges, dtype=np.int32)
        rows = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
        cols = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
        adjacency = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(num_images, num_images))
        _, labels = connected_components(adjacency, directed=False)
        return labels.astype(np.int32)

    def _filter_clusters_by_size(self, cluster_labels: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Filter clusters to keep only those within size bounds, remap to consecutive IDs."""
        cluster_counts = Counter(cluster_labels[cluster_labels >= 0])
        valid_clusters = {cid for cid, count in cluster_counts.items()
                          if self.min_cluster_size <= count <= self.max_cluster_size}

        new_labels = np.full_like(cluster_labels, -1)
        cluster_to_images: dict[int, np.ndarray] = {}

        for new_cluster_id, old_cluster_id in enumerate(sorted(valid_clusters)):
            image_indices = np.where(cluster_labels == old_cluster_id)[0]
            new_labels[image_indices] = new_cluster_id
            cluster_to_images[new_cluster_id] = image_indices.astype(np.int32)

        return new_labels, cluster_to_images


class _SimpleImageDataset(Dataset):
    """Dataset for embedding - returns (tensor, index/bytes) without augmentation."""

    def __init__(self, dataset: ParquetTwoViewDataset | BinaryImageDataset):
        # Check if this is a binary dataset
        if isinstance(dataset, BinaryImageDataset):
            self._is_binary = True
            self._binary_images: np.ndarray = dataset.images
            self._rows: list[tuple[bytes, str]] = []
        else:
            self._is_binary = False
            self._binary_images = np.array([])
            self._rows = []
            from src.data.file_utils import list_parquet_files
            for path in list_parquet_files(dataset.parquet_glob):
                table = pq.read_table(path, columns=["image_bytes", "identity_id"])
                for i in range(table.num_rows):
                    identity_id = table["identity_id"][i].as_py()
                    if dataset.allowed_identities is None or identity_id in dataset.allowed_identities:
                        self._rows.append((table["image_bytes"][i].as_py(), identity_id))

    def __len__(self) -> int:
        return len(self._binary_images) if self._is_binary else len(self._rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, bytes]:
        if self._is_binary:
            # Fast path: direct numpy->tensor, no PIL
            img = self._binary_images[idx]  # (H, W, 3) uint8
            tensor = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
            tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
            return tensor, idx.to_bytes(4, "little")
        else:
            # Parquet: decode and transform
            image_bytes, _ = self._rows[idx]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = transforms.functional.to_tensor(image)
            tensor = (tensor - 0.5) / 0.5
            return tensor, image_bytes
