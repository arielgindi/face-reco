"""Pseudo-ID mining via mutual k-NN graph for self-supervised face recognition."""

from __future__ import annotations

import logging
import os
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from src.model import MoCo

from src.data.binary_dataset import BinaryImageDataset
from src.utils.platform import get_optimal_batch_size

logger = logging.getLogger(__name__)

# Suppress PyTorch warning about non-writable numpy arrays (we use read-only mmap intentionally)
warnings.filterwarnings("ignore", message=".*given NumPy array is not writable.*")


def _is_main_process() -> bool:
    """Check if this is the main process (rank 0) in distributed training."""
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    return rank == 0


# Rich console - only outputs on main process
_console = Console(force_terminal=True, highlight=False)


def _fmt_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


class _MmapEmbedDataset(Dataset):
    """Dataset wrapper for mmap array (module-level for Windows pickling)."""

    def __init__(self, arr: np.ndarray, length: int):
        self._arr_ref = arr  # Keep reference to original
        self._arr = None  # Will be loaded in worker
        self.length = length

    def __len__(self) -> int:
        return self.length

    @property
    def arr(self) -> np.ndarray:
        if self._arr is None:
            self._arr = self._arr_ref
        return self._arr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.arr[idx]
        if not img.flags["C_CONTIGUOUS"]:
            img = np.ascontiguousarray(img)
        return torch.from_numpy(img), idx

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_arr_ref"] = None  # Don't pickle the array
        state["_arr"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


@dataclass
class PseudoIDState:
    """Pseudo-ID cluster assignments for all training images."""

    image_to_cluster: np.ndarray
    cluster_to_images: dict[int, np.ndarray]
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

    def sample_partner(
        self, image_idx: int, rng: np.random.Generator, rejection_tries: int = 10
    ) -> int | None:
        """Sample a different image from the same cluster."""
        cluster_id = self.get_cluster(image_idx)
        if cluster_id < 0:
            return None
        cluster_images = self.cluster_to_images.get(cluster_id)
        if cluster_images is None or len(cluster_images) < 2:
            return None
        for _ in range(rejection_tries):
            sampled = rng.choice(cluster_images)
            if sampled != image_idx:
                return int(sampled)
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
            cluster_to_images={
                k: np.array(v, dtype=np.int32) for k, v in cluster_to_images.items()
            },
            num_images=data["num_images"],
            num_clusters=data["num_clusters"],
            num_clustered=data["num_clustered"],
            last_refresh_epoch=data["last_refresh_epoch"],
            sim_threshold_used=data["sim_threshold_used"],
        )


@dataclass
class PseudoIDManager:
    """Pseudo-ID mining: embed -> kNN -> mutual filter -> cluster."""

    knn_k: int
    mutual_topk: int
    min_cluster_size: int
    max_cluster_size: int
    sim_threshold: float
    batch_size_base: int = 8192
    embed_batch_multiplier: int = 4
    embed_workers_multiplier: int = 2
    rejection_sampling_tries: int = 10
    faiss_temp_memory_gb: int = 2
    faiss_nprobe: int = 64
    state: PseudoIDState | None = field(default=None, init=False)
    _image_bytes_cache: dict[int, bytes] = field(default_factory=dict, init=False)
    _binary_images: np.ndarray | None = field(default=None, init=False)
    _embed_start_time: float = field(default=0.0, init=False)

    @property
    def num_clusters(self) -> int:
        return self.state.num_clusters if self.state else 0

    @property
    def last_refresh_epoch(self) -> int:
        return self.state.last_refresh_epoch if self.state else -1

    def clear(self) -> None:
        self.state = None
        self._image_bytes_cache.clear()
        self._binary_images = None

    def get_threshold(self) -> float:
        return self.sim_threshold

    def get_image(self, idx: int) -> np.ndarray | bytes | None:
        if self._binary_images is not None:
            return self._binary_images[idx] if 0 <= idx < len(self._binary_images) else None
        return self._image_bytes_cache.get(idx)

    def __getstate__(self) -> dict:
        """Custom pickle: exclude mmap array (worker processes will reload it lazily)."""
        state = self.__dict__.copy()
        # Don't pickle the large mmap array - workers will get it from the dataset
        state["_binary_images"] = None
        state["_image_bytes_cache"] = {}  # Also clear this to save memory
        return state

    def __setstate__(self, state: dict) -> None:
        """Custom unpickle: restore state without mmap (accessed via dataset instead)."""
        self.__dict__.update(state)

    def _extract_embeddings(
        self, model: MoCo, datasets: list, device: torch.device, batch_size: int, num_workers: int, max_images: int | None = None
    ) -> tuple[np.ndarray, list[bytes]]:
        from src.model import l2_normalize

        model.eval()
        if hasattr(datasets[0], "images"):
            result = self._extract_binary_fast(model, datasets[0].images, device, max_images)
            model.train()
            return result

        all_embeddings, all_image_bytes = [], []
        embed_batch_size = batch_size * self.embed_batch_multiplier
        embed_workers = max(8, num_workers * self.embed_workers_multiplier)

        for dataset in datasets:
            simple_ds = _SimpleImageDataset(dataset)
            total_batches = (len(simple_ds) + embed_batch_size - 1) // embed_batch_size
            loader = DataLoader(
                simple_ds, batch_size=embed_batch_size, num_workers=embed_workers, pin_memory=True
            )
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch_idx, (images, image_bytes_batch) in enumerate(loader):
                    embeddings = l2_normalize(
                        model.backbone_k(images.to(device, non_blocking=True)), dim=1
                    )
                    all_embeddings.append(embeddings.float().cpu())
                    all_image_bytes.extend(image_bytes_batch)
                    if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                        logger.info(
                            f"        Embedding: {(batch_idx + 1) / total_batches * 100:.0f}%"
                        )

        model.train()
        return torch.cat(all_embeddings).numpy().astype(np.float32), all_image_bytes

    def _extract_binary_fast(
        self, model: MoCo, images: np.ndarray, device: torch.device, max_images: int | None = None
    ) -> tuple[np.ndarray, list[bytes]]:
        from src.model import l2_normalize
        from src.utils.platform import calculate_optimal_embed_batch_size

        total_images = len(images)
        num_images = min(total_images, max_images) if max_images is not None else total_images
        is_main = _is_main_process()

        if max_images is not None and num_images < total_images and is_main:
            logger.info(f"        Subset: {num_images:,}/{total_images:,} ({num_images/total_images:.1%})")

        embed_dim = model.cfg.embedding_dim
        batch_size = calculate_optimal_embed_batch_size(device)

        all_embeddings = torch.empty((num_images, embed_dim), dtype=torch.float32)
        total_batches = (num_images + batch_size - 1) // batch_size

        # Only show progress on main process
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Embedding"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("{task.completed:,}/{task.total:,}"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("→"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TextColumn("[cyan]{task.fields[ips]:,} img/s"),
            console=_console,
            disable=not is_main,
        )

        with progress:
            task = progress.add_task("embed", total=num_images, ips=0)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch_idx in range(total_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, num_images)

                    # Zero-copy mmap access
                    batch_slice = images[start_idx:end_idx]
                    if not batch_slice.flags["C_CONTIGUOUS"]:
                        batch_slice = np.ascontiguousarray(batch_slice)

                    batch_gpu = torch.from_numpy(batch_slice).to(device, non_blocking=True)
                    # Fused preprocessing
                    batch_gpu = batch_gpu.permute(0, 3, 1, 2).to(torch.float32).mul_(2.0/255.0).sub_(1.0)
                    embeddings = l2_normalize(model.backbone_k(batch_gpu), dim=1)
                    all_embeddings[start_idx:end_idx] = embeddings.cpu()
                    del batch_gpu, embeddings

                    # Update progress with images/sec
                    elapsed = time.perf_counter() - self._embed_start_time if self._embed_start_time else 1
                    ips = int(end_idx / elapsed) if elapsed > 0 else 0
                    progress.update(task, completed=end_idx, ips=ips)

        return all_embeddings.numpy(), [i.to_bytes(4, "little") for i in range(num_images)]

    def _build_knn_graph(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_images, embed_dim = embeddings.shape
        search_k = self.knn_k + 1
        is_main = _is_main_process()

        def _log(msg: str) -> None:
            if is_main:
                _console.print(f"        [dim]{msg}[/dim]")

        try:
            import faiss

            use_gpu = False
            num_gpus = 0
            try:
                num_gpus = faiss.get_num_gpus()
                use_gpu = num_gpus > 0
            except Exception:
                pass  # Silently fall back to CPU

            if use_gpu:
                # Use single GPU to avoid OOM when training also uses GPU memory
                # Multi-GPU replicates the entire index on each GPU (2.5GB+)
                _log(f"FAISS: GPU 0 (of {num_gpus})")
                t0 = time.perf_counter()
                index = faiss.IndexFlatIP(embed_dim)
                gpu_resources = faiss.StandardGpuResources()
                gpu_resources.setTempMemory(self.faiss_temp_memory_gb << 30)
                index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

                index.add(embeddings)
                add_time = time.perf_counter() - t0
                _log(f"Added {num_images:,} vectors in {add_time:.1f}s")

                t0 = time.perf_counter()
                similarities, indices = index.search(embeddings, search_k)
                search_time = time.perf_counter() - t0
                _log(f"Searched k={self.knn_k} in {search_time:.1f}s ({num_images/search_time:,.0f} q/s)")
            else:
                nlist = min(4096, num_images // 100)
                quantizer = faiss.IndexFlatIP(embed_dim)
                index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.nprobe = self.faiss_nprobe
                _log(f"FAISS-CPU: IVF nlist={nlist}")
                index.train(embeddings)
                index.add(embeddings)
                similarities, indices = index.search(embeddings, search_k)
        except ImportError:
            _log("Backend: sklearn (slow)")
            from sklearn.neighbors import NearestNeighbors
            nn_model = NearestNeighbors(n_neighbors=search_k, metric="cosine").fit(embeddings)
            distances, indices = nn_model.kneighbors(embeddings)
            similarities = 1.0 - distances

        return similarities[:, 1:search_k].astype(np.float32), indices[:, 1:search_k].astype(
            np.int32
        )

    def _filter_mutual_edges(
        self, knn_indices: np.ndarray, knn_similarities: np.ndarray, threshold: float
    ) -> list[tuple[int, int]]:
        from scipy.sparse import csr_matrix

        num_images = knn_indices.shape[0]
        top_neighbors = knn_indices[:, : self.mutual_topk]
        top_similarities = knn_similarities[:, : self.mutual_topk]
        above_threshold = top_similarities >= threshold

        rows = np.repeat(np.arange(num_images), self.mutual_topk)[above_threshold.ravel()]
        cols = top_neighbors.ravel()[above_threshold.ravel()]
        if len(rows) == 0:
            return []

        adjacency = csr_matrix(
            (top_similarities.ravel()[above_threshold.ravel()], (rows, cols)),
            shape=(num_images, num_images),
        )
        mutual = adjacency.multiply(adjacency.T.tocsr().astype(bool)).tocoo()
        return [(int(i), int(j)) for i, j in zip(mutual.row, mutual.col, strict=True) if i < j]

    def _cluster_components(
        self, edges: list[tuple[int, int]], num_images: int
    ) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        if not edges:
            return np.full(num_images, -1, dtype=np.int32), {}

        edge_array = np.array(edges, dtype=np.int32)
        rows = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
        cols = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
        adjacency = csr_matrix(
            (np.ones(len(rows), np.int8), (rows, cols)), shape=(num_images, num_images)
        )
        _, labels = connected_components(adjacency, directed=False)
        cluster_labels = labels.astype(np.int32)

        cluster_counts = Counter(cluster_labels[cluster_labels >= 0])
        valid_clusters = {
            cid
            for cid, count in cluster_counts.items()
            if self.min_cluster_size <= count <= self.max_cluster_size
        }

        new_labels = np.full_like(cluster_labels, -1)
        cluster_to_images: dict[int, np.ndarray] = {}
        for new_cluster_id, old_cluster_id in enumerate(sorted(valid_clusters)):
            image_indices = np.where(cluster_labels == old_cluster_id)[0]
            new_labels[image_indices] = new_cluster_id
            cluster_to_images[new_cluster_id] = image_indices.astype(np.int32)

        return new_labels, cluster_to_images

    def refresh(
        self,
        model: MoCo,
        datasets: list,
        epoch: int,
        sim_threshold: float,
        device: torch.device,
        batch_size: int = 256,
        num_workers: int = 2,
        max_images: int | None = None,
    ) -> dict[str, Any]:
        """Run pseudo-ID mining: embed -> kNN -> mutual filter -> cluster."""
        start_time = time.perf_counter()
        step_times: dict[str, float] = {}
        step_results: dict[str, str] = {}
        is_main = _is_main_process()

        # Header
        if is_main:
            _console.print()
            _console.print(Panel(
                f"[bold cyan]PSEUDO-ID MINING[/bold cyan]  •  Epoch {epoch}  •  Threshold {sim_threshold:.2f}",
                style="blue",
                padding=(0, 2),
            ))

        # Step 1: Embedding
        if is_main:
            _console.print("  [bold yellow]►[/bold yellow] [1/4] Embedding...", end="")
        t0 = time.perf_counter()
        self._embed_start_time = t0
        embeddings, image_bytes_list = self._extract_embeddings(
            model, datasets, device, batch_size, num_workers, max_images
        )
        num_images = len(embeddings)
        step_times["embed"] = time.perf_counter() - t0
        embed_ips = num_images / step_times["embed"] if step_times["embed"] > 0 else 0
        step_results["embed"] = f"{num_images:,} images @ {embed_ips:,.0f} img/s"
        if is_main:
            _console.print(f"\r  [bold green]✓[/bold green] [1/4] Embedding     {step_results['embed']} [dim]({_fmt_time(step_times['embed'])})[/dim]")

        # Step 2: k-NN graph
        if is_main:
            _console.print(f"  [bold yellow]►[/bold yellow] [2/4] k-NN Graph (k={self.knn_k})...", end="")
        t0 = time.perf_counter()
        knn_similarities, knn_indices = self._build_knn_graph(embeddings)
        step_times["knn"] = time.perf_counter() - t0
        step_results["knn"] = f"{num_images:,} queries"
        if is_main:
            _console.print(f"\r  [bold green]✓[/bold green] [2/4] k-NN Graph    {step_results['knn']} [dim]({_fmt_time(step_times['knn'])})[/dim]     ")

        # Step 3: Mutual edges
        if is_main:
            _console.print(f"  [bold yellow]►[/bold yellow] [3/4] Mutual Edges (threshold={sim_threshold:.2f})...", end="")
        t0 = time.perf_counter()
        mutual_edges = self._filter_mutual_edges(knn_indices, knn_similarities, sim_threshold)
        step_times["mutual"] = time.perf_counter() - t0
        step_results["mutual"] = f"{len(mutual_edges):,} edges"
        if is_main:
            _console.print(f"\r  [bold green]✓[/bold green] [3/4] Mutual Edges  {step_results['mutual']} [dim]({_fmt_time(step_times['mutual'])})[/dim]             ")

        # Step 4: Clustering
        if is_main:
            _console.print("  [bold yellow]►[/bold yellow] [4/4] Clustering...", end="")
        t0 = time.perf_counter()
        cluster_labels, cluster_to_images = self._cluster_components(mutual_edges, num_images)
        step_times["cluster"] = time.perf_counter() - t0

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
        if hasattr(datasets[0], "images"):
            self._binary_images = datasets[0].images
            self._image_bytes_cache.clear()
        else:
            self._binary_images = None
            self._image_bytes_cache = dict(enumerate(image_bytes_list))

        elapsed = time.perf_counter() - start_time
        accept_rate = num_clustered / num_images if num_images > 0 else 0.0
        avg_cluster_size = num_clustered / len(cluster_to_images) if cluster_to_images else 0.0
        max_size = max((len(v) for v in cluster_to_images.values()), default=0)

        step_results["cluster"] = f"{len(cluster_to_images):,} clusters"
        if is_main:
            _console.print(f"\r  [bold green]✓[/bold green] [4/4] Clustering    {step_results['cluster']} [dim]({_fmt_time(step_times['cluster'])})[/dim]")

            # Summary table
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="dim")
            table.add_column(style="bold")
            table.add_row("Total time", _fmt_time(elapsed))
            table.add_row("Clusters", f"{len(cluster_to_images):,}")
            table.add_row("Clustered", f"{num_clustered:,}/{num_images:,} ({accept_rate:.1%})")
            table.add_row("Avg/Max size", f"{avg_cluster_size:.1f} / {max_size}")
            _console.print(Panel(table, title="[bold green]Summary[/bold green]", style="green", padding=(0, 1)))
            _console.print()

        return {
            "pseudo/cluster_count": len(cluster_to_images),
            "pseudo/images_clustered": num_clustered,
            "pseudo/accept_rate": accept_rate,
            "pseudo/avg_cluster_size": avg_cluster_size,
            "pseudo/max_cluster_size": max_size,
            "pseudo/sim_threshold": sim_threshold,
            "pseudo/refresh_time_sec": elapsed,
        }


class _SimpleImageDataset(Dataset):
    """Dataset for embedding - returns (tensor, index) for binary datasets."""

    def __init__(self, dataset: BinaryImageDataset):
        self._binary_images: np.ndarray = dataset.images

    def __len__(self) -> int:
        return len(self._binary_images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, bytes]:
        img = self._binary_images[idx]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        tensor = (tensor - 0.5) / 0.5
        return tensor, idx.to_bytes(4, "little")
