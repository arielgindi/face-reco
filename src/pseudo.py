"""Pseudo-ID mining via mutual k-NN graph for self-supervised face recognition."""

from __future__ import annotations

import copy
import logging
import os
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from threading import Thread
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
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


def _make_gpu_progress_table(
    gpu_progress: list[tuple[int, int]],  # [(completed, total), ...]
    title: str,
    speed_text: str = "",
) -> Table:
    """Create a table showing per-GPU progress bars."""
    table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    table.add_column("GPU", style="cyan", width=6)
    table.add_column("Bar", ratio=1)
    table.add_column("%", style="bold", width=5, justify="right")

    for gpu_id, (completed, total) in enumerate(gpu_progress):
        pct = (completed / total * 100) if total > 0 else 0
        bar_width = 40
        filled = int(bar_width * pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        table.add_row(f"GPU {gpu_id}", f"[green]{bar}[/green]", f"{pct:3.0f}%")

    # Add totals row
    total_completed = sum(c for c, _ in gpu_progress)
    total_total = sum(t for _, t in gpu_progress)
    total_pct = (total_completed / total_total * 100) if total_total > 0 else 0

    table.add_row("", "", "")  # Spacer
    total_bar_filled = int(40 * total_pct / 100)
    total_bar = "█" * total_bar_filled + "░" * (40 - total_bar_filled)
    speed_suffix = f"  [cyan]{speed_text}[/cyan]" if speed_text else ""
    table.add_row("[bold]Total", f"[blue]{total_bar}[/blue]{speed_suffix}", f"[bold]{total_pct:3.0f}%")

    return Panel(table, title=f"[bold]{title}[/bold]", border_style="blue")


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

        embed_dim = model.cfg.embedding_dim
        batch_size = calculate_optimal_embed_batch_size(device)
        backbone = model.backbone_k
        start_time = time.perf_counter()

        # Get distributed info
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Calculate this rank's shard (for distributed embedding)
        shard_size = (num_images + world_size - 1) // world_size
        shard_start = rank * shard_size
        shard_end = min(shard_start + shard_size, num_images)
        local_count = shard_end - shard_start

        # Each rank embeds its own shard
        local_embeddings = torch.empty((shard_size, embed_dim), dtype=torch.float32)
        local_progress = [0]

        def process_embedding() -> None:
            num_batches = (local_count + batch_size - 1) // batch_size
            with torch.no_grad(), torch.amp.autocast("cuda"):
                for batch_idx in range(num_batches):
                    b_start = batch_idx * batch_size
                    b_end = min(b_start + batch_size, local_count)
                    global_start = shard_start + b_start
                    global_end = shard_start + b_end

                    batch_slice = images[global_start:global_end]
                    if not batch_slice.flags["C_CONTIGUOUS"]:
                        batch_slice = np.ascontiguousarray(batch_slice)

                    batch_tensor = torch.from_numpy(batch_slice).to(device, non_blocking=True)
                    batch_tensor = batch_tensor.permute(0, 3, 1, 2).to(torch.float32).mul_(2.0/255.0).sub_(1.0)
                    emb = l2_normalize(backbone(batch_tensor), dim=1)
                    local_embeddings[b_start:b_end] = emb.cpu()
                    del batch_tensor, emb

                    local_progress[0] = b_end

        # Run embedding with progress display (only rank 0 shows UI)
        if is_main and world_size > 1:
            # Multi-GPU: show per-GPU progress bars
            gpu_progress = [[0, shard_size] for _ in range(world_size)]
            with Live(_make_gpu_progress_table([(0, shard_size) for _ in range(world_size)], "[1/4] EMBEDDING"),
                      console=_console, refresh_per_second=4) as live:
                embed_thread = Thread(target=process_embedding)
                embed_thread.start()
                while embed_thread.is_alive():
                    elapsed = time.perf_counter() - start_time
                    # Rank 0 only knows its own progress; show estimate for others
                    gpu_progress[0][0] = local_progress[0]
                    for i in range(1, world_size):
                        gpu_progress[i][0] = local_progress[0]  # Approximate (they run in parallel)
                    total_done = sum(p[0] for p in gpu_progress)
                    ips = int(total_done / elapsed) if elapsed > 0 else 0
                    live.update(_make_gpu_progress_table(
                        [(p[0], shard_size) for p in gpu_progress],
                        "[1/4] EMBEDDING",
                        f"{ips:,} img/s"
                    ))
                    time.sleep(0.1)
                embed_thread.join()
                elapsed = time.perf_counter() - start_time
                ips = int(num_images / elapsed) if elapsed > 0 else 0
                live.update(_make_gpu_progress_table(
                    [(shard_size, shard_size) for _ in range(world_size)],
                    "[1/4] EMBEDDING ✓",
                    f"{ips:,} img/s"
                ))
        elif is_main:
            # Single GPU
            with Live(_make_gpu_progress_table([(0, num_images)], "[1/4] EMBEDDING"),
                      console=_console, refresh_per_second=4) as live:
                embed_thread = Thread(target=process_embedding)
                embed_thread.start()
                while embed_thread.is_alive():
                    elapsed = time.perf_counter() - start_time
                    done = local_progress[0]
                    ips = int(done / elapsed) if elapsed > 0 else 0
                    live.update(_make_gpu_progress_table(
                        [(done, num_images)],
                        "[1/4] EMBEDDING",
                        f"{ips:,} img/s"
                    ))
                    time.sleep(0.1)
                embed_thread.join()
                elapsed = time.perf_counter() - start_time
                ips = int(num_images / elapsed) if elapsed > 0 else 0
                live.update(_make_gpu_progress_table(
                    [(num_images, num_images)],
                    "[1/4] EMBEDDING ✓",
                    f"{ips:,} img/s"
                ))
        else:
            # Non-main ranks: just run embedding without progress display
            process_embedding()

        # Gather embeddings from all ranks using all_gather
        if torch.distributed.is_initialized() and world_size > 1:
            # Pad local embeddings to shard_size (last rank may have fewer)
            # local_embeddings is already shard_size, but only local_count are valid
            # Zero-pad the rest (already zeros from empty init)

            # all_gather requires CUDA tensors with NCCL backend
            local_cuda = local_embeddings.to(device)
            gathered = [torch.empty((shard_size, embed_dim), dtype=torch.float32, device=device) for _ in range(world_size)]
            torch.distributed.all_gather(gathered, local_cuda)

            # Concatenate and trim to actual num_images, move back to CPU
            all_embeddings = torch.cat(gathered, dim=0)[:num_images].cpu()
            del local_cuda, gathered
        else:
            all_embeddings = local_embeddings[:local_count]

        return all_embeddings.numpy(), [i.to_bytes(4, "little") for i in range(num_images)]

    def _build_knn_graph(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        num_images, embed_dim = embeddings.shape
        search_k = self.knn_k + 1
        is_main = _is_main_process()

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
                # Clear GPU memory before FAISS
                for i in range(num_gpus):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()

                t0 = time.perf_counter()
                shard_size = (num_images + num_gpus - 1) // num_gpus

                if num_gpus > 1 and is_main:
                    # Build shards with per-GPU progress
                    gpu_progress = [[0, 2] for _ in range(num_gpus)]  # 2 steps: add + ready

                    with Live(_make_gpu_progress_table([(0, 2) for _ in range(num_gpus)], "[2/4] K-NN GRAPH - Building Index"),
                              console=_console, refresh_per_second=4) as live:
                        index = faiss.IndexShards(embed_dim, True, False)

                        for i in range(num_gpus):
                            gpu_res = faiss.StandardGpuResources()
                            gpu_res.setTempMemory(self.faiss_temp_memory_gb << 30)
                            sub_index = faiss.IndexFlatIP(embed_dim)
                            gpu_index = faiss.index_cpu_to_gpu(gpu_res, i, sub_index)

                            start = i * shard_size
                            end = min(start + shard_size, num_images)
                            if start < num_images:
                                gpu_index.add(embeddings[start:end])
                            gpu_progress[i][0] = 1  # Adding done

                            live.update(_make_gpu_progress_table(
                                [(p[0], p[1]) for p in gpu_progress],
                                "[2/4] K-NN GRAPH - Building Index",
                                f"{(i+1)*shard_size:,} vectors"
                            ))

                            index.add_shard(gpu_index)
                            gpu_progress[i][0] = 2  # Shard ready

                        live.update(_make_gpu_progress_table(
                            [(2, 2) for _ in range(num_gpus)],
                            "[2/4] K-NN GRAPH - Index Built ✓",
                            f"{num_images:,} vectors"
                        ))

                    add_time = time.perf_counter() - t0

                    # Search phase with progress
                    t0 = time.perf_counter()
                    search_batch = 50000  # Search in batches to show progress
                    all_similarities = np.empty((num_images, search_k), dtype=np.float32)
                    all_indices = np.empty((num_images, search_k), dtype=np.int64)

                    with Live(_make_gpu_progress_table([(0, shard_size) for _ in range(num_gpus)], "[2/4] K-NN GRAPH - Searching"),
                              console=_console, refresh_per_second=4) as live:
                        for batch_start in range(0, num_images, search_batch):
                            batch_end = min(batch_start + search_batch, num_images)
                            sim, idx = index.search(embeddings[batch_start:batch_end], search_k)
                            all_similarities[batch_start:batch_end] = sim
                            all_indices[batch_start:batch_end] = idx

                            # Update per-GPU progress (approximate based on batch position)
                            for gpu_id in range(num_gpus):
                                gpu_start = gpu_id * shard_size
                                gpu_done = min(max(0, batch_end - gpu_start), shard_size)
                                gpu_progress[gpu_id][0] = gpu_done

                            elapsed = time.perf_counter() - t0
                            qps = int(batch_end / elapsed) if elapsed > 0 else 0
                            live.update(_make_gpu_progress_table(
                                [(p[0], shard_size) for p in gpu_progress],
                                "[2/4] K-NN GRAPH - Searching",
                                f"{qps:,} q/s"
                            ))

                        live.update(_make_gpu_progress_table(
                            [(shard_size, shard_size) for _ in range(num_gpus)],
                            "[2/4] K-NN GRAPH ✓",
                            f"{int(num_images / (time.perf_counter() - t0)):,} q/s"
                        ))

                    similarities, indices = all_similarities, all_indices

                elif num_gpus > 1:
                    # Multi-GPU but not main process
                    index = faiss.IndexShards(embed_dim, True, False)
                    for i in range(num_gpus):
                        gpu_res = faiss.StandardGpuResources()
                        gpu_res.setTempMemory(self.faiss_temp_memory_gb << 30)
                        sub_index = faiss.IndexFlatIP(embed_dim)
                        gpu_index = faiss.index_cpu_to_gpu(gpu_res, i, sub_index)
                        start = i * shard_size
                        end = min(start + shard_size, num_images)
                        if start < num_images:
                            gpu_index.add(embeddings[start:end])
                        index.add_shard(gpu_index)
                    similarities, indices = index.search(embeddings, search_k)
                else:
                    # Single GPU
                    gpu_resources = faiss.StandardGpuResources()
                    gpu_resources.setTempMemory(self.faiss_temp_memory_gb << 30)
                    index = faiss.index_cpu_to_gpu(gpu_resources, 0, faiss.IndexFlatIP(embed_dim))
                    index.add(embeddings)
                    similarities, indices = index.search(embeddings, search_k)
            else:
                nlist = min(4096, num_images // 100)
                quantizer = faiss.IndexFlatIP(embed_dim)
                index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.nprobe = self.faiss_nprobe
                if is_main:
                    _console.print(f"        [dim]FAISS-CPU: IVF nlist={nlist}[/dim]")
                index.train(embeddings)
                index.add(embeddings)
                similarities, indices = index.search(embeddings, search_k)
        except ImportError:
            if is_main:
                _console.print("        [dim]Backend: sklearn (slow)[/dim]")
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
        is_main = _is_main_process()

        # Header
        if is_main:
            _console.print()
            _console.print(Panel(
                f"[bold cyan]PSEUDO-ID MINING[/bold cyan]  •  Epoch {epoch}  •  Threshold {sim_threshold:.2f}",
                style="blue",
                padding=(0, 2),
            ))

        # Step 1: Embedding (has its own per-GPU progress display)
        t0 = time.perf_counter()
        self._embed_start_time = t0
        embeddings, image_bytes_list = self._extract_embeddings(
            model, datasets, device, batch_size, num_workers, max_images
        )
        num_images = len(embeddings)
        step_times["embed"] = time.perf_counter() - t0
        embed_ips = num_images / step_times["embed"] if step_times["embed"] > 0 else 0
        if is_main:
            _console.print(f"  [bold green]✓[/bold green] [1/4] Embedding     {num_images:,} images @ {embed_ips:,.0f} img/s [dim]({_fmt_time(step_times['embed'])})[/dim]")

        # Steps 2-4: FAISS k-NN, mutual edges, and clustering run ONLY on rank 0
        # Other ranks wait at broadcast and receive the cluster labels
        if is_main:
            # Step 2: k-NN graph (has its own per-GPU progress display)
            t0 = time.perf_counter()
            knn_similarities, knn_indices = self._build_knn_graph(embeddings)
            step_times["knn"] = time.perf_counter() - t0
            knn_qps = num_images / step_times["knn"] if step_times["knn"] > 0 else 0
            _console.print(f"  [bold green]✓[/bold green] [2/4] k-NN Graph    {num_images:,} queries @ {knn_qps:,.0f} q/s [dim]({_fmt_time(step_times['knn'])})[/dim]")

            # Step 3: Mutual edges (CPU operation - simple progress)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold yellow][3/4] Mutual Edges[/bold yellow]"),
                TextColumn(f"threshold={sim_threshold:.2f}"),
                console=_console,
            ) as progress:
                progress.add_task("mutual", total=None)
                t0 = time.perf_counter()
                mutual_edges = self._filter_mutual_edges(knn_indices, knn_similarities, sim_threshold)
                step_times["mutual"] = time.perf_counter() - t0
            _console.print(f"  [bold green]✓[/bold green] [3/4] Mutual Edges  {len(mutual_edges):,} edges [dim]({_fmt_time(step_times['mutual'])})[/dim]")

            # Step 4: Clustering (CPU operation - simple progress)
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold yellow][4/4] Clustering[/bold yellow]"),
                console=_console,
            ) as progress:
                progress.add_task("cluster", total=None)
                t0 = time.perf_counter()
                cluster_labels, cluster_to_images = self._cluster_components(mutual_edges, num_images)
                step_times["cluster"] = time.perf_counter() - t0

            # Convert to tensor for broadcast
            cluster_labels_tensor = torch.from_numpy(cluster_labels).to(device)
        else:
            # Non-main ranks: create empty tensor to receive broadcast
            cluster_labels_tensor = torch.empty(num_images, dtype=torch.int32, device=device)
            step_times["knn"] = 0.0
            step_times["mutual"] = 0.0
            step_times["cluster"] = 0.0

        # Broadcast cluster labels from rank 0 to all ranks
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(cluster_labels_tensor, src=0)

        # Convert back to numpy and rebuild cluster_to_images on non-main ranks
        cluster_labels = cluster_labels_tensor.cpu().numpy()
        if not is_main:
            # Rebuild cluster_to_images dict from the broadcasted labels
            cluster_to_images: dict[int, np.ndarray] = {}
            for idx, cid in enumerate(cluster_labels):
                if cid >= 0:
                    cluster_to_images.setdefault(int(cid), []).append(idx)
            cluster_to_images = {k: np.array(v, dtype=np.int32) for k, v in cluster_to_images.items()}

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

        if is_main:
            _console.print(f"  [bold green]✓[/bold green] [4/4] Clustering    {len(cluster_to_images):,} clusters [dim]({_fmt_time(step_times['cluster'])})[/dim]")

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
