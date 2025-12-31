"""Utility modules."""

from __future__ import annotations

from src.utils.distributed import (
    DistributedContext,
    cleanup_distributed,
    compute_epoch_batch_counts,
    configure_precision,
    set_seed,
    setup_distributed,
)
from src.utils.platform import (
    get_gpu_memory_gb,
    get_optimal_batch_size,
    is_windows,
    supports_fork,
    supports_torch_compile,
)

__all__ = [
    "DistributedContext",
    "cleanup_distributed",
    "compute_epoch_batch_counts",
    "configure_precision",
    "get_gpu_memory_gb",
    "get_optimal_batch_size",
    "is_windows",
    "set_seed",
    "setup_distributed",
    "supports_fork",
    "supports_torch_compile",
]
