"""Shared utilities for DataLoader worker and DDP rank management."""

from __future__ import annotations

import torch.distributed as dist
from torch.utils.data import get_worker_info


def _worker_info() -> tuple[int, int]:
    """Get (worker_id, num_workers) for DataLoader sharding."""
    info = get_worker_info()
    return (0, 1) if info is None else (info.id, info.num_workers)


def _distributed_info() -> tuple[int, int]:
    """Get (rank, world_size) for distributed training."""
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _global_worker_info() -> tuple[int, int]:
    """Get global (worker_id, total_workers) across all DDP ranks and DataLoader workers."""
    wid, nw = _worker_info()
    rank, world_size = _distributed_info()
    return rank * nw + wid, world_size * nw
