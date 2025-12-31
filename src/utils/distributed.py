"""General utility functions."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig


@dataclass(frozen=True)
class DistributedContext:
    """Distributed training context - immutable after initialization."""

    enabled: bool  # True if running distributed
    rank: int  # Global rank (0 = main process)
    local_rank: int  # Local rank on this node (for device selection)
    world_size: int  # Total number of processes
    device: torch.device  # Device for this process

    @property
    def is_main(self) -> bool:
        """True if this is the main process (rank 0)."""
        return self.rank == 0


def setup_distributed() -> DistributedContext:
    """Initialize distributed training from environment variables.

    Auto-detects torchrun/torch.distributed.launch environment.
    Falls back to single-GPU mode if not in distributed context.

    Returns:
        DistributedContext with rank, world_size, and device info.
    """
    # Check for torchrun environment variables
    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")

    if rank_env is not None and world_env is not None:
        # Distributed mode via torchrun
        rank = int(rank_env)
        world_size = int(world_env)
        local_rank = int(local_rank_env) if local_rank_env else rank

        # Set device before init (required for NCCL)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

        return DistributedContext(
            enabled=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
        )

    # Single-GPU fallback
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(
        enabled=False,
        rank=0,
        local_rank=0,
        world_size=1,
        device=device,
    )


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed: int, deterministic: bool, rank: int = 0) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility.

    Args:
        seed: Base random seed.
        deterministic: If True, use deterministic algorithms (slower).
        rank: Process rank for distributed training (adds offset for diversity).
    """
    # Offset seed by rank for data augmentation diversity across GPUs
    effective_seed = seed + rank * 1000
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def select_device() -> torch.device:
    """Pick CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_precision(cfg: DictConfig) -> tuple[bool, torch.dtype, bool]:
    """Return (amp_enabled, amp_dtype, tf32_enabled)."""
    precision_cfg = cfg.get("train", {}).get("precision", {})
    amp_enabled = bool(precision_cfg.get("amp", True))
    amp_dtype_str = str(precision_cfg.get("amp_dtype", "fp16")).lower()
    if amp_dtype_str not in {"fp16", "bf16"}:
        raise ValueError("train.precision.amp_dtype must be 'fp16' or 'bf16'.")

    amp_dtype = torch.float16 if amp_dtype_str == "fp16" else torch.bfloat16
    tf32_enabled = bool(precision_cfg.get("tf32_matmul", True))
    return amp_enabled, amp_dtype, tf32_enabled


def compute_epoch_batch_counts(
    *,
    base_samples: int,
    batch_size: int,
    grad_accum_steps: int,
) -> tuple[int, int]:
    """Convert a target sample count into (num_batches, num_samples)."""
    if base_samples < batch_size:
        raise ValueError("samples_per_epoch must be >= batch_size.")

    num_batches = base_samples // batch_size
    num_batches = (num_batches // grad_accum_steps) * grad_accum_steps
    if num_batches <= 0:
        raise ValueError("samples_per_epoch too small after enforcing grad_accum_steps.")
    num_samples = num_batches * batch_size
    return num_batches, num_samples
