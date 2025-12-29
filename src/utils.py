"""General utility functions."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int, deterministic: bool) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
