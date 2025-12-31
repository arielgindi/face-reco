"""Platform abstraction layer for cross-platform compatibility."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def get_gpu_memory_gb() -> float:
    """Get GPU memory in GB.

    Returns:
        Total GPU memory in GB, or 0.0 if no GPU or error.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0

        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)  # Convert bytes to GB
    except Exception as e:
        logger.warning(f"Failed to detect GPU memory: {e}")
        return 0.0


def get_optimal_batch_size(*, base_size: int, windows_multiplier: float = 0.25) -> int:
    """Get GPU memory-optimized batch size.

    Auto-scales batch size based on available GPU memory:
    - < 16GB VRAM: base_size // 4 (e.g., RTX 4070 Ti 12GB → 2048 from 8192)
    - 16-24GB VRAM: base_size // 2 (e.g., RTX 4090 24GB → 4096 from 8192)
    - >= 24GB VRAM: base_size (e.g., RTX 5090 32GB → 8192)

    Args:
        base_size: Target batch size for high-end GPUs (24GB+ VRAM)
        windows_multiplier: Deprecated, kept for API compatibility

    Returns:
        GPU memory-optimized batch size
    """
    gpu_memory_gb = get_gpu_memory_gb()

    if gpu_memory_gb == 0.0:
        logger.warning("No GPU detected, using conservative batch size")
        return base_size // 4

    # Auto-scale based on GPU memory
    if gpu_memory_gb < 16:
        scaled_batch = base_size // 4
        logger.info(
            f"GPU: {gpu_memory_gb:.1f}GB VRAM → batch_size={scaled_batch} (base={base_size})"
        )
    elif gpu_memory_gb < 24:
        scaled_batch = base_size // 2
        logger.info(
            f"GPU: {gpu_memory_gb:.1f}GB VRAM → batch_size={scaled_batch} (base={base_size})"
        )
    else:
        scaled_batch = base_size
        logger.info(
            f"GPU: {gpu_memory_gb:.1f}GB VRAM → batch_size={scaled_batch} (base={base_size})"
        )

    return scaled_batch


def supports_fork() -> bool:
    """Check if platform supports fork() multiprocessing."""
    return sys.platform != "win32"


def supports_torch_compile() -> bool:
    """Check if platform supports torch.compile()."""
    return sys.platform != "win32"
