"""Platform abstraction layer for cross-platform compatibility."""

from __future__ import annotations

import sys


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def get_optimal_batch_size(*, base_size: int, windows_multiplier: float = 0.25) -> int:
    """Get platform-optimized batch size (Windows gets reduced due to spawn)."""
    if is_windows():
        return int(base_size * windows_multiplier)
    return base_size


def calculate_optimal_embed_batch_size(device, target_vram_usage: float = 0.6) -> int:
    """Calculate optimal batch size for embedding extraction based on FREE GPU memory.

    This is specifically for inference-only embedding extraction, not training.
    Uses actual free GPU memory to determine batch size (model may already be loaded).

    Args:
        device: torch.device
        target_vram_usage: Target utilization of free VRAM (0.0-1.0), default 0.6

    Returns:
        Optimal batch size (power of 2)
    """
    import torch

    if device.type != "cuda":
        return 2048

    try:
        # Use FREE memory, not total - model is already loaded
        free_memory = torch.cuda.mem_get_info(device)[0]
        free_memory_gb = free_memory / (1024 ** 3)

        # Each image: 112x112x3 = 37,632 bytes input
        # Forward pass activations: ~4MB per image for iresnet50
        # Use conservative estimate: ~5MB per image including gradients
        bytes_per_image = 5 * 1024 * 1024  # 5 MB
        usable_memory = free_memory * target_vram_usage

        # Calculate batch size, round down to nearest power of 2
        max_batch = int(usable_memory / bytes_per_image)
        batch_size = 1
        while batch_size * 2 <= max_batch:
            batch_size *= 2

        # Clamp to reasonable range
        return max(256, min(batch_size, 4096))
    except Exception:
        # Fallback if GPU detection fails
        return 1024


def supports_fork() -> bool:
    """Check if platform supports fork() multiprocessing."""
    return sys.platform != "win32"


def supports_torch_compile() -> bool:
    """Check if platform supports torch.compile()."""
    return sys.platform != "win32"
