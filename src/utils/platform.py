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
    """Calculate optimal batch size for embedding extraction based on GPU memory.

    This is specifically for inference-only embedding extraction, not training.
    Uses actual GPU memory to determine batch size instead of platform heuristics.

    Args:
        device: torch.device
        target_vram_usage: Target VRAM utilization (0.0-1.0), default 0.6

    Returns:
        Optimal batch size (power of 2)
    """
    import torch

    if device.type != "cuda":
        return 2048

    try:
        props = torch.cuda.get_device_properties(device)
        total_memory_gb = props.total_memory / (1024 ** 3)
        gpu_name = props.name

        # GPU-specific optimizations based on memory and architecture
        if "4070" in gpu_name:
            # RTX 4070 Ti: 12GB VRAM, conservative batch for iresnet50
            return 2048
        elif "4090" in gpu_name or total_memory_gb >= 24:
            # RTX 4090, A100, H100: 24GB+
            return 8192
        elif "4080" in gpu_name or "3090" in gpu_name or total_memory_gb >= 16:
            # RTX 4080, 3090: 16-24GB
            return 6144
        elif total_memory_gb >= 12:
            # 12-16GB cards
            return 4096
        elif total_memory_gb >= 8:
            # 8-12GB cards
            return 2048
        else:
            # Smaller GPUs
            return 1024
    except Exception:
        # Fallback if GPU detection fails
        return 2048


def supports_fork() -> bool:
    """Check if platform supports fork() multiprocessing."""
    return sys.platform != "win32"


def supports_torch_compile() -> bool:
    """Check if platform supports torch.compile()."""
    return sys.platform != "win32"
