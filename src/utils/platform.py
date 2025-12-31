"""Platform abstraction layer for cross-platform compatibility."""

from __future__ import annotations

import sys


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def get_optimal_batch_size(*, base_size: int, windows_multiplier: float = 0.25) -> int:
    """Get platform-optimized batch size.

    Note: mmap block shuffling eliminates Windows memory pressure penalty.
    Returns base_size on all platforms.
    """
    return base_size


def supports_fork() -> bool:
    """Check if platform supports fork() multiprocessing."""
    return sys.platform != "win32"


def supports_torch_compile() -> bool:
    """Check if platform supports torch.compile()."""
    return sys.platform != "win32"
