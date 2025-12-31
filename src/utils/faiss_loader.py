"""FAISS loader with automatic GPU detection and Blackwell (sm_120) support."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import faiss as faiss_module

logger = logging.getLogger(__name__)

# Minimum compute capability for native faiss-gpu-cu12 support (Ada Lovelace)
_MAX_NATIVE_CC = 89  # CC 8.9 = Ada Lovelace (RTX 4090)
_MIN_BLACKWELL_CC = 100  # CC 10.0+ = Blackwell (RTX 5090)


class FaissBackend(Enum):
    """FAISS backend type."""

    GPU = "gpu"
    CPU_IVF = "cpu_ivf"
    SKLEARN = "sklearn"


@dataclass
class FaissIndex:
    """Wrapper for FAISS index with metadata."""

    index: Any
    backend: FaissBackend
    gpu_resources: Any = None


def _get_cuda_compute_capability() -> int | None:
    """Get CUDA compute capability as integer (e.g., 89 for CC 8.9).

    Returns:
        Compute capability * 10 (e.g., 89 for 8.9), or None if unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return major * 10 + minor
    except Exception:
        return None


def _enable_ptx_jit_if_needed() -> bool:
    """Enable PTX JIT compilation for Blackwell GPUs.

    Returns:
        True if PTX JIT was enabled, False otherwise.
    """
    cc = _get_cuda_compute_capability()
    if cc is None:
        return False

    if cc >= _MIN_BLACKWELL_CC:
        os.environ["CUDA_FORCE_PTX_JIT"] = "1"
        logger.info(f"Enabled PTX JIT for Blackwell GPU (CC {cc // 10}.{cc % 10})")
        return True
    return False


def load_faiss() -> faiss_module | None:
    """Load FAISS with automatic GPU setup for Blackwell.

    For Blackwell GPUs (sm_120+), enables PTX JIT compilation before import
    to allow forward-compatible execution of Ada Lovelace kernels.

    Returns:
        The faiss module, or None if not installed.
    """
    _enable_ptx_jit_if_needed()

    try:
        import faiss

        return faiss
    except ImportError:
        logger.warning("FAISS not installed, falling back to sklearn for k-NN")
        return None


def create_gpu_index(
    faiss: faiss_module,
    embed_dim: int,
    temp_memory_gb: int = 2,
) -> FaissIndex | None:
    """Create a FAISS GPU index if available.

    Args:
        faiss: The faiss module.
        embed_dim: Embedding dimension.
        temp_memory_gb: Temporary GPU memory allocation in GB.

    Returns:
        FaissIndex with GPU backend, or None if GPU unavailable.
    """
    try:
        gpu_resources = faiss.StandardGpuResources()
        gpu_resources.setTempMemory(temp_memory_gb << 30)
        index = faiss.IndexFlatIP(embed_dim)
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
        logger.info("k-NN backend: FAISS-GPU")
        return FaissIndex(index=index, backend=FaissBackend.GPU, gpu_resources=gpu_resources)
    except AttributeError as e:
        # faiss-cpu doesn't have StandardGpuResources
        logger.warning(f"FAISS GPU unavailable: {e}")
        return None
    except Exception as e:
        logger.warning(f"FAISS GPU init failed: {e}")
        return None


def create_cpu_index(
    faiss: faiss_module,
    embed_dim: int,
    num_samples: int,
    nprobe: int = 64,
) -> FaissIndex:
    """Create a FAISS CPU index with IVF for large-scale search.

    Args:
        faiss: The faiss module.
        embed_dim: Embedding dimension.
        num_samples: Number of samples (used to compute nlist).
        nprobe: Number of clusters to probe during search.

    Returns:
        FaissIndex with CPU IVF backend.
    """
    nlist = min(4096, num_samples // 100)
    quantizer = faiss.IndexFlatIP(embed_dim)
    index = faiss.IndexIVFFlat(quantizer, embed_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.nprobe = nprobe
    logger.info(f"k-NN backend: FAISS-CPU (IVF, nlist={nlist}, nprobe={nprobe})")
    return FaissIndex(index=index, backend=FaissBackend.CPU_IVF)
