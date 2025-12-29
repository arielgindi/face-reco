"""Binary numpy dataset with fork-based COW sharing for maximum speed."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger(__name__)


class BinaryImageDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Ultra-fast dataset loading pre-decoded images from .npy file.

    Design:
    - Loads ENTIRE array into RAM in __init__ (main process)
    - Uses fork() workers for copy-on-write memory sharing (zero duplication)
    - Applies albumentations (OpenCV backend) for max speed
    - Infinite iteration with per-epoch shuffling

    Expected .npy format: shape (N, H, W, 3), dtype uint8

    Args:
        npy_path: Path to .npy file with pre-decoded images
        transform_q: Albumentations Compose for query view
        transform_k: Albumentations Compose for key view
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        npy_path: str | Path,
        transform_q,  # A.Compose
        transform_k,  # A.Compose
        *,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.npy_path = Path(npy_path)
        self.transform_q = transform_q
        self.transform_k = transform_k
        self.seed = seed

        if not self.npy_path.exists():
            raise FileNotFoundError(
                f"Binary cache not found: {self.npy_path}\n"
                f"Run: python fast_convert.py"
            )

        # Load array - use mmap on Windows (spawn), direct load on Unix (fork COW)
        import sys
        logger.info(f"Loading binary cache: {self.npy_path}")
        if sys.platform == "win32":
            # Windows uses spawn - each worker would reload entire array
            # Use memory-mapping for true zero-copy shared access
            self.images: np.ndarray = np.load(str(self.npy_path), mmap_mode='r')
        else:
            # Unix fork() gives COW - load once, workers share via page tables
            self.images = np.load(str(self.npy_path))

        if self.images.ndim != 4 or self.images.shape[3] != 3:
            raise ValueError(f"Expected (N, H, W, 3), got {self.images.shape}")

        # Skip contiguity check for mmap (read-only, handled by OS)
        if sys.platform != "win32" and not self.images.flags['C_CONTIGUOUS']:
            logger.warning("Making array contiguous...")
            self.images = np.ascontiguousarray(self.images)

        self._num_images = len(self.images)
        logger.info(f"Loaded {self._num_images:,} images, {self.images.nbytes / 1e9:.1f} GB")

    def __len__(self) -> int:
        return self._num_images

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (view_q, view_k) pairs forever with per-epoch shuffling."""
        info = get_worker_info()
        if info is None:
            wid, nw = 0, 1
        else:
            wid, nw = info.id, info.num_workers

        # Each worker gets interleaved indices
        all_indices = np.arange(self._num_images)
        my_indices = all_indices[wid::nw]

        # Per-worker RNG
        rng = np.random.default_rng(self.seed + wid * 1009)

        epoch = 0
        while True:
            # Shuffle for this epoch
            epoch_indices = my_indices.copy()
            rng.shuffle(epoch_indices)

            for idx in epoch_indices:
                img = self.images[idx]  # (H, W, 3) uint8, zero-copy

                try:
                    # Albumentations: numpy HWC -> Tensor CHW
                    view_q = self.transform_q(image=img)["image"]
                    view_k = self.transform_k(image=img)["image"]
                    yield view_q, view_k
                except Exception:
                    continue

            epoch += 1


class BinaryMixDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """
    Curriculum mix of BinaryImageDataset (like CurriculumMixTwoViewDataset).

    Mixes two datasets with probability p_a for dataset_a.
    """

    def __init__(
        self,
        dataset_a: BinaryImageDataset,
        dataset_b: BinaryImageDataset | None,
        p_a: float,
        num_samples: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.p_a = float(p_a)
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        if info is None:
            wid, nw = 0, 1
        else:
            wid, nw = info.id, info.num_workers

        target = self.num_samples // nw + (1 if wid < self.num_samples % nw else 0)
        rng = np.random.default_rng(self.seed + wid * 9176)

        it_a = iter(self.dataset_a)
        it_b = iter(self.dataset_b) if self.dataset_b else None

        produced = 0
        while produced < target:
            if it_b is None or self.p_a >= 1.0:
                sample = next(it_a)
            elif self.p_a <= 0.0:
                sample = next(it_b)
            else:
                sample = next(it_a) if rng.random() < self.p_a else next(it_b)

            yield sample
            produced += 1
