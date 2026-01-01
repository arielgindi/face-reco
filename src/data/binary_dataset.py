"""Binary numpy dataset with fork-based COW sharing for maximum speed."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.data.worker_utils import _global_worker_info

logger = logging.getLogger(__name__)


def get_binary_dataset_length(npy_path: str | Path) -> int:
    """Get dataset length without loading the full array into memory."""
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Binary cache not found: {npy_path}")

    # Use mmap to read just the header (shape info)
    arr = np.load(str(npy_path), mmap_mode="r")
    return len(arr)


class BinaryImageDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Ultra-fast dataset loading pre-decoded images from .npy file (N, H, W, 3) uint8."""

    def __init__(
        self,
        npy_path: str | Path,
        transform_q,  # A.Compose
        transform_k,  # A.Compose
        *,
        seed: int = 42,
        block_size: int = 8192,
        worker_seed_offset: int = 7919,
    ) -> None:
        super().__init__()
        self.npy_path = Path(npy_path)
        self.transform_q = transform_q
        self.transform_k = transform_k
        self.seed = seed
        self.block_size = block_size
        self.worker_seed_offset = worker_seed_offset

        if not self.npy_path.exists():
            raise FileNotFoundError(
                f"Binary cache not found: {self.npy_path}\nRun: python fast_convert.py"
            )

        # Lazy load: only store metadata, load mmap in worker processes
        # This avoids pickling the large mmap array on Windows (spawn)
        self._images: np.ndarray | None = None
        self._num_images: int | None = None
        self._shape: tuple[int, ...] | None = None

        # Pre-compute metadata (fast, just reads npy header)
        self._init_metadata()

    def _init_metadata(self) -> None:
        """Load just the shape info without keeping mmap reference."""
        arr = np.load(str(self.npy_path), mmap_mode="r")
        if arr.ndim != 4 or arr.shape[3] != 3:
            raise ValueError(f"Expected (N, H, W, 3), got {arr.shape}")
        self._num_images = len(arr)
        self._shape = arr.shape
        size_gb = (self._num_images * arr.shape[1] * arr.shape[2] * 3) / 1e9
        logger.info(f"Loaded {self._num_images:,} images, {size_gb:.1f} GB (Virtual)")
        # Don't store arr - let it be garbage collected

    @property
    def images(self) -> np.ndarray:
        """Lazy-load mmap array (called in worker process on Windows)."""
        if self._images is None:
            self._images = np.load(str(self.npy_path), mmap_mode="r")
        return self._images

    def __len__(self) -> int:
        return self._num_images

    def __getstate__(self) -> dict:
        """Custom pickle: exclude mmap array (worker processes will reload it lazily)."""
        state = self.__dict__.copy()
        # Don't pickle the large mmap array - workers will reload it via the images property
        state["_images"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Custom unpickle: restore state without mmap (will be loaded lazily on first access)."""
        self.__dict__.update(state)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield (view_q, view_k) pairs with block shuffle for sequential IO."""
        gid, total = _global_worker_info()

        N = self._num_images

        # Contiguous shard per global worker (DDP rank * DataLoader workers)
        per_worker = (N + total - 1) // total
        start = gid * per_worker
        end = min(start + per_worker, N)
        my_indices = np.arange(start, end, dtype=np.int64)

        # Per-worker RNG (unique across all ranks + workers)
        rng = np.random.default_rng(self.seed + gid * self.worker_seed_offset)

        # Block shuffle: keeps IO mostly sequential
        epoch = 0
        while True:
            # Shuffle blocks, not individual indices
            blocks = np.arange(0, len(my_indices), self.block_size, dtype=np.int64)
            rng.shuffle(blocks)

            for b in blocks:
                block = my_indices[b : b + self.block_size].copy()
                # Light shuffle within block (still local/sequential-ish)
                rng.shuffle(block)

                for idx in block:
                    img = self.images[idx]  # (H, W, 3) uint8

                    try:
                        # Albumentations: numpy HWC -> Tensor CHW
                        view_q = self.transform_q(image=img)["image"]
                        view_k = self.transform_k(image=img)["image"]
                        yield view_q, view_k
                    except (KeyError, ValueError, TypeError) as e:
                        logger.debug(f"Transform failed for image {idx}: {e}")
                        continue

            epoch += 1


class BinaryMixDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Curriculum mix of BinaryImageDataset with probability p_a for dataset_a."""

    def __init__(
        self,
        dataset_a: BinaryImageDataset,
        dataset_b: BinaryImageDataset | None,
        p_a: float,
        num_samples: int,
        seed: int,
        worker_seed_offset: int = 7919,
    ) -> None:
        super().__init__()
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.p_a = float(p_a)
        self.num_samples = num_samples
        self.seed = seed
        self.worker_seed_offset = worker_seed_offset

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        gid, total = _global_worker_info()

        target = self.num_samples // total + (1 if gid < self.num_samples % total else 0)
        rng = np.random.default_rng(self.seed + gid * self.worker_seed_offset)

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


class PseudoPairTwoViewDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Wraps a base dataset and samples pseudo-ID pairs with probability p_pseudo."""

    def __init__(
        self,
        base_dataset: BinaryImageDataset | BinaryMixDataset,
        pseudo_manager,
        transform_q,
        transform_k,
        p_pseudo: float,
        num_samples: int,
        seed: int,
        worker_seed_offset: int = 7919,
    ) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.pseudo_manager = pseudo_manager
        self.transform_q = transform_q
        self.transform_k = transform_k
        self.p_pseudo = float(p_pseudo)
        self.num_samples = num_samples
        self.seed = seed
        self.worker_seed_offset = worker_seed_offset

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        gid, total = _global_worker_info()

        target = self.num_samples // total + (1 if gid < self.num_samples % total else 0)
        rng = np.random.default_rng(self.seed + gid * self.worker_seed_offset)

        it_base = iter(self.base_dataset)
        produced = 0

        # Get base dataset images for pseudo-ID sampling
        if hasattr(self.base_dataset, "dataset_a") and hasattr(
            self.base_dataset.dataset_a, "images"
        ):
            base_images = self.base_dataset.dataset_a.images
        elif hasattr(self.base_dataset, "images"):
            base_images = self.base_dataset.images
        else:
            raise ValueError("Cannot access images from base_dataset")

        while produced < target:
            # With probability p_pseudo, sample a pseudo-ID pair
            if (
                self.pseudo_manager
                and self.pseudo_manager.state is not None
                and rng.random() < self.p_pseudo
            ):
                # Sample random image index
                img_idx = rng.integers(0, len(base_images))

                # Try to get pseudo-ID partner
                partner_idx = self.pseudo_manager.state.sample_partner(img_idx, rng)

                if partner_idx is not None:
                    # Apply transforms to both images
                    img_q = base_images[img_idx]
                    img_k = base_images[partner_idx]

                    try:
                        view_q = self.transform_q(image=img_q)["image"]
                        view_k = self.transform_k(image=img_k)["image"]
                        yield view_q, view_k
                        produced += 1
                        continue
                    except (KeyError, ValueError, TypeError) as e:
                        logger.debug(
                            f"Transform failed for pseudo pair ({img_idx}, {partner_idx}): {e}"
                        )

            # Fall back to base dataset
            try:
                yield next(it_base)
                produced += 1
            except StopIteration:
                # Restart base iterator if we run out
                it_base = iter(self.base_dataset)
                yield next(it_base)
                produced += 1
