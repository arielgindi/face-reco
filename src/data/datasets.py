"""Streaming parquet datasets for self-supervised face learning."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from src.data.file_utils import PILImage, decode_image, list_parquet_files

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamParams:
    """Parquet streaming configuration."""
    shuffle_files: bool = True
    batch_read_rows: int = 2048
    shuffle_within_batch: bool = True
    shuffle_buffer_size: int = 2048
    seed: int = 42


class ShuffleBuffer:
    """Reservoir sampling buffer for streaming shuffle."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity, self.rng, self.buffer = capacity, rng, []

    def add(self, item: Any) -> Any | None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
            return None
        idx = self.rng.integers(0, len(self.buffer))
        out, self.buffer[idx] = self.buffer[idx], item
        return out

    def flush(self) -> Iterator[Any]:
        self.rng.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()


def _worker_info() -> tuple[int, int]:
    """Get (worker_id, num_workers) for DataLoader sharding."""
    info = get_worker_info()
    return (0, 1) if info is None else (info.id, info.num_workers)


def _shard_files(files: list[Path]) -> list[Path]:
    """Shard files across DataLoader workers."""
    wid, nw = _worker_info()
    return files[wid::nw]


class ParquetTwoViewDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Streaming dataset yielding two augmented views per image."""

    def __init__(self, parquet_glob: str, transform_q: Callable[[PILImage], torch.Tensor],
                 transform_k: Callable[[PILImage], torch.Tensor], *, stream: StreamParams | None = None,
                 allowed_identities: set[str] | None = None) -> None:
        super().__init__()
        self.parquet_glob, self.transform_q, self.transform_k = parquet_glob, transform_q, transform_k
        self.stream = stream or StreamParams()
        self.allowed_identities = allowed_identities
        self._files = list_parquet_files(parquet_glob)
        if not self._files:
            raise FileNotFoundError(f"No parquet files: {parquet_glob}")

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        files = _shard_files(list(self._files))
        wid, _ = _worker_info()
        rng = np.random.default_rng(self.stream.seed + wid * 1009)
        buf = ShuffleBuffer(self.stream.shuffle_buffer_size, rng) if self.stream.shuffle_buffer_size > 0 else None
        need_filter = self.allowed_identities is not None
        cols = ["identity_id", "image_bytes"] if need_filter else ["image_bytes"]

        while True:
            if self.stream.shuffle_files:
                rng.shuffle(files)
            for fp in files:
                try:
                    pf = pq.ParquetFile(fp)
                except Exception:
                    continue
                for batch in pf.iter_batches(batch_size=self.stream.batch_read_rows, columns=cols):
                    if need_filter:
                        rows = list(zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True))
                    else:
                        rows = [(None, b) for b in batch.column(0).to_pylist()]
                    if self.stream.shuffle_within_batch and len(rows) > 1:
                        rows = [rows[i] for i in rng.permutation(len(rows))]
                    for iid, img_bytes in rows:
                        if img_bytes is None:
                            continue
                        if need_filter and str(iid) not in self.allowed_identities:
                            continue
                        try:
                            img = decode_image(img_bytes)
                            sample = (self.transform_q(img), self.transform_k(img))
                        except Exception:
                            continue
                        if buf:
                            out = buf.add(sample)
                            if out: yield out
                        else:
                            yield sample
            if buf:
                yield from buf.flush()


class ParquetEmbedDataset(IterableDataset[tuple[str, str, torch.Tensor]]):
    """Streaming dataset for embedding export (identity_id, filename, tensor)."""

    def __init__(self, parquet_glob: str, transform: Callable[[PILImage], torch.Tensor], *,
                 batch_read_rows: int = 2048, allowed_identities: set[str] | None = None) -> None:
        super().__init__()
        self.parquet_glob, self.transform = parquet_glob, transform
        self.batch_read_rows, self.allowed_identities = batch_read_rows, allowed_identities
        self._files = list_parquet_files(parquet_glob)
        if not self._files:
            raise FileNotFoundError(f"No parquet files: {parquet_glob}")

    def __iter__(self) -> Iterator[tuple[str, str, torch.Tensor]]:
        for fp in _shard_files(list(self._files)):
            try:
                pf = pq.ParquetFile(fp)
            except Exception:
                continue
            for batch in pf.iter_batches(batch_size=self.batch_read_rows,
                                          columns=["identity_id", "image_filename", "image_bytes"]):
                for iid, fn, img_bytes in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(),
                                               batch.column(2).to_pylist(), strict=True):
                    if img_bytes is None:
                        continue
                    if self.allowed_identities and str(iid) not in self.allowed_identities:
                        continue
                    try:
                        yield str(iid), str(fn), self.transform(decode_image(img_bytes))
                    except Exception:
                        continue


class CurriculumMixTwoViewDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Mix DigiFace and Digi2Real streams with probability-based curriculum."""

    def __init__(self, digiface: ParquetTwoViewDataset, digi2real: ParquetTwoViewDataset | None,
                 p_digiface: float, num_samples: int, seed: int, chunk_size: int = 2048) -> None:
        super().__init__()
        self.digiface, self.digi2real = digiface, digi2real
        self.p_digiface, self.num_samples, self.seed, self.chunk_size = float(p_digiface), num_samples, seed, chunk_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        wid, nw = _worker_info()
        target = self.num_samples // nw + (1 if wid < self.num_samples % nw else 0)
        rng = np.random.default_rng(self.seed + wid * 9176)

        it_a, it_b = iter(self.digiface), iter(self.digi2real) if self.digi2real else None
        produced, cnt_a, cnt_b = 0, 0, 0

        while produced < target:
            chunk = min(self.chunk_size, target - produced)
            # Decide which dataset to sample from
            if it_b is None or self.p_digiface >= 1.0:
                pick_a = True
            elif self.p_digiface <= 0.0:
                pick_a = False
            else:
                total = cnt_a + cnt_b
                pick_a = rng.random() < self.p_digiface if total == 0 else (cnt_a / total) < self.p_digiface

            it = it_a if pick_a else it_b
            got = 0
            while got < chunk and produced < target:
                try:
                    yield next(it)
                    produced += 1
                    got += 1
                except StopIteration:
                    if pick_a:
                        it_a = iter(self.digiface)
                        it = it_a
                    else:
                        it_b = iter(self.digi2real) if self.digi2real else None
                        it = it_b
            if pick_a:
                cnt_a += got
            else:
                cnt_b += got


class PseudoPairTwoViewDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Sample cross-image pairs from pseudo-clusters, with fallback to augmentation pairs."""

    def __init__(self, base_dataset: CurriculumMixTwoViewDataset, pseudo_manager: Any,
                 transform_q: Callable[[PILImage], torch.Tensor], transform_k: Callable[[PILImage], torch.Tensor],
                 p_pseudo: float, num_samples: int, seed: int) -> None:
        super().__init__()
        self.base, self.mgr = base_dataset, pseudo_manager
        self.transform_q, self.transform_k = transform_q, transform_k
        self.p_pseudo, self.num_samples, self.seed = float(p_pseudo), num_samples, seed
        # Cache clustered indices
        state = pseudo_manager.state if pseudo_manager else None
        self._clustered = [i for i, c in enumerate(state.image_to_cluster) if c >= 0] if state else []

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, int]]:
        wid, nw = _worker_info()
        target = self.num_samples // nw + (1 if wid < self.num_samples % nw else 0)
        rng = np.random.default_rng(self.seed + wid * 7919)
        base_it = iter(self.base)
        produced = 0

        while produced < target:
            state = self.mgr.state if self.mgr else None
            # Try pseudo pair
            if self.p_pseudo > 0 and self._clustered and rng.random() < self.p_pseudo and state:
                result = self._sample_pseudo(rng, state)
                if result:
                    yield result
                    produced += 1
                    continue
            # Fallback to augmentation pair
            try:
                vq, vk = next(base_it)
                yield vq, vk, -1
                produced += 1
            except StopIteration:
                base_it = iter(self.base)

    def _apply_transform(self, transform: Any, img: np.ndarray) -> torch.Tensor:
        """Apply transform supporting both albumentations and torchvision."""
        # Try albumentations (expects numpy + keyword "image")
        try:
            out = transform(image=img)
            if isinstance(out, dict) and "image" in out:
                return out["image"]
        except TypeError:
            pass
        # Fallback to torchvision/PIL path
        return transform(Image.fromarray(img))

    def _sample_pseudo(self, rng: np.random.Generator, state: Any) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        idx_q = rng.choice(self._clustered)
        cid = state.get_cluster(idx_q)
        if cid < 0:
            return None
        idx_k = state.sample_partner(idx_q, rng)
        if idx_k is None:
            return None
        img_q, img_k = self.mgr.get_image(idx_q), self.mgr.get_image(idx_k)
        if img_q is None or img_k is None:
            return None
        try:
            # Handle both numpy arrays (binary) and bytes (parquet)
            if isinstance(img_q, np.ndarray) and isinstance(img_k, np.ndarray):
                return self._apply_transform(self.transform_q, img_q), self._apply_transform(self.transform_k, img_k), cid
            # Bytes/parquet path
            pil_q = decode_image(img_q)
            pil_k = decode_image(img_k)
            return self.transform_q(pil_q), self.transform_k(pil_k), cid
        except Exception:
            return None
