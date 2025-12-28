"""Data IO utilities for face_sniper_ssl.

This module contains:
- Identity-disjoint splitting helpers (train/test by identity_id)
- Optional split materialization (write train/test shards for each dataset)
- Streaming Parquet IterableDatasets for training (two-view) and embedding export
- Shuffle buffer for proper randomization during streaming

Parquet schema expectation (zstd or any compression):
  - identity_id: string
  - image_filename: string
  - image_bytes: binary (PNG/JPG bytes)
"""

from __future__ import annotations

import glob
import io
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from tqdm import tqdm

logger = logging.getLogger(__name__)

PILImage = Image.Image


def list_parquet_files(glob_pattern: str) -> list[Path]:
    """Return sorted parquet file paths for a glob pattern."""
    # recursive=True is needed for ** patterns to work on Windows
    paths = [Path(p) for p in glob.glob(glob_pattern, recursive=True)]
    return sorted(paths)


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def collect_unique_identities(globs: Sequence[str]) -> list[str]:
    """Collect unique identity IDs across one or more parquet globs.

    This is typically used once to create an identity-disjoint train/test split.
    """
    if not globs:
        raise ValueError("No input globs provided.")

    lazy_frames = []
    for g in globs:
        files = list_parquet_files(g)
        if not files:
            logger.warning(f"No files found for glob: {g}")
            continue
        lf = pl.scan_parquet(g).select(pl.col("identity_id"))
        lazy_frames.append(lf)

    if not lazy_frames:
        raise ValueError("No parquet files found in any of the provided globs.")

    all_ids = (
        pl.concat(lazy_frames).unique().collect(streaming=True).get_column("identity_id").to_list()
    )

    logger.info(f"Collected {len(all_ids)} unique identities")
    return sorted(str(x) for x in all_ids)


def write_identity_splits(
    identity_ids: Sequence[str],
    out_path: Path,
    train_ratio: float,
    seed: int,
) -> None:
    """Write identity_id -> split (train/test) mapping as parquet.

    The split is identity-disjoint by construction.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if not identity_ids:
        raise ValueError("No identity IDs provided.")

    ids = np.array(list(identity_ids), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    n_train = round(len(ids) * train_ratio)
    split = np.array(["train"] * len(ids), dtype=object)
    split[n_train:] = "test"

    df = pl.DataFrame({"identity_id": ids.tolist(), "split": split.tolist()})
    ensure_dir(out_path.parent)
    df.write_parquet(out_path, compression="zstd")

    logger.info(f"Split {len(ids)} identities: {n_train} train, {len(ids) - n_train} test")


def load_identity_splits(path: Path) -> pl.DataFrame:
    """Load identity splits parquet as a Polars DataFrame."""
    return pl.read_parquet(path).select(["identity_id", "split"])


def get_identity_set(splits_path: Path, split_name: str) -> set[str]:
    """Get the set of identity IDs for a given split (train or test).

    Args:
        splits_path: Path to the identity_splits.parquet file
        split_name: Either "train" or "test"

    Returns:
        Set of identity_id strings belonging to that split
    """
    if split_name not in {"train", "test"}:
        raise ValueError("split_name must be 'train' or 'test'.")

    df = load_identity_splits(splits_path)
    ids = df.filter(pl.col("split") == split_name).get_column("identity_id").to_list()
    return set(str(x) for x in ids)


def get_or_create_splits(
    globs: Sequence[str],
    cache_dir: Path,
    train_ratio: float,
    seed: int,
) -> Path:
    """Get existing splits file or create a new one.

    This ensures the same split is used consistently across runs.
    The splits file is cached based on a hash of the input configuration.

    Returns:
        Path to the identity_splits.parquet file
    """
    import hashlib

    # Create a deterministic hash of the configuration
    config_str = f"{sorted(globs)}|{train_ratio}|{seed}"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    splits_path = cache_dir / f"identity_splits_{config_hash}.parquet"

    if splits_path.exists():
        logger.info(f"Using cached identity splits: {splits_path}")
        return splits_path

    logger.info("Creating new identity splits (this may take a moment)...")
    ensure_dir(cache_dir)

    # Collect all unique identities
    identity_ids = collect_unique_identities(globs)

    # Write the splits
    write_identity_splits(identity_ids, splits_path, train_ratio, seed)

    return splits_path


def materialize_split(
    input_glob: str,
    splits_df: pl.DataFrame,
    out_dir: Path,
    split_name: str,
    *,
    dataset_name: str,
) -> int:
    """Materialize a train/test split for a dataset into clean parquet shards.

    Each input parquet file becomes (0..1) output parquet file per split.
    Output parquet files keep the original columns:
      identity_id, image_filename, image_bytes

    Returns the number of rows written.
    """
    if split_name not in {"train", "test"}:
        raise ValueError("split_name must be 'train' or 'test'.")

    ensure_dir(out_dir)
    out_path = out_dir / split_name / dataset_name
    ensure_dir(out_path)

    files = list_parquet_files(input_glob)
    if not files:
        raise FileNotFoundError(f"No parquet files found for glob: {input_glob}")

    splits = splits_df.lazy()
    total_rows = 0

    for fp in tqdm(files, desc=f"materialize {dataset_name}:{split_name}", unit="file"):
        lf = pl.scan_parquet(str(fp))
        lf = (
            lf.join(splits, on="identity_id", how="inner")
            .filter(pl.col("split") == split_name)
            .drop("split")
        )
        df = lf.collect(streaming=True)
        if df.height == 0:
            continue
        df.write_parquet(out_path / fp.name, compression="zstd")
        total_rows += df.height

    logger.info(f"Materialized {total_rows} rows for {dataset_name}:{split_name}")
    return total_rows


def count_parquet_rows(glob_pattern: str) -> int:
    """Count total rows across parquet files via metadata (fast)."""
    total = 0
    for fp in list_parquet_files(glob_pattern):
        parquet_file = pq.ParquetFile(fp)
        total += parquet_file.metadata.num_rows
    return int(total)


def decode_image(image_bytes: bytes) -> PILImage:
    """Decode raw PNG/JPG bytes into a RGB PIL image."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.load()
        return img


@dataclass(frozen=True)
class StreamParams:
    """Performance knobs for parquet streaming."""

    shuffle_files: bool = True
    batch_read_rows: int = 2048
    shuffle_within_batch: bool = True
    shuffle_buffer_size: int = 2048  # Reduced from 10K to prevent memory bloat
    seed: int = 42


class ShuffleBuffer:
    """In-memory buffer for shuffling streaming data.

    This is crucial for training quality when data is read sequentially
    from files that may have correlated samples (e.g., same identity).
    """

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        self.capacity = capacity
        self.rng = rng
        self.buffer: list[Any] = []

    def add(self, item: Any) -> Any | None:
        """Add item to buffer, possibly returning a random item."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
            return None

        # Buffer is full - randomly swap and return
        idx = self.rng.integers(0, len(self.buffer))
        out = self.buffer[idx]
        self.buffer[idx] = item
        return out

    def flush(self) -> Iterator[Any]:
        """Yield remaining items in random order."""
        self.rng.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()


class ParquetTwoViewDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Streaming iterable dataset that yields two augmented views per sample.

    The dataset loops forever over the provided parquet files (cycling).
    Use a wrapper (e.g., CurriculumMixTwoViewDataset) to bound samples per epoch.

    Features:
    - Worker-aware file sharding for multi-process DataLoader
    - Shuffle buffer for better randomization
    - Identity filtering for train/test split enforcement
    - Robust error handling with logging
    """

    def __init__(
        self,
        parquet_glob: str,
        transform_q: Callable[[PILImage], torch.Tensor],
        transform_k: Callable[[PILImage], torch.Tensor],
        *,
        stream: StreamParams | None = None,
        allowed_identities: set[str] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            parquet_glob: Glob pattern for parquet files
            transform_q: Transform for query view
            transform_k: Transform for key view
            stream: Streaming parameters
            allowed_identities: If set, only yield samples where identity_id
                               is in this set. Used to enforce train/test splits.
        """
        super().__init__()
        self.parquet_glob = parquet_glob
        self.transform_q = transform_q
        self.transform_k = transform_k
        self.stream = stream or StreamParams()
        self.allowed_identities = allowed_identities

        files = list_parquet_files(parquet_glob)
        if not files:
            raise FileNotFoundError(f"No parquet files found for glob: {parquet_glob}")
        self._files = files

        filter_msg = ""
        if allowed_identities is not None:
            filter_msg = f", filtering to {len(allowed_identities)} identities"
        logger.debug(f"ParquetTwoViewDataset: found {len(files)} files{filter_msg}")

    def _shard_files_for_worker(self, files: list[Path]) -> list[Path]:
        """Shard files across DataLoader workers."""
        info = get_worker_info()
        if info is None:
            return files
        return files[info.id :: info.num_workers]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        files = self._shard_files_for_worker(list(self._files))

        info = get_worker_info()
        worker_id = 0 if info is None else int(info.id)
        seed = int(self.stream.seed) + worker_id * 1009

        rng = np.random.default_rng(seed)

        # Initialize shuffle buffer
        shuffle_buffer: ShuffleBuffer | None = None
        if self.stream.shuffle_buffer_size > 0:
            shuffle_buffer = ShuffleBuffer(self.stream.shuffle_buffer_size, rng)

        decode_errors = 0
        max_decode_errors_to_log = 10

        # Determine which columns to read
        need_identity_filter = self.allowed_identities is not None
        columns = ["identity_id", "image_bytes"] if need_identity_filter else ["image_bytes"]

        while True:
            if self.stream.shuffle_files:
                rng.shuffle(files)

            for fp in files:
                try:
                    parquet_file = pq.ParquetFile(fp)
                except Exception as e:
                    logger.warning(f"Failed to open parquet file {fp}: {e}")
                    continue

                batch_iter = parquet_file.iter_batches(
                    batch_size=self.stream.batch_read_rows,
                    columns=columns,
                )

                for record_batch in batch_iter:
                    if need_identity_filter:
                        identity_ids = record_batch.column(0).to_pylist()
                        image_bytes_list = record_batch.column(1).to_pylist()
                        # Pair identity with image bytes for filtering
                        rows = list(zip(identity_ids, image_bytes_list, strict=True))
                    else:
                        # No filtering needed
                        rows = [(None, b) for b in record_batch.column(0).to_pylist()]

                    if self.stream.shuffle_within_batch and len(rows) > 1:
                        order = rng.permutation(len(rows))
                        rows = [rows[i] for i in order]

                    for identity_id, image_bytes in rows:
                        if image_bytes is None:
                            continue

                        # Filter by identity if required
                        if need_identity_filter:
                            if str(identity_id) not in self.allowed_identities:  # type: ignore[operator]
                                continue

                        try:
                            img = decode_image(image_bytes)
                            view_q = self.transform_q(img)
                            view_k = self.transform_k(img)
                            sample = (view_q, view_k)
                        except Exception as e:
                            decode_errors += 1
                            if decode_errors <= max_decode_errors_to_log:
                                logger.debug(f"Failed to decode image: {e}")
                            continue

                        # Use shuffle buffer if enabled
                        if shuffle_buffer is not None:
                            out = shuffle_buffer.add(sample)
                            if out is not None:
                                yield out
                        else:
                            yield sample

            # Flush shuffle buffer at end of epoch cycle
            if shuffle_buffer is not None:
                yield from shuffle_buffer.flush()


class ParquetEmbedDataset(IterableDataset[tuple[str, str, torch.Tensor]]):
    """Streaming dataset for embedding export.

    Yields:
      (identity_id, image_filename, image_tensor)

    The dataset iterates each file once (finite).
    """

    def __init__(
        self,
        parquet_glob: str,
        transform: Callable[[PILImage], torch.Tensor],
        *,
        batch_read_rows: int = 2048,
        allowed_identities: set[str] | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            parquet_glob: Glob pattern for parquet files
            transform: Transform to apply to images
            batch_read_rows: Number of rows to read per batch
            allowed_identities: If set, only yield samples where identity_id
                               is in this set. Used to enforce train/test splits.
        """
        super().__init__()
        self.parquet_glob = parquet_glob
        self.transform = transform
        self.batch_read_rows = batch_read_rows
        self.allowed_identities = allowed_identities

        files = list_parquet_files(parquet_glob)
        if not files:
            raise FileNotFoundError(f"No parquet files found for glob: {parquet_glob}")
        self._files = files

        filter_msg = ""
        if allowed_identities is not None:
            filter_msg = f", filtering to {len(allowed_identities)} identities"
        logger.debug(f"ParquetEmbedDataset: found {len(files)} files{filter_msg}")

    def _shard_files_for_worker(self, files: list[Path]) -> list[Path]:
        info = get_worker_info()
        if info is None:
            return files
        return files[info.id :: info.num_workers]

    def __iter__(self) -> Iterator[tuple[str, str, torch.Tensor]]:
        files = self._shard_files_for_worker(list(self._files))
        decode_errors = 0
        max_errors_to_log = 10

        for fp in files:
            try:
                parquet_file = pq.ParquetFile(fp)
            except Exception as e:
                logger.warning(f"Failed to open parquet file {fp}: {e}")
                continue

            batch_iter = parquet_file.iter_batches(
                batch_size=self.batch_read_rows,
                columns=["identity_id", "image_filename", "image_bytes"],
            )

            for record_batch in batch_iter:
                ids = record_batch.column(0).to_pylist()
                fns = record_batch.column(1).to_pylist()
                bts = record_batch.column(2).to_pylist()

                for identity_id, filename, image_bytes in zip(
                    ids,
                    fns,
                    bts,
                    strict=True,
                ):
                    if image_bytes is None:
                        continue

                    # Filter by identity if required
                    if self.allowed_identities is not None:
                        if str(identity_id) not in self.allowed_identities:
                            continue

                    try:
                        img = decode_image(image_bytes)
                        tensor = self.transform(img)
                    except Exception as e:
                        decode_errors += 1
                        if decode_errors <= max_errors_to_log:
                            logger.debug(f"Failed to decode image {filename}: {e}")
                        continue
                    yield str(identity_id), str(filename), tensor


class CurriculumMixTwoViewDataset(IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Mix two two-view streams according to a probability.

    This yields a *finite* number of samples (`num_samples`) and then stops.
    That allows using a DataLoader iterator per epoch without manual breaks.

    Uses chunk-level mixing (not sample-level) to preserve I/O locality.
    Reads `chunk_size` consecutive samples from one dataset before switching.
    This prevents Parquet row group thrashing and cache invalidation.
    """

    def __init__(
        self,
        digiface: ParquetTwoViewDataset,
        digi2real: ParquetTwoViewDataset | None,
        *,
        p_digiface: float,
        num_samples: int,
        seed: int,
        chunk_size: int = 2048,  # Aligned with batch_read_rows for optimal I/O
    ) -> None:
        super().__init__()
        if not 0.0 <= p_digiface <= 1.0:
            raise ValueError("p_digiface must be in [0, 1].")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        self.digiface = digiface
        self.digi2real = digi2real
        self.p_digiface = float(p_digiface)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        info = get_worker_info()
        worker_id = 0 if info is None else int(info.id)
        num_workers = 1 if info is None else info.num_workers

        # Distribute samples across workers
        samples_per_worker = self.num_samples // num_workers
        if worker_id < self.num_samples % num_workers:
            samples_per_worker += 1

        rng = np.random.default_rng(self.seed + worker_id * 9176)

        it_a = iter(self.digiface)
        it_b = iter(self.digi2real) if self.digi2real is not None else None

        produced = 0
        # Track cumulative counts for ratio-aware chunk scheduling
        count_a = 0
        count_b = 0

        while produced < samples_per_worker:
            # Determine chunk size (may be smaller at end of epoch)
            remaining = samples_per_worker - produced
            current_chunk_size = min(self.chunk_size, remaining)

            # Ratio-aware dataset selection:
            # Pick the dataset that is furthest behind its target ratio
            if it_b is None:
                pick_a = True
            elif self.p_digiface >= 1.0:
                pick_a = True
            elif self.p_digiface <= 0.0:
                pick_a = False
            else:
                total = count_a + count_b
                if total == 0:
                    # First chunk: use probability
                    pick_a = rng.random() < self.p_digiface
                else:
                    # Pick dataset that's behind target ratio
                    current_ratio_a = count_a / total
                    pick_a = current_ratio_a < self.p_digiface

            # Read a full chunk from the selected dataset
            chunk_produced = 0
            current_iter = it_a if pick_a else it_b

            while chunk_produced < current_chunk_size and produced < samples_per_worker:
                try:
                    sample = next(current_iter)  # type: ignore[arg-type]
                except StopIteration:
                    # Underlying streams are expected to be infinite, but recover if not
                    if pick_a:
                        it_a = iter(self.digiface)
                        current_iter = it_a
                    else:
                        it_b = iter(self.digi2real) if self.digi2real is not None else None
                        current_iter = it_b
                    continue

                yield sample
                produced += 1
                chunk_produced += 1

            # Update counters for ratio tracking
            if pick_a:
                count_a += chunk_produced
            else:
                count_b += chunk_produced
