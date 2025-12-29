"""Identity split management for train/test splitting."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from tqdm import tqdm

from src.data.file_utils import ensure_dir, list_parquet_files

logger = logging.getLogger(__name__)


def collect_unique_identities(globs: Sequence[str]) -> list[str]:
    """Collect unique identity IDs across one or more parquet globs."""
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
    """Write identity_id -> split (train/test) mapping as parquet."""
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
    """Get the set of identity IDs for a given split (train or test)."""
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
    """
    config_str = f"{sorted(globs)}|{train_ratio}|{seed}"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:12]
    splits_path = cache_dir / f"identity_splits_{config_hash}.parquet"

    if splits_path.exists():
        logger.info(f"Using cached identity splits: {splits_path}")
        return splits_path

    logger.info("Creating new identity splits (this may take a moment)...")
    ensure_dir(cache_dir)

    identity_ids = collect_unique_identities(globs)
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
    """Materialize a train/test split for a dataset into clean parquet shards."""
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
