"""File and parquet utilities."""

from __future__ import annotations

import glob
import io
import logging
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

logger = logging.getLogger(__name__)

PILImage = Image.Image


def list_parquet_files(glob_pattern: str) -> list[Path]:
    """Return sorted parquet file paths for a glob pattern."""
    paths = [Path(p) for p in glob.glob(glob_pattern, recursive=True)]
    return sorted(paths, key=lambda p: str(p.resolve()))


def ensure_dir(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


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
