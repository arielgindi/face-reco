"""Data module - file utilities, splits, and datasets."""

from src.data.datasets import (
    CurriculumMixTwoViewDataset,
    ParquetEmbedDataset,
    ParquetTwoViewDataset,
    PseudoPairTwoViewDataset,
    ShuffleBuffer,
    StreamParams,
)
from src.data.file_utils import (
    PILImage,
    count_parquet_rows,
    decode_image,
    ensure_dir,
    list_parquet_files,
)
from src.data.splits import (
    collect_unique_identities,
    get_identity_set,
    get_or_create_splits,
    load_identity_splits,
    materialize_split,
    write_identity_splits,
)

__all__ = [
    "CurriculumMixTwoViewDataset",
    "PILImage",
    "ParquetEmbedDataset",
    "ParquetTwoViewDataset",
    "PseudoPairTwoViewDataset",
    "ShuffleBuffer",
    "StreamParams",
    "collect_unique_identities",
    "count_parquet_rows",
    "decode_image",
    "ensure_dir",
    "get_identity_set",
    "get_or_create_splits",
    "list_parquet_files",
    "load_identity_splits",
    "materialize_split",
    "write_identity_splits",
]
