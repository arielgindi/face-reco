"""SniperFace - Label-free face encoder training with MoCo + MarginNCE."""

__version__ = "0.1.0"

from src.data import (
    CurriculumMixTwoViewDataset,
    ParquetEmbedDataset,
    ParquetTwoViewDataset,
    StreamParams,
    count_parquet_rows,
    get_identity_set,
    get_or_create_splits,
)
from src.model import (
    IResNet,
    MoCo,
    MoCoConfig,
    backbone_state_from_checkpoint,
    build_backbone,
    build_moco,
    l2_normalize,
)

__all__ = [
    "CurriculumMixTwoViewDataset",
    "IResNet",
    "MoCo",
    "MoCoConfig",
    "ParquetEmbedDataset",
    "ParquetTwoViewDataset",
    "StreamParams",
    "__version__",
    "backbone_state_from_checkpoint",
    "build_backbone",
    "build_moco",
    "count_parquet_rows",
    "get_identity_set",
    "get_or_create_splits",
    "l2_normalize",
]
