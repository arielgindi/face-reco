"""SniperFace - Label-free face encoder training with MoCo + MarginNCE."""

from sniperface.dataio import (
    CurriculumMixTwoViewDataset,
    ParquetEmbedDataset,
    ParquetTwoViewDataset,
    get_identity_set,
    get_or_create_splits,
)
from sniperface.moco import MoCo, build_backbone, build_moco

__version__ = "0.1.0"
__all__ = [
    "CurriculumMixTwoViewDataset",
    "MoCo",
    "ParquetEmbedDataset",
    "ParquetTwoViewDataset",
    "build_backbone",
    "build_moco",
    "get_identity_set",
    "get_or_create_splits",
]
