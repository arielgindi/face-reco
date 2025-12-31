"""SniperFace - Label-free face encoder training with MoCo + MarginNCE."""

__version__ = "0.1.0"

from src.data import BinaryImageDataset, BinaryMixDataset
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
    "BinaryImageDataset",
    "BinaryMixDataset",
    "IResNet",
    "MoCo",
    "MoCoConfig",
    "__version__",
    "backbone_state_from_checkpoint",
    "build_backbone",
    "build_moco",
    "l2_normalize",
]
