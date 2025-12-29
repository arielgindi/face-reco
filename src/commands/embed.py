"""Embedding export command."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import data
from src.augmentations import build_embed_transform
from src.model import backbone_state_from_checkpoint, build_backbone, l2_normalize
from src.utils import select_device

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


def _embed_dataset(
    glob_pattern: str,
    backbone: nn.Module,
    transform: object,
    allowed_identities: set[str],
    writer: pq.ParquetWriter,
    schema: pa.Schema,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp_enabled: bool,
    l2_norm: bool,
    emb_dim: int,
    desc: str,
) -> int:
    """Embed all images from one dataset glob pattern."""
    try:
        ds = data.ParquetEmbedDataset(
            glob_pattern,
            transform=transform,
            allowed_identities=allowed_identities,
        )
    except FileNotFoundError:
        logger.warning(f"No files found for {desc}: {glob_pattern}")
        return 0

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    count = 0
    with torch.no_grad():
        for ids, fns, imgs in tqdm(loader, desc=desc, unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                emb = backbone(imgs)
                if l2_norm:
                    emb = l2_normalize(emb, dim=1)
            emb_np = emb.cpu().numpy().astype(np.float32, copy=False)

            values = pa.array(emb_np.reshape(-1), type=pa.float32())
            emb_arr = pa.FixedSizeListArray.from_arrays(values, emb_dim)

            table = pa.Table.from_arrays(
                [
                    pa.array(list(ids), type=pa.string()),
                    pa.array(list(fns), type=pa.string()),
                    emb_arr,
                ],
                schema=schema,
            )
            writer.write_table(table)
            count += len(ids)

    return count


def cmd_embed(cfg: DictConfig) -> None:
    """Export embeddings from trained model.

    Embeds images from BOTH digiface and digi2real datasets.
    """
    embed_cfg = cfg.get("embed", {})
    split_name = embed_cfg.get("split", "test")

    # Find checkpoint
    ckpt_dir = Path(os.getcwd()) / "checkpoints"
    if not ckpt_dir.exists():
        ckpt_dir = Path("checkpoints")

    ckpts = sorted(ckpt_dir.glob("epoch_*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        raise FileNotFoundError("No checkpoints found. Run training first.")

    ckpt_path = ckpts[-1]
    logger.info(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {})

    # Get backbone config (from current config or checkpoint)
    backbone_cfg = cfg.get("model", {}).get("backbone", {})
    if not backbone_cfg:
        backbone_cfg = ckpt_cfg.get("model", {}).get("backbone", {})

    input_size = tuple(backbone_cfg.get("input_size", [112, 112]))
    l2_norm = bool(backbone_cfg.get("l2_normalize", True))
    emb_dim = int(backbone_cfg.get("embedding_dim", 512))

    # Load backbone
    device = select_device()
    backbone = build_backbone(backbone_cfg)
    backbone.load_state_dict(backbone_state_from_checkpoint(ckpt), strict=True)
    backbone.to(device)
    backbone.eval()
    logger.info(f"Loaded backbone, device: {device}")

    # Get data globs (from current config or checkpoint)
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})

    digiface_glob = data_cfg.get("digiface_glob")
    digi2real_glob = data_cfg.get("digi2real_glob")
    if not digiface_glob:
        digiface_glob = ckpt_cfg.get("data", {}).get("digiface_glob")
    if not digi2real_glob:
        digi2real_glob = ckpt_cfg.get("data", {}).get("digi2real_glob")

    if not digiface_glob:
        raise ValueError("No data glob found in config or checkpoint.")

    # Get identity splits (use BOTH globs for consistent hash)
    cache_dir = Path(split_cfg.get("cache_dir", ".cache/splits"))
    train_ratio = float(split_cfg.get("train_ratio", 0.75))
    split_seed = int(split_cfg.get("seed", cfg.get("experiment", {}).get("seed", 42)))

    data_globs = [digiface_glob]
    if digi2real_glob:
        data_globs.append(digi2real_glob)

    splits_path = data.get_or_create_splits(
        globs=data_globs,
        cache_dir=cache_dir,
        train_ratio=train_ratio,
        seed=split_seed,
    )

    allowed_identities = data.get_identity_set(splits_path, split_name)
    logger.info(f"Embedding {len(allowed_identities)} {split_name} identities")

    # Prepare transform and writer
    transform = build_embed_transform(input_size)
    out_path = Path(os.getcwd()) / f"embeddings_{split_name}.parquet"

    schema = pa.schema([
        ("identity_id", pa.string()),
        ("image_filename", pa.string()),
        ("embedding", pa.list_(pa.float32(), emb_dim)),
    ])
    schema = schema.with_metadata({"checkpoint": str(ckpt_path.name)})
    writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd")

    # Common embedding params
    batch_size = int(embed_cfg.get("batch_size", 256))
    num_workers = int(cfg.data.streaming.get("num_workers", 4))
    amp_enabled = bool(embed_cfg.get("amp", True)) and device.type == "cuda"

    embed_params = {
        "backbone": backbone,
        "transform": transform,
        "allowed_identities": allowed_identities,
        "writer": writer,
        "schema": schema,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": device,
        "amp_enabled": amp_enabled,
        "l2_norm": l2_norm,
        "emb_dim": emb_dim,
    }

    # Embed from BOTH datasets
    total = _embed_dataset(digiface_glob, desc="digiface", **embed_params)
    if digi2real_glob:
        total += _embed_dataset(digi2real_glob, desc="digi2real", **embed_params)

    writer.close()
    logger.info(f"Wrote {total} embeddings to: {out_path}")
