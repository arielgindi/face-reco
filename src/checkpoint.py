"""Checkpoint save/load/prune utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src import data

logger = logging.getLogger(__name__)


def load_checkpoint_for_resume(
    resume_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> int:
    """Load checkpoint and return the epoch to resume from."""
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    logger.info(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])
    logger.info("Loaded model state")

    optimizer.load_state_dict(ckpt["optimizer"])
    logger.info("Loaded optimizer state")

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
        logger.info("Loaded AMP scaler state")

    start_epoch = int(ckpt["epoch"])
    logger.info(f"Resuming from epoch {start_epoch}")

    return start_epoch


def save_checkpoint(
    out_dir: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    cfg: DictConfig,
) -> Path:
    """Save a training checkpoint."""
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"epoch_{epoch:03d}.pt"

    payload: dict[str, Any] = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    torch.save(payload, path)
    return path


def prune_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    """Keep only the newest N checkpoints."""
    if keep_last <= 0:
        return
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if len(ckpts) <= keep_last:
        return
    for fp in ckpts[:-keep_last]:
        fp.unlink(missing_ok=True)
        logger.debug(f"Pruned checkpoint: {fp}")


def prewarm_datasets(
    digiface_ds: data.ParquetTwoViewDataset,
    digi2real_ds: data.ParquetTwoViewDataset | None,
    num_workers: int,
    device: torch.device,
) -> None:
    """Pre-warm both datasets in parallel by iterating one sample from each."""
    import concurrent.futures

    if num_workers <= 0:
        logger.info("Skipping dataset pre-warm (num_workers=0)")
        return

    logger.info("Pre-warming datasets to initialize workers...")

    def warm_one(ds: data.ParquetTwoViewDataset, name: str, p_digi: float) -> str:
        """Warm a single dataset."""
        warm_ds = data.CurriculumMixTwoViewDataset(
            digiface=digiface_ds,
            digi2real=ds if p_digi < 1.0 else None,
            p_digiface=p_digi,
            num_samples=num_workers * 2,
            seed=0,
        )
        warm_loader = DataLoader(
            warm_ds,
            batch_size=1,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=False,
        )
        for _batch in warm_loader:
            break
        del warm_loader, warm_ds
        return f"  {name} dataset warmed"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(warm_one, digiface_ds, "DigiFace", 1.0)]
        if digi2real_ds is not None:
            futures.append(executor.submit(warm_one, digi2real_ds, "Digi2Real", 0.0))

        for future in concurrent.futures.as_completed(futures):
            logger.info(future.result())

    logger.info("Dataset pre-warming complete")
