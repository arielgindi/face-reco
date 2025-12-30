"""Checkpoint save/load/prune utilities."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import wandb
from src import data

logger = logging.getLogger(__name__)


def find_latest_checkpoint(project: str, local_dir: Path | None = None) -> Path:
    """Find the latest checkpoint from W&B or local folder.

    Args:
        project: W&B project name (e.g., "sniperface-v2")
        local_dir: Local checkpoints directory to search if W&B fails

    Returns:
        Path to the latest checkpoint file

    Raises:
        FileNotFoundError: If no checkpoint found in W&B or locally
    """
    # Try W&B first
    try:
        api = wandb.Api()
        # Get latest version of checkpoint artifact
        artifact = api.artifact(f"{project}/checkpoint:latest")
        # Download to temp directory
        tmp_dir = Path(tempfile.gettempdir()) / "wandb_checkpoints"
        tmp_dir.mkdir(exist_ok=True)
        artifact_dir = Path(artifact.download(root=str(tmp_dir)))
        # Find the .pt file in the artifact - sort by epoch number to get newest
        pt_files = sorted(artifact_dir.glob("epoch_*.pt"), reverse=True)
        if pt_files:
            logger.info(f"Found checkpoint in W&B: {artifact.name}")
            return pt_files[0]
    except Exception as e:
        logger.debug(f"W&B checkpoint not found: {e}")

    # Fall back to local folder
    if local_dir and local_dir.exists():
        ckpts = sorted(local_dir.glob("epoch_*.pt"))
        if ckpts:
            logger.info(f"Found local checkpoint: {ckpts[-1].name}")
            return ckpts[-1]

    raise FileNotFoundError("No checkpoint found in W&B or locally")


def load_checkpoint_for_resume(
    resume_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    *,
    pseudo_manager: Any = None,
    warm_start: bool = False,
) -> int:
    """Load checkpoint and return the epoch to resume from.

    Args:
        resume_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scaler: Optional AMP scaler
        device: Target device
        pseudo_manager: Optional PseudoIDManager to load state into
        warm_start: If True, reset epoch to 0 and clear pseudo state
    """
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_epoch = int(ckpt.get("epoch", 0))

    # For warm start, filter out queue buffers (they may have different sizes)
    ckpt_state = ckpt["model"]
    if warm_start:
        # Only load backbone and projector weights, skip queue buffers
        skip_prefixes = ("queue", "queue_ptr", "queue_cluster_ids")
        ckpt_state = {
            k: v for k, v in ckpt_state.items()
            if not k.startswith(skip_prefixes)
        }

    # Load model with strict=False to handle new/missing buffers
    model.load_state_dict(ckpt_state, strict=False)

    if not warm_start:
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        # Load pseudo-ID state if available
        if pseudo_manager is not None and "pseudo" in ckpt:
            from src.pseudo import PseudoIDState
            pseudo_manager.state = PseudoIDState.from_dict(ckpt["pseudo"])

        start_epoch = ckpt_epoch
        logger.info(f"Resumed from {path.name} at epoch {start_epoch}")
    else:
        # Warm start: reset epoch, don't load optimizer/scaler/pseudo
        start_epoch = 0
        if pseudo_manager is not None:
            pseudo_manager.clear()
        logger.info(f"Warm start from {path.name} (checkpoint epoch {ckpt_epoch})")

    return start_epoch


def save_checkpoint(
    out_dir: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    cfg: DictConfig,
    pseudo_manager: Any = None,
) -> Path:
    """Save a training checkpoint.

    Args:
        out_dir: Output directory
        epoch: Current epoch (1-indexed, after completion)
        model: Model to save
        optimizer: Optimizer to save
        scaler: Optional AMP scaler
        cfg: Config to save
        pseudo_manager: Optional PseudoIDManager to save state from
    """
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

    # Save pseudo-ID state if available
    if pseudo_manager is not None and pseudo_manager.state is not None:
        payload["pseudo"] = pseudo_manager.state.to_dict()

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
