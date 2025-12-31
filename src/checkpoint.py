"""Checkpoint save/load/prune utilities."""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import wandb

logger = logging.getLogger(__name__)


def find_latest_checkpoint(project: str, local_dir: Path | None = None) -> Path:
    """Find the latest checkpoint from W&B or local folder."""
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
    except (wandb.errors.CommError, ValueError, KeyError) as e:
        logger.debug(f"W&B checkpoint not found (network/API error): {e}")
    except OSError as e:
        logger.debug(f"W&B checkpoint download failed (I/O error): {e}")

    # Fall back to local folder
    if local_dir and local_dir.exists():
        ckpts = sorted(local_dir.glob("epoch_*.pt"))
        if ckpts:
            logger.info(f"Found local checkpoint: {ckpts[-1].name}")
            return ckpts[-1]

    raise FileNotFoundError("No checkpoint found in W&B or locally")


def _filter_warm_start_state(ckpt_state: dict) -> dict:
    """Filter out queue buffers for warm start."""
    if not isinstance(ckpt_state, dict):
        raise TypeError(f"Expected model state_dict to be dict, got {type(ckpt_state)}")
    skip_prefixes = ("queue", "queue_ptr", "queue_cluster_ids")
    return {
        k: v
        for k, v in ckpt_state.items()
        if isinstance(k, str) and not k.startswith(skip_prefixes)
    }


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
    """Load checkpoint and return the epoch to resume from."""
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    # Try secure loading first, fallback to unsafe for old checkpoints
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError) as e:
        logger.warning(
            f"Checkpoint uses unsafe pickle format. Loading with weights_only=False. Error: {e}"
        )
        ckpt = torch.load(path, map_location=device, weights_only=False)
    ckpt_epoch = int(ckpt.get("epoch", 0))
    ckpt_state = _filter_warm_start_state(ckpt["model"]) if warm_start else ckpt["model"]

    if not isinstance(ckpt_state, dict):
        raise TypeError(f"Expected model state_dict to be dict, got {type(ckpt_state)}")
    model.load_state_dict(ckpt_state, strict=False)

    if warm_start:
        if pseudo_manager is not None:
            pseudo_manager.clear()
        logger.info(f"Warm start from {path.name} (checkpoint epoch {ckpt_epoch})")
        return 0

    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if pseudo_manager is not None and "pseudo" in ckpt:
        from src.pseudo import PseudoIDState

        pseudo_manager.state = PseudoIDState.from_dict(ckpt["pseudo"])

    logger.info(f"Resumed from {path.name} at epoch {ckpt_epoch}")
    return ckpt_epoch


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
