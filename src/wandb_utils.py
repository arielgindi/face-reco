"""Weights & Biases integration utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

import wandb

logger = logging.getLogger(__name__)


def get_system_info() -> dict[str, Any]:
    """Collect system information for W&B logging."""
    info: dict[str, Any] = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_memory_total_gb": round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
                ),
            }
        )

    return info


def log_gpu_memory() -> dict[str, float]:
    """Get current GPU memory usage for logging."""
    if not torch.cuda.is_available():
        return {}

    return {
        "gpu/memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
        "gpu/memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
        "gpu/max_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    }


def init_wandb(cfg: DictConfig, out_dir: Path) -> bool:
    """Initialize Weights & Biases if enabled. Returns True if W&B is active."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        logger.info("W&B disabled")
        return False

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    system_info = get_system_info()
    config_dict["system"] = system_info

    wandb.init(
        project=wandb_cfg.get("project", "sniperface"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        tags=list(wandb_cfg.get("tags", [])),
        notes=wandb_cfg.get("notes"),
        config=config_dict,
        dir=str(out_dir),
        resume="allow",
    )

    for key, value in system_info.items():
        wandb.run.summary[f"system/{key}"] = value

    logger.info(f"W&B initialized: {wandb.run.url}")
    return True


def log_wandb(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log metrics to W&B if active."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb() -> None:
    """Finish W&B run if active."""
    if wandb.run is not None:
        wandb.finish()
