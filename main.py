#!/usr/bin/env python3
"""SniperFace - Label-free face encoder training with MoCo + MarginNCE.

Uses Hydra for configuration and Weights & Biases for experiment tracking.

Commands:
  uv run python main.py command=train                    # Train with defaults
  uv run python main.py command=train train.epochs=100   # Override epochs
  uv run python main.py command=train wandb.enabled=false # Disable W&B
  uv run python main.py command=embed                    # Export embeddings
  uv run python main.py command=eval                     # Evaluate metrics

Reference: https://ar5iv.org/pdf/2211.07371 (USynthFace - MarginNCE)
"""

from __future__ import annotations

# Enable PTX JIT for Blackwell GPUs (sm_120+) before any CUDA imports
import os
import subprocess

def _enable_blackwell_ptx_jit() -> None:
    """Enable PTX JIT compilation for Blackwell GPUs before CUDA init."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            cc = result.stdout.strip().split("\n")[0]  # e.g., "12.0"
            major = int(cc.split(".")[0])
            if major >= 10:  # Blackwell is CC 10.0+
                os.environ["CUDA_FORCE_PTX_JIT"] = "1"
    except Exception:
        pass  # Silently ignore - not critical

_enable_blackwell_ptx_jit()

import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.commands import cmd_train

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    command = cfg.get("command", "train")

    logger.info(f"Running command: {command}")
    logger.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if command == "train":
        cmd_train(cfg)
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
