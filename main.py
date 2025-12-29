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

import logging
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from src.commands import cmd_embed, cmd_eval, cmd_train

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    command = cfg.get("command", "train")

    logger.info(f"Running command: {command}")
    logger.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if command == "train":
        cmd_train(cfg)
    elif command == "embed":
        cmd_embed(cfg)
    elif command == "eval":
        cmd_eval(cfg)
    else:
        logger.error(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
