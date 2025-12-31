"""Curriculum learning schedule."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class PseudoMixPhase:
    """Phase for pseudo positive pair mixing schedule."""

    start_epoch: int
    end_epoch: int
    pseudo_prob: float


def build_pseudo_schedule(cfg: DictConfig) -> tuple[PseudoMixPhase, ...]:
    """Build pseudo-ID positive mixing schedule from config."""
    pseudo_cfg = cfg.get("pseudo", {})
    if not pseudo_cfg.get("enabled", False):
        return ()

    schedule_cfg = pseudo_cfg.get("pos_mix", {}).get("schedule", [])
    if not schedule_cfg:
        return ()

    return tuple(
        PseudoMixPhase(
            start_epoch=int(p.start_epoch),
            end_epoch=int(p.end_epoch),
            pseudo_prob=float(p.pseudo_prob),
        )
        for p in schedule_cfg
    )


def get_pseudo_prob(epoch: int, schedule: tuple[PseudoMixPhase, ...]) -> float:
    """Return pseudo positive probability for a given epoch."""
    for phase in schedule:
        if phase.start_epoch <= epoch <= phase.end_epoch:
            return float(phase.pseudo_prob)
    return 0.0


def should_refresh_pseudo(epoch: int, cfg: DictConfig) -> bool:
    """Check if pseudo-IDs should be refreshed (every 2 epochs starting from epoch 2)."""
    pseudo_cfg = cfg.get("pseudo", {})
    if not pseudo_cfg.get("enabled", False):
        return False
    return epoch > 0 and epoch % 2 == 0


def get_sim_threshold(cfg: DictConfig) -> float:
    """Get fixed similarity threshold for pseudo-ID mining.

    Config format:
        pseudo:
          sim_threshold: 0.60

    Args:
        cfg: Training configuration

    Returns:
        Fixed threshold value (default: 0.60)
    """
    pseudo_cfg = cfg.get("pseudo", {})
    return float(pseudo_cfg.get("sim_threshold", 0.60))
