"""Curriculum learning schedule."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig


@dataclass(frozen=True)
class TrainSchedulePhase:
    """Epoch-based curriculum mixing schedule."""

    start_epoch: int
    end_epoch: int
    p_digiface: float


def build_curriculum_schedule(cfg: DictConfig) -> tuple[TrainSchedulePhase, ...]:
    """Build curriculum schedule from config."""
    schedule_cfg = cfg.get("curriculum", {}).get("schedule", [])
    if not schedule_cfg:
        return (
            TrainSchedulePhase(start_epoch=0, end_epoch=15, p_digiface=1.0),
            TrainSchedulePhase(start_epoch=16, end_epoch=30, p_digiface=0.5),
            TrainSchedulePhase(start_epoch=31, end_epoch=10_000, p_digiface=0.3),
        )

    return tuple(
        TrainSchedulePhase(
            start_epoch=int(p.start_epoch),
            end_epoch=int(p.end_epoch),
            p_digiface=float(p.p_digiface),
        )
        for p in schedule_cfg
    )


def curriculum_p_digiface(epoch: int, schedule: tuple[TrainSchedulePhase, ...]) -> float:
    """Return digiface sampling probability for a given epoch."""
    for phase in schedule:
        if phase.start_epoch <= epoch <= phase.end_epoch:
            return float(phase.p_digiface)
    return 1.0


# =============================================================================
# Pseudo-ID Schedule
# =============================================================================


@dataclass(frozen=True)
class PseudoMixPhase:
    """Phase for pseudo positive pair mixing schedule."""

    start_epoch: int
    end_epoch: int
    pseudo_prob: float


def build_pseudo_schedule(cfg: DictConfig) -> tuple[PseudoMixPhase, ...]:
    """Build pseudo-ID positive mixing schedule from config.

    Config format:
        pseudo:
          pos_mix:
            schedule:
              - start_epoch: 0
                end_epoch: 4
                pseudo_prob: 0.0
              - start_epoch: 5
                end_epoch: 9
                pseudo_prob: 0.30
              ...
    """
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


def get_sim_threshold(epoch: int, cfg: DictConfig) -> float:
    """Compute linearly decaying similarity threshold for pseudo-ID mining.

    Config format:
        pseudo:
          sim_threshold:
            start: 0.72
            end: 0.52
            decay_end_epoch: 50
    """
    pseudo_cfg = cfg.get("pseudo", {})
    threshold_cfg = pseudo_cfg.get("sim_threshold", {})

    start = float(threshold_cfg.get("start", 0.72))
    end = float(threshold_cfg.get("end", 0.52))
    decay_end = int(threshold_cfg.get("decay_end_epoch", 50))

    if epoch >= decay_end:
        return end

    # Linear interpolation
    if decay_end <= 0:
        return start

    alpha = epoch / decay_end
    return start + alpha * (end - start)


def get_refresh_epochs(cfg: DictConfig) -> set[int]:
    """Get set of epochs at which to refresh pseudo-IDs.

    Config format:
        pseudo:
          refresh_epochs: [4, 9, 14, 24, 34, 49, 64]
    """
    pseudo_cfg = cfg.get("pseudo", {})
    if not pseudo_cfg.get("enabled", False):
        return set()

    refresh_list = pseudo_cfg.get("refresh_epochs", [])
    return set(int(e) for e in refresh_list)
