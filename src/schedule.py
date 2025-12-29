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
