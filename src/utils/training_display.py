"""Simple progress bar display for SSL face recognition training."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any


def _fmt_time(s: float) -> str:
    """Format seconds as human-readable string."""
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{int(s // 3600)}h{int((s % 3600) // 60)}m"


def _make_bar(value: float, max_val: float, width: int = 20) -> str:
    """Create a simple progress bar string."""
    if max_val <= 0:
        return "░" * width
    pct = min(1.0, max(0.0, value / max_val))
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


@dataclass
class TrainingState:
    """Current training state for display."""

    # Core metrics
    images_per_sec: float = 0.0
    total_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    pos_sim: float = 0.0
    neg_sim: float = 0.0

    # Pseudo-ID
    pseudo_prob: float = 0.0
    num_clusters: int = 0
    clustered_pct: float = 0.0

    # Progress
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    epoch_start_time: float = 0.0
    train_start_time: float = 0.0


class TrainingDisplay:
    """Simple single-line progress display for training."""

    def __init__(
        self,
        world_size: int = 1,
        is_main: bool = True,
        refresh_rate: float = 4.0,
    ):
        self.world_size = world_size
        self.is_main = is_main
        self.state = TrainingState()
        self._last_update = 0.0
        self._min_interval = 1.0 / refresh_rate

    def start(self) -> None:
        """Start the display (no-op for simple display)."""
        pass

    def stop(self) -> None:
        """Stop the display and clear the line."""
        if self.is_main:
            sys.stdout.write("\r" + " " * 120 + "\r")
            sys.stdout.flush()

    def update(
        self,
        step: int,
        total_steps: int,
        stats: dict[str, Any],
        lr: float,
        grad_norm: float,
        img_count: int,
        epoch: int,
        pseudo_prob: float = 0.0,
        num_clusters: int = 0,
        clustered_pct: float = 0.0,
        temperature: float = 0.07,
        margin: float = 0.0,
        queue_size: int = 65536,
    ) -> None:
        """Update display with new training stats."""
        if not self.is_main:
            return

        now = time.perf_counter()

        # Rate limit updates
        if now - self._last_update < self._min_interval:
            return
        self._last_update = now

        s = self.state

        # Calculate metrics
        elapsed = now - s.epoch_start_time if s.epoch_start_time > 0 else 0.001
        s.images_per_sec = (img_count * self.world_size) / elapsed if elapsed > 0 else 0
        s.total_loss = stats.get("loss", 0.0)
        s.learning_rate = lr
        s.grad_norm = grad_norm
        s.pos_sim = stats.get("pos_sim", 0.0)
        s.neg_sim = stats.get("neg_sim", 0.0)
        s.pseudo_prob = pseudo_prob
        s.num_clusters = num_clusters
        s.clustered_pct = clustered_pct
        s.epoch = epoch
        s.step = step
        s.total_steps = total_steps

        # Calculate ETA
        if step > 0 and total_steps > 0:
            time_per_step = elapsed / step
            remaining = time_per_step * (total_steps - step)
            eta_str = _fmt_time(remaining)
        else:
            eta_str = "--"

        # Build progress bar
        pct = (step + 1) / total_steps * 100 if total_steps > 0 else 0
        bar = _make_bar(step + 1, total_steps, width=20)

        # Gap (pos_sim - neg_sim)
        gap = s.pos_sim - s.neg_sim

        # Build single line output
        line = (
            f"\r[E{epoch:03d}] {bar} {pct:5.1f}% | "
            f"{s.images_per_sec:,.0f} IPS | "
            f"L:{s.total_loss:.3f} | "
            f"gap:{gap:.3f} | "
            f"lr:{lr:.0e} | "
            f"gn:{grad_norm:.1f} | "
            f"ETA:{eta_str}"
        )

        # Add pseudo info if active
        if pseudo_prob > 0:
            line += f" | P:{pseudo_prob:.0%}"

        # Pad and write
        line = line.ljust(140)
        sys.stdout.write(line)
        sys.stdout.flush()

    def set_epoch_start(self, epoch_start_time: float, train_start_time: float) -> None:
        """Set the epoch and training start times."""
        self.state.epoch_start_time = epoch_start_time
        self.state.train_start_time = train_start_time
