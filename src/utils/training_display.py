"""Rich training display for SSL face recognition training."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _fmt_time(s: float) -> str:
    """Format seconds as human-readable string."""
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{int(s // 3600)}h{int((s % 3600) // 60)}m"


def _make_bar(value: float, max_val: float, width: int = 20, color: str = "green") -> str:
    """Create a progress bar string."""
    if max_val <= 0:
        return " " * width
    pct = min(1.0, max(0.0, value / max_val))
    filled = int(width * pct)
    return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/{color}]"


def _trend_arrow(current: float, previous: float, threshold: float = 0.001) -> str:
    """Return trend arrow based on change."""
    diff = current - previous
    if abs(diff) < threshold:
        return "[dim]→[/dim]"
    return "[red]↑[/red]" if diff > 0 else "[green]↓[/green]"


@dataclass
class TrainingState:
    """Current training state for display."""

    # Speed metrics
    batch_time: float = 0.0
    images_per_sec: float = 0.0
    images_per_sec_per_gpu: list[float] = field(default_factory=list)

    # Loss metrics
    total_loss: float = 0.0
    moco_loss: float = 0.0
    pseudo_loss: float = 0.0
    loss_history: deque = field(default_factory=lambda: deque(maxlen=50))

    # Training state
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    temperature: float = 0.07
    margin: float = 0.0
    queue_ptr: int = 0
    queue_size: int = 65536

    # Pseudo-ID
    pseudo_prob: float = 0.0
    num_clusters: int = 0
    clustered_pct: float = 0.0
    neg_masked_pct: float = 0.0

    # GPU memory (list for multi-GPU)
    gpu_memory_used: list[float] = field(default_factory=list)
    gpu_memory_total: list[float] = field(default_factory=list)

    # Time
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    epoch_start_time: float = 0.0
    train_start_time: float = 0.0

    # Extra stats
    pos_sim: float = 0.0
    neg_sim: float = 0.0
    emb_std: float = 0.0


class TrainingDisplay:
    """Rich panel display for training progress."""

    def __init__(
        self,
        world_size: int = 1,
        is_main: bool = True,
        refresh_rate: float = 4.0,
    ):
        """Initialize training display.

        Args:
            world_size: Number of GPUs for distributed training.
            is_main: Whether this is the main process (only rank 0 shows display).
            refresh_rate: Refresh rate in Hz.
        """
        self.world_size = world_size
        self.is_main = is_main
        self.refresh_rate = refresh_rate
        self.state = TrainingState()
        self._console = Console(force_terminal=True, highlight=False)
        self._live: Live | None = None
        self._last_loss = 0.0

    def start(self) -> None:
        """Start the live display."""
        if not self.is_main:
            return
        self._live = Live(
            self._build_panel(),
            console=self._console,
            refresh_per_second=self.refresh_rate,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

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
        s = self.state

        # Timing
        elapsed = now - s.epoch_start_time if s.epoch_start_time > 0 else 0.001
        s.batch_time = elapsed / max(1, step + 1)
        s.images_per_sec = (img_count * self.world_size) / elapsed if elapsed > 0 else 0

        # Per-GPU IPS (approximate - assumes equal distribution)
        ips_per_gpu = s.images_per_sec / self.world_size if self.world_size > 0 else 0
        s.images_per_sec_per_gpu = [ips_per_gpu] * self.world_size

        # Loss
        s.total_loss = stats.get("loss", 0.0)
        s.moco_loss = stats.get("loss", 0.0)  # In this model, moco_loss is the total loss
        s.pseudo_loss = 0.0  # Not separate in current model
        s.loss_history.append(s.total_loss)

        # Training state
        s.learning_rate = lr
        s.grad_norm = grad_norm
        s.temperature = temperature
        s.margin = margin
        s.queue_ptr = stats.get("queue_ptr", 0)
        s.queue_size = queue_size

        # Pseudo-ID
        s.pseudo_prob = pseudo_prob
        s.num_clusters = num_clusters
        s.clustered_pct = clustered_pct
        s.neg_masked_pct = stats.get("neg_masked_pct", 0.0)

        # GPU memory
        s.gpu_memory_used = []
        s.gpu_memory_total = []
        if torch.cuda.is_available():
            for i in range(self.world_size):
                try:
                    if i == 0 or self.world_size == 1:
                        # Only access the local GPU
                        used = torch.cuda.memory_allocated() / (1024**3)
                        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    else:
                        # For other GPUs in DDP, we can only estimate
                        used = torch.cuda.memory_allocated() / (1024**3)
                        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    s.gpu_memory_used.append(used)
                    s.gpu_memory_total.append(total)
                except Exception:
                    s.gpu_memory_used.append(0.0)
                    s.gpu_memory_total.append(24.0)

        # Time
        s.epoch = epoch
        s.step = step
        s.total_steps = total_steps

        # Extra stats
        s.pos_sim = stats.get("pos_sim", 0.0)
        s.neg_sim = stats.get("neg_sim", 0.0)
        s.emb_std = stats.get("emb_std", 0.0)

        # Update display
        if self._live is not None:
            self._live.update(self._build_panel())

        self._last_loss = s.total_loss

    def set_epoch_start(self, epoch_start_time: float, train_start_time: float) -> None:
        """Set the epoch and training start times."""
        self.state.epoch_start_time = epoch_start_time
        self.state.train_start_time = train_start_time

    def _build_panel(self) -> Panel:
        """Build the complete training panel."""
        s = self.state

        # Create main grid
        grid = Table.grid(padding=(0, 2), expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        # Row 1: Speed | Loss | Training State
        speed_section = self._build_speed_section()
        loss_section = self._build_loss_section()
        training_section = self._build_training_section()
        grid.add_row(speed_section, loss_section, training_section)

        # Row 2: Pseudo-ID | GPU Memory | Time
        pseudo_section = self._build_pseudo_section()
        gpu_section = self._build_gpu_section()
        time_section = self._build_time_section()
        grid.add_row(pseudo_section, gpu_section, time_section)

        # Determine phase
        phase = "A:Stabilize" if s.epoch < 5 else "B:Bootstrap" if s.epoch < 35 else "C:Refine"

        # Progress info
        pct = (s.step + 1) / s.total_steps * 100 if s.total_steps > 0 else 0
        progress_bar = _make_bar(s.step + 1, s.total_steps, width=40, color="cyan")

        # Title with progress
        title = Text()
        title.append(f" Epoch {s.epoch:03d} ", style="bold white on blue")
        title.append(f" [{phase}] ", style="bold yellow")
        title.append(f" {progress_bar} ", style="")
        title.append(f" {s.step + 1}/{s.total_steps} ({pct:.1f}%) ", style="dim")

        return Panel(
            grid,
            title=title,
            border_style="blue",
            padding=(0, 1),
        )

    def _build_speed_section(self) -> Panel:
        """Build speed metrics section."""
        s = self.state
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("Label", style="dim", width=12)
        table.add_column("Value", style="bold")

        # Per-GPU IPS
        if self.world_size > 1:
            for i, ips in enumerate(s.images_per_sec_per_gpu[:4]):  # Show max 4 GPUs
                table.add_row(f"GPU {i} IPS", f"{ips:,.0f}")
            if self.world_size > 4:
                table.add_row("...", f"+{self.world_size - 4} GPUs")

        # Total IPS
        table.add_row("Total IPS", f"[bold cyan]{s.images_per_sec:,.0f}[/bold cyan]")

        # Batch time
        table.add_row("Batch time", f"{s.batch_time * 1000:.1f}ms")

        return Panel(table, title="[bold]Speed[/bold]", border_style="green", padding=(0, 0))

    def _build_loss_section(self) -> Panel:
        """Build loss metrics section."""
        s = self.state
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("Label", style="dim", width=10)
        table.add_column("Value", width=8)
        table.add_column("Bar", width=12)
        table.add_column("Trend", width=3)

        # Loss trend
        prev_loss = list(s.loss_history)[-2] if len(s.loss_history) > 1 else s.total_loss
        trend = _trend_arrow(s.total_loss, prev_loss, threshold=0.01)

        # Total loss with color coding
        loss_color = "green" if s.total_loss < 3.0 else "yellow" if s.total_loss < 5.0 else "red"
        bar = _make_bar(s.total_loss, 8.0, width=12, color=loss_color)
        table.add_row("Total", f"[{loss_color}]{s.total_loss:.4f}[/{loss_color}]", bar, trend)

        # Similarities
        gap = s.pos_sim - s.neg_sim
        gap_color = "green" if gap > 0.3 else "yellow" if gap > 0.1 else "red"
        table.add_row("Pos sim", f"{s.pos_sim:.3f}", "", "")
        table.add_row("Neg sim", f"{s.neg_sim:.3f}", "", "")
        table.add_row("Gap", f"[{gap_color}]{gap:.3f}[/{gap_color}]", "", "")

        return Panel(table, title="[bold]Loss[/bold]", border_style="yellow", padding=(0, 0))

    def _build_training_section(self) -> Panel:
        """Build training state section."""
        s = self.state
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("Label", style="dim", width=10)
        table.add_column("Value", style="bold")

        # Learning rate
        table.add_row("LR", f"{s.learning_rate:.2e}")

        # Gradient norm with color
        gn_color = "green" if s.grad_norm < 10 else "yellow" if s.grad_norm < 50 else "red"
        table.add_row("Grad norm", f"[{gn_color}]{s.grad_norm:.2f}[/{gn_color}]")

        # Temperature and margin
        table.add_row("Temp (t)", f"{s.temperature:.3f}")
        table.add_row("Margin (m)", f"{s.margin:.2f}")

        # Queue utilization
        queue_pct = s.queue_ptr / s.queue_size * 100 if s.queue_size > 0 else 0
        queue_bar = _make_bar(s.queue_ptr, s.queue_size, width=8, color="cyan")
        table.add_row("Queue", f"{queue_bar} {queue_pct:.0f}%")

        return Panel(
            table, title="[bold]Training[/bold]", border_style="magenta", padding=(0, 0)
        )

    def _build_pseudo_section(self) -> Panel:
        """Build pseudo-ID section."""
        s = self.state
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("Label", style="dim", width=12)
        table.add_column("Value", style="bold")

        if s.pseudo_prob > 0 or s.num_clusters > 0:
            # Mix ratio bar
            mix_bar = _make_bar(s.pseudo_prob, 1.0, width=10, color="cyan")
            table.add_row("Mix ratio", f"{mix_bar} {s.pseudo_prob:.0%}")

            # Clusters
            table.add_row("Clusters", f"{s.num_clusters:,}")

            # Clustered %
            clust_color = "green" if s.clustered_pct > 0.7 else "yellow" if s.clustered_pct > 0.4 else "dim"
            table.add_row("Clustered", f"[{clust_color}]{s.clustered_pct:.1%}[/{clust_color}]")

            # Neg masked
            if s.neg_masked_pct > 0:
                table.add_row("Neg masked", f"{s.neg_masked_pct:.1%}")
        else:
            table.add_row("[dim]Inactive[/dim]", "[dim]--[/dim]")

        return Panel(table, title="[bold]Pseudo-ID[/bold]", border_style="cyan", padding=(0, 0))

    def _build_gpu_section(self) -> Panel:
        """Build GPU memory section."""
        s = self.state
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("GPU", style="dim", width=6)
        table.add_column("Bar", width=10)
        table.add_column("Usage", style="bold", width=12)

        if s.gpu_memory_used:
            for i, (used, total) in enumerate(
                zip(s.gpu_memory_used[:4], s.gpu_memory_total[:4], strict=False)
            ):
                pct = used / total if total > 0 else 0
                mem_color = "green" if pct < 0.7 else "yellow" if pct < 0.9 else "red"
                bar = _make_bar(used, total, width=10, color=mem_color)
                table.add_row(f"GPU {i}", bar, f"{used:.1f}/{total:.1f}GB")
        else:
            table.add_row("[dim]N/A[/dim]", "", "[dim]No GPU[/dim]")

        return Panel(table, title="[bold]GPU Memory[/bold]", border_style="red", padding=(0, 0))

    def _build_time_section(self) -> Panel:
        """Build time section."""
        s = self.state
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("Label", style="dim", width=10)
        table.add_column("Value", style="bold")

        now = time.perf_counter()

        # Epoch elapsed
        epoch_elapsed = now - s.epoch_start_time if s.epoch_start_time > 0 else 0
        table.add_row("Elapsed", _fmt_time(epoch_elapsed))

        # ETA for epoch
        if s.step > 0 and s.total_steps > 0:
            time_per_step = epoch_elapsed / s.step
            remaining_steps = s.total_steps - s.step
            eta = time_per_step * remaining_steps
            table.add_row("Epoch ETA", _fmt_time(eta))
        else:
            table.add_row("Epoch ETA", "--")

        # Total training time
        if s.train_start_time > 0:
            total_elapsed = now - s.train_start_time
            table.add_row("Total", _fmt_time(total_elapsed))

        # Emb std (quality indicator)
        std_color = "green" if 0.03 < s.emb_std < 0.2 else "yellow"
        table.add_row("Emb std", f"[{std_color}]{s.emb_std:.4f}[/{std_color}]")

        return Panel(table, title="[bold]Time[/bold]", border_style="white", padding=(0, 0))
