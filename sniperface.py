#!/usr/bin/env python3
"""SniperFace - Label-free face encoder training with MoCo + MarginNCE.

Uses Hydra for configuration and Weights & Biases for experiment tracking.

Commands:
  uv run python sniperface.py train                    # Train with defaults
  uv run python sniperface.py train train.epochs=100   # Override epochs
  uv run python sniperface.py train wandb.enabled=false # Disable W&B
  uv run python sniperface.py embed                    # Export embeddings
  uv run python sniperface.py eval                     # Evaluate metrics

Reference: https://ar5iv.org/pdf/2211.07371 (USynthFace - MarginNCE)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torchvision.transforms as T
import wandb
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataio
import moco

logger = logging.getLogger(__name__)


# =============================================================================
# Training Schedule
# =============================================================================


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
        # Default schedule
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
# Utility Functions
# =============================================================================


def set_seed(seed: int, deterministic: bool) -> None:
    """Set Python/NumPy/PyTorch seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def select_device() -> torch.device:
    """Pick CUDA if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def configure_precision(cfg: DictConfig) -> tuple[bool, torch.dtype, bool]:
    """Return (amp_enabled, amp_dtype, tf32_enabled)."""
    precision_cfg = cfg.get("train", {}).get("precision", {})
    amp_enabled = bool(precision_cfg.get("amp", True))
    amp_dtype_str = str(precision_cfg.get("amp_dtype", "fp16")).lower()
    if amp_dtype_str not in {"fp16", "bf16"}:
        raise ValueError("train.precision.amp_dtype must be 'fp16' or 'bf16'.")

    amp_dtype = torch.float16 if amp_dtype_str == "fp16" else torch.bfloat16
    tf32_enabled = bool(precision_cfg.get("tf32_matmul", True))
    return amp_enabled, amp_dtype, tf32_enabled


def compute_epoch_batch_counts(
    *,
    base_samples: int,
    batch_size: int,
    grad_accum_steps: int,
) -> tuple[int, int]:
    """Convert a target sample count into (num_batches, num_samples)."""
    if base_samples < batch_size:
        raise ValueError("samples_per_epoch must be >= batch_size.")

    num_batches = base_samples // batch_size
    num_batches = (num_batches // grad_accum_steps) * grad_accum_steps
    if num_batches <= 0:
        raise ValueError("samples_per_epoch too small after enforcing grad_accum_steps.")
    num_samples = num_batches * batch_size
    return num_batches, num_samples


# =============================================================================
# Custom Augmentations
# =============================================================================


class RandomJPEGCompression:
    """Randomly JPEG-compress a PIL image."""

    def __init__(self, p: float, quality_min: int, quality_max: int) -> None:
        self.p = float(p)
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)

    def __call__(self, img: dataio.PILImage) -> dataio.PILImage:
        if np.random.rand() >= self.p:
            return img

        q = int(np.random.randint(self.quality_min, self.quality_max + 1))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out.load()
        return out


class RandomISONoise:
    """Add pixel-wise Gaussian noise to a tensor image."""

    def __init__(self, p: float, sigma_min: float, sigma_max: float) -> None:
        self.p = float(p)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self.p:
            return x
        sigma = float(np.random.uniform(self.sigma_min, self.sigma_max))
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp_(0.0, 1.0)


# =============================================================================
# Transform Builders
# =============================================================================


def build_view_transform(view_cfg: DictConfig, *, input_size: tuple[int, int]) -> T.Compose:
    """Build one view augmentation pipeline from config."""
    h, w = int(input_size[0]), int(input_size[1])

    ops: list[Any] = []

    # Random resized crop
    rrc = view_cfg.get("random_resized_crop", {})
    scale = tuple(rrc.get("scale", [0.2, 1.0]))
    ratio = tuple(rrc.get("ratio", [0.75, 1.33]))
    ops.append(T.RandomResizedCrop(size=(h, w), scale=scale, ratio=ratio))

    # Horizontal flip
    flip_p = float(view_cfg.get("horizontal_flip_p", 0.5))
    ops.append(T.RandomHorizontalFlip(p=flip_p))

    # RandAugment
    ra_cfg = view_cfg.get("randaugment", {})
    if ra_cfg:
        num_ops = int(ra_cfg.get("num_ops", 4))
        magnitude = int(ra_cfg.get("magnitude", 16))
        ops.append(T.RandAugment(num_ops=num_ops, magnitude=magnitude))

    # Color jitter
    cj_cfg = view_cfg.get("color_jitter", {})
    if cj_cfg and float(cj_cfg.get("p", 0.0)) > 0.0:
        p = float(cj_cfg.get("p", 0.8))
        cj = T.ColorJitter(
            brightness=float(cj_cfg.get("brightness", 0.4)),
            contrast=float(cj_cfg.get("contrast", 0.4)),
            saturation=float(cj_cfg.get("saturation", 0.4)),
            hue=float(cj_cfg.get("hue", 0.1)),
        )
        ops.append(T.RandomApply([cj], p=p))

    # Grayscale
    gs_p = float(view_cfg.get("grayscale_p", 0.0))
    if gs_p > 0.0:
        ops.append(T.RandomGrayscale(p=gs_p))

    # Gaussian blur
    blur_cfg = view_cfg.get("gaussian_blur", {})
    if blur_cfg and float(blur_cfg.get("p", 0.0)) > 0.0:
        p = float(blur_cfg.get("p", 0.5))
        sigma = tuple(blur_cfg.get("sigma", [0.1, 2.0]))
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=sigma)], p=p))

    # JPEG compression
    jpeg_cfg = view_cfg.get("jpeg_compression", {})
    if jpeg_cfg and float(jpeg_cfg.get("p", 0.0)) > 0.0:
        p = float(jpeg_cfg.get("p", 0.3))
        qmin, qmax = jpeg_cfg.get("quality", [30, 95])
        ops.append(RandomJPEGCompression(p=p, quality_min=int(qmin), quality_max=int(qmax)))

    # Convert to tensor
    ops.append(T.ToTensor())

    # ISO noise
    iso_cfg = view_cfg.get("iso_noise", {})
    if iso_cfg and float(iso_cfg.get("p", 0.0)) > 0.0:
        p = float(iso_cfg.get("p", 0.2))
        smin, smax = iso_cfg.get("sigma", [0.0, 0.08])
        ops.append(RandomISONoise(p=p, sigma_min=float(smin), sigma_max=float(smax)))

    # Cutout
    cut_cfg = view_cfg.get("cutout", {})
    if cut_cfg and float(cut_cfg.get("p", 0.0)) > 0.0:
        p = float(cut_cfg.get("p", 0.2))
        holes = int(cut_cfg.get("holes", 1))
        size_ratio = tuple(cut_cfg.get("size_ratio", [0.10, 0.30]))
        for _ in range(max(1, holes)):
            ops.append(T.RandomErasing(p=p, scale=size_ratio, ratio=(0.3, 3.3), value="random"))

    # Normalize
    ops.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return T.Compose(ops)


def build_embed_transform(input_size: tuple[int, int]) -> T.Compose:
    """Deterministic preprocessing for embedding export."""
    h, w = int(input_size[0]), int(input_size[1])
    return T.Compose(
        [
            T.Resize(size=(h, w), antialias=True),
            T.CenterCrop(size=(h, w)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


# =============================================================================
# Checkpointing
# =============================================================================


def load_checkpoint_for_resume(
    resume_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> int:
    """Load checkpoint and return the epoch to resume from.

    Returns the next epoch to train (checkpoint_epoch, since we save after epoch completion).
    """
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    logger.info(f"Resuming from checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Load model state
    model.load_state_dict(ckpt["model"])
    logger.info("Loaded model state")

    # Load optimizer state
    optimizer.load_state_dict(ckpt["optimizer"])
    logger.info("Loaded optimizer state")

    # Load scaler state if available
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
        logger.info("Loaded AMP scaler state")

    start_epoch = int(ckpt["epoch"])
    logger.info(f"Resuming from epoch {start_epoch}")

    return start_epoch


def save_checkpoint(
    out_dir: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: DictConfig,
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
    digiface_ds: dataio.ParquetTwoViewDataset,
    digi2real_ds: dataio.ParquetTwoViewDataset | None,
    num_workers: int,
    device: torch.device,
) -> None:
    """Pre-warm both datasets by iterating one sample from each.

    This initializes file handles and DataLoader workers for both datasets
    BEFORE the curriculum switch at epoch 16, preventing Windows deadlocks.
    """
    if num_workers <= 0:
        logger.info("Skipping dataset pre-warm (num_workers=0)")
        return

    logger.info("Pre-warming datasets to initialize workers...")

    # Pre-warm digiface
    warm_ds = dataio.CurriculumMixTwoViewDataset(
        digiface=digiface_ds,
        digi2real=None,
        p_digiface=1.0,
        num_samples=num_workers * 2,  # Enough to engage all workers
        seed=0,
    )
    warm_loader = DataLoader(
        warm_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,  # Don't persist for warmup
    )
    for batch in warm_loader:
        break  # Just need one batch to initialize
    del warm_loader, warm_ds
    logger.info("  DigiFace dataset warmed")

    # Pre-warm digi2real if available
    if digi2real_ds is not None:
        warm_ds = dataio.CurriculumMixTwoViewDataset(
            digiface=digiface_ds,
            digi2real=digi2real_ds,
            p_digiface=0.0,  # Force digi2real
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
        for batch in warm_loader:
            break
        del warm_loader, warm_ds
        logger.info("  Digi2Real dataset warmed")

    logger.info("Dataset pre-warming complete")


# =============================================================================
# W&B Integration
# =============================================================================


def init_wandb(cfg: DictConfig, out_dir: Path) -> bool:
    """Initialize Weights & Biases if enabled. Returns True if W&B is active."""
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        logger.info("W&B disabled")
        return False

    # Convert config to dict for W&B
    config_dict = OmegaConf.to_container(cfg, resolve=True)

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


# =============================================================================
# Training Command
# =============================================================================


def cmd_train(cfg: DictConfig) -> None:
    """Train MoCo + MarginNCE encoder.

    Training automatically excludes test identities based on split config.
    """
    exp_cfg = cfg.get("experiment", {})
    seed = int(exp_cfg.get("seed", 42))
    deterministic = bool(exp_cfg.get("deterministic", False))
    set_seed(seed, deterministic)

    amp_enabled, amp_dtype, tf32_enabled = configure_precision(cfg)
    if tf32_enabled and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = select_device()
    logger.info(f"Device: {device}")

    # Output directory (Hydra sets this)
    out_dir = Path(os.getcwd())
    logger.info(f"Output directory: {out_dir}")

    # Initialize W&B
    wandb_active = init_wandb(cfg, out_dir)
    wandb_cfg = cfg.get("wandb", {})
    log_every = int(wandb_cfg.get("log_every_steps", 50))

    # Data configuration
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})
    streaming_cfg = data_cfg.get("streaming", {})

    digiface_glob = data_cfg.get("digiface_glob")
    digi2real_glob = data_cfg.get("digi2real_glob")

    if not digiface_glob:
        raise ValueError("data.digiface_glob not specified in config.")

    # Split configuration
    train_ratio = float(split_cfg.get("train_ratio", 0.75))
    split_seed = int(split_cfg.get("seed", seed))
    cache_dir = Path(split_cfg.get("cache_dir", ".cache/splits"))

    data_globs = [digiface_glob]
    if digi2real_glob:
        data_globs.append(digi2real_glob)

    # Get or create identity splits
    splits_path = dataio.get_or_create_splits(
        globs=data_globs,
        cache_dir=cache_dir,
        train_ratio=train_ratio,
        seed=split_seed,
    )

    train_identities = dataio.get_identity_set(splits_path, "train")
    logger.info(
        f"Training on {len(train_identities)} identities "
        f"({train_ratio:.0%} of total, test set protected)"
    )

    # Save splits to output
    shutil.copy2(splits_path, out_dir / "identity_splits.parquet")

    # Build model
    model = moco.build_moco(cfg, device=device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    if wandb_active:
        wandb.watch(model, log="gradients", log_freq=log_every * 10)

    # Training config
    batch_size = int(cfg.train.batch.size)
    grad_accum = int(cfg.train.batch.grad_accum_steps)
    epochs = int(cfg.train.epochs)

    optim_cfg = cfg.train.optimizer
    optimizer = SGD(
        model.parameters(),
        lr=float(optim_cfg.lr),
        momentum=float(optim_cfg.momentum),
        weight_decay=float(optim_cfg.weight_decay),
        nesterov=bool(optim_cfg.nesterov),
    )

    scaler: torch.cuda.amp.GradScaler | None = None
    if amp_enabled and device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.cuda.amp.GradScaler()

    # Handle resume from checkpoint
    resume_path = cfg.train.get("resume")
    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint_for_resume(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

    # Transforms
    input_size = tuple(cfg.model.backbone.input_size)
    aug_cfg = cfg.augmentation
    t_q = build_view_transform(aug_cfg.view_1, input_size=input_size)
    t_k = build_view_transform(aug_cfg.view_2, input_size=input_size)

    # Streaming parameters
    stream_params = dataio.StreamParams(
        shuffle_files=True,
        batch_read_rows=int(streaming_cfg.get("batch_read_rows", 2048)),
        shuffle_within_batch=True,
        shuffle_buffer_size=int(streaming_cfg.get("shuffle_buffer_size", 10000)),
        seed=seed,
    )

    # Datasets with train identity filtering
    digiface_ds = dataio.ParquetTwoViewDataset(
        digiface_glob,
        transform_q=t_q,
        transform_k=t_k,
        stream=stream_params,
        allowed_identities=train_identities,
    )

    digi2real_ds: dataio.ParquetTwoViewDataset | None = None
    if digi2real_glob:
        digi2real_ds = dataio.ParquetTwoViewDataset(
            digi2real_glob,
            transform_q=t_q,
            transform_k=t_k,
            stream=stream_params,
            allowed_identities=train_identities,
        )

    # Compute epoch size
    base_samples = int(cfg.train.get("samples_per_epoch", 0))
    if base_samples <= 0:
        base_samples = dataio.count_parquet_rows(digiface_glob)

    num_batches, num_samples = compute_epoch_batch_counts(
        base_samples=base_samples,
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
    )
    logger.info(
        f"Epoch: {num_samples:,} samples, {num_batches:,} batches, "
        f"batch_size={batch_size}, grad_accum={grad_accum}"
    )

    # Learning rate schedule
    sched_cfg = cfg.train.lr_schedule
    milestones = list(sched_cfg.milestones)
    gamma = float(sched_cfg.gamma)
    warm_cfg = sched_cfg.warmup
    warm_enabled = bool(warm_cfg.enabled)
    warm_epochs = int(warm_cfg.epochs)
    warm_start_lr = float(warm_cfg.start_lr)
    base_lr = float(optim_cfg.lr)

    def lr_for_epoch(epoch: int) -> float:
        decay_steps = sum(epoch >= m for m in milestones)
        lr = base_lr * (gamma**decay_steps)
        if warm_enabled and epoch < warm_epochs:
            if warm_epochs <= 1:
                return lr
            alpha = epoch / float(warm_epochs - 1)
            return warm_start_lr + alpha * (lr - warm_start_lr)
        return lr

    grad_clip_norm = float(cfg.train.regularization.grad_clip_norm)
    save_every = int(cfg.train.checkpointing.save_every_epochs)
    keep_last = int(cfg.train.checkpointing.keep_last)
    num_workers = int(streaming_cfg.get("num_workers", 4))

    # Pre-warm datasets to prevent Windows multiprocessing deadlock
    # This initializes file handles for both datasets before training starts
    prewarm_datasets(digiface_ds, digi2real_ds, num_workers, device)

    # Curriculum schedule
    schedule = build_curriculum_schedule(cfg)

    # Calculate global_step for resume (approximate based on batches per epoch)
    steps_per_epoch = num_batches // grad_accum
    global_step = start_epoch * steps_per_epoch
    optimizer.zero_grad(set_to_none=True)

    if start_epoch > 0:
        logger.info(f"Resuming training from epoch {start_epoch} (global_step={global_step})")
    logger.info(f"Training for epochs {start_epoch} to {epochs - 1}")

    for epoch in range(start_epoch, epochs):
        lr = lr_for_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        p_digiface = curriculum_p_digiface(epoch, schedule)

        epoch_ds = dataio.CurriculumMixTwoViewDataset(
            digiface=digiface_ds,
            digi2real=digi2real_ds,
            p_digiface=p_digiface,
            num_samples=num_samples,
            seed=seed + epoch * 17,
        )

        # Windows-safe DataLoader settings:
        # - persistent_workers: keeps workers alive between epochs (prevents respawn deadlock)
        # - prefetch_factor: buffers batches for smoother data flow
        loader_kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": device.type == "cuda",
            "drop_last": True,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        loader = DataLoader(epoch_ds, **loader_kwargs)

        epoch_loss_sum = 0.0
        epoch_pos_sum = 0.0
        epoch_neg_sum = 0.0
        epoch_embstd_sum = 0.0
        n_logged = 0

        pbar = tqdm(loader, total=num_batches, desc=f"epoch {epoch:03d}", unit="batch")
        for step, (im_q, im_k) in enumerate(pbar):
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(
                enabled=amp_enabled and device.type == "cuda",
                dtype=amp_dtype,
            ):
                loss, stats = model(im_q, im_k)
                loss_to_backprop = loss / float(grad_accum)

            if scaler is not None:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if (step + 1) % grad_accum == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                model.update_momentum_encoder()
                global_step += 1

            epoch_loss_sum += float(stats["loss"])
            epoch_pos_sum += float(stats["pos_sim"])
            epoch_neg_sum += float(stats["neg_sim"])
            epoch_embstd_sum += float(stats["emb_std"])
            n_logged += 1

            # Log to W&B
            if (
                wandb_active
                and log_every > 0
                and global_step % log_every == 0
                and (step + 1) % grad_accum == 0
            ):
                log_wandb(
                    {
                        "train/loss": stats["loss"],
                        "train/pos_sim": stats["pos_sim"],
                        "train/neg_sim": stats["neg_sim"],
                        "train/emb_std": stats["emb_std"],
                        "train/lr": lr,
                        "train/p_digiface": p_digiface,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                pos=f"{stats['pos_sim']:.3f}",
                neg=f"{stats['neg_sim']:.3f}",
            )

        # Epoch summary
        if n_logged > 0:
            epoch_metrics = {
                "epoch/loss": epoch_loss_sum / n_logged,
                "epoch/pos_sim": epoch_pos_sum / n_logged,
                "epoch/neg_sim": epoch_neg_sum / n_logged,
                "epoch/emb_std": epoch_embstd_sum / n_logged,
                "epoch/lr": lr,
                "epoch/p_digiface": p_digiface,
            }
            if wandb_active:
                log_wandb(epoch_metrics, step=global_step)

            logger.info(
                f"Epoch {epoch:03d}: loss={epoch_metrics['epoch/loss']:.4f}, "
                f"pos_sim={epoch_metrics['epoch/pos_sim']:.3f}, "
                f"neg_sim={epoch_metrics['epoch/neg_sim']:.3f}"
            )

        # Checkpointing
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt_path = save_checkpoint(
                out_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
            )
            prune_checkpoints(out_dir / "checkpoints", keep_last=keep_last)
            logger.info(f"Saved checkpoint: {ckpt_path}")

            # Save to W&B artifacts
            if wandb_active and wandb_cfg.get("save_artifacts", False):
                artifact = wandb.Artifact(
                    f"model-epoch-{epoch + 1:03d}",
                    type="model",
                    metadata={"epoch": epoch + 1},
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)

    logger.info("Training complete!")
    finish_wandb()


# =============================================================================
# Embed Command
# =============================================================================


def cmd_embed(cfg: DictConfig) -> None:
    """Export embeddings from trained model.

    By default exports TEST identities only.
    """
    embed_cfg = cfg.get("embed", {})
    split_name = embed_cfg.get("split", "test")

    # Find the latest checkpoint
    ckpt_dir = Path(os.getcwd()) / "checkpoints"
    if not ckpt_dir.exists():
        # Try to find in Hydra output structure
        ckpt_dir = Path("checkpoints")

    ckpts = sorted(ckpt_dir.glob("epoch_*.pt")) if ckpt_dir.exists() else []
    if not ckpts:
        raise FileNotFoundError("No checkpoints found. Run training first.")

    ckpt_path = ckpts[-1]  # Latest
    logger.info(f"Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt.get("config", {})

    backbone_cfg = cfg.get("model", {}).get("backbone", {})
    if not backbone_cfg:
        backbone_cfg = ckpt_cfg.get("model", {}).get("backbone", {})

    input_size = tuple(backbone_cfg.get("input_size", [112, 112]))
    l2_norm = bool(backbone_cfg.get("l2_normalize", True))

    device = select_device()
    backbone = moco.build_backbone(backbone_cfg)
    backbone.load_state_dict(moco.backbone_state_from_checkpoint(ckpt), strict=True)
    backbone.to(device)
    backbone.eval()
    logger.info(f"Loaded backbone, device: {device}")

    # Data configuration
    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})

    digiface_glob = data_cfg.get("digiface_glob")
    if not digiface_glob:
        digiface_glob = ckpt_cfg.get("data", {}).get("digiface_glob")

    if not digiface_glob:
        raise ValueError("No data glob found in config or checkpoint.")

    # Get splits
    cache_dir = Path(split_cfg.get("cache_dir", ".cache/splits"))
    train_ratio = float(split_cfg.get("train_ratio", 0.75))
    split_seed = int(split_cfg.get("seed", cfg.get("experiment", {}).get("seed", 42)))

    splits_path = dataio.get_or_create_splits(
        globs=[digiface_glob],
        cache_dir=cache_dir,
        train_ratio=train_ratio,
        seed=split_seed,
    )

    allowed_identities = dataio.get_identity_set(splits_path, split_name)
    logger.info(f"Embedding {len(allowed_identities)} {split_name} identities")

    transform = build_embed_transform(input_size)

    ds = dataio.ParquetEmbedDataset(
        digiface_glob,
        transform=transform,
        allowed_identities=allowed_identities,
    )

    loader = DataLoader(
        ds,
        batch_size=int(embed_cfg.get("batch_size", 256)),
        num_workers=int(cfg.data.streaming.get("num_workers", 4)),
        pin_memory=(device.type == "cuda"),
    )

    out_path = Path(os.getcwd()) / f"embeddings_{split_name}.parquet"
    emb_dim = int(backbone_cfg.get("embedding_dim", 512))

    schema = pa.schema(
        [
            ("identity_id", pa.string()),
            ("image_filename", pa.string()),
            ("embedding", pa.fixed_size_list(pa.float32(), emb_dim)),
        ]
    )
    writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd")

    amp_enabled = bool(embed_cfg.get("amp", True)) and device.type == "cuda"

    total_embedded = 0
    with torch.no_grad():
        for ids, fns, imgs in tqdm(loader, desc="embed", unit="batch"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
                emb = backbone(imgs)
                if l2_norm:
                    emb = moco.l2_normalize(emb, dim=1)
            emb_np = emb.detach().cpu().numpy().astype(np.float32, copy=False)

            values = pa.array(emb_np.reshape(-1), type=pa.float32())
            emb_arr = pa.FixedSizeListArray.from_arrays(values, emb_dim)

            table = pa.Table.from_arrays(
                [
                    pa.array(list(ids), type=pa.string()),
                    pa.array(list(fns), type=pa.string()),
                    emb_arr,
                ],
                schema=schema,
            )
            writer.write_table(table)
            total_embedded += len(ids)

    writer.close()
    logger.info(f"Wrote {total_embedded} embeddings to: {out_path}")


# =============================================================================
# Eval Command
# =============================================================================


def cmd_eval(cfg: DictConfig) -> None:
    """Evaluate retrieval metrics on embeddings."""
    import faiss

    eval_cfg = cfg.get("eval", {})
    emb_path = Path(os.getcwd()) / "embeddings_test.parquet"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}. Run embed first.")

    parquet_file = pq.ParquetFile(emb_path)
    schema = parquet_file.schema_arrow

    emb_field = schema.field("embedding")
    dim = int(emb_field.type.list_size)

    enroll_per_id = int(eval_cfg.get("enroll_per_id", 5))
    top_k = int(eval_cfg.get("top_k", 5))
    seed = int(cfg.get("experiment", {}).get("seed", 42))

    id_to_embs: dict[str, list[np.ndarray]] = {}
    total_rows = 0

    for batch in tqdm(parquet_file.iter_batches(batch_size=8192), desc="read", unit="batch"):
        ids = batch.column(0).to_pylist()
        emb_col = batch.column(batch.schema.get_field_index("embedding"))
        values = emb_col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        embs = values.reshape(-1, dim)

        for identity_id, vec in zip(ids, embs, strict=True):
            key = str(identity_id)
            id_to_embs.setdefault(key, []).append(vec.copy())
        total_rows += len(ids)

    identity_ids = sorted(id_to_embs.keys())
    n_ids = len(identity_ids)
    logger.info(f"Loaded {total_rows} embeddings from {n_ids} identities")

    rng = np.random.default_rng(seed)

    centroids = np.zeros((n_ids, dim), dtype=np.float32)
    enroll_sets: dict[str, set[int]] = {}

    for i, identity_id in enumerate(tqdm(identity_ids, desc="centroids", unit="id")):
        embs = np.stack(id_to_embs[identity_id], axis=0)
        n = embs.shape[0]
        perm = rng.permutation(n)
        e = min(enroll_per_id, n)
        enroll_idx = perm[:e]
        enroll_sets[identity_id] = set(int(x) for x in enroll_idx.tolist())

        centroid = embs[enroll_idx].mean(axis=0)
        norm = np.linalg.norm(centroid) + 1e-12
        centroids[i] = (centroid / norm).astype(np.float32)

    index = faiss.IndexFlatIP(dim)
    index.add(centroids)

    n_queries = 0
    rank1 = 0
    topk = 0

    batch_q: list[np.ndarray] = []
    batch_true: list[int] = []

    def flush() -> None:
        nonlocal n_queries, rank1, topk, batch_q, batch_true
        if not batch_q:
            return
        q = np.stack(batch_q, axis=0).astype(np.float32, copy=False)
        true = np.array(batch_true, dtype=np.int64)
        _, idx = index.search(q, top_k)

        n_queries += q.shape[0]
        rank1 += int(np.sum(idx[:, 0] == true))
        topk += int(np.sum(np.any(idx == true[:, None], axis=1)))

        batch_q = []
        batch_true = []

    query_batch_size = int(eval_cfg.get("query_batch", 50000))

    for true_idx, identity_id in enumerate(tqdm(identity_ids, desc="queries", unit="id")):
        embs = np.stack(id_to_embs[identity_id], axis=0)
        enroll_idx = enroll_sets[identity_id]
        query_idx = [j for j in range(embs.shape[0]) if j not in enroll_idx]
        if not query_idx:
            continue

        for vec in embs[query_idx]:
            batch_q.append(vec.astype(np.float32, copy=False))
            batch_true.append(true_idx)

        if len(batch_q) >= query_batch_size:
            flush()

    flush()

    metrics = {
        "num_identities": n_ids,
        "num_embeddings": total_rows,
        "num_queries": n_queries,
        "enroll_per_id": enroll_per_id,
        "top_k": top_k,
        "rank_1": (rank1 / n_queries) if n_queries else 0.0,
        "top_k_acc": (topk / n_queries) if n_queries else 0.0,
    }

    out_path = Path(os.getcwd()) / "metrics.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Rank-1: {metrics['rank_1']:.4f}, Top-{top_k}: {metrics['top_k_acc']:.4f}")

    # Log to W&B if active
    if wandb.run is not None:
        wandb.log({"eval/rank_1": metrics["rank_1"], "eval/top_k_acc": metrics["top_k_acc"]})


# =============================================================================
# Hydra Entry Point
# =============================================================================


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    # Determine command from command line (first positional arg before Hydra takes over)
    # Since Hydra captures all args, we use a config key or default to train
    command = cfg.get("command", "train")

    # Log config
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
