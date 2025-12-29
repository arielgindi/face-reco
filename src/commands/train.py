"""Training command for MoCo + MarginNCE encoder."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src import data
from src.augmentations import build_view_transform
from src.checkpoint import (
    load_checkpoint_for_resume,
    prewarm_datasets,
    prune_checkpoints,
    save_checkpoint,
)
from src.model import build_moco
from src.schedule import build_curriculum_schedule, curriculum_p_digiface
from src.utils import (
    compute_epoch_batch_counts,
    configure_precision,
    select_device,
    set_seed,
)
from src.wandb_utils import finish_wandb, init_wandb, log_gpu_memory, log_wandb

logger = logging.getLogger(__name__)


def cmd_train(cfg: DictConfig) -> None:
    """Train MoCo + MarginNCE encoder."""
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

    out_dir = Path(os.getcwd())
    logger.info(f"Output directory: {out_dir}")

    wandb_active = init_wandb(cfg, out_dir)
    wandb_cfg = cfg.get("wandb", {})
    log_every = int(wandb_cfg.get("log_every_steps", 50))

    data_cfg = cfg.get("data", {})
    split_cfg = data_cfg.get("split", {})
    streaming_cfg = data_cfg.get("streaming", {})

    digiface_glob = data_cfg.get("digiface_glob")
    digi2real_glob = data_cfg.get("digi2real_glob")

    if not digiface_glob:
        raise ValueError("data.digiface_glob not specified in config.")

    train_ratio = float(split_cfg.get("train_ratio", 0.75))
    split_seed = int(split_cfg.get("seed", seed))
    cache_dir = Path(split_cfg.get("cache_dir", ".cache/splits"))

    data_globs = [digiface_glob]
    if digi2real_glob:
        data_globs.append(digi2real_glob)

    splits_path = data.get_or_create_splits(
        globs=data_globs,
        cache_dir=cache_dir,
        train_ratio=train_ratio,
        seed=split_seed,
    )

    train_identities = data.get_identity_set(splits_path, "train")
    logger.info(
        f"Training on {len(train_identities)} identities "
        f"({train_ratio:.0%} of total, test set protected)"
    )

    shutil.copy2(splits_path, out_dir / "identity_splits.parquet")

    model = build_moco(cfg, device=device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

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

    scaler: torch.amp.GradScaler | None = None
    if amp_enabled and device.type == "cuda" and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")

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

    use_torch_compile = bool(cfg.train.precision.get("torch_compile", False))
    if device.type == "cuda" and sys.platform != "win32" and use_torch_compile:
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("Applied torch.compile with reduce-overhead mode")
    elif device.type == "cuda" and sys.platform == "win32":
        logger.info("Skipping torch.compile (Triton not available on Windows)")
    elif device.type == "cuda" and not use_torch_compile:
        logger.info("Skipping torch.compile (disabled in config)")

    input_size = tuple(cfg.model.backbone.input_size)
    aug_cfg = cfg.augmentation
    t_q = build_view_transform(aug_cfg.view_1, input_size=input_size)
    t_k = build_view_transform(aug_cfg.view_2, input_size=input_size)

    stream_params = data.StreamParams(
        shuffle_files=True,
        batch_read_rows=int(streaming_cfg.get("batch_read_rows", 2048)),
        shuffle_within_batch=True,
        shuffle_buffer_size=int(streaming_cfg.get("shuffle_buffer_size", 10000)),
        seed=seed,
    )

    digiface_ds = data.ParquetTwoViewDataset(
        digiface_glob,
        transform_q=t_q,
        transform_k=t_k,
        stream=stream_params,
        allowed_identities=train_identities,
    )

    digi2real_ds: data.ParquetTwoViewDataset | None = None
    if digi2real_glob:
        digi2real_ds = data.ParquetTwoViewDataset(
            digi2real_glob,
            transform_q=t_q,
            transform_k=t_k,
            stream=stream_params,
            allowed_identities=train_identities,
        )

    base_samples = int(cfg.train.get("samples_per_epoch", 0))
    if base_samples <= 0:
        base_samples = data.count_parquet_rows(digiface_glob)

    num_batches, num_samples = compute_epoch_batch_counts(
        base_samples=base_samples,
        batch_size=batch_size,
        grad_accum_steps=grad_accum,
    )
    logger.info(
        f"Epoch: {num_samples:,} samples, {num_batches:,} batches, "
        f"batch_size={batch_size}, grad_accum={grad_accum}"
    )

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

    if sys.platform == "win32":
        prewarm_datasets(digiface_ds, digi2real_ds, num_workers, device)

    schedule = build_curriculum_schedule(cfg)

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

        epoch_ds = data.CurriculumMixTwoViewDataset(
            digiface=digiface_ds,
            digi2real=digi2real_ds,
            p_digiface=p_digiface,
            num_samples=num_samples,
            seed=seed + epoch * 17,
        )

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
        epoch_start_time = time.perf_counter()
        images_processed = 0

        pbar = tqdm(loader, total=num_batches, desc=f"epoch {epoch:03d}", unit="batch")
        for step, (im_q, im_k) in enumerate(pbar):
            images_processed += im_q.shape[0]
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
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

            if (
                wandb_active
                and log_every > 0
                and global_step % log_every == 0
                and (step + 1) % grad_accum == 0
            ):
                metrics_to_log = {
                    "train/loss": stats["loss"],
                    "train/pos_sim": stats["pos_sim"],
                    "train/neg_sim": stats["neg_sim"],
                    "train/emb_std": stats["emb_std"],
                    "train/sim_gap": stats["pos_sim"] - stats["neg_sim"],
                    "train/lr": lr,
                    "train/p_digiface": p_digiface,
                    "train/epoch": epoch,
                }
                metrics_to_log.update(log_gpu_memory())
                log_wandb(metrics_to_log, step=global_step)

            elapsed = time.perf_counter() - epoch_start_time
            img_per_sec = images_processed / elapsed if elapsed > 0 else 0

            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                pos=f"{stats['pos_sim']:.3f}",
                neg=f"{stats['neg_sim']:.3f}",
                img_s=f"{img_per_sec:.0f}",
            )

        epoch_elapsed = time.perf_counter() - epoch_start_time
        epoch_img_per_sec = images_processed / epoch_elapsed if epoch_elapsed > 0 else 0

        if n_logged > 0:
            avg_pos = epoch_pos_sum / n_logged
            avg_neg = epoch_neg_sum / n_logged
            epoch_metrics = {
                "epoch/loss": epoch_loss_sum / n_logged,
                "epoch/pos_sim": avg_pos,
                "epoch/neg_sim": avg_neg,
                "epoch/sim_gap": avg_pos - avg_neg,
                "epoch/emb_std": epoch_embstd_sum / n_logged,
                "epoch/images_per_sec": epoch_img_per_sec,
                "epoch/lr": lr,
                "epoch/p_digiface": p_digiface,
                "epoch/number": epoch,
            }
            epoch_metrics.update(log_gpu_memory())
            if wandb_active:
                log_wandb(epoch_metrics, step=global_step)

            logger.info(
                f"Epoch {epoch:03d}: loss={epoch_metrics['epoch/loss']:.4f}, "
                f"pos_sim={epoch_metrics['epoch/pos_sim']:.3f}, "
                f"neg_sim={epoch_metrics['epoch/neg_sim']:.3f}, "
                f"img/s={epoch_img_per_sec:.0f}"
            )

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

            if wandb_active and wandb_cfg.get("save_artifacts", False):
                artifact = wandb.Artifact(
                    f"model-epoch-{epoch + 1:03d}",
                    type="model",
                    metadata={"epoch": epoch + 1},
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    logger.info("Training complete!")
    finish_wandb()
