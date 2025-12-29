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
from src.pseudo import PseudoIDManager
from src.schedule import (
    build_curriculum_schedule,
    build_pseudo_schedule,
    curriculum_p_digiface,
    get_pseudo_prob,
    get_refresh_epochs,
    get_sim_threshold,
)
from src.utils import (
    compute_epoch_batch_counts,
    configure_precision,
    select_device,
    set_seed,
)
from src.wandb_utils import finish_wandb, init_wandb, log_gpu_memory, log_wandb

logger = logging.getLogger(__name__)


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def _log_training_config(
    cfg: DictConfig,
    device: torch.device,
    num_params: int,
    train_identities: int,
    num_samples: int,
    num_batches: int,
    start_epoch: int,
) -> None:
    """Log a formatted training configuration summary."""
    ssl_cfg = cfg.get("ssl", {})
    pseudo_cfg = cfg.get("pseudo", {})
    train_cfg = cfg.train

    # Get GPU info
    gpu_name = "N/A"
    gpu_mem = "N/A"
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"

    lines = [
        "",
        "=" * 70,
        "  SNIPERFACE TRAINING",
        "=" * 70,
        "",
        "  Hardware:",
        f"    GPU:              {gpu_name} ({gpu_mem})",
        f"    Precision:        {'FP16 (AMP)' if train_cfg.precision.amp else 'FP32'}",
        "",
        "  Model:",
        f"    Backbone:         {cfg.model.backbone.name}",
        f"    Parameters:       {num_params:,}",
        f"    Embedding dim:    {cfg.model.backbone.embedding_dim}",
        "",
        "  SSL (MoCo + MarginNCE):",
        f"    Queue size:       {ssl_cfg.get('queue_size', 32768):,}",
        f"    Temperature:      {ssl_cfg.get('temperature', 0.07)}",
        f"    Margin:           {ssl_cfg.get('margin_nce', {}).get('margin', 0.1)}",
        "",
        "  Training:",
        f"    Epochs:           {start_epoch} -> {train_cfg.epochs - 1} ({train_cfg.epochs - start_epoch} total)",
        f"    Batch size:       {train_cfg.batch.size} x {train_cfg.batch.grad_accum_steps} = {train_cfg.batch.size * train_cfg.batch.grad_accum_steps}",
        f"    Samples/epoch:    {num_samples:,}",
        f"    Steps/epoch:      {num_batches // train_cfg.batch.grad_accum_steps:,}",
        f"    Learning rate:    {train_cfg.optimizer.lr} (warmup: {train_cfg.lr_schedule.warmup.epochs} epochs)",
        "",
        "  Data:",
        f"    Train identities: {train_identities:,}",
        f"    DigiFace glob:    {cfg.data.digiface_glob}",
    ]

    if cfg.data.get("digi2real_glob"):
        lines.append(f"    Digi2Real glob:   {cfg.data.digi2real_glob}")

    if pseudo_cfg.get("enabled", False):
        threshold_cfg = pseudo_cfg.get("sim_threshold", {})
        lines.extend([
            "",
            "  Pseudo-ID Mining:",
            f"    kNN k:            {pseudo_cfg.get('knn_k', 20)}",
            f"    Mutual top-k:     {pseudo_cfg.get('mutual_topk', 5)}",
            f"    Sim threshold:    {threshold_cfg.get('start', 0.72)} -> {threshold_cfg.get('end', 0.52)}",
            f"    Cluster size:     {pseudo_cfg.get('min_cluster_size', 2)}-{pseudo_cfg.get('max_cluster_size', 50)}",
            f"    Refresh epochs:   {pseudo_cfg.get('refresh_epochs', [])}",
        ])

    lines.extend([
        "",
        "=" * 70,
        "",
    ])

    for line in lines:
        logger.info(line)


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
    out_dir = Path(os.getcwd())

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
    shutil.copy2(splits_path, out_dir / "identity_splits.parquet")

    model = build_moco(cfg, device=device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())

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

    # Initialize pseudo-ID manager if enabled
    pseudo_cfg = cfg.get("pseudo", {})
    pseudo_enabled = bool(pseudo_cfg.get("enabled", False))
    pseudo_manager: PseudoIDManager | None = None

    if pseudo_enabled:
        pseudo_manager = PseudoIDManager(
            knn_k=int(pseudo_cfg.get("knn_k", 20)),
            mutual_topk=int(pseudo_cfg.get("mutual_topk", 5)),
            min_cluster_size=int(pseudo_cfg.get("min_cluster_size", 2)),
            max_cluster_size=int(pseudo_cfg.get("max_cluster_size", 50)),
        )

    resume_path = cfg.train.get("resume")
    warm_start = bool(cfg.train.get("warm_start", False))
    start_epoch = 0
    if resume_path:
        start_epoch = load_checkpoint_for_resume(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            pseudo_manager=pseudo_manager,
            warm_start=warm_start,
        )

    use_torch_compile = bool(cfg.train.precision.get("torch_compile", False))
    if device.type == "cuda" and sys.platform != "win32" and use_torch_compile:
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model, mode="reduce-overhead")

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

    # Log training configuration summary
    _log_training_config(
        cfg=cfg,
        device=device,
        num_params=num_params,
        train_identities=len(train_identities),
        num_samples=num_samples,
        num_batches=num_batches,
        start_epoch=start_epoch,
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
    pseudo_schedule = build_pseudo_schedule(cfg) if pseudo_enabled else ()
    refresh_epochs = get_refresh_epochs(cfg)

    # Pseudo-ID negative masking config
    neg_cfg = pseudo_cfg.get("negatives", {}) if pseudo_enabled else {}
    mask_same_cluster = bool(neg_cfg.get("mask_same_pseudo_in_queue", True))
    mask_topk = int(neg_cfg.get("mask_topk_most_similar", 8))
    reset_queue_on_refresh = bool(neg_cfg.get("reset_queue_on_refresh", True))

    steps_per_epoch = num_batches // grad_accum
    global_step = start_epoch * steps_per_epoch
    optimizer.zero_grad(set_to_none=True)

    training_start_time = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        lr = lr_for_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        p_digiface = curriculum_p_digiface(epoch, schedule)
        pseudo_prob = get_pseudo_prob(epoch, pseudo_schedule) if pseudo_enabled else 0.0

        # Check if we need to refresh pseudo-IDs at start of this epoch
        if pseudo_enabled and pseudo_manager is not None and epoch in refresh_epochs:
            sim_threshold = get_sim_threshold(epoch, cfg)
            datasets_for_mining = [digiface_ds]
            if digi2real_ds is not None:
                datasets_for_mining.append(digi2real_ds)

            refresh_stats = pseudo_manager.refresh(
                model=model,
                datasets=datasets_for_mining,
                epoch=epoch,
                sim_threshold=sim_threshold,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # Log refresh stats to W&B
            if wandb_active:
                log_wandb(refresh_stats, step=global_step)

            # Reset queue after refresh
            if reset_queue_on_refresh:
                model.reset_queue()
                logger.info("Queue reset after pseudo-ID refresh")

        # Build curriculum dataset
        curriculum_ds = data.CurriculumMixTwoViewDataset(
            digiface=digiface_ds,
            digi2real=digi2real_ds,
            p_digiface=p_digiface,
            num_samples=num_samples,
            seed=seed + epoch * 17,
        )

        # Wrap with pseudo-pair sampling if enabled and have clusters
        if pseudo_enabled and pseudo_manager is not None and pseudo_manager.state is not None:
            epoch_ds: Any = data.PseudoPairTwoViewDataset(
                base_dataset=curriculum_ds,
                pseudo_manager=pseudo_manager,  # FIX: Pass manager, not state
                transform_q=t_q,
                transform_k=t_k,
                p_pseudo=pseudo_prob,
                num_samples=num_samples,
                seed=seed + epoch * 17,
            )
            use_pseudo_pairs = True
        else:
            epoch_ds = curriculum_ds
            use_pseudo_pairs = False

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
        last_log_time = epoch_start_time
        last_log_images = 0

        # Determine training phase for display
        if epoch < 5:
            phase = "A: Stabilize"
        elif epoch < 35:
            phase = "B: Bootstrap"
        else:
            phase = "C: Refine"

        pbar = tqdm(loader, total=num_batches, desc=f"Epoch {epoch:03d} [{phase}]", unit="batch")
        for step, batch in enumerate(pbar):
            # Handle 2-tuple (no pseudo) or 3-tuple (with pseudo cluster IDs)
            if use_pseudo_pairs and len(batch) == 3:
                im_q, im_k, cluster_ids = batch
                # FIX: Cast to int32 to match queue_cluster_ids dtype
                cluster_ids = cluster_ids.to(dtype=torch.int32, device=device, non_blocking=True)
            else:
                im_q, im_k = batch[0], batch[1]
                cluster_ids = None

            images_processed += im_q.shape[0]
            im_q = im_q.to(device, non_blocking=True)
            im_k = im_k.to(device, non_blocking=True)

            with torch.amp.autocast(
                "cuda",
                enabled=amp_enabled and device.type == "cuda",
                dtype=amp_dtype,
            ):
                loss, stats = model(
                    im_q,
                    im_k,
                    cluster_ids=cluster_ids,
                    mask_same_cluster=mask_same_cluster and use_pseudo_pairs,
                    mask_topk=mask_topk if use_pseudo_pairs else 0,
                )
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
                # Add pseudo-ID metrics if enabled
                if pseudo_enabled:
                    metrics_to_log["train/pseudo_prob"] = pseudo_prob
                    metrics_to_log["train/neg_masked_pct"] = stats.get("neg_masked_pct", 0.0)
                metrics_to_log.update(log_gpu_memory())
                log_wandb(metrics_to_log, step=global_step)

            # Calculate instantaneous img/s (last ~1 second window)
            current_time = time.perf_counter()
            time_since_log = current_time - last_log_time
            if time_since_log >= 1.0:
                images_since_log = images_processed - last_log_images
                img_per_sec = images_since_log / time_since_log
                last_log_time = current_time
                last_log_images = images_processed
            else:
                # Use average for first second
                elapsed = current_time - epoch_start_time
                img_per_sec = images_processed / elapsed if elapsed > 0 else 0

            pbar.set_postfix(
                loss=f"{stats['loss']:.4f}",
                pos=f"{stats['pos_sim']:.3f}",
                neg=f"{stats['neg_sim']:.3f}",
                ips=f"{img_per_sec:.0f}",
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
            # Add pseudo-ID epoch metrics
            if pseudo_enabled:
                epoch_metrics["epoch/pseudo_prob"] = pseudo_prob
                if pseudo_manager is not None and pseudo_manager.state is not None:
                    epoch_metrics["epoch/pseudo_clusters"] = pseudo_manager.num_clusters
            epoch_metrics.update(log_gpu_memory())
            if wandb_active:
                log_wandb(epoch_metrics, step=global_step)

            # Calculate ETA
            epochs_done = epoch - start_epoch + 1
            epochs_remaining = epochs - epoch - 1
            total_elapsed = time.perf_counter() - training_start_time
            avg_epoch_time = total_elapsed / epochs_done
            eta_seconds = avg_epoch_time * epochs_remaining

            logger.info(
                f"Epoch {epoch:03d} complete | "
                f"loss={epoch_metrics['epoch/loss']:.4f} | "
                f"pos={epoch_metrics['epoch/pos_sim']:.3f} neg={epoch_metrics['epoch/neg_sim']:.3f} gap={epoch_metrics['epoch/sim_gap']:.3f} | "
                f"{epoch_img_per_sec:.0f} img/s | "
                f"ETA: {_format_time(eta_seconds)}"
            )

        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt_path = save_checkpoint(
                out_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                cfg=cfg,
                pseudo_manager=pseudo_manager,
            )
            prune_checkpoints(out_dir / "checkpoints", keep_last=keep_last)
            logger.info(f"Checkpoint saved: {ckpt_path.name}")

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

    total_training_time = time.perf_counter() - training_start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"  Training complete in {_format_time(total_training_time)}")
    logger.info(f"  Final epoch: {epochs - 1}")
    logger.info(f"  Checkpoints: {out_dir / 'checkpoints'}")
    logger.info("=" * 70)
    logger.info("")
    finish_wandb()
