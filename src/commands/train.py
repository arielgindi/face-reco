"""Training command for MoCo + MarginNCE face encoder."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src import data
from src.augmentations import build_album_transform, build_view_transform
from src.checkpoint import (
    find_latest_checkpoint,
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
)
from src.utils import compute_epoch_batch_counts, configure_precision, select_device, set_seed
from src.wandb_utils import finish_wandb, init_wandb, log_gpu_memory, log_wandb

logger = logging.getLogger(__name__)


def _fmt_time(s: float) -> str:
    """Format seconds as human-readable string."""
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{int(s//3600)}h {int((s%3600)//60)}m"


def _log_config(cfg: DictConfig, device: torch.device, num_params: int,
                n_ids: int, n_samples: int, n_batches: int, start: int) -> None:
    """Log training configuration summary."""
    ssl, pseudo, train = cfg.get("ssl", {}), cfg.get("pseudo", {}), cfg.train
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    mem = f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if device.type == "cuda" else ""

    logger.info("\n" + "="*70)
    logger.info("  SNIPERFACE TRAINING")
    logger.info("="*70)
    logger.info(f"  GPU: {gpu} ({mem})  |  Precision: {'FP16' if train.precision.amp else 'FP32'}")
    logger.info(f"  Model: {cfg.model.backbone.name} ({num_params:,} params, {cfg.model.backbone.embedding_dim}D)")
    logger.info(f"  SSL: queue={ssl.get('queue_size',32768):,} temp={ssl.get('temperature',0.07)} margin={ssl.get('margin_nce',{}).get('margin',0.1)}")
    logger.info(f"  Training: epochs {start}->{train.epochs-1} | batch {train.batch.size}x{train.batch.grad_accum_steps} | lr={train.optimizer.lr}")
    logger.info(f"  Data: {n_ids:,} identities | {n_samples:,} samples/epoch | {n_batches//train.batch.grad_accum_steps:,} steps")
    if pseudo.get("enabled"):
        th = pseudo.get("sim_threshold", {})
        logger.info(f"  Pseudo-ID: k={pseudo.get('knn_k',20)} mutual={pseudo.get('mutual_topk',5)} threshold={th.get('start',0.72)}->{th.get('end',0.52)}")
    logger.info("="*70 + "\n")


def _get_phase(epoch: int) -> str:
    """Get training phase name for display."""
    if epoch < 5:
        return "A:Stabilize"
    if epoch < 35:
        return "B:Bootstrap"
    return "C:Refine"


def _build_lr_fn(cfg: DictConfig):
    """Build learning rate schedule function."""
    sched = cfg.train.lr_schedule
    milestones, gamma = list(sched.milestones), float(sched.gamma)
    base_lr = float(cfg.train.optimizer.lr)
    warm_enabled, warm_epochs = bool(sched.warmup.enabled), int(sched.warmup.epochs)
    warm_start_lr = float(sched.warmup.start_lr)

    def lr_for_epoch(epoch: int) -> float:
        lr = base_lr * (gamma ** sum(epoch >= m for m in milestones))
        if warm_enabled and epoch < warm_epochs and warm_epochs > 1:
            return warm_start_lr + (epoch / (warm_epochs - 1)) * (lr - warm_start_lr)
        return lr
    return lr_for_epoch


def cmd_train(cfg: DictConfig) -> None:
    """Train MoCo + MarginNCE face encoder with optional pseudo-ID bootstrapping."""
    # Setup
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed, bool(cfg.get("experiment", {}).get("deterministic", False)))
    amp_enabled, amp_dtype, tf32 = configure_precision(cfg)
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    device = select_device()
    out_dir = Path(os.getcwd())
    wandb_active = init_wandb(cfg, out_dir)
    log_every = int(cfg.get("wandb", {}).get("log_every_steps", 50))

    # Data splits
    data_cfg = cfg.get("data", {})
    digiface_glob, digi2real_glob = data_cfg.get("digiface_glob"), data_cfg.get("digi2real_glob")
    if not digiface_glob:
        raise ValueError("data.digiface_glob not specified")

    split_cfg = data_cfg.get("split", {})
    globs = [digiface_glob] + ([digi2real_glob] if digi2real_glob else [])
    splits_path = data.get_or_create_splits(
        globs=globs, cache_dir=Path(split_cfg.get("cache_dir", ".cache/splits")),
        train_ratio=float(split_cfg.get("train_ratio", 0.75)), seed=int(split_cfg.get("seed", seed)),
    )
    train_ids = data.get_identity_set(splits_path, "train")
    shutil.copy2(splits_path, out_dir / "identity_splits.parquet")

    # Model & optimizer
    model = build_moco(cfg, device=device)
    model.train()
    num_params = sum(p.numel() for p in model.parameters())
    batch_size, grad_accum, epochs = int(cfg.train.batch.size), int(cfg.train.batch.grad_accum_steps), int(cfg.train.epochs)

    opt_cfg = cfg.train.optimizer
    optimizer = SGD(model.parameters(), lr=float(opt_cfg.lr), momentum=float(opt_cfg.momentum),
                    weight_decay=float(opt_cfg.weight_decay), nesterov=bool(opt_cfg.nesterov))
    scaler = torch.amp.GradScaler("cuda") if amp_enabled and device.type == "cuda" and amp_dtype == torch.float16 else None

    # Pseudo-ID manager
    pseudo_cfg = cfg.get("pseudo", {})
    pseudo_enabled = bool(pseudo_cfg.get("enabled", False))
    if pseudo_enabled:
        threshold_cfg = pseudo_cfg.sim_threshold
        pseudo_mgr = PseudoIDManager(
            knn_k=int(pseudo_cfg.knn_k),
            mutual_topk=int(pseudo_cfg.mutual_topk),
            min_cluster_size=int(pseudo_cfg.min_cluster_size),
            max_cluster_size=int(pseudo_cfg.max_cluster_size),
            threshold_start=float(threshold_cfg.start),
            threshold_end=float(threshold_cfg.end),
            target_coverage=float(threshold_cfg.target_coverage),
            adaptation_rate=float(threshold_cfg.adaptation_rate),
            adaptive_mode=str(threshold_cfg.mode).lower() == "adaptive",
        )
    else:
        pseudo_mgr = None

    # Resume checkpoint
    start_epoch = 0
    resume = cfg.train.get("resume", "auto")
    if resume:
        if resume == "auto":
            # Find latest from W&B or local
            wandb_project = cfg.get("wandb", {}).get("project", "")
            resume_path = find_latest_checkpoint(wandb_project, out_dir / "checkpoints")
        else:
            resume_path = Path(resume)
        start_epoch = load_checkpoint_for_resume(
            resume_path, model, optimizer, scaler, device,
            pseudo_manager=pseudo_mgr, warm_start=bool(cfg.train.get("warm_start", False)),
        )

    # Torch compile (Linux only)
    if device.type == "cuda" and sys.platform != "win32" and cfg.train.precision.get("torch_compile"):
        torch._dynamo.config.capture_scalar_outputs = True
        cc = torch.cuda.get_device_capability(0)
        if cc[0] >= 12:  # Blackwell (sm_120) - use cudagraphs backend (Triton has issues)
            logger.info(f"Using cudagraphs backend for Blackwell GPU (sm_{cc[0]}{cc[1]})")
            model = torch.compile(model, backend="cudagraphs")
        else:
            model = torch.compile(model, mode="reduce-overhead")

    # Datasets
    input_size = tuple(cfg.model.backbone.input_size)
    stream_cfg = data_cfg.get("streaming", {})
    num_workers = int(stream_cfg.get("num_workers", 4))

    # Check for binary cache (pre-decoded images for max speed)
    binary_cache = data_cfg.get("binary_cache_path")
    use_binary = binary_cache and Path(binary_cache).exists()

    if use_binary:
        logger.info(f"Using binary cache: {binary_cache}")
        t_q = build_album_transform(cfg.augmentation.view_1, input_size=input_size)
        t_k = build_album_transform(cfg.augmentation.view_2, input_size=input_size)
        digiface_ds = data.BinaryImageDataset(binary_cache, t_q, t_k, seed=seed)
        digi2real_ds = None  # Binary mode uses single pre-merged dataset
        base_samples = len(digiface_ds)
    else:
        t_q = build_view_transform(cfg.augmentation.view_1, input_size=input_size)
        t_k = build_view_transform(cfg.augmentation.view_2, input_size=input_size)
        stream = data.StreamParams(
            shuffle_files=True, batch_read_rows=int(stream_cfg.get("batch_read_rows", 2048)),
            shuffle_within_batch=True, shuffle_buffer_size=int(stream_cfg.get("shuffle_buffer_size", 10000)), seed=seed,
        )
        digiface_ds = data.ParquetTwoViewDataset(digiface_glob, t_q, t_k, stream=stream, allowed_identities=train_ids)
        digi2real_ds = data.ParquetTwoViewDataset(digi2real_glob, t_q, t_k, stream=stream, allowed_identities=train_ids) if digi2real_glob else None
        base_samples = int(cfg.train.get("samples_per_epoch", 0)) or data.count_parquet_rows(digiface_glob)

    num_batches, num_samples = compute_epoch_batch_counts(
        base_samples=base_samples, batch_size=batch_size, grad_accum_steps=grad_accum
    )

    # Log config summary
    _log_config(cfg, device, num_params, len(train_ids), num_samples, num_batches, start_epoch)

    # Schedules
    lr_fn = _build_lr_fn(cfg)
    curriculum = build_curriculum_schedule(cfg)
    pseudo_sched = build_pseudo_schedule(cfg) if pseudo_enabled else ()
    refresh_epochs = get_refresh_epochs(cfg) if pseudo_enabled else set()
    neg_cfg = pseudo_cfg.get("negatives", {})
    mask_cluster, mask_topk = bool(neg_cfg.get("mask_same_pseudo_in_queue", True)), int(neg_cfg.get("mask_topk_most_similar", 8))
    reset_queue = bool(neg_cfg.get("reset_queue_on_refresh", True))
    grad_clip = float(cfg.train.regularization.grad_clip_norm)
    save_every, keep_last = int(cfg.train.checkpointing.save_every_epochs), int(cfg.train.checkpointing.keep_last)
    save_local = bool(cfg.train.checkpointing.get("save_local", False))

    if sys.platform == "win32" and not use_binary:
        prewarm_datasets(digiface_ds, digi2real_ds, num_workers, device)

    # Training loop
    global_step = start_epoch * (num_batches // grad_accum)
    optimizer.zero_grad(set_to_none=True)
    train_start = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        lr = lr_fn(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        p_digi = curriculum_p_digiface(epoch, curriculum)
        p_pseudo = get_pseudo_prob(epoch, pseudo_sched) if pseudo_enabled else 0.0

        # Pseudo-ID refresh
        if pseudo_mgr and epoch in refresh_epochs:
            datasets = [digiface_ds] + ([digi2real_ds] if digi2real_ds else [])
            stats = pseudo_mgr.refresh(model, datasets, epoch, pseudo_mgr.get_threshold(), device, batch_size, num_workers)
            if wandb_active:
                log_wandb(stats, step=global_step)
            if reset_queue:
                model.reset_queue()

        # Build epoch dataset
        if use_binary:
            # Binary mode with pseudo-ID support
            base_ds = data.BinaryMixDataset(digiface_ds, None, 1.0, num_samples, seed + epoch * 17)
            use_pseudo = pseudo_mgr and pseudo_mgr.state is not None and p_pseudo > 0
            epoch_ds = (
                data.PseudoPairTwoViewDataset(base_ds, pseudo_mgr, t_q, t_k, p_pseudo, num_samples, seed + epoch * 17)
                if use_pseudo else base_ds
            )
        else:
            curr_ds = data.CurriculumMixTwoViewDataset(digiface_ds, digi2real_ds, p_digi, num_samples, seed + epoch * 17)
            use_pseudo = pseudo_mgr and pseudo_mgr.state is not None
            epoch_ds = data.PseudoPairTwoViewDataset(curr_ds, pseudo_mgr, t_q, t_k, p_pseudo, num_samples, seed + epoch * 17) if use_pseudo else curr_ds

        loader_kw = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": device.type == "cuda", "drop_last": True}
        if num_workers > 0:
            loader_kw.update(persistent_workers=True, prefetch_factor=2)
            # Use fork for binary mode (COW memory sharing)
            if use_binary and sys.platform != "win32":
                loader_kw["multiprocessing_context"] = "fork"
        loader = DataLoader(epoch_ds, **loader_kw)

        # Epoch tracking
        loss_sum = pos_sum = neg_sum = std_sum = grad_norm_sum = 0.0
        n_steps = 0
        grad_norm = 0.0
        epoch_start = time.perf_counter()
        img_count = 0

        pbar = tqdm(loader, total=num_batches, desc=f"Epoch {epoch:03d} [{_get_phase(epoch)}]", unit="batch")
        for step, batch in enumerate(pbar):
            # Unpack batch
            if use_pseudo and len(batch) == 3:
                im_q, im_k, cids = batch
                cids = cids.to(dtype=torch.int32, device=device, non_blocking=True)
            else:
                im_q, im_k, cids = batch[0], batch[1], None

            img_count += im_q.shape[0]
            im_q, im_k = im_q.to(device, non_blocking=True), im_k.to(device, non_blocking=True)

            # Forward + backward
            with torch.amp.autocast("cuda", enabled=amp_enabled and device.type == "cuda", dtype=amp_dtype):
                loss, stats = model(im_q, im_k, cluster_ids=cids,
                                    mask_same_cluster=mask_cluster and use_pseudo, mask_topk=mask_topk if use_pseudo else 0)
            (scaler.scale(loss / grad_accum) if scaler else (loss / grad_accum)).backward()

            # Optimizer step
            if (step + 1) % grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip > 0 else float("inf"))
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.update_momentum_encoder()
                global_step += 1

            # Accumulate stats
            loss_sum += stats["loss"]
            pos_sum += stats["pos_sim"]
            neg_sum += stats["neg_sim"]
            std_sum += stats["emb_std"]
            if (step + 1) % grad_accum == 0:
                grad_norm_sum += float(grad_norm)
            n_steps += 1

            # Log to wandb
            if wandb_active and log_every > 0 and global_step % log_every == 0 and (step + 1) % grad_accum == 0:
                m = {"train/loss": stats["loss"], "train/pos_sim": stats["pos_sim"], "train/neg_sim": stats["neg_sim"],
                     "train/sim_gap": stats["pos_sim"] - stats["neg_sim"], "train/emb_std": stats["emb_std"],
                     "train/grad_norm": float(grad_norm), "train/lr": lr, "train/epoch": epoch, "train/p_digiface": p_digi}
                if pseudo_enabled:
                    m.update({"train/pseudo_prob": p_pseudo, "train/neg_masked_pct": stats.get("neg_masked_pct", 0)})
                m.update(log_gpu_memory())
                log_wandb(m, step=global_step)

            # Update progress bar with avg img/s
            now = time.perf_counter()
            avg_ips = img_count / (now - epoch_start) if now > epoch_start else 0
            pbar.set_postfix(loss=f"{stats['loss']:.4f}", pos=f"{stats['pos_sim']:.3f}", neg=f"{stats['neg_sim']:.3f}", ips=f"{avg_ips:.0f}")

        # Epoch summary
        elapsed = time.perf_counter() - epoch_start
        if n_steps > 0:
            avg_loss, avg_pos, avg_neg = loss_sum/n_steps, pos_sum/n_steps, neg_sum/n_steps
            gap, ips = avg_pos - avg_neg, img_count / elapsed

            if wandb_active:
                num_opt_steps = n_steps // grad_accum
                em = {"epoch/loss": avg_loss, "epoch/pos_sim": avg_pos, "epoch/neg_sim": avg_neg, "epoch/sim_gap": gap,
                      "epoch/emb_std": std_sum/n_steps, "epoch/grad_norm": grad_norm_sum/num_opt_steps if num_opt_steps > 0 else 0,
                      "epoch/images_per_sec": ips, "epoch/lr": lr, "epoch/number": epoch}
                if pseudo_mgr and pseudo_mgr.state:
                    em["epoch/pseudo_clusters"] = pseudo_mgr.num_clusters
                em.update(log_gpu_memory())
                log_wandb(em, step=global_step)

            done = epoch - start_epoch + 1
            eta = _fmt_time((time.perf_counter() - train_start) / done * (epochs - epoch - 1))
            logger.info(f"Epoch {epoch:03d} | loss={avg_loss:.4f} pos={avg_pos:.3f} neg={avg_neg:.3f} gap={gap:.3f} | {ips:.0f} ips | ETA: {eta}")

        # Checkpoint
        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            path = save_checkpoint(out_dir, epoch=epoch+1, model=model, optimizer=optimizer, scaler=scaler, cfg=cfg, pseudo_manager=pseudo_mgr)

            # Upload to W&B
            if wandb_active and cfg.get("wandb", {}).get("save_artifacts"):
                art = wandb.Artifact("checkpoint", type="model", metadata={"epoch": epoch+1})
                art.add_file(str(path))
                wandb.log_artifact(art)
                logger.info(f"Uploaded: {path.name}")
                # Prune old artifact versions, keep last 5
                try:
                    api = wandb.Api()
                    collection = api.artifact_collection("model", f"{wandb.run.entity}/{wandb.run.project}/checkpoint")
                    versions = sorted(collection.versions(), key=lambda a: int(a.version.lstrip("v")), reverse=True)
                    for old in versions[5:]:
                        old.delete()
                except Exception:
                    pass  # Don't crash training if cleanup fails

            # Handle local storage
            if save_local:
                prune_checkpoints(out_dir / "checkpoints", keep_last)
                logger.info(f"Saved locally: {path.name}")
            else:
                path.unlink(missing_ok=True)  # Delete after W&B upload

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Done
    total = _fmt_time(time.perf_counter() - train_start)
    logger.info(f"\n{'='*70}\n  Training complete in {total}\n  Checkpoints: {out_dir/'checkpoints'}\n{'='*70}\n")
    finish_wandb()
