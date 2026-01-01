"""Training command for MoCo + MarginNCE face encoder."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src import data
from src.augmentations import build_album_transform
from src.checkpoint import (
    find_latest_checkpoint,
    load_checkpoint_for_resume,
    prune_checkpoints,
    save_checkpoint,
)
from src.model import build_moco
from src.pseudo import PseudoIDManager
from src.schedule import (
    build_pseudo_schedule,
    get_pseudo_prob,
    should_refresh_pseudo,
)
from src.utils import (
    DistributedContext,
    cleanup_distributed,
    compute_epoch_batch_counts,
    configure_precision,
    set_seed,
    setup_distributed,
)
from src.utils.platform import (
    supports_fork,
    supports_torch_compile,
)
from src.wandb_utils import finish_wandb, init_wandb, log_gpu_memory, log_wandb

logger = logging.getLogger(__name__)


def _fmt_time(s: float) -> str:
    """Format seconds as human-readable string."""
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{int(s // 3600)}h {int((s % 3600) // 60)}m"


def _build_lr_fn(cfg: DictConfig) -> Callable[[int], float]:
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


def _setup_training(
    cfg: DictConfig, dist_ctx: DistributedContext, out_dir: Path
) -> tuple[Any, SGD, Any, PseudoIDManager | None, int, int]:
    """Setup training components: model, optimizer, scaler, pseudo-ID manager, and resume checkpoint.

    Returns:
        Tuple of (model, optimizer, scaler, pseudo_manager, start_epoch, global_step)
    """
    device = dist_ctx.device

    # Model & optimizer
    model = build_moco(cfg, device=device)
    model.train()
    batch_size, grad_accum = (
        int(cfg.train.batch.size),
        int(cfg.train.batch.grad_accum_steps),
    )

    # Validate queue_size compatibility with DDP settings
    queue_size = (
        model.cfg.queue_size
        if not dist_ctx.enabled
        else model.module.cfg.queue_size
        if hasattr(model, "module")
        else model.cfg.queue_size
    )
    effective_batch = batch_size * dist_ctx.world_size
    if queue_size % effective_batch != 0 and dist_ctx.is_main:
        logger.debug(
            f"Queue size ({queue_size}) not divisible by batch ({effective_batch}), minor efficiency impact."
        )

    opt_cfg = cfg.train.optimizer
    optimizer = SGD(
        model.parameters(),
        lr=float(opt_cfg.lr),
        momentum=float(opt_cfg.momentum),
        weight_decay=float(opt_cfg.weight_decay),
        nesterov=bool(opt_cfg.nesterov),
    )

    # Setup AMP scaler
    amp_enabled, amp_dtype, _ = configure_precision(cfg)
    scaler = (
        torch.amp.GradScaler("cuda")
        if amp_enabled and device.type == "cuda" and amp_dtype == torch.float16
        else None
    )

    # Pseudo-ID manager
    pseudo_cfg = cfg.get("pseudo", {})
    pseudo_enabled = bool(pseudo_cfg.get("enabled", False))
    if pseudo_enabled:
        embed_cfg = pseudo_cfg.embed
        pseudo_mgr = PseudoIDManager(
            knn_k=int(pseudo_cfg.knn_k),
            mutual_topk=int(pseudo_cfg.mutual_topk),
            min_cluster_size=int(pseudo_cfg.min_cluster_size),
            max_cluster_size=int(pseudo_cfg.max_cluster_size),
            sim_threshold=float(pseudo_cfg.get("sim_threshold", 0.60)),
            batch_size_base=int(embed_cfg.batch_size_base),
            embed_batch_multiplier=int(embed_cfg.embed_batch_multiplier),
            embed_workers_multiplier=int(embed_cfg.embed_workers_multiplier),
            rejection_sampling_tries=int(embed_cfg.rejection_sampling_tries),
            faiss_temp_memory_gb=int(embed_cfg.faiss_temp_memory_gb),
            faiss_nprobe=int(embed_cfg.get("faiss_nprobe", 64)),
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
            resume_path,
            model,
            optimizer,
            scaler,
            device,
            pseudo_manager=pseudo_mgr,
            warm_start=bool(cfg.train.get("warm_start", False)),
        )

    # Wrap in DDP if distributed (before torch.compile)
    if dist_ctx.enabled:
        model = DDP(model, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank)
        if dist_ctx.is_main:
            logger.info(f"Wrapped model in DDP ({dist_ctx.world_size} GPUs)")

    # Torch compile (Unix only, after DDP wrap)
    if (
        device.type == "cuda"
        and supports_torch_compile()
        and cfg.train.precision.get("torch_compile")
    ):
        torch._dynamo.config.capture_scalar_outputs = True
        cc = torch.cuda.get_device_capability(device)
        if cc[0] >= 12:  # Blackwell (sm_120) - use cudagraphs backend (Triton has issues)
            if dist_ctx.is_main:
                logger.info(f"Using cudagraphs backend for Blackwell GPU (sm_{cc[0]}{cc[1]})")
            model = torch.compile(model, backend="cudagraphs")
        else:
            model = torch.compile(model, mode="reduce-overhead")

    # Compute global step from start epoch (binary mode only)
    data_cfg = cfg.get("data", {})
    binary_cache = data_cfg.get("binary_cache_path")
    if not binary_cache or not Path(binary_cache).exists():
        raise ValueError("Binary cache required: data.binary_cache_path")

    # Get dataset length without loading full array (avoid duplicate load)
    base_samples = data.get_binary_dataset_length(binary_cache)

    num_batches, _ = compute_epoch_batch_counts(
        base_samples=base_samples, batch_size=batch_size, grad_accum_steps=grad_accum
    )
    global_step = start_epoch * (num_batches // grad_accum)

    return model, optimizer, scaler, pseudo_mgr, start_epoch, global_step


def _save_checkpoint(
    cfg: DictConfig,
    out_dir: Path,
    epoch: int,
    model: Any,
    optimizer: SGD,
    scaler: Any,
    pseudo_mgr: PseudoIDManager | None,
    wandb_active: bool,
    save_local: bool,
    keep_last: int,
) -> None:
    """Save checkpoint and optionally upload to wandb."""
    path = save_checkpoint(
        out_dir,
        epoch=epoch + 1,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        cfg=cfg,
        pseudo_manager=pseudo_mgr,
    )

    # Upload to W&B with retry
    upload_success = False
    if wandb_active and cfg.get("wandb", {}).get("save_artifacts"):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                art = wandb.Artifact("checkpoint", type="model", metadata={"epoch": epoch + 1})
                art.add_file(str(path))
                wandb.log_artifact(art, aliases=["latest"])
                # Wait for upload with timeout (10 minutes)
                art.wait(timeout=600)
                upload_success = True
                logger.info(f"Uploaded: {path.name}")
                break
            except wandb.errors.CommError as e:
                # Network errors - retry with exponential backoff
                wait_time = 5 * (2**attempt)  # 5s, 10s, 20s
                logger.warning(
                    f"Network error on attempt {attempt + 1}/{max_attempts}: {e}. Retrying in {wait_time}s..."
                )
                if attempt < max_attempts - 1:
                    time.sleep(wait_time)
            except (PermissionError, wandb.errors.AuthenticationError) as e:
                # Auth errors - don't retry
                logger.error(f"Authentication/permission error during upload: {e}")
                break
            except TimeoutError as e:
                # Timeout - retry with exponential backoff
                wait_time = 5 * (2**attempt)
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{max_attempts}: {e}. Retrying in {wait_time}s..."
                )
                if attempt < max_attempts - 1:
                    time.sleep(wait_time)
            except OSError as e:
                # File I/O errors - don't retry
                logger.error(f"File I/O error during upload: {e}")
                break

        if not upload_success:
            logger.error(
                f"Failed to upload {path.name} after {max_attempts} attempts - keeping local copy"
            )

        # Prune old artifact versions, keep last 5
        if upload_success:
            try:
                api = wandb.Api()
                collection = api.artifact_collection(
                    "model", f"{wandb.run.entity}/{wandb.run.project}/checkpoint"
                )
                versions = sorted(
                    collection.versions(),
                    key=lambda a: int(a.version.lstrip("v")),
                    reverse=True,
                )
                for old in versions[5:]:
                    old.delete()
            except (wandb.errors.CommError, ValueError, KeyError) as e:
                logger.warning(f"Failed to prune old artifact versions: {e}")
            except Exception as e:
                # Catch unexpected errors to prevent training crash
                logger.warning(f"Unexpected error during artifact cleanup: {e}")

    # Handle local storage - keep if upload failed
    if save_local or not upload_success:
        prune_checkpoints(out_dir / "checkpoints", keep_last)
        logger.info(f"Saved locally: {path.name}")
    else:
        path.unlink(missing_ok=True)  # Delete only after confirmed upload


def cmd_train(cfg: DictConfig) -> None:
    """Train MoCo + MarginNCE face encoder with optional pseudo-ID bootstrapping."""
    # Setup distributed (auto-detects torchrun or single-GPU)
    dist_ctx = setup_distributed()
    device = dist_ctx.device

    seed = int(cfg.get("experiment", {}).get("seed", 42))
    set_seed(seed, bool(cfg.get("experiment", {}).get("deterministic", False)), rank=dist_ctx.rank)
    amp_enabled, amp_dtype, tf32 = configure_precision(cfg)
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(os.getcwd())

    # Only init wandb on main process
    wandb_active = init_wandb(cfg, out_dir) if dist_ctx.is_main else False
    log_every = int(cfg.get("wandb", {}).get("log_every_steps", 50))

    # Data configuration (binary mode only)
    data_cfg = cfg.get("data", {})

    # Setup training components
    model, optimizer, scaler, pseudo_mgr, start_epoch, global_step = _setup_training(
        cfg, dist_ctx, out_dir
    )
    num_params = sum(p.numel() for p in (model.module if dist_ctx.enabled else model).parameters())
    batch_size, grad_accum, epochs = (
        int(cfg.train.batch.size),
        int(cfg.train.batch.grad_accum_steps),
        int(cfg.train.epochs),
    )

    # Pseudo-ID configuration
    pseudo_cfg = cfg.get("pseudo", {})
    pseudo_enabled = bool(pseudo_cfg.get("enabled", False))

    # Binary dataset configuration
    binary_cache = data_cfg.get("binary_cache_path")
    if not binary_cache or not Path(binary_cache).exists():
        raise ValueError("Binary cache required: data.binary_cache_path")

    input_size = tuple(cfg.model.backbone.input_size)
    num_workers = int(data_cfg.get("num_workers", 8))
    prefetch_factor = int(data_cfg.get("prefetch_factor", 4))
    data_fraction = float(data_cfg.get("data_fraction", 1.0))
    if not 0.0 < data_fraction <= 1.0:
        raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction}")

    logger.info(f"Using binary cache: {binary_cache}")
    t_q = build_album_transform(cfg.augmentation.view_1, input_size=input_size)
    t_k = build_album_transform(cfg.augmentation.view_2, input_size=input_size)
    digiface_ds = data.BinaryImageDataset(binary_cache, t_q, t_k, seed=seed)
    base_samples = int(len(digiface_ds) * data_fraction)
    if data_fraction < 1.0:
        logger.info(f"Using {data_fraction:.1%} of data: {base_samples:,} samples")

    num_batches, num_samples = compute_epoch_batch_counts(
        base_samples=base_samples, batch_size=batch_size, grad_accum_steps=grad_accum
    )

    # Log config summary (main process only)
    if dist_ctx.is_main:
        pseudo, train = cfg.get("pseudo", {}), cfg.train
        gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"

        # Show distributed info
        if dist_ctx.enabled:
            logger.info(f"Distributed: {dist_ctx.world_size} GPUs | {gpu}")
        else:
            logger.info(f"Device: {gpu}")

        logger.info(
            f"Model: {cfg.model.backbone.name} ({num_params:,} params) | "
            f"Precision: {'FP16' if train.precision.amp else 'FP32'}"
        )

        # Show effective batch size for DDP
        batch_per_gpu = train.batch.size
        effective_batch = batch_per_gpu * dist_ctx.world_size * train.batch.grad_accum_steps
        logger.info(
            f"Training: {start_epoch}->{train.epochs - 1} epochs | batch={effective_batch} | lr={train.optimizer.lr}"
        )
        logger.info(f"Data: {num_samples:,} samples/epoch")
        if pseudo.get("enabled"):
            logger.info(
                f"Pseudo-ID: k={pseudo.get('knn_k', 20)} mutual={pseudo.get('mutual_topk', 5)}"
            )

    # Schedules
    lr_fn = _build_lr_fn(cfg)
    pseudo_sched = build_pseudo_schedule(cfg) if pseudo_enabled else ()
    neg_cfg = pseudo_cfg.get("negatives", {})
    mask_cluster, mask_topk = (
        bool(neg_cfg.get("mask_same_pseudo_in_queue", True)),
        int(neg_cfg.get("mask_topk_most_similar", 8)),
    )
    reset_queue = bool(neg_cfg.get("reset_queue_on_refresh", True))
    grad_clip = float(cfg.train.regularization.grad_clip_norm)
    save_every, keep_last = (
        int(cfg.train.checkpointing.save_every_epochs),
        int(cfg.train.checkpointing.keep_last),
    )
    save_local = bool(cfg.train.checkpointing.get("save_local", False))

    # Helper to access underlying model (unwrap DDP if needed)
    def get_moco():
        return model.module if dist_ctx.enabled else model

    # Training loop
    global_step = start_epoch * (num_batches // grad_accum)
    optimizer.zero_grad(set_to_none=True)
    train_start = time.perf_counter()

    try:
        for epoch in range(start_epoch, epochs):
            lr = lr_fn(epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            p_pseudo = get_pseudo_prob(epoch, pseudo_sched) if pseudo_enabled else 0.0

            # Pseudo-ID refresh (all ranks participate, but only main logs)
            # Refresh every 2 epochs starting from epoch 2
            if pseudo_mgr and should_refresh_pseudo(epoch, cfg):
                datasets = [digiface_ds]
                moco = get_moco()
                stats = pseudo_mgr.refresh(
                    moco,
                    datasets,
                    epoch,
                    pseudo_mgr.get_threshold(),
                    device,
                    batch_size,
                    num_workers,
                    max_images=base_samples,
                )
                if wandb_active and dist_ctx.is_main:
                    log_wandb(stats, step=global_step)
                if reset_queue:
                    moco.reset_queue()

            # Build epoch dataset (binary mode with optional pseudo-ID)
            base_ds = data.BinaryMixDataset(
                digiface_ds, None, 1.0, num_samples, seed + epoch * 17
            )
            use_pseudo = pseudo_mgr and pseudo_mgr.state is not None and p_pseudo > 0
            epoch_ds = (
                data.PseudoPairTwoViewDataset(
                    base_ds, pseudo_mgr, t_q, t_k, p_pseudo, num_samples, seed + epoch * 17
                )
                if use_pseudo
                else base_ds
            )

            loader_kw = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": device.type == "cuda",
                "drop_last": True,
            }
            if num_workers > 0:
                loader_kw.update(persistent_workers=True, prefetch_factor=prefetch_factor)
                # Use fork for binary mode (COW memory sharing)
                if supports_fork():
                    loader_kw["multiprocessing_context"] = "fork"
            loader = DataLoader(epoch_ds, **loader_kw)

            # Epoch tracking
            loss_sum = pos_sum = neg_sum = std_sum = grad_norm_sum = 0.0
            n_steps = 0
            grad_norm = 0.0
            epoch_start = time.perf_counter()
            img_count = 0

            # Determine training phase for display
            phase = "A:Stabilize" if epoch < 5 else "B:Bootstrap" if epoch < 35 else "C:Refine"
            pbar = tqdm(
                loader,
                total=num_batches,
                desc=f"Epoch {epoch:03d} [{phase}]",
                unit="batch",
                disable=not dist_ctx.is_main,
            )  # Only show progress on main process
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
                with torch.amp.autocast(
                    "cuda", enabled=amp_enabled and device.type == "cuda", dtype=amp_dtype
                ):
                    loss, stats = model(
                        im_q,
                        im_k,
                        cluster_ids=cids,
                        mask_same_cluster=mask_cluster and use_pseudo,
                        mask_topk=mask_topk if use_pseudo else 0,
                    )
                (scaler.scale(loss / grad_accum) if scaler else (loss / grad_accum)).backward()

                # Optimizer step
                if (step + 1) % grad_accum == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip if grad_clip > 0 else float("inf")
                    )
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    get_moco().update_momentum_encoder()
                    global_step += 1

                # Accumulate stats
                loss_sum += stats["loss"]
                pos_sum += stats["pos_sim"]
                neg_sum += stats["neg_sim"]
                std_sum += stats["emb_std"]
                if (step + 1) % grad_accum == 0:
                    grad_norm_sum += float(grad_norm)
                n_steps += 1

                # Log to wandb (main process only)
                if (
                    wandb_active
                    and dist_ctx.is_main
                    and log_every > 0
                    and global_step % log_every == 0
                    and (step + 1) % grad_accum == 0
                ):
                    m = {
                        "train/loss": stats["loss"],
                        "train/pos_sim": stats["pos_sim"],
                        "train/neg_sim": stats["neg_sim"],
                        "train/sim_gap": stats["pos_sim"] - stats["neg_sim"],
                        "train/emb_std": stats["emb_std"],
                        "train/grad_norm": float(grad_norm),
                        "train/lr": lr,
                        "train/epoch": epoch,
                    }
                    if pseudo_enabled:
                        m.update(
                            {
                                "train/pseudo_prob": p_pseudo,
                                "train/neg_masked_pct": stats.get("neg_masked_pct", 0),
                            }
                        )
                    m.update(log_gpu_memory())
                    log_wandb(m, step=global_step)

                # Update progress bar with avg img/s
                now = time.perf_counter()
                avg_ips = img_count / (now - epoch_start) if now > epoch_start else 0
                pbar.set_postfix(
                    loss=f"{stats['loss']:.4f}",
                    pos=f"{stats['pos_sim']:.3f}",
                    neg=f"{stats['neg_sim']:.3f}",
                    ips=f"{avg_ips:.0f}",
                )

            # Epoch summary (main process only for logging)
            elapsed = time.perf_counter() - epoch_start
            if n_steps > 0:
                avg_loss, avg_pos, avg_neg = (
                    loss_sum / n_steps,
                    pos_sum / n_steps,
                    neg_sum / n_steps,
                )
                # For DDP, multiply ips by world_size (each GPU processes batch_size images)
                ips = (img_count / elapsed * dist_ctx.world_size) if elapsed > 0 else 0
                gap = avg_pos - avg_neg

                if wandb_active and dist_ctx.is_main:
                    num_opt_steps = n_steps // grad_accum
                    em = {
                        "epoch/loss": avg_loss,
                        "epoch/pos_sim": avg_pos,
                        "epoch/neg_sim": avg_neg,
                        "epoch/sim_gap": gap,
                        "epoch/emb_std": std_sum / n_steps,
                        "epoch/grad_norm": grad_norm_sum / num_opt_steps
                        if num_opt_steps > 0
                        else 0,
                        "epoch/images_per_sec": ips,
                        "epoch/lr": lr,
                        "epoch/number": epoch,
                    }
                    if pseudo_mgr and pseudo_mgr.state:
                        em["epoch/pseudo_clusters"] = pseudo_mgr.num_clusters
                    em.update(log_gpu_memory())
                    log_wandb(em, step=global_step)

                if dist_ctx.is_main:
                    done = epoch - start_epoch + 1
                    eta = _fmt_time(
                        (time.perf_counter() - train_start) / done * (epochs - epoch - 1)
                    )
                    logger.info(
                        f"Epoch {epoch:03d} | loss={avg_loss:.4f} pos={avg_pos:.3f} neg={avg_neg:.3f} gap={gap:.3f} | {ips:.0f} ips | ETA: {eta}"
                    )

            # Checkpoint (main process only)
            if dist_ctx.is_main and ((epoch + 1) % save_every == 0 or epoch + 1 == epochs):
                # For DDP, save the underlying model (unwrap DDP wrapper)
                save_model = get_moco()
                _save_checkpoint(
                    cfg,
                    out_dir,
                    epoch,
                    save_model,
                    optimizer,
                    scaler,
                    pseudo_mgr,
                    wandb_active,
                    save_local,
                    keep_last,
                )

            # Synchronize all ranks after checkpoint (ensures all wait for main to finish saving/uploading)
            if dist_ctx.enabled:
                dist.barrier()

            if device.type == "cuda":
                torch.cuda.empty_cache()

    except Exception as e:
        # Log the error and ensure cleanup happens
        logger.error(f"Training failed with error: {e}", exc_info=True)
        # Re-raise to allow caller to handle if needed
        raise
    finally:
        # Always clean up distributed resources, even on failure
        # Done
        if dist_ctx.is_main:
            if "train_start" in locals():
                total = _fmt_time(time.perf_counter() - train_start)
                logger.info(
                    f"\n{'=' * 70}\n  Training complete in {total}\n  Checkpoints: {out_dir / 'checkpoints'}\n{'=' * 70}\n"
                )
            finish_wandb()

        # Clean up distributed
        cleanup_distributed()
