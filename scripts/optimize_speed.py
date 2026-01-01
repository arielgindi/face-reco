#!/usr/bin/env python3
"""Bayesian optimization for finding optimal training speed parameters."""

import subprocess
import re
import time
from dataclasses import dataclass

# Try to import optuna, fall back to simple grid search
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("optuna not found, using manual search")


@dataclass
class TrialResult:
    batch_size: int
    num_workers: int
    lr: float
    avg_ips: float
    min_ips: float
    max_ips: float
    success: bool
    error: str = ""


def run_trial(batch_size: int, num_workers: int, n_epochs: int = 2) -> TrialResult:
    """Run a single training trial and measure ips."""
    # Calculate LR based on batch size (linear scaling from base 0.02 at batch 256)
    effective_batch = batch_size * 3  # 3 GPUs
    lr = 0.02 * (effective_batch / 256)

    cmd = [
        "/root/.local/bin/uv", "run", "torchrun",
        "--standalone", "--nproc_per_node=3", "main.py",
        f"data.binary_cache_path=/dev/shm/images.npy",
        f"data.data_fraction=0.03",  # 3% for faster trials
        f"train.epochs={n_epochs}",
        "train.resume=null",
        f"train.batch.size={batch_size}",
        "train.batch.grad_accum_steps=1",
        f"train.optimizer.lr={lr:.4f}",
        f"data.num_workers={num_workers}",
        "wandb.enabled=false",
    ]

    print(f"\n{'='*60}")
    print(f"Trial: batch={batch_size}, workers={num_workers}, lr={lr:.4f}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            cwd="/workspace/face-reco",
            env={
                "UV_CACHE_DIR": "/root/.uv_cache",
                "UV_PROJECT_ENVIRONMENT": "/root/venv",
                "PATH": "/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
            }
        )
        output = result.stdout + result.stderr

        # Check for OOM
        if "CUDA out of memory" in output or "OutOfMemoryError" in output:
            print(f"  -> OOM at batch_size={batch_size}")
            return TrialResult(batch_size, num_workers, lr, 0, 0, 0, False, "OOM")

        # Parse ips values from epoch summaries (format: "| XXXX ips |")
        ips_pattern = r'\| (\d+) ips \|'
        ips_values = [int(m) for m in re.findall(ips_pattern, output)]

        if not ips_values:
            print(f"  -> No ips found in output")
            print(f"  Output: {output[-500:]}")  # Last 500 chars
            return TrialResult(batch_size, num_workers, lr, 0, 0, 0, False, "No ips")

        # Skip first epoch (warmup), use rest for average
        if len(ips_values) > 1:
            ips_values = ips_values[1:]  # Skip warmup epoch

        avg_ips = sum(ips_values) / len(ips_values)
        min_ips = min(ips_values)
        max_ips = max(ips_values)

        print(f"  -> IPS: avg={avg_ips:.0f}, min={min_ips}, max={max_ips}")
        return TrialResult(batch_size, num_workers, lr, avg_ips, min_ips, max_ips, True)

    except subprocess.TimeoutExpired:
        print(f"  -> Timeout")
        return TrialResult(batch_size, num_workers, lr, 0, 0, 0, False, "Timeout")
    except Exception as e:
        print(f"  -> Error: {e}")
        return TrialResult(batch_size, num_workers, lr, 0, 0, 0, False, str(e))


def objective(trial) -> float:
    """Optuna objective function."""
    batch_size = trial.suggest_int("batch_size", 256, 448, step=32)
    num_workers = trial.suggest_int("num_workers", 8, 16, step=2)

    result = run_trial(batch_size, num_workers, n_epochs=2)

    if not result.success:
        return 0.0  # Failed trial

    return result.avg_ips


def run_optuna_optimization(n_trials: int = 15):
    """Run Bayesian optimization with Optuna."""
    print("\n" + "="*60)
    print("BAYESIAN OPTIMIZATION (Optuna)")
    print("="*60)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best IPS: {study.best_value:.0f}")
    print(f"Best params: {study.best_params}")

    return study


def run_grid_search():
    """Run manual grid search if optuna not available."""
    print("\n" + "="*60)
    print("GRID SEARCH OPTIMIZATION")
    print("="*60)

    results = []

    # Test key batch sizes
    for batch_size in [288, 320, 352, 384, 416]:
        for num_workers in [10, 12, 14]:
            result = run_trial(batch_size, num_workers, n_epochs=2)
            results.append(result)

            if not result.success:
                # If OOM, skip larger batch sizes for this worker count
                if result.error == "OOM":
                    break

    # Find best
    successful = [r for r in results if r.success]
    if successful:
        best = max(successful, key=lambda r: r.avg_ips)
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Best IPS: {best.avg_ips:.0f}")
        print(f"Best params: batch_size={best.batch_size}, num_workers={best.num_workers}")

        # Print all results
        print("\nAll results:")
        for r in sorted(successful, key=lambda x: -x.avg_ips):
            print(f"  batch={r.batch_size}, workers={r.num_workers}: {r.avg_ips:.0f} ips (min={r.min_ips}, max={r.max_ips})")

    return results


if __name__ == "__main__":
    print("="*60)
    print("TRAINING SPEED OPTIMIZATION")
    print("="*60)
    print(f"Optuna available: {HAS_OPTUNA}")

    if HAS_OPTUNA:
        study = run_optuna_optimization(n_trials=12)
    else:
        run_grid_search()
