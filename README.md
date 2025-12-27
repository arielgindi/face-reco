# SniperFace

Minimal (3-file) codebase to train a **label-free** face embedding model using
**MoCo + MarginNCE**, then export embeddings and evaluate retrieval accuracy on
**unseen identities**.

Uses **Hydra** for configuration and **Weights & Biases** for experiment tracking.

## Install / Run (uv-first)

```bash
uv sync
```

All commands are run via Hydra config overrides:

```bash
uv run python sniperface.py                    # Train with defaults
uv run python sniperface.py train.epochs=100   # Override any config value
uv run python sniperface.py wandb.enabled=false # Disable W&B
```

## 1) Train (MoCo + MarginNCE)

Training **automatically** creates identity-disjoint splits (75% train / 25% test)
and excludes test identities. No manual split step needed.

```bash
uv run python sniperface.py
```

Override settings from CLI:

```bash
uv run python sniperface.py train.epochs=30 train.batch.size=64
```

Outputs (in Hydra output dir):

- `checkpoints/epoch_XXX.pt`
- `identity_splits.parquet`

## 2) Embed (export embeddings)

```bash
uv run python sniperface.py command=embed
```

By default, exports embeddings for **test identities only** (the 25% held out).

## 3) Evaluate (Rank-1 / Top-K)

```bash
uv run python sniperface.py command=eval
```

## Configuration

All settings are in `conf/config.yaml`. Key sections:

- `data` - Parquet file globs and split settings
- `train` - Epochs, batch size, optimizer, LR schedule
- `wandb` - W&B project, tags, logging frequency
- `augmentation` - Two-view augmentations for contrastive learning

## Notes

- The training step only consumes `(view_q, view_k)` tensors. Identity is never
  passed into the SSL loss.
- Test identities (25%) are **automatically protected** during training.
- `conf/config.yaml` controls model/training/augmentation hyperparameters.
