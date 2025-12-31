# SniperFace

Minimal codebase to train a **label-free** face embedding model using
**MoCo + MarginNCE** with mutual k-NN pseudo-ID bootstrapping.

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

## Training (MoCo + MarginNCE + Pseudo-ID)

Training uses **binary-only mode** for maximum speed (pre-decoded images in .npy format).
Pseudo-IDs are automatically mined via mutual k-NN clustering every 2 epochs.

```bash
uv run python sniperface.py
```

Override settings from CLI:

```bash
uv run python sniperface.py train.epochs=30 train.batch.size=64
```

Outputs (in Hydra output dir):

- `checkpoints/epoch_XXX.pt`

## Data Preparation

Convert parquet datasets to binary format:

```bash
uv run python fast_convert.py data/digiface1m_*.parquet
```

Or download pre-built binary cache:

```bash
uv run python gdrive_download.py <file_id> images.npy
```

## Configuration

All settings are in `conf/config.yaml`. Key sections:

- `data` - Binary dataset settings and conversion config
- `train` - Epochs, batch size, optimizer, LR schedule
- `pseudo` - Pseudo-ID mining parameters (k-NN, thresholds, refresh schedule)
- `wandb` - W&B project, tags, logging frequency
- `augmentation` - Two-view augmentations for contrastive learning

## Notes

- **Binary-only mode**: All data must be converted to .npy format first
- **Pseudo-ID training**: Identity labels are never used during training
- **Automatic refresh**: Pseudo-IDs are re-mined every 2 epochs
- `conf/config.yaml` controls all model/training/pseudo-ID hyperparameters
