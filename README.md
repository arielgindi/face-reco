# face_sniper_ssl

Minimal (3-file) codebase to train a **label-free** face embedding model using
**MoCo + MarginNCE**, then export embeddings and evaluate retrieval accuracy on
**unseen identities**.

## Install / Run (uv-first)

```bash
uv sync
```

All commands are run as:

```bash
uv run python sniperface.py <command> ...
```

## 1) Split (identity-disjoint)

```bash
uv run python sniperface.py split \
  --digiface-glob "data/digiface1m_*.parquet" \
  --digi2real-glob "data/digi2real_*.parquet" \
  --out-dir "data/splits" \
  --train-ratio 0.75 \
  --seed 42 \
  --materialize
```

Outputs:

- `data/splits/identity_splits.parquet`
- `data/splits/train/{digiface,digi2real}/*.parquet`
- `data/splits/test/{digiface,digi2real}/*.parquet`

## 2) Train (MoCo + MarginNCE)

```bash
uv run python sniperface.py train \
  --config config.yaml \
  --train-digiface-glob "data/splits/train/digiface/*.parquet" \
  --train-digi2real-glob "data/splits/train/digi2real/*.parquet" \
  --out "runs/run_001" \
  --num-workers 4
```

Outputs:

- `runs/run_001/checkpoints/epoch_XXX.pt`
- `runs/run_001/train_log.jsonl`

## 3) Embed (export embeddings)

```bash
uv run python sniperface.py embed \
  --checkpoint "runs/run_001/checkpoints/epoch_050.pt" \
  --test-glob "data/splits/test/*/*.parquet" \
  --out "embeddings/test_embeddings.parquet" \
  --batch-size 256
```

## 4) Evaluate (Rank-1 / Top-K)

```bash
uv run python sniperface.py eval \
  --embeddings "embeddings/test_embeddings.parquet" \
  --enroll-per-id 5 \
  --seed 42 \
  --top-k 5 \
  --out "runs/run_001/metrics.json"
```

## Notes

- The training step only consumes `(view_q, view_k)` tensors. Identity is never
  passed into the SSL loss.
- `config.yaml` controls model/training/augmentation hyperparameters.
