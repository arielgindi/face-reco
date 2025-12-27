# Project Structure

Minimal codebase with 3 Python files, runnable via `uv`.

---

## 1) Repo Layout

```
face_sniper_ssl/
  pyproject.toml
  conf/
    config.yaml                # Hydra config (training hyperparams)
  sniperface.py                # single CLI entrypoint (train/embed/eval via Hydra)
  moco.py                      # model + loss + queue + momentum encoder
  dataio.py                    # parquet streaming + split/materialize helpers
  README.md                    # short run instructions
```

That’s **3 Python files total**.

---

## 2) “uv-first” workflow

### Environment

* We’ll use `uv` to manage dependencies and run commands.
* You’ll run everything like:
  `uv run python sniperface.py ...`

### Dependencies (high level)

In `pyproject.toml`, we’ll declare (no code yet):

* **torch**, **torchvision**
* **pyarrow** + **polars** (fast parquet scan/filter/materialize)
* **numpy**
* **pyyaml**
* **tqdm** (progress bars)
* **faiss-cpu** (evaluation indexing)

Optional (only if you want heavier augmentations / faster GPU aug):

* **kornia**

---

## 3) CLI design (one entrypoint, 4 commands)

All commands are subcommands of `sniperface.py`:

### A) `split`

**Purpose:** Create an identity‑disjoint split and (recommended) materialize split shards.

Inputs:

* `--digiface-glob data/digiface1m_*.parquet`
* `--digi2real-glob data/digi2real_*.parquet`
* `--out-dir data/splits/`
* `--train-ratio 0.75`
* `--seed 42`
* `--materialize true|false` (default true)

Outputs:

* `data/splits/identity_splits.parquet`
  Columns: `identity_id`, `split`
* (If materialize)
  `data/splits/train/digiface/*.parquet`
  `data/splits/train/digi2real/*.parquet`
  `data/splits/test/digiface/*.parquet`
  `data/splits/test/digi2real/*.parquet`

**Why materialize?**
It removes expensive “is this identity in train?” checks during every epoch and makes training *much* faster and simpler.

---

### B) `train`

**Purpose:** Train MoCo + MarginNCE **without using identity IDs**.

Inputs:

* `--config config.yaml`
* `--train-digiface-glob data/splits/train/digiface/*.parquet`
* `--train-digi2real-glob data/splits/train/digi2real/*.parquet`
* `--out runs/run_001/`

What it does:

* Loads hyperparams from `config.yaml`
* Builds a streaming dataset that yields **images only**
* Applies **two-view augmentations**
* Runs MoCo training for `epochs=50`
* Saves:

  * `runs/run_001/checkpoints/epoch_XX.pt`
  * `runs/run_001/checkpoints/best.pt` (optional)
  * `runs/run_001/train_log.jsonl` (loss, pos/neg similarity stats)

Important enforcement:

* The batch returned to the training step is only:

  * `view1_tensor`, `view2_tensor`
* Identity is never passed into the training step.

---

### C) `embed`

**Purpose:** Freeze encoder and export embeddings for evaluation.

Inputs:

* `--checkpoint runs/run_001/checkpoints/epoch_50.pt`
* `--test-glob data/splits/test/*/*.parquet`
* `--out embeddings/test_embeddings.parquet`
* `--batch-size 256`

Outputs:

* `test_embeddings.parquet`:

  * `identity_id` (only for evaluation/analysis)
  * `image_filename`
  * `embedding` (float32[512])

---

### D) `eval`

**Purpose:** Compute Rank‑1/Top‑5 on the unseen 25% identities using centroid enrollment.

Inputs:

* `--embeddings embeddings/test_embeddings.parquet`
* `--enroll-per-id 5`
* `--seed 42`
* `--out runs/run_001/metrics.json`

What it does:

* For each identity in the test set:

  * choose E=5 enrollment embeddings
  * compute centroid (mean → L2 normalize)
* Build FAISS `IndexFlatIP` over all centroids
* Query remaining images, measure:

  * Rank‑1, Top‑5 (and optionally Top‑10)

Outputs:

* `runs/run_001/metrics.json`
* Optional: histograms/plots later (not required for minimal run)

---

## 4) What each Python file contains (planned)

### `dataio.py` (data + splitting + streaming)

Responsibilities:

1. **Split builder**

   * `collect_unique_identities(globs) -> list[str]`
   * `write_identity_splits(ids, out_path, train_ratio, seed)`
2. **Materializer**

   * `materialize_split(input_glob, splits_table, out_dir, split_name)`
   * Writes clean train/test shards
3. **Training stream dataset**

   * `ParquetImageStream(globs, shuffle_files=True, shuffle_buffer=N, batch_read_rows=M)`
   * Produces `image_tensor` (and optionally filename)
   * Drops identity entirely for training
4. **Curriculum mixer**

   * `CurriculumMixStream(digiface_stream, digi2real_stream, epoch, schedule)`
   * Implements epoch‑based mixing ratios

Key design choice:

* Use **streaming scan** of Parquet into batches → shuffle inside batch → decode bytes → yield tensors

---

### `moco.py` (model + MoCo mechanics)

Responsibilities:

1. iResNet‑50 backbone builder (112×112 friendly)
2. Projection head (MLP)
3. MoCo wrapper:

   * momentum encoder update
   * queue enqueue/dequeue
4. MarginNCE loss:

   * temperature
   * margin applied to positive logit

Everything needed to compute:

* `loss, pos_sim, neg_sim, queue_ptr`

---

### `sniperface.py` (the only "script")

Responsibilities:

* CLI via Hydra (config overrides from command line)
* Load config from `conf/config.yaml`
* Wire together data stream + model + optimizer + AMP scaler
* Run the loop for `train`
* Run `embed` and `eval`
* Handle output folders + logging

This file stays small because the heavy logic lives in `dataio.py` and `moco.py`.

---

## 5) Run order (the exact sequence you'll execute)

All commands use Hydra config overrides. Training **automatically** creates identity-disjoint splits and excludes test identities (25%).

1. **Train** (auto-creates splits, excludes 25% test identities)

* `uv run python sniperface.py`
* `uv run python sniperface.py train.epochs=100`

2. **Embed**

* `uv run python sniperface.py command=embed`

3. **Evaluate**

* `uv run python sniperface.py command=eval`

---

## 6) Identity filtering (automatic)

Training **automatically** filters identities during streaming:

* 75% identities → training (automatically excluded from test)
* 25% identities → test (automatically excluded from training)

No manual materialization needed. The `dataio.py` `allowed_identities` parameter filters during Parquet streaming.