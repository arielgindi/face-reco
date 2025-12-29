"""Evaluation command for retrieval metrics."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import DictConfig
from tqdm import tqdm

import wandb

logger = logging.getLogger(__name__)


def cmd_eval(cfg: DictConfig) -> None:
    """Evaluate retrieval metrics on embeddings.

    Uses adaptive enroll_per_id to ensure all identities contribute queries.
    """
    import faiss

    eval_cfg = cfg.get("eval", {})
    emb_path = Path(os.getcwd()) / "embeddings_test.parquet"

    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}. Run embed first.")

    # Read embeddings
    parquet_file = pq.ParquetFile(emb_path)
    schema = parquet_file.schema_arrow

    if schema.metadata:
        ckpt_name = schema.metadata.get(b"checkpoint", b"unknown").decode()
        logger.info(f"Embeddings from checkpoint: {ckpt_name}")

    emb_field = schema.field("embedding")
    dim = int(emb_field.type.list_size)

    # Load all embeddings grouped by identity
    id_to_embs: dict[str, list[np.ndarray]] = {}
    total_rows = 0

    for batch in tqdm(parquet_file.iter_batches(batch_size=8192), desc="read", unit="batch"):
        ids = batch.column(0).to_pylist()
        emb_col = batch.column(batch.schema.get_field_index("embedding"))
        values = emb_col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        embs = values.reshape(-1, dim)

        for identity_id, vec in zip(ids, embs, strict=True):
            id_to_embs.setdefault(str(identity_id), []).append(vec.copy())
        total_rows += len(ids)

    identity_ids = sorted(id_to_embs.keys())
    n_ids = len(identity_ids)
    logger.info(f"Loaded {total_rows} embeddings from {n_ids} identities")

    # Compute adaptive enroll_per_id
    # Leave at least 1 image per identity for queries
    min_images = min(len(embs) for embs in id_to_embs.values())
    max_enroll = max(1, min_images - 1)
    requested_enroll = int(eval_cfg.get("enroll_per_id", 5))
    enroll_per_id = min(requested_enroll, max_enroll)

    if enroll_per_id < requested_enroll:
        logger.warning(
            f"Reduced enroll_per_id from {requested_enroll} to {enroll_per_id} "
            f"(min images per identity: {min_images})"
        )
    logger.info(f"Using enroll_per_id={enroll_per_id}")

    top_k = int(eval_cfg.get("top_k", 5))
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    rng = np.random.default_rng(seed)

    # Build centroids from enrollment images
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

    # Build FAISS index
    index = faiss.IndexFlatIP(dim)
    index.add(centroids)

    # Evaluate queries
    n_queries = 0
    rank1_hits = 0
    topk_hits = 0
    query_batch_size = int(eval_cfg.get("query_batch", 50000))

    batch_q: list[np.ndarray] = []
    batch_true: list[int] = []

    def flush_batch() -> None:
        nonlocal n_queries, rank1_hits, topk_hits, batch_q, batch_true
        if not batch_q:
            return

        q = np.stack(batch_q, axis=0).astype(np.float32, copy=False)
        true = np.array(batch_true, dtype=np.int64)
        _, idx = index.search(q, top_k)

        n_queries += q.shape[0]
        rank1_hits += int(np.sum(idx[:, 0] == true))
        topk_hits += int(np.sum(np.any(idx == true[:, None], axis=1)))

        batch_q.clear()
        batch_true.clear()

    for true_idx, identity_id in enumerate(tqdm(identity_ids, desc="queries", unit="id")):
        embs = np.stack(id_to_embs[identity_id], axis=0)
        enroll_idx = enroll_sets[identity_id]
        query_idx = [j for j in range(embs.shape[0]) if j not in enroll_idx]

        for j in query_idx:
            batch_q.append(embs[j])
            batch_true.append(true_idx)

            if len(batch_q) >= query_batch_size:
                flush_batch()

    flush_batch()

    # Compute metrics
    rank_1 = (rank1_hits / n_queries) if n_queries else 0.0
    top_k_acc = (topk_hits / n_queries) if n_queries else 0.0

    metrics = {
        "num_identities": n_ids,
        "num_embeddings": total_rows,
        "num_queries": n_queries,
        "enroll_per_id": enroll_per_id,
        "top_k": top_k,
        "rank_1": rank_1,
        "top_k_acc": top_k_acc,
    }

    out_path = Path(os.getcwd()) / "metrics.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Rank-1: {rank_1:.4f}, Top-{top_k}: {top_k_acc:.4f}")
    logger.info(f"({n_queries:,} queries from {n_ids:,} identities)")

    if wandb.run is not None:
        wandb.log({"eval/rank_1": rank_1, "eval/top_k_acc": top_k_acc})
