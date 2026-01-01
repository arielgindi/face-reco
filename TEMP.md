# Guide for Claude Code: Fixing Multi-GPU Pseudo-ID Mining

## The Problem You Must Understand

When `torchrun --nproc_per_node=3` runs, it creates **3 separate Python processes**. Each process:
- Has its own copy of the entire program
- Has its own model already on its designated GPU (rank 0 → cuda:0, rank 1 → cuda:1, rank 2 → cuda:2)
- Shares NOTHING with other processes unless you explicitly use `torch.distributed` collectives

**The current code treats multi-GPU as "one process controlling multiple GPUs" but DDP is "multiple processes, each with one GPU".**

---

## Current Bugs (In Order of Severity)

### Bug 1: FAISS Runs on ALL Ranks, Each Grabbing ALL GPUs

In `_build_knn_graph()`, there is NO `if is_main:` guard. All 3 ranks execute:

```python
for i in range(num_gpus):  # 0, 1, 2
    gpu_index = faiss.index_cpu_to_gpu(gpu_res, i, sub_index)
```

**Result**: 3 processes × 3 GPUs = 9 FAISS allocations fighting for 3 GPUs.

### Bug 2: Embedding Wastes 2 GPUs

Only rank 0 embeds. Ranks 1 and 2 wait at `broadcast()`. But ranks 1 and 2 have perfectly good models sitting idle on their GPUs.

### Bug 3: Clustering Runs 3 Times Identically

`_cluster_components()` has no rank guard. All 3 ranks compute the same clustering.

### Bug 4: Broadcast Sends Wrong Data

The broadcast after embedding sends `all_embeddings` from rank 0, but ranks 1 and 2 have empty tensors. This works but is wasteful - we computed on 1 GPU what could have been computed on 3.

---

## The Clean Solution: Two Phases

### Phase 1: Distributed Embedding (All Ranks Participate)

Each rank embeds its SHARD of the data using its LOCAL model:

```
Total images: 1,200,000

Rank 0 (cuda:0): embeds images [0 ... 400,000)      → local_emb_0
Rank 1 (cuda:1): embeds images [400,000 ... 800,000) → local_emb_1  
Rank 2 (cuda:2): embeds images [800,000 ... 1,200,000) → local_emb_2
```

Then use `torch.distributed.all_gather()` to combine:

```
After all_gather, ALL ranks have:
full_embeddings = [local_emb_0 | local_emb_1 | local_emb_2]  (1,200,000 × 512)
```

**Key insight**: You do NOT need to copy models. Each rank already has `model.backbone_k` on its own device. Just use it directly.

### Phase 2: Centralized FAISS + Clustering (Rank 0 Only)

Only rank 0 does the expensive graph operations:

```python
if is_main:
    knn_sim, knn_idx = self._build_knn_graph(embeddings)  # Uses all 3 GPUs safely
    mutual_edges = self._filter_mutual_edges(...)
    cluster_labels, cluster_to_images = self._cluster_components(...)
else:
    cluster_labels = torch.empty(num_images, dtype=torch.int32, device=device)

# Broadcast results to all ranks
torch.distributed.broadcast(cluster_labels, src=0)
```

**Why this is safe**: When only rank 0 runs FAISS, it can use all 3 GPUs without contention. Ranks 1 and 2 are waiting at the broadcast, not competing for GPU memory.

---

## Implementation Steps

### Step 1: Modify `_extract_binary_fast()` for Distributed Embedding

```
1. Calculate this rank's shard:
   - rank = torch.distributed.get_rank()
   - world_size = torch.distributed.get_world_size()
   - shard_size = (num_images + world_size - 1) // world_size
   - start = rank * shard_size
   - end = min(start + shard_size, num_images)

2. Each rank embeds ONLY images[start:end] using model.backbone_k (already on correct device)

3. Pad local embeddings to same size (for all_gather):
   - local_emb shape: (shard_size, 512) - pad with zeros if last shard is smaller

4. all_gather into list of tensors:
   - gathered = [torch.empty_like(local_emb) for _ in range(world_size)]
   - torch.distributed.all_gather(gathered, local_emb)

5. Concatenate and trim to actual num_images:
   - full_embeddings = torch.cat(gathered, dim=0)[:num_images]

6. Return full_embeddings (all ranks now have identical complete embeddings)
```

### Step 2: Guard FAISS and Clustering with `if is_main:`

In `refresh()` method:

```
1. After embedding (which now uses all GPUs efficiently):
   
2. ONLY rank 0 does:
   - _build_knn_graph()
   - _filter_mutual_edges()  
   - _cluster_components()

3. Rank 0 creates cluster_labels tensor on its device

4. All ranks: broadcast cluster_labels from rank 0

5. All ranks: rebuild cluster_to_images dict from the broadcasted labels
   (This is fast - just a Python dict comprehension)
```

### Step 3: Fix FAISS to Only Run on Main Rank

The FAISS code itself is fine for multi-GPU usage. The problem is that all 3 ranks call it. Simply wrap the entire `_build_knn_graph()` call site with `if is_main:`.

---

## Progress Display Fix

Currently the progress shows GPU 0, 1, 2 bars but they represent FAISS shards (sequential), not parallel work.

After the fix:
- Embedding phase: Show 3 GPU bars that fill simultaneously (true parallelism)
- FAISS phase: Only rank 0 shows progress (others are waiting)

For embedding progress with all_gather, each rank can only update its own progress. Use `all_reduce` to sum progress across ranks for the total bar, or just show rank 0's view of total progress.

---

## Testing Instructions

### Test 1: Verify Multi-GPU Embedding Works (1% Data)

```bash
uv run torchrun --standalone --nproc_per_node=3 main.py \
  train.batch.size=384 \
  train.optimizer.lr=0.03 \
  data.binary_cache_path=/dev/shm/images.npy \
  data.data_fraction=0.01 \
  train.epochs=1
```

**Expected with 1% data (~12,000 images)**:
- Embedding should complete in ~15 seconds (not 45 seconds)
- You should see activity on all 3 GPUs during embedding (use `nvidia-smi -l 1` in another terminal)
- Total throughput should be ~9,000 img/s (3x single GPU)

### Test 2: Verify FAISS Doesn't Crash

After embedding completes, FAISS k-NN should:
- Only show "Loading faiss" message ONCE (not 3 times)
- Complete without memory errors
- Show reasonable query speed (~7,000 q/s)

### Test 3: Verify Training Speed

After pseudo-ID mining completes:
- Training should show ~1,300-1,500 total ips (3 GPUs × ~450 ips each)
- Each epoch should take ~15 minutes for full data

---

## Expected Timeline After Fix

| Phase | Before Fix | After Fix |
|-------|-----------|-----------|
| Embedding 1.2M images | 6.5 min (1 GPU) | ~2.2 min (3 GPUs) |
| FAISS k-NN | 3.0 min (fighting) | ~2.5 min (no contention) |
| Clustering | 13s × 3 = wasted | 13s × 1 |
| **Total pseudo-ID refresh** | **~10 min** | **~5 min** |

---

## Common Mistakes to Avoid

1. **Don't try to copy models across GPUs** - This causes the device placement bugs you saw. Each rank already has its model.

2. **Don't use ThreadPoolExecutor for multi-GPU** - That's for one process controlling multiple GPUs. DDP is multiple processes.

3. **Don't forget to handle uneven shards** - If 1,219,995 images ÷ 3 = 406,665 each, the last rank gets 406,665 images. Use padding for all_gather.

4. **Don't broadcast large tensors unnecessarily** - Embeddings (1.2M × 512 × 4 bytes = 2.4GB) should use all_gather, not broadcast. Cluster labels (1.2M × 4 bytes = 5MB) are fine to broadcast.

5. **Always synchronize before accessing shared results** - After all_gather/broadcast, all ranks have the data. No need for barriers, but don't access before the collective completes.

---

## Verification Checklist

After implementing, confirm:

- [ ] `nvidia-smi` shows all 3 GPUs active during embedding phase
- [ ] "Loading faiss" appears only ONCE in logs
- [ ] Embedding speed is ~9,000 img/s (not ~3,000)
- [ ] No CUDA device mismatch errors
- [ ] Training loop still works (DDP gradient sync)
- [ ] Checkpoint saving works (only rank 0 saves)
- [ ] 1% data test completes in < 2 minutes total