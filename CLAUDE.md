# Face Recognition Project - Knowledge Base

## Server Access

```bash
# SSH to RunPod server
ssh root@213.173.102.207 -p 14960

# SSH key location (Windows)
C:\Users\ariel\.ssh\id_ed25519
```

## Environment Setup (Remote Server)

```bash
# Required environment variables
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT=/root/venv

# Data location (RAM disk for speed)
/dev/shm/images.npy  # 43GB, 1.2M images
```

## Training Commands

```bash
# Full training (3 GPUs) - Optimized settings
uv run torchrun --nproc_per_node=3 -m src.commands.train \
  data.binary_cache_path=/dev/shm/images.npy \
  train.batch.size=384 \
  train.batch.grad_accum_steps=1 \
  train.optimizer.lr=0.03 \
  train.lr_schedule.warmup.start_lr=0.003

# Test with 1% data
uv run torchrun --nproc_per_node=3 -m src.commands.train \
  data.binary_cache_path=/dev/shm/images.npy \
  train.batch.size=384 \
  train.optimizer.lr=0.03 \
  data.data_fraction=0.01

# Resume from specific checkpoint
train.resume=/tmp/wandb_checkpoints/epoch_014.pt
```

## tmux Session

```bash
tmux new -s train          # Create new session
tmux attach -t train       # Reattach
Ctrl+B, D                  # Detach
```

## Hardware

- 3x NVIDIA RTX A4500 (20GB each)
- 251GB RAM
- RunPod cloud instance

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| Embedding speed | 9,000 img/s (3 GPUs parallel) |
| k-NN search | 13,000 q/s |
| Training IPS | ~1,400 total (473/GPU) |
| Batch size | 384/GPU × 3 = 1,152 effective |
| Steps per epoch | ~1,059 (per GPU) |
| Epoch time | ~5 min (training) + 4 min (pseudo-ID) |

## Key Config Paths

```
conf/config.yaml          # Main config (Hydra)
src/pseudo.py             # Pseudo-ID mining (embedding, FAISS, clustering)
src/commands/train.py     # Training loop
src/model/moco.py         # MoCo model
src/model/backbone.py     # IResNet50 backbone
```

## DDP Architecture

- `torchrun --nproc_per_node=3` creates 3 separate Python processes
- Each process has its own model on its designated GPU (rank 0 → cuda:0, etc.)
- Processes communicate via `torch.distributed` collectives (all_gather, broadcast)
- Only rank 0 should print logs, save checkpoints, run W&B

## Pseudo-ID Mining Flow

1. **Embedding**: Each rank embeds 1/3 of images → `all_gather` to combine
2. **FAISS k-NN**: Only rank 0 runs, uses all 3 GPUs via IndexShards
3. **Clustering**: Only rank 0, then broadcasts cluster_labels to all ranks
4. **Training**: All ranks use clusters for cross-image positive pairs

## Common Issues

### SIGBUS Error
- Usually memory/mmap issue with /dev/shm
- Check: `ls -la /dev/shm/images.npy && free -h`

### CUDA Device Mismatch
- Don't try to copy models across GPUs with deepcopy
- Each rank already has its model on the correct device

### Checkpoint Corruption
- Don't run tests that create checkpoints with partial data
- Delete bad checkpoints: `rm /tmp/wandb_checkpoints/epoch_*.pt`
- Keep only the real full-training checkpoint

## W&B Project

- Project: `sniperface-v2`
- Entity: `arielgindi`
- URL: https://wandb.ai/arielgindi/sniperface-v2

## Git Workflow

```bash
# Local (Windows)
git add . && git commit -m "message" && git push

# Remote (pull changes)
ssh root@213.173.102.207 -p 14960 "cd /workspace/face-reco && git pull"
```
