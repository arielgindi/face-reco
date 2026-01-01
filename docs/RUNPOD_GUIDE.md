# RunPod Training Guide

Complete guide for running SniperFace training on RunPod cloud GPUs.

## Overview

This guide covers deploying and running the MoCo + MarginNCE face encoder training on RunPod with optimal settings.

**Training specs:**
- 1.2M images (DigiFace-1M dataset)
- 75 epochs with pseudo-ID bootstrapping
- ~48 hours on 3x RTX A4500

---

## Step 1: RunPod Setup

### 1.1 Prerequisites
- RunPod account with billing configured
- Network volume with dataset (`images.npy` - 43GB binary cache)
- GitHub repo cloned locally

### 1.2 Create Network Volume (one-time)
1. Go to **Storage** → **Network Volumes**
2. Create volume: `facereco-s3-volume` (50GB minimum)
3. Select data center (e.g., `EU-RO-1`)
4. Upload `images.npy` to the volume

### 1.3 Deploy Pod
1. Go to **Pods** → **Deploy**
2. Filter by your network volume's data center
3. Select GPU: **RTX A4500** (best value: $0.25/hr, 20GB VRAM)
   - Recommended: 3x GPUs for faster training
4. Select template: **Better PyTorch 2.6.0 CUDA12.4**
5. Attach your network volume
6. Deploy (On-Demand for reliability)

### 1.4 Get SSH Connection
1. Click on your pod → **Connect**
2. Copy SSH command: `ssh root@<IP> -p <PORT>`

---

## Step 2: Initial Setup (First Time Only)

Run these commands after SSH'ing into the pod:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# Clone repository
cd /workspace
git clone https://github.com/arielgindi/face-reco.git
cd face-reco

# Configure environment (prevents conflicts with system Python)
export UV_CACHE_DIR=/root/.uv_cache
export UV_PROJECT_ENVIRONMENT=/root/venv

# Install dependencies with CUDA 12.4 support
uv sync --extra cu124
```

---

## Step 3: Prepare Training

### 3.1 Copy Dataset to RAM (Required Each Session)

The dataset must be in `/dev/shm` (RAM) for maximum speed:

```bash
# Check available RAM
df -h /dev/shm

# Copy dataset (takes ~2-3 minutes)
cp /workspace/images.npy /dev/shm/

# Verify
ls -lh /dev/shm/images.npy
```

### 3.2 Pull Latest Code

```bash
cd /workspace/face-reco
git pull
```

---

## Step 4: Run Training

### 4.1 Quick Test (Optional)

Test with 1% of data to verify setup:

```bash
cd /workspace/face-reco
export UV_CACHE_DIR=/root/.uv_cache
export UV_PROJECT_ENVIRONMENT=/root/venv

uv run main.py \
    data.binary_cache_path=/dev/shm/images.npy \
    data.data_fraction=0.01 \
    train.epochs=2 \
    train.resume=null
```

### 4.2 Full Training (Background)

Run training in background so it continues after SSH disconnects:

```bash
cd /workspace/face-reco
export UV_CACHE_DIR=/root/.uv_cache
export UV_PROJECT_ENVIRONMENT=/root/venv

nohup uv run main.py data.binary_cache_path=/dev/shm/images.npy > /workspace/training.log 2>&1 &
```

### 4.3 Resume Training

Training auto-resumes from W&B checkpoint. Just run the same command:

```bash
nohup uv run main.py data.binary_cache_path=/dev/shm/images.npy > /workspace/training.log 2>&1 &
```

---

## Step 5: Monitor Training

### View Live Logs
```bash
tail -f /workspace/training.log
```

### Check if Training is Running
```bash
ps aux | grep python
```

### Check GPU Usage
```bash
nvidia-smi
```

### W&B Dashboard
Training logs to: https://wandb.ai/arielgindi/sniperface-v2

---

## Quick Reference: All-in-One Commands

### Fresh Pod Setup + Training
```bash
# One-time setup
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
cd /workspace && git clone https://github.com/arielgindi/face-reco.git && cd face-reco
export UV_CACHE_DIR=/root/.uv_cache && export UV_PROJECT_ENVIRONMENT=/root/venv
uv sync --extra cu124

# Copy data to RAM
cp /workspace/images.npy /dev/shm/

# Start training
nohup uv run main.py data.binary_cache_path=/dev/shm/images.npy > /workspace/training.log 2>&1 &
```

### Returning to Existing Pod
```bash
# Set environment
cd /workspace/face-reco
export UV_CACHE_DIR=/root/.uv_cache
export UV_PROJECT_ENVIRONMENT=/root/venv

# Copy data to RAM (if pod was restarted)
cp /workspace/images.npy /dev/shm/

# Pull latest code and resume
git pull
nohup uv run main.py data.binary_cache_path=/dev/shm/images.npy > /workspace/training.log 2>&1 &
```

### Remote Commands (From Local Machine)
```bash
# Check training status
ssh -o StrictHostKeyChecking=no root@<IP> -p <PORT> "tail -20 /workspace/training.log"

# Check if running
ssh -o StrictHostKeyChecking=no root@<IP> -p <PORT> "ps aux | grep python | grep -v grep"

# View GPU usage
ssh -o StrictHostKeyChecking=no root@<IP> -p <PORT> "nvidia-smi"
```

---

## Troubleshooting

### CUDA Out of Memory
The embedding batch size auto-calculates based on free GPU memory. If OOM still occurs:
```bash
# Edit conf/config.yaml and reduce batch size
train:
  batch:
    size: 128  # Reduce from 256
```

### FAISS Not Using GPU
Verify FAISS GPU support:
```bash
uv run python -c "import faiss; print(f'GPUs: {faiss.get_num_gpus()}')"
```
Should show `GPUs: 3` (or your GPU count).

### Training Crashed / Pod Restarted
1. Data in `/dev/shm` is lost on restart - recopy it
2. Training auto-resumes from last W&B checkpoint
3. Just run the training command again

### Wrong Python Environment
Always set these before running:
```bash
export UV_CACHE_DIR=/root/.uv_cache
export UV_PROJECT_ENVIRONMENT=/root/venv
```

---

## Cost Optimization

| GPU | VRAM | $/hr | Speed | $/epoch |
|-----|------|------|-------|---------|
| RTX A4500 | 20GB | $0.25 | ~460 ips | ~$0.18 |
| RTX 4090 | 24GB | $0.69 | ~800 ips | ~$0.21 |
| A100 | 80GB | $1.89 | ~1500 ips | ~$0.31 |

**Best value**: 3x RTX A4500 ($0.75/hr total) - good balance of cost and speed.

---

## Training Timeline

With 3x RTX A4500:
- **Per epoch**: ~44 minutes (1.2M images @ 460 ips)
- **Pseudo-ID refresh**: ~10 minutes (every 2 epochs)
- **Total 75 epochs**: ~48-50 hours
- **Estimated cost**: ~$37-40

---

## Files Reference

```
/workspace/
├── images.npy              # Dataset on network volume
└── face-reco/
    ├── main.py             # Entry point
    ├── conf/config.yaml    # All hyperparameters
    ├── src/
    │   ├── commands/train.py
    │   ├── pseudo.py       # Pseudo-ID mining
    │   └── model/          # MoCo architecture
    └── outputs/            # Hydra output dirs
        └── YYYY-MM-DD/HH-MM-SS/
            └── checkpoints/

/dev/shm/
└── images.npy              # RAM copy for speed
```
