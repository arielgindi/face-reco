"""MoCo-style self-supervised learning for face embeddings.

This module contains:
- A projection MLP head (MoCo v2 style)
- MoCo wrapper with a momentum encoder and a queue of negative keys
- MarginNCE loss (additive margin on the positive logit)

Reference: https://arxiv.org/abs/1911.05722 (MoCo)
Reference: https://ar5iv.org/pdf/2211.07371 (USynthFace - MarginNCE)
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.backbone import IResNet, iresnet50, l2_normalize

logger = logging.getLogger(__name__)


class MLPProjector(nn.Module):
    """MoCo v2-style projection head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class MoCoConfig:
    """Configuration for the MoCo model and its loss."""

    embedding_dim: int = 512
    projection_dim: int = 128
    projection_hidden_dim: int = 2048
    queue_size: int = 32768
    momentum: float = 0.999
    temperature: float = 0.07
    margin: float = 0.10
    l2_normalize_backbone: bool = True


class MoCo(nn.Module):
    """MoCo with a momentum encoder and a queue.

    Forward expects two augmented views of the same image:
      - im_q: view for query encoder (gradient)
      - im_k: view for key encoder (no gradient)

    Returns:
      - loss
      - stats dict (pos/neg similarities, queue ptr, embedding stats)
    """

    def __init__(
        self,
        backbone: nn.Module,
        cfg: MoCoConfig,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # Query encoder
        self.backbone_q = backbone
        self.projector_q = MLPProjector(
            in_dim=cfg.embedding_dim,
            hidden_dim=cfg.projection_hidden_dim,
            out_dim=cfg.projection_dim,
        )

        # Key encoder (momentum updated) - deep copy
        self.backbone_k = copy.deepcopy(backbone)
        self.projector_k = MLPProjector(
            in_dim=cfg.embedding_dim,
            hidden_dim=cfg.projection_hidden_dim,
            out_dim=cfg.projection_dim,
        )

        # Initialize key encoder to match query encoder
        self._copy_params_q_to_k()

        # Create the queue (dim, K)
        queue = torch.randn(cfg.projection_dim, cfg.queue_size)
        queue = l2_normalize(queue, dim=0)
        self.register_buffer("queue", queue)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Pseudo-ID tracking for negative masking
        # -1 means "no cluster assigned" (treat as normal negative)
        self.register_buffer(
            "queue_cluster_ids",
            torch.full((cfg.queue_size,), -1, dtype=torch.int32),
        )

        if device is not None:
            self.to(device)

    @torch.no_grad()
    def reset_queue(self) -> None:
        """Reset queue to random embeddings (call after pseudo-ID refresh)."""
        queue = torch.randn_like(self.queue)
        queue = l2_normalize(queue, dim=0)
        self.queue.copy_(queue)
        self.queue_ptr.zero_()
        self.queue_cluster_ids.fill_(-1)

    def _copy_params_q_to_k(self) -> None:
        """Initialize momentum encoder to match the query encoder."""
        for param_q, param_k in zip(
            self.backbone_q.parameters(),
            self.backbone_k.parameters(),
            strict=True,
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(
            self.projector_q.parameters(),
            self.projector_k.parameters(),
            strict=True,
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def update_momentum_encoder(self) -> None:
        """Momentum update of the key encoder."""
        m = self.cfg.momentum
        for param_q, param_k in zip(
            self.backbone_q.parameters(),
            self.backbone_k.parameters(),
            strict=True,
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

        for param_q, param_k in zip(
            self.projector_q.parameters(),
            self.projector_k.parameters(),
            strict=True,
        ):
            param_k.data.mul_(m).add_(param_q.data, alpha=1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        keys: torch.Tensor,
        cluster_ids: torch.Tensor | None = None,
    ) -> None:
        """Update queue with the latest batch of key projections."""
        keys = keys.detach()
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        k = self.cfg.queue_size

        # Default cluster IDs to -1 if not provided
        if cluster_ids is None:
            cluster_ids = torch.full(
                (batch_size,), -1, dtype=torch.int32, device=keys.device
            )

        # Handle wraparound
        if ptr + batch_size > k:
            space_left = k - ptr
            self.queue[:, ptr:k] = keys[:space_left].T
            self.queue_cluster_ids[ptr:k] = cluster_ids[:space_left]
            remaining = batch_size - space_left
            self.queue[:, :remaining] = keys[space_left:].T
            self.queue_cluster_ids[:remaining] = cluster_ids[space_left:]
            ptr = remaining
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            self.queue_cluster_ids[ptr : ptr + batch_size] = cluster_ids
            ptr = (ptr + batch_size) % k

        self.queue_ptr[0] = ptr

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce the backbone embedding for retrieval (512-D by default)."""
        emb = self.backbone_q(x)
        if self.cfg.l2_normalize_backbone:
            emb = l2_normalize(emb, dim=1)
        return emb

    def forward(
        self,
        im_q: torch.Tensor,
        im_k: torch.Tensor,
        cluster_ids: torch.Tensor | None = None,
        mask_same_cluster: bool = False,
        mask_topk: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute MarginNCE loss for a batch.

        Args:
            im_q: Query images (B, C, H, W)
            im_k: Key images (B, C, H, W)
            cluster_ids: Optional pseudo-cluster IDs for this batch (B,)
            mask_same_cluster: If True, mask queue negatives with same cluster ID
            mask_topk: Additionally mask top-k most similar negatives (likely false negatives)
        """
        # Query embeddings
        emb_q = self.backbone_q(im_q)
        if self.cfg.l2_normalize_backbone:
            emb_q = l2_normalize(emb_q, dim=1)
        q = self.projector_q(emb_q)
        q = l2_normalize(q, dim=1)

        # Key embeddings (no gradient)
        with torch.no_grad():
            emb_k = self.backbone_k(im_k)
            if self.cfg.l2_normalize_backbone:
                emb_k = l2_normalize(emb_k, dim=1)
            k = self.projector_k(emb_k)
            k = l2_normalize(k, dim=1)

        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(1)

        # Negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", q, self.queue.clone().detach())

        # ===== NEGATIVE MASKING =====
        neg_masked_count = 0
        if (mask_same_cluster or mask_topk > 0) and cluster_ids is not None:
            mask = torch.zeros_like(l_neg, dtype=torch.bool)

            # Mask 1: Same pseudo-cluster in queue
            if mask_same_cluster:
                # Only mask where both query and queue have valid cluster IDs (>= 0)
                valid_query = cluster_ids >= 0  # (B,)
                valid_queue = self.queue_cluster_ids >= 0  # (K,)

                # (B, 1) == (1, K) -> (B, K)
                same_cluster = cluster_ids.unsqueeze(1) == self.queue_cluster_ids.unsqueeze(0)
                # Only apply where both are valid
                same_cluster = same_cluster & valid_query.unsqueeze(1) & valid_queue.unsqueeze(0)
                mask = mask | same_cluster

            # Mask 2: Top-k most similar (likely false negatives beyond cluster)
            if mask_topk > 0:
                _, topk_indices = l_neg.topk(mask_topk, dim=1)  # (B, mask_topk)
                mask.scatter_(1, topk_indices, True)

            # Apply mask by setting to large negative (excluded from softmax)
            neg_masked_count = int(mask.sum().item())
            l_neg = l_neg.masked_fill(mask, float("-inf"))

        # Apply margin to positive logits (MarginNCE)
        if self.cfg.margin > 0:
            l_pos = l_pos - self.cfg.margin

        # Concatenate and scale by temperature
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.cfg.temperature

        # Labels: positive is always index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # Update queue with cluster IDs
        self._dequeue_and_enqueue(k, cluster_ids)

        # Collect stats for logging
        batch_size = im_q.shape[0]
        queue_size = self.cfg.queue_size
        stats: dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "pos_sim": float((l_pos + self.cfg.margin).mean().detach().cpu()),
            "neg_sim": float(l_neg[l_neg > float("-inf")].mean().detach().cpu()) if neg_masked_count < batch_size * queue_size else 0.0,
            "queue_ptr": int(self.queue_ptr.item()),
            "emb_std": float(emb_q.detach().std(dim=0).mean().cpu()),
            "neg_masked_pct": neg_masked_count / (batch_size * queue_size) if queue_size > 0 else 0.0,
        }
        return loss, stats


def build_backbone(backbone_cfg: Mapping[str, Any]) -> IResNet:
    """Build the embedding backbone from a config mapping."""
    name = str(backbone_cfg.get("name", "iresnet50")).lower()
    embedding_dim = int(backbone_cfg.get("embedding_dim", 512))
    bn_eps = float(backbone_cfg.get("bn_eps", 1.0e-5))
    bn_momentum = float(backbone_cfg.get("bn_momentum", 0.1))
    dropout = float(backbone_cfg.get("dropout", 0.0))

    if name != "iresnet50":
        raise ValueError(f"Unsupported backbone: {name!r} (expected 'iresnet50')")

    return iresnet50(
        embedding_dim=embedding_dim,
        dropout=dropout,
        bn_eps=bn_eps,
        bn_momentum=bn_momentum,
    )


def build_moco(cfg: Mapping[str, Any], device: torch.device | None = None) -> MoCo:
    """Build a MoCo model from the overall YAML config dict."""
    model_cfg = cfg.get("model", {})
    backbone_cfg = model_cfg.get("backbone", {})
    backbone = build_backbone(backbone_cfg)

    ssl_cfg = cfg.get("ssl", {})
    margin_cfg = ssl_cfg.get("margin_nce", {}) if isinstance(ssl_cfg, dict) else {}

    margin_enabled = bool(margin_cfg.get("enabled", True))
    margin = float(margin_cfg.get("margin", 0.10)) if margin_enabled else 0.0

    mlp_head_cfg = ssl_cfg.get("mlp_head", {})
    projection_hidden_dim = int(mlp_head_cfg.get("hidden_dim", 2048))
    projection_dim = int(mlp_head_cfg.get("out_dim", 128))

    moco_cfg = MoCoConfig(
        embedding_dim=int(backbone_cfg.get("embedding_dim", 512)),
        projection_dim=projection_dim,
        projection_hidden_dim=projection_hidden_dim,
        queue_size=int(ssl_cfg.get("queue_size", 32768)),
        momentum=float(ssl_cfg.get("momentum_encoder", 0.999)),
        temperature=float(ssl_cfg.get("temperature", 0.07)),
        margin=margin,
        l2_normalize_backbone=bool(backbone_cfg.get("l2_normalize", True)),
    )

    logger.info(
        f"Building MoCo: queue_size={moco_cfg.queue_size}, "
        f"margin={moco_cfg.margin}, temp={moco_cfg.temperature}"
    )

    return MoCo(backbone=backbone, cfg=moco_cfg, device=device)


def backbone_state_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Extract backbone_q weights from a saved checkpoint."""
    model_sd = checkpoint.get("model", checkpoint)
    if not isinstance(model_sd, dict):
        raise ValueError("Checkpoint does not contain a state dict under key 'model'.")

    prefix = "backbone_q."
    out: dict[str, torch.Tensor] = {}
    for k, v in model_sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v

    if not out:
        raise ValueError(
            "Could not find backbone weights in checkpoint. "
            "Expected keys starting with 'backbone_q.'"
        )
    return out
