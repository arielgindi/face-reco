"""MoCo-style self-supervised learning for face embeddings.

This module contains:
- iResNet-50 backbone (proper face recognition architecture with PReLU, BN placement)
- A projection MLP head (MoCo v2 style)
- MoCo wrapper with a momentum encoder and a queue of negative keys
- MarginNCE loss (additive margin on the positive logit)

The encoder backbone outputs 512-D embeddings intended for downstream retrieval.
The projection head outputs a lower-D representation used only for the SSL loss.

Reference: https://arxiv.org/abs/1901.01815 (ArcFace iResNet)
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

logger = logging.getLogger(__name__)


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize a tensor along a dimension."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


# =============================================================================
# iResNet-50 Backbone (Face Recognition Architecture)
# =============================================================================


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class IBasicBlock(nn.Module):
    """Basic Block for iResNet (PreAct-like structure with PReLU).

    Key differences from standard ResNet BasicBlock:
    - Uses PReLU instead of ReLU
    - BN-before-conv structure (pre-activation style)
    - BN after second conv before addition
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=bn_eps, momentum=bn_momentum)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class IResNet(nn.Module):
    """iResNet backbone for face recognition (112x112 input).

    This is the standard backbone used in ArcFace, CosFace, and related
    face recognition methods. Key characteristics:
    - 3x3 conv1 with stride=1 (preserves 112x112 initially)
    - No maxpool layer
    - Stride-2 at start of each residual stage
    - Final spatial size: 7x7 for 112x112 input
    - FC embedding head with BN "neck"

    Reference: https://arxiv.org/abs/1901.01815
    """

    def __init__(
        self,
        block: type[IBasicBlock],
        layers: list[int],
        embedding_dim: int = 512,
        dropout: float = 0.0,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        # Stem: 3x3 stride 1 (preserves 112x112)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=bn_eps, momentum=bn_momentum)
        self.prelu = nn.PReLU(64)

        # Layers: Stride 2 at start of each layer -> 112 -> 56 -> 28 -> 14 -> 7
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=bn_eps, momentum=bn_momentum)
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else nn.Identity()

        # Face Recognition Embedding Head: Flatten -> FC -> BN
        # For 112x112 input: final spatial is 7x7
        self.fc = nn.Linear(512 * block.expansion * 7 * 7, embedding_dim)
        self.features = nn.BatchNorm1d(embedding_dim, eps=bn_eps, momentum=bn_momentum)

        self._init_weights()

    def _make_layer(
        self,
        block: type[IBasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    eps=self.bn_eps,
                    momentum=self.bn_momentum,
                ),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                bn_eps=self.bn_eps,
                bn_momentum=self.bn_momentum,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    bn_eps=self.bn_eps,
                    bn_momentum=self.bn_momentum,
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize weights following best practices for face recognition."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)

        return x


def iresnet50(
    embedding_dim: int = 512,
    dropout: float = 0.0,
    bn_eps: float = 1e-5,
    bn_momentum: float = 0.1,
) -> IResNet:
    """Create iResNet-50 backbone for face recognition."""
    return IResNet(
        IBasicBlock,
        [3, 4, 14, 3],
        embedding_dim=embedding_dim,
        dropout=dropout,
        bn_eps=bn_eps,
        bn_momentum=bn_momentum,
    )


# =============================================================================
# Projection Head and MoCo Components
# =============================================================================


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

        if device is not None:
            self.to(device)

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
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Update queue with the latest batch of key projections.

        Handles wraparound correctly for any batch size.
        """
        keys = keys.detach()
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        k = self.cfg.queue_size

        # Handle wraparound
        if ptr + batch_size > k:
            # Split into two parts
            space_left = k - ptr
            self.queue[:, ptr:k] = keys[:space_left].T
            remaining = batch_size - space_left
            self.queue[:, :remaining] = keys[space_left:].T
            ptr = remaining
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % k

        self.queue_ptr[0] = ptr

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce the backbone embedding for retrieval (512-D by default).

        The embedding is L2-normalized if configured.
        """
        emb = self.backbone_q(x)
        if self.cfg.l2_normalize_backbone:
            emb = l2_normalize(emb, dim=1)
        return emb

    def forward(
        self,
        im_q: torch.Tensor,
        im_k: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute MarginNCE loss for a batch."""
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

        # Apply margin to positive logits (MarginNCE)
        if self.cfg.margin > 0:
            l_pos = l_pos - self.cfg.margin

        # Concatenate and scale by temperature
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.cfg.temperature

        # Labels: positive is always index 0
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        # Collect stats for logging
        stats: dict[str, Any] = {
            "loss": float(loss.detach().cpu()),
            "pos_sim": float((l_pos + self.cfg.margin).mean().detach().cpu()),
            "neg_sim": float(l_neg.mean().detach().cpu()),
            "queue_ptr": int(self.queue_ptr.item()),
            "emb_std": float(emb_q.detach().std(dim=0).mean().cpu()),
        }
        return loss, stats


# =============================================================================
# Factory Functions
# =============================================================================


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

    # Get projection dims from ssl config or use defaults
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
    """Extract backbone_q weights from a saved checkpoint.

    Expected checkpoint format:
      {"model": state_dict, ...}
    """
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
