"""IResNet backbone for face recognition.

Reference: https://arxiv.org/abs/1901.01815 (ArcFace iResNet)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize a tensor along a dimension."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


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
