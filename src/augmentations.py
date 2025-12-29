"""Data augmentation pipelines and custom transforms."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image

from src.data import PILImage


class RandomJPEGCompression:
    """Randomly JPEG-compress a PIL image."""

    def __init__(self, p: float, quality_min: int, quality_max: int) -> None:
        self.p = float(p)
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)

    def __call__(self, img: PILImage) -> PILImage:
        if np.random.rand() >= self.p:
            return img

        q = int(np.random.randint(self.quality_min, self.quality_max + 1))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=False)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        out.load()
        return out


class RandomISONoise:
    """Add pixel-wise Gaussian noise to a tensor image."""

    def __init__(self, p: float, sigma_min: float, sigma_max: float) -> None:
        self.p = float(p)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self.p:
            return x
        sigma = float(np.random.uniform(self.sigma_min, self.sigma_max))
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp_(0.0, 1.0)


def build_view_transform(view_cfg: DictConfig, *, input_size: tuple[int, int]) -> T.Compose:
    """Build one view augmentation pipeline from config."""
    h, w = int(input_size[0]), int(input_size[1])

    ops: list[Any] = []

    # Random resized crop
    rrc = view_cfg.get("random_resized_crop", {})
    scale = tuple(rrc.get("scale", [0.2, 1.0]))
    ratio = tuple(rrc.get("ratio", [0.75, 1.33]))
    ops.append(T.RandomResizedCrop(size=(h, w), scale=scale, ratio=ratio))

    # Horizontal flip
    flip_p = float(view_cfg.get("horizontal_flip_p", 0.5))
    ops.append(T.RandomHorizontalFlip(p=flip_p))

    # RandAugment
    ra_cfg = view_cfg.get("randaugment", {})
    if ra_cfg:
        num_ops = int(ra_cfg.get("num_ops", 4))
        magnitude = int(ra_cfg.get("magnitude", 16))
        ops.append(T.RandAugment(num_ops=num_ops, magnitude=magnitude))

    # Color jitter
    cj_cfg = view_cfg.get("color_jitter", {})
    if cj_cfg and float(cj_cfg.get("p", 0.0)) > 0.0:
        p = float(cj_cfg.get("p", 0.8))
        cj = T.ColorJitter(
            brightness=float(cj_cfg.get("brightness", 0.4)),
            contrast=float(cj_cfg.get("contrast", 0.4)),
            saturation=float(cj_cfg.get("saturation", 0.4)),
            hue=float(cj_cfg.get("hue", 0.1)),
        )
        ops.append(T.RandomApply([cj], p=p))

    # Grayscale
    gs_p = float(view_cfg.get("grayscale_p", 0.0))
    if gs_p > 0.0:
        ops.append(T.RandomGrayscale(p=gs_p))

    # Gaussian blur
    blur_cfg = view_cfg.get("gaussian_blur", {})
    if blur_cfg and float(blur_cfg.get("p", 0.0)) > 0.0:
        p = float(blur_cfg.get("p", 0.5))
        sigma = tuple(blur_cfg.get("sigma", [0.1, 2.0]))
        ops.append(T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=sigma)], p=p))

    # JPEG compression
    jpeg_cfg = view_cfg.get("jpeg_compression", {})
    if jpeg_cfg and float(jpeg_cfg.get("p", 0.0)) > 0.0:
        p = float(jpeg_cfg.get("p", 0.3))
        qmin, qmax = jpeg_cfg.get("quality", [30, 95])
        ops.append(RandomJPEGCompression(p=p, quality_min=int(qmin), quality_max=int(qmax)))

    # Convert to tensor
    ops.append(T.ToTensor())

    # ISO noise
    iso_cfg = view_cfg.get("iso_noise", {})
    if iso_cfg and float(iso_cfg.get("p", 0.0)) > 0.0:
        p = float(iso_cfg.get("p", 0.2))
        smin, smax = iso_cfg.get("sigma", [0.0, 0.08])
        ops.append(RandomISONoise(p=p, sigma_min=float(smin), sigma_max=float(smax)))

    # Cutout
    cut_cfg = view_cfg.get("cutout", {})
    if cut_cfg and float(cut_cfg.get("p", 0.0)) > 0.0:
        p = float(cut_cfg.get("p", 0.2))
        holes = int(cut_cfg.get("holes", 1))
        size_ratio = tuple(cut_cfg.get("size_ratio", [0.10, 0.30]))
        for _ in range(max(1, holes)):
            ops.append(T.RandomErasing(p=p, scale=size_ratio, ratio=(0.3, 3.3), value="random"))

    # Normalize
    ops.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return T.Compose(ops)


def build_embed_transform(input_size: tuple[int, int]) -> T.Compose:
    """Deterministic preprocessing for embedding export."""
    h, w = int(input_size[0]), int(input_size[1])
    return T.Compose(
        [
            T.Resize(size=(h, w), antialias=True),
            T.CenterCrop(size=(h, w)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def build_album_transform(view_cfg: DictConfig, *, input_size: tuple[int, int]) -> A.Compose:
    """Build albumentations pipeline (2-3x faster than torchvision).

    Input: numpy array (H, W, 3) uint8
    Output: torch.Tensor (3, H, W) float32, normalized to [-1, 1]
    """
    h, w = int(input_size[0]), int(input_size[1])
    ops: list[A.BasicTransform] = []

    # Random resized crop
    rrc = view_cfg.get("random_resized_crop", {})
    scale = tuple(rrc.get("scale", [0.2, 1.0]))
    ratio = tuple(rrc.get("ratio", [0.75, 1.33]))
    ops.append(A.RandomResizedCrop(size=(h, w), scale=scale, ratio=ratio))

    # Horizontal flip
    flip_p = float(view_cfg.get("horizontal_flip_p", 0.5))
    ops.append(A.HorizontalFlip(p=flip_p))

    # Color jitter
    cj_cfg = view_cfg.get("color_jitter", {})
    if cj_cfg and float(cj_cfg.get("p", 0.0)) > 0.0:
        p = float(cj_cfg.get("p", 0.8))
        ops.append(A.ColorJitter(
            brightness=(1 - float(cj_cfg.get("brightness", 0.4)), 1 + float(cj_cfg.get("brightness", 0.4))),
            contrast=(1 - float(cj_cfg.get("contrast", 0.4)), 1 + float(cj_cfg.get("contrast", 0.4))),
            saturation=(1 - float(cj_cfg.get("saturation", 0.4)), 1 + float(cj_cfg.get("saturation", 0.4))),
            hue=(-float(cj_cfg.get("hue", 0.1)), float(cj_cfg.get("hue", 0.1))),
            p=p,
        ))

    # Grayscale
    gs_p = float(view_cfg.get("grayscale_p", 0.0))
    if gs_p > 0.0:
        ops.append(A.ToGray(p=gs_p))

    # Gaussian blur
    blur_cfg = view_cfg.get("gaussian_blur", {})
    if blur_cfg and float(blur_cfg.get("p", 0.0)) > 0.0:
        p = float(blur_cfg.get("p", 0.5))
        sigma = tuple(blur_cfg.get("sigma", [0.1, 2.0]))
        ops.append(A.GaussianBlur(blur_limit=(7, 9), sigma_limit=sigma, p=p))

    # JPEG compression
    jpeg_cfg = view_cfg.get("jpeg_compression", {})
    if jpeg_cfg and float(jpeg_cfg.get("p", 0.0)) > 0.0:
        p = float(jpeg_cfg.get("p", 0.3))
        qmin, qmax = jpeg_cfg.get("quality", [30, 95])
        ops.append(A.ImageCompression(quality_range=(int(qmin), int(qmax)), p=p))

    # ISO noise (Gaussian noise)
    iso_cfg = view_cfg.get("iso_noise", {})
    if iso_cfg and float(iso_cfg.get("p", 0.0)) > 0.0:
        p = float(iso_cfg.get("p", 0.2))
        smin, smax = iso_cfg.get("sigma", [0.0, 0.08])
        ops.append(A.GaussNoise(std_range=(float(smin), float(smax)), p=p))

    # Cutout (CoarseDropout)
    cut_cfg = view_cfg.get("cutout", {})
    if cut_cfg and float(cut_cfg.get("p", 0.0)) > 0.0:
        p = float(cut_cfg.get("p", 0.2))
        holes = int(cut_cfg.get("holes", 1))
        size_ratio = tuple(cut_cfg.get("size_ratio", [0.10, 0.30]))
        ops.append(A.CoarseDropout(
            num_holes_range=(1, holes),
            hole_height_range=(int(h * size_ratio[0]), int(h * size_ratio[1])),
            hole_width_range=(int(w * size_ratio[0]), int(w * size_ratio[1])),
            p=p,
        ))

    # Normalize and convert to tensor
    ops.append(A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    ops.append(ToTensorV2())

    return A.Compose(ops)
