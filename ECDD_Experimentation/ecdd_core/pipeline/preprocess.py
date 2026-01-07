"""Deterministic preprocessing (resize + normalize) for ECDD.

This module is part of the preprocessing equivalence contract and is used
for both training and inference.

Phase alignment:
- E1.8 fixed interpolation kernel test
- E1.9 normalization constants test
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    resize_hw: Tuple[int, int] = (256, 256)
    resize_kernel: Literal["bilinear", "bicubic"] = "bilinear"

    # normalization mean/std in RGB order (ImageNet default in many pipelines)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # output dtype
    output_dtype: Literal["float32"] = "float32"


def _pil_resample(kernel: str) -> int:
    if kernel == "bilinear":
        return Image.BILINEAR
    if kernel == "bicubic":
        return Image.BICUBIC
    raise ValueError(f"Unsupported resize kernel: {kernel}")


def resize_rgb_uint8(rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Resize RGB uint8 (H,W,3) deterministically."""
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize(cfg.resize_hw[::-1], resample=_pil_resample(cfg.resize_kernel))
    out = np.array(img, dtype=np.uint8)
    return out


def normalize_rgb_uint8(resized_rgb: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Normalize to float32 CHW tensor."""
    x = resized_rgb.astype(np.float32) / 255.0
    mean = np.array(cfg.mean, dtype=np.float32)
    std = np.array(cfg.std, dtype=np.float32)
    x = (x - mean) / std
    # HWC -> CHW
    x = np.transpose(x, (2, 0, 1))
    return x.astype(np.float32)


def sha256_ndarray(arr: np.ndarray) -> str:
    # Deterministic byte representation
    return hashlib.sha256(arr.tobytes()).hexdigest()
