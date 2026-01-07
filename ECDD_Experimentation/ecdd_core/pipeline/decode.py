"""Authoritative image decoding.

This module enforces a single decode path and implements:
- file integrity checks
- supported format allowlist
- EXIF orientation application
- alpha handling policy
- RGB channel order

It is designed to satisfy Phase 1 experiments E1.2–E1.7.

NOTE: We intentionally avoid OpenCV here to reduce RGB/BGR mismatch risk.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

from .reason_codes import CODES


SupportedFormat = Literal["jpeg", "png", "webp"]


@dataclass(frozen=True)
class DecodeConfig:
    # Allowlist
    allowed_formats: Tuple[SupportedFormat, ...] = ("jpeg", "png", "webp")

    # Alpha handling
    # - "reject": fail if alpha exists
    # - "composite_black": composite over black background
    # - "composite_white": composite over white background
    alpha_policy: Literal["reject", "composite_black", "composite_white"] = "composite_black"

    # Output dtype/range policy
    # - "uint8": uint8 0..255
    output_dtype: Literal["uint8"] = "uint8"


class DecodeError(ValueError):
    def __init__(self, message: str, reason_code: str):
        super().__init__(message)
        self.reason_code = reason_code


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_format(pil_image: Image.Image) -> Optional[SupportedFormat]:
    fmt = (pil_image.format or "").lower()
    if fmt in ("jpeg", "jpg"):
        return "jpeg"
    if fmt == "png":
        return "png"
    if fmt == "webp":
        return "webp"
    return None


def decode_image_bytes(data: bytes, cfg: DecodeConfig) -> np.ndarray:
    """Decode bytes into an RGB uint8 array (H, W, 3).

    Raises DecodeError with a reason code on failure.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise DecodeError(f"Corrupt/invalid image: {e}", CODES.CORRUPT_FILE)

    fmt = detect_format(img)
    if fmt is None or fmt not in cfg.allowed_formats:
        raise DecodeError(f"Unsupported format: {img.format}", CODES.UNSUPPORTED_FORMAT)

    # EXIF orientation
    img = ImageOps.exif_transpose(img)

    # Alpha handling
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        if cfg.alpha_policy == "reject":
            raise DecodeError("Alpha channel rejected by policy", CODES.ALPHA_POLICY_REJECTED)
        # composite
        bg = (0, 0, 0) if cfg.alpha_policy == "composite_black" else (255, 255, 255)
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, bg + (255,))
        img = Image.alpha_composite(background, img).convert("RGB")
    else:
        img = img.convert("RGB")

    arr = np.array(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise DecodeError("Decoded image is not RGB", CODES.PIPELINE_INVARIANT_FAIL)

    return arr
