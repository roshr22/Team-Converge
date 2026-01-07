"""Deterministic transform suite for Phase 6 (E6.3).

Applies:
- JPEG re-encode Q in {95, 75, 50, 30}
- resize down-up chains
- blur severities
- screenshot-like resampling (nearest downscale then bilinear upscale)

All transforms are deterministic given parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Tuple

import io
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class TransformSpec:
    name: str
    params: dict


def jpeg_reencode(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def resize_chain(img: Image.Image, down: int, up: int, down_kernel=Image.BILINEAR, up_kernel=Image.BILINEAR) -> Image.Image:
    img2 = img.resize((down, down), resample=down_kernel)
    return img2.resize((up, up), resample=up_kernel)


def blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def screenshot_like(img: Image.Image, down: int, up: int) -> Image.Image:
    # nearest downscale simulates screenshot / pixelated resampling
    img2 = img.resize((down, down), resample=Image.NEAREST)
    return img2.resize((up, up), resample=Image.BILINEAR)


def default_transform_suite() -> List[TransformSpec]:
    suite: List[TransformSpec] = []

    for q in [95, 75, 50, 30]:
        suite.append(TransformSpec(name="jpeg", params={"quality": q}))

    # Resize chains
    for down in [128, 96]:
        suite.append(TransformSpec(name="resize_chain", params={"down": down, "up": 256}))

    # Blur
    for r in [2.0, 4.0, 8.0]:
        suite.append(TransformSpec(name="blur", params={"radius": r}))

    # Screenshot-like
    for down in [128, 96]:
        suite.append(TransformSpec(name="screenshot", params={"down": down, "up": 256}))

    return suite


def apply_transform(img: Image.Image, spec: TransformSpec) -> Image.Image:
    if spec.name == "jpeg":
        return jpeg_reencode(img, quality=int(spec.params["quality"]))
    if spec.name == "resize_chain":
        return resize_chain(img, down=int(spec.params["down"]), up=int(spec.params["up"]))
    if spec.name == "blur":
        return blur(img, radius=float(spec.params["radius"]))
    if spec.name == "screenshot":
        return screenshot_like(img, down=int(spec.params["down"]), up=int(spec.params["up"]))
    raise ValueError(f"Unknown transform: {spec.name}")
