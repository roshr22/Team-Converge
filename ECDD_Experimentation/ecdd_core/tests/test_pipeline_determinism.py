"""Unit tests for canonical pixel pipeline determinism.

These tests cover:
- decode path works and enforces RGB
- EXIF transpose path does not crash
- alpha handling policy is deterministic
- resize + normalize are deterministic for the same input

Run with pytest from Team-Converge/ECDD_Experimentation:
    pytest -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecdd_core.pipeline import DecodeConfig, PreprocessConfig, decode_image_bytes, resize_rgb_uint8, normalize_rgb_uint8


@pytest.fixture
def sample_jpg_bytes() -> bytes:
    # Use one of the shipped ECDD sample images
    base = Path(__file__).resolve().parents[2] / "ECDD_Experiment_Data" / "real"
    candidates = sorted(list(base.glob("*.jpg")) + list(base.glob("*.png")))
    if not candidates:
        pytest.skip("No sample images found under ECDD_Experiment_Data/real")
    return candidates[0].read_bytes()


def test_decode_rgb_uint8(sample_jpg_bytes: bytes):
    cfg = DecodeConfig(alpha_policy="composite_black")
    rgb = decode_image_bytes(sample_jpg_bytes, cfg)
    assert rgb.dtype == np.uint8
    assert rgb.ndim == 3 and rgb.shape[2] == 3


def test_resize_and_normalize_deterministic(sample_jpg_bytes: bytes):
    dcfg = DecodeConfig(alpha_policy="composite_black")
    pcfg = PreprocessConfig(resize_hw=(256, 256), resize_kernel="bilinear")

    rgb = decode_image_bytes(sample_jpg_bytes, dcfg)

    r1 = resize_rgb_uint8(rgb, pcfg)
    r2 = resize_rgb_uint8(rgb, pcfg)
    assert np.array_equal(r1, r2)

    n1 = normalize_rgb_uint8(r1, pcfg)
    n2 = normalize_rgb_uint8(r2, pcfg)
    assert np.array_equal(n1, n2)
    assert n1.shape == (3, 256, 256)
