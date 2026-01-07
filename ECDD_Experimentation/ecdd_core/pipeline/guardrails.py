"""Guardrails and routing policy.

Implements Phase 2 guardrails:
- No-face abstain
- face confidence threshold
- min face size
- blur metric gate
- compression proxy gate (best-effort)

Outputs explicit reason codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .reason_codes import CODES
from .face import FaceDetectorConfig, detect_faces, select_faces, crop_faces


@dataclass(frozen=True)
class QualityGateConfig:
    # Blur gate: variance of Laplacian threshold
    blur_laplacian_var_threshold: float = 50.0

    # Minimum resolution (after crop) gate
    min_face_hw: int = 64

    # Compression proxy (heuristic). If enabled, reject/abstain when estimated quality is too low.
    # This is a placeholder and can be improved using JPEG quant tables.
    enable_compression_proxy: bool = True
    min_jpeg_quality_proxy: float = 0.2  # 0..1 heuristic


@dataclass
class GuardrailResult:
    ok: bool
    reason_codes: List[str]
    faces: List[np.ndarray]
    face_boxes: List[Tuple[int, int, int, int]]
    face_confidences: List[float]


def variance_of_laplacian(rgb_uint8: np.ndarray) -> float:
    """Simple blur metric: variance of Laplacian on grayscale.

    Implemented without OpenCV to avoid dependency; uses finite differences.
    """
    # grayscale
    gray = (0.2989 * rgb_uint8[..., 0] + 0.5870 * rgb_uint8[..., 1] + 0.1140 * rgb_uint8[..., 2]).astype(np.float32)

    # Approx Laplacian via second derivatives
    # Pad edges
    g = np.pad(gray, 1, mode="edge")
    lap = (
        -4 * g[1:-1, 1:-1]
        + g[0:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, 0:-2]
        + g[1:-1, 2:]
    )
    return float(np.var(lap))


def jpeg_quality_proxy(rgb_uint8: np.ndarray) -> float:
    """Heuristic compression proxy in [0,1].

    This is NOT a true JPEG quality estimator.
    It approximates "blockiness/high-frequency loss" via edge energy.

    Higher is better.
    """
    gray = (0.2989 * rgb_uint8[..., 0] + 0.5870 * rgb_uint8[..., 1] + 0.1140 * rgb_uint8[..., 2]).astype(np.float32) / 255.0
    # simple gradient magnitude
    dx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    dy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    edge_energy = float(dx + dy)
    # map to [0,1] roughly
    return float(np.clip(edge_energy * 4.0, 0.0, 1.0))


def apply_guardrails(rgb_uint8: np.ndarray, face_cfg: FaceDetectorConfig, q_cfg: QualityGateConfig) -> GuardrailResult:
    reason_codes: List[str] = []

    det = detect_faces(rgb_uint8, face_cfg)
    det = select_faces(det, face_cfg)

    if not det.boxes:
        return GuardrailResult(
            ok=False,
            reason_codes=[CODES.NO_FACE],
            faces=[],
            face_boxes=[],
            face_confidences=[],
        )

    # Confidence gate
    if det.confidences and max(det.confidences) < face_cfg.min_confidence:
        reason_codes.append(CODES.LOW_FACE_CONF)
        return GuardrailResult(False, reason_codes, [], det.boxes, det.confidences)

    # Crop faces
    crops = crop_faces(rgb_uint8, det)

    # Min face size gate
    filtered_faces: List[np.ndarray] = []
    filtered_boxes: List[Tuple[int, int, int, int]] = []
    filtered_confs: List[float] = []
    for crop, box, conf in zip(crops, det.boxes, det.confidences):
        if crop.shape[0] < q_cfg.min_face_hw or crop.shape[1] < q_cfg.min_face_hw:
            continue
        filtered_faces.append(crop)
        filtered_boxes.append(box)
        filtered_confs.append(conf)

    if not filtered_faces:
        return GuardrailResult(
            ok=False,
            reason_codes=[CODES.FACE_TOO_SMALL],
            faces=[],
            face_boxes=det.boxes,
            face_confidences=det.confidences,
        )

    # Blur gate (if any face too blurry, we abstain)
    # Alternative policies exist; spec requires deterministic behavior.
    blur_vals = [variance_of_laplacian(f) for f in filtered_faces]
    if min(blur_vals) < q_cfg.blur_laplacian_var_threshold:
        return GuardrailResult(
            ok=False,
            reason_codes=[CODES.TOO_BLURRY],
            faces=[],
            face_boxes=filtered_boxes,
            face_confidences=filtered_confs,
        )

    # Compression proxy
    if q_cfg.enable_compression_proxy:
        q = min(jpeg_quality_proxy(f) for f in filtered_faces)
        if q < q_cfg.min_jpeg_quality_proxy:
            return GuardrailResult(
                ok=False,
                reason_codes=[CODES.TOO_COMPRESSED],
                faces=[],
                face_boxes=filtered_boxes,
                face_confidences=filtered_confs,
            )

    return GuardrailResult(
        ok=True,
        reason_codes=[],
        faces=filtered_faces,
        face_boxes=filtered_boxes,
        face_confidences=filtered_confs,
    )
