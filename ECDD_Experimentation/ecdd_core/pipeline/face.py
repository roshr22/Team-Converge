"""Face detection wrapper and multi-face policy.

Phase 2 requirements (from experimentation spec):
- version pinning / auditable configuration
- deterministic outputs as much as possible
- multi-face policy (max vs largest-face)
- minimum face size threshold

Implementation note:
We avoid adding heavy dependencies by default. This wrapper supports:
1) Optional `mediapipe` face detection if installed.
2) Fallback stub that returns no faces (useful for unit tests and wiring).

Teams can later swap in a pinned detector (e.g., BlazeFace, RetinaFace) as long
as the API contract stays identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


@dataclass(frozen=True)
class FaceDetectorConfig:
    backend: Literal["mediapipe", "stub"] = "stub"
    min_confidence: float = 0.5

    # Multi-face policy:
    # - "largest": choose largest face
    # - "max": evaluate all faces and take max p_fake later
    multi_face_policy: Literal["largest", "max"] = "max"

    # Minimum crop size (pixels) below which we abstain
    min_face_size: int = 64


@dataclass
class FaceDetection:
    boxes: List[BBox]
    confidences: List[float]


def _clip_box(box: BBox, w: int, h: int) -> BBox:
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


class FaceBackendUnavailable(RuntimeError):
    pass


def detect_faces(rgb_uint8: np.ndarray, cfg: FaceDetectorConfig) -> FaceDetection:
    """Detect faces in an RGB uint8 image.

    Returns boxes in pixel coordinates.

    IMPORTANT: If cfg.backend == 'mediapipe' and mediapipe is not installed,
    we fail loudly. Silent fallback makes audits non-reproducible.
    """
    h, w, _ = rgb_uint8.shape

    if cfg.backend == "stub":
        return FaceDetection(boxes=[], confidences=[])

    if cfg.backend == "mediapipe":
        try:
            import mediapipe as mp
        except Exception as e:
            raise FaceBackendUnavailable(
                "mediapipe is required for FaceDetectorConfig(backend='mediapipe'). "
                "Install with: pip install mediapipe"
            ) from e

        mp_face = mp.solutions.face_detection
        with mp_face.FaceDetection(model_selection=0, min_detection_confidence=cfg.min_confidence) as face_det:
            # mediapipe expects RGB
            results = face_det.process(rgb_uint8)

        boxes: List[BBox] = []
        confs: List[float] = []
        if results and results.detections:
            for det in results.detections:
                score = float(det.score[0]) if det.score else 0.0
                b = det.location_data.relative_bounding_box
                x1 = int(b.xmin * w)
                y1 = int(b.ymin * h)
                x2 = int((b.xmin + b.width) * w)
                y2 = int((b.ymin + b.height) * h)
                box = _clip_box((x1, y1, x2, y2), w, h)
                boxes.append(box)
                confs.append(score)

        return FaceDetection(boxes=boxes, confidences=confs)

    raise ValueError(f"Unknown backend: {cfg.backend}")


def select_faces(detection: FaceDetection, cfg: FaceDetectorConfig) -> FaceDetection:
    """Apply multi-face policy to detected faces."""
    if not detection.boxes:
        return detection

    if cfg.multi_face_policy == "max":
        return detection

    if cfg.multi_face_policy == "largest":
        areas = []
        for (x1, y1, x2, y2) in detection.boxes:
            areas.append(max(0, x2 - x1) * max(0, y2 - y1))
        idx = int(np.argmax(np.array(areas)))
        return FaceDetection(boxes=[detection.boxes[idx]], confidences=[detection.confidences[idx]])

    raise ValueError(f"Unknown multi_face_policy: {cfg.multi_face_policy}")


def crop_faces(rgb_uint8: np.ndarray, detection: FaceDetection) -> List[np.ndarray]:
    """Crop face regions as RGB uint8 arrays."""
    crops: List[np.ndarray] = []
    for (x1, y1, x2, y2) in detection.boxes:
        crop = rgb_uint8[y1:y2, x1:x2, :]
        crops.append(crop)
    return crops
