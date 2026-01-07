"""Pipeline contracts and stage artifacts (S0–S8).

Based on `ECDD_Paper_DR_3_Experimentation.md` golden set hashing requirements.
We model the stage artifacts and provide a consistent serialization shape.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StageHashes:
    """Golden stage hashes and summaries for one image."""

    # S0
    s0_client_bytes_sha256: Optional[str] = None
    s0_server_bytes_sha256: Optional[str] = None

    # S1
    s1_decoded_rgb_sha256: Optional[str] = None

    # S2
    s2_face_boxes: Optional[List[Tuple[int, int, int, int]]] = None
    s2_face_crops_sha256: Optional[List[str]] = None

    # S3
    s3_resized_sha256: Optional[str] = None

    # S4
    s4_normalized_sha256: Optional[str] = None

    # S5
    s5_patch_map_shape: Optional[Tuple[int, int]] = None
    s5_patch_map_stats: Optional[Dict[str, float]] = None
    s5_patch_map_checksum: Optional[str] = None

    # S6
    s6_pooled_logit: Optional[float] = None

    # S7
    s7_calibrated_logit: Optional[float] = None
    s7_calibrated_prob: Optional[float] = None

    # S8
    s8_decision_label: Optional[int] = None
    s8_reason_codes: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceDecision:
    """Final decision emitted by inference."""

    p_fake: float
    decision: int
    abstained: bool
    reason_codes: List[str]
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
