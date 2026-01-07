"""Reason code taxonomy.

These codes should align with `policy_contract.yaml` (reason_codes section).
They are emitted by guardrails and inference routing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasonCodes:
    # Guardrail / routing
    NO_FACE: str = "GR-001"
    LOW_FACE_CONF: str = "GR-002"
    MULTI_FACE_POLICY: str = "GR-003"
    FACE_TOO_SMALL: str = "GR-004"

    # Quality gates
    TOO_BLURRY: str = "QG-001"
    TOO_COMPRESSED: str = "QG-002"
    TOO_LOW_RES: str = "QG-003"

    # Input policy / integrity
    UNSUPPORTED_FORMAT: str = "IN-001"
    CORRUPT_FILE: str = "IN-002"
    TOO_LARGE: str = "IN-003"
    ALPHA_POLICY_REJECTED: str = "IN-004"

    # Model behavior
    ABSTAIN_LOW_CONF: str = "MD-001"
    ABSTAIN_OOD: str = "MD-002"

    # System
    PIPELINE_INVARIANT_FAIL: str = "SYS-001"


CODES = ReasonCodes()
