"""Pipeline package: decode + preprocess + guardrails.

This package is the authoritative implementation of the pixel contract.
"""

from .decode import DecodeConfig, DecodeError, decode_image_bytes, sha256_bytes
from .preprocess import PreprocessConfig, resize_rgb_uint8, normalize_rgb_uint8, sha256_ndarray
from .reason_codes import CODES, ReasonCodes
from .contracts import StageHashes, InferenceDecision
from .face import FaceDetectorConfig, FaceDetection, detect_faces, select_faces, crop_faces
from .guardrails import QualityGateConfig, GuardrailResult, apply_guardrails

__all__ = [
    "DecodeConfig",
    "DecodeError",
    "decode_image_bytes",
    "sha256_bytes",
    "PreprocessConfig",
    "resize_rgb_uint8",
    "normalize_rgb_uint8",
    "sha256_ndarray",
    "CODES",
    "ReasonCodes",
    "StageHashes",
    "InferenceDecision",
    "FaceDetectorConfig",
    "FaceDetection",
    "detect_faces",
    "select_faces",
    "crop_faces",
    "QualityGateConfig",
    "GuardrailResult",
    "apply_guardrails",
]
