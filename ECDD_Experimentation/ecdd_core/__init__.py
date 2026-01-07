"""ECDD Core: canonical training+inference pipeline, calibration, export/parity, monitoring, and governance.

This package is the single source of truth for:
- pixel pipeline equivalence (decode → EXIF → alpha/gamma → resize → normalize)
- guardrails and routing
- calibration and operating point thresholds
- quantization + parity gates
- monitoring schema (privacy-preserving)
- dataset governance and lineage enforcement

NOTE: Per project constraint, this codebase lives entirely under ECDD_Experimentation and
must not modify deepfake-patch-audit.
"""
