"""Generate golden stage hashes S0–S4 (initial) for the golden manifest.

This is the backbone for Phase 1 gating and CI smoke tests.

We start with S0–S4 (bytes → decoded RGB → resized → normalized).
S5–S8 require model integration (patch logits, pooling, calibration, decision).

Outputs a JSONL or JSON file mapping golden item id -> StageHashes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from ..pipeline import (
    DecodeConfig,
    PreprocessConfig,
    FaceDetectorConfig,
    decode_image_bytes,
    sha256_bytes,
    sha256_ndarray,
    detect_faces,
    select_faces,
    crop_faces,
    resize_rgb_uint8,
    normalize_rgb_uint8,
)
from ..pipeline.contracts import StageHashes
from .golden_sets import default_golden_manifest


def compute_stage_hashes_for_file(
    path: Path,
    decode_cfg: DecodeConfig,
    pp_cfg: PreprocessConfig,
    face_cfg: FaceDetectorConfig,
) -> StageHashes:
    data = path.read_bytes()
    s0 = sha256_bytes(data)

    rgb = decode_image_bytes(data, decode_cfg)
    s1 = sha256_ndarray(rgb)

    # S2: face boxes + crop hashes (best-effort; may be empty if detector is stub)
    det = select_faces(detect_faces(rgb, face_cfg), face_cfg)
    crops = crop_faces(rgb, det)
    crop_hashes = [sha256_ndarray(c) for c in crops]

    resized = resize_rgb_uint8(rgb, pp_cfg)
    s3 = sha256_ndarray(resized)

    normalized = normalize_rgb_uint8(resized, pp_cfg)
    s4 = sha256_ndarray(normalized)

    return StageHashes(
        s0_client_bytes_sha256=s0,
        s0_server_bytes_sha256=None,
        s1_decoded_rgb_sha256=s1,
        s2_face_boxes=det.boxes if det.boxes else [],
        s2_face_crops_sha256=crop_hashes,
        s3_resized_sha256=s3,
        s4_normalized_sha256=s4,
    )


def generate(base_dir: Path, out_path: Path, face_backend: str = "stub") -> Dict[str, Dict]:
    manifest = default_golden_manifest(base_dir)
    decode_cfg = DecodeConfig()
    pp_cfg = PreprocessConfig()

    face_backend = (face_backend or "stub").lower()
    if face_backend not in ("stub", "mediapipe"):
        raise ValueError(f"Unsupported face backend: {face_backend}")

    face_cfg = FaceDetectorConfig(backend=face_backend)  # type: ignore[arg-type]

    results: Dict[str, Dict] = {}
    for item in manifest:
        hashes = compute_stage_hashes_for_file(item.path, decode_cfg, pp_cfg, face_cfg)
        results[item.id] = hashes.to_dict()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    # Usage:
    #   python -m ecdd_core.golden.generate_golden_hashes --face-backend mediapipe
    # or:
    #   python ecdd_core/golden/generate_golden_hashes.py --face-backend mediapipe
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--face-backend", default="stub", choices=["stub", "mediapipe"], help="Face detector backend")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[2] / "ECDD_Experiment_Data"
    out = Path(__file__).resolve().parents[1] / f"golden_hashes_s0_s4_s2_{args.face_backend}.json"

    print(f"Generating golden hashes from: {base}")
    print(f"Face backend: {args.face_backend}")
    print(f"Writing: {out}")
    generate(base, out, face_backend=args.face_backend)
    print("Done")
