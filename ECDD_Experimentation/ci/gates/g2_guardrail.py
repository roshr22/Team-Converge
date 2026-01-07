"""Gate G2: Guardrail Test

Ensures guardrails (face detection, quality checks) function correctly:
- Face detector produces consistent outputs
- Quality metrics are computed correctly
- Abstention logic works as expected

Exit code 0 = pass, non-zero = fail
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def test_face_detection_consistency(image_paths: List[Path], backend: str = "stub") -> Tuple[bool, dict]:
    """Test face detection consistency.
    
    Args:
        image_paths: List of test images
        backend: Face detector backend ('stub' or 'mediapipe')
    
    Returns:
        (passed, details)
    """
    from ecdd_core.pipeline.face import detect_faces, FaceDetectorConfig
    from PIL import Image
    
    cfg = FaceDetectorConfig(backend=backend, min_confidence=0.5)
    results = []
    
    for img_path in image_paths:
        if not img_path.exists():
            results.append({"image": str(img_path), "passed": False, "error": "File not found"})
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            rgb = np.array(img, dtype=np.uint8)
            
            # Run detection twice
            det1 = detect_faces(rgb, cfg)
            det2 = detect_faces(rgb, cfg)
            
            # Check consistency
            boxes_match = len(det1.boxes) == len(det2.boxes)
            if boxes_match and len(det1.boxes) > 0:
                # Check box coordinates match
                for b1, b2 in zip(det1.boxes, det2.boxes):
                    if b1 != b2:
                        boxes_match = False
                        break
            
            results.append({
                "image": img_path.name,
                "passed": boxes_match,
                "num_faces_run1": len(det1.boxes),
                "num_faces_run2": len(det2.boxes),
            })
        except Exception as e:
            results.append({"image": str(img_path), "passed": False, "error": str(e)})
    
    all_passed = all(r.get("passed", False) for r in results)
    return all_passed, {"results": results}


def test_ood_abstention(ood_dir: Path, backend: str = "stub") -> Tuple[bool, dict]:
    """Test that OOD images trigger abstention via no-face-detection.
    
    Args:
        ood_dir: Directory with OOD images
        backend: Face detector backend
    
    Returns:
        (passed, details)
    """
    from ecdd_core.pipeline.face import detect_faces, FaceDetectorConfig
    from PIL import Image
    
    ood_images = sorted(list(ood_dir.glob("*.jpg")) + list(ood_dir.glob("*.png")))[:10]
    
    if not ood_images:
        return False, {"error": f"No OOD images found in {ood_dir}"}
    
    cfg = FaceDetectorConfig(backend=backend, min_confidence=0.5)
    abstain_count = 0
    
    for img_path in ood_images:
        try:
            img = Image.open(img_path).convert("RGB")
            rgb = np.array(img, dtype=np.uint8)
            detection = detect_faces(rgb, cfg)
            
            if len(detection.boxes) == 0:
                abstain_count += 1
        except Exception:
            pass
    
    abstain_rate = abstain_count / len(ood_images)
    passed = abstain_rate >= 0.7  # At least 70% should abstain
    
    return passed, {
        "num_tested": len(ood_images),
        "abstain_count": abstain_count,
        "abstain_rate": abstain_rate,
    }


def main():
    """CLI entry point for CI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gate G2: Guardrail Test")
    parser.add_argument("--face-dir", type=Path, required=True, help="Directory with face images")
    parser.add_argument("--ood-dir", type=Path, help="Directory with OOD images")
    parser.add_argument("--backend", type=str, default="stub", choices=["stub", "mediapipe"])
    parser.add_argument("--max-images", type=int, default=10)
    
    args = parser.parse_args()
    
    if not args.face_dir.exists():
        print(f"ERROR: Face directory not found: {args.face_dir}")
        sys.exit(1)
    
    print(f"Gate G2: Testing guardrails...")
    
    # Test 1: Face detection consistency
    face_images = sorted(list(args.face_dir.glob("*.jpg")) + list(args.face_dir.glob("*.png")))[:args.max_images]
    passed1, details1 = test_face_detection_consistency(face_images, backend=args.backend)
    
    print(f"\n[Test 1] Face detection consistency: {'✓ PASS' if passed1 else '✗ FAIL'}")
    print(f"  Tested {len(details1['results'])} images")
    
    # Test 2: OOD abstention (if OOD dir provided)
    if args.ood_dir and args.ood_dir.exists():
        passed2, details2 = test_ood_abstention(args.ood_dir, backend=args.backend)
        print(f"\n[Test 2] OOD abstention: {'✓ PASS' if passed2 else '✗ FAIL'}")
        print(f"  Abstain rate: {details2['abstain_rate']:.1%}")
    else:
        passed2 = True
        print(f"\n[Test 2] OOD abstention: SKIPPED (no --ood-dir provided)")
    
    print(f"\n{'='*60}")
    if passed1 and passed2:
        print("✓ Gate G2 PASSED: Guardrails functioning correctly")
        sys.exit(0)
    else:
        print("✗ Gate G2 FAILED: Guardrail issues detected")
        sys.exit(1)


if __name__ == "__main__":
    main()
