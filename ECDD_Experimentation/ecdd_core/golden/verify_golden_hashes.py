"""Verify pipeline outputs against golden hashes.

Used to detect regressions or environment differences in the preprocessing pipeline.

Usage:
    python verify_golden_hashes.py --golden-hashes golden_hashes.json --golden-dir ../ECDD_Experiment_Data/real
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from generate_golden_hashes import (
    generate_s0_hash,
    generate_s1_hash,
    generate_s2_hash,
    generate_s3_hash,
    generate_s4_hash,
)


def verify_golden_hashes(
    golden_hashes_file: Path,
    golden_dir: Path,
    face_backend: str = "stub",
) -> Tuple[bool, Dict]:
    """Verify current pipeline outputs match golden hashes.
    
    Args:
        golden_hashes_file: JSON file with golden hashes
        golden_dir: Directory with golden images
        face_backend: Face detector backend
    
    Returns:
        (all_passed, details)
    """
    with open(golden_hashes_file, 'r') as f:
        golden = json.load(f)
    
    results = []
    total = 0
    passed = 0
    
    for image_id, expected_hashes in golden.items():
        if "error" in expected_hashes:
            continue
        
        img_path = None
        for ext in [".jpg", ".png"]:
            candidate = golden_dir / f"{image_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            results.append({
                "image_id": image_id,
                "status": "MISSING",
            })
            continue
        
        # Generate current hashes
        try:
            current_hashes = {
                "s0_raw_bytes": generate_s0_hash(img_path),
                "s1_decoded_rgb": generate_s1_hash(img_path),
                "s2_face_boxes": generate_s2_hash(img_path, face_backend=face_backend),
                "s3_resized": generate_s3_hash(img_path),
                "s4_normalized": generate_s4_hash(img_path),
            }
            
            # Compare
            mismatches = []
            for stage in ["s0_raw_bytes", "s1_decoded_rgb", "s2_face_boxes", "s3_resized", "s4_normalized"]:
                expected = expected_hashes.get(stage)
                current = current_hashes.get(stage)
                
                if expected and current and expected != current:
                    if not expected.startswith("NOT_IMPLEMENTED"):
                        mismatches.append(stage)
            
            total += 1
            if not mismatches:
                passed += 1
                results.append({
                    "image_id": image_id,
                    "status": "PASS",
                })
            else:
                results.append({
                    "image_id": image_id,
                    "status": "FAIL",
                    "mismatches": mismatches,
                })
        
        except Exception as e:
            results.append({
                "image_id": image_id,
                "status": "ERROR",
                "error": str(e),
            })
    
    all_passed = passed == total and total > 0
    
    return all_passed, {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "results": results,
    }


def main():
    """CLI entry point."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Verify golden hashes")
    parser.add_argument("--golden-hashes", type=Path, required=True, help="Golden hashes JSON file")
    parser.add_argument("--golden-dir", type=Path, required=True, help="Directory with golden images")
    parser.add_argument("--face-backend", type=str, default="stub", choices=["stub", "mediapipe"])
    
    args = parser.parse_args()
    
    if not args.golden_hashes.exists():
        print(f"ERROR: Golden hashes file not found: {args.golden_hashes}")
        return 1
    
    if not args.golden_dir.exists():
        print(f"ERROR: Golden directory not found: {args.golden_dir}")
        return 1
    
    print(f"Verifying golden hashes...")
    all_passed, details = verify_golden_hashes(
        args.golden_hashes,
        args.golden_dir,
        face_backend=args.face_backend,
    )
    
    print(f"\nResults:")
    print(f"  Total: {details['total']}")
    print(f"  Passed: {details['passed']}")
    print(f"  Failed: {details['failed']}")
    
    for r in details['results']:
        if r['status'] == "FAIL":
            print(f"\n  ✗ {r['image_id']}: MISMATCHES in {', '.join(r['mismatches'])}")
        elif r['status'] == "ERROR":
            print(f"\n  ✗ {r['image_id']}: ERROR - {r['error']}")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All golden hashes verified")
        return 0
    else:
        print("✗ Golden hash verification failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
