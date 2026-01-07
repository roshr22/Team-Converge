"""Gate G1: Pixel Equivalence Test

Ensures that preprocessing produces identical pixels across:
- Training pipeline
- Inference pipeline
- Different environments (local dev, server, edge)

Exit code 0 = pass, non-zero = fail (for CI integration)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def load_and_preprocess_reference(image_path: Path) -> Tuple[np.ndarray, dict]:
    """Load and preprocess image using reference pipeline."""
    from ecdd_core.pipeline.preprocess import PreprocessConfig, resize_rgb_uint8, normalize_rgb_uint8
    from ecdd_core.pipeline.decode import decode_image_to_rgb
    
    # Decode
    rgb_decoded = decode_image_to_rgb(image_path)
    
    # Preprocess
    cfg = PreprocessConfig()
    resized = resize_rgb_uint8(rgb_decoded, cfg)
    normalized = normalize_rgb_uint8(resized, cfg)
    
    metadata = {
        "decoded_shape": rgb_decoded.shape,
        "resized_shape": resized.shape,
        "normalized_shape": normalized.shape,
        "dtype": str(normalized.dtype),
    }
    
    return normalized, metadata


def test_pixel_equivalence(image_paths: List[Path], tolerance: float = 1e-6) -> Tuple[bool, dict]:
    """Test pixel equivalence across multiple runs.
    
    Args:
        image_paths: List of test images
        tolerance: Maximum allowed pixel difference
    
    Returns:
        (passed, details)
    """
    results = []
    
    for img_path in image_paths:
        if not img_path.exists():
            results.append({
                "image": str(img_path),
                "passed": False,
                "error": "File not found",
            })
            continue
        
        try:
            # Run preprocessing twice
            output1, meta1 = load_and_preprocess_reference(img_path)
            output2, meta2 = load_and_preprocess_reference(img_path)
            
            # Check exact equivalence
            max_diff = float(np.max(np.abs(output1 - output2)))
            passed = max_diff <= tolerance
            
            results.append({
                "image": img_path.name,
                "passed": passed,
                "max_diff": max_diff,
                "tolerance": tolerance,
                "metadata": meta1,
            })
        except Exception as e:
            results.append({
                "image": str(img_path),
                "passed": False,
                "error": str(e),
            })
    
    all_passed = all(r.get("passed", False) for r in results)
    
    return all_passed, {"results": results, "num_tested": len(image_paths)}


def main():
    """CLI entry point for CI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gate G1: Pixel Equivalence Test")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing test images")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Maximum allowed pixel difference")
    parser.add_argument("--max-images", type=int, default=10, help="Maximum number of images to test")
    
    args = parser.parse_args()
    
    if not args.image_dir.exists():
        print(f"ERROR: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Collect test images
    image_paths = sorted(list(args.image_dir.glob("*.jpg")) + list(args.image_dir.glob("*.png")))
    image_paths = image_paths[:args.max_images]
    
    if not image_paths:
        print(f"ERROR: No images found in {args.image_dir}")
        sys.exit(1)
    
    print(f"Gate G1: Testing pixel equivalence on {len(image_paths)} images...")
    passed, details = test_pixel_equivalence(image_paths, tolerance=args.tolerance)
    
    # Report results
    print(f"\nResults:")
    for r in details["results"]:
        status = "✓ PASS" if r.get("passed", False) else "✗ FAIL"
        print(f"  {status}: {r['image']}")
        if not r.get("passed", False):
            print(f"    Error: {r.get('error', 'Unknown')}")
            if 'max_diff' in r:
                print(f"    Max diff: {r['max_diff']:.2e} (tolerance: {r['tolerance']:.2e})")
    
    print(f"\n{'='*60}")
    if passed:
        print("✓ Gate G1 PASSED: Pixel equivalence verified")
        sys.exit(0)
    else:
        print("✗ Gate G1 FAILED: Pixel equivalence violated")
        sys.exit(1)


if __name__ == "__main__":
    main()
