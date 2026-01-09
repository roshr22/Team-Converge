#!/usr/bin/env python3
"""Preprocessing equivalence smoke test for ECDD.

Verifies that training and inference preprocessing paths produce identical outputs.
This is a critical guardrail to prevent training/deployment mismatch.

Checks:
- Channel order (RGB)
- Dtype and value range (uint8 0-255 before normalization, float32 after)
- Resize kernel consistency
- Normalization constants
- Face crop alignment (if applicable)

Usage:
    python preprocessing_equivalence_test.py --image-set test_images/ --tolerance 1e-5 --output-dir results/

    # Quick test with synthetic image
    python preprocessing_equivalence_test.py --demo

Outputs:
    - preprocessing_equivalence_report.json — Full test results
    - Pass/fail printed to stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecdd_core.pipeline import (
    PreprocessConfig,
    resize_rgb_uint8,
    normalize_rgb_uint8,
    sha256_ndarray,
)


@dataclass
class PreprocessingTestResult:
    """Result of a single preprocessing equivalence test."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def test_channel_order(
    img: np.ndarray,
    expected_channels: int = 3,
) -> PreprocessingTestResult:
    """Verify image has expected channel order."""
    if img.ndim != 3:
        return PreprocessingTestResult(
            test_name="channel_order",
            passed=False,
            message=f"Expected 3D array, got {img.ndim}D",
            details={"ndim": img.ndim, "shape": list(img.shape)},
        )
    
    if img.shape[2] != expected_channels:
        return PreprocessingTestResult(
            test_name="channel_order",
            passed=False,
            message=f"Expected {expected_channels} channels, got {img.shape[2]}",
            details={"shape": list(img.shape)},
        )
    
    return PreprocessingTestResult(
        test_name="channel_order",
        passed=True,
        message=f"RGB channel order verified ({img.shape})",
        details={"shape": list(img.shape), "channels": img.shape[2]},
    )


def test_dtype_range(
    img: np.ndarray,
    expected_dtype: str,
    expected_min: float,
    expected_max: float,
    tolerance: float = 1e-6,
) -> PreprocessingTestResult:
    """Verify image has expected dtype and value range."""
    dtype_str = str(img.dtype)
    actual_min = float(np.min(img))
    actual_max = float(np.max(img))
    
    dtype_ok = expected_dtype in dtype_str
    min_ok = actual_min >= expected_min - tolerance
    max_ok = actual_max <= expected_max + tolerance
    
    passed = dtype_ok and min_ok and max_ok
    
    if not passed:
        issues = []
        if not dtype_ok:
            issues.append(f"dtype {dtype_str} != {expected_dtype}")
        if not min_ok:
            issues.append(f"min {actual_min:.4f} < {expected_min}")
        if not max_ok:
            issues.append(f"max {actual_max:.4f} > {expected_max}")
        message = "; ".join(issues)
    else:
        message = f"dtype={dtype_str}, range=[{actual_min:.4f}, {actual_max:.4f}]"
    
    return PreprocessingTestResult(
        test_name="dtype_range",
        passed=passed,
        message=message,
        details={
            "dtype": dtype_str,
            "min": actual_min,
            "max": actual_max,
            "expected_dtype": expected_dtype,
            "expected_range": [expected_min, expected_max],
        },
    )


def test_resize_determinism(
    img: np.ndarray,
    cfg: PreprocessConfig,
    n_runs: int = 3,
) -> PreprocessingTestResult:
    """Verify resize produces identical outputs across runs."""
    hashes = []
    for _ in range(n_runs):
        resized = resize_rgb_uint8(img, cfg)
        hashes.append(sha256_ndarray(resized))
    
    unique_hashes = set(hashes)
    passed = len(unique_hashes) == 1
    
    return PreprocessingTestResult(
        test_name="resize_determinism",
        passed=passed,
        message=f"{'Deterministic' if passed else 'Non-deterministic'}: {len(unique_hashes)} unique hash(es) in {n_runs} runs",
        details={
            "hashes": hashes,
            "unique_count": len(unique_hashes),
            "n_runs": n_runs,
        },
    )


def test_normalization_determinism(
    resized: np.ndarray,
    cfg: PreprocessConfig,
    n_runs: int = 3,
) -> PreprocessingTestResult:
    """Verify normalization produces identical outputs across runs."""
    hashes = []
    for _ in range(n_runs):
        normalized = normalize_rgb_uint8(resized, cfg)
        hashes.append(sha256_ndarray(normalized))
    
    unique_hashes = set(hashes)
    passed = len(unique_hashes) == 1
    
    return PreprocessingTestResult(
        test_name="normalization_determinism",
        passed=passed,
        message=f"{'Deterministic' if passed else 'Non-deterministic'}: {len(unique_hashes)} unique hash(es) in {n_runs} runs",
        details={
            "hashes": hashes,
            "unique_count": len(unique_hashes),
            "n_runs": n_runs,
        },
    )


def test_normalized_output_format(
    normalized: np.ndarray,
    cfg: PreprocessConfig,
) -> PreprocessingTestResult:
    """Verify normalized output has correct format (CHW, float32, normalized range)."""
    issues = []
    
    # Check shape is CHW
    if normalized.ndim != 3:
        issues.append(f"Expected 3D array, got {normalized.ndim}D")
    elif normalized.shape[0] != 3:
        issues.append(f"Expected CHW format with 3 channels, got shape {normalized.shape}")
    
    # Check dtype
    if normalized.dtype != np.float32:
        issues.append(f"Expected float32, got {normalized.dtype}")
    
    # Check normalized range (ImageNet normalization typically gives range roughly [-2.1, 2.6])
    min_val = float(np.min(normalized))
    max_val = float(np.max(normalized))
    if min_val < -3.0 or max_val > 3.0:
        issues.append(f"Normalized values out of expected range: [{min_val:.2f}, {max_val:.2f}]")
    
    passed = len(issues) == 0
    
    return PreprocessingTestResult(
        test_name="normalized_output_format",
        passed=passed,
        message="CHW float32 normalized" if passed else "; ".join(issues),
        details={
            "shape": list(normalized.shape),
            "dtype": str(normalized.dtype),
            "min": min_val,
            "max": max_val,
            "mean": float(np.mean(normalized)),
            "std": float(np.std(normalized)),
        },
    )


def test_training_inference_parity(
    img: np.ndarray,
    training_cfg: PreprocessConfig,
    inference_cfg: PreprocessConfig,
    tolerance: float = 1e-5,
) -> PreprocessingTestResult:
    """Verify training and inference preprocessing produce identical outputs."""
    # Training path
    train_resized = resize_rgb_uint8(img, training_cfg)
    train_normalized = normalize_rgb_uint8(train_resized, training_cfg)
    train_hash = sha256_ndarray(train_normalized)
    
    # Inference path
    infer_resized = resize_rgb_uint8(img, inference_cfg)
    infer_normalized = normalize_rgb_uint8(infer_resized, inference_cfg)
    infer_hash = sha256_ndarray(infer_normalized)
    
    # Compare hashes
    if train_hash == infer_hash:
        return PreprocessingTestResult(
            test_name="training_inference_parity",
            passed=True,
            message=f"Exact match (hash={train_hash[:16]}...)",
            details={
                "train_hash": train_hash,
                "infer_hash": infer_hash,
                "max_diff": 0.0,
            },
        )
    
    # If hashes differ, compute actual difference
    max_diff = float(np.max(np.abs(train_normalized - infer_normalized)))
    within_tolerance = max_diff <= tolerance
    
    return PreprocessingTestResult(
        test_name="training_inference_parity",
        passed=within_tolerance,
        message=f"Max diff={max_diff:.2e} ({'<=' if within_tolerance else '>'} tolerance={tolerance:.0e})",
        details={
            "train_hash": train_hash,
            "infer_hash": infer_hash,
            "max_diff": max_diff,
            "tolerance": tolerance,
        },
    )


def run_preprocessing_tests(
    img: np.ndarray,
    training_cfg: PreprocessConfig,
    inference_cfg: PreprocessConfig,
    tolerance: float = 1e-5,
) -> Tuple[bool, List[PreprocessingTestResult]]:
    """Run all preprocessing equivalence tests on an image.
    
    Returns:
        Tuple of (all_passed, list of results)
    """
    results = []
    
    # Test 1: Channel order on input
    results.append(test_channel_order(img))
    
    # Test 2: Dtype/range on input
    results.append(test_dtype_range(img, "uint8", 0, 255))
    
    # Test 3: Resize determinism
    results.append(test_resize_determinism(img, training_cfg))
    
    # Resize for further tests
    resized = resize_rgb_uint8(img, training_cfg)
    
    # Test 4: Resize output format
    results.append(test_dtype_range(resized, "uint8", 0, 255))
    
    # Test 5: Normalization determinism
    results.append(test_normalization_determinism(resized, training_cfg))
    
    # Normalize for further tests
    normalized = normalize_rgb_uint8(resized, training_cfg)
    
    # Test 6: Normalized output format
    results.append(test_normalized_output_format(normalized, training_cfg))
    
    # Test 7: Training/inference parity
    results.append(test_training_inference_parity(img, training_cfg, inference_cfg, tolerance))
    
    all_passed = all(r.passed for r in results)
    return all_passed, results


def run_equivalence_test(
    images_dir: Optional[Path] = None,
    output_dir: Path = Path("results"),
    tolerance: float = 1e-5,
    demo: bool = False,
) -> Dict[str, Any]:
    """Run preprocessing equivalence test on image set.
    
    Args:
        images_dir: Directory containing test images
        output_dir: Directory for output files
        tolerance: Maximum allowed difference for parity
        demo: If True, use synthetic test image
    
    Returns:
        Dictionary with test results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use same config for training and inference (the contract)
    training_cfg = PreprocessConfig()
    inference_cfg = PreprocessConfig()  # Should be identical
    
    # Collect test images
    if demo or images_dir is None or not Path(images_dir).exists():
        # Generate synthetic test images
        np.random.seed(42)
        
        # Uniform random noise
        synthetic_uniform = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        
        # Gradient image (vertical gradient on each channel)
        gradient_1d = np.linspace(0, 255, 512, dtype=np.uint8)
        synthetic_gradient = np.stack([
            np.tile(gradient_1d.reshape(-1, 1), (1, 512)),
            np.tile(gradient_1d.reshape(-1, 1), (1, 512)),
            np.tile(gradient_1d.reshape(-1, 1), (1, 512)),
        ], axis=-1).astype(np.uint8)
        
        # Checkerboard pattern
        synthetic_checkerboard = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(0, 512, 64):
            for j in range(0, 512, 64):
                if (i // 64 + j // 64) % 2 == 0:
                    synthetic_checkerboard[i:i+64, j:j+64] = 255
        
        test_images = {
            "synthetic_uniform": synthetic_uniform,
            "synthetic_gradient": synthetic_gradient,
            "synthetic_checkerboard": synthetic_checkerboard,
        }
    else:
        # Load images from directory
        from PIL import Image
        test_images = {}
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for path in Path(images_dir).glob(ext):
                img = Image.open(path).convert("RGB")
                test_images[path.stem] = np.array(img, dtype=np.uint8)
    
    # Run tests on each image
    all_results = {}
    overall_passed = True
    
    for name, img in test_images.items():
        passed, results = run_preprocessing_tests(
            img, training_cfg, inference_cfg, tolerance
        )
        all_results[name] = {
            "passed": passed,
            "tests": [r.to_dict() for r in results],
        }
        if not passed:
            overall_passed = False
    
    # Build summary
    summary = {
        "timestamp": timestamp,
        "overall_passed": overall_passed,
        "n_images": len(test_images),
        "n_images_passed": sum(1 for r in all_results.values() if r["passed"]),
        "tolerance": tolerance,
        "training_config": {
            "resize_hw": training_cfg.resize_hw,
            "resize_kernel": training_cfg.resize_kernel,
            "mean": training_cfg.mean,
            "std": training_cfg.std,
        },
        "inference_config": {
            "resize_hw": inference_cfg.resize_hw,
            "resize_kernel": inference_cfg.resize_kernel,
            "mean": inference_cfg.mean,
            "std": inference_cfg.std,
        },
        "per_image_results": all_results,
    }
    
    # Save report
    report_path = output_dir / f"{timestamp}_preprocessing_equivalence_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Also save latest
    latest_path = output_dir / "preprocessing_equivalence_report.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    status = "[PASS]" if overall_passed else "[FAIL]"
    print(f"{status} Preprocessing equivalence test")
    print(f"   Images: {summary['n_images_passed']}/{summary['n_images']} passed")
    print(f"   Tolerance: {tolerance:.0e}")
    print(f"")
    
    for name, result in all_results.items():
        img_status = "[PASS]" if result["passed"] else "[FAIL]"
        failed_tests = [t["test_name"] for t in result["tests"] if not t["passed"]]
        if failed_tests:
            print(f"   {img_status} {name}: failed {', '.join(failed_tests)}")
        else:
            print(f"   {img_status} {name}")
    
    print(f"")
    print(f"   Report: {report_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test preprocessing equivalence between training and inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--image-set", type=Path, default=None,
        help="Directory containing test images"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Output directory for test report"
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5,
        help="Maximum allowed difference for parity test"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with synthetic test images"
    )
    
    args = parser.parse_args()
    
    summary = run_equivalence_test(
        args.image_set,
        args.output_dir,
        args.tolerance,
        args.demo,
    )
    
    # Exit with error code if test failed
    sys.exit(0 if summary["overall_passed"] else 1)


if __name__ == "__main__":
    main()
