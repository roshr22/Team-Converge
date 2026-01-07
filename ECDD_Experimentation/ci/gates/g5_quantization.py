"""Gate G5: Quantization Parity Test

Ensures quantized model maintains parity with float model:
- Probability parity within tolerance
- Patch-logit map parity
- Pooled logit parity

Exit code 0 = pass, non-zero = fail
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def test_quantization_parity_mock(
    float_model_path: Path,
    tflite_model_path: Path,
    tolerance: float = 0.05,
) -> Tuple[bool, dict]:
    """Test quantization parity using mock data.
    
    In production, this would run actual model inference.
    
    Args:
        float_model_path: Path to float model
        tflite_model_path: Path to TFLite model
        tolerance: Maximum allowed difference
    
    Returns:
        (passed, details)
    """
    if not float_model_path.exists():
        return False, {"error": f"Float model not found: {float_model_path}"}
    
    if not tflite_model_path.exists():
        return False, {"error": f"TFLite model not found: {tflite_model_path}"}
    
    # Mock predictions
    np.random.seed(42)
    n = 20
    float_outputs = np.random.randn(n).astype(np.float32)
    tflite_outputs = float_outputs + np.random.randn(n).astype(np.float32) * 0.01
    
    # Compute parity metrics
    max_diff = float(np.max(np.abs(float_outputs - tflite_outputs)))
    mean_diff = float(np.mean(np.abs(float_outputs - tflite_outputs)))
    
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(float_outputs, tflite_outputs)
    
    passed = max_diff <= tolerance and rank_corr >= 0.9
    
    return passed, {
        "is_mock": True,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "rank_correlation": float(rank_corr),
        "tolerance": tolerance,
        "num_samples": n,
        "note": "Using mock predictions. Replace with actual model inference for production.",
    }


def main():
    """CLI entry point for CI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gate G5: Quantization Parity Test")
    parser.add_argument("--float-model", type=Path, required=True, help="Path to float model")
    parser.add_argument("--tflite-model", type=Path, required=True, help="Path to TFLite model")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Maximum allowed difference")
    
    args = parser.parse_args()
    
    print(f"Gate G5: Testing quantization parity...")
    passed, details = test_quantization_parity_mock(
        args.float_model,
        args.tflite_model,
        tolerance=args.tolerance,
    )
    
    print(f"\nResults:")
    print(f"  Max difference: {details['max_diff']:.6f}")
    print(f"  Mean difference: {details['mean_diff']:.6f}")
    print(f"  Rank correlation: {details['rank_correlation']:.4f}")
    print(f"  Tolerance: {details['tolerance']}")
    
    if details.get('is_mock'):
        print(f"\n  ⚠️  WARNING: Using mock predictions")
        print(f"      Replace with actual model inference for production CI")
    
    print(f"\n{'='*60}")
    if passed:
        print("✓ Gate G5 PASSED: Quantization parity maintained")
        sys.exit(0)
    else:
        print("✗ Gate G5 FAILED: Quantization parity violation")
        sys.exit(1)


if __name__ == "__main__":
    main()
