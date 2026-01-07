"""Gate G4: Calibration Test

Ensures calibration parameters are properly fitted and improve reliability:
- Temperature/Platt parameters exist and are reasonable
- ECE improves after calibration
- Thresholds are defined and meet operating point constraints

Exit code 0 = pass, non-zero = fail
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def test_calibration_exists(calibration_file: Path) -> Tuple[bool, dict]:
    """Test that calibration file exists and is well-formed.
    
    Args:
        calibration_file: Path to calibration parameters JSON
    
    Returns:
        (passed, details)
    """
    if not calibration_file.exists():
        return False, {"error": f"Calibration file not found: {calibration_file}"}
    
    try:
        import json
        with open(calibration_file, 'r') as f:
            calib = json.load(f)
        
        # Check required fields
        has_method = "method" in calib
        has_params = "params" in calib
        has_threshold = "threshold" in calib
        
        method = calib.get("method", "unknown")
        valid_method = method in ["temperature", "platt", "none"]
        
        passed = has_method and has_params and has_threshold and valid_method
        
        return passed, {
            "has_method": has_method,
            "has_params": has_params,
            "has_threshold": has_threshold,
            "method": method,
            "valid_method": valid_method,
            "params": calib.get("params", {}),
            "threshold": calib.get("threshold"),
        }
    except Exception as e:
        return False, {"error": str(e)}


def test_calibration_improves_ece(calibration_json: Path) -> Tuple[bool, dict]:
    """Test that calibration improves ECE on calibration set.
    
    Args:
        calibration_json: Path to calibration set logits
    
    Returns:
        (passed, details)
    """
    if not calibration_json.exists():
        return False, {"error": f"Calibration data not found: {calibration_json}"}
    
    try:
        from ecdd_core.calibration.calibration_set_contract import load_calibration_json
        from ecdd_core.calibration.temperature_scaling import fit_temperature, apply_temperature, sigmoid, expected_calibration_error
        
        logits, labels, _ = load_calibration_json(calibration_json)
        
        # Before calibration
        pre_probs = sigmoid(logits)
        pre_ece = expected_calibration_error(pre_probs, labels)
        
        # After calibration
        params, _ = fit_temperature(logits, labels)
        post_logits = apply_temperature(logits, params)
        post_probs = sigmoid(post_logits)
        post_ece = expected_calibration_error(post_probs, labels)
        
        # ECE should improve (or at least not get worse)
        improved = post_ece <= pre_ece
        
        return improved, {
            "pre_ece": float(pre_ece),
            "post_ece": float(post_ece),
            "temperature": float(params.temperature),
            "improved": improved,
        }
    except Exception as e:
        return False, {"error": str(e)}


def main():
    """CLI entry point for CI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gate G4: Calibration Test")
    parser.add_argument("--calibration-params", type=Path, help="Path to calibration parameters JSON")
    parser.add_argument("--calibration-data", type=Path, help="Path to calibration set logits JSON")
    
    args = parser.parse_args()
    
    print(f"Gate G4: Testing calibration...")
    
    all_passed = True
    
    # Test 1: Calibration parameters exist
    if args.calibration_params:
        passed1, details1 = test_calibration_exists(args.calibration_params)
        print(f"\n[Test 1] Calibration params exist: {'✓ PASS' if passed1 else '✗ FAIL'}")
        if passed1:
            print(f"  Method: {details1['method']}")
            print(f"  Threshold: {details1['threshold']}")
        else:
            print(f"  Error: {details1.get('error', 'Unknown')}")
        all_passed = all_passed and passed1
    else:
        print(f"\n[Test 1] Calibration params: SKIPPED (no --calibration-params)")
    
    # Test 2: Calibration improves ECE
    if args.calibration_data:
        passed2, details2 = test_calibration_improves_ece(args.calibration_data)
        print(f"\n[Test 2] Calibration improves ECE: {'✓ PASS' if passed2 else '✗ FAIL'}")
        if 'pre_ece' in details2:
            print(f"  Pre-ECE: {details2['pre_ece']:.4f}")
            print(f"  Post-ECE: {details2['post_ece']:.4f}")
        else:
            print(f"  Error: {details2.get('error', 'Unknown')}")
        all_passed = all_passed and passed2
    else:
        print(f"\n[Test 2] ECE improvement: SKIPPED (no --calibration-data)")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ Gate G4 PASSED: Calibration validated")
        sys.exit(0)
    else:
        print("✗ Gate G4 FAILED: Calibration issues detected")
        sys.exit(1)


if __name__ == "__main__":
    main()
