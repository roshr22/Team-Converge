"""Gate G6: Release Gate

Final pre-deployment validation:
- All previous gates (G1-G5) passed
- Operating point metrics meet requirements
- Abstain rate is acceptable
- Documentation and versioning are complete

Exit code 0 = pass, non-zero = fail
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import json


def check_gate_results(gates_dir: Path) -> Tuple[bool, dict]:
    """Check that all previous gates passed.
    
    Args:
        gates_dir: Directory containing gate result files
    
    Returns:
        (passed, details)
    """
    required_gates = ["g1", "g2", "g3", "g4", "g5"]
    gate_status = {}
    
    for gate in required_gates:
        result_file = gates_dir / f"{gate}_result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                gate_status[gate] = result.get("passed", False)
            except Exception:
                gate_status[gate] = False
        else:
            gate_status[gate] = False
    
    all_passed = all(gate_status.values())
    
    return all_passed, {"gate_status": gate_status}


def check_operating_point(metrics_file: Path, min_tpr: float = 0.8, max_fpr: float = 0.05) -> Tuple[bool, dict]:
    """Check operating point metrics.
    
    Args:
        metrics_file: JSON file with evaluation metrics
        min_tpr: Minimum acceptable TPR
        max_fpr: Maximum acceptable FPR
    
    Returns:
        (passed, details)
    """
    if not metrics_file.exists():
        return False, {"error": f"Metrics file not found: {metrics_file}"}
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        tpr = metrics.get("tpr", 0.0)
        fpr = metrics.get("fpr", 1.0)
        
        passed = tpr >= min_tpr and fpr <= max_fpr
        
        return passed, {
            "tpr": tpr,
            "fpr": fpr,
            "min_tpr": min_tpr,
            "max_fpr": max_fpr,
            "tpr_ok": tpr >= min_tpr,
            "fpr_ok": fpr <= max_fpr,
        }
    except Exception as e:
        return False, {"error": str(e)}


def check_abstain_rate(metrics_file: Path, max_abstain: float = 0.15) -> Tuple[bool, dict]:
    """Check abstain rate is acceptable.
    
    Args:
        metrics_file: JSON file with evaluation metrics
        max_abstain: Maximum acceptable abstain rate
    
    Returns:
        (passed, details)
    """
    if not metrics_file.exists():
        return False, {"error": f"Metrics file not found: {metrics_file}"}
    
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        abstain_rate = metrics.get("abstain_rate", 1.0)
        passed = abstain_rate <= max_abstain
        
        return passed, {
            "abstain_rate": abstain_rate,
            "max_abstain": max_abstain,
            "acceptable": passed,
        }
    except Exception as e:
        return False, {"error": str(e)}


def check_versioning(bundle_dir: Path) -> Tuple[bool, dict]:
    """Check deployment bundle has required versioning artifacts.
    
    Args:
        bundle_dir: Directory with deployment bundle
    
    Returns:
        (passed, details)
    """
    if not bundle_dir.exists():
        return False, {"error": f"Bundle directory not found: {bundle_dir}"}
    
    required_files = [
        "model_hash.txt",
        "calibration_params.json",
        "policy_contract.yaml",
    ]
    
    found = {}
    for fname in required_files:
        found[fname] = (bundle_dir / fname).exists()
    
    all_present = all(found.values())
    
    return all_present, {"required_files": found}


def main():
    """CLI entry point for CI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Gate G6: Release Gate")
    parser.add_argument("--gates-dir", type=Path, help="Directory with previous gate results")
    parser.add_argument("--metrics-file", type=Path, help="Evaluation metrics JSON")
    parser.add_argument("--bundle-dir", type=Path, help="Deployment bundle directory")
    parser.add_argument("--skip-gates", action="store_true", help="Skip checking previous gates")
    
    args = parser.parse_args()
    
    print(f"Gate G6: Release validation...")
    
    all_passed = True
    
    # Test 1: Previous gates
    if not args.skip_gates and args.gates_dir:
        passed1, details1 = check_gate_results(args.gates_dir)
        print(f"\n[Test 1] Previous gates: {'✓ PASS' if passed1 else '✗ FAIL'}")
        for gate, status in details1["gate_status"].items():
            print(f"  {gate.upper()}: {'✓' if status else '✗'}")
        all_passed = all_passed and passed1
    else:
        print(f"\n[Test 1] Previous gates: SKIPPED")
    
    # Test 2: Operating point
    if args.metrics_file:
        passed2, details2 = check_operating_point(args.metrics_file)
        print(f"\n[Test 2] Operating point: {'✓ PASS' if passed2 else '✗ FAIL'}")
        if 'tpr' in details2:
            print(f"  TPR: {details2['tpr']:.3f} (min: {details2['min_tpr']})")
            print(f"  FPR: {details2['fpr']:.3f} (max: {details2['max_fpr']})")
        all_passed = all_passed and passed2
    else:
        print(f"\n[Test 2] Operating point: SKIPPED")
    
    # Test 3: Abstain rate
    if args.metrics_file:
        passed3, details3 = check_abstain_rate(args.metrics_file)
        print(f"\n[Test 3] Abstain rate: {'✓ PASS' if passed3 else '✗ FAIL'}")
        if 'abstain_rate' in details3:
            print(f"  Rate: {details3['abstain_rate']:.1%} (max: {details3['max_abstain']:.1%})")
        all_passed = all_passed and passed3
    else:
        print(f"\n[Test 3] Abstain rate: SKIPPED")
    
    # Test 4: Versioning
    if args.bundle_dir:
        passed4, details4 = check_versioning(args.bundle_dir)
        print(f"\n[Test 4] Versioning artifacts: {'✓ PASS' if passed4 else '✗ FAIL'}")
        for fname, exists in details4["required_files"].items():
            print(f"  {fname}: {'✓' if exists else '✗'}")
        all_passed = all_passed and passed4
    else:
        print(f"\n[Test 4] Versioning: SKIPPED")
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ Gate G6 PASSED: Release approved")
        sys.exit(0)
    else:
        print("✗ Gate G6 FAILED: Release blocked")
        sys.exit(1)


if __name__ == "__main__":
    main()
