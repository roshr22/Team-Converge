#!/usr/bin/env python3
"""Temperature scaling calibration CLI for ECDD.

Fits a temperature scalar on calibration set logits to minimize NLL.
Produces reliability curves and calibration reports.

Usage:
    # From logits/labels files
    python fit_temp.py --logits-file calib_logits.npy --labels-file calib_labels.npy --output-dir results/calibration/
    
    # Demo with synthetic data
    python fit_temp.py --demo --output-dir results/calibration/

Outputs:
    - temperature_params.json — Fitted temperature and metadata
    - calibration_report.json — ECE/NLL before/after calibration
    - reliability_curve.png — Reliability diagram before/after
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecdd_core.calibration.temperature_scaling import (
    fit_temperature,
    apply_temperature,
    expected_calibration_error,
    TemperatureScalingParams,
    nll_from_logits,
    sigmoid,
)
from evaluation.plot_diagnostics import plot_reliability_curve


def compute_calibration_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """Compute calibration metrics for given logits and labels.
    
    Args:
        logits: Raw model logits
        labels: Ground truth labels (0 or 1)
        temperature: Temperature to apply before computing metrics
    
    Returns:
        Dictionary with NLL, ECE, and accuracy metrics
    """
    scaled_logits = logits / temperature
    probs = sigmoid(scaled_logits)
    preds = (probs >= 0.5).astype(int)
    
    nll = nll_from_logits(scaled_logits, labels)
    ece = expected_calibration_error(probs, labels)
    accuracy = float(np.mean(preds == labels))
    
    return {
        "nll": float(nll),
        "ece": float(ece),
        "accuracy": float(accuracy),
        "mean_confidence": float(np.mean(probs)),
        "mean_label": float(np.mean(labels)),
    }


def run_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    t_min: float = 0.05,
    t_max: float = 10.0,
    steps: int = 200,
) -> Dict[str, Any]:
    """Run temperature scaling calibration and generate all outputs.
    
    Args:
        logits: Raw model logits
        labels: Ground truth labels (0 or 1)
        output_dir: Directory for output files
        t_min: Minimum temperature to search
        t_max: Maximum temperature to search
        steps: Number of grid search steps
    
    Returns:
        Dictionary with calibration results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Compute before-calibration metrics
    before_metrics = compute_calibration_metrics(logits, labels, temperature=1.0)
    
    # Fit temperature
    params, fit_details = fit_temperature(logits, labels, t_min, t_max, steps)
    
    # Compute after-calibration metrics
    after_metrics = compute_calibration_metrics(logits, labels, params.temperature)
    
    # Compute probabilities for plotting
    probs_before = sigmoid(logits)
    probs_after = sigmoid(apply_temperature(logits, params))
    
    # Generate reliability curve
    reliability_path = output_dir / f"{timestamp}_reliability_curve.png"
    plot_reliability_curve(
        labels, probs_before, reliability_path,
        title=f"Calibration: T={params.temperature:.3f}",
        before_label=f"Before (ECE={before_metrics['ece']:.4f})",
        calibrated_probs=probs_after,
        after_label=f"After (ECE={after_metrics['ece']:.4f})",
    )
    
    # Build results
    results = {
        "timestamp": timestamp,
        "temperature": params.temperature,
        "fit_details": fit_details,
        "before_calibration": before_metrics,
        "after_calibration": after_metrics,
        "improvement": {
            "nll_reduction": before_metrics["nll"] - after_metrics["nll"],
            "ece_reduction": before_metrics["ece"] - after_metrics["ece"],
        },
        "n_samples": len(labels),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
    }
    
    # Save temperature parameters (minimal, for deployment)
    params_path = output_dir / f"{timestamp}_temperature_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            "temperature": params.temperature,
            "timestamp": timestamp,
            "n_calibration_samples": len(labels),
        }, f, indent=2)
    
    # Save full calibration report
    report_path = output_dir / f"{timestamp}_calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Also save a "latest" symlink-style copy for easy access
    latest_params_path = output_dir / "temperature_params.json"
    with open(latest_params_path, "w", encoding="utf-8") as f:
        json.dump({
            "temperature": params.temperature,
            "timestamp": timestamp,
            "n_calibration_samples": len(labels),
        }, f, indent=2)
    
    print(f"[OK] Calibration complete!")
    print(f"   Temperature: {params.temperature:.4f}")
    print(f"   ECE: {before_metrics['ece']:.4f} -> {after_metrics['ece']:.4f}")
    print(f"   NLL: {before_metrics['nll']:.4f} -> {after_metrics['nll']:.4f}")
    print(f"")
    print(f"   Params: {params_path}")
    print(f"   Report: {report_path}")
    print(f"   Plot:   {reliability_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fit temperature scaling on calibration set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--logits-file", type=Path,
        help="Path to logits file (.npy or .json)"
    )
    parser.add_argument(
        "--labels-file", type=Path,
        help="Path to labels file (.npy or .json)"
    )
    parser.add_argument(
        "--predictions-file", type=Path,
        help="Path to combined predictions file with 'logits' and 'labels' keys"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/calibration"),
        help="Output directory for calibration artifacts"
    )
    parser.add_argument(
        "--t-min", type=float, default=0.05,
        help="Minimum temperature for grid search"
    )
    parser.add_argument(
        "--t-max", type=float, default=10.0,
        help="Maximum temperature for grid search"
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of grid search steps"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with synthetic data"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Generate synthetic calibration data
        print("Running demo calibration with synthetic data...")
        np.random.seed(42)
        n = 500
        
        labels = np.random.binomial(1, 0.5, n)
        # Simulate over-confident model (needs T > 1)
        true_logits = 2.0 * (2 * labels - 1)  # ±2 for fake/real
        noise = np.random.normal(0, 0.5, n)
        logits = true_logits + noise
        
        run_calibration(
            logits, labels, args.output_dir,
            args.t_min, args.t_max, args.steps
        )
        return
    
    # Load from files
    if args.predictions_file:
        ext = args.predictions_file.suffix.lower()
        if ext == ".npy":
            data = np.load(args.predictions_file, allow_pickle=True).item()
        else:
            with open(args.predictions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        logits = np.array(data["logits"])
        labels = np.array(data["labels"])
    elif args.logits_file and args.labels_file:
        ext = args.logits_file.suffix.lower()
        if ext == ".npy":
            logits = np.load(args.logits_file)
            labels = np.load(args.labels_file)
        else:
            with open(args.logits_file, "r", encoding="utf-8") as f:
                logits = np.array(json.load(f))
            with open(args.labels_file, "r", encoding="utf-8") as f:
                labels = np.array(json.load(f))
    else:
        parser.error("Either --predictions-file, (--logits-file and --labels-file), or --demo is required")
        return
    
    run_calibration(
        logits, labels, args.output_dir,
        args.t_min, args.t_max, args.steps
    )


if __name__ == "__main__":
    main()
