#!/usr/bin/env python3
"""Model evaluation CLI for ECDD.

Runs evaluation on indexed datasets and produces metrics grouped by source and compression.

Usage:
    python evaluate_model.py --index-file dataset_index.json --model-path model.pth --output-dir results/

Outputs:
    - <timestamp>_metrics.json — Full metrics in JSON format
    - <timestamp>_metrics.csv — Metrics summary in CSV format
    - <timestamp>_confusion.png — Confusion matrix heatmap
    - <timestamp>_misclassified_confidence.png — FP/FN confidence histogram
    - <timestamp>_roc.png — ROC curve
    - <timestamp>_auc_by_source.png — AUC bar chart by source
    - <timestamp>_auc_by_compression.png — AUC bar chart by compression
"""

from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.dataset_index import DatasetIndex, SampleEntry, load_index
from evaluation.metrics import (
    compute_binary_metrics,
    compute_grouped_metrics,
    get_misclassified_samples,
    MetricsResult,
)
from evaluation.plot_diagnostics import (
    plot_confidence_histogram,
    plot_reliability_curve,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_grouped_metrics_bar,
)


def get_model_predictions_stub(
    index: DatasetIndex,
    model_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stub function for getting model predictions.
    
    In production, this would load the model and run inference on all samples.
    For now, returns random predictions for testing the evaluation pipeline.
    
    Args:
        index: Dataset index with sample paths
        model_path: Path to model weights (not used in stub)
    
    Returns:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        sources: Source labels for each sample
        compressions: Compression labels for each sample
    """
    n_samples = len(index.samples)
    
    y_true = np.array([s.label for s in index.samples])
    sources = np.array([s.source for s in index.samples])
    compressions = np.array([s.compression for s in index.samples])
    
    # Generate mock predictions - slightly correlated with true labels
    np.random.seed(42)
    base_score = 0.3 + 0.4 * y_true  # Real: ~0.3, Fake: ~0.7
    noise = np.random.normal(0, 0.15, n_samples)
    y_score = np.clip(base_score + noise, 0.01, 0.99)
    
    return y_true, y_score, sources, compressions


def load_predictions_from_file(
    predictions_file: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load predictions from a JSON file.
    
    Expected format:
    {
        "predictions": [
            {"path": "...", "label": 0, "score": 0.3, "source": "...", "compression": "..."},
            ...
        ]
    }
    """
    with open(predictions_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    preds = data["predictions"]
    y_true = np.array([p["label"] for p in preds])
    y_score = np.array([p["score"] for p in preds])
    sources = np.array([p.get("source", "unknown") for p in preds])
    compressions = np.array([p.get("compression", "unknown") for p in preds])
    
    return y_true, y_score, sources, compressions


def run_evaluation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sources: np.ndarray,
    compressions: np.ndarray,
    output_dir: Path,
    threshold: float = 0.5,
    model_version: str = "unknown",
) -> Dict[str, Any]:
    """Run full evaluation and generate all outputs.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        sources: Source labels for each sample
        compressions: Compression labels for each sample
        output_dir: Directory for output files
        threshold: Classification threshold
        model_version: Model version string for metadata
    
    Returns:
        Dictionary with all computed metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Overall metrics
    overall_metrics = compute_binary_metrics(
        y_true, y_score, threshold, group="_overall"
    )
    
    # Grouped by source
    source_metrics = compute_grouped_metrics(
        y_true, y_score, sources, threshold
    )
    
    # Grouped by compression
    compression_metrics = compute_grouped_metrics(
        y_true, y_score, compressions, threshold
    )
    
    # Combined source x compression groups
    source_compression_groups = np.array([
        f"{s}|{c}" for s, c in zip(sources, compressions)
    ])
    source_compression_metrics = compute_grouped_metrics(
        y_true, y_score, source_compression_groups, threshold
    )
    
    # Get misclassified samples
    fp_indices, fn_indices, fp_scores, fn_scores = get_misclassified_samples(
        y_true, y_score, threshold
    )
    
    # ========== Generate Plots ==========
    
    # Confusion matrix
    plot_confusion_matrix(
        overall_metrics.tp, overall_metrics.tn,
        overall_metrics.fp, overall_metrics.fn,
        output_dir / f"{timestamp}_confusion.png",
        title=f"Confusion Matrix (threshold={threshold})"
    )
    
    # Misclassified confidence histogram
    plot_confidence_histogram(
        fp_scores, fn_scores,
        output_dir / f"{timestamp}_misclassified_confidence.png",
        threshold=threshold,
        title="Confidence Distribution of Misclassified Samples"
    )
    
    # ROC curve
    plot_roc_curve(
        y_true, y_score,
        output_dir / f"{timestamp}_roc.png",
        auc_value=overall_metrics.auc,
        title="ROC Curve"
    )
    
    # AUC by source
    plot_grouped_metrics_bar(
        source_metrics, "auc",
        output_dir / f"{timestamp}_auc_by_source.png",
        title="AUC by Source Family"
    )
    
    # AUC by compression
    plot_grouped_metrics_bar(
        compression_metrics, "auc",
        output_dir / f"{timestamp}_auc_by_compression.png",
        title="AUC by Compression Level"
    )
    
    # ========== Build Results Dict ==========
    
    results = {
        "metadata": {
            "timestamp": timestamp,
            "model_version": model_version,
            "threshold": threshold,
            "n_samples": int(len(y_true)),
            "n_real": int(np.sum(y_true == 0)),
            "n_fake": int(np.sum(y_true == 1)),
        },
        "overall": overall_metrics.to_dict(),
        "by_source": {k: v.to_dict() for k, v in source_metrics.items()},
        "by_compression": {k: v.to_dict() for k, v in compression_metrics.items()},
        "by_source_compression": {k: v.to_dict() for k, v in source_compression_metrics.items()},
        "misclassified": {
            "n_false_positives": len(fp_indices),
            "n_false_negatives": len(fn_indices),
            "fp_indices": fp_indices.tolist(),
            "fn_indices": fn_indices.tolist(),
        },
        "output_files": {
            "confusion": f"{timestamp}_confusion.png",
            "misclassified_confidence": f"{timestamp}_misclassified_confidence.png",
            "roc": f"{timestamp}_roc.png",
            "auc_by_source": f"{timestamp}_auc_by_source.png",
            "auc_by_compression": f"{timestamp}_auc_by_compression.png",
        }
    }
    
    # ========== Save JSON ==========
    json_path = output_dir / f"{timestamp}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # ========== Save CSV ==========
    csv_path = output_dir / f"{timestamp}_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "group_type", "group", "n_samples", "auc", "ap", "f1", 
            "accuracy", "precision", "recall", "tpr@fpr_0.05", "fpr@tpr_0.95"
        ])
        
        # Overall
        m = overall_metrics
        writer.writerow([
            "overall", "_all", m.n_samples, f"{m.auc:.4f}", f"{m.ap:.4f}", 
            f"{m.f1:.4f}", f"{m.accuracy:.4f}", f"{m.precision:.4f}", f"{m.recall:.4f}",
            f"{m.tpr_at_fpr.get('0.05', 0):.4f}", f"{m.fpr_at_tpr.get('0.95', 0):.4f}"
        ])
        
        # By source
        for group, m in source_metrics.items():
            if group != "_overall":
                writer.writerow([
                    "source", group, m.n_samples, f"{m.auc:.4f}", f"{m.ap:.4f}",
                    f"{m.f1:.4f}", f"{m.accuracy:.4f}", f"{m.precision:.4f}", f"{m.recall:.4f}",
                    f"{m.tpr_at_fpr.get('0.05', 0):.4f}", f"{m.fpr_at_tpr.get('0.95', 0):.4f}"
                ])
        
        # By compression
        for group, m in compression_metrics.items():
            if group != "_overall":
                writer.writerow([
                    "compression", group, m.n_samples, f"{m.auc:.4f}", f"{m.ap:.4f}",
                    f"{m.f1:.4f}", f"{m.accuracy:.4f}", f"{m.precision:.4f}", f"{m.recall:.4f}",
                    f"{m.tpr_at_fpr.get('0.05', 0):.4f}", f"{m.fpr_at_tpr.get('0.95', 0):.4f}"
                ])
    
    print(f"[OK] Evaluation complete!")
    print(f"   JSON: {json_path}")
    print(f"   CSV:  {csv_path}")
    print(f"   Plots: {output_dir}/{timestamp}_*.png")
    print(f"")
    print(f"   Overall AUC: {overall_metrics.auc:.4f}")
    print(f"   Overall F1:  {overall_metrics.f1:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ECDD model on indexed dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--index-file", type=Path,
        help="Path to dataset index JSON file"
    )
    parser.add_argument(
        "--predictions-file", type=Path,
        help="Path to pre-computed predictions JSON file (alternative to --model-path)"
    )
    parser.add_argument(
        "--model-path", type=Path,
        help="Path to model weights (not implemented - uses stub)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Output directory for metrics and plots"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold"
    )
    parser.add_argument(
        "--model-version", type=str, default="unknown",
        help="Model version string for metadata"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo evaluation with synthetic data"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Generate synthetic demo data
        print("Running demo evaluation with synthetic data...")
        np.random.seed(42)
        n = 1000
        
        y_true = np.random.binomial(1, 0.5, n)
        base_score = 0.3 + 0.4 * y_true
        y_score = np.clip(base_score + np.random.normal(0, 0.15, n), 0.01, 0.99)
        
        sources = np.random.choice(
            ["celebv2_real", "stable_diffusion", "midjourney", "dall_e"],
            n
        )
        compressions = np.random.choice(["high", "medium", "low"], n)
        
        run_evaluation(
            y_true, y_score, sources, compressions,
            args.output_dir, args.threshold, args.model_version
        )
        return
    
    if args.predictions_file:
        y_true, y_score, sources, compressions = load_predictions_from_file(
            args.predictions_file
        )
    elif args.index_file:
        index = load_index(args.index_file)
        y_true, y_score, sources, compressions = get_model_predictions_stub(
            index, args.model_path
        )
    else:
        parser.error("Either --index-file, --predictions-file, or --demo is required")
        return
    
    run_evaluation(
        y_true, y_score, sources, compressions,
        args.output_dir, args.threshold, args.model_version
    )


if __name__ == "__main__":
    main()
