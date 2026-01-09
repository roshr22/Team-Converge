#!/usr/bin/env python3
"""Checkpoint selection for deployment-relevant criteria.

Selects the best model checkpoint based on deployment metrics (not training loss):
- FPR at fixed TPR (e.g., FPR@TPR=0.95)
- F1 at a declared operating threshold
- TPR at fixed FPR (e.g., TPR@FPR=0.05)

Usage:
    python checkpoint_selection.py --checkpoints-dir runs/ --metric fpr_at_tpr --target 0.95 --output best_model.json

Outputs:
    - best_model.json — Selected checkpoint with metadata
    - selection_report.json — Comparison of all checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Literal, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint."""
    checkpoint_path: str
    epoch: int
    train_loss: float
    val_loss: float
    auc: float
    f1: float
    tpr_at_fpr: Dict[str, float]  # e.g., {"0.05": 0.92}
    fpr_at_tpr: Dict[str, float]  # e.g., {"0.95": 0.03}
    threshold: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectedCheckpoint:
    """Selected best checkpoint with selection rationale."""
    checkpoint_path: str
    selection_criterion: str
    target_value: float
    achieved_value: float
    operating_threshold: float
    epoch: int
    auc: float
    f1: float
    temperature: Optional[float]
    bundle_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_checkpoint_metrics(metrics_file: Path) -> CheckpointMetrics:
    """Load metrics from a checkpoint metrics JSON file.
    
    Expected format:
    {
        "checkpoint_path": "...",
        "epoch": 10,
        "train_loss": 0.1,
        "val_loss": 0.15,
        "auc": 0.95,
        "f1": 0.88,
        "tpr_at_fpr": {"0.05": 0.92},
        "fpr_at_tpr": {"0.95": 0.03},
        "threshold": 0.5
    }
    """
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return CheckpointMetrics(
        checkpoint_path=data.get("checkpoint_path", str(metrics_file)),
        epoch=data.get("epoch", 0),
        train_loss=data.get("train_loss", float("inf")),
        val_loss=data.get("val_loss", float("inf")),
        auc=data.get("auc", 0.0),
        f1=data.get("f1", 0.0),
        tpr_at_fpr=data.get("tpr_at_fpr", {}),
        fpr_at_tpr=data.get("fpr_at_tpr", {}),
        threshold=data.get("threshold", 0.5),
        metadata=data.get("metadata", {}),
    )


def select_best_checkpoint(
    checkpoints: List[CheckpointMetrics],
    criterion: str,
    target: float,
    minimize: bool = True,
) -> Tuple[CheckpointMetrics, float]:
    """Select the best checkpoint based on deployment criterion.
    
    Args:
        checkpoints: List of checkpoint metrics
        criterion: Selection criterion:
            - "fpr_at_tpr": Select checkpoint with lowest FPR at target TPR
            - "tpr_at_fpr": Select checkpoint with highest TPR at target FPR
            - "f1": Select checkpoint with highest F1
            - "auc": Select checkpoint with highest AUC
        target: Target value for fpr_at_tpr or tpr_at_fpr metrics
        minimize: Whether to minimize (True for fpr_at_tpr) or maximize
    
    Returns:
        Tuple of (best checkpoint, achieved metric value)
    """
    if not checkpoints:
        raise ValueError("No checkpoints provided")
    
    def get_metric_value(cp: CheckpointMetrics) -> float:
        if criterion == "fpr_at_tpr":
            return cp.fpr_at_tpr.get(str(target), float("inf"))
        elif criterion == "tpr_at_fpr":
            return cp.tpr_at_fpr.get(str(target), 0.0)
        elif criterion == "f1":
            return cp.f1
        elif criterion == "auc":
            return cp.auc
        elif criterion == "val_loss":
            return cp.val_loss
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    if minimize:
        best = min(checkpoints, key=get_metric_value)
    else:
        best = max(checkpoints, key=get_metric_value)
    
    return best, get_metric_value(best)


def create_bundle_metadata(
    checkpoint: CheckpointMetrics,
    temperature: Optional[float] = None,
    operating_threshold: float = 0.5,
    selection_criterion: str = "",
) -> Dict[str, Any]:
    """Create metadata for deployment bundle.
    
    This metadata should be persisted with the exported model.
    """
    return {
        "model": {
            "checkpoint_path": checkpoint.checkpoint_path,
            "epoch": checkpoint.epoch,
            "auc": checkpoint.auc,
            "f1": checkpoint.f1,
        },
        "calibration": {
            "temperature": temperature,
            "method": "temperature_scaling" if temperature else None,
        },
        "operating_point": {
            "threshold": operating_threshold,
            "selection_criterion": selection_criterion,
            "tpr_at_fpr": checkpoint.tpr_at_fpr,
            "fpr_at_tpr": checkpoint.fpr_at_tpr,
        },
        "timestamp": datetime.now().isoformat(),
    }


def run_checkpoint_selection(
    checkpoints_dir: Path,
    output_dir: Path,
    criterion: str = "fpr_at_tpr",
    target: float = 0.95,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """Run checkpoint selection and generate outputs.
    
    Args:
        checkpoints_dir: Directory containing checkpoint metrics JSON files
        output_dir: Directory for output files
        criterion: Selection criterion
        target: Target value for criterion
        temperature: Optional calibration temperature to include in bundle
    
    Returns:
        Dictionary with selection results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find all metrics files
    metrics_files = list(Path(checkpoints_dir).glob("**/metrics.json"))
    if not metrics_files:
        metrics_files = list(Path(checkpoints_dir).glob("**/*metrics*.json"))
    
    if not metrics_files:
        # Demo mode: generate synthetic checkpoints
        print("No metrics files found. Generating synthetic checkpoints for demo...")
        checkpoints = []
        for epoch in range(1, 11):
            np.random.seed(epoch)
            checkpoints.append(CheckpointMetrics(
                checkpoint_path=f"checkpoints/epoch_{epoch}.pth",
                epoch=epoch,
                train_loss=0.5 / epoch + np.random.uniform(0, 0.05),
                val_loss=0.6 / epoch + np.random.uniform(0, 0.1),
                auc=0.85 + 0.01 * epoch + np.random.uniform(0, 0.02),
                f1=0.80 + 0.015 * epoch + np.random.uniform(0, 0.02),
                tpr_at_fpr={"0.05": 0.80 + 0.02 * epoch, "0.01": 0.70 + 0.02 * epoch},
                fpr_at_tpr={"0.95": 0.10 - 0.008 * epoch, "0.90": 0.06 - 0.005 * epoch},
                threshold=0.5,
                metadata={"synthetic": True},
            ))
    else:
        checkpoints = [load_checkpoint_metrics(f) for f in metrics_files]
    
    # Determine if we should minimize or maximize
    minimize = criterion in ["fpr_at_tpr", "val_loss"]
    
    # Select best checkpoint
    best, achieved_value = select_best_checkpoint(
        checkpoints, criterion, target, minimize
    )
    
    # Create bundle metadata
    bundle_metadata = create_bundle_metadata(
        best, temperature, best.threshold, criterion
    )
    
    # Build results
    selected = SelectedCheckpoint(
        checkpoint_path=best.checkpoint_path,
        selection_criterion=criterion,
        target_value=target,
        achieved_value=achieved_value,
        operating_threshold=best.threshold,
        epoch=best.epoch,
        auc=best.auc,
        f1=best.f1,
        temperature=temperature,
        bundle_metadata=bundle_metadata,
    )
    
    results = {
        "selected": selected.to_dict(),
        "all_checkpoints": [cp.to_dict() for cp in checkpoints],
        "selection_summary": {
            "criterion": criterion,
            "target": target,
            "n_candidates": len(checkpoints),
            "best_epoch": best.epoch,
            "best_value": achieved_value,
        },
        "timestamp": timestamp,
    }
    
    # Save selected checkpoint info
    selected_path = output_dir / f"{timestamp}_best_model.json"
    with open(selected_path, "w", encoding="utf-8") as f:
        json.dump(selected.to_dict(), f, indent=2)
    
    # Save full selection report
    report_path = output_dir / f"{timestamp}_selection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    # Save latest (for easy access)
    latest_path = output_dir / "best_model.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(selected.to_dict(), f, indent=2)
    
    print(f"[OK] Checkpoint selection complete!")
    print(f"   Criterion: {criterion}={target}")
    print(f"   Best: epoch {best.epoch}, {criterion}={achieved_value:.4f}")
    print(f"   AUC: {best.auc:.4f}, F1: {best.f1:.4f}")
    print(f"")
    print(f"   Selected: {selected_path}")
    print(f"   Report:   {report_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Select best checkpoint by deployment-relevant criteria",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--checkpoints-dir", type=Path, default=Path("checkpoints"),
        help="Directory containing checkpoint metrics files"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/checkpoints"),
        help="Output directory for selection artifacts"
    )
    parser.add_argument(
        "--criterion", type=str, default="fpr_at_tpr",
        choices=["fpr_at_tpr", "tpr_at_fpr", "f1", "auc", "val_loss"],
        help="Selection criterion"
    )
    parser.add_argument(
        "--target", type=float, default=0.95,
        help="Target value for fpr_at_tpr or tpr_at_fpr"
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Calibration temperature to include in bundle metadata"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo with synthetic checkpoints"
    )
    
    args = parser.parse_args()
    
    if args.demo:
        print("Running demo checkpoint selection...")
        args.checkpoints_dir = Path("nonexistent_for_demo")
    
    run_checkpoint_selection(
        args.checkpoints_dir,
        args.output_dir,
        args.criterion,
        args.target,
        args.temperature,
    )


if __name__ == "__main__":
    main()
