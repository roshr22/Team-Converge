"""Diagnostic plots for ECDD evaluation.

Generates:
- Confidence histograms for misclassified samples (FP vs FN)
- Reliability curves for calibration analysis
- Confusion matrix heatmaps
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Use a non-interactive backend for headless operation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_confidence_histogram(
    fp_scores: np.ndarray,
    fn_scores: np.ndarray,
    output_path: Path,
    threshold: float = 0.5,
    bins: int = 30,
    title: str = "Misclassified Sample Confidence Distribution",
) -> Path:
    """Plot histogram of predicted probabilities for misclassified samples.
    
    Args:
        fp_scores: Predicted P(fake) for false positives (real predicted as fake)
        fn_scores: Predicted P(fake) for false negatives (fake predicted as real)
        output_path: Path to save the plot
        threshold: Classification threshold (shown as vertical line)
        bins: Number of histogram bins
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot FP distribution (above threshold)
    if len(fp_scores) > 0:
        ax.hist(fp_scores, bins=bins, alpha=0.6, color='red', 
                label=f'False Positives (n={len(fp_scores)})', density=True)
    
    # Plot FN distribution (below threshold)
    if len(fn_scores) > 0:
        ax.hist(fn_scores, bins=bins, alpha=0.6, color='blue',
                label=f'False Negatives (n={len(fn_scores)})', density=True)
    
    # Add threshold line
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold = {threshold}')
    
    ax.set_xlabel('Predicted P(Fake)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_reliability_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
    title: str = "Reliability Curve (Calibration)",
    before_label: str = "Model",
    calibrated_probs: Optional[np.ndarray] = None,
    after_label: str = "After Calibration",
) -> Path:
    """Plot reliability diagram showing calibration quality.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        output_path: Path to save the plot
        n_bins: Number of bins for reliability curve
        title: Plot title
        before_label: Label for uncalibrated curve
        calibrated_probs: Optional calibrated probabilities for comparison
        after_label: Label for calibrated curve
    
    Returns:
        Path to saved plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    def compute_reliability_bins(probs, labels, n_bins):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
            if np.sum(mask) > 0:
                bin_centers.append((lo + hi) / 2)
                bin_accuracies.append(np.mean(labels[mask]))
                bin_counts.append(np.sum(mask))
        
        return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Reliability curve
    centers, accuracies, counts = compute_reliability_bins(y_score, y_true, n_bins)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax1.plot(centers, accuracies, 'o-', color='blue', markersize=8, label=before_label)
    
    if calibrated_probs is not None:
        cal_centers, cal_accuracies, _ = compute_reliability_bins(calibrated_probs, y_true, n_bins)
        ax1.plot(cal_centers, cal_accuracies, 's-', color='green', markersize=8, label=after_label)
    
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Reliability Curve', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Histogram of predictions
    ax2.hist(y_score, bins=n_bins, alpha=0.6, color='blue', label=before_label, density=True)
    if calibrated_probs is not None:
        ax2.hist(calibrated_probs, bins=n_bins, alpha=0.6, color='green', label=after_label, density=True)
    
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_confusion_matrix(
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    output_path: Path,
    title: str = "Confusion Matrix",
    class_names: Tuple[str, str] = ("Real", "Fake"),
) -> Path:
    """Plot confusion matrix heatmap.
    
    Args:
        tp, tn, fp, fn: Confusion matrix values
        output_path: Path to save the plot
        title: Plot title
        class_names: Names for negative and positive classes
    
    Returns:
        Path to saved plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cm = np.array([[tn, fp], [fn, tp]])
    total = tn + fp + fn + tp
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    
    # Add labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f'Pred {class_names[0]}', f'Pred {class_names[1]}'])
    ax.set_yticklabels([f'True {class_names[0]}', f'True {class_names[1]}'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            text = f'{cm[i, j]}\n({100*cm[i, j]/total:.1f}%)'
            ax.text(j, i, text, ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black", fontsize=12)
    
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    output_path: Path,
    auc_value: Optional[float] = None,
    title: str = "ROC Curve",
) -> Path:
    """Plot ROC curve.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        output_path: Path to save the plot
        auc_value: Pre-computed AUC (if None, will be computed)
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    # Compute ROC curve points
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    
    tpr = np.concatenate([[0], tps / n_pos])
    fpr = np.concatenate([[0], fps / n_neg])
    
    if auc_value is None:
        try:
            auc_value = np.trapezoid(tpr, fpr)  # numpy 2.0+
        except AttributeError:
            auc_value = np.trapz(tpr, fpr)  # numpy 1.x
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_value:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def plot_grouped_metrics_bar(
    grouped_metrics: Dict[str, Any],
    metric_name: str,
    output_path: Path,
    title: str = None,
) -> Path:
    """Plot bar chart of a metric across groups.
    
    Args:
        grouped_metrics: Dict mapping group name to MetricsResult
        metric_name: Name of metric to plot (e.g., 'auc', 'f1')
        output_path: Path to save the plot
        title: Plot title
    
    Returns:
        Path to saved plot
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    groups = []
    values = []
    
    for group_name, result in grouped_metrics.items():
        if group_name != "_overall":
            groups.append(group_name)
            values.append(getattr(result, metric_name, 0))
    
    if not groups:
        return output_path
    
    # Sort by value
    sorted_indices = np.argsort(values)
    groups = [groups[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(groups) * 0.4)))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(groups)))
    bars = ax.barh(groups, values, color=colors)
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(min(value + 0.02, 0.95), bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel(metric_name.upper(), fontsize=12)
    ax.set_ylabel('Group', fontsize=12)
    ax.set_title(title or f'{metric_name.upper()} by Group', fontsize=14)
    ax.set_xlim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return output_path
