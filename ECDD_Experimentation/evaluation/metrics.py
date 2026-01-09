"""Binary classification metrics for ECDD evaluation.

Computes AUC, AP, F1, TPR@FPR, FPR@TPR, and confusion matrix metrics
for deepfake detection evaluation, with grouping by source and compression.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    
    # Core metrics
    auc: float = 0.0
    ap: float = 0.0  # Average Precision
    f1: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    
    # Operating point metrics
    tpr_at_fpr: Dict[str, float] = field(default_factory=dict)  # e.g., {"0.01": 0.85, "0.05": 0.92}
    fpr_at_tpr: Dict[str, float] = field(default_factory=dict)  # e.g., {"0.90": 0.03, "0.95": 0.08}
    
    # Threshold used for F1/accuracy/precision/recall
    threshold: float = 0.5
    
    # Sample counts
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    
    # Confusion matrix values
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    
    # Group identifier (for grouped metrics)
    group: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve.
    
    Uses trapezoidal integration without sklearn dependency.
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    if len(np.unique(y_true)) < 2:
        return 0.5  # Undefined, return chance level
    
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    y_score = y_score[desc_score_indices]
    
    # Count positives and negatives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Compute TPR and FPR at each threshold
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add origin point
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Trapezoidal integration (numpy 2.0 compatibility)
    try:
        auc = np.trapezoid(tpr, fpr)  # numpy 2.0+
    except AttributeError:
        auc = np.trapz(tpr, fpr)  # numpy 1.x
    return float(auc)


def compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Average Precision (area under precision-recall curve)."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    n_pos = np.sum(y_true == 1)
    if n_pos == 0:
        return 0.0
    
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = y_true[desc_score_indices]
    
    # Compute precision at each threshold
    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)
    
    precision = tps / (tps + fps)
    recall = tps / n_pos
    
    # AP = sum of precision at each recall change point
    recall_diff = np.diff(np.concatenate([[0], recall]))
    ap = np.sum(precision * recall_diff)
    
    return float(ap)


def compute_tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, 
                        target_fprs: List[float] = None) -> Dict[str, float]:
    """Compute TPR at fixed FPR thresholds."""
    if target_fprs is None:
        target_fprs = [0.01, 0.05, 0.10]
    
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return {str(fpr): 0.0 for fpr in target_fprs}
    
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    result = {}
    for target_fpr in target_fprs:
        # Find first index where FPR exceeds target
        mask = fpr <= target_fpr
        if np.any(mask):
            idx = np.sum(mask) - 1
            result[str(target_fpr)] = float(tpr[idx])
        else:
            result[str(target_fpr)] = 0.0
    
    return result


def compute_fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray,
                        target_tprs: List[float] = None) -> Dict[str, float]:
    """Compute FPR at fixed TPR thresholds."""
    if target_tprs is None:
        target_tprs = [0.90, 0.95, 0.99]
    
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return {str(tpr): 1.0 for tpr in target_tprs}
    
    # Sort by score descending
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    result = {}
    for target_tpr in target_tprs:
        # Find first index where TPR reaches target
        mask = tpr >= target_tpr
        if np.any(mask):
            idx = np.argmax(mask)
            result[str(target_tpr)] = float(fpr[idx])
        else:
            result[str(target_tpr)] = 1.0
    
    return result


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute confusion matrix values (TP, TN, FP, FN)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    return tp, tn, fp, fn


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    target_fprs: List[float] = None,
    target_tprs: List[float] = None,
    group: Optional[str] = None,
) -> MetricsResult:
    """Compute comprehensive binary classification metrics.
    
    Args:
        y_true: Ground truth labels (0=real, 1=fake)
        y_score: Predicted probabilities of fake class
        threshold: Threshold for binary classification
        target_fprs: FPR values for TPR@FPR computation
        target_tprs: TPR values for FPR@TPR computation
        group: Optional group identifier
    
    Returns:
        MetricsResult with all computed metrics
    """
    if target_fprs is None:
        target_fprs = [0.01, 0.05, 0.10]
    if target_tprs is None:
        target_tprs = [0.90, 0.95, 0.99]
    
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    y_pred = (y_score >= threshold).astype(int)
    
    # Basic counts
    n_samples = len(y_true)
    n_positive = int(np.sum(y_true == 1))
    n_negative = int(np.sum(y_true == 0))
    
    # Confusion matrix
    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)
    
    # Derived metrics
    accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Ranking metrics
    auc = compute_auc(y_true, y_score)
    ap = compute_ap(y_true, y_score)
    
    # Operating point metrics
    tpr_at_fpr = compute_tpr_at_fpr(y_true, y_score, target_fprs)
    fpr_at_tpr = compute_fpr_at_tpr(y_true, y_score, target_tprs)
    
    return MetricsResult(
        auc=auc,
        ap=ap,
        f1=f1,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        tpr_at_fpr=tpr_at_fpr,
        fpr_at_tpr=fpr_at_tpr,
        threshold=threshold,
        n_samples=n_samples,
        n_positive=n_positive,
        n_negative=n_negative,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        group=group,
    )


def compute_grouped_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    threshold: float = 0.5,
    target_fprs: List[float] = None,
    target_tprs: List[float] = None,
) -> Dict[str, MetricsResult]:
    """Compute metrics grouped by a categorical variable.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        groups: Group labels for each sample
        threshold: Classification threshold
        target_fprs: FPR values for TPR@FPR
        target_tprs: TPR values for FPR@TPR
    
    Returns:
        Dictionary mapping group name to MetricsResult
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    groups = np.asarray(groups).ravel()
    
    unique_groups = np.unique(groups)
    results = {}
    
    # Overall metrics
    results["_overall"] = compute_binary_metrics(
        y_true, y_score, threshold, target_fprs, target_tprs, group="_overall"
    )
    
    # Per-group metrics
    for group in unique_groups:
        mask = groups == group
        if np.sum(mask) > 0:
            results[str(group)] = compute_binary_metrics(
                y_true[mask],
                y_score[mask],
                threshold,
                target_fprs,
                target_tprs,
                group=str(group),
            )
    
    return results


def get_misclassified_samples(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get indices and scores of misclassified samples.
    
    Returns:
        Tuple of (fp_indices, fn_indices, fp_scores, fn_scores)
    """
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    y_pred = (y_score >= threshold).astype(int)
    
    # False positives: real predicted as fake
    fp_mask = (y_true == 0) & (y_pred == 1)
    fp_indices = np.where(fp_mask)[0]
    fp_scores = y_score[fp_mask]
    
    # False negatives: fake predicted as real
    fn_mask = (y_true == 1) & (y_pred == 0)
    fn_indices = np.where(fn_mask)[0]
    fn_scores = y_score[fn_mask]
    
    return fp_indices, fn_indices, fp_scores, fn_scores
