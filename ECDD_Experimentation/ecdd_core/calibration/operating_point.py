"""Operating point selection for deployment.

Given calibrated probabilities and labels, select thresholds to satisfy an error budget.
Implements E4.4.

Common constraint from docs: FPR on real faces <= 5%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    target_fpr: float


def compute_fpr_tpr(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Tuple[float, float]:
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    preds = (probs >= threshold).astype(int)

    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    tp = np.sum((preds == 1) & (labels == 1))
    fn = np.sum((preds == 0) & (labels == 1))

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(fpr), float(tpr)


def select_threshold_at_fpr(probs: np.ndarray, labels: np.ndarray, target_fpr: float = 0.05) -> Tuple[OperatingPoint, Dict]:
    """Select the highest threshold that satisfies FPR <= target_fpr."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    # Candidate thresholds are unique probs
    thresholds = np.unique(probs)
    thresholds = np.sort(thresholds)

    best_t = 0.5
    best_tpr = -1.0
    best_fpr = None

    for t in thresholds:
        fpr, tpr = compute_fpr_tpr(probs, labels, float(t))
        if fpr <= target_fpr:
            # choose highest TPR (or highest threshold with acceptable FPR)
            if tpr > best_tpr:
                best_tpr = tpr
                best_t = float(t)
                best_fpr = fpr

    if best_fpr is None:
        # No threshold meets constraint; return default with details
        best_t = float(np.quantile(probs, 1 - target_fpr))
        best_fpr, best_tpr = compute_fpr_tpr(probs, labels, best_t)

    op = OperatingPoint(threshold=best_t, target_fpr=target_fpr)
    return op, {"selected_fpr": best_fpr, "selected_tpr": best_tpr}
