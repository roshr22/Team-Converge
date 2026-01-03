"""Evaluation metrics: AUC, Accuracy@τ, and other metrics."""

import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix


def compute_roc_curve(y_true, y_scores):
    """
    Compute ROC curve.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities for positive class

    Returns:
        fpr, tpr, thresholds, auc_score
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def compute_accuracy_at_threshold(y_true, y_scores, threshold):
    """
    Compute accuracy at specific threshold.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        threshold: Decision threshold

    Returns:
        Accuracy at threshold
    """
    predictions = (y_scores > threshold).astype(int)
    accuracy = np.mean(predictions == y_true)
    return accuracy


def compute_metrics_at_threshold(y_true, y_scores, threshold):
    """
    Compute multiple metrics at specific threshold.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        threshold: Decision threshold

    Returns:
        dict with accuracy, precision, recall, f1
    """
    predictions = (y_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def find_optimal_threshold(y_true, y_scores, metric="f1"):
    """
    Find optimal decision threshold based on metric.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted probabilities
        metric: 'f1', 'accuracy', 'precision', 'recall'

    Returns:
        Optimal threshold and metric value
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_scores, threshold)
        score = metrics[metric]

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
