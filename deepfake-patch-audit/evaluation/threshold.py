"""Quantization-aware threshold tuning."""

import torch
import numpy as np
from .metrics import compute_metrics_at_threshold


class ThresholdTuner:
    """
    Tune decision threshold using validation set.
    Supports quantization-aware threshold selection.
    """

    def __init__(self, validation_loader, device="cuda"):
        """
        Args:
            validation_loader: Validation data loader
            device: Device for computation
        """
        self.validation_loader = validation_loader
        self.device = device

    def get_predictions(self, model):
        """Get predictions on validation set."""
        all_scores = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch in self.validation_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                fake_probs = probs[:, 1]

                all_scores.append(fake_probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        return labels, scores

    def find_best_threshold(self, model, metric="f1", search_range=None):
        """
        Find best threshold based on metric.

        Args:
            model: Model to evaluate
            metric: 'f1', 'accuracy', 'precision', 'recall'
            search_range: Custom range to search (default: 0.0-1.0)

        Returns:
            Best threshold and its metric value
        """
        labels, scores = self.get_predictions(model)

        if search_range is None:
            thresholds = np.arange(0.0, 1.01, 0.01)
        else:
            thresholds = np.array(search_range)

        best_threshold = 0.5
        best_score = 0.0
        all_results = []

        for threshold in thresholds:
            metrics = compute_metrics_at_threshold(labels, scores, threshold)
            score = metrics[metric]
            all_results.append(metrics)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score, all_results

    def compare_float_vs_quantized(self, float_model, quant_model, tolerance=0.01):
        """
        Compare predictions between float and quantized models.

        Args:
            float_model: Full precision model
            quant_model: Quantized model
            tolerance: Maximum allowed difference

        Returns:
            dict with comparison results
        """
        float_labels, float_scores = self.get_predictions(float_model)
        quant_labels, quant_scores = self.get_predictions(quant_model)

        # Compute differences
        score_diff = np.abs(float_scores - quant_scores)
        max_diff = score_diff.max()
        mean_diff = score_diff.mean()

        # Count mismatches
        float_pred = (float_scores > 0.5).astype(int)
        quant_pred = (quant_scores > 0.5).astype(int)
        mismatches = (float_pred != quant_pred).sum()
        mismatch_rate = mismatches / len(float_pred)

        return {
            "max_difference": float(max_diff),
            "mean_difference": float(mean_diff),
            "num_mismatches": int(mismatches),
            "mismatch_rate": float(mismatch_rate),
            "within_tolerance": bool(max_diff <= tolerance),
            "float_scores": float_scores.tolist(),
            "quant_scores": quant_scores.tolist(),
        }
