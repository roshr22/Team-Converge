"""Evaluation loop for validation and testing."""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from pathlib import Path


class Evaluator:
    """
    Evaluate model on validation or test set.
    """

    def __init__(self, model, device="cuda"):
        """
        Args:
            model: Model to evaluate
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate(self, data_loader, return_predictions=False):
        """
        Evaluate model on dataset.

        Args:
            data_loader: Data loader for evaluation
            return_predictions: Return predictions and targets

        Returns:
            dict with metrics, optionally predictions
        """
        all_logits = []
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch in data_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(images)
                probs = F.softmax(logits, dim=1)

                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
                all_predictions.append(probs.cpu())

        # Concatenate all batches
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        predictions = torch.cat(all_predictions, dim=0)

        # Convert to numpy
        labels = labels.numpy()
        predictions = predictions.numpy()
        predicted_classes = predictions.argmax(axis=1)
        fake_probs = predictions[:, 1]

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(labels, predicted_classes),
            "precision": precision_score(labels, predicted_classes, zero_division=0),
            "recall": recall_score(labels, predicted_classes, zero_division=0),
            "f1": f1_score(labels, predicted_classes, zero_division=0),
            "auc": roc_auc_score(labels, fake_probs),
        }

        result = {"metrics": metrics}

        if return_predictions:
            result["predictions"] = predicted_classes.tolist()
            result["probabilities"] = fake_probs.tolist()
            result["labels"] = labels.tolist()

        return result

    def save_results(self, results, output_path):
        """Save evaluation results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def threshold_analysis(self, data_loader, thresholds=None):
        """
        Analyze accuracy at different decision thresholds.

        Args:
            data_loader: Data loader for evaluation
            thresholds: List of thresholds to test (default: 0.1 to 0.9)

        Returns:
            dict with accuracy at each threshold
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        results = self.evaluate(data_loader, return_predictions=True)
        labels = results["labels"]
        fake_probs = results["probabilities"]

        threshold_results = {}

        for threshold in thresholds:
            predicted = (fake_probs > threshold).astype(int)
            acc = accuracy_score(labels, predicted)
            threshold_results[float(threshold)] = {
                "accuracy": acc,
                "predictions": predicted.tolist(),
            }

        return threshold_results
