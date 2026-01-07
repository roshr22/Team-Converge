"""Temperature scaling calibration.

Fits a single scalar temperature T on logits to minimize NLL.
This module is used for Phase 4 experiments E4.2 and policy_contract calibration.method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class TemperatureScalingParams:
    temperature: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def nll_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    # Binary cross-entropy on logits
    # Stable implementation
    x = logits
    y = labels
    # log(1+exp(x)) - y*x
    return float(np.mean(np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) - y * x))


def fit_temperature(logits: np.ndarray, labels: np.ndarray, 
                    t_min: float = 0.05, t_max: float = 10.0, steps: int = 200) -> Tuple[TemperatureScalingParams, Dict]:
    """Fit temperature via grid search (robust, dependency-free).

    For research/engineering reproducibility, a deterministic grid search is acceptable.
    If needed later, we can swap to LBFGS.
    """
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    assert logits.shape == labels.shape

    best_t = 1.0
    best_nll = float("inf")

    grid = np.linspace(t_min, t_max, steps)
    for t in grid:
        scaled = logits / t
        loss = nll_from_logits(scaled, labels)
        if loss < best_nll:
            best_nll = loss
            best_t = float(t)

    params = TemperatureScalingParams(temperature=best_t)
    details = {"best_nll": best_nll, "t_min": t_min, "t_max": t_max, "steps": steps}
    return params, details


def apply_temperature(logits: np.ndarray, params: TemperatureScalingParams) -> np.ndarray:
    return np.asarray(logits) / float(params.temperature)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute ECE for binary classification."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(labels[mask]))
        conf = float(np.mean(probs[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(acc - conf)
    return float(ece)
