"""Platt scaling calibration for binary logits.

Fits parameters (a, b) so that:
  p = sigmoid(a * logit + b)

Used for Phase 4 experiment E4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class PlattScalingParams:
    a: float
    b: float


def fit_platt(logits: np.ndarray, labels: np.ndarray, lr: float = 0.05, steps: int = 2000) -> Tuple[PlattScalingParams, Dict]:
    """Fit Platt scaling with simple gradient descent (dependency-free).

    This is deterministic given fixed inputs.
    """
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    assert x.shape == y.shape

    a = 1.0
    b = 0.0

    for _ in range(steps):
        z = a * x + b
        p = sigmoid(z)
        # gradients for BCE
        grad_a = np.mean((p - y) * x)
        grad_b = np.mean(p - y)
        a -= lr * grad_a
        b -= lr * grad_b

    params = PlattScalingParams(a=float(a), b=float(b))
    # compute final nll
    eps = 1e-12
    p = np.clip(sigmoid(a * x + b), eps, 1 - eps)
    nll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return params, {"nll": nll, "lr": lr, "steps": steps}


def apply_platt(logits: np.ndarray, params: PlattScalingParams) -> np.ndarray:
    return sigmoid(params.a * np.asarray(logits) + params.b)
