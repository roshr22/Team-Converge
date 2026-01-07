"""Calibration set contract utilities.

The experimentation spec requires:
- a deployment-like calibration set
- disjoint from training and golden test sets
- versioned artifacts

To keep this lightweight and runnable on partner laptop, we define a simple
JSONL/JSON schema for calibration logits:

Schema (JSON list):
[
  {
    "id": "unique_sample_id",
    "logit": 1.2345,
    "label": 0,
    "meta": {"source": "device/platform", "timestamp": "..."}
  },
  ...
]

This file can be produced by any inference runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class CalibrationSetInfo:
    path: str
    num_samples: int
    label_counts: Dict[str, int]


def load_calibration_json(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    data = json.loads(path.read_text())
    logits = np.array([row["logit"] for row in data], dtype=np.float64)
    labels = np.array([row["label"] for row in data], dtype=np.int64)
    return logits, labels, data


def describe_calibration_set(logits: np.ndarray, labels: np.ndarray, path: Path) -> CalibrationSetInfo:
    counts = {
        "0": int(np.sum(labels == 0)),
        "1": int(np.sum(labels == 1)),
    }
    return CalibrationSetInfo(path=str(path), num_samples=int(labels.size), label_counts=counts)
