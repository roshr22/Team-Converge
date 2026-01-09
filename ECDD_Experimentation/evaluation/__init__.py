"""Evaluation package for ECDD model diagnostics.

This package provides tools for evaluating deepfake detection models
on large online datasets with metrics grouped by source and compression level.
"""

from .dataset_index import DatasetIndex, SampleEntry, load_index, save_index
from .metrics import (
    compute_binary_metrics,
    compute_confusion_matrix,
    compute_grouped_metrics,
    MetricsResult,
)
from .plot_diagnostics import (
    plot_confidence_histogram,
    plot_reliability_curve,
    plot_confusion_matrix,
)

__all__ = [
    "DatasetIndex",
    "SampleEntry",
    "load_index",
    "save_index",
    "compute_binary_metrics",
    "compute_confusion_matrix",
    "compute_grouped_metrics",
    "MetricsResult",
    "plot_confidence_histogram",
    "plot_reliability_curve",
    "plot_confusion_matrix",
]
