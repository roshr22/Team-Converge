"""Calibration utilities (temperature scaling, Platt scaling, operating point selection)."""

from .temperature_scaling import TemperatureScalingParams, fit_temperature, apply_temperature, expected_calibration_error
from .platt_scaling import PlattScalingParams, fit_platt, apply_platt
from .operating_point import OperatingPoint, select_threshold_at_fpr, compute_fpr_tpr

__all__ = [
    "TemperatureScalingParams",
    "fit_temperature",
    "apply_temperature",
    "expected_calibration_error",
    "PlattScalingParams",
    "fit_platt",
    "apply_platt",
    "OperatingPoint",
    "select_threshold_at_fpr",
    "compute_fpr_tpr",
]
