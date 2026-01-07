"""Model parity validation between float and quantized models.

Used in Phase 5 experiments to ensure quantization doesn't break predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ParityResult:
    """Results of parity validation between two models."""
    passed: bool
    max_abs_diff: float
    mean_abs_diff: float
    rank_correlation: float
    details: Dict


def compute_parity_metrics(
    float_outputs: np.ndarray,
    quantized_outputs: np.ndarray,
    tolerance: float = 0.1,
) -> ParityResult:
    """Compute parity metrics between float and quantized model outputs.
    
    Args:
        float_outputs: Outputs from float model (N,)
        quantized_outputs: Outputs from quantized model (N,)
        tolerance: Maximum allowed absolute difference
    
    Returns:
        ParityResult with pass/fail and metrics
    """
    float_outputs = np.asarray(float_outputs).flatten()
    quantized_outputs = np.asarray(quantized_outputs).flatten()
    
    if float_outputs.shape != quantized_outputs.shape:
        return ParityResult(
            passed=False,
            max_abs_diff=float('inf'),
            mean_abs_diff=float('inf'),
            rank_correlation=0.0,
            details={"error": "Shape mismatch", "float_shape": float_outputs.shape, "quant_shape": quantized_outputs.shape},
        )
    
    # Difference metrics
    abs_diff = np.abs(float_outputs - quantized_outputs)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))
    
    # Rank correlation
    from scipy.stats import spearmanr
    rank_corr, _ = spearmanr(float_outputs, quantized_outputs)
    
    passed = max_abs_diff <= tolerance and rank_corr >= 0.9
    
    return ParityResult(
        passed=passed,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rank_correlation=float(rank_corr),
        details={
            "tolerance": tolerance,
            "num_samples": len(float_outputs),
            "percentiles": {
                "p50": float(np.percentile(abs_diff, 50)),
                "p95": float(np.percentile(abs_diff, 95)),
                "p99": float(np.percentile(abs_diff, 99)),
            },
        },
    )


def validate_probability_parity(
    float_probs: np.ndarray,
    tflite_probs: np.ndarray,
    epsilon: float = 0.05,
) -> ParityResult:
    """Validate probability parity (E5.1).
    
    Args:
        float_probs: Probabilities from float model
        tflite_probs: Probabilities from TFLite model
        epsilon: Maximum allowed probability difference
    
    Returns:
        ParityResult
    """
    return compute_parity_metrics(float_probs, tflite_probs, tolerance=epsilon)


def validate_patch_logit_parity(
    float_patch_maps: np.ndarray,
    tflite_patch_maps: np.ndarray,
    epsilon: float = 0.1,
) -> ParityResult:
    """Validate patch-logit map parity (E5.2).
    
    Args:
        float_patch_maps: Patch logit maps from float model (N, H, W)
        tflite_patch_maps: Patch logit maps from TFLite model (N, H, W)
        epsilon: Maximum allowed difference
    
    Returns:
        ParityResult with additional argmax stability check
    """
    result = compute_parity_metrics(float_patch_maps, tflite_patch_maps, tolerance=epsilon)
    
    # Check argmax stability across images
    float_flat = float_patch_maps.reshape(float_patch_maps.shape[0], -1)
    tflite_flat = tflite_patch_maps.reshape(tflite_patch_maps.shape[0], -1)
    
    float_argmax = np.argmax(float_flat, axis=1)
    tflite_argmax = np.argmax(tflite_flat, axis=1)
    
    argmax_agreement = float(np.mean(float_argmax == tflite_argmax))
    
    result.details["argmax_agreement"] = argmax_agreement
    result.passed = result.passed and argmax_agreement >= 0.9
    
    return result


def validate_pooled_logit_parity(
    float_logits: np.ndarray,
    tflite_logits: np.ndarray,
    epsilon: float = 0.05,
) -> ParityResult:
    """Validate pooled logit parity (E5.3).
    
    Args:
        float_logits: Pooled logits from float model
        tflite_logits: Pooled logits from TFLite model
        epsilon: Maximum allowed difference
    
    Returns:
        ParityResult
    """
    return compute_parity_metrics(float_logits, tflite_logits, tolerance=epsilon)


def run_parity_suite(
    float_model_path: Path,
    tflite_model_path: Path,
    test_images: List[Path],
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, ParityResult]:
    """Run full parity test suite across all stages.
    
    Args:
        float_model_path: Path to float model
        tflite_model_path: Path to TFLite model
        test_images: List of test image paths
        tolerances: Dictionary of tolerances for each stage
    
    Returns:
        Dictionary mapping stage name to ParityResult
    """
    if tolerances is None:
        tolerances = {
            "probability": 0.05,
            "patch_logit": 0.1,
            "pooled_logit": 0.05,
        }
    
    # TODO: Implement actual model loading and inference
    # This requires:
    # 1. Load both models
    # 2. Run inference on test images
    # 3. Extract intermediate outputs (patch logits, pooled logits, probs)
    # 4. Compute parity for each stage
    
    raise NotImplementedError(
        "Parity suite requires actual model loading and inference. "
        "Use individual parity functions with precomputed outputs for now."
    )
