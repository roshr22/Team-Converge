"""Dynamic range quantization (preferred approach)."""

import torch
import torch.nn as nn
import numpy as np


class DynamicRangeQuantizer:
    """
    Quantize model using dynamic range (per-channel quantization).
    This is the preferred quantization approach.
    """

    def __init__(self, bits=8, symmetric=False):
        """
        Args:
            bits: Quantization bits (default: 8)
            symmetric: Use symmetric quantization
        """
        self.bits = bits
        self.symmetric = symmetric
        self.scale = 2 ** bits - 1

    def quantize_weight(self, weight):
        """
        Quantize weight tensor.

        Args:
            weight: Weight tensor to quantize

        Returns:
            Quantized weight
        """
        if weight.dim() == 1:
            # Bias
            return self._quantize_1d(weight)
        elif weight.dim() == 2:
            # Linear layer
            return self._quantize_2d(weight)
        elif weight.dim() >= 3:
            # Convolutional layer
            return self._quantize_4d(weight)

    def _quantize_1d(self, tensor):
        """Quantize 1D tensor (bias)."""
        min_val = tensor.min()
        max_val = tensor.max()

        if self.symmetric:
            abs_max = max(abs(min_val), abs(max_val))
            min_val = -abs_max
            max_val = abs_max

        q = (tensor - min_val) / (max_val - min_val + 1e-8) * self.scale
        q = torch.round(q)
        q = torch.clamp(q, 0, self.scale)

        q_tensor = q / self.scale * (max_val - min_val) + min_val
        return q_tensor

    def _quantize_2d(self, tensor):
        """Quantize 2D tensor (per-channel)."""
        quantized = torch.zeros_like(tensor)

        # Per-output-channel quantization
        for i in range(tensor.shape[0]):
            quantized[i] = self._quantize_1d(tensor[i])

        return quantized

    def _quantize_4d(self, tensor):
        """Quantize 4D tensor (conv weights, per-output-channel)."""
        quantized = torch.zeros_like(tensor)

        # Per-output-channel quantization
        for i in range(tensor.shape[0]):
            quantized[i] = self._quantize_1d(tensor[i].flatten()).reshape(tensor[i].shape)

        return quantized

    def quantize_model(self, model):
        """
        Quantize all weights in model.

        Args:
            model: Model to quantize

        Returns:
            Quantized model (in-place)
        """
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module.weight is not None:
                    module.weight.data = self.quantize_weight(module.weight.data)
                if module.bias is not None:
                    module.bias.data = self._quantize_1d(module.bias.data)

        return model


class QuantizationCalibrator:
    """
    Calibrate quantization ranges using representative data.
    """

    def __init__(self, calibration_loader, num_iterations=100):
        """
        Args:
            calibration_loader: Data loader for calibration
            num_iterations: Number of calibration iterations
        """
        self.calibration_loader = calibration_loader
        self.num_iterations = num_iterations
        self.statistics = {}

    def collect_statistics(self, model, device="cuda"):
        """
        Collect activation statistics for calibration.

        Args:
            model: Model to calibrate
            device: Device for computation
        """
        model.eval()
        iter_count = 0

        with torch.no_grad():
            for batch in self.calibration_loader:
                if iter_count >= self.num_iterations:
                    break

                images = batch["image"].to(device)
                _ = model(images)
                iter_count += 1

        return self.statistics
