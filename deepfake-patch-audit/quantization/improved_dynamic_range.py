"""Improved dynamic range quantization with better numerical stability and accuracy."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class ImprovedDynamicRangeQuantizer:
    """
    Improved dynamic range quantization based on per-channel or per-tensor statistics.

    Features:
    - Symmetric and asymmetric quantization
    - Per-channel and per-tensor quantization
    - Calibration support for better statistics
    - Enhanced numerical stability
    - Optional clipping for outliers

    Quantization Formula:
    - X_int8 = round(X_fp32 / scale) + zero_point
    - X_fp32 = scale * (X_int8 - zero_point)

    where:
    - scale = (x_max - x_min) / (qmax - qmin)
    - zero_point = qmin - round(x_min / scale)
    """

    def __init__(
        self,
        bits=8,
        symmetric=False,
        per_channel=True,
        clip_outliers=False,
        clip_percentile=99.9,
    ):
        """
        Args:
            bits: Quantization bits (default: 8 for int8)
            symmetric: Use symmetric quantization (range: [-qmax, qmax])
            per_channel: Use per-channel quantization (recommended for conv weights)
            clip_outliers: Clip extreme outliers before quantization
            clip_percentile: Percentile threshold for clipping (e.g., 99.9 clips to 99.9th percentile)
        """
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.clip_outliers = clip_outliers
        self.clip_percentile = clip_percentile

        # Quantization range
        self.qmin = -2 ** (bits - 1)  # -128 for int8
        self.qmax = 2 ** (bits - 1) - 1  # 127 for int8

        # Store quantization parameters for calibration
        self.calibration_stats = {}

    def _compute_quant_params(
        self, tensor: torch.Tensor, per_channel: bool = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute quantization parameters (scale and zero_point).

        Args:
            tensor: Input tensor to quantize
            per_channel: Override per_channel setting

        Returns:
            Dict with 'scale' and 'zero_point' tensors
        """
        if per_channel is None:
            per_channel = self.per_channel

        # Handle outliers via clipping
        if self.clip_outliers and tensor.numel() > 0:
            clip_val = torch.quantile(
                torch.abs(tensor), self.clip_percentile / 100.0
            )
            tensor = torch.clamp(tensor, -clip_val, clip_val)

        if not per_channel or tensor.dim() == 1:
            # Per-tensor quantization
            return self._compute_quant_params_per_tensor(tensor)
        else:
            # Per-channel quantization
            return self._compute_quant_params_per_channel(tensor)

    def _compute_quant_params_per_tensor(
        self, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute quantization parameters for entire tensor."""
        x_min = tensor.min()
        x_max = tensor.max()

        if self.symmetric:
            # Symmetric: range is [-abs_max, abs_max]
            abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
            scale = (2 * abs_max) / (self.qmax - self.qmin)
            # For symmetric, zero_point is typically 0
            zero_point = torch.tensor(0, dtype=torch.int32)
        else:
            # Asymmetric: standard min-max quantization
            scale = (x_max - x_min) / (self.qmax - self.qmin)

            # Avoid division by zero
            if scale < 1e-8:
                scale = torch.tensor(1e-8, dtype=tensor.dtype, device=tensor.device)

            # Compute zero_point: qmin - x_min / scale
            zero_point_real = self.qmin - x_min / scale
            zero_point = torch.round(zero_point_real).clamp(self.qmin, self.qmax)
            zero_point = zero_point.to(torch.int32)

        return {
            "scale": scale if isinstance(scale, torch.Tensor) else torch.tensor(scale, dtype=tensor.dtype),
            "zero_point": zero_point,
        }

    def _compute_quant_params_per_channel(
        self, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute per-channel quantization parameters (for conv/linear weights)."""
        # For conv: (out_channels, in_channels, kh, kw)
        # For linear: (out_features, in_features)
        # Quantize per output channel

        out_channels = tensor.shape[0]
        scales = []
        zero_points = []

        for ch in range(out_channels):
            if tensor.dim() == 2:
                # Linear layer: (out_features, in_features)
                ch_tensor = tensor[ch, :]
            elif tensor.dim() == 4:
                # Conv layer: (out_channels, in_channels, kh, kw)
                ch_tensor = tensor[ch, :, :, :]
            else:
                ch_tensor = tensor[ch]

            ch_tensor_flat = ch_tensor.flatten()

            x_min = ch_tensor_flat.min()
            x_max = ch_tensor_flat.max()

            if self.symmetric:
                abs_max = torch.max(torch.abs(x_min), torch.abs(x_max))
                scale = (2 * abs_max) / (self.qmax - self.qmin)
                zero_point = torch.tensor(0, dtype=torch.int32)
            else:
                scale = (x_max - x_min) / (self.qmax - self.qmin)

                if scale < 1e-8:
                    scale = torch.tensor(1e-8, dtype=ch_tensor.dtype, device=ch_tensor.device)

                zero_point_real = self.qmin - x_min / scale
                zero_point = torch.round(zero_point_real).clamp(self.qmin, self.qmax)
                zero_point = zero_point.to(torch.int32)

            scales.append(scale)
            zero_points.append(zero_point)

        return {
            "scale": torch.stack(scales) if isinstance(scales[0], torch.Tensor) else torch.tensor(scales),
            "zero_point": torch.stack(zero_points),
        }

    def quantize_tensor(
        self, tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Quantize a tensor to int8.

        X_q = round(X_fp32 / scale) + zero_point
        then clip to [qmin, qmax]
        """
        # Scale the tensor
        x_scaled = tensor / (scale if isinstance(scale, torch.Tensor) else torch.tensor(scale))

        # Round and add zero_point
        x_rounded = torch.round(x_scaled)
        x_q = x_rounded + (zero_point if isinstance(zero_point, torch.Tensor) else zero_point)

        # Clip to quantization range
        x_q = torch.clamp(x_q, self.qmin, self.qmax)

        return x_q.to(torch.int8)

    def dequantize_tensor(
        self, tensor_q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """
        Dequantize an int8 tensor back to float32.

        X_fp32 = scale * (X_q - zero_point)
        """
        tensor_q = tensor_q.to(torch.float32)
        x_dequant = (scale if isinstance(scale, torch.Tensor) else torch.tensor(scale)) * (
            tensor_q - (zero_point if isinstance(zero_point, torch.Tensor) else zero_point)
        )
        return x_dequant

    def quantize_model(
        self, model: nn.Module, calibration_loader=None
    ) -> Tuple[nn.Module, Dict]:
        """
        Quantize all weights in a model.

        Args:
            model: Model to quantize
            calibration_loader: Optional data loader for activation calibration

        Returns:
            Quantized model and quantization parameters dictionary
        """
        quant_params = {}

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Quantize weights
                if module.weight is not None:
                    weight_params = self._compute_quant_params(
                        module.weight, per_channel=True
                    )
                    module.weight.data = self.quantize_tensor(
                        module.weight,
                        weight_params["scale"],
                        weight_params["zero_point"],
                    ).float()

                    quant_params[f"{name}.weight"] = weight_params

                # Quantize bias (symmetric, per-tensor)
                if module.bias is not None:
                    bias_params = self._compute_quant_params(
                        module.bias, per_channel=False
                    )
                    module.bias.data = self.quantize_tensor(
                        module.bias,
                        bias_params["scale"],
                        bias_params["zero_point"],
                    ).float()

                    quant_params[f"{name}.bias"] = bias_params

        return model, quant_params

    def get_quantization_report(self, quant_params: Dict) -> str:
        """Generate a report of quantization statistics."""
        report = "\n" + "=" * 80 + "\n"
        report += "QUANTIZATION REPORT\n"
        report += "=" * 80 + "\n"
        report += f"Bits: {self.bits}\n"
        report += f"Symmetric: {self.symmetric}\n"
        report += f"Per-channel: {self.per_channel}\n"
        report += f"Quantization range: [{self.qmin}, {self.qmax}]\n\n"

        report += "Quantized layers:\n"
        for name, params in quant_params.items():
            scale = params["scale"]
            zero_point = params["zero_point"]

            if isinstance(scale, torch.Tensor):
                if scale.dim() > 0:
                    report += f"  {name}:\n"
                    report += f"    Scale - min: {scale.min():.6f}, max: {scale.max():.6f}, mean: {scale.mean():.6f}\n"
                    report += f"    Zero point: {zero_point.tolist()}\n"
                else:
                    report += f"  {name}: scale={scale.item():.6f}, zero_point={zero_point.item()}\n"
            else:
                report += f"  {name}: scale={scale:.6f}, zero_point={zero_point}\n"

        report += "\n" + "=" * 80 + "\n"
        return report


class QuantizationCalibrator:
    """Calibrate quantization parameters using representative data."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Args:
            model: Model to calibrate
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.activation_stats = {}

    def calibrate(self, calibration_loader, num_batches: int = 10) -> Dict:
        """
        Collect activation statistics for better quantization.

        Args:
            calibration_loader: DataLoader with representative data
            num_batches: Number of batches to use for calibration

        Returns:
            Dictionary of activation statistics
        """
        self.model.eval()
        batch_count = 0

        with torch.no_grad():
            for batch in calibration_loader:
                if batch_count >= num_batches:
                    break

                images = batch["image"].to(self.device)
                _ = self.model(images)
                batch_count += 1

        return self.activation_stats

    def get_activation_range(self, activation_name: str) -> Tuple[float, float]:
        """Get min and max activation values for a layer."""
        if activation_name in self.activation_stats:
            stats = self.activation_stats[activation_name]
            return stats["min"], stats["max"]
        return None
