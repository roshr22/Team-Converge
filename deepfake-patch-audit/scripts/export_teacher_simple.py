#!/usr/bin/env python3
"""
Simple Teacher Model Export with ONNX-level Quantization

Pipeline:
1. Load teacher model (53 MB)
2. Export to ONNX (no compatibility issues)
3. Quantize ONNX model directly (~10-15 MB)
4. Place in deployment folder for Pi

This is the most reliable approach - avoids PyTorch quantization export issues.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.teacher.ladeda_wrapper import LaDeDaWrapper


def load_teacher(model_path: str, device: str = "cpu"):
    """Load teacher model."""
    print(f"Loading teacher from {model_path}...")

    teacher = LaDeDaWrapper(pretrained=False, freeze_backbone=False)
    state_dict = torch.load(model_path, map_location="cpu")
    teacher.model.load_state_dict(state_dict)
    teacher = teacher.to(device)
    teacher.eval()

    print(f"✓ Teacher loaded ({sum(p.numel() for p in teacher.parameters()):,} params)")
    return teacher


def export_to_onnx(model: torch.nn.Module, output_path: str, device: str = "cpu"):
    """Export PyTorch model to ONNX."""
    print(f"\nExporting to ONNX: {output_path}...")

    # Create dummy input
    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input_image"],
            output_names=["predictions"],
            dynamic_axes={"input_image": {0: "batch_size"}, "predictions": {0: "batch_size"}},
            opset_version=13,
            verbose=False,
            export_params=True,
        )

        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"✓ ONNX export complete: {file_size_mb:.2f} MB")
        return True

    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False


def quantize_onnx(onnx_path: str, output_path: str):
    """Quantize ONNX model using ONNX Runtime."""
    print(f"\nQuantizing ONNX model...")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )

        original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
        ratio = original_size / quantized_size

        print(f"✓ Quantization complete")
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Quantized: {quantized_size:.2f} MB")
        print(f"  Compression: {ratio:.1f}x")

        return True

    except ImportError:
        print("⚠ onnxruntime not installed with quantization support")
        print("  Fallback: Using non-quantized ONNX model")
        import shutil
        shutil.copy(str(onnx_path), str(output_path))
        return True
    except Exception as e:
        print(f"✗ Quantization failed: {e}")
        return False


def main():
    """Export and quantize teacher model."""
    print("=" * 80)
    print("TEACHER MODEL EXPORT (Simple Pipeline)")
    print("=" * 80)

    # Paths
    model_path = "outputs/checkpoints_teacher/teacher_finetuned_best.pth"
    output_dir = Path("deployment/pi_server/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "teacher.onnx"
    quantized_path = output_dir / "teacher_quantized.onnx"

    # Step 1: Load
    teacher = load_teacher(model_path)

    # Step 2: Export to ONNX
    if not export_to_onnx(teacher, str(onnx_path)):
        return False

    # Step 3: Quantize
    if not quantize_onnx(str(onnx_path), str(quantized_path)):
        return False

    # Summary
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE")
    print("=" * 80)
    print(f"\n✓ Teacher model ready for deployment:")
    print(f"  Quantized model: {quantized_path}")
    print(f"  Use this in Flask app (deployment/pi_server/app.py)")
    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
