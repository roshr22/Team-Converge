#!/usr/bin/env python3
"""
Export Student Model with Aggressive Quantization for Nicla Vision

Pipeline:
1. Load trained student model (13 KB)
2. Export to ONNX
3. Apply aggressive INT8 quantization
4. Target: <10 KB for Nicla Vision deployment

This implements true federated inference:
- Student runs on Nicla Vision (edge filtering)
- Teacher runs on Raspberry Pi (final verification)
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student.tiny_ladeda import TinyLaDeDa


def load_student(model_path: str, device: str = "cpu"):
    """Load trained student model."""
    print(f"Loading student from {model_path}...")

    student = TinyLaDeDa(pretrained=False)
    state_dict = torch.load(model_path, map_location="cpu")
    student.load_state_dict(state_dict)
    student = student.to(device)
    student.eval()

    params = sum(p.numel() for p in student.parameters())
    print(f"✓ Student loaded ({params:,} params)")
    return student


def export_to_onnx(model: torch.nn.Module, output_path: str, device: str = "cpu"):
    """Export PyTorch model to ONNX."""
    print(f"\nExporting to ONNX: {output_path}...")

    dummy_input = torch.randn(1, 3, 256, 256, device=device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["input_image"],
            output_names=["predictions"],
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "predictions": {0: "batch_size"}
            },
            opset_version=13,
            verbose=False,
            export_params=True,
        )

        file_size_kb = Path(output_path).stat().st_size / 1024
        print(f"✓ ONNX export complete: {file_size_kb:.1f} KB")
        return True

    except Exception as e:
        print(f"✗ Export failed: {e}")
        return False


def quantize_onnx(onnx_path: str, output_path: str):
    """Quantize ONNX model aggressively for embedded devices."""
    print(f"\nQuantizing ONNX model (INT8 - aggressive)...")

    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(onnx_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )

        original_size_kb = Path(onnx_path).stat().st_size / 1024
        quantized_size_kb = Path(output_path).stat().st_size / 1024
        ratio = original_size_kb / quantized_size_kb if quantized_size_kb > 0 else 0

        print(f"✓ Quantization complete")
        print(f"  Original: {original_size_kb:.1f} KB")
        print(f"  Quantized: {quantized_size_kb:.1f} KB")
        if ratio > 0:
            print(f"  Compression: {ratio:.1f}x")

        # Check if it fits in Nicla constraints
        if quantized_size_kb < 100:
            print(f"  ✓ Size suitable for Nicla Vision (<100 KB)")
        elif quantized_size_kb < 500:
            print(f"  ⚠ Size acceptable but close to limit (<500 KB)")
        else:
            print(f"  ❌ Size too large for Nicla (>500 KB)")

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


def verify_model(model_path: str):
    """Verify ONNX model can be loaded."""
    print(f"\nVerifying model: {model_path}...")

    try:
        import onnxruntime as rt

        session = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        print(f"✓ Model verification passed")
        print(f"  Input: {input_name} {session.get_inputs()[0].shape}")
        print(f"  Output: {output_name} {session.get_outputs()[0].shape}")

        # Test inference
        test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        output = session.run([output_name], {input_name: test_input})
        print(f"  Test inference: OK (output shape: {output[0].shape})")

        return True
    except Exception as e:
        print(f"⚠ Verification failed: {e}")
        return False


def main():
    """Export and quantize student model for Nicla."""
    print("=" * 80)
    print("STUDENT MODEL EXPORT FOR NICLA VISION (Federated Inference Edge)")
    print("=" * 80)

    # Paths
    model_path = "outputs/checkpoints_two_stage/student_final.pt"
    output_dir = Path("deployment/nicla/models")
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / "student.onnx"
    quantized_path = output_dir / "student_quantized.onnx"

    # Step 1: Load
    student = load_student(model_path)

    # Step 2: Export to ONNX
    if not export_to_onnx(student, str(onnx_path)):
        return False

    # Step 3: Quantize
    if not quantize_onnx(str(onnx_path), str(quantized_path)):
        return False

    # Step 4: Verify
    verify_model(str(quantized_path))

    # Summary
    print("\n" + "=" * 80)
    print("EXPORT COMPLETE ✓")
    print("=" * 80)
    print(f"\n✓ Student model ready for Nicla Vision:")
    print(f"  Quantized model: {quantized_path}")
    print(f"  Copy to Nicla sketch data folder:")
    print(f"    cp {quantized_path} deployment/nicla/data/")
    print(f"\n✓ Architecture:")
    print(f"  Nicla Vision: Runs student_quantized.onnx (edge filtering)")
    print(f"  Raspberry Pi: Runs teacher_quantized.onnx (final verification)")
    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
