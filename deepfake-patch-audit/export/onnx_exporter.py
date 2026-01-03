"""PyTorch to ONNX model export with validation."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple
import onnx
import onnxruntime as rt


class ONNXExporter:
    """Export PyTorch models to ONNX format with validation."""

    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Args:
            model: PyTorch model to export
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def export(
        self,
        output_path: str,
        input_shape: Tuple = (1, 3, 256, 256),
        opset_version: int = 12,
        verbose: bool = True,
    ) -> bool:
        """
        Export PyTorch model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            input_shape: Shape of dummy input (batch_size, channels, height, width)
            opset_version: ONNX opset version (12+ recommended)
            verbose: Print export details

        Returns:
            True if export successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create dummy input
            dummy_input = torch.randn(input_shape, device=self.device)

            if verbose:
                print("\n" + "=" * 80)
                print("PYTORCH TO ONNX EXPORT")
                print("=" * 80)
                print(f"Input shape: {input_shape}")
                print(f"Opset version: {opset_version}")

            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                str(output_path),
                input_names=["image_input"],
                output_names=["output"],
                dynamic_axes={
                    "image_input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=verbose,
            )

            if verbose:
                print(f"✓ Model exported to {output_path}")

            # Validate ONNX model
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)

            if verbose:
                print(f"✓ ONNX model validation passed")

            return True

        except Exception as e:
            print(f"✗ ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def verify_onnx_model(
        self, onnx_path: str, test_input_shape: Tuple = (1, 3, 256, 256)
    ) -> bool:
        """
        Verify ONNX model by running inference and comparing with PyTorch.

        Args:
            onnx_path: Path to ONNX model
            test_input_shape: Shape of test input

        Returns:
            True if outputs match, False otherwise
        """
        try:
            print("\n" + "=" * 80)
            print("ONNX MODEL VERIFICATION")
            print("=" * 80)

            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            # Create ONNX Runtime session
            sess = rt.InferenceSession(
                onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Generate test input
            test_input = torch.randn(test_input_shape, device=self.device)
            test_input_np = test_input.cpu().numpy()

            # PyTorch inference
            with torch.no_grad():
                pytorch_output = self.model(test_input)

            # ONNX inference
            input_name = sess.get_inputs()[0].name
            output_name = sess.get_outputs()[0].name
            onnx_output = sess.run(
                [output_name], {input_name: test_input_np}
            )[0]

            # Compare outputs
            pytorch_np = pytorch_output.cpu().numpy()
            max_diff = abs(pytorch_np - onnx_output).max()
            mean_diff = abs(pytorch_np - onnx_output).mean()

            if max_diff < 1e-4:
                print(f"✓ Outputs match!")
                print(f"  Max difference: {max_diff:.6e}")
                print(f"  Mean difference: {mean_diff:.6e}")
                return True
            else:
                print(f"⚠ Outputs differ!")
                print(f"  Max difference: {max_diff:.6e}")
                print(f"  Mean difference: {mean_diff:.6e}")
                return False

        except Exception as e:
            print(f"✗ ONNX verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_onnx_model_info(self, onnx_path: str) -> dict:
        """Get information about ONNX model."""
        try:
            onnx_model = onnx.load(onnx_path)

            graph = onnx_model.graph

            info = {
                "inputs": [],
                "outputs": [],
                "num_nodes": len(graph.node),
                "num_initializers": len(graph.initializer),
            }

            # Input information
            for input_tensor in graph.input:
                shape = [
                    dim.dim_value if dim.dim_value != 0 else "dynamic"
                    for dim in input_tensor.type.tensor_type.shape.dim
                ]
                info["inputs"].append(
                    {"name": input_tensor.name, "shape": shape}
                )

            # Output information
            for output_tensor in graph.output:
                shape = [
                    dim.dim_value if dim.dim_value != 0 else "dynamic"
                    for dim in output_tensor.type.tensor_type.shape.dim
                ]
                info["outputs"].append(
                    {"name": output_tensor.name, "shape": shape}
                )

            return info

        except Exception as e:
            print(f"✗ Failed to get ONNX model info: {e}")
            return None
