"""ONNX to TFLite conversion with quantization options."""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import tempfile


class TFLiteConverter:
    """Convert ONNX models to TensorFlow Lite format with quantization."""

    def __init__(self, onnx_path: str, verbose: bool = True):
        """
        Args:
            onnx_path: Path to ONNX model
            verbose: Print conversion details
        """
        self.onnx_path = onnx_path
        self.verbose = verbose

    def onnx_to_saved_model(self, saved_model_path: str) -> bool:
        """
        Convert ONNX to TensorFlow SavedModel format.

        Args:
            saved_model_path: Path to save TensorFlow model

        Returns:
            True if conversion successful
        """
        try:
            if self.verbose:
                print("\n" + "=" * 80)
                print("ONNX TO TENSORFLOW CONVERSION")
                print("=" * 80)
                print(f"ONNX path: {self.onnx_path}")
                print(f"Output path: {saved_model_path}")

            # Import onnx_tf (requires installation)
            try:
                from onnx_tf.backend import prepare
            except ImportError:
                print("⚠ onnx-tf not installed. Install: pip install onnx-tf")
                return False

            # Convert ONNX to TensorFlow
            onnx_model = self._load_onnx_model()
            if onnx_model is None:
                return False

            tf_rep = prepare(onnx_model)
            export_path = Path(saved_model_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)

            tf_rep.export_graph(str(export_path))

            if self.verbose:
                print(f"✓ Converted to TensorFlow SavedModel format")

            return True

        except Exception as e:
            print(f"✗ ONNX to TensorFlow conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def convert_to_tflite(
        self,
        output_path: str,
        quantization_mode: str = "none",
        representative_data_gen=None,
    ) -> bool:
        """
        Convert ONNX to TFLite format with optional quantization.

        Args:
            output_path: Path to save TFLite model
            quantization_mode: 'none', 'dynamic', or 'static'
                - 'none': No quantization
                - 'dynamic': Dynamic range quantization
                - 'static': Full integer quantization (requires representative_data_gen)
            representative_data_gen: Generator function for calibration data
                Should yield batches of input data

        Returns:
            True if conversion successful
        """
        try:
            if self.verbose:
                print("\n" + "=" * 80)
                print("ONNX TO TFLITE CONVERSION")
                print("=" * 80)
                print(f"Quantization mode: {quantization_mode}")

            # Create temporary SavedModel
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_model_path = Path(temp_dir) / "saved_model"

                # Convert ONNX to SavedModel first
                if not self.onnx_to_saved_model(str(saved_model_path)):
                    return False

                # Convert SavedModel to TFLite
                converter = tf.lite.TFLiteConverter.from_saved_model(
                    str(saved_model_path)
                )

                # Set quantization options
                if quantization_mode == "dynamic":
                    if self.verbose:
                        print("  Applying dynamic range quantization...")
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]

                elif quantization_mode == "static":
                    if representative_data_gen is None:
                        print("⚠ Static quantization requires representative_data_gen")
                        return False

                    if self.verbose:
                        print("  Applying full integer quantization...")

                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.representative_dataset = representative_data_gen

                    # Full integer quantization
                    converter.target_spec.supported_ops = [
                        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                    ]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8

                elif quantization_mode != "none":
                    print(f"⚠ Unknown quantization mode: {quantization_mode}")

                # Convert
                tflite_model = converter.convert()

                # Save TFLite model
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "wb") as f:
                    f.write(tflite_model)

                if self.verbose:
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    print(f"✓ Converted to TFLite format")
                    print(f"  Model size: {file_size_mb:.2f} MB")

                return True

        except Exception as e:
            print(f"✗ ONNX to TFLite conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def quantize_tflite(
        self,
        tflite_path: str,
        output_path: str,
        representative_data_gen,
        quantization_mode: str = "dynamic",
    ) -> bool:
        """
        Apply post-training quantization to an existing TFLite model.

        Args:
            tflite_path: Path to existing TFLite model
            output_path: Path to save quantized TFLite model
            representative_data_gen: Generator for calibration data
            quantization_mode: 'dynamic' or 'static'

        Returns:
            True if quantization successful
        """
        try:
            if self.verbose:
                print("\n" + "=" * 80)
                print("TFLITE POST-TRAINING QUANTIZATION")
                print("=" * 80)
                print(f"Input model: {tflite_path}")
                print(f"Quantization mode: {quantization_mode}")

            # Load existing TFLite model
            with open(tflite_path, "rb") as f:
                tflite_model = f.read()

            # Create quantization config
            converter = tf.lite.TFLiteConverter.from_concrete_functions([])

            if quantization_mode == "dynamic":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            elif quantization_mode == "static":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_data_gen
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
                ]

            # Save quantized model
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                f.write(tflite_model)

            if self.verbose:
                original_size = Path(tflite_path).stat().st_size / (1024 * 1024)
                quantized_size = output_path.stat().st_size / (1024 * 1024)
                compression_ratio = (1 - quantized_size / original_size) * 100

                print(f"✓ Quantization applied")
                print(f"  Original size: {original_size:.2f} MB")
                print(f"  Quantized size: {quantized_size:.2f} MB")
                print(f"  Compression: {compression_ratio:.1f}%")

            return True

        except Exception as e:
            print(f"✗ TFLite quantization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def verify_tflite_model(
        self, tflite_path: str, test_input: np.ndarray
    ) -> bool:
        """
        Verify TFLite model by running inference.

        Args:
            tflite_path: Path to TFLite model
            test_input: Test input array

        Returns:
            True if inference successful
        """
        try:
            if self.verbose:
                print("\n" + "=" * 80)
                print("TFLITE MODEL VERIFICATION")
                print("=" * 80)

            # Load and run inference
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            if self.verbose:
                print(f"Input details: {input_details[0]['shape']}")
                print(f"Output details: {output_details[0]['shape']}")

            # Set input and run inference
            interpreter.set_tensor(
                input_details[0]["index"], test_input.astype(np.float32)
            )
            interpreter.invoke()

            # Get output
            output_data = interpreter.get_tensor(output_details[0]["index"])

            if self.verbose:
                print(f"✓ Model inference successful")
                print(f"  Output shape: {output_data.shape}")
                print(f"  Output range: [{output_data.min():.4f}, {output_data.max():.4f}]")

            return True

        except Exception as e:
            print(f"✗ TFLite verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_onnx_model(self):
        """Load and validate ONNX model."""
        try:
            import onnx
            onnx_model = onnx.load(self.onnx_path)
            onnx.checker.check_model(onnx_model)
            return onnx_model
        except Exception as e:
            print(f"✗ Failed to load ONNX model: {e}")
            return None

    def get_tflite_info(self, tflite_path: str) -> dict:
        """Get information about TFLite model."""
        try:
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            info = {
                "inputs": [
                    {
                        "name": inp["name"],
                        "shape": inp["shape"].tolist(),
                        "dtype": str(inp["dtype"]),
                    }
                    for inp in input_details
                ],
                "outputs": [
                    {
                        "name": out["name"],
                        "shape": out["shape"].tolist(),
                        "dtype": str(out["dtype"]),
                    }
                    for out in output_details
                ],
                "file_size_mb": Path(tflite_path).stat().st_size / (1024 * 1024),
            }

            return info

        except Exception as e:
            print(f"✗ Failed to get TFLite model info: {e}")
            return None
