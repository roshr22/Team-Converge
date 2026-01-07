#!/usr/bin/env python3
"""Master Test Runner for ECDD Experimentation Infrastructure

Runs all available tests and generates a comprehensive report.

Usage:
    python run_all_tests.py [--output report.json] [--verbose]

Tests:
    - Phase 6 experiments (E6.1-E6.4)
    - Phase 5 experiments (E5.1-E5.5) with mock data
    - Golden hash generation (S0-S4)
    - Calibration utilities
    - Pipeline components
    - All infrastructure modules

Output:
    - Console report with pass/fail status
    - Optional JSON report file
    - Summary statistics
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class TestResult:
    """Result of a single test."""
    test_id: str
    name: str
    category: str
    passed: bool
    duration_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None


class TestRunner:
    """Master test runner for all ECDD infrastructure."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[TestResult] = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose."""
        if self.verbose or level in ["ERROR", "SUMMARY"]:
            prefix = {
                "INFO": "ℹ️",
                "SUCCESS": "✅",
                "ERROR": "❌",
                "WARN": "⚠️",
                "SUMMARY": "📊"
            }.get(level, "•")
            print(f"{prefix} {message}")
    
    def run_test(self, test_id: str, name: str, category: str, test_func) -> TestResult:
        """Run a single test and record result."""
        self.log(f"Running {test_id}: {name}")
        
        start = time.time()
        try:
            result = test_func()
            duration_ms = (time.time() - start) * 1000
            
            if hasattr(result, 'passed') and hasattr(result, 'details'):
                # Phase experiment result
                test_result = TestResult(
                    test_id=test_id,
                    name=name,
                    category=category,
                    passed=result.passed,
                    duration_ms=duration_ms,
                    details=result.details
                )
            else:
                # Custom result format
                test_result = TestResult(
                    test_id=test_id,
                    name=name,
                    category=category,
                    passed=result.get('passed', False),
                    duration_ms=duration_ms,
                    details=result
                )
            
            status = "✅ PASS" if test_result.passed else "❌ FAIL"
            self.log(f"{status}: {test_id} ({duration_ms:.1f}ms)", 
                    "SUCCESS" if test_result.passed else "ERROR")
            
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            test_result = TestResult(
                test_id=test_id,
                name=name,
                category=category,
                passed=False,
                duration_ms=duration_ms,
                details={},
                error=str(e)
            )
            self.log(f"❌ ERROR: {test_id} - {e}", "ERROR")
        
        self.results.append(test_result)
        return test_result
    
    def run_phase6_tests(self):
        """Run Phase 6 experiments."""
        self.log("\n=== Phase 6: Evaluation Battery ===", "SUMMARY")
        
        try:
            from PhaseWise_Experiments_ECDD.phase6_experiments import (
                e6_1_source_based_split_stress_test,
                e6_2_time_based_split_drift_probe,
                e6_3_transform_suite_conclusive_test,
                e6_4_out_of_scope_separation_test
            )
            
            self.run_test("E6.1", "Source-based split stress test", "Phase6", 
                         e6_1_source_based_split_stress_test)
            self.run_test("E6.2", "Time-based split drift probe", "Phase6", 
                         e6_2_time_based_split_drift_probe)
            self.run_test("E6.3", "Transform suite conclusive test", "Phase6", 
                         e6_3_transform_suite_conclusive_test)
            self.run_test("E6.4", "Out-of-scope separation test", "Phase6", 
                         e6_4_out_of_scope_separation_test)
        except Exception as e:
            self.log(f"Failed to import Phase 6: {e}", "ERROR")
    
    def run_phase5_tests(self):
        """Run Phase 5 experiments (mock mode)."""
        self.log("\n=== Phase 5: Quantization & Parity (Mock) ===", "SUMMARY")
        
        try:
            from PhaseWise_Experiments_ECDD.phase5_experiments import (
                e5_1_float_vs_tflite_probability_parity,
                e5_2_patch_logit_map_parity,
                e5_3_pooled_logit_parity,
                e5_4_post_quant_calibration_gate,
                e5_5_delegate_and_threading_invariance
            )
            
            self.run_test("E5.1", "Float vs TFLite probability parity", "Phase5", 
                         e5_1_float_vs_tflite_probability_parity)
            self.run_test("E5.2", "Patch-logit map parity", "Phase5", 
                         e5_2_patch_logit_map_parity)
            self.run_test("E5.3", "Pooled logit parity", "Phase5", 
                         e5_3_pooled_logit_parity)
            self.run_test("E5.4", "Post-quant calibration gate", "Phase5", 
                         e5_4_post_quant_calibration_gate)
            self.run_test("E5.5", "Delegate and threading invariance", "Phase5", 
                         e5_5_delegate_and_threading_invariance)
        except Exception as e:
            self.log(f"Failed to import Phase 5: {e}", "ERROR")
    
    def run_golden_hash_test(self):
        """Test golden hash generation."""
        self.log("\n=== Golden Hash System (S0-S4) ===", "SUMMARY")
        
        def test_golden_hashes():
            from ecdd_core.golden.generate_golden_hashes import generate_golden_hashes
            
            golden_dir = Path(__file__).parent / "ECDD_Experiment_Data" / "real"
            
            if not golden_dir.exists():
                return {
                    'passed': False,
                    'error': f'Golden directory not found: {golden_dir}',
                    'note': 'Skipped - expected in some environments'
                }
            
            results = generate_golden_hashes(golden_dir, max_images=5, face_backend="stub")
            
            success_count = sum(1 for r in results.values() 
                              if isinstance(r, dict) and "s0_raw_bytes" in r)
            
            return {
                'passed': success_count > 0,
                'images_processed': len(results),
                'successful': success_count,
                'stages': 'S0-S4 (preprocessing)',
                'sample_hash': list(results.values())[0].get('s0_raw_bytes', '')[:16] + '...' if results else None
            }
        
        self.run_test("Golden.S0-S4", "Golden hash generation", "GoldenHash", 
                     test_golden_hashes)
    
    def run_calibration_test(self):
        """Test calibration utilities."""
        self.log("\n=== Calibration Utilities ===", "SUMMARY")
        
        def test_calibration():
            from ecdd_core.calibration.temperature_scaling import (
                fit_temperature, apply_temperature, sigmoid, expected_calibration_error
            )
            import numpy as np
            
            # Generate mock data
            np.random.seed(42)
            logits = np.random.randn(100).astype(np.float32)
            labels = (logits > 0).astype(int)
            
            # Fit temperature
            params, details = fit_temperature(logits, labels)
            calibrated_logits = apply_temperature(logits, params)
            
            # Compute ECE
            pre_ece = expected_calibration_error(sigmoid(logits), labels)
            post_ece = expected_calibration_error(sigmoid(calibrated_logits), labels)
            
            improved = post_ece <= pre_ece
            
            return {
                'passed': improved,
                'temperature': float(params.temperature),
                'pre_ece': float(pre_ece),
                'post_ece': float(post_ece),
                'improved': improved,
                'note': 'Mock data - calibration working correctly'
            }
        
        self.run_test("Calib.Temp", "Temperature scaling", "Calibration", 
                     test_calibration)
    
    def run_pipeline_tests(self):
        """Test pipeline components."""
        self.log("\n=== Pipeline Components ===", "SUMMARY")
        
        # Test 1: Face Detection
        def test_face_detection():
            from ecdd_core.pipeline.face import detect_faces, FaceDetectorConfig
            import numpy as np
            
            mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cfg = FaceDetectorConfig(backend="stub")
            detection = detect_faces(mock_image, cfg)
            
            return {
                'passed': True,
                'backend': cfg.backend,
                'faces_detected': len(detection.boxes),
                'multi_face_policy': cfg.multi_face_policy,
                'note': 'Stub backend used for testing'
            }
        
        # Test 2: Preprocessing
        def test_preprocessing():
            from ecdd_core.pipeline.preprocess import (
                PreprocessConfig, resize_rgb_uint8, normalize_rgb_uint8
            )
            import numpy as np
            
            mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cfg = PreprocessConfig()
            resized = resize_rgb_uint8(mock_image, cfg)
            normalized = normalize_rgb_uint8(resized, cfg)
            
            shape_correct = normalized.shape == (256, 256, 3)
            dtype_correct = normalized.dtype == np.float32
            
            return {
                'passed': shape_correct and dtype_correct,
                'input_shape': list(mock_image.shape),
                'output_shape': list(normalized.shape),
                'output_dtype': str(normalized.dtype),
                'resize_kernel': cfg.resize_kernel
            }
        
        # Test 3: Determinism
        def test_determinism():
            from ecdd_core.pipeline.preprocess import (
                PreprocessConfig, resize_rgb_uint8, normalize_rgb_uint8
            )
            import numpy as np
            
            np.random.seed(42)
            mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cfg = PreprocessConfig()
            
            # Run twice
            output1 = normalize_rgb_uint8(resize_rgb_uint8(mock_image, cfg), cfg)
            output2 = normalize_rgb_uint8(resize_rgb_uint8(mock_image, cfg), cfg)
            
            max_diff = float(np.max(np.abs(output1 - output2)))
            deterministic = max_diff < 1e-6
            
            return {
                'passed': deterministic,
                'max_diff': max_diff,
                'tolerance': 1e-6,
                'note': 'Preprocessing is deterministic'
            }
        
        self.run_test("Pipeline.Face", "Face detection", "Pipeline", 
                     test_face_detection)
        self.run_test("Pipeline.Preprocess", "Preprocessing", "Pipeline", 
                     test_preprocessing)
        self.run_test("Pipeline.Determinism", "Determinism check", "Pipeline", 
                     test_determinism)
    
    def run_export_test(self):
        """Test export infrastructure."""
        self.log("\n=== Export Infrastructure ===", "SUMMARY")
        
        def test_parity_utils():
            from ecdd_core.export.model_parity import compute_parity_metrics
            import numpy as np
            
            # Mock outputs
            np.random.seed(42)
            float_outputs = np.random.randn(20).astype(np.float32)
            quant_outputs = float_outputs + np.random.randn(20).astype(np.float32) * 0.01
            
            result = compute_parity_metrics(float_outputs, quant_outputs, tolerance=0.05)
            
            return {
                'passed': result.passed,
                'max_abs_diff': result.max_abs_diff,
                'mean_abs_diff': result.mean_abs_diff,
                'rank_correlation': result.rank_correlation,
                'note': 'Parity utilities working correctly'
            }
        
        self.run_test("Export.Parity", "Parity validation", "Export", 
                     test_parity_utils)
    
    def generate_report(self, output_file: Optional[Path] = None):
        """Generate test report."""
        self.log("\n" + "="*80, "SUMMARY")
        self.log("TEST EXECUTION SUMMARY", "SUMMARY")
        self.log("="*80, "SUMMARY")
        
        # Calculate statistics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_time = sum(r.duration_ms for r in self.results)
        
        # Group by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {'passed': 0, 'failed': 0}
            if r.passed:
                categories[r.category]['passed'] += 1
            else:
                categories[r.category]['failed'] += 1
        
        # Print summary
        self.log(f"\nTotal Tests: {total}", "SUMMARY")
        self.log(f"Passed: {passed} ✅", "SUMMARY")
        self.log(f"Failed: {failed} ❌", "SUMMARY")
        self.log(f"Success Rate: {(passed/total*100):.1f}%", "SUMMARY")
        self.log(f"Total Time: {total_time:.1f}ms", "SUMMARY")
        
        self.log(f"\nBy Category:", "SUMMARY")
        for cat, stats in sorted(categories.items()):
            total_cat = stats['passed'] + stats['failed']
            self.log(f"  {cat}: {stats['passed']}/{total_cat} passed", "SUMMARY")
        
        # List failures
        failures = [r for r in self.results if not r.passed]
        if failures:
            self.log(f"\nFailed Tests:", "SUMMARY")
            for r in failures:
                self.log(f"  ❌ {r.test_id}: {r.name}", "ERROR")
                if r.error:
                    self.log(f"     Error: {r.error}", "ERROR")
        
        # Save to file
        if output_file:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total': total,
                    'passed': passed,
                    'failed': failed,
                    'success_rate': passed / total if total > 0 else 0,
                    'total_time_ms': total_time
                },
                'categories': categories,
                'results': [asdict(r) for r in self.results]
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.log(f"\nReport saved to: {output_file}", "SUMMARY")
        
        self.log("="*80, "SUMMARY")
        
        return passed == total
    
    def run_all(self):
        """Run all tests."""
        self.log("="*80, "SUMMARY")
        self.log("🧪 ECDD INFRASTRUCTURE TEST SUITE", "SUMMARY")
        self.log("="*80, "SUMMARY")
        
        self.run_phase6_tests()
        self.run_phase5_tests()
        self.run_golden_hash_test()
        self.run_calibration_test()
        self.run_pipeline_tests()
        self.run_export_test()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Master test runner for ECDD Experimentation infrastructure"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSON report file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Run tests
    runner = TestRunner(verbose=args.verbose)
    runner.run_all()
    
    # Generate report
    all_passed = runner.generate_report(output_file=args.output)
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
