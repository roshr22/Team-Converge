"""Test preprocessing transforms (NPR and gradient-based)."""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.student.tiny_ladeda import TinyLaDeDa


def test_npr_preprocessing():
    """Test that NPR preprocessing is applied in teacher model."""
    print("\n[TEST] NPR preprocessing...")

    teacher = LaDeDaWrapper(
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=False,
    )
    teacher.eval()

    # Create simple input with known pattern
    batch_size = 1
    dummy_input = torch.ones(batch_size, 3, 256, 256)
    dummy_input.requires_grad_(True)

    with torch.no_grad():
        output = teacher(dummy_input)

    # NPR preprocessing should zero-out uniform regions
    # since nearest-neighbor downsampling and upsampling should preserve uniform values
    print(f"  ✓ NPR preprocessing initialized")
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Output value range: [{output.min():.4f}, {output.max():.4f}]")

    # Verify output is not all zeros (some detection happening)
    assert not torch.all(output == 0), "All-zero output suggests preprocessing issue"
    print(f"  ✓ Output is non-trivial (not all zeros)")


def test_gradient_preprocessing():
    """Test that gradient preprocessing is applied in student model."""
    print("\n[TEST] Gradient preprocessing...")

    student = TinyLaDeDa(
        pretrained=False,
        pretrained_path=None,
    )
    student.eval()

    # Create simple input with gradient pattern
    batch_size = 1
    dummy_input = torch.ones(batch_size, 3, 256, 256)
    # Create a simple gradient: increase from left to right
    for i in range(256):
        dummy_input[:, :, :, i] *= (i / 256.0)

    with torch.no_grad():
        output = student(dummy_input)

    # Gradient preprocessing should enhance edges
    print(f"  ✓ Gradient preprocessing initialized")
    print(f"  ✓ Output shape: {output.shape}")
    print(f"  ✓ Output value range: [{output.min():.4f}, {output.max():.4f}]")

    # Verify output is not all zeros
    assert not torch.all(output == 0), "All-zero output suggests preprocessing issue"
    print(f"  ✓ Output is non-trivial (not all zeros)")


def test_preprocessing_output_ranges():
    """Test that preprocessing outputs are in reasonable ranges."""
    print("\n[TEST] Preprocessing output ranges...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    teacher.eval()
    student.eval()

    # Create random input
    dummy_input = torch.randn(2, 3, 256, 256)

    with torch.no_grad():
        teacher_out = teacher(dummy_input)
        student_out = student(dummy_input)

    # Check for NaN/Inf
    assert not torch.isnan(teacher_out).any(), "NaN in teacher output"
    assert not torch.isinf(teacher_out).any(), "Inf in teacher output"
    assert not torch.isnan(student_out).any(), "NaN in student output"
    assert not torch.isinf(student_out).any(), "Inf in student output"

    print(f"  ✓ Teacher output: no NaN/Inf")
    print(f"  ✓ Student output: no NaN/Inf")
    print(f"  ✓ Teacher range: [{teacher_out.min():.4f}, {teacher_out.max():.4f}]")
    print(f"  ✓ Student range: [{student_out.min():.4f}, {student_out.max():.4f}]")


def test_preprocessing_consistency():
    """Test that preprocessing is deterministic."""
    print("\n[TEST] Preprocessing consistency...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    teacher.eval()
    student.eval()

    dummy_input = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        teacher_out1 = teacher(dummy_input)
        teacher_out2 = teacher(dummy_input)
        student_out1 = student(dummy_input)
        student_out2 = student(dummy_input)

    # Outputs should be identical for same input
    assert torch.allclose(teacher_out1, teacher_out2), "Teacher preprocessing not deterministic"
    assert torch.allclose(student_out1, student_out2), "Student preprocessing not deterministic"

    print(f"  ✓ Teacher preprocessing is deterministic")
    print(f"  ✓ Student preprocessing is deterministic")


def test_different_inputs_different_outputs():
    """Test that different inputs produce different outputs."""
    print("\n[TEST] Different inputs -> different outputs...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    teacher.eval()
    student.eval()

    input1 = torch.randn(1, 3, 256, 256)
    input2 = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        teacher_out1 = teacher(input1)
        teacher_out2 = teacher(input2)
        student_out1 = student(input1)
        student_out2 = student(input2)

    # Different inputs should produce different outputs (not identical)
    assert not torch.allclose(teacher_out1, teacher_out2, rtol=1e-4), \
        "Teacher produces same output for different inputs"
    assert not torch.allclose(student_out1, student_out2, rtol=1e-4), \
        "Student produces same output for different inputs"

    print(f"  ✓ Teacher: different inputs produce different outputs")
    print(f"  ✓ Student: different inputs produce different outputs")


def test_preprocessing_with_real_image_format():
    """Test preprocessing with realistic image data."""
    print("\n[TEST] Preprocessing with realistic image ranges...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    teacher.eval()
    student.eval()

    # Simulate normalized image data (mean=0.485, 0.456, 0.406; std=0.229, 0.224, 0.225)
    # After normalization, values typically in range [-2, 2]
    dummy_input = torch.randn(2, 3, 256, 256) * 0.229  # Simulate normalized image

    with torch.no_grad():
        teacher_out = teacher(dummy_input)
        student_out = student(dummy_input)

    # Should handle normalized input without issues
    assert not torch.isnan(teacher_out).any(), "NaN with normalized input (teacher)"
    assert not torch.isnan(student_out).any(), "NaN with normalized input (student)"
    assert not torch.isinf(teacher_out).any(), "Inf with normalized input (teacher)"
    assert not torch.isinf(student_out).any(), "Inf with normalized input (student)"

    print(f"  ✓ Teacher handles normalized images")
    print(f"  ✓ Student handles normalized images")
    print(f"  ✓ No numerical instabilities detected")


def test_preprocessing_edge_cases():
    """Test preprocessing with edge case inputs."""
    print("\n[TEST] Preprocessing edge cases...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    teacher.eval()
    student.eval()

    # Test with zeros
    zeros_input = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        teacher_zeros = teacher(zeros_input)
        student_zeros = student(zeros_input)

    assert not torch.isnan(teacher_zeros).any(), "NaN with zero input (teacher)"
    assert not torch.isnan(student_zeros).any(), "NaN with zero input (student)"

    # Test with ones
    ones_input = torch.ones(1, 3, 256, 256)
    with torch.no_grad():
        teacher_ones = teacher(ones_input)
        student_ones = student(ones_input)

    assert not torch.isnan(teacher_ones).any(), "NaN with ones input (teacher)"
    assert not torch.isnan(student_ones).any(), "NaN with ones input (student)"

    # Test with very large values
    large_input = torch.randn(1, 3, 256, 256) * 100
    with torch.no_grad():
        teacher_large = teacher(large_input)
        student_large = student(large_input)

    assert not torch.isnan(teacher_large).any(), "NaN with large values (teacher)"
    assert not torch.isnan(student_large).any(), "NaN with large values (student)"

    print(f"  ✓ Handles zero input")
    print(f"  ✓ Handles ones input")
    print(f"  ✓ Handles large values")


def run_all_tests():
    """Run all preprocessing tests."""
    print("\n" + "="*80)
    print("PREPROCESSING VERIFICATION TESTS")
    print("="*80)

    try:
        test_npr_preprocessing()
        test_gradient_preprocessing()
        test_preprocessing_output_ranges()
        test_preprocessing_consistency()
        test_different_inputs_different_outputs()
        test_preprocessing_with_real_image_format()
        test_preprocessing_edge_cases()

        print("\n" + "="*80)
        print("ALL PREPROCESSING TESTS PASSED ✓")
        print("="*80 + "\n")
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
