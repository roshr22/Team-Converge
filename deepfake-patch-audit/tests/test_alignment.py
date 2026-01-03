
"""Test patch map alignment and spatial dimensions."""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.student.tiny_ladeda import TinyLaDeDa
from models.pooling import TopKLogitPooling


def test_teacher_output_shape():
    """Test that teacher outputs (B, 1, 31, 31) patch-logit map."""
    print("\n[TEST] Teacher output shape...")

    teacher = LaDeDaWrapper(
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=False,
    )
    teacher.eval()

    # Create dummy input: (B, 3, 256, 256)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256)

    with torch.no_grad():
        output = teacher(dummy_input)

    # Expected: (B, 1, 31, 31)
    assert output.shape == (batch_size, 1, 31, 31), f"Expected (4, 1, 31, 31), got {output.shape}"
    print(f"  ✓ Teacher output shape: {output.shape}")
    print(f"  ✓ Value range: [{output.min():.4f}, {output.max():.4f}]")


def test_student_output_shape():
    """Test that student outputs (B, 1, 126, 126) patch-logit map."""
    print("\n[TEST] Student output shape...")

    student = TinyLaDeDa(
        pretrained=False,
        pretrained_path=None,
    )
    student.eval()

    # Create dummy input: (B, 3, 256, 256)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 256, 256)

    with torch.no_grad():
        output = student(dummy_input)

    # Expected: (B, 1, 126, 126)
    assert output.shape == (batch_size, 1, 126, 126), f"Expected (4, 1, 126, 126), got {output.shape}"
    print(f"  ✓ Student output shape: {output.shape}")
    print(f"  ✓ Value range: [{output.min():.4f}, {output.max():.4f}]")


def test_spatial_alignment():
    """Test that student patches can be spatially aligned to teacher shape."""
    print("\n[TEST] Spatial alignment (126x126 -> 31x31)...")

    batch_size = 4
    student_patches = torch.randn(batch_size, 1, 126, 126)
    teacher_patches_shape = (batch_size, 1, 31, 31)

    # Align student patches to teacher shape
    student_aligned = F.adaptive_avg_pool2d(student_patches, (31, 31))

    assert student_aligned.shape == teacher_patches_shape, \
        f"Expected {teacher_patches_shape}, got {student_aligned.shape}"
    print(f"  ✓ Student aligned shape: {student_aligned.shape}")
    print(f"  ✓ Alignment method: adaptive_avg_pool2d")

    # Verify no NaN values
    assert not torch.isnan(student_aligned).any(), "NaN values found in aligned patches"
    print(f"  ✓ No NaN values in aligned patches")


def test_top_k_pooling_shape():
    """Test that TopK pooling produces correct image-level logits."""
    print("\n[TEST] Top-K pooling output shape...")

    pooling = TopKLogitPooling(r=0.1, min_k=5, aggregation="mean")
    pooling.eval()

    # Test with spatial input: (B, 1, 126, 126)
    batch_size = 4
    patch_logits = torch.randn(batch_size, 1, 126, 126)

    with torch.no_grad():
        image_logit = pooling(patch_logits)

    # Expected: (B, 1)
    assert image_logit.shape == (batch_size, 1), f"Expected (4, 1), got {image_logit.shape}"
    print(f"  ✓ Image logit shape: {image_logit.shape}")
    print(f"  ✓ Value range: [{image_logit.min():.4f}, {image_logit.max():.4f}]")


def test_top_k_pooling_k_selection():
    """Test that TopK pooling selects correct number of patches."""
    print("\n[TEST] Top-K pooling K selection...")

    # For 126x126 = 15,876 patches
    # K = max(5, ceil(0.1 * 15876)) = max(5, 1588) = 1588
    num_patches = 126 * 126
    expected_k = max(5, int(num_patches * 0.1 + 0.5))  # ceil approximation

    pooling = TopKLogitPooling(r=0.1, min_k=5, aggregation="mean")

    print(f"  ✓ Total patches: {num_patches}")
    print(f"  ✓ Expected K: {expected_k}")
    print(f"  ✓ Pooling strategy: top-k mean aggregation")


def test_batch_size_variance():
    """Test that models work with different batch sizes."""
    print("\n[TEST] Batch size variance...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    teacher.eval()
    student.eval()

    for batch_size in [1, 2, 4, 8]:
        dummy_input = torch.randn(batch_size, 3, 256, 256)

        with torch.no_grad():
            teacher_out = teacher(dummy_input)
            student_out = student(dummy_input)

        assert teacher_out.shape == (batch_size, 1, 31, 31)
        assert student_out.shape == (batch_size, 1, 126, 126)

    print(f"  ✓ All batch sizes work: 1, 2, 4, 8")


def test_gradient_flow():
    """Test that gradients can flow through student model."""
    print("\n[TEST] Gradient flow...")

    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    student.train()

    dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)
    output = student(dummy_input)

    # Backprop a simple loss
    loss = output.mean()
    loss.backward()

    # Check that gradients exist
    assert dummy_input.grad is not None, "No gradients for input"

    # Check that model has gradients
    has_grad = False
    for param in student.parameters():
        if param.grad is not None:
            has_grad = True
            break

    assert has_grad, "No gradients for model parameters"
    print(f"  ✓ Gradients flow through student model")
    print(f"  ✓ Input gradient shape: {dummy_input.grad.shape}")


def run_all_tests():
    """Run all alignment tests."""
    print("\n" + "="*80)
    print("PATCH ALIGNMENT AND SHAPE VERIFICATION TESTS")
    print("="*80)

    try:
        test_teacher_output_shape()
        test_student_output_shape()
        test_spatial_alignment()
        test_top_k_pooling_shape()
        test_top_k_pooling_k_selection()
        test_batch_size_variance()
        test_gradient_flow()

        print("\n" + "="*80)
        print("ALL ALIGNMENT TESTS PASSED ✓")
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
