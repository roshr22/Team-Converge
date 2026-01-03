"""Test model architectures, parameter counts, and forward passes."""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.teacher.ladeda_wrapper import LaDeDaWrapper
from models.student.tiny_ladeda import TinyLaDeDa
from models.pooling import TopKLogitPooling
from losses.distillation import PatchDistillationLoss


def count_parameters(model):
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def test_teacher_parameter_count():
    """Test that teacher has expected parameter count."""
    print("\n[TEST] Teacher parameter count...")

    teacher = LaDeDaWrapper(
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=False,
    )

    num_params = count_parameters(teacher)
    print(f"  ✓ Teacher parameters: {num_params:,}")
    print(f"  ✓ Expected: ~13.6M (LaDeDa9)")

    # Should be in millions (ResNet-50 based)
    assert num_params > 1_000_000, f"Too few parameters: {num_params}"
    print(f"  ✓ Parameter count is reasonable")


def test_student_parameter_count():
    """Test that student has expected parameter count (1,297)."""
    print("\n[TEST] Student parameter count...")

    student = TinyLaDeDa(
        pretrained=False,
        pretrained_path=None,
    )

    num_params = count_parameters(student)
    expected_params = 1_297
    print(f"  ✓ Student parameters: {num_params:,}")
    print(f"  ✓ Expected: {expected_params:,}")

    # Allow small tolerance in parameter count (e.g., bias terms)
    tolerance = 50
    assert abs(num_params - expected_params) <= tolerance, \
        f"Parameter count mismatch: {num_params} vs {expected_params}"
    print(f"  ✓ Parameter count matches specification")


def test_teacher_forward_pass():
    """Test teacher forward pass with various input sizes."""
    print("\n[TEST] Teacher forward pass...")

    teacher = LaDeDaWrapper(
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=False,
    )
    teacher.eval()

    # Test standard input
    standard_input = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        output = teacher(standard_input)

    assert output.shape == (4, 1, 31, 31), f"Wrong shape: {output.shape}"
    print(f"  ✓ 256×256 input: {standard_input.shape} → {output.shape}")

    # Test different batch sizes
    for batch_size in [1, 2, 8]:
        test_input = torch.randn(batch_size, 3, 256, 256)
        with torch.no_grad():
            output = teacher(test_input)
        assert output.shape == (batch_size, 1, 31, 31)
    print(f"  ✓ Works with batch sizes: 1, 2, 4, 8")


def test_student_forward_pass():
    """Test student forward pass with various input sizes."""
    print("\n[TEST] Student forward pass...")

    student = TinyLaDeDa(
        pretrained=False,
        pretrained_path=None,
    )
    student.eval()

    # Test standard input
    standard_input = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        output = student(standard_input)

    assert output.shape == (4, 1, 126, 126), f"Wrong shape: {output.shape}"
    print(f"  ✓ 256×256 input: {standard_input.shape} → {output.shape}")

    # Test different batch sizes
    for batch_size in [1, 2, 8]:
        test_input = torch.randn(batch_size, 3, 256, 256)
        with torch.no_grad():
            output = student(test_input)
        assert output.shape == (batch_size, 1, 126, 126)
    print(f"  ✓ Works with batch sizes: 1, 2, 4, 8")


def test_models_can_be_frozen():
    """Test that models can be frozen for evaluation."""
    print("\n[TEST] Model freezing...")

    teacher = LaDeDaWrapper(
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=True,
    )

    # Check that parameters require no gradients
    num_frozen = sum(1 for p in teacher.parameters() if not p.requires_grad)
    num_total = sum(1 for p in teacher.parameters())
    print(f"  ✓ Teacher frozen: {num_frozen}/{num_total} params frozen")

    student = TinyLaDeDa(
        pretrained=False,
        pretrained_path=None,
    )
    student.eval()

    # Set to eval mode
    for param in student.parameters():
        param.requires_grad = True
    print(f"  ✓ Student can be unfrozen for training")


def test_teacher_is_frozen_by_default():
    """Test that teacher is frozen in training setup."""
    print("\n[TEST] Teacher frozen in training setup...")

    teacher = LaDeDaWrapper(
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=True,
    )
    teacher.eval()

    # All params should require no gradients
    all_frozen = all(not p.requires_grad for p in teacher.parameters())
    assert all_frozen, "Teacher should be frozen"
    print(f"  ✓ All teacher parameters frozen")


def test_student_is_trainable():
    """Test that student is trainable."""
    print("\n[TEST] Student is trainable...")

    student = TinyLaDeDa(
        pretrained=False,
        pretrained_path=None,
    )
    student.train()

    # All params should require gradients
    all_trainable = all(p.requires_grad for p in student.parameters())
    assert all_trainable, "Student parameters should be trainable"
    print(f"  ✓ All student parameters trainable")


def test_end_to_end_forward_pass():
    """Test complete forward pass: student → pooling."""
    print("\n[TEST] End-to-end forward pass...")

    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    pooling = TopKLogitPooling(r=0.1, min_k=5, aggregation="mean")
    student.eval()
    pooling.eval()

    dummy_input = torch.randn(4, 3, 256, 256)

    with torch.no_grad():
        student_patches = student(dummy_input)  # (B, 1, 126, 126)
        image_logit = pooling(student_patches)  # (B, 1)

    assert student_patches.shape == (4, 1, 126, 126)
    assert image_logit.shape == (4, 1)
    print(f"  ✓ Input: {dummy_input.shape}")
    print(f"  ✓ Student patches: {student_patches.shape}")
    print(f"  ✓ Image logit: {image_logit.shape}")


def test_loss_computation():
    """Test loss computation with batch data."""
    print("\n[TEST] Loss computation...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    pooling = TopKLogitPooling(r=0.1, min_k=5, aggregation="mean")
    criterion = PatchDistillationLoss(
        alpha_distill=0.5,
        alpha_task=0.5,
    )

    teacher.eval()
    student.eval()
    pooling.eval()
    criterion.eval()

    # Create batch data
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 256, 256)
    dummy_labels = torch.randint(0, 2, (batch_size,))

    with torch.no_grad():
        student_patches = student(dummy_images)
        student_image_logit = pooling(student_patches)
        teacher_patches = teacher(dummy_images)

        loss, distill_loss, task_loss = criterion(
            student_patches,
            teacher_patches,
            student_image_logit,
            dummy_labels,
        )

    # Check that losses are scalar tensors
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert distill_loss.dim() == 0, f"Distill loss should be scalar"
    assert task_loss.dim() == 0, f"Task loss should be scalar"

    # Check that losses are finite
    assert not torch.isnan(loss), "NaN loss"
    assert not torch.isinf(loss), "Inf loss"

    print(f"  ✓ Total loss: {loss.item():.6f}")
    print(f"  ✓ Distill loss: {distill_loss.item():.6f}")
    print(f"  ✓ Task loss: {task_loss.item():.6f}")
    print(f"  ✓ Loss computation verified")


def test_gradient_computation():
    """Test that gradients can be computed for student."""
    print("\n[TEST] Gradient computation...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)
    pooling = TopKLogitPooling(r=0.1, min_k=5, aggregation="mean")
    criterion = PatchDistillationLoss(
        alpha_distill=0.5,
        alpha_task=0.5,
    )

    teacher.eval()
    student.train()

    # Create batch data
    dummy_images = torch.randn(4, 3, 256, 256)
    dummy_labels = torch.randint(0, 2, (4,)).float()

    # Forward pass
    student_patches = student(dummy_images)
    student_image_logit = pooling(student_patches)

    with torch.no_grad():
        teacher_patches = teacher(dummy_images)

    loss, _, _ = criterion(
        student_patches,
        teacher_patches,
        student_image_logit,
        dummy_labels,
    )

    # Backward pass
    loss.backward()

    # Check that gradients exist
    has_grads = False
    for param in student.parameters():
        if param.grad is not None:
            has_grads = True
            break

    assert has_grads, "No gradients computed for student"
    print(f"  ✓ Gradients computed for student model")


def test_model_device_compatibility():
    """Test that models work on both CPU and CUDA (if available)."""
    print("\n[TEST] Device compatibility...")

    teacher = LaDeDaWrapper(pretrained=False, pretrained_path=None)
    student = TinyLaDeDa(pretrained=False, pretrained_path=None)

    # Test CPU
    teacher.cpu()
    student.cpu()
    dummy_input = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        _ = teacher(dummy_input)
        _ = student(dummy_input)

    print(f"  ✓ Models work on CPU")

    # Test CUDA if available
    if torch.cuda.is_available():
        teacher.cuda()
        student.cuda()
        dummy_input = dummy_input.cuda()

        with torch.no_grad():
            _ = teacher(dummy_input)
            _ = student(dummy_input)

        print(f"  ✓ Models work on CUDA")
    else:
        print(f"  ✓ CUDA not available (CPU only)")


def run_all_tests():
    """Run all model tests."""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE AND FORWARD PASS TESTS")
    print("="*80)

    try:
        test_teacher_parameter_count()
        test_student_parameter_count()
        test_teacher_forward_pass()
        test_student_forward_pass()
        test_models_can_be_frozen()
        test_teacher_is_frozen_by_default()
        test_student_is_trainable()
        test_end_to_end_forward_pass()
        test_loss_computation()
        test_gradient_computation()
        test_model_device_compatibility()

        print("\n" + "="*80)
        print("ALL MODEL TESTS PASSED ✓")
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
