#!/usr/bin/env python3
"""Test script for two-stage training implementation."""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student.tiny_ladeda import TinyLaDeDa
from models.teacher.ladeda_wrapper import LaDeDaWrapper
from datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from losses.distillation_improved import ImprovedPatchDistillationLoss
from models.pooling import TopKLogitPooling
from training.train_student_improved import ImprovedTwoStagePatchStudentTrainer


def count_trainable_params(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_initialization():
    """Test 1: Model initialization."""
    print("\n" + "=" * 80)
    print("TEST 1: Model Initialization")
    print("=" * 80)

    student = TinyLaDeDa(pretrained=False)
    teacher = LaDeDaWrapper(pretrained=False)

    print(f"✓ Student model initialized: {student.count_parameters()} parameters")
    print(f"✓ Teacher model initialized")

    return student, teacher


def test_freeze_backbone(student):
    """Test 2: Freeze backbone logic."""
    print("\n" + "=" * 80)
    print("TEST 2: Freeze Backbone Logic")
    print("=" * 80)

    # Get actual model (handle wrapper)
    model = student.model if hasattr(student, 'model') else student

    # Count trainable params before freeze
    before = count_trainable_params(student)
    print(f"Trainable params before freeze: {before}")

    # Simulate freezing (what stage 1 does)
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.conv2.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False

    after_freeze = count_trainable_params(student)
    print(f"Trainable params after freezing backbone: {after_freeze}")

    # Only fc should be trainable
    fc_params = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    print(f"FC layer params: {fc_params}")

    # Verify only fc is trainable
    assert after_freeze == fc_params, "Only FC should be trainable after freeze!"
    print(f"✓ Freeze backbone test PASSED")

    return student


def test_unfreeze_layer1(student):
    """Test 3: Unfreeze layer1 logic."""
    print("\n" + "=" * 80)
    print("TEST 3: Unfreeze Layer1 Logic")
    print("=" * 80)

    # Get actual model (handle wrapper)
    model = student.model if hasattr(student, 'model') else student

    before = count_trainable_params(student)
    print(f"Trainable params before unfreeze: {before}")

    # Simulate unfreezing layer1 (what stage 2 does)
    for param in model.layer1.parameters():
        param.requires_grad = True

    after_unfreeze = count_trainable_params(student)
    print(f"Trainable params after unfreezing layer1: {after_unfreeze}")

    # Verify layer1 was unfrozen
    layer1_params = sum(p.numel() for p in model.layer1.parameters() if p.requires_grad)
    fc_params = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    total = layer1_params + fc_params

    print(f"Layer1 params: {layer1_params}")
    print(f"FC params: {fc_params}")
    print(f"Total trainable: {total}")

    assert after_unfreeze == total, "Layer1 and FC should be trainable!"
    print(f"✓ Unfreeze layer1 test PASSED")


def test_dummy_forward_pass():
    """Test 4: Forward pass with dummy data."""
    print("\n" + "=" * 80)
    print("TEST 4: Dummy Forward Pass")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    student = TinyLaDeDa(pretrained=False).to(device)
    teacher = LaDeDaWrapper(pretrained=False).to(device)

    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)

    # Student forward pass
    with torch.no_grad():
        student_output = student(dummy_input)
    print(f"✓ Student output shape: {student_output.shape} (expected: torch.Size([{batch_size}, 1, 126, 126]))")
    assert student_output.shape == (batch_size, 1, 126, 126), "Wrong student output shape!"

    # Teacher forward pass
    with torch.no_grad():
        teacher_output = teacher(dummy_input)
    print(f"✓ Teacher output shape: {teacher_output.shape} (expected: torch.Size([{batch_size}, 1, 31, 31]))")
    assert teacher_output.shape == (batch_size, 1, 31, 31), "Wrong teacher output shape!"

    print(f"✓ Forward pass test PASSED")


def test_loss_computation():
    """Test 5: Loss computation."""
    print("\n" + "=" * 80)
    print("TEST 5: Loss Computation")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    student = TinyLaDeDa(pretrained=False).to(device)
    teacher = LaDeDaWrapper(pretrained=False).to(device)
    pooling = TopKLogitPooling(r=0.1, min_k=5)
    criterion = ImprovedPatchDistillationLoss(alpha_distill=0.5, alpha_task=0.5, temperature=4.0, use_kl_loss=True, enable_scale_matching=True)

    # Dummy batch
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_labels = torch.tensor([0, 1]).to(device)

    with torch.no_grad():
        student_patches = student(dummy_input)
        teacher_patches = teacher(dummy_input)
        student_image_logit = pooling(student_patches)

    # Compute loss
    total_loss, distill_loss, task_loss = criterion(
        student_patches, teacher_patches, student_image_logit, dummy_labels
    )

    print(f"✓ Total loss: {total_loss.item():.4f}")
    print(f"✓ Distill loss: {distill_loss.item():.4f}")
    print(f"✓ Task loss: {task_loss.item():.4f}")
    print(f"✓ Loss computation test PASSED")


def test_trainer_initialization():
    """Test 6: Trainer initialization and stage setup."""
    print("\n" + "=" * 80)
    print("TEST 6: Trainer Initialization and Stage Setup")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    student = TinyLaDeDa(pretrained=False).to(device)
    teacher = LaDeDaWrapper(pretrained=False).to(device)

    # Create minimal dummy loaders
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4
        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 256, 256),
                "label": torch.randint(0, 2, (1,)).item()
            }

    train_loader = DataLoader(DummyDataset(), batch_size=2)
    val_loader = DataLoader(DummyDataset(), batch_size=2)

    criterion = ImprovedPatchDistillationLoss(alpha_distill=0.5, alpha_task=0.5, temperature=4.0, use_kl_loss=True, enable_scale_matching=True)
    pooling = TopKLogitPooling(r=0.1, min_k=5)

    trainer = ImprovedTwoStagePatchStudentTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        pooling=pooling,
        device=device,
        stage1_epochs=1,
        stage2_epochs=1,
        stage1_lr=0.001,
        stage2_lr=0.0001,
        stage1_warmup_epochs=0.5,
        stage2_warmup_epochs=0.5,
    )

    print(f"✓ Trainer initialized successfully")
    print(f"✓ Stage 1 epochs: {trainer.stage1_epochs}")
    print(f"✓ Stage 2 epochs: {trainer.stage2_epochs}")
    print(f"✓ Stage 1 LR: {trainer.stage1_lr}")
    print(f"✓ Stage 2 LR: {trainer.stage2_lr}")

    # Test freeze/unfreeze
    print("\nTesting Stage 1 freeze logic...")
    trainer._freeze_backbone()
    trainable_stage1 = count_trainable_params(trainer.student_model)
    print(f"  Trainable params in Stage 1: {trainable_stage1}")

    print("\nTesting Stage 2 unfreeze logic...")
    trainer._unfreeze_layer1()
    trainable_stage2 = count_trainable_params(trainer.student_model)
    print(f"  Trainable params in Stage 2: {trainable_stage2}")

    assert trainable_stage2 > trainable_stage1, "Stage 2 should have more trainable params than Stage 1!"
    print(f"✓ Trainer initialization test PASSED")


def test_mini_training_loop():
    """Test 7: Mini training loop (1 epoch each stage)."""
    print("\n" + "=" * 80)
    print("TEST 7: Mini Training Loop (1 epoch Stage 1 + 1 epoch Stage 2)")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    student = TinyLaDeDa(pretrained=False).to(device)
    teacher = LaDeDaWrapper(pretrained=False).to(device)

    # Create minimal dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 8
        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 256, 256),
                "label": torch.randint(0, 2, (1,)).item()
            }

    train_loader = DataLoader(DummyDataset(), batch_size=2)
    val_loader = DataLoader(DummyDataset(), batch_size=2)

    criterion = ImprovedPatchDistillationLoss(alpha_distill=0.5, alpha_task=0.5, temperature=4.0, use_kl_loss=True, enable_scale_matching=True)
    pooling = TopKLogitPooling(r=0.1, min_k=5)

    trainer = ImprovedTwoStagePatchStudentTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        pooling=pooling,
        device=device,
        stage1_epochs=1,
        stage2_epochs=1,
        stage1_lr=0.001,
        stage2_lr=0.0001,
        stage1_warmup_epochs=0.5,
        stage2_warmup_epochs=0.5,
    )

    print("\n--- Stage 1: Classifier Training ---")
    trainer._freeze_backbone()
    trainer._setup_stage1_optimizer()

    try:
        train_loss, distill_loss, task_loss = trainer.train_epoch(stage=1)
        val_loss, val_distill, val_task, val_acc, val_auc = trainer.validate()

        print(f"✓ Stage 1 training completed")
        print(f"  Train Loss: {train_loss:.4f} (distill: {distill_loss:.4f}, task: {task_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (distill: {val_distill:.4f}, task: {val_task:.4f})")
        print(f"  Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    except Exception as e:
        print(f"✗ Stage 1 training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n--- Stage 2: Fine-tuning ---")
    trainer._unfreeze_layer1()
    trainer._setup_stage2_optimizer()

    try:
        train_loss, distill_loss, task_loss = trainer.train_epoch(stage=2)
        val_loss, val_distill, val_task, val_acc, val_auc = trainer.validate()

        print(f"✓ Stage 2 training completed")
        print(f"  Train Loss: {train_loss:.4f} (distill: {distill_loss:.4f}, task: {task_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (distill: {val_distill:.4f}, task: {val_task:.4f})")
        print(f"  Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")

    except Exception as e:
        print(f"✗ Stage 2 training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n✓ Mini training loop test PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("TWO-STAGE TRAINING IMPLEMENTATION TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Model initialization
        student, teacher = test_model_initialization()

        # Test 2: Freeze backbone
        student = test_freeze_backbone(student)

        # Test 3: Unfreeze layer1
        test_unfreeze_layer1(student)

        # Test 4: Dummy forward pass
        test_dummy_forward_pass()

        # Test 5: Loss computation
        test_loss_computation()

        # Test 6: Trainer initialization
        test_trainer_initialization()

        # Test 7: Mini training loop
        success = test_mini_training_loop()

        # Final summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        if success:
            print("✓ ALL TESTS PASSED!")
            print("\nTwo-stage training is working correctly:")
            print("  1. Freezing/unfreezing logic works")
            print("  2. Optimizer setup works")
            print("  3. Forward pass works")
            print("  4. Loss computation works")
            print("  5. Training loop executes successfully")
            print("\nYou can now run the full training with:")
            print("  python3 scripts/train_student_two_stage.py --epochs-s1 5 --epochs-s2 20")
        else:
            print("✗ SOME TESTS FAILED")
            return False

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
