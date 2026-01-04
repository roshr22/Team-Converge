#!/usr/bin/env python3
"""
Quick training script using improved components.
This script uses:
- ImprovedPatchDistillationLoss (with scale matching, KL divergence)
- ImprovedTwoStagePatchStudentTrainer (with all stability fixes)
- Real dataset from dataset/train and dataset/val
"""

import sys
import torch
import random
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.student.tiny_ladeda import TinyLaDeDa
from models.teacher.ladeda_wrapper import LaDeDaWrapper
from datasets.base_dataset import BaseDataset
from losses.distillation_improved import ImprovedPatchDistillationLoss, ModeCollapsePrevention
from models.pooling import TopKLogitPooling
from training.train_student_improved import ImprovedTwoStagePatchStudentTrainer


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    print("=" * 80)
    print("IMPROVED STUDENT MODEL TRAINING")
    print("=" * 80)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[1] Device: {device}")
    set_seed(42)

    # Dataset paths
    dataset_root = Path("deepfake-patch-audit/dataset")
    train_root = dataset_root / "train"
    val_root = dataset_root / "val"

    print(f"\n[2] Loading dataset...")
    print(f"    Train: {train_root}")
    print(f"    Val: {val_root}")

    # Create datasets
    train_dataset = BaseDataset(
        root_dir=str(train_root),
        split="train",
        image_format="jpg",
        resize_size=256,
        normalize=True,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        split_file=None
    )

    val_dataset = BaseDataset(
        root_dir=str(val_root),
        split="val",
        image_format="jpg",
        resize_size=256,
        normalize=True,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
        split_file=None
    )

    print(f"    Train samples: {len(train_dataset)}")
    print(f"    Val samples: {len(val_dataset)}")

    # Create data loaders (Windows-safe: num_workers=0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=0,  # CRITICAL FOR WINDOWS
        pin_memory=False,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        shuffle=False
    )

    print(f"    Train batches: {len(train_loader)}")
    print(f"    Val batches: {len(val_loader)}")

    # Load models
    print(f"\n[3] Loading models...")
    try:
        teacher = LaDeDaWrapper(pretrained="wildrf").to(device)
        print(f"    ✓ Teacher (LaDeDa9) loaded")
    except Exception as e:
        print(f"    ✗ Teacher load failed: {e}")
        sys.exit(1)

    try:
        student = TinyLaDeDa().to(device)
        print(f"    ✓ Student (TinyLaDeDa) loaded")
    except Exception as e:
        print(f"    ✗ Student load failed: {e}")
        sys.exit(1)

    # Setup pooling
    pooling = TopKLogitPooling(r=0.1, min_k=5, aggregation="mean")
    print(f"    ✓ Pooling (TopK, r=0.1, min_k=5, mean) created")

    # Setup improved loss
    print(f"\n[4] Setting up improved loss...")
    criterion = ImprovedPatchDistillationLoss(
        alpha_distill=0.3,  # REDUCED from 0.5
        alpha_task=0.7,     # INCREASED from 0.5
        temperature=4.0,    # Soft targets
        use_kl_loss=True,   # Scale-invariant
        enable_scale_matching=True,  # Fix scale mismatch
        enable_gradient_monitoring=True
    )
    print(f"    ✓ Loss: ImprovedPatchDistillationLoss")
    print(f"      - α_distill=0.3, α_task=0.7")
    print(f"      - Temperature=4.0 (soft targets)")
    print(f"      - KL divergence (not MSE)")
    print(f"      - Scale matching enabled")

    # Setup trainer
    print(f"\n[5] Setting up trainer...")
    trainer = ImprovedTwoStagePatchStudentTrainer(
        student_model=student,
        teacher_model=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        pooling=pooling,
        device=device,
        # Stage 1 config
        stage1_epochs=5,
        stage1_lr=0.0001,  # REDUCED from 0.001
        stage1_warmup_epochs=0.5,
        # Stage 2 config
        stage2_epochs=20,
        stage2_lr=0.00005,  # REDUCED from 0.0001
        stage2_backbone_lr=0.000005,  # Layer-wise
        stage2_warmup_epochs=1.0,
        # Gradient control
        gradient_clip_norm=0.5,
        gradient_clip_value=1.0,
        # Checkpoint
        checkpoint_dir="outputs/checkpoints_improved",
        save_best_only=True
    )
    print(f"    ✓ Trainer created")
    print(f"      - Stage 1: 5 epochs, LR=0.0001, warmup=0.5")
    print(f"      - Stage 2: 20 epochs, LR=0.00005, warmup=1.0")
    print(f"      - Gradient clipping: norm=0.5, value=1.0")

    # Run training
    print(f"\n[6] Starting training...")
    print("=" * 80)
    try:
        trainer.train()

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETED!")
        print("=" * 80)
        print(f"\nResults:")
        print(f"  Best validation AUC: {trainer.best_val_auc:.4f}")
        print(f"  Best checkpoint: {trainer.best_checkpoint_path}")
        print(f"\nNext steps:")
        print(f"  1. Evaluate on test set")
        print(f"  2. Export to ONNX: python3 scripts/export_student.py --checkpoint {trainer.best_checkpoint_path}")
        print(f"  3. Deploy to Raspberry Pi or Nicla Vision")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
