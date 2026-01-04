#!/usr/bin/env python3
"""Two-stage student model training script with progressive unfreezing."""

import argparse
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.student.tiny_ladeda import TinyLaDeDa
from models.teacher.ladeda_wrapper import LaDeDaWrapper
from datasets.base_dataset import BaseDataset
from torch.utils.data import DataLoader
from losses.distillation import PatchDistillationLoss
from models.pooling import TopKLogitPooling
from training.train_student_two_stage import TwoStagePatchStudentTrainer


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validate_config(config):
    """Validate required configuration keys exist."""
    required_keys = {
        'model': {
            'teacher': ['pretrained', 'pretrained_path'],
            'student': ['pretrained', 'pretrained_path']
        },
        'dataset': ['resize_size', 'num_workers', 'pin_memory'],
        'training': {
            'distillation': ['alpha_distill', 'alpha_task']
        },
        'pooling': ['r', 'min_k', 'aggregation']
    }
    
    def check_nested(cfg, keys, path=""):
        for key, value in keys.items():
            if key not in cfg:
                raise KeyError(f"Missing required config key: {path}{key}")
            if isinstance(value, dict):
                check_nested(cfg[key], value, f"{path}{key}.")
            elif isinstance(value, list):
                for subkey in value:
                    if subkey not in cfg[key]:
                        raise KeyError(f"Missing required config key: {path}{key}.{subkey}")
    
    try:
        check_nested(config, required_keys)
        print("✓ Config validation passed")
        return True
    except KeyError as e:
        print(f"✗ Config validation failed: {e}")
        print("\nPlease check your config/base.yaml file")
        sys.exit(1)


def load_config(config_path="config/base.yaml"):
    """Load and validate configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        validate_config(config)
        return config
    except FileNotFoundError:
        print(f"✗ Config file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"✗ Error parsing YAML config: {e}")
        sys.exit(1)


def auto_detect_dataset_structure(dataset_root):
    """Auto-detect dataset structure and return appropriate paths."""
    dataset_root = Path(dataset_root)

    # Check if we have train/val subdirectories
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    # Check if train/val have real/fake subdirectories
    if (train_dir / "real").exists() and (train_dir / "fake").exists():
        if (val_dir / "real").exists() and (val_dir / "fake").exists():
            return {
                "mode": "directory",
                "train_root": str(train_dir),
                "val_root": str(val_dir),
            }

    # Check for CSV files
    train_csv = dataset_root / "data" / "splits" / "train.csv"
    val_csv = dataset_root / "data" / "splits" / "val.csv"

    if train_csv.exists() and val_csv.exists():
        return {
            "mode": "csv",
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
        }

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage training for student model with knowledge distillation"
    )
    parser.add_argument("--epochs-s1", type=int, default=5, help="Stage 1 epochs")
    parser.add_argument("--epochs-s2", type=int, default=20, help="Stage 2 epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr-s1", type=float, default=0.001, help="Stage 1 learning rate")
    parser.add_argument(
        "--lr-s2", type=float, default=0.0001, help="Stage 2 learning rate"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--dataset-root", type=str, default="dataset", help="Dataset root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/checkpoints_two_stage",
        help="Output directory",
    )
    parser.add_argument(
        "--teacher-weights",
        type=str,
        default=None,
        choices=["wildrf", "ForenSynth", "finetuned"],  # Fixed typo: forensyth → ForenSynth
        help="Teacher model weights: 'wildrf'/'ForenSynth' (pretrained) or 'finetuned' (fine-tuned)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"✓ Random seed set to {args.seed}")

    # Auto-detect dataset structure
    dataset_info = auto_detect_dataset_structure(args.dataset_root)

    if dataset_info is None:
        print("\n" + "=" * 80)
        print("ERROR: Dataset structure not recognized!")
        print("=" * 80)
        print("\nExpected one of:")
        print("1. Directory structure:")
        print("   dataset/train/real/")
        print("   dataset/train/fake/")
        print("   dataset/val/real/")
        print("   dataset/val/fake/")
        print("\n2. CSV files:")
        print("   dataset/data/splits/train.csv")
        print("   dataset/data/splits/val.csv")
        sys.exit(1)  # Fixed: exit(1) → sys.exit(1)

    print("\n" + "=" * 80)
    print("Dataset Auto-Detection")
    print("=" * 80)
    print(f"✓ Detected mode: {dataset_info['mode'].upper()}")
    if dataset_info["mode"] == "directory":
        print(f"  Train: {dataset_info['train_root']}")
        print(f"  Val:   {dataset_info['val_root']}")
    else:
        print(f"  Train CSV: {dataset_info['train_csv']}")
        print(f"  Val CSV:   {dataset_info['val_csv']}")

    # Load config
    config = load_config()

    # Override teacher weights if specified via command-line
    if args.teacher_weights:
        if args.teacher_weights.lower() == "wildrf":
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/WildRF_LaDeDa.pth"
        elif args.teacher_weights.lower() == "forensynth":  # Fixed case handling
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/ForenSynth_LaDeDa.pth"
        elif args.teacher_weights.lower() == "finetuned":
            config["model"]["teacher"]["pretrained_path"] = "weights/teacher/teacher_finetuned_best.pth"
            config["model"]["teacher"]["pretrained"] = True  # Will load checkpoint

    # Print final resolved paths
    print("\n" + "=" * 80)
    print("RESOLVED MODEL PATHS")
    print("=" * 80)
    print(f"Teacher: {config['model']['teacher']['pretrained_path']}")
    print(f"Student: {config['model']['student']['pretrained_path']}")
    
    print("\n" + "=" * 80)
    print("TWO-STAGE STUDENT TRAINING - Configuration")
    print("=" * 80)
    print(f"Stage 1 epochs: {args.epochs_s1}")
    print(f"Stage 2 epochs: {args.epochs_s2}")
    print(f"Batch size: {args.batch_size}")
    print(f"Stage 1 LR: {args.lr_s1}")
    print(f"Stage 2 LR: {args.lr_s2}")
    print(f"Device: {args.device}")
    print(f"Distillation alpha: {config['training']['distillation']['alpha_distill']}")
    print(f"Task alpha: {config['training']['distillation']['alpha_task']}")

    # =========================================================================
    # Load Models
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading Models")
    print("=" * 80)

    # Student model - safe dictionary access with .get()
    student_model = TinyLaDeDa(
        pretrained=config["model"]["student"].get("pretrained", False),
        pretrained_path=config["model"]["student"].get(
            "pretrained_path", "weights/student/ForenSynth_Tiny_LaDeDa.pth"
        ),
    )
    print(f"✓ Student model loaded: {student_model.count_parameters()} parameters")

    # Teacher model - safe dictionary access with .get()
    teacher_model = LaDeDaWrapper(
        pretrained=config["model"]["teacher"].get("pretrained", True),
        pretrained_path=config["model"]["teacher"].get(
            "pretrained_path", "weights/teacher/WildRF_LaDeDa.pth"
        ),
    )
    print(f"✓ Teacher model loaded")

    # =========================================================================
    # Load Datasets
    # =========================================================================
    print("\n" + "=" * 80)
    print("Loading Datasets")
    print("=" * 80)

    # Load datasets based on detected structure
    if dataset_info["mode"] == "directory":
        # Use directory structure (train/real, train/fake, etc.)
        train_dataset = BaseDataset(
            root_dir=dataset_info["train_root"],
            split="train",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            split_file=None,  # Use directory structure
        )

        val_dataset = BaseDataset(
            root_dir=dataset_info["val_root"],
            split="val",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            split_file=None,  # Use directory structure
        )
    else:
        # Use CSV split files
        train_dataset = BaseDataset(
            root_dir=args.dataset_root,
            split="train",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            split_file=dataset_info["train_csv"],
        )

        val_dataset = BaseDataset(
            root_dir=args.dataset_root,
            split="val",
            resize_size=config["dataset"]["resize_size"],
            normalize=True,
            split_file=dataset_info["val_csv"],
        )

    # Conditionally set pin_memory based on device
    use_pin_memory = config["dataset"].get("pin_memory", True) and args.device == "cuda"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=use_pin_memory,  # Only pin memory when using CUDA
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=use_pin_memory,  # Only pin memory when using CUDA
    )

    print(f"✓ Training dataset: {len(train_dataset)} samples")
    print(f"✓ Validation dataset: {len(val_dataset)} samples")
    print(f"✓ pin_memory: {use_pin_memory} (device={args.device})")

    # =========================================================================
    # Setup Loss, Pooling, and Trainer
    # =========================================================================
    print("\n" + "=" * 80)
    print("Setting Up Loss and Pooling")
    print("=" * 80)

    criterion = PatchDistillationLoss(
        alpha_distill=config["training"]["distillation"]["alpha_distill"],
        alpha_task=config["training"]["distillation"]["alpha_task"],
    )
    print(
        f"✓ Loss: PatchDistillationLoss "
        f"(alpha_distill={config['training']['distillation']['alpha_distill']}, "
        f"alpha_task={config['training']['distillation']['alpha_task']})"
    )

    pooling = TopKLogitPooling(
        r=config["pooling"]["r"],
        min_k=config["pooling"]["min_k"],
        aggregation=config["pooling"]["aggregation"],
    )
    print(f"✓ Pooling: TopKLogitPooling (r={config['pooling']['r']}, min_k={config['pooling']['min_k']})")

    # =========================================================================
    # Verify Patch Grid Compatibility
    # =========================================================================
    print("\n" + "=" * 80)
    print("Verifying Patch Grid Compatibility")
    print("=" * 80)
    
    # Get expected patch grids from config
    teacher_grid = tuple(config.get("patches", {}).get("teacher_grid", [31, 31]))
    student_grid = tuple(config.get("patches", {}).get("student_grid", [126, 126]))
    
    print(f"Expected teacher grid: {teacher_grid[0]}×{teacher_grid[1]}")
    print(f"Expected student grid: {student_grid[0]}×{student_grid[1]}")
    print("✓ Patch grids configured (will be validated during first forward pass)")

    # =========================================================================
    # Create Trainer and Train
    # =========================================================================
    trainer = TwoStagePatchStudentTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        pooling=pooling,
        device=args.device,
        stage1_epochs=args.epochs_s1,
        stage2_epochs=args.epochs_s2,
        stage1_lr=args.lr_s1,
        stage2_lr=args.lr_s2,
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    history = trainer.train(checkpoint_dir=args.output_dir)

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Checkpoints saved to: {args.output_dir}")
    print(f"Final validation AUC: {history['val_auc'][-1]:.4f}")


if __name__ == "__main__":
    main()
