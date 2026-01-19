#!/usr/bin/env python
"""Staged Fine-tuning Script (Step 9).

Implements micro-stepped training:
- Stage A: Head-only stabilization (1-2 epochs)
- Stage B: Partial unfreeze - layer4 (3-8 epochs)
- Stage C: Optional deeper unfreeze (if needed)

Usage:
    # Run all stages
    python train_staged.py --config config.yaml --stages A B
    
    # Run specific stage
    python train_staged.py --config config.yaml --stages B --resume best_model.pt
    
    # Colab
    !python train_staged.py --config config.yaml \\
        --override dataset.ffpp_root=/content/data/raw/ffpp \\
        --override caching.cache_dir=/content/cache/faces \\
        --stages A B
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from utils import (
    load_config,
    set_global_seed,
    FFppDataset,
    ConstrainedBatchSampler,
    DeploymentRealismAugmentation,
    ValidationTransform,
)
from utils.staged_training import (
    FinetuneStage,
    StageConfig,
    LayerFreezer,
    CheckpointManager,
    CalibrationLogger,
    get_stage_configs_from_config,
    DEFAULT_STAGES,
    get_git_commit_hash,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Staged Fine-tuning (Step 9)")
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--override', action='append', default=[], help='Config overrides')
    parser.add_argument('--stages', nargs='+', default=['A', 'B'], 
                       choices=['A', 'B', 'C'], help='Which stages to run')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Setup only, no training')
    return parser.parse_args()


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config."""
    for override in overrides:
        key, value = override.split('=', 1)
        
        # Parse value
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
        
        parts = key.split('.')
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
        logger.info(f"Override: {key} = {value}")
    
    return config


def create_model(config: dict) -> nn.Module:
    """Create model with attention pooling head."""
    from torchvision import models
    
    model_cfg = config.get('model', {})
    
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Replace FC with attention pooling + binary classification head
    in_features = model.fc.in_features
    hidden_dim = model_cfg.get('attention_hidden_dim', 512)
    dropout = model_cfg.get('dropout_rate', 0.4)
    
    # Simple attention pooling head
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    
    return model


def create_dataloaders(config: dict) -> tuple:
    """Create train and validation dataloaders."""
    training_cfg = config.get('training', {})
    
    train_transform = DeploymentRealismAugmentation.from_config(config)
    val_transform = ValidationTransform(
        target_size=config.get('dataset', {}).get('crop_size', 256)
    )
    
    logger.info("Creating datasets...")
    train_dataset = FFppDataset.from_config(config, split='train', transform=train_transform)
    val_dataset = FFppDataset.from_config(config, split='val', transform=val_transform)
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    train_sampler = ConstrainedBatchSampler.from_dataset(train_dataset, config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=training_cfg.get('pin_memory', True),
        collate_fn=collate_batch,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get('batch_size', 24),
        shuffle=False,
        num_workers=training_cfg.get('num_workers', 4),
        pin_memory=training_cfg.get('pin_memory', True),
        collate_fn=collate_batch,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def collate_batch(batch: list) -> Dict[str, Any]:
    """Custom collate function."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    
    return {
        'image': images,
        'label': labels,
        'sample_id': [item['sample_id'] for item in batch],
        'video_id': [item['video_id'] for item in batch],
        'group_id': [item['group_id'] for item in batch],
        'method': [item['method'] for item in batch],
    }


def get_class_weights(train_dataset) -> Optional[torch.Tensor]:
    """Compute class weights for balanced loss."""
    labels = [s.label for s in train_dataset.samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return None
    
    # pos_weight for BCEWithLogitsLoss
    pos_weight = n_neg / n_pos
    logger.info(f"Class balance: neg={n_neg}, pos={n_pos}, pos_weight={pos_weight:.2f}")
    
    return torch.tensor([pos_weight])


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    freezer: LayerFreezer,
    stage_config: StageConfig,
) -> Dict[str, float]:
    """Train for one epoch with proper BN handling."""
    model.train()
    
    # Freeze BatchNorm in frozen layers
    if stage_config.freeze_bn:
        frozen_backbone = [l for l in LayerFreezer.BACKBONE_LAYERS 
                         if l not in stage_config.unfreeze_layers]
        freezer.freeze_batchnorm(frozen_backbone)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            logger.info(
                f"[Epoch {epoch}][{batch_idx+1}/{len(train_loader)}] "
                f"loss={total_loss/(batch_idx+1):.4f}"
            )
    
    return {
        "train_loss": total_loss / len(train_loader),
        "train_acc": correct / total,
    }


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    calibration_logger: Optional[CalibrationLogger] = None,
) -> Dict[str, float]:
    """Validate for one epoch, optionally logging logits."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())
            
            if calibration_logger:
                calibration_logger.record(outputs, labels, batch['sample_id'])
    
    # Compute AUC
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels.numpy(), torch.sigmoid(all_logits).numpy())
    except Exception:
        auc = 0.5
    
    return {
        "val_loss": total_loss / len(val_loader),
        "val_acc": correct / total,
        "val_auc": auc,
    }


def run_stage(
    stage: FinetuneStage,
    stage_config: StageConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: dict,
    checkpoint_manager: CheckpointManager,
    calibration_logger: CalibrationLogger,
    start_epoch: int = 1,
) -> Dict[str, Any]:
    """Run a single training stage.
    
    Returns:
        Dict with final metrics and best epoch
    """
    logger.info("=" * 70)
    logger.info(f"STARTING {stage_config.name}")
    logger.info("=" * 70)
    
    # Setup freezer
    freezer = LayerFreezer(model)
    
    # Freeze backbone, unfreeze specified layers
    freezer.freeze_backbone(except_layers=stage_config.unfreeze_layers)
    freezer.log_freeze_status()
    
    # Setup optimizer with differential LR
    param_groups = freezer.get_parameter_groups(
        head_lr=stage_config.head_lr,
        backbone_lr=stage_config.get_backbone_lr(),
        weight_decay=stage_config.weight_decay,
    )
    
    optimizer = torch.optim.AdamW(param_groups)
    
    # Loss with class weighting
    pos_weight = get_class_weights(train_loader.dataset)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=stage_config.epochs,
        eta_min=1e-7,
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(start_epoch, start_epoch + stage_config.epochs):
        logger.info(f"\n--- Epoch {epoch} ({stage.value}) ---")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, freezer, stage_config
        )
        
        # Validate
        calibration_logger.clear()
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, calibration_logger
        )
        
        # Combine metrics
        metrics = {**train_metrics, **val_metrics}
        
        logger.info(
            f"[Epoch {epoch}] "
            f"train_loss={metrics['train_loss']:.4f}, train_acc={metrics['train_acc']:.4f}, "
            f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc']:.4f}, "
            f"val_auc={metrics['val_auc']:.4f}"
        )
        
        # Record
        checkpoint_manager.record_metrics(epoch, stage.value, metrics)
        
        # Save logits for calibration
        calibration_logger.save(epoch, stage.value)
        
        # Checkpoint
        checkpoint_manager.save_checkpoint(
            model, optimizer, epoch, metrics, config, stage.value, scheduler
        )
        
        # Early stopping
        if metrics['val_loss'] < best_val_loss:
            best_val_loss = metrics['val_loss']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= stage_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        scheduler.step()
    
    # Save stage checkpoint
    checkpoint_manager.save_stage_checkpoint(model, stage.value, epoch, metrics)
    
    return {
        "stage": stage.value,
        "final_epoch": epoch,
        "final_metrics": metrics,
        "best_val_loss": best_val_loss,
    }


def main():
    args = parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    
    # Set seed
    seed = config.get('seed', 42)
    set_global_seed(seed)
    
    # Output directory
    output_dir = Path(args.output_dir or config.get('output_dir', 'artifacts/models/staged'))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Resume if specified
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create dataloaders
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(config)
    
    if args.dry_run:
        logger.info("Dry run complete. Setup successful.")
        return
    
    # Setup managers
    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        save_best_only=True,
        metric_name="val_loss",
        mode="min",
    )
    
    calibration_logger = CalibrationLogger(output_dir / "calibration")
    
    # Get stage configs
    stage_configs = get_stage_configs_from_config(config)
    
    # Map stage letters to enums
    stage_map = {
        'A': FinetuneStage.STAGE_A,
        'B': FinetuneStage.STAGE_B,
        'C': FinetuneStage.STAGE_C,
    }
    
    # Run stages
    results = []
    current_epoch = 1
    
    for stage_letter in args.stages:
        stage = stage_map[stage_letter]
        stage_config = stage_configs.get(stage, DEFAULT_STAGES[stage])
        
        result = run_stage(
            stage=stage,
            stage_config=stage_config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            checkpoint_manager=checkpoint_manager,
            calibration_logger=calibration_logger,
            start_epoch=current_epoch,
        )
        
        results.append(result)
        current_epoch = result['final_epoch'] + 1
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    
    for result in results:
        logger.info(
            f"Stage {result['stage']}: epochs={result['final_epoch']}, "
            f"best_val_loss={result['best_val_loss']:.4f}"
        )
    
    logger.info(f"\nBest checkpoint: {output_dir / 'best_model.pt'}")
    logger.info(f"Training history: {output_dir / 'training_history.json'}")
    logger.info(f"Calibration logits: {output_dir / 'calibration/'}")


if __name__ == "__main__":
    main()
