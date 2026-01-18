#!/usr/bin/env python
"""FF++ Training Script with Lazy Caching, Constrained Batching, and Deployment Realism Augmentations.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --override dataset.ffpp_root=/content/data/raw/ffpp

Audit Mode (verify Step 7 + Step 8):
    python train.py --config config.yaml --audit_steps 300 --audit_every 25 --dump_aug 32

Colab Usage:
    from google.colab import drive
    drive.mount('/content/drive')
    !python copy_to_local.py --source /content/drive/MyDrive/data/raw/ffpp --dest /content/data/raw/ffpp
    !python train.py --config config.yaml \
        --override dataset.ffpp_root=/content/data/raw/ffpp \
        --override caching.cache_dir=/content/cache/faces \
        --audit_steps 300 --audit_every 25 --dump_aug 32
"""

import argparse
import logging
import sys
import time
import os
import json
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from utils import (
    load_config,
    save_config,
    set_global_seed,
    write_runlog,
    FFppDataset,
    ConstrainedBatchSampler,
    DeploymentRealismAugmentation,
    ValidationTransform,
)
from utils.batch_sampler import validate_batch_constraints, print_batch_audit


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def parse_override(override_str: str) -> tuple:
    """Parse override string like 'dataset.ffpp_root=/content/data'."""
    key, value = override_str.split('=', 1)
    
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
    
    return key, value


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config."""
    for override in overrides:
        key, value = parse_override(override)
        parts = key.split('.')
        
        d = config
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        
        d[parts[-1]] = value
        logger.info(f"Override: {key} = {value}")
    
    return config


def create_dataloaders(config: dict, for_audit: bool = False) -> tuple:
    """Create train and validation dataloaders."""
    training_cfg = config.get('training', {})
    
    # Create transforms
    train_transform = DeploymentRealismAugmentation.from_config(config)
    val_transform = ValidationTransform(
        target_size=config.get('dataset', {}).get('crop_size', 256)
    )
    
    # Create datasets
    logger.info("Creating train dataset...")
    train_dataset = FFppDataset.from_config(config, split='train', transform=train_transform)
    
    logger.info("Creating validation dataset...")
    val_dataset = FFppDataset.from_config(config, split='val', transform=val_transform)
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Create constrained batch sampler for training
    logger.info("Creating constrained batch sampler...")
    train_sampler = ConstrainedBatchSampler.from_dataset(train_dataset, config)
    
    # For audit, use fewer workers to ensure determinism
    num_workers = 0 if for_audit else training_cfg.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=training_cfg.get('pin_memory', True) and not for_audit,
        collate_fn=collate_batch,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get('batch_size', 24),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=training_cfg.get('pin_memory', True) and not for_audit,
        collate_fn=collate_batch,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset, train_transform


def collate_batch(batch: list) -> Dict[str, Any]:
    """Custom collate function for batches."""
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


def dump_augmented_samples(
    batch: Dict[str, Any], 
    output_dir: Path, 
    count: int,
    batch_idx: int,
):
    """Dump augmented sample images for visual inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = batch['image']
    sample_ids = batch['sample_id']
    methods = batch['method']
    labels = batch['label'].tolist()
    
    for i in range(min(count, len(images))):
        img_tensor = images[i]
        sample_id = sample_ids[i]
        method = methods[i]
        label = int(labels[i])
        
        # Convert tensor to image
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        # Save with descriptive filename
        filename = f"batch{batch_idx:04d}_{sample_id}_{method}_label{label}.jpg"
        img.save(output_dir / filename, quality=95)
    
    logger.info(f"Dumped {min(count, len(images))} augmented samples to {output_dir}")


def run_audit_mode(
    config: dict,
    audit_steps: int,
    audit_every: int,
    dump_aug: int,
):
    """Run audit mode to verify Step 7 and Step 8 correctness.
    
    Args:
        config: Full config dict
        audit_steps: Number of steps to run
        audit_every: Print audit report every N batches
        dump_aug: Number of augmented samples to dump
    """
    logger.info("=" * 70)
    logger.info("AUDIT MODE: Verifying Step 7 (batching) + Step 8 (augmentations)")
    logger.info("=" * 70)
    
    # Create dataloaders in audit mode
    train_loader, _, train_dataset, _, train_transform = create_dataloaders(config, for_audit=True)
    
    batch_cfg = config.get('batch_sampling', {})
    max_per_video = batch_cfg.get('max_samples_per_video', 1)
    max_per_group = batch_cfg.get('max_samples_per_group', 1)
    
    # Statistics
    audit_results = {
        'total_batches': 0,
        'valid_batches': 0,
        'method_mixing_ok': 0,
        'video_violations': 0,
        'group_violations': 0,
        'group_relaxed': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'data_times': [],
        'step_times': [],
        'method_histogram': Counter(),
    }
    
    output_dir = Path('artifacts/aug_debug')
    aug_dumped = False
    
    start_time = time.time()
    prev_time = start_time
    
    logger.info(f"Running audit for {audit_steps} steps, reporting every {audit_every}...")
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= audit_steps:
            break
        
        step_start = time.time()
        data_time = step_start - prev_time
        
        # Build sample dicts for validation
        batch_samples = []
        for i in range(len(batch['sample_id'])):
            batch_samples.append({
                'method': batch['method'][i],
                'video_id': batch['video_id'][i],
                'group_id': batch['group_id'][i],
                'label': int(batch['label'][i].item()),
            })
        
        # Validate constraints
        valid, audit = validate_batch_constraints(
            batch_samples,
            require_mixing=batch_cfg.get('require_method_mixing', True),
            max_per_video=max_per_video,
            max_per_group=max_per_group,
        )
        
        # Collect stats
        audit_results['total_batches'] += 1
        if valid:
            audit_results['valid_batches'] += 1
        
        if audit['real_count'] > 0 and audit['fake_count'] > 0:
            audit_results['method_mixing_ok'] += 1
        
        if audit['video_duplicates'] > 0:
            audit_results['video_violations'] += 1
        
        if audit['group_duplicates'] > 0:
            audit_results['group_violations'] += 1
        
        # Track methods
        for method, count in audit['method_histogram'].items():
            audit_results['method_histogram'][method] += count
        
        # Timing
        step_time = time.time() - step_start
        audit_results['data_times'].append(data_time)
        audit_results['step_times'].append(step_time)
        
        # Dump augmented samples (first qualifying batch)
        if dump_aug > 0 and not aug_dumped:
            dump_augmented_samples(batch, output_dir, dump_aug, batch_idx)
            aug_dumped = True
        
        # Periodic audit report
        if (batch_idx + 1) % audit_every == 0:
            print_batch_audit(audit, batch_idx)
            
            # Cache stats
            cache_stats = train_dataset.cache_stats.get_stats()
            logger.info(
                f"[Step {batch_idx+1}] Cache: "
                f"hits={cache_stats['hits']}, misses={cache_stats['misses']}, "
                f"hit_rate={cache_stats['hit_rate']:.2%}"
            )
            
            # Timing stats
            avg_data_time = np.mean(audit_results['data_times'][-audit_every:])
            avg_step_time = np.mean(audit_results['step_times'][-audit_every:])
            logger.info(
                f"[Step {batch_idx+1}] Timing: "
                f"avg_data_time={avg_data_time:.3f}s, avg_step_time={avg_step_time:.3f}s"
            )
        
        prev_time = time.time()
    
    # Get sampler violations
    sampler = train_loader.batch_sampler
    sampler_violations = sampler.get_constraint_violations()
    audit_results['group_relaxed'] = sampler_violations.get('group_relaxed', 0)
    
    # Final cache stats
    cache_stats = train_dataset.cache_stats.get_stats()
    audit_results['cache_hits'] = cache_stats['hits']
    audit_results['cache_misses'] = cache_stats['misses']
    
    # Print summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    
    total = audit_results['total_batches']
    print(f"\nTotal batches audited: {total}")
    print(f"Valid batches: {audit_results['valid_batches']} ({100*audit_results['valid_batches']/total:.1f}%)")
    print(f"Method mixing OK: {audit_results['method_mixing_ok']} ({100*audit_results['method_mixing_ok']/total:.1f}%)")
    print(f"\n--- Constraint Violations ---")
    print(f"Video ID violations (>1 per batch): {audit_results['video_violations']}")
    print(f"Group ID violations (>1 per batch): {audit_results['group_violations']}")
    print(f"Sampler group relaxations: {audit_results['group_relaxed']}")
    
    print(f"\n--- Cache Statistics ---")
    print(f"Hits: {audit_results['cache_hits']}")
    print(f"Misses: {audit_results['cache_misses']}")
    total_cache = audit_results['cache_hits'] + audit_results['cache_misses']
    hit_rate = audit_results['cache_hits'] / total_cache if total_cache > 0 else 0
    print(f"Hit rate: {hit_rate:.2%}")
    
    print(f"\n--- Timing ---")
    print(f"Avg data_time: {np.mean(audit_results['data_times']):.3f}s")
    print(f"Avg step_time: {np.mean(audit_results['step_times']):.3f}s")
    
    print(f"\n--- Method Distribution ---")
    for method, count in sorted(audit_results['method_histogram'].items()):
        pct = 100 * count / sum(audit_results['method_histogram'].values())
        print(f"  {method}: {count} ({pct:.1f}%)")
    
    print(f"\n--- Augmentation Debug ---")
    if aug_dumped:
        print(f"Samples dumped to: {output_dir.absolute()}")
    else:
        print("No samples dumped (dump_aug=0 or no batches)")
    
    print("=" * 70)
    
    # Save audit report
    report_path = Path('artifacts/reports/audit_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'audit_steps': audit_steps,
        'total_batches': total,
        'valid_batches': audit_results['valid_batches'],
        'method_mixing_ok': audit_results['method_mixing_ok'],
        'video_violations': audit_results['video_violations'],
        'group_violations': audit_results['group_violations'],
        'group_relaxed': audit_results['group_relaxed'],
        'cache_hits': audit_results['cache_hits'],
        'cache_misses': audit_results['cache_misses'],
        'cache_hit_rate': hit_rate,
        'avg_data_time': float(np.mean(audit_results['data_times'])),
        'avg_step_time': float(np.mean(audit_results['step_times'])),
        'method_histogram': dict(audit_results['method_histogram']),
        'aug_dump_path': str(output_dir.absolute()) if aug_dumped else None,
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAudit report saved to: {report_path}")
    
    return report


def log_cache_stats(dataset: FFppDataset, epoch: int):
    """Log cache statistics."""
    stats = dataset.cache_stats.get_stats()
    logger.info(
        f"[Epoch {epoch}] Cache stats: "
        f"hits={stats['hits']}, misses={stats['misses']}, "
        f"hit_rate={stats['hit_rate']:.2%}"
    )


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: dict,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    log_interval = config.get('logging', {}).get('log_every_steps', 50)
    
    start_time = time.time()
    data_time = 0.0
    batch_end = start_time
    
    for batch_idx, batch in enumerate(train_loader):
        data_end = time.time()
        if batch_idx > 0:
            data_time += data_end - batch_end
        
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if outputs.dim() > 1:
            outputs = outputs.squeeze(-1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        grad_clip = config.get('training', {}).get('gradient_clip_norm', 1.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        batch_end = time.time()
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            elapsed = time.time() - start_time
            throughput = (batch_idx + 1) * len(images) / elapsed
            
            logger.info(
                f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                f"loss={avg_loss:.4f} throughput={throughput:.1f} samples/s "
                f"data_wait={data_time:.1f}s"
            )
    
    return total_loss / max(num_batches, 1)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            if outputs.dim() > 1:
                outputs = outputs.squeeze(-1)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / max(len(val_loader), 1)
    accuracy = correct / max(total, 1)
    
    return avg_loss, accuracy


def create_model(config: dict) -> nn.Module:
    """Create model from config."""
    from torchvision import models
    
    model_cfg = config.get('model', {})
    architecture = model_cfg.get('architecture', 'ladeda_resnet50')
    
    if 'resnet50' in architecture:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    freeze_layers = model_cfg.get('freeze_layers', [])
    for name, param in model.named_parameters():
        for freeze_name in freeze_layers:
            if freeze_name in name:
                param.requires_grad = False
                break
    
    return model


def main():
    parser = argparse.ArgumentParser(description="FF++ Training with Caching & Constraints")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--override', action='append', default=[], help='Config overrides (key=value)')
    parser.add_argument('--dry-run', action='store_true', help='Only create dataloaders, no training')
    parser.add_argument('--validate-batches', type=int, default=0, help='Validate N batches and exit')
    
    # Audit mode arguments
    parser.add_argument('--audit_steps', type=int, default=0, help='Run audit mode for N steps')
    parser.add_argument('--audit_every', type=int, default=25, help='Print audit report every N batches')
    parser.add_argument('--dump_aug', type=int, default=0, help='Dump N augmented samples to artifacts/aug_debug/')
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    
    # Set seed
    seed = config.get('seed', 42)
    set_global_seed(seed)
    logger.info(f"Set global seed: {seed}")
    
    # Audit mode
    if args.audit_steps > 0:
        run_audit_mode(config, args.audit_steps, args.audit_every, args.dump_aug)
        return
    
    # Create dataloaders
    train_loader, val_loader, train_dataset, val_dataset, _ = create_dataloaders(config)
    
    # Batch validation mode
    if args.validate_batches > 0:
        from utils.batch_sampler import validate_batch_constraints, print_batch_audit
        
        logger.info(f"Validating {args.validate_batches} batches...")
        valid_count = 0
        for i, batch in enumerate(train_loader):
            if i >= args.validate_batches:
                break
            
            batch_samples = []
            for j in range(len(batch['sample_id'])):
                batch_samples.append({
                    'method': batch['method'][j],
                    'video_id': batch['video_id'][j],
                    'group_id': batch['group_id'][j],
                    'label': int(batch['label'][j].item()),
                })
            
            valid, audit = validate_batch_constraints(batch_samples)
            valid_count += int(valid)
            print_batch_audit(audit, i)
        
        logger.info(f"Validation complete: {valid_count}/{args.validate_batches} valid batches")
        return
    
    if args.dry_run:
        logger.info("Dry run complete. Dataloaders created successfully.")
        return
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = create_model(config)
    model = model.to(device)
    
    training_cfg = config.get('training', {})
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 3e-5),
        weight_decay=training_cfg.get('weight_decay', 0.001),
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    max_epochs = training_cfg.get('max_epochs', 30)
    best_val_loss = float('inf')
    
    logger.info(f"Starting training for {max_epochs} epochs")
    
    for epoch in range(1, max_epochs + 1):
        train_dataset.set_epoch(epoch)
        train_loader.batch_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config
        )
        
        log_cache_stats(train_dataset, epoch)
        
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        logger.info(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_dir = Path(config.get('output_dir', 'artifacts/models'))
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
