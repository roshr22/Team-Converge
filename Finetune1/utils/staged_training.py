"""Staged fine-tuning utilities for Step 9.

Implements micro-stepped training with:
- Stage A: Head-only stabilization
- Stage B: Partial unfreeze (layer4 only)
- Stage C: Optional deeper unfreeze (layer3+layer4)

Key features:
- Differential learning rates (backbone LR = head LR / ratio)
- BatchNorm freezing when layers are frozen
- Loss/calibration utilities
- Checkpoint management with config/git hash
"""

import logging
import subprocess
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path

import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class FinetuneStage(Enum):
    """Fine-tuning stages."""
    STAGE_A = "head_only"       # Freeze backbone, train head only
    STAGE_B = "partial"         # Unfreeze layer4
    STAGE_C = "deeper"          # Unfreeze layer3 + layer4


@dataclass
class StageConfig:
    """Configuration for a fine-tuning stage."""
    name: str
    epochs: int
    unfreeze_layers: List[str]      # Which layers to unfreeze
    head_lr: float                   # Learning rate for head
    backbone_lr_ratio: float = 0.1   # backbone_lr = head_lr * ratio
    weight_decay: float = 0.001
    freeze_bn: bool = True           # Freeze BN in frozen layers
    early_stopping_patience: int = 3
    
    def get_backbone_lr(self) -> float:
        return self.head_lr * self.backbone_lr_ratio


# Default stage configurations
DEFAULT_STAGES = {
    FinetuneStage.STAGE_A: StageConfig(
        name="Stage A: Head-only stabilization",
        epochs=2,
        unfreeze_layers=[],  # Everything frozen except head
        head_lr=1e-3,
        backbone_lr_ratio=0.0,  # Backbone completely frozen
        freeze_bn=True,
        early_stopping_patience=2,
    ),
    FinetuneStage.STAGE_B: StageConfig(
        name="Stage B: Partial unfreeze (layer4)",
        epochs=8,
        unfreeze_layers=["layer4"],
        head_lr=3e-5,
        backbone_lr_ratio=0.1,  # backbone_lr = 3e-6
        freeze_bn=True,
        early_stopping_patience=5,
    ),
    FinetuneStage.STAGE_C: StageConfig(
        name="Stage C: Deeper unfreeze (layer3+layer4)",
        epochs=5,
        unfreeze_layers=["layer3", "layer4"],
        head_lr=1e-5,
        backbone_lr_ratio=0.05,  # backbone_lr = 5e-7
        freeze_bn=True,
        early_stopping_patience=3,
    ),
}


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class LayerFreezer:
    """Manages layer freezing/unfreezing with proper BatchNorm handling."""
    
    # ResNet50 layer names
    BACKBONE_LAYERS = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    HEAD_LAYERS = ["fc", "attention", "patch_head", "pooling"]
    
    def __init__(self, model: nn.Module):
        self.model = model
        self._frozen_layers: Set[str] = set()
        self._bn_frozen: Set[str] = set()
    
    def freeze_backbone(self, except_layers: Optional[List[str]] = None):
        """Freeze all backbone layers except specified ones.
        
        Args:
            except_layers: Layers to keep unfrozen (e.g., ["layer4"])
        """
        except_layers = except_layers or []
        
        for name, module in self.model.named_modules():
            layer_name = self._get_layer_name(name)
            
            if layer_name in self.BACKBONE_LAYERS and layer_name not in except_layers:
                self._freeze_module(name, module)
        
        logger.info(f"Frozen backbone layers: {self._frozen_layers}")
        logger.info(f"Unfrozen layers: {except_layers}")
    
    def unfreeze_layers(self, layers: List[str]):
        """Unfreeze specific layers.
        
        Args:
            layers: List of layer names to unfreeze (e.g., ["layer4"])
        """
        for name, module in self.model.named_modules():
            layer_name = self._get_layer_name(name)
            
            if layer_name in layers:
                self._unfreeze_module(name, module)
        
        # Update frozen set
        for layer in layers:
            self._frozen_layers.discard(layer)
        
        logger.info(f"Unfroze layers: {layers}")
    
    def freeze_batchnorm(self, frozen_layers: Optional[List[str]] = None):
        """Set BatchNorm layers to eval mode in frozen layers.
        
        This prevents running stats from drifting during training.
        
        Args:
            frozen_layers: Which layers' BNs to freeze. If None, use self._frozen_layers.
        """
        frozen_layers = frozen_layers or list(self._frozen_layers)
        bn_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                layer_name = self._get_layer_name(name)
                if layer_name in frozen_layers:
                    module.eval()
                    # Prevent parameters from being updated
                    for param in module.parameters():
                        param.requires_grad = False
                    self._bn_frozen.add(name)
                    bn_count += 1
        
        logger.info(f"Froze {bn_count} BatchNorm layers in: {frozen_layers}")
        return bn_count
    
    def get_parameter_groups(
        self, 
        head_lr: float, 
        backbone_lr: float,
        weight_decay: float = 0.001,
    ) -> List[Dict]:
        """Get parameter groups with differential learning rates.
        
        Args:
            head_lr: Learning rate for head/new layers
            backbone_lr: Learning rate for backbone layers
            weight_decay: Weight decay for all params
            
        Returns:
            List of parameter group dicts for optimizer
        """
        head_params = []
        backbone_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            layer_name = self._get_layer_name(name)
            
            if layer_name in self.BACKBONE_LAYERS:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = []
        
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "name": "head",
            })
        
        if backbone_params and backbone_lr > 0:
            param_groups.append({
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "name": "backbone",
            })
        
        logger.info(
            f"Parameter groups: head={len(head_params)} params @ lr={head_lr:.2e}, "
            f"backbone={len(backbone_params)} params @ lr={backbone_lr:.2e}"
        )
        
        return param_groups
    
    def log_freeze_status(self):
        """Log current freeze status for debugging."""
        trainable = 0
        frozen = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable += param.numel()
            else:
                frozen += param.numel()
        
        total = trainable + frozen
        logger.info(
            f"[FREEZE STATUS] Trainable: {trainable:,} ({100*trainable/total:.1f}%) | "
            f"Frozen: {frozen:,} ({100*frozen/total:.1f}%)"
        )
        logger.info(f"[FREEZE STATUS] Frozen BN layers: {len(self._bn_frozen)}")
    
    def _get_layer_name(self, module_name: str) -> str:
        """Extract layer name from full module path."""
        parts = module_name.split(".")
        if parts:
            return parts[0]
        return module_name
    
    def _freeze_module(self, name: str, module: nn.Module):
        """Freeze a module's parameters."""
        for param in module.parameters():
            param.requires_grad = False
        
        layer_name = self._get_layer_name(name)
        self._frozen_layers.add(layer_name)
    
    def _unfreeze_module(self, name: str, module: nn.Module):
        """Unfreeze a module's parameters."""
        for param in module.parameters():
            param.requires_grad = True


class CheckpointManager:
    """Manages checkpointing with config and git hash."""
    
    def __init__(
        self, 
        output_dir: Path,
        save_best_only: bool = True,
        metric_name: str = "val_loss",
        mode: str = "min",  # "min" or "max"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.mode = mode
        
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.history: List[Dict] = []
    
    def is_better(self, metric: float) -> bool:
        """Check if metric is better than best."""
        if self.mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict,
        stage: str,
        scheduler = None,
    ) -> Optional[Path]:
        """Save checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Dict of metrics (must include self.metric_name)
            config: Full config dict
            stage: Current training stage name
            scheduler: Optional scheduler state
            
        Returns:
            Path to saved checkpoint, or None if not saved
        """
        metric_value = metrics.get(self.metric_name, float("inf"))
        
        # Check if we should save
        if self.save_best_only and not self.is_better(metric_value):
            return None
        
        # Update best
        self.best_metric = metric_value
        self.best_epoch = epoch
        
        # Build checkpoint
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
            "git_commit": get_git_commit_hash(),
            "config": config,
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        
        # Save
        checkpoint_path = self.output_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save config separately as JSON
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        # Log
        logger.info(
            f"[CHECKPOINT] Saved best model @ epoch {epoch} "
            f"({self.metric_name}={metric_value:.4f})"
        )
        
        return checkpoint_path
    
    def save_stage_checkpoint(
        self,
        model: nn.Module,
        stage: str,
        epoch: int,
        metrics: Dict[str, float],
    ):
        """Save checkpoint at end of stage."""
        checkpoint_path = self.output_dir / f"stage_{stage}_epoch{epoch}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "stage": stage,
            "epoch": epoch,
            "metrics": metrics,
        }, checkpoint_path)
        logger.info(f"[CHECKPOINT] Saved stage checkpoint: {checkpoint_path}")
    
    def record_metrics(self, epoch: int, stage: str, metrics: Dict[str, float]):
        """Record metrics history."""
        self.history.append({
            "epoch": epoch,
            "stage": stage,
            **metrics,
        })
        
        # Save history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(
        self, 
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        path: Optional[Path] = None,
    ) -> Dict:
        """Load checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optional optimizer to load state into
            path: Checkpoint path (defaults to best_model.pt)
            
        Returns:
            Checkpoint dict with metadata
        """
        path = path or (self.output_dir / "best_model.pt")
        
        checkpoint = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        logger.info(
            f"[CHECKPOINT] Loaded from {path} "
            f"(epoch={checkpoint.get('epoch')}, stage={checkpoint.get('stage')})"
        )
        
        return checkpoint


class CalibrationLogger:
    """Logs logits for threshold calibration on validation set."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logits: List[float] = []
        self.labels: List[int] = []
        self.sample_ids: List[str] = []
    
    def record(self, logits: torch.Tensor, labels: torch.Tensor, sample_ids: List[str]):
        """Record batch of logits."""
        self.logits.extend(logits.detach().cpu().tolist())
        self.labels.extend(labels.detach().cpu().int().tolist())
        self.sample_ids.extend(sample_ids)
    
    def save(self, epoch: int, stage: str):
        """Save logits to file."""
        output = {
            "epoch": epoch,
            "stage": stage,
            "logits": self.logits,
            "labels": self.labels,
            "sample_ids": self.sample_ids,
        }
        
        path = self.output_dir / f"val_logits_epoch{epoch}.json"
        with open(path, "w") as f:
            json.dump(output, f)
        
        logger.info(f"[CALIBRATION] Saved {len(self.logits)} val logits to {path}")
    
    def clear(self):
        """Clear accumulated logits."""
        self.logits = []
        self.labels = []
        self.sample_ids = []


def get_stage_configs_from_config(config: dict) -> Dict[FinetuneStage, StageConfig]:
    """Build stage configs from config.yaml."""
    stages_cfg = config.get("finetuning_stages", {})
    
    stages = {}
    
    # Stage A
    stage_a_cfg = stages_cfg.get("stage_a", {})
    stages[FinetuneStage.STAGE_A] = StageConfig(
        name="Stage A: Head-only stabilization",
        epochs=stage_a_cfg.get("epochs", 2),
        unfreeze_layers=[],
        head_lr=stage_a_cfg.get("head_lr", 1e-3),
        backbone_lr_ratio=0.0,
        freeze_bn=stage_a_cfg.get("freeze_bn", True),
        early_stopping_patience=stage_a_cfg.get("patience", 2),
    )
    
    # Stage B
    stage_b_cfg = stages_cfg.get("stage_b", {})
    stages[FinetuneStage.STAGE_B] = StageConfig(
        name="Stage B: Partial unfreeze (layer4)",
        epochs=stage_b_cfg.get("epochs", 8),
        unfreeze_layers=stage_b_cfg.get("unfreeze", ["layer4"]),
        head_lr=stage_b_cfg.get("head_lr", 3e-5),
        backbone_lr_ratio=stage_b_cfg.get("backbone_lr_ratio", 0.1),
        freeze_bn=stage_b_cfg.get("freeze_bn", True),
        early_stopping_patience=stage_b_cfg.get("patience", 5),
    )
    
    # Stage C
    stage_c_cfg = stages_cfg.get("stage_c", {})
    stages[FinetuneStage.STAGE_C] = StageConfig(
        name="Stage C: Deeper unfreeze (layer3+layer4)",
        epochs=stage_c_cfg.get("epochs", 5),
        unfreeze_layers=stage_c_cfg.get("unfreeze", ["layer3", "layer4"]),
        head_lr=stage_c_cfg.get("head_lr", 1e-5),
        backbone_lr_ratio=stage_c_cfg.get("backbone_lr_ratio", 0.05),
        freeze_bn=stage_c_cfg.get("freeze_bn", True),
        early_stopping_patience=stage_c_cfg.get("patience", 3),
    )
    
    return stages


if __name__ == "__main__":
    print("Staged fine-tuning utilities loaded!")
    print(f"Default stages: {list(DEFAULT_STAGES.keys())}")
