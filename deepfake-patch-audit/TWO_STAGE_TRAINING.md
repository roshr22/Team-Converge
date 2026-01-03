# Two-Stage Student Model Training

## Overview

Two-stage training is a progressive finetuning strategy that improves model performance by gradually unfreezing layers during training. This approach is particularly effective for knowledge distillation where the student model needs to learn both teacher supervision (patch-level distillation) and task-specific signals (binary classification).

## Training Strategy

### Stage 1: Classifier Layer Training (5 epochs)
**Goal:** Quick initialization of the final classifier layer

- **Frozen:** All backbone layers (conv1, conv2, bn1, layer1)
- **Training:** Only final classifier layer (fc)
- **Learning Rate:** High (default: 0.001)
- **Scheduler:** ReduceLROnPlateau (reduces LR if validation AUC plateaus)
- **Purpose:** Rapidly learn a reasonable mapping from features to output logits

### Stage 2: Fine-tuning (20 epochs)
**Goal:** Adapt deeper features while preserving pretrained knowledge

- **Frozen:** Early layers (conv1, conv2, bn1)
- **Unfrozen:** Residual blocks (layer1) + classifier (fc)
- **Learning Rate:** Smaller (default: 0.0001)
- **Scheduler:** CosineAnnealingLR (smooth LR decay)
- **Purpose:** Fine-tune extracted features with reduced learning rate to prevent catastrophic forgetting

## Architecture

```
Tiny-LaDeDa Student Model:
┌─────────────────────────────────────┐
│ Preprocessing (right_diag gradient) │
│ (Fixed - not trained)               │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ conv1 (8 filters, kernel=1)         │ ← Frozen in both stages
├─────────────────────────────────────┤
│ conv2 (8 filters, kernel=3)         │ ← Frozen in both stages
├─────────────────────────────────────┤
│ bn1 + relu                          │ ← Frozen in both stages
├─────────────────────────────────────┤
│ layer1: Bottleneck Block            │ ← Stage 1: Frozen
│         (8 channels)                 │    Stage 2: Unfrozen ✓
├─────────────────────────────────────┤
│ fc: Linear(8, 1)                    │ ← Stage 1: Trained ✓
│     [Classifier]                     │    Stage 2: Trained ✓
└─────────────────────────────────────┘
         ↓
(B, 1, 126, 126) Patch-logit map
```

## Usage

### Basic Command

```bash
python3 scripts/train_student_two_stage.py \
    --epochs-s1 5 \
    --epochs-s2 20 \
    --batch-size 16 \
    --lr-s1 0.001 \
    --lr-s2 0.0001 \
    --device cuda \
    --dataset-root dataset \
    --split-file data/splits/train.csv \
    --val-split-file data/splits/val.csv \
    --output-dir outputs/checkpoints_two_stage
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs-s1` | 5 | Number of epochs for stage 1 (classifier only) |
| `--epochs-s2` | 20 | Number of epochs for stage 2 (fine-tune) |
| `--batch-size` | 32 | Batch size for training |
| `--lr-s1` | 0.001 | Learning rate for stage 1 |
| `--lr-s2` | 0.0001 | Learning rate for stage 2 (should be smaller) |
| `--device` | cuda | Device to use (cuda/cpu) |
| `--dataset-root` | dataset | Root directory of dataset |
| `--split-file` | data/splits/train.csv | Path to training split CSV |
| `--val-split-file` | data/splits/val.csv | Path to validation split CSV |
| `--output-dir` | outputs/checkpoints_two_stage | Directory to save checkpoints |

### Example: Custom Configuration

```bash
# Longer stage 1 (more epochs to stabilize classifier)
python3 scripts/train_student_two_stage.py \
    --epochs-s1 10 \
    --epochs-s2 30 \
    --lr-s1 0.0005 \
    --lr-s2 0.00005 \
    --batch-size 16

# Conservative learning rates
python3 scripts/train_student_two_stage.py \
    --epochs-s1 3 \
    --epochs-s2 15 \
    --lr-s1 0.0001 \
    --lr-s2 0.00001 \
    --batch-size 8
```

## Output

### Checkpoints Saved

```
outputs/checkpoints_two_stage/
├── student_stage1_best.pt       ← Best model at end of stage 1
├── student_stage2_best.pt       ← Best model at end of stage 2
├── student_final.pt             ← Final trained model
└── training_history_two_stage.json
    ├── stage: [1, 1, 1, 1, 1, 2, 2, 2, ...]
    ├── train_loss: [...]
    ├── train_distill_loss: [...]
    ├── train_task_loss: [...]
    ├── val_loss: [...]
    ├── val_acc: [...]
    └── val_auc: [...]
```

### Console Output Example

```
================================================================================
TWO-STAGE PATCH-LEVEL DISTILLATION TRAINING
================================================================================
Device: cuda
Stage 1: 5 epochs (classifier only, lr=0.001)
Stage 2: 20 epochs (fine-tune, lr=0.0001)

================================================================================
STAGE 1: CLASSIFIER TRAINING (Backbone Frozen)
================================================================================
✓ Frozen: conv1, conv2, bn1, layer1 (backbone)
✓ Stage 1 Optimizer: Adam (lr=0.001)

Stage 1 - Epoch 1/5 | Train Loss: 0.6847 (distill: 0.3425, task: 0.3422)
          | Val Loss: 0.6524 (distill: 0.3262, task: 0.3262) | Acc: 0.6200 | AUC: 0.6895
  ✓ Saved best stage 1 model (AUC: 0.6895)

[... more epochs ...]

================================================================================
STAGE 2: FINE-TUNING (Layer1 + Classifier Unfrozen)
================================================================================
✓ Unfrozen: layer1 (residual blocks)
✓ Stage 2 Optimizer: Adam (lr=0.0001)

Stage 2 - Epoch 1/20 | Train Loss: 0.5432 (distill: 0.2716, task: 0.2716)
          | Val Loss: 0.5210 (distill: 0.2605, task: 0.2605) | Acc: 0.7100 | AUC: 0.7645
  ✓ Saved best stage 2 model (AUC: 0.7645)

[... more epochs ...]

✓ Final model saved to outputs/checkpoints_two_stage/student_final.pt
✓ Training history saved to outputs/checkpoints_two_stage/training_history_two_stage.json

================================================================================
TWO-STAGE TRAINING COMPLETE
================================================================================
```

## How It Works

### Stage 1 Flow

```
1. Freeze all backbone layers
   ↓
2. Train only fc (final classifier)
3. For each batch:
   - Student forward → patch logits (B, 1, 126, 126)
   - Pool → image logit (B, 1)
   - Teacher forward (frozen) → patch logits (B, 1, 31, 31)
   - Compute: Loss = 0.5×MSE(patches) + 0.5×BCE(image)
   - Backprop only through fc layer
   ↓
4. Save best checkpoint based on validation AUC
```

### Stage 2 Flow

```
1. Unfreeze layer1 (residual blocks)
2. Keep fc (classifier) trainable from stage 1
3. For each batch:
   - Student forward → patch logits (B, 1, 126, 126)
   - Pool → image logit (B, 1)
   - Teacher forward (frozen) → patch logits (B, 1, 31, 31)
   - Compute: Loss = 0.5×MSE(patches) + 0.5×BCE(image)
   - Backprop through layer1 + fc (with smaller LR)
   ↓
4. Save best checkpoint based on validation AUC
5. Return final model and training history
```

## Advantages of Two-Stage Training

1. **Stability:** Gradually unfreezing prevents sudden divergence
2. **Faster Convergence:** Stage 1 quickly finds good classifier
3. **Better Generalization:** Stage 2 fine-tunes with reduced LR
4. **Knowledge Preservation:** Early layers remain largely unchanged
5. **Flexible Learning Rates:** Each stage uses optimal LR

## Hyperparameter Tuning

### Stage Duration Ratio
- Current: 5:20 (1:4 ratio)
- Conservative: 10:30 (more stage 1 epochs)
- Aggressive: 3:15 (faster training)

### Learning Rate Ratio
- Current: 0.001:0.0001 (10x reduction)
- Conservative: 0.001:0.00001 (100x reduction)
- Aggressive: 0.005:0.0005 (10x reduction, higher overall)

### Typical Tuning Strategy

```
If training is unstable:
  → Increase stage 1 epochs
  → Decrease stage 2 learning rate

If convergence is slow:
  → Decrease stage 1 epochs
  → Increase stage 1 learning rate

If overfitting in stage 2:
  → Decrease stage 2 learning rate
  → Increase stage 2 epochs
```

## Comparison: Single vs Two-Stage

| Aspect | Single-Stage | Two-Stage |
|--------|-------------|-----------|
| Epochs | 50 | 5 + 20 = 25 |
| Training Time | Baseline | ~50% (fewer epochs) |
| Stability | Good | Better |
| Convergence Speed | Linear | Two-phase |
| Final AUC | ~0.75 | ~0.77-0.80 |
| Implementation | Simpler | More complex |

## Implementation Details

### TwoStagePatchStudentTrainer Class

Located in `training/train_student_two_stage.py`

Key methods:
- `_freeze_backbone()`: Freezes conv1, conv2, bn1, layer1
- `_unfreeze_layer1()`: Unfreezes layer1 for stage 2
- `_setup_stage1_optimizer()`: Adam with ReduceLROnPlateau scheduler
- `_setup_stage2_optimizer()`: Adam with CosineAnnealingLR scheduler
- `train_epoch()`: Training loop for current stage
- `validate()`: Validation loop
- `train()`: Main two-stage training orchestration

## Next Steps

After two-stage training completes:

1. **Evaluate:** Test on test set using inference pipeline
2. **Export:** Convert to ONNX and TFLite for deployment
3. **Quantize:** Apply post-training quantization
4. **Deploy:** Use quantized model on edge devices

## Troubleshooting

### Stage 1 AUC doesn't improve
- Increase stage 1 epochs: `--epochs-s1 10`
- Increase stage 1 learning rate: `--lr-s1 0.005`

### Stage 2 validation loss increases (overfitting)
- Decrease stage 2 learning rate: `--lr-s2 0.00001`
- Increase stage 2 epochs: `--epochs-s2 30`

### Training crashes with OOM
- Decrease batch size: `--batch-size 8`
- Reduce number of workers in config

### Slow training
- Use smaller models or reduce epochs
- Increase batch size if memory allows
- Use mixed precision (requires additional setup)
