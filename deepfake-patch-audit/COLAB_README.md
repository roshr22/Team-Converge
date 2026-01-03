# Google Colab Notebook Guide

## File Location
```
colab_training_inference_deployment.ipynb
```

## ✅ Notebook Status
- **Format**: Valid Jupyter Notebook (.ipynb)
- **Size**: 44.8 KB
- **Cells**: 16 (3 markdown, 13 code)
- **Status**: Ready for Google Colab ✓

## 📋 Cell Breakdown

### Setup Cells (1-4)
- **Cell 1**: Check GPU availability
- **Cell 2**: Install required packages
- **Cell 3**: Mount Google Drive and configure paths
- **Cell 4**: Import libraries and set device

### Core Components (5-7)
- **Cell 5**: `BaseDataset` class - Dataset loading
- **Cell 6**: `TinyLaDeDa` model - Student architecture
- **Cell 7**: `TopKLogitPooling` and `PatchDistillationLoss`

### Training (8-9)
- **Cell 8**: `PatchStudentTrainer` - Single-stage training
- **Cell 9**: `TwoStagePatchStudentTrainer` - Two-stage fine-tuning

### Inference & Deployment (10-11)
- **Cell 10**: `InferencePipeline` - Inference + heatmap visualization
- **Cell 11**: `ImprovedDynamicRangeQuantizer` - Int8 quantization

### Utilities (12-13)
- **Cell 12**: Data loading and model initialization utilities
- **Cell 13**: Helper functions for visualization and analysis

### Documentation (14-15)
- **Cell 14**: Usage examples
- **Cell 15**: Summary and quick reference

---

## 🚀 Quick Start Guide

### Step 1: Open in Google Colab
1. Go to https://colab.research.google.com
2. Upload the notebook OR
3. Connect to GitHub and select the file

### Step 2: Run Setup Cells
```
Cell 1 → Cell 2 → Cell 3 → Cell 4
```
This installs dependencies and mounts your Google Drive.

### Step 3: Prepare Dataset
Create the following structure in Google Drive:
```
MyDrive/
└── deepfake-patch-audit/
    └── data/
        ├── train/
        │   ├── real/   (your real images)
        │   └── fake/   (your fake images)
        └── val/
            ├── real/   (your real images)
            └── fake/   (your fake images)
```

### Step 4: Load Data and Models
```python
# Cell 12 provides utilities
dataset_root = "/content/drive/MyDrive/deepfake-patch-audit/data"
train_loader, val_loader = create_sample_data_loaders(dataset_root, batch_size=16)
student, teacher, pooling, criterion = initialize_training_pipeline()
```

### Step 5: Choose Training Method

**Option A: Single-Stage Training**
```python
trainer = PatchStudentTrainer(student, teacher, train_loader, val_loader,
                             criterion, pooling, device=DEVICE)
history = trainer.train(epochs=20)
```

**Option B: Two-Stage Fine-Tuning (Recommended)**
```python
trainer = TwoStagePatchStudentTrainer(student, teacher, train_loader, val_loader,
                                     criterion, pooling, device=DEVICE)
history = trainer.train(epochs_s1=5, epochs_s2=20)
```

### Step 6: Run Inference
```python
pipeline = InferencePipeline(student, pooling, device=DEVICE)
result = pipeline.predict("path/to/image.jpg")

# Visualize
fig = pipeline.visualize_heatmap("path/to/image.jpg", result)
plt.show()
```

### Step 7: Quantize & Export
```python
quantizer = ImprovedDynamicRangeQuantizer(bits=8, per_channel=True)
quantized_model, params = quantizer.quantize_model(student)
print(quantizer.get_quantization_report(params))
```

---

## 📊 All Available Classes

| Class | Purpose |
|-------|---------|
| `BaseDataset` | Load images (directory or CSV mode) |
| `TinyLaDeDa` | Ultra-lightweight model (1,297 params) |
| `TopKLogitPooling` | Aggregate patch logits to image level |
| `PatchDistillationLoss` | MSE (patch) + BCE (image) loss |
| `PatchStudentTrainer` | Single-stage training |
| `TwoStagePatchStudentTrainer` | Two-stage fine-tuning with progressive unfreezing |
| `InferencePipeline` | Inference with patch-level heatmaps |
| `ImprovedDynamicRangeQuantizer` | Per-channel int8 quantization |

---

## 🔧 Utilities

| Function | Purpose |
|----------|---------|
| `create_sample_data_loaders()` | Create train/val dataloaders |
| `initialize_training_pipeline()` | Setup models, pooling, loss |
| `plot_training_history()` | Visualize loss/acc/auc curves |
| `print_model_summary()` | Display model architecture |
| `compute_model_size()` | Calculate model size in MB |

---

## 💾 Checkpoint Files

Training automatically saves:
```
outputs/
└── checkpoints/  (single-stage)
    ├── student_best.pt
    └── student_final.pt

outputs/
└── checkpoints_two_stage/  (two-stage)
    ├── student_best.pt
    └── student_final.pt
```

Download these from Colab after training completes.

---

## ⚙️ Hyperparameters You Can Modify

### Data Loading
```python
batch_size = 16
image_size = 256  # Always use 256x256 with bicubic
num_workers = 0   # Keep 0 for Colab
```

### Training
```python
# Single-stage
trainer.train(epochs=20)

# Two-stage
trainer.train(epochs_s1=5, epochs_s2=20)  # 5 epochs freeze, 20 epochs fine-tune
```

### Learning Rates
```python
# Two-stage trainer
TwoStagePatchStudentTrainer(
    ...,
    stage1_lr=0.001,      # Freeze backbone, train classifier
    stage2_lr=0.0001,     # Unfreeze, fine-tune with smaller LR
    weight_decay=1e-4
)
```

### Pooling
```python
TopKLogitPooling(k_percent=0.1)  # Use top 10% of patches
```

### Quantization
```python
ImprovedDynamicRangeQuantizer(
    bits=8,
    symmetric=False,        # Asymmetric (better for weights)
    per_channel=True,       # Per-output-channel (better accuracy)
    clip_outliers=True,
    clip_percentile=99.9    # Clip extreme outliers
)
```

---

## 📈 Expected Performance

After training with typical hyperparameters:

| Metric | Value |
|--------|-------|
| Model Size | ~10 KB (PyTorch), ~8 KB (ONNX), ~5 KB (TFLite) |
| Inference Speed (CPU) | ~300-500 ms |
| Inference Speed (GPU) | ~10-50 ms |
| Validation AUC | 0.85-0.95 (depends on data) |

---

## ⚠️ Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution**: Run Cell 2 to install dependencies

### Issue: "Google Drive not mounted"
**Solution**: Run Cell 3 and authenticate when prompted

### Issue: "Dataset not found"
**Solution**:
1. Check your dataset path matches the expected structure
2. Ensure files are in MyDrive/deepfake-patch-audit/data/
3. Verify real/fake subdirectories exist

### Issue: "Out of memory (OOM)"
**Solution**: Reduce batch_size from 16 to 8 or 4

### Issue: "CUDA out of memory"
**Solution**:
- Reduce batch size
- Use CPU device instead: `DEVICE = 'cpu'`

---

## 📝 Notes

- All images are resized to **256×256 with bicubic interpolation** (deterministic, no augmentation)
- Images are normalized with **ImageNet mean/std** [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
- Output is a **(B, 1, 126, 126) patch-logit map** (one logit per spatial location)
- Sigmoid is applied for probability: **P(Fake) = sigmoid(logit)**
- Two-stage training is **strongly recommended** for better accuracy

---

## 🎯 Tips for Best Results

1. **Dataset Quality**: Ensure good real/fake examples
2. **Batch Size**: Use 16 or higher if GPU memory allows
3. **Epochs**:
   - Single-stage: 20-50 epochs
   - Two-stage: 5 epochs (stage 1) + 20 epochs (stage 2)
4. **Monitoring**: Watch validation AUC - stop if it plateaus
5. **Checkpoints**: Always save best model (automatic in trainer)

---

## 🔗 File Paths in Colab

```
/content/drive/MyDrive/deepfake-patch-audit/
├── colab_training_inference_deployment.ipynb
├── data/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── val/
│       ├── real/
│       └── fake/
└── outputs/  (created during training)
    ├── checkpoints/
    └── checkpoints_two_stage/
```

---

**Happy Training! 🚀**
