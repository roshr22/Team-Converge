# 🚀 Training on Kaggle (Free GPU)

## Quick Start Guide

### Step 1: Upload Your Dataset to Kaggle Datasets

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"+ New Dataset"**
3. Upload your training data with this structure:
   ```
   ecdd-training-data/
   ├── train/
   │   ├── real/
   │   │   └── *.jpg (real face images)
   │   └── fake/
   │       └── *.jpg (deepfake images)
   ├── val/
   │   ├── real/
   │   └── fake/
   └── test/
       ├── real/
       └── fake/
   ```
4. Name it: `ecdd-training-data`
5. Make it **Private** (or Public if you want)
6. Click **Create**

**Your local data is at**: `ECDD_Experimentation/ECDD_Training_Data/processed/splits/`

### Step 2: Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code)
2. Click **"+ New Notebook"**
3. Enable GPU:
   - Settings (right panel) → Accelerator → **GPU P100**
4. Add your dataset:
   - Click **"+ Add Data"** (right panel)
   - Search for your `ecdd-training-data`
   - Click **Add**

### Step 3: Copy the Training Code

1. Open `kaggle_training_notebook.py` from this folder
2. Copy the entire content
3. Paste into your Kaggle notebook
4. **Important**: Update `DATA_PATH` if needed:
   ```python
   DATA_PATH = "/kaggle/input/ecdd-training-data"
   ```

### Step 4: Run Training

1. Click **"Run All"** or run cells one by one
2. Training takes ~30-45 minutes for 15 epochs on P100
3. Monitor progress in the output

### Step 5: Download Trained Model

After training completes:
1. Go to the **Output** section
2. Download:
   - `best_model.pth` - Your trained model
   - `results.json` - Training metrics
   - `training_history.png` - Loss/accuracy plots

---

## Alternative: Using Public Datasets

If you don't want to upload your own data, use these public datasets:

### Option A: Use CelebDF-v2
```python
DATA_PATH = "/kaggle/input/celeb-df-v2"
```
Search for "Celeb-DF" in Kaggle Datasets

### Option B: Use FaceForensics++
```python
DATA_PATH = "/kaggle/input/faceforensics"
```
Search for "FaceForensics" in Kaggle Datasets

---

## Training Configuration

Default settings (modify in the notebook):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Epochs | 15 | Increase for better results |
| Batch Size | 16 | Reduce to 8 if GPU OOM |
| Learning Rate | 1e-4 | Lower for fine-tuning |
| Frozen Layers | conv1, layer1 | Freeze early layers |
| Image Size | 256×256 | ECDD-locked |

---

## Expected Results

With balanced dataset (~500+ images per class):

| Metric | Expected Range |
|--------|----------------|
| Val Accuracy | 85-95% |
| Val F1 | 0.80-0.92 |
| Test F1 | 0.78-0.90 |

---

## Troubleshooting

**GPU Out of Memory (OOM)**
- Reduce `batch_size` to 8 or 4
- Reduce `num_workers` to 0

**Dataset Not Found**
- Check DATA_PATH matches your dataset name
- Ensure dataset is added to notebook

**Slow Training**
- Verify GPU is enabled (check for "GPU" badge)
- Use P100 (faster than T4)

---

## Files in This Folder

- `kaggle_training_notebook.py` - Complete training notebook
- `finetune_script.py` - Original local training script
- `models/ladeda_resnet.py` - Model architecture

---

## Kaggle Free Tier Limits

- **30 hours/week** GPU quota
- **P100 or T4** GPU available
- **16GB** GPU memory
- **~20GB** disk space

This training should use ~2-3 hours of your weekly quota.
