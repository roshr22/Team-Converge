# Quick Start Guide - Automatic Dataset Detection

## Overview

The training scripts now **automatically detect your dataset structure**. No code changes needed!

---

## Supported Dataset Structures

### Option 1: Directory Structure (Recommended)

```
dataset/
├── train/
│   ├── real/      (real face images)
│   └── fake/      (deepfake images)
├── val/
│   ├── real/
│   └── fake/
└── test/          (optional, for evaluation)
    ├── real/
    └── fake/
```

**Example:**
```
dataset/
├── train/
│   ├── real/image_001.jpg
│   ├── real/image_002.jpg
│   ├── fake/deepfake_001.jpg
│   └── fake/deepfake_002.jpg
├── val/
│   ├── real/
│   ├── fake/
```

### Option 2: CSV Files

```
dataset/
├── data/splits/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
```

**CSV Format:**
```csv
path,label
dataset/train/real/image_001.jpg,0
dataset/train/real/image_002.jpg,0
dataset/train/fake/deepfake_001.jpg,1
dataset/train/fake/deepfake_002.jpg,1
...
```

---

## How Auto-Detection Works

When you run training, the script:

1. **Checks for directory structure** (train/real, train/fake, val/real, val/fake)
2. **If found**: Uses directory-based loading
3. **If not found**: Looks for CSV files (data/splits/train.csv, data/splits/val.csv)
4. **If CSV found**: Uses CSV-based loading
5. **If neither found**: Shows error with expected structure

```python
# Pseudo-code
if (dataset/train/real exists AND dataset/train/fake exists AND
    dataset/val/real exists AND dataset/val/fake exists):
    use directory structure
elif (data/splits/train.csv exists AND data/splits/val.csv exists):
    use CSV files
else:
    raise error("Dataset not found")
```

---

## Running Training

### Two-Stage Training (Recommended)

```bash
# Simplest - auto-detects everything
python3 scripts/train_student_two_stage.py

# With custom parameters
python3 scripts/train_student_two_stage.py \
    --epochs-s1 5 \
    --epochs-s2 20 \
    --batch-size 16 \
    --lr-s1 0.001 \
    --lr-s2 0.0001

# Different dataset location
python3 scripts/train_student_two_stage.py \
    --dataset-root /path/to/dataset
```

### Single-Stage Training

```bash
# Auto-detects dataset structure
python3 scripts/train_student.py --epochs 50 --batch-size 32

# Different dataset
python3 scripts/train_student.py \
    --dataset-root /path/to/dataset \
    --epochs 50
```

---

## Example Commands

### Scenario 1: Standard Directory Structure

If your dataset is:
```
dataset/
├── train/ (2099 images)
│   ├── real/ (1050)
│   └── fake/ (1049)
└── val/ (449 images)
    ├── real/ (225)
    └── fake/ (224)
```

Just run:
```bash
python3 scripts/train_student_two_stage.py --epochs-s1 5 --epochs-s2 20
```

### Scenario 2: Dataset in Different Location

If your dataset is at `/data/deepfakes/`:
```bash
python3 scripts/train_student_two_stage.py \
    --dataset-root /data/deepfakes \
    --epochs-s1 5 \
    --epochs-s2 20
```

### Scenario 3: CSV Split Files

If you have CSV files:
```
dataset/data/splits/train.csv
dataset/data/splits/val.csv
```

Just run:
```bash
python3 scripts/train_student_two_stage.py --epochs-s1 5 --epochs-s2 20
```

---

## What Gets Detected

### Console Output Example

```
================================================================================
Dataset Auto-Detection
================================================================================
✓ Detected mode: DIRECTORY
  Train: dataset/train
  Val:   dataset/val

✓ Student model loaded: 1297 parameters
✓ Teacher model loaded

✓ Training dataset: 2099 samples
✓ Validation dataset: 449 samples
```

---

## Dataset Requirements

### Minimum Configuration
- At least one of:
  - `dataset/train/{real,fake}` + `dataset/val/{real,fake}`
  - `data/splits/train.csv` + `data/splits/val.csv`

### Recommended File Formats
- **Images**: JPG, PNG, BMP
- **Count**:
  - Training: 1000+ images (minimum)
  - Validation: 100+ images (minimum)
  - Recommended: 2000+ train, 500+ val

### Label Format
- **Directory**: Folder name (real/fake)
- **CSV**: 0 = real, 1 = fake

---

## Troubleshooting

### Error: "Dataset structure not recognized!"

**Solution**: Create one of the supported structures:

```bash
# Option 1: Directory structure
mkdir -p dataset/train/{real,fake}
mkdir -p dataset/val/{real,fake}
cp your_real_images/* dataset/train/real/
cp your_fake_images/* dataset/train/fake/
cp your_val_real_images/* dataset/val/real/
cp your_val_fake_images/* dataset/val/fake/

# Option 2: CSV files
python3 << EOF
import pandas as pd

samples = []
for path in [paths_to_real_images]:
    samples.append({'path': path, 'label': 0})
for path in [paths_to_fake_images]:
    samples.append({'path': path, 'label': 1})

df = pd.DataFrame(samples)
df.to_csv('data/splits/train.csv', index=False)
EOF
```

### Error: "FileNotFoundError: [Errno 2] No such file or directory"

**Solution**: Ensure images exist in the correct directories:
```bash
# Check dataset structure
tree dataset -d -L 2

# Expected output:
# dataset
# ├── train
# │   ├── fake
# │   └── real
# └── val
#     ├── fake
#     └── real
```

### Error: "Empty dataset (0 samples)"

**Solution**:
1. Check image format (default: .jpg)
2. Verify files exist in real/fake directories
3. Check file permissions

```bash
# Verify images
ls dataset/train/real/ | head -5
ls dataset/train/fake/ | head -5

# Count images
find dataset -name "*.jpg" | wc -l
```

---

## No Code Changes Needed! ✅

The auto-detection feature means:

- ✅ Copy code to any system
- ✅ Organize dataset as train/real, train/fake, val/real, val/fake
- ✅ Run training immediately
- ✅ No config file changes
- ✅ No command-line argument changes
- ✅ Works with both directory and CSV structures

---

## Advanced: Custom Dataset Path

```bash
# Your dataset is at /home/data/deepfakes/
python3 scripts/train_student_two_stage.py \
    --dataset-root /home/data/deepfakes \
    --epochs-s1 5 \
    --epochs-s2 20 \
    --batch-size 16
```

The script will look for:
- `/home/data/deepfakes/train/{real,fake}`
- `/home/data/deepfakes/val/{real,fake}`

Or:
- `/home/data/deepfakes/data/splits/train.csv`
- `/home/data/deepfakes/data/splits/val.csv`

---

## Summary

| Feature | Supported |
|---------|-----------|
| Auto-detect directory structure | ✅ |
| Auto-detect CSV files | ✅ |
| No code changes needed | ✅ |
| Custom dataset paths | ✅ |
| Both single & two-stage training | ✅ |
| Clear error messages | ✅ |

**Just copy the code, organize your dataset, and start training!** 🚀
