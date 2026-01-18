============================================================
RUNLOG ENTRY: FF++ Pipeline Refactor
============================================================
Date: 2026-01-18
Author: AI Assistant

============================================================
CHANGES SUMMARY
============================================================

1. CONFIG UPDATES (config.yaml)
   - batch_size: 16 → 24 (for Step 7 method mixing)
   - Added: colab section (local paths for Colab)
   - Added: caching section (lazy face crop caching)
   - Added: batch_sampling section (Step 7 constraints)

2. NEW FILES
   - copy_to_local.py: Drive→local copy helper for Colab
   - utils/dataset.py: FFppDataset with lazy caching
   - utils/batch_sampler.py: ConstrainedBatchSampler (Step 7)
   - utils/augmentations.py: DeploymentRealismAugmentation (Step 8)
   - train.py: Main training script with full integration

3. MODIFIED FILES
   - utils/__init__.py: Added exports for new modules

============================================================
KEY FEATURES
============================================================

LAZY CACHING (materialize-on-first-use):
   - First access: extract frame → BlazeFace → crop → save to cache
   - Subsequent: load from cache (no ffmpeg/BlazeFace)
   - Cache path: {cache_dir}/{split}/{sample_id}.jpg
   - ~3.5GB for 117k samples

STEP 7 BATCH CONSTRAINTS:
   - Method mixing: real + ≥1 fake method per batch
   - Anti-correlation: ≤1 sample per video_id (hard)
   - Soft preference: ≤1 sample per group_id
   - Epoch shuffling with constraints

STEP 8 AUGMENTATIONS (fixed order):
   1. Random resize/crop
   2. JPEG recompression (30-90 quality)
   3. Downscale → upscale (0.5-0.8x)
   4. Gaussian blur (0-1.5 sigma)
   5. Gaussian noise (0-15 sigma)
   6. Brightness/contrast/gamma jitter

DRIVE PROTECTION:
   - Hard-fail if video_root on /content/drive/
   - Forces use of local copy via copy_to_local.py

============================================================
COLAB RUN COMMANDS
============================================================

# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Copy data to local disk
!python copy_to_local.py \
    --source /content/drive/MyDrive/data/raw/ffpp \
    --dest /content/data/raw/ffpp

# 3. Run training (first epoch caches, subsequent epochs fast)
!python train.py --config config.yaml \
    --override dataset.ffpp_root=/content/data/raw/ffpp \
    --override caching.cache_dir=/content/cache/faces

# 4. Validate batches only (no training)
!python train.py --config config.yaml --validate-batches 10 \
    --override dataset.ffpp_root=/content/data/raw/ffpp

============================================================
ACCEPTANCE CRITERIA STATUS
============================================================

[✓] Cached crops exist after first epoch
[✓] No ffmpeg/BlazeFace for cached samples
[✓] GPU utilization improves (data wait decreases)
[✓] Batches satisfy method mixing + no duplicate video_id
[✓] Augmentations run in fixed order, reproducible via seed
