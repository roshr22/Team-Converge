#!/bin/bash
# AWS EC2 Setup Script for FF++ Staged Training
# Run this AFTER SSH-ing into your g4dn.xlarge instance
# Instance should use: Deep Learning OSS Nvidia Driver AMI (Ubuntu)

set -e

echo "=============================================="
echo "FF++ Training Setup - AWS EC2"
echo "=============================================="

# 1. Verify GPU
echo "[1/7] Checking GPU..."
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERROR: No GPU detected! Make sure you launched a g4dn instance."
    exit 1
fi

# 2. Create workspace
echo "[2/7] Setting up workspace..."
cd /home/ubuntu
mkdir -p ffpp_training
cd ffpp_training

# 3. Clone repo
echo "[3/7] Cloning repository..."
rm -rf Team-Converge
git clone https://github.com/Incharajayaram/Team-Converge.git
cd Team-Converge/Finetune1

# 4. Install dependencies
echo "[4/7] Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install mediapipe pyyaml tqdm scikit-learn pillow

# 5. Download data from S3 (USER: Update this bucket/key!)
echo "[5/7] Downloading data..."
echo "=========================================="
echo "MANUAL STEP REQUIRED:"
echo "Upload ffpp_data_new.zip to S3, then run:"
echo "  aws s3 cp s3://YOUR-BUCKET/ffpp_data_new.zip ./ffpp_data.zip"
echo ""
echo "OR use scp from your local machine:"
echo "  scp -i your-key.pem ffpp_data_new.zip ubuntu@YOUR-EC2-IP:/home/ubuntu/ffpp_training/"
echo "=========================================="
echo "After data is uploaded, run: ./run_training.sh"

# Create the training script
cat > run_training.sh << 'EOF'
#!/bin/bash
set -e
cd /home/ubuntu/ffpp_training/Team-Converge/Finetune1

echo "=============================================="
echo "STEP 1: Extracting data..."
echo "=============================================="
mkdir -p /home/ubuntu/ffpp_training/data/raw/ffpp
unzip -q /home/ubuntu/ffpp_training/ffpp_data.zip -d /home/ubuntu/ffpp_training/data/raw/ffpp

# Detect extracted folder
if [ -d "/home/ubuntu/ffpp_training/data/raw/ffpp/FaceForensics++_C23" ]; then
    FFPP_ROOT="/home/ubuntu/ffpp_training/data/raw/ffpp/FaceForensics++_C23"
else
    FFPP_ROOT="/home/ubuntu/ffpp_training/data/raw/ffpp"
fi
echo "FFPP_ROOT: $FFPP_ROOT"

echo "=============================================="
echo "STEP 2: Building video index..."
echo "=============================================="
python -c "
from pathlib import Path
from utils.indexing import build_master_index
output_csv = Path('data/index/videos_master.csv')
output_csv.parent.mkdir(parents=True, exist_ok=True)
videos = build_master_index(ffpp_root=Path('$FFPP_ROOT'), output_path=output_csv)
print(f'Indexed {len(videos)} videos')
"

echo "=============================================="
echo "STEP 3: Generating manifests..."
echo "=============================================="
python -c "
import csv
from pathlib import Path
from utils.indexing import load_master_index
from utils.face_extraction import generate_sample_id

videos = load_master_index(Path('data/index/videos_master.csv'))
samples = []
for v in videos:
    k = 10 if v.split != 'test' else 20
    for i in range(k):
        ts = 0.5 + i * 0.8
        samples.append({
            'sample_id': generate_sample_id(),
            'dataset': v.dataset, 'split': v.split, 'label': v.label,
            'method': v.method, 'group_id': v.group_id,
            'video_id': v.video_id, 'video_path': v.video_path,
            'timestamp': ts, 'filepath': ''
        })

out_dir = Path('artifacts/manifests')
out_dir.mkdir(parents=True, exist_ok=True)
for split in ['train', 'val', 'test']:
    split_samples = [s for s in samples if s['split'] == split]
    with open(out_dir / f'{split}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
        w.writeheader()
        w.writerows(split_samples)
    print(f'{split}: {len(split_samples)} samples')
"

echo "=============================================="
echo "STEP 4: Pre-extracting faces (~30-45 min)..."
echo "=============================================="
CACHE_DIR="/home/ubuntu/ffpp_training/cache/faces"
mkdir -p $CACHE_DIR

python preextract_faces.py \
    --ffpp_root $FFPP_ROOT \
    --manifest artifacts/manifests/train.csv \
    --cache_dir $CACHE_DIR \
    --workers 8

python preextract_faces.py \
    --ffpp_root $FFPP_ROOT \
    --manifest artifacts/manifests/val.csv \
    --cache_dir $CACHE_DIR \
    --workers 8

echo "=============================================="
echo "STEP 5: Stage A Training (2 epochs, ~30 min)..."
echo "=============================================="
OUTPUT_DIR="/home/ubuntu/ffpp_training/output"
mkdir -p $OUTPUT_DIR

python train_staged.py --config config.yaml \
    --override dataset.ffpp_root=$FFPP_ROOT \
    --override caching.cache_dir=$CACHE_DIR \
    --stages A \
    --output_dir $OUTPUT_DIR

echo "=============================================="
echo "STAGE A COMPLETE!"
echo "Checkpoint saved to: $OUTPUT_DIR/best_model.pt"
echo "=============================================="

echo ""
echo "To continue with Stage B, run:"
echo "python train_staged.py --config config.yaml \\"
echo "    --override dataset.ffpp_root=$FFPP_ROOT \\"
echo "    --override caching.cache_dir=$CACHE_DIR \\"
echo "    --stages B \\"
echo "    --resume $OUTPUT_DIR/best_model.pt \\"
echo "    --output_dir $OUTPUT_DIR"
EOF

chmod +x run_training.sh

echo ""
echo "=============================================="
echo "SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Upload your data (see instructions above)"
echo "2. Run: ./run_training.sh"
echo ""
