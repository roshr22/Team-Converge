# AWS Fine-tuning Guide for FF++ Pipeline

## Option 1: EC2 Spot Instance (Cheapest)

### Step 1: Launch Instance
```bash
# From AWS Console or CLI
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \  # Deep Learning AMI (Ubuntu)
    --instance-type g4dn.xlarge \        # T4 GPU ~$0.35/hr spot
    --spot-market-options '{"SpotInstanceType":"one-time"}' \
    --key-name your-key \
    --security-group-ids sg-xxx
```

**Instance options:**
| Type | GPU | vCPU | RAM | Spot Price |
|------|-----|------|-----|------------|
| g4dn.xlarge | T4 16GB | 4 | 16GB | ~$0.15-0.35/hr |
| g4dn.2xlarge | T4 16GB | 8 | 32GB | ~$0.25-0.50/hr |
| p3.2xlarge | V100 16GB | 8 | 61GB | ~$0.90-1.50/hr |

### Step 2: SSH and Setup
```bash
ssh -i your-key.pem ubuntu@<instance-ip>

# Clone repo
git clone https://github.com/Incharajayaram/Team-Converge.git
cd Team-Converge/Finetune1

# Install deps
pip install mediapipe pyyaml tqdm scikit-learn
```

### Step 3: Upload Data to S3
```bash
# On your local machine
aws s3 cp ffpp_data.zip s3://your-bucket/data/

# On EC2
aws s3 cp s3://your-bucket/data/ffpp_data.zip /home/ubuntu/data/
unzip ffpp_data.zip -d /home/ubuntu/data/raw/ffpp/
```

### Step 4: Run Training
```bash
python train_staged.py --config config.yaml \
    --override dataset.ffpp_root=/home/ubuntu/data/raw/ffpp \
    --override caching.cache_dir=/home/ubuntu/cache/faces \
    --stages A B \
    --output_dir /home/ubuntu/checkpoints
```

### Step 5: Save Checkpoints to S3
```bash
aws s3 sync /home/ubuntu/checkpoints s3://your-bucket/checkpoints/
```

---

## Option 2: SageMaker Training Job

### Step 1: Create training script wrapper
```python
# sagemaker_entry.py
import os
import sys
sys.path.insert(0, '/opt/ml/code')

from train_staged import main
main()
```

### Step 2: Submit job
```python
import sagemaker
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='sagemaker_entry.py',
    source_dir='.',
    role='arn:aws:iam::xxx:role/SageMakerRole',
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version='2.0',
    py_version='py310',
    hyperparameters={
        'stages': 'A B',
    },
)

estimator.fit({'training': 's3://your-bucket/data/'})
```

---

## Cost Estimates

| Setup | Duration | Est. Cost |
|-------|----------|-----------|
| Stage A+B on g4dn.xlarge | ~3-4 hours | $1-2 |
| Stage A+B on p3.2xlarge | ~1-2 hours | $2-3 |
| Full training (A+B+C) | ~5-6 hours | $2-5 |

---

## Quick Start (Recommended)

1. **Create S3 bucket**: `aws s3 mb s3://ffpp-training`
2. **Upload zip**: `aws s3 cp ffpp_data.zip s3://ffpp-training/data/`
3. **Launch spot instance**: g4dn.xlarge with Deep Learning AMI
4. **SSH in and run training**
5. **Sync checkpoints to S3 periodically**
