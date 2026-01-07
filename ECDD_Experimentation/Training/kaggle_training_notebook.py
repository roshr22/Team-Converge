# %% [markdown]
# # ECDD LaDeDa Deepfake Detection Training
# 
# **Free GPU Training on Kaggle**
# 
# This notebook trains a LaDeDa-style ResNet50 for deepfake detection.
# 
# ## Setup Instructions:
# 1. Upload your dataset to Kaggle Datasets (see cell below)
# 2. Enable GPU: Settings → Accelerator → GPU P100
# 3. Run all cells
# 
# **Expected Training Time**: ~30-45 min for 15 epochs on P100

# %% [markdown]
# ## 1. Setup & Imports

# %%
# Install any missing packages
# pip install -q torchvision pillow tqdm scikit-learn  # Uncomment with ! prefix in Jupyter

# %%
import os
import sys
import json
import random
import io
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image, ImageOps, ImageEnhance
from tqdm.notebook import tqdm
from typing import Tuple, Optional

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Dataset Setup
# 
# ### Option A: Upload to Kaggle Datasets (Recommended)
# 1. Go to kaggle.com → Your Work → Datasets → New Dataset
# 2. Upload your training data with structure:
# ```
# ecdd-training-data/
# ├── train/
# │   ├── real/
# │   │   └── *.jpg
# │   └── fake/
# │       └── *.jpg
# ├── val/
# │   ├── real/
# │   └── fake/
# └── test/
#     ├── real/
#     └── fake/
# ```
# 3. Add dataset to this notebook: "+ Add Data" → Your Datasets
# 
# ### Option B: Use Existing Public Dataset
# You can also use Celeb-DF or FaceForensics++ from Kaggle

# %%
# ========== CONFIGURE YOUR DATASET PATH HERE ==========
# If you uploaded your own dataset:
DATA_PATH = "/kaggle/input/ecdd-training-data"

# Alternative: Use a public dataset (uncomment one):
# DATA_PATH = "/kaggle/input/celeb-df-v2"
# DATA_PATH = "/kaggle/input/faceforensics"

# Check if path exists
if os.path.exists(DATA_PATH):
    print(f"✅ Dataset found at: {DATA_PATH}")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(DATA_PATH, split)
        if os.path.exists(split_path):
            real_count = len(list(Path(split_path).glob('real/*.jpg')))
            fake_count = len(list(Path(split_path).glob('fake/*.jpg')))
            print(f"   {split}: {real_count} real, {fake_count} fake")
else:
    print(f"❌ Dataset not found at: {DATA_PATH}")
    print("Please upload your dataset or modify DATA_PATH")

# %% [markdown]
# ## 3. Model Architecture: LaDeDa ResNet50

# %%
class AttentionPooling(nn.Module):
    """
    Attention-based pooling over patch logits.
    Learns to weight patches based on their "importance" for the final decision.
    """
    
    def __init__(self, in_channels: int = 2048, hidden_dim: int = 512):
        super().__init__()
        self.attention_fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)
        )
        
    def forward(self, features: torch.Tensor, patch_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute attention scores
        attention_scores = self.attention_fc(features)  # (B, 1, H, W)
        
        # Flatten spatial dimensions for softmax
        B, _, H, W = attention_scores.shape
        attention_flat = attention_scores.view(B, -1)  # (B, H*W)
        
        # Apply softmax for normalized weights
        attention_weights_flat = F.softmax(attention_flat, dim=1)  # (B, H*W)
        attention_weights = attention_weights_flat.view(B, 1, H, W)  # (B, 1, H, W)
        
        # Weighted sum of patch logits
        patch_logits_flat = patch_logits.view(B, -1)  # (B, H*W)
        pooled_logit = (patch_logits_flat * attention_weights_flat).sum(dim=1, keepdim=True)  # (B, 1)
        
        return pooled_logit, attention_weights


class LaDeDaResNet50(nn.Module):
    """
    LaDeDa-style ResNet50 for patch-based deepfake detection.
    
    Key modifications:
    - Replace 7x7 conv with 3x3 (smaller receptive field)
    - Remove maxpool
    - Patch-level classification with attention pooling
    """
    
    def __init__(self, 
                 pretrained: bool = True,
                 freeze_layers: Optional[list] = None,
                 num_classes: int = 1):
        super().__init__()
        
        # Load base ResNet50
        if pretrained:
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            base_model = resnet50(weights=None)
        
        # MODIFICATION 1: Replace conv1 (7x7 -> 3x3)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        if pretrained:
            with torch.no_grad():
                original_weight = base_model.conv1.weight.data
                center = original_weight[:, :, 2:5, 2:5]
                self.conv1.weight.data = center
        
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        # MODIFICATION 2: NO maxpool (removed)
        
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # MODIFICATION 3: Patch classifier
        self.patch_classifier = nn.Conv2d(2048, num_classes, kernel_size=1)
        
        # MODIFICATION 4: Attention pooling
        self.attention_pool = AttentionPooling(in_channels=2048)
        
        # Freeze layers
        self.freeze_layers = freeze_layers or []
        self._freeze_layers()
        
    def _freeze_layers(self):
        freeze_map = {
            'conv1': [self.conv1, self.bn1],
            'layer1': [self.layer1],
            'layer2': [self.layer2],
            'layer3': [self.layer3],
            'layer4': [self.layer4],
        }
        
        for layer_name in self.freeze_layers:
            if layer_name in freeze_map:
                for module in freeze_map[layer_name]:
                    for param in module.parameters():
                        param.requires_grad = False
                print(f"Frozen: {layer_name}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # NO maxpool
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        features = x
        patch_logits = self.patch_classifier(features)
        pooled_logit, attention_map = self.attention_pool(features, patch_logits)
        
        return pooled_logit, patch_logits, attention_map


def create_ladeda_model(pretrained=True, freeze_layers=None):
    """Create LaDeDa model."""
    return LaDeDaResNet50(pretrained=pretrained, freeze_layers=freeze_layers)


# Test model
print("Testing model architecture...")
test_model = create_ladeda_model(pretrained=True, freeze_layers=['conv1', 'layer1'])
test_model.eval()
with torch.no_grad():
    x = torch.randn(2, 3, 256, 256)
    pooled, patches, attention = test_model(x)
print(f"✅ Model OK - Pooled: {pooled.shape}, Patches: {patches.shape}, Attention: {attention.shape}")

total = sum(p.numel() for p in test_model.parameters())
trainable = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
print(f"Parameters: Total={total:,}, Trainable={trainable:,}")
del test_model

# %% [markdown]
# ## 4. Dataset & Preprocessing

# %%
# ECDD-locked preprocessing constants
TARGET_SIZE = (256, 256)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DeepfakeDataset(Dataset):
    """Dataset for deepfake detection with ECDD-compliant preprocessing."""
    
    def __init__(self, data_dir: str, split: str = "train", augment: bool = True):
        self.data_dir = Path(data_dir) / split
        self.augment = augment and (split == "train")
        
        self.images = []
        self.labels = []
        
        # Load images
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for f in (self.data_dir / "real").glob(ext):
                self.images.append(f)
                self.labels.append(0)
            for f in (self.data_dir / "fake").glob(ext):
                self.images.append(f)
                self.labels.append(1)
        
        print(f"Loaded {split}: {len(self.images)} images (Real: {self.labels.count(0)}, Fake: {self.labels.count(1)})")
    
    def __len__(self):
        return len(self.images)
    
    def _augment(self, img):
        """Apply training augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Random JPEG compression (simulate social media)
        if random.random() > 0.5:
            quality = random.randint(50, 95)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
            img.load()
        
        # Random brightness
        if random.random() > 0.7:
            factor = random.uniform(0.9, 1.1)
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        return img
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess (ECDD-locked)
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # Fix EXIF orientation
        img = img.convert('RGB')
        
        if self.augment:
            img = self._augment(img)
        
        # Resize with Lanczos (ECDD-locked)
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        
        # To tensor and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = (img_array - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        return img_tensor, torch.tensor(label, dtype=torch.float32)

# %% [markdown]
# ## 5. Training Functions

# %%
def compute_metrics(outputs, labels, threshold=0.5):
    """Compute accuracy, precision, recall, F1."""
    probs = torch.sigmoid(outputs).cpu().numpy()
    preds = (probs > threshold).astype(int)
    labels = labels.cpu().numpy().astype(int)
    
    accuracy = (preds == labels).mean()
    
    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            pooled_logit, _, _ = model(images)
            loss = criterion(pooled_logit.squeeze(), labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        all_outputs.append(pooled_logit.squeeze().detach())
        all_labels.append(labels)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            pooled_logit, _, _ = model(images)
            loss = criterion(pooled_logit.squeeze(), labels)
            
            total_loss += loss.item()
            all_outputs.append(pooled_logit.squeeze())
            all_labels.append(labels)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_outputs, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

# %% [markdown]
# ## 6. Training Configuration

# %%
# ========== TRAINING CONFIG ==========
CONFIG = {
    'name': 'ladeda_deepfake',
    'epochs': 15,
    'batch_size': 16,  # Reduce to 8 if OOM
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'freeze_layers': ['conv1', 'layer1'],  # Freeze early layers
}

OUTPUT_DIR = "/kaggle/working/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Training Configuration:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 7. Run Training

# %%
# Create datasets
train_dataset = DeepfakeDataset(DATA_PATH, split="train", augment=True)
val_dataset = DeepfakeDataset(DATA_PATH, split="val", augment=False)
test_dataset = DeepfakeDataset(DATA_PATH, split="test", augment=False)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                        shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=False, num_workers=2)

print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

# %%
# Create model
model = create_ladeda_model(pretrained=True, freeze_layers=CONFIG['freeze_layers'])
model = model.to(device)

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG['lr'],
    weight_decay=CONFIG['weight_decay']
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
scaler = GradScaler()

print(f"Model loaded on {device}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %%
# Training loop
best_val_f1 = 0
history = {'train': [], 'val': []}

print("\n" + "="*60)
print("Starting Training")
print("="*60)

for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    print("-" * 40)
    
    train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
    val_metrics = validate(model, val_loader, criterion, device)
    
    scheduler.step()
    
    history['train'].append(train_metrics)
    history['val'].append(val_metrics)
    
    print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
    
    # Save best model
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': CONFIG
        }, f"{OUTPUT_DIR}/best_model.pth")
        print(f"  → Saved best model (Val F1: {best_val_f1:.4f})")

# %% [markdown]
# ## 8. Final Evaluation

# %%
print("\n" + "="*60)
print("Final Test Evaluation")
print("="*60)

# Load best model
checkpoint = torch.load(f"{OUTPUT_DIR}/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

test_metrics = validate(model, test_loader, criterion, device)
print(f"Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
print(f"       Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")

# Save results
results = {
    'config': CONFIG,
    'best_val_f1': best_val_f1,
    'test_metrics': test_metrics,
    'history': history,
    'timestamp': datetime.now().isoformat()
}

with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Training complete! Results saved to {OUTPUT_DIR}")

# %% [markdown]
# ## 9. Download Model

# %%
# Your trained model is saved at:
print("📦 Download your trained model from:")
print(f"   {OUTPUT_DIR}/best_model.pth")
print(f"   {OUTPUT_DIR}/results.json")

# List files in output
print("\n📁 Output files:")
for f in Path(OUTPUT_DIR).glob("*"):
    size = f.stat().st_size / 1024  # KB
    print(f"   {f.name}: {size:.1f} KB")

# %% [markdown]
# ## 10. Training History Visualization

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss
axes[0].plot([m['loss'] for m in history['train']], label='Train')
axes[0].plot([m['loss'] for m in history['val']], label='Val')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True)

# Accuracy
axes[1].plot([m['accuracy'] for m in history['train']], label='Train')
axes[1].plot([m['accuracy'] for m in history['val']], label='Val')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training & Validation Accuracy')
axes[1].legend()
axes[1].grid(True)

# F1 Score
axes[2].plot([m['f1'] for m in history['train']], label='Train')
axes[2].plot([m['f1'] for m in history['val']], label='Val')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('F1 Score')
axes[2].set_title('Training & Validation F1')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/training_history.png", dpi=150)
plt.show()

print(f"📈 Training history saved to {OUTPUT_DIR}/training_history.png")
