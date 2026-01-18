"""Test model with exact notebook architecture."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

# Exact model from notebook
class AttentionPooling(nn.Module):
    def __init__(self, in_channels=2048, hidden_dim=512):
        super().__init__()
        self.attention_fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(self, features, patch_logits):
        attn_scores = self.attention_fc(features)
        B, _, H, W = attn_scores.shape
        attn_flat = attn_scores.view(B, -1)
        attn_flat = attn_flat - attn_flat.max(dim=1, keepdim=True)[0]
        attn = F.softmax(attn_flat, dim=1).view(B, 1, H, W)
        pooled = (patch_logits * attn).sum(dim=(2, 3))
        return pooled, attn


class LaDeDaResNet50_Notebook(nn.Module):
    def __init__(self, pretrained=False, freeze_layers=None, dropout_rate=0.4):
        super().__init__()
        base = resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.patch_classifier = nn.Conv2d(2048, 1, kernel_size=1)
        self.attention_pool = AttentionPooling(2048)
        self.freeze_layers = freeze_layers or []

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        patch_logits = self.patch_classifier(x)
        pooled_logits, attn = self.attention_pool(x, patch_logits)
        pooled_logits = pooled_logits.view(-1)
        return pooled_logits, patch_logits, attn


def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    img = ImageOps.exif_transpose(img)
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def main():
    # Load model with notebook architecture
    print("Loading model...")
    model = LaDeDaResNet50_Notebook(pretrained=False, dropout_rate=0.4)
    checkpoint = torch.load('Training/weights/best_model_finetune1.pth', map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded!")
    
    # Test on some images
    test_images = [
        ("ECDD_Experiment_Data/fake/Gemini_Generated_Image_tcds8etcds8etcds (1).png", "FAKE"),
        ("ECDD_Experiment_Data/fake/face2face_frame.png", "FAKE"),
        ("ECDD_Experiment_Data/fake/01_02__outside_talking_still_laughing__YVGY8LOK_frame.png", "FAKE"),
        ("ECDD_Experiment_Data/real/00023.jpg", "REAL"),
        ("ECDD_Experiment_Data/real/00472.jpg", "REAL"),
        ("ECDD_Experiment_Data/real/565037942_1315519440042831_8848975727375564547_n.jpg", "REAL"),
    ]
    
    print("\n" + "="*60)
    print(f"{'Image':<50} {'True':<6} {'Pred':<6} {'Logit':>8} {'Prob':>6}")
    print("="*60)
    
    correct = 0
    for img_path, true_label in test_images:
        tensor = preprocess(img_path)
        with torch.no_grad():
            logit, _, _ = model(tensor)
            prob = torch.sigmoid(logit).item()
        
        pred = "FAKE" if prob > 0.5 else "REAL"
        is_correct = pred == true_label
        correct += is_correct
        
        name = Path(img_path).name[:45]
        mark = "OK" if is_correct else "WRONG"
        print(f"{name:<50} {true_label:<6} {pred:<6} {logit.item():>8.4f} {prob:>6.3f} {mark}")
    
    print("="*60)
    print(f"Accuracy: {correct}/{len(test_images)} = {100*correct/len(test_images):.1f}%")


if __name__ == "__main__":
    main()
