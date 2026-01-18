"""Generate calibration logits from trained model on test images.

Usage:
    python generate_calibration_logits.py --model path/to/weights.pth --output calibration_logits.json

Output format: JSON list of {id, logit, label} entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image, ImageOps


def load_model(weights_path: Path):
    """Load LaDeDa model with trained weights."""
    from Training.models.ladeda_resnet import create_ladeda_model
    
    model = create_ladeda_model(pretrained=False)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def preprocess_image(img_path: Path, size: int = 256) -> np.ndarray:
    """Preprocess image for model input."""
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.exif_transpose(img)  # Handle EXIF orientation
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    # Normalize with ImageNet stats
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    
    return arr


def run_inference(model, images: List[np.ndarray]) -> np.ndarray:
    """Run model inference on batch of images."""
    with torch.no_grad():
        input_tensor = torch.from_numpy(np.stack(images, axis=0)).float()  # Ensure float32
        pooled_logits, _, _ = model(input_tensor)
        return pooled_logits.numpy().squeeze()


def generate_calibration_data(
    model,
    real_dir: Path,
    fake_dir: Path,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Generate calibration data from real and fake images."""
    results = []
    
    # Collect all image paths
    real_images = sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png")))
    fake_images = sorted(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png")))
    
    print(f"Found {len(real_images)} real and {len(fake_images)} fake images")
    
    # Process real images (label=0)
    for img_path in real_images:
        try:
            preprocessed = preprocess_image(img_path)
            logit = run_inference(model, [preprocessed])
            # Handle scalar or array logit
            logit_val = float(logit.item()) if hasattr(logit, 'item') else float(logit)
            results.append({
                "id": img_path.stem,
                "logit": logit_val,
                "label": 0,  # Real = 0
                "path": str(img_path),
            })
            print(f"  [REAL] {img_path.name}: logit={logit_val:.4f}")
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
    
    # Process fake images (label=1)
    for img_path in fake_images:
        try:
            preprocessed = preprocess_image(img_path)
            logit = run_inference(model, [preprocessed])
            # Handle scalar or array logit
            logit_val = float(logit.item()) if hasattr(logit, 'item') else float(logit)
            results.append({
                "id": img_path.stem,
                "logit": logit_val,
                "label": 1,  # Fake = 1
                "path": str(img_path),
            })
            print(f"  [FAKE] {img_path.name}: logit={logit_val:.4f}")
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate calibration logits from trained model")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--output", type=str, default="calibration_logits.json", help="Output JSON path")
    parser.add_argument("--real-dir", type=str, default=None, help="Directory with real images")
    parser.add_argument("--fake-dir", type=str, default=None, help="Directory with fake images")
    args = parser.parse_args()
    
    # Default directories
    base_dir = Path(__file__).parent.parent
    real_dir = Path(args.real_dir) if args.real_dir else base_dir / "ECDD_Experiment_Data" / "real"
    fake_dir = Path(args.fake_dir) if args.fake_dir else base_dir / "ECDD_Experiment_Data" / "fake"
    
    print(f"Loading model from {args.model}...")
    model = load_model(Path(args.model))
    print("Model loaded successfully!")
    
    print(f"\nGenerating calibration data...")
    print(f"  Real images: {real_dir}")
    print(f"  Fake images: {fake_dir}")
    
    results = generate_calibration_data(model, real_dir, fake_dir)
    
    # Write output
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} calibration entries to {output_path}")
    
    # Summary stats
    real_logits = [r["logit"] for r in results if r["label"] == 0]
    fake_logits = [r["logit"] for r in results if r["label"] == 1]
    
    print(f"\nSummary:")
    print(f"  Real: n={len(real_logits)}, mean={np.mean(real_logits):.4f}, std={np.std(real_logits):.4f}")
    print(f"  Fake: n={len(fake_logits)}, mean={np.mean(fake_logits):.4f}, std={np.std(fake_logits):.4f}")


if __name__ == "__main__":
    main()
