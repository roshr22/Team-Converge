"""Main inference pipeline for deepfake detection."""

import torch
from PIL import Image
import numpy as np


class InferencePipeline:
    """
    Main inference pipeline for deepfake detection.
    Handles model loading, image preprocessing, patch extraction, pooling, and prediction.

    Pipeline:
    1. Image/frame → Resize 256×256 → Normalize
    2. Student model → Patch-logit map (B, 1, 126, 126)
    3. Top-K pooling → Image-level logit (B, 1)
    4. Sigmoid → P-Fake ∈ [0, 1]
    5. Threshold → Real vs Deepfake decision
    6. Return: prediction + patch heatmap
    """

    def __init__(self, model, pooling=None, device="cuda", threshold=0.5):
        """
        Args:
            model: Trained student model
            pooling: Pooling layer (e.g., TopKLogitPooling) for aggregating patch logits
                    If None, uses mean pooling across all patches
            device: Device to run inference on
            threshold: Decision threshold for classification
        """
        self.model = model.to(device)
        self.pooling = pooling.to(device) if pooling is not None else None
        self.device = device
        self.threshold = threshold
        self.model.eval()
        if self.pooling is not None:
            self.pooling.eval()

    def preprocess(self, image_path, resize_size=256):
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file
            resize_size: Target size for resizing (default 256×256 to match training)

        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        image = Image.open(image_path).convert("RGB")
        image = image.resize((resize_size, resize_size), Image.BICUBIC)
        image = np.array(image, dtype=np.float32) / 255.0

        # Normalize (ImageNet)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image.to(self.device)

    @torch.no_grad()
    def predict(self, image_path):
        """
        Predict if image is deepfake or real with patch-level heatmap.

        Args:
            image_path: Path to image file

        Returns:
            dict with prediction results and patch heatmap
        """
        image = self.preprocess(image_path)

        # Step 1: Student model forward pass → patch-logit map
        # Output: (B, 1, H, W) where H=W=126 for Tiny-LaDeDa
        patch_logits = self.model(image)

        # Step 2: Apply pooling to aggregate patch logits → image-level logit
        if self.pooling is not None:
            image_logit = self.pooling(patch_logits)  # (B, 1)
        else:
            # Fallback: mean pooling across all patches
            image_logit = patch_logits.view(patch_logits.size(0), -1).mean(dim=1, keepdim=True)

        # Step 3: Apply sigmoid to convert logit to probability
        fake_prob = torch.sigmoid(image_logit).item()
        real_prob = 1.0 - fake_prob

        # Step 4: Threshold for decision
        is_fake = fake_prob > self.threshold
        confidence = max(fake_prob, real_prob)

        # Step 5: Generate patch-level probability heatmap
        patch_heatmap = torch.sigmoid(patch_logits).squeeze(1)  # (B, H, W) → (H, W) for single image

        return {
            "is_fake": is_fake,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": confidence,
            "image_logit": image_logit.cpu().numpy(),
            "patch_logits": patch_logits.cpu().numpy(),  # Raw patch logits (B, 1, H, W)
            "patch_heatmap": patch_heatmap.cpu().numpy(),  # Patch probabilities (H, W)
        }

    @torch.no_grad()
    def predict_batch(self, image_paths):
        """
        Predict batch of images with patch-level heatmaps.

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction dicts with heatmaps
        """
        images = torch.stack([self.preprocess(path).squeeze(0) for path in image_paths])

        # Step 1: Student model forward pass → patch-logit maps
        # Output: (B, 1, H, W)
        patch_logits = self.model(images)

        # Step 2: Apply pooling to aggregate patch logits → image-level logits
        if self.pooling is not None:
            image_logits = self.pooling(patch_logits)  # (B, 1)
        else:
            # Fallback: mean pooling across all patches
            image_logits = patch_logits.view(patch_logits.size(0), -1).mean(dim=1, keepdim=True)

        # Step 3: Apply sigmoid to convert logits to probabilities
        fake_probs = torch.sigmoid(image_logits).squeeze(1)  # (B,)

        # Step 4: Generate patch-level probability heatmaps
        patch_heatmaps = torch.sigmoid(patch_logits)  # (B, 1, H, W)

        results = []
        for i in range(len(image_paths)):
            fake_prob = fake_probs[i].item()
            real_prob = 1.0 - fake_prob

            results.append({
                "image": image_paths[i],
                "is_fake": fake_prob > self.threshold,
                "fake_probability": fake_prob,
                "real_probability": real_prob,
                "confidence": max(fake_prob, real_prob),
                "image_logit": image_logits[i].cpu().numpy(),
                "patch_logits": patch_logits[i].cpu().numpy(),  # Raw patch logits (1, H, W)
                "patch_heatmap": patch_heatmaps[i].squeeze(0).cpu().numpy(),  # Patch probabilities (H, W)
            })

        return results

    def set_threshold(self, threshold):
        """Update decision threshold."""
        self.threshold = threshold

    def visualize_heatmap(self, image_path, prediction_result, output_path=None):
        """
        Visualize patch heatmap overlaid on original image.

        Args:
            image_path: Path to original image
            prediction_result: Output dict from predict() method
            output_path: Path to save visualization (optional)

        Returns:
            PIL Image of visualization
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠ matplotlib required for visualization. Install: pip install matplotlib")
            return None

        # Load original image
        original_image = Image.open(image_path).convert("RGB")
        original_image = np.array(original_image)

        # Get heatmap from prediction result
        heatmap = prediction_result["patch_heatmap"]  # (H, W)

        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=12)
        axes[0].axis("off")

        # Plot 2: Patch heatmap
        im = axes[1].imshow(heatmap, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes[1].set_title(f"Patch Heatmap (Fake Prob)", fontsize=12)
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], label="Fake Probability")

        # Plot 3: Heatmap overlaid on image (with transparency)
        axes[2].imshow(original_image, alpha=0.5)
        im_overlay = axes[2].imshow(heatmap, cmap="RdYlGn_r", vmin=0, vmax=1, alpha=0.5)
        axes[2].set_title(
            f"Heatmap Overlay\nPrediction: {'FAKE' if prediction_result['is_fake'] else 'REAL'} "
            f"(P={prediction_result['fake_probability']:.3f})",
            fontsize=12,
        )
        axes[2].axis("off")
        plt.colorbar(im_overlay, ax=axes[2], label="Fake Probability")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"✓ Visualization saved to {output_path}")

        return fig
