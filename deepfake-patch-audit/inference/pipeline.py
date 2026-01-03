"""Main inference pipeline for deepfake detection."""

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np


class InferencePipeline:
    """
    Main inference pipeline for deepfake detection.
    Handles model loading, image preprocessing, and prediction.
    """

    def __init__(self, model, device="cuda", threshold=0.5):
        """
        Args:
            model: Trained model
            device: Device to run inference on
            threshold: Decision threshold for classification
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.model.eval()

    def preprocess(self, image_path, resize_size=224):
        """
        Preprocess image for inference.

        Args:
            image_path: Path to image file
            resize_size: Target size for resizing

        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        image = Image.open(image_path).convert("RGB")
        image = image.resize((resize_size, resize_size), Image.BILINEAR)
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
        Predict if image is deepfake or real.

        Args:
            image_path: Path to image file

        Returns:
            dict with prediction results
        """
        image = self.preprocess(image_path)

        # Forward pass
        logits = self.model(image)
        probs = F.softmax(logits, dim=1)

        fake_prob = probs[0, 1].item()
        real_prob = probs[0, 0].item()

        is_fake = fake_prob > self.threshold
        confidence = max(fake_prob, real_prob)

        return {
            "is_fake": is_fake,
            "fake_probability": fake_prob,
            "real_probability": real_prob,
            "confidence": confidence,
            "logits": logits.cpu().numpy(),
        }

    @torch.no_grad()
    def predict_batch(self, image_paths):
        """
        Predict batch of images.

        Args:
            image_paths: List of image paths

        Returns:
            List of prediction dicts
        """
        images = torch.stack([self.preprocess(path).squeeze(0) for path in image_paths])

        logits = self.model(images)
        probs = F.softmax(logits, dim=1)

        results = []
        for i in range(len(image_paths)):
            fake_prob = probs[i, 1].item()
            real_prob = probs[i, 0].item()

            results.append({
                "image": image_paths[i],
                "is_fake": fake_prob > self.threshold,
                "fake_probability": fake_prob,
                "real_probability": real_prob,
                "confidence": max(fake_prob, real_prob),
            })

        return results

    def set_threshold(self, threshold):
        """Update decision threshold."""
        self.threshold = threshold
