"""Data transformations: normalization, JPEG compression shifts, and augmentations."""

import numpy as np
import torch
from PIL import Image
import io


def normalize(image, mean=None, std=None):
    """
    Normalize image using standard normalization.

    Args:
        image: numpy array or torch tensor
        mean: normalization mean (ImageNet default if None)
        std: normalization std (ImageNet default if None)

    Returns:
        Normalized image
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])

    if isinstance(image, torch.Tensor):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
    else:
        mean = np.array(mean)
        std = np.array(std)

    return (image - mean) / std


def denormalize(image, mean=None, std=None):
    """Reverse normalization."""
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])

    if isinstance(image, torch.Tensor):
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
    else:
        mean = np.array(mean)
        std = np.array(std)

    return image * std + mean


def apply_jpeg_compression(image, quality=95):
    """
    Apply JPEG compression to simulate distribution shift.

    Args:
        image: PIL Image or numpy array
        quality: JPEG quality (1-100)

    Returns:
        Compressed image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray((image * 255).astype(np.uint8))

    # Save and reload with specified quality
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert("RGB")

    return compressed


class CompressorAugmentation:
    """Apply JPEG compression as data augmentation."""

    def __init__(self, quality_range=(50, 95), p=0.5):
        """
        Args:
            quality_range: (min, max) JPEG quality
            p: probability of applying compression
        """
        self.quality_range = quality_range
        self.p = p

    def __call__(self, image):
        if np.random.random() < self.p:
            quality = np.random.randint(*self.quality_range)
            return apply_jpeg_compression(image, quality=quality)
        return image
