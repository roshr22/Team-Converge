"""Deployment realism augmentation pipeline (Step 8).

Fixed-order augmentation pipeline simulating real-world deployment conditions:
1. Random resize/crop (small scale jitter)
2. JPEG recompression simulation
3. Downscale → upscale (resolution degradation)
4. Mild Gaussian blur
5. Mild Gaussian noise
6. Brightness/contrast/gamma jitter

All parameters are logged for reproducibility.
"""

import io
import random
import logging
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field, asdict

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

import torch
from torchvision import transforms


logger = logging.getLogger(__name__)


@dataclass
class AugmentationParams:
    """Record of augmentation parameters applied."""
    # Resize/crop
    scale_factor: float = 1.0
    crop_x: int = 0
    crop_y: int = 0
    
    # JPEG
    jpeg_quality: int = 95
    jpeg_applied: bool = False
    
    # Downscale
    downscale_factor: float = 1.0
    downscale_applied: bool = False
    
    # Blur
    blur_sigma: float = 0.0
    blur_applied: bool = False
    
    # Noise
    noise_sigma: float = 0.0
    noise_applied: bool = False
    
    # Color
    brightness_factor: float = 1.0
    contrast_factor: float = 1.0
    gamma: float = 1.0
    color_applied: bool = False


class DeploymentRealismAugmentation:
    """Sequential augmentations for deployment realism.
    
    Order is fixed for reproducibility:
    1. resize_crop → 2. jpeg → 3. downscale_upscale → 4. blur → 5. noise → 6. color
    
    Args:
        target_size: Final output size (square)
        
        # Resize/crop params
        p_resize_crop: Probability of applying resize/crop
        scale_range: Range for scale jitter (min, max)
        
        # JPEG params
        p_jpeg: Probability of JPEG recompression
        jpeg_quality_range: Range for quality (min, max)
        
        # Downscale params  
        p_downscale: Probability of downscale/upscale
        downscale_range: Range for scale (min, max)
        
        # Blur params
        p_blur: Probability of Gaussian blur
        blur_sigma_range: Range for sigma (min, max)
        
        # Noise params
        p_noise: Probability of Gaussian noise
        noise_sigma_range: Range for sigma (min, max)
        
        # Color params
        p_color: Probability of color jitter
        brightness_range: Range for brightness factor (min, max)
        contrast_range: Range for contrast factor (min, max)
        gamma_range: Range for gamma (min, max)
        
        seed: Random seed
        log_interval: Log params every N samples (0 to disable)
    """
    
    def __init__(
        self,
        target_size: int = 256,
        # Resize/crop
        p_resize_crop: float = 0.5,
        scale_range: Tuple[float, float] = (0.9, 1.0),
        # JPEG
        p_jpeg: float = 0.4,
        jpeg_quality_range: Tuple[int, int] = (30, 90),
        # Downscale
        p_downscale: float = 0.3,
        downscale_range: Tuple[float, float] = (0.5, 0.8),
        # Blur
        p_blur: float = 0.2,
        blur_sigma_range: Tuple[float, float] = (0.0, 1.5),
        # Noise
        p_noise: float = 0.2,
        noise_sigma_range: Tuple[float, float] = (0, 15),
        # Color
        p_color: float = 0.3,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        gamma_range: Tuple[float, float] = (0.9, 1.1),
        # Other
        seed: Optional[int] = None,
        log_interval: int = 1000,
    ):
        self.target_size = target_size
        
        self.p_resize_crop = p_resize_crop
        self.scale_range = scale_range
        
        self.p_jpeg = p_jpeg
        self.jpeg_quality_range = jpeg_quality_range
        
        self.p_downscale = p_downscale
        self.downscale_range = downscale_range
        
        self.p_blur = p_blur
        self.blur_sigma_range = blur_sigma_range
        
        self.p_noise = p_noise
        self.noise_sigma_range = noise_sigma_range
        
        self.p_color = p_color
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        
        self.seed = seed
        self.log_interval = log_interval
        
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
        self._sample_count = 0
        self._last_params: Optional[AugmentationParams] = None
    
    def set_seed(self, seed: int):
        """Reset random state with new seed."""
        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """Apply augmentation pipeline to image.
        
        Args:
            img: PIL Image (RGB)
            
        Returns:
            Tensor of shape (3, H, W) normalized to [0, 1]
        """
        params = AugmentationParams()
        
        # 1. Resize/crop (small scale jitter)
        if self._rng.random() < self.p_resize_crop:
            img, params = self._apply_resize_crop(img, params)
        
        # 2. JPEG recompression
        if self._rng.random() < self.p_jpeg:
            img, params = self._apply_jpeg(img, params)
        
        # 3. Downscale → upscale
        if self._rng.random() < self.p_downscale:
            img, params = self._apply_downscale(img, params)
        
        # 4. Gaussian blur
        if self._rng.random() < self.p_blur:
            img, params = self._apply_blur(img, params)
        
        # 5. Gaussian noise
        if self._rng.random() < self.p_noise:
            img, params = self._apply_noise(img, params)
        
        # 6. Brightness/contrast/gamma
        if self._rng.random() < self.p_color:
            img, params = self._apply_color(img, params)
        
        # Final resize to target size
        if img.size != (self.target_size, self.target_size):
            img = img.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        # Store params and log periodically
        self._last_params = params
        self._sample_count += 1
        
        if self.log_interval > 0 and self._sample_count % self.log_interval == 0:
            self._log_params(params)
        
        # Convert to tensor
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        
        return tensor
    
    def _apply_resize_crop(
        self, 
        img: Image.Image, 
        params: AugmentationParams,
    ) -> Tuple[Image.Image, AugmentationParams]:
        """Apply random resize and crop."""
        w, h = img.size
        scale = self._rng.uniform(*self.scale_range)
        
        # Compute crop size
        crop_w = int(w * scale)
        crop_h = int(h * scale)
        
        # Random crop position
        max_x = w - crop_w
        max_y = h - crop_h
        x = self._rng.randint(0, max(0, max_x))
        y = self._rng.randint(0, max(0, max_y))
        
        img = img.crop((x, y, x + crop_w, y + crop_h))
        
        params.scale_factor = scale
        params.crop_x = x
        params.crop_y = y
        
        return img, params
    
    def _apply_jpeg(
        self, 
        img: Image.Image, 
        params: AugmentationParams,
    ) -> Tuple[Image.Image, AugmentationParams]:
        """Apply JPEG recompression."""
        quality = self._rng.randint(*self.jpeg_quality_range)
        
        # Encode to JPEG in memory
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img = Image.open(buffer)
        img.load()
        
        params.jpeg_quality = quality
        params.jpeg_applied = True
        
        return img, params
    
    def _apply_downscale(
        self, 
        img: Image.Image, 
        params: AugmentationParams,
    ) -> Tuple[Image.Image, AugmentationParams]:
        """Apply downscale then upscale (resolution degradation)."""
        scale = self._rng.uniform(*self.downscale_range)
        
        w, h = img.size
        small_w = int(w * scale)
        small_h = int(h * scale)
        
        # Downscale
        img = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
        # Upscale back
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        
        params.downscale_factor = scale
        params.downscale_applied = True
        
        return img, params
    
    def _apply_blur(
        self, 
        img: Image.Image, 
        params: AugmentationParams,
    ) -> Tuple[Image.Image, AugmentationParams]:
        """Apply Gaussian blur."""
        sigma = self._rng.uniform(*self.blur_sigma_range)
        
        if sigma > 0.01:
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        params.blur_sigma = sigma
        params.blur_applied = True
        
        return img, params
    
    def _apply_noise(
        self, 
        img: Image.Image, 
        params: AugmentationParams,
    ) -> Tuple[Image.Image, AugmentationParams]:
        """Apply Gaussian noise."""
        sigma = self._rng.uniform(*self.noise_sigma_range)
        
        if sigma > 0.1:
            arr = np.array(img).astype(np.float32)
            noise = self._np_rng.normal(0, sigma, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        
        params.noise_sigma = sigma
        params.noise_applied = True
        
        return img, params
    
    def _apply_color(
        self, 
        img: Image.Image, 
        params: AugmentationParams,
    ) -> Tuple[Image.Image, AugmentationParams]:
        """Apply brightness/contrast/gamma adjustment."""
        brightness = self._rng.uniform(*self.brightness_range)
        contrast = self._rng.uniform(*self.contrast_range)
        gamma = self._rng.uniform(*self.gamma_range)
        
        # Brightness
        if abs(brightness - 1.0) > 0.01:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        # Contrast
        if abs(contrast - 1.0) > 0.01:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        # Gamma
        if abs(gamma - 1.0) > 0.01:
            arr = np.array(img).astype(np.float32) / 255.0
            arr = np.power(arr, gamma)
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        
        params.brightness_factor = brightness
        params.contrast_factor = contrast
        params.gamma = gamma
        params.color_applied = True
        
        return img, params
    
    def _log_params(self, params: AugmentationParams):
        """Log augmentation parameters."""
        logger.info(
            f"[AUG #{self._sample_count}] "
            f"jpeg={params.jpeg_quality if params.jpeg_applied else 'N/A'} "
            f"downscale={params.downscale_factor:.2f if params.downscale_applied else 'N/A'} "
            f"blur={params.blur_sigma:.2f if params.blur_applied else 'N/A'} "
            f"noise={params.noise_sigma:.1f if params.noise_applied else 'N/A'} "
            f"brightness={params.brightness_factor:.2f if params.color_applied else 'N/A'} "
            f"gamma={params.gamma:.2f if params.color_applied else 'N/A'}"
        )
    
    def get_last_params(self) -> Optional[Dict[str, Any]]:
        """Get parameters from last augmentation."""
        if self._last_params:
            return asdict(self._last_params)
        return None
    
    @classmethod
    def from_config(cls, config: dict) -> 'DeploymentRealismAugmentation':
        """Create augmentation pipeline from config.
        
        Args:
            config: Full config dict
            
        Returns:
            Configured DeploymentRealismAugmentation
        """
        aug_cfg = config.get('augmentation', {})
        dataset_cfg = config.get('dataset', {})
        
        return cls(
            target_size=dataset_cfg.get('crop_size', 256),
            p_resize_crop=aug_cfg.get('p_random_crop', 0.3),
            scale_range=tuple(aug_cfg.get('random_crop_scale', [0.85, 1.0])),
            p_jpeg=aug_cfg.get('p_jpeg', 0.4),
            jpeg_quality_range=(
                min(aug_cfg.get('jpeg_qualities', [30, 90])),
                max(aug_cfg.get('jpeg_qualities', [30, 90])),
            ),
            p_downscale=aug_cfg.get('p_resize', 0.3),
            downscale_range=(
                min(aug_cfg.get('resize_scales', [0.5, 0.8])),
                max(aug_cfg.get('resize_scales', [0.5, 0.8])),
            ),
            p_blur=aug_cfg.get('p_blur', 0.2),
            blur_sigma_range=(0.0, max(aug_cfg.get('blur_sigmas', [0.5, 1.5]))),
            p_noise=aug_cfg.get('p_noise', 0.2),
            noise_sigma_range=(0, max(aug_cfg.get('noise_sigmas', [5, 15]))),
            p_color=aug_cfg.get('p_color', 0.3),
            brightness_range=tuple(aug_cfg.get('brightness_range', [0.8, 1.2])),
            contrast_range=tuple(aug_cfg.get('contrast_range', [0.8, 1.2])),
            seed=config.get('seed', 42),
        )


class ValidationTransform:
    """Simple transform for validation/test (no augmentation).
    
    Just resizes to target size and converts to tensor.
    """
    
    def __init__(self, target_size: int = 256):
        self.target_size = target_size
    
    def __call__(self, img: Image.Image) -> torch.Tensor:
        if img.size != (self.target_size, self.target_size):
            img = img.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        
        return tensor


if __name__ == "__main__":
    print("DeploymentRealismAugmentation module loaded!")
    print("Order: resize_crop → jpeg → downscale_upscale → blur → noise → color")
