"""PyTorch Dataset for FF++ with lazy face crop caching.

Implements cache-on-first-use: on first access, extracts frame via ffmpeg,
runs BlazeFace detection, crops/resizes, saves to cache. Subsequent accesses
load directly from cache.

This dramatically speeds up training after the first epoch.
"""

import csv
import os
import io
import sys
import tempfile
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from .manifest import SampleRecord, load_samples_csv
from .face_extraction import (
    FaceDetector, 
    decode_frame_ffmpeg, 
    expand_bbox, 
    crop_and_resize, 
    encode_jpeg,
    DEFAULT_MARGIN,
    DEFAULT_CROP_SIZE,
    DEFAULT_JPEG_QUALITY,
)


# Drive path patterns to detect (hard-fail if detected)
DRIVE_PATH_PATTERNS = [
    "/content/drive/",
    "/content/gdrive/",
    "drive/MyDrive/",
]


@dataclass
class ExtractionFailure:
    """Record of a failed face extraction."""
    sample_id: str
    video_path: str
    timestamp: float
    error_type: str
    error_message: str
    epoch: int


class FailureLogger:
    """Thread-safe logger for extraction failures."""
    
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialized = False
    
    def _ensure_header(self):
        if not self._initialized:
            if not self.log_path.exists():
                with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['sample_id', 'video_path', 'timestamp', 
                                     'error_type', 'error_message', 'epoch'])
            self._initialized = True
    
    def log(self, failure: ExtractionFailure):
        with self._lock:
            self._ensure_header()
            with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    failure.sample_id,
                    failure.video_path,
                    failure.timestamp,
                    failure.error_type,
                    failure.error_message,
                    failure.epoch,
                ])


class CacheStats:
    """Thread-safe cache hit/miss statistics."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
    
    def record_hit(self):
        with self._lock:
            self.hits += 1
    
    def record_miss(self):
        with self._lock:
            self.misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total": total,
                "hit_rate": self.hits / total if total > 0 else 0.0,
            }
    
    def reset(self):
        with self._lock:
            self.hits = 0
            self.misses = 0


def check_drive_path(path: str) -> bool:
    """Check if path is on Google Drive (should be avoided).
    
    Returns:
        True if path appears to be on Drive
    """
    path_str = str(path)
    for pattern in DRIVE_PATH_PATTERNS:
        if pattern in path_str:
            return True
    return False


class FFppDataset(Dataset):
    """FF++ dataset with lazy face crop caching.
    
    On first access to a sample:
    1. Check if cached crop exists at cache_dir/{split}/{sample_id}.jpg
    2. If cached: load JPEG and return
    3. If not cached: extract frame, detect face, crop, save to cache, return
    
    Args:
        manifest_csv: Path to sample manifest CSV (train.csv, val.csv, etc.)
        video_root: Root directory containing video files
        cache_dir: Directory for cached face crops
        cache_enabled: Whether to use caching (default True for train)
        transform: Optional torchvision transform to apply
        margin: Bounding box expansion factor
        crop_size: Output face crop size
        jpeg_quality: JPEG quality for cached crops
        failure_log_path: Path to log extraction failures
        epoch: Current epoch number (for failure logging)
    """
    
    def __init__(
        self,
        manifest_csv: Path,
        video_root: Path,
        cache_dir: Path,
        cache_enabled: bool = True,
        transform = None,
        margin: float = DEFAULT_MARGIN,
        crop_size: int = DEFAULT_CROP_SIZE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        failure_log_path: Optional[Path] = None,
        epoch: int = 0,
    ):
        self.video_root = Path(video_root)
        self.cache_dir = Path(cache_dir)
        self.cache_enabled = cache_enabled
        self.transform = transform
        self.margin = margin
        self.crop_size = crop_size
        self.jpeg_quality = jpeg_quality
        self.epoch = epoch
        
        # Hard-fail if video_root is on Drive
        if check_drive_path(str(self.video_root)):
            raise RuntimeError(
                f"FATAL: video_root points to Google Drive: {self.video_root}\n"
                "Training from Drive is extremely slow and not supported.\n"
                "Use copy_to_local.py to copy data to /content/data first."
            )
        
        # Load manifest
        self.samples = load_samples_csv(manifest_csv)
        
        # Create cache directories
        if self.cache_enabled:
            for split in ['train', 'val', 'test']:
                (self.cache_dir / split).mkdir(parents=True, exist_ok=True)
        
        # Initialize face detector (lazy)
        self._detector = None
        self._detector_lock = threading.Lock()
        
        # Failure logging
        if failure_log_path:
            self.failure_logger = FailureLogger(failure_log_path)
        else:
            self.failure_logger = None
        
        # Cache statistics
        self.cache_stats = CacheStats()
        
        # Build index by sample_id for fallback lookups
        self._sample_index = {s.sample_id: i for i, s in enumerate(self.samples)}
        
        # Track failed samples (to skip in future epochs)
        self._failed_samples: set = set()
    
    @property
    def detector(self) -> FaceDetector:
        """Lazily initialize face detector."""
        if self._detector is None:
            with self._detector_lock:
                if self._detector is None:
                    self._detector = FaceDetector()
        return self._detector
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _get_cache_path(self, sample: SampleRecord) -> Path:
        """Get cache file path for a sample."""
        return self.cache_dir / sample.split / f"{sample.sample_id}.jpg"
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Image.Image]:
        """Load image from cache if it exists."""
        if cache_path.exists():
            try:
                img = Image.open(cache_path)
                img.load()  # Force load to catch truncated images
                return img
            except Exception:
                # Corrupted cache file - delete and re-extract
                try:
                    cache_path.unlink()
                except Exception:
                    pass
        return None
    
    def _extract_and_cache(
        self, 
        sample: SampleRecord,
        cache_path: Path,
    ) -> Optional[Image.Image]:
        """Extract face from video and cache it.
        
        Returns:
            PIL Image if successful, None if extraction failed
        """
        video_path = self.video_root / sample.video_path
        
        # Decode frame
        try:
            frame = decode_frame_ffmpeg(video_path, sample.timestamp)
            if frame is None:
                self._log_failure(sample, "decode_failed", "ffmpeg returned None")
                return None
        except Exception as e:
            self._log_failure(sample, "decode_error", str(e))
            return None
        
        # Detect face
        try:
            bbox = self.detector.get_primary_face(frame)
            if bbox is None:
                self._log_failure(sample, "no_face", "No face detected")
                return None
        except Exception as e:
            self._log_failure(sample, "detection_error", str(e))
            return None
        
        # Expand and crop
        try:
            expanded = expand_bbox(bbox, margin=self.margin)
            crop = crop_and_resize(frame, expanded, target_size=self.crop_size)
        except Exception as e:
            self._log_failure(sample, "crop_error", str(e))
            return None
        
        # Save to cache (atomic write)
        if self.cache_enabled:
            try:
                # Write to temp file first, then rename (atomic)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_fd, temp_path = tempfile.mkstemp(
                    suffix='.jpg', 
                    dir=cache_path.parent
                )
                os.close(temp_fd)
                
                crop.save(temp_path, format='JPEG', quality=self.jpeg_quality)
                os.replace(temp_path, cache_path)
            except Exception as e:
                # Cache write failed - continue without caching
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        
        return crop
    
    def _log_failure(self, sample: SampleRecord, error_type: str, error_msg: str):
        """Log an extraction failure."""
        self._failed_samples.add(sample.sample_id)
        
        if self.failure_logger:
            self.failure_logger.log(ExtractionFailure(
                sample_id=sample.sample_id,
                video_path=sample.video_path,
                timestamp=sample.timestamp,
                error_type=error_type,
                error_message=error_msg,
                epoch=self.epoch,
            ))
    
    def _get_fallback_sample(self, current_idx: int) -> Tuple[Image.Image, int]:
        """Get a fallback sample when extraction fails.
        
        Returns:
            Tuple of (image, label) from a different sample with same label
        """
        current_sample = self.samples[current_idx]
        target_label = current_sample.label
        
        # Try to find a cached sample with same label
        for offset in range(1, min(100, len(self.samples))):
            fallback_idx = (current_idx + offset) % len(self.samples)
            fallback_sample = self.samples[fallback_idx]
            
            if fallback_sample.label != target_label:
                continue
            
            if fallback_sample.sample_id in self._failed_samples:
                continue
            
            cache_path = self._get_cache_path(fallback_sample)
            img = self._load_from_cache(cache_path)
            if img is not None:
                return img, fallback_sample.label
        
        # Last resort: return a blank image
        blank = Image.new('RGB', (self.crop_size, self.crop_size), (128, 128, 128))
        return blank, target_label
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a sample, extracting and caching if necessary.
        
        Returns:
            Dict with keys: image, label, sample_id, video_id, group_id, method
        """
        sample = self.samples[idx]
        cache_path = self._get_cache_path(sample)
        
        # Try cache first
        if self.cache_enabled:
            img = self._load_from_cache(cache_path)
            if img is not None:
                self.cache_stats.record_hit()
            else:
                self.cache_stats.record_miss()
                img = self._extract_and_cache(sample, cache_path)
        else:
            # No caching - always extract
            img = self._extract_and_cache(sample, cache_path)
        
        # Handle extraction failure
        if img is None:
            img, label = self._get_fallback_sample(idx)
        else:
            label = sample.label
        
        # Apply transform
        if self.transform:
            img = self.transform(img)
        else:
            # Default: convert to tensor
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        return {
            'image': img,
            'label': label,
            'sample_id': sample.sample_id,
            'video_id': sample.video_id,
            'group_id': sample.group_id,
            'method': sample.method,
        }
    
    def set_epoch(self, epoch: int):
        """Update current epoch (for failure logging)."""
        self.epoch = epoch
    
    def get_method_indices(self) -> Dict[str, List[int]]:
        """Get all sample indices grouped by method.
        
        Returns:
            Dict mapping method name to list of indices
        """
        method_indices = {}
        for i, sample in enumerate(self.samples):
            if sample.method not in method_indices:
                method_indices[sample.method] = []
            method_indices[sample.method].append(i)
        return method_indices
    
    def get_video_indices(self) -> Dict[str, List[int]]:
        """Get all sample indices grouped by video_id.
        
        Returns:
            Dict mapping video_id to list of indices
        """
        video_indices = {}
        for i, sample in enumerate(self.samples):
            if sample.video_id not in video_indices:
                video_indices[sample.video_id] = []
            video_indices[sample.video_id].append(i)
        return video_indices
    
    def get_group_indices(self) -> Dict[str, List[int]]:
        """Get all sample indices grouped by group_id.
        
        Returns:
            Dict mapping group_id to list of indices
        """
        group_indices = {}
        for i, sample in enumerate(self.samples):
            if sample.group_id not in group_indices:
                group_indices[sample.group_id] = []
            group_indices[sample.group_id].append(i)
        return group_indices
    
    @classmethod
    def from_config(
        cls,
        config: dict,
        split: str = 'train',
        transform = None,
    ) -> 'FFppDataset':
        """Create dataset from config.
        
        Args:
            config: Full config dict
            split: train/val/test
            transform: Optional transform to apply
            
        Returns:
            Configured FFppDataset
        """
        dataset_cfg = config.get('dataset', {})
        caching_cfg = config.get('caching', {})
        
        manifest_csv = Path(dataset_cfg.get('manifests_dir', 'artifacts/manifests')) / f"{split}.csv"
        video_root = Path(dataset_cfg.get('ffpp_root', 'data/raw/ffpp'))
        cache_dir = Path(caching_cfg.get('cache_dir', 'cache/faces'))
        failure_log = Path(caching_cfg.get('failure_log', 'artifacts/reports/extraction_failures.csv'))
        
        return cls(
            manifest_csv=manifest_csv,
            video_root=video_root,
            cache_dir=cache_dir,
            cache_enabled=caching_cfg.get('enabled', True),
            transform=transform,
            margin=dataset_cfg.get('margin_factor', 0.3) + 1.0,
            crop_size=dataset_cfg.get('crop_size', 256),
            jpeg_quality=caching_cfg.get('jpeg_quality', 95),
            failure_log_path=failure_log,
        )
