"""Face extraction pipeline for deepfake detection training.

Implements consistent face detection and cropping with:
- Single detector (BlazeFace) with frozen version
- Primary face selection by largest area
- Fixed margin expansion (1.3x)
- Consistent resize and JPEG re-encoding
- UUID-only filenames (no metadata leakage)
- Split-separated storage
"""

import io
import uuid
import subprocess
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from PIL import Image

# BlazeFace detection thresholds (frozen)
BLAZEFACE_VERSION = "mediapipe==0.10.9"
DETECTION_CONFIDENCE = 0.5
MIN_DETECTION_CONFIDENCE = 0.5

# Processing constants
DEFAULT_MARGIN = 1.3  # 30% expansion around face
DEFAULT_CROP_SIZE = 256
DEFAULT_JPEG_QUALITY = 95  # Consistent re-encoding quality


@dataclass
class BoundingBox:
    """Face bounding box with normalized coordinates."""
    x_min: float  # 0-1 normalized
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    
    @property
    def area(self) -> float:
        """Compute area of bounding box."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)
    
    def to_pixels(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Convert to pixel coordinates."""
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )


@dataclass
class FaceCrop:
    """A single extracted face crop."""
    sample_id: str  # UUID
    image_bytes: bytes  # JPEG encoded
    bbox: BoundingBox
    original_size: Tuple[int, int]


class FaceDetector:
    """Face detector using MediaPipe BlazeFace.
    
    Provides consistent detection across the entire pipeline.
    """
    
    def __init__(self, min_confidence: float = MIN_DETECTION_CONFIDENCE):
        """Initialize detector.
        
        Args:
            min_confidence: Minimum detection confidence threshold
        """
        self.min_confidence = min_confidence
        self._detector = None
        self._mp_face = None
    
    def _lazy_init(self):
        """Lazily initialize MediaPipe to avoid import overhead."""
        if self._detector is None:
            try:
                import mediapipe as mp
                self._mp_face = mp.solutions.face_detection
                self._detector = self._mp_face.FaceDetection(
                    model_selection=1,  # Full range model
                    min_detection_confidence=self.min_confidence,
                )
            except ImportError:
                raise ImportError(
                    f"MediaPipe not installed. Install with: pip install {BLAZEFACE_VERSION}"
                )
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """Detect faces in an image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            List of BoundingBox objects, sorted by area (largest first)
        """
        self._lazy_init()
        
        # MediaPipe expects RGB
        results = self._detector.process(image)
        
        boxes = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Convert to our format (x_min, y_min, x_max, y_max)
                x_min = max(0, bbox.xmin)
                y_min = max(0, bbox.ymin)
                x_max = min(1, bbox.xmin + bbox.width)
                y_max = min(1, bbox.ymin + bbox.height)
                
                boxes.append(BoundingBox(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    confidence=float(detection.score[0]) if len(detection.score) > 0 else 0.0,
                ))
        
        # Sort by area (largest first)
        boxes.sort(key=lambda b: b.area, reverse=True)
        
        return boxes
    
    def get_primary_face(self, image: np.ndarray) -> Optional[BoundingBox]:
        """Get the primary (largest) face in an image.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            BoundingBox of primary face, or None if no face detected
        """
        boxes = self.detect(image)
        return boxes[0] if boxes else None
    
    def close(self):
        """Release detector resources."""
        if self._detector:
            self._detector.close()
            self._detector = None


def expand_bbox(
    bbox: BoundingBox,
    margin: float = DEFAULT_MARGIN,
    image_width: int = 1,
    image_height: int = 1,
) -> BoundingBox:
    """Expand bounding box by a margin factor.
    
    Args:
        bbox: Original bounding box
        margin: Expansion factor (1.3 = 30% larger)
        image_width: Image width for clamping
        image_height: Image height for clamping
        
    Returns:
        Expanded BoundingBox
    """
    width = bbox.x_max - bbox.x_min
    height = bbox.y_max - bbox.y_min
    
    # Compute expansion
    expand_w = width * (margin - 1) / 2
    expand_h = height * (margin - 1) / 2
    
    # Apply and clamp
    return BoundingBox(
        x_min=max(0, bbox.x_min - expand_w),
        y_min=max(0, bbox.y_min - expand_h),
        x_max=min(1, bbox.x_max + expand_w),
        y_max=min(1, bbox.y_max + expand_h),
        confidence=bbox.confidence,
    )


def crop_and_resize(
    image: np.ndarray,
    bbox: BoundingBox,
    target_size: int = DEFAULT_CROP_SIZE,
) -> Image.Image:
    """Crop face region and resize to target size.
    
    Args:
        image: RGB numpy array
        bbox: Face bounding box
        target_size: Output size (square)
        
    Returns:
        Cropped and resized PIL Image
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.to_pixels(w, h)
    
    # Ensure valid crop region
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        # Invalid box - return center crop
        cx, cy = w // 2, h // 2
        size = min(w, h) // 2
        x1, y1 = cx - size, cy - size
        x2, y2 = cx + size, cy + size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
    
    # Crop
    crop = image[y1:y2, x1:x2]
    
    # Convert to PIL and resize
    pil_img = Image.fromarray(crop)
    pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return pil_img


def encode_jpeg(image: Image.Image, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    """Encode image as JPEG bytes with consistent quality.
    
    Args:
        image: PIL Image
        quality: JPEG quality (0-100)
        
    Returns:
        JPEG encoded bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, subsampling=0)
    return buffer.getvalue()


def decode_frame_ffmpeg(video_path: Path, timestamp: float) -> Optional[np.ndarray]:
    """Decode a single frame from video at timestamp using ffmpeg.
    
    Args:
        video_path: Path to video file
        timestamp: Time in seconds
        
    Returns:
        RGB numpy array or None on failure
    """
    cmd = [
        'ffmpeg',
        '-ss', f'{timestamp:.3f}',
        '-i', str(video_path),
        '-vframes', '1',
        '-f', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-loglevel', 'error',
        '-'
    ]
    
    try:
        # First get video dimensions
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'csv=p=0',
            str(video_path)
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        width, height = map(int, probe_result.stdout.strip().split(','))
        
        # Extract frame
        result = subprocess.run(cmd, capture_output=True, check=True)
        
        # Convert to numpy
        frame = np.frombuffer(result.stdout, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))
        
        return frame
        
    except (subprocess.CalledProcessError, ValueError):
        return None


def generate_sample_id() -> str:
    """Generate a unique sample ID (UUID).
    
    Returns:
        UUID string (no dashes, lowercase)
    """
    return uuid.uuid4().hex


class FaceExtractor:
    """Complete face extraction pipeline.
    
    Extracts faces from video frames and saves to disk with UUID filenames.
    """
    
    def __init__(
        self,
        output_root: Path,
        margin: float = DEFAULT_MARGIN,
        crop_size: int = DEFAULT_CROP_SIZE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        min_confidence: float = MIN_DETECTION_CONFIDENCE,
    ):
        """Initialize extractor.
        
        Args:
            output_root: Root directory for face crops (will contain train/val/test subdirs)
            margin: Bounding box expansion factor
            crop_size: Output face size
            jpeg_quality: JPEG re-encoding quality
            min_confidence: Minimum face detection confidence
        """
        self.output_root = Path(output_root)
        self.margin = margin
        self.crop_size = crop_size
        self.jpeg_quality = jpeg_quality
        
        self.detector = FaceDetector(min_confidence=min_confidence)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            (self.output_root / split).mkdir(parents=True, exist_ok=True)
    
    def extract_from_frame(
        self,
        frame: np.ndarray,
        split: str,
    ) -> Optional[FaceCrop]:
        """Extract primary face from a frame.
        
        Args:
            frame: RGB numpy array
            split: train/val/test (for storage location)
            
        Returns:
            FaceCrop object or None if no face detected
        """
        # Detect primary face
        bbox = self.detector.get_primary_face(frame)
        if bbox is None:
            return None
        
        # Expand bbox
        expanded = expand_bbox(bbox, margin=self.margin)
        
        # Crop and resize
        crop = crop_and_resize(frame, expanded, target_size=self.crop_size)
        
        # Encode as JPEG
        image_bytes = encode_jpeg(crop, quality=self.jpeg_quality)
        
        # Generate UUID
        sample_id = generate_sample_id()
        
        return FaceCrop(
            sample_id=sample_id,
            image_bytes=image_bytes,
            bbox=expanded,
            original_size=(frame.shape[1], frame.shape[0]),
        )
    
    def extract_and_save(
        self,
        video_path: Path,
        timestamp: float,
        split: str,
    ) -> Optional[Tuple[str, Path]]:
        """Extract face from video at timestamp and save to disk.
        
        Args:
            video_path: Path to video
            timestamp: Frame timestamp in seconds
            split: train/val/test
            
        Returns:
            Tuple of (sample_id, saved_path) or None if extraction failed
        """
        # Decode frame
        frame = decode_frame_ffmpeg(video_path, timestamp)
        if frame is None:
            return None
        
        # Extract face
        crop = self.extract_from_frame(frame, split)
        if crop is None:
            return None
        
        # Save with UUID filename only (no metadata in filename!)
        output_path = self.output_root / split / f"{crop.sample_id}.jpg"
        with open(output_path, 'wb') as f:
            f.write(crop.image_bytes)
        
        return crop.sample_id, output_path
    
    def close(self):
        """Release resources."""
        self.detector.close()
    
    @classmethod
    def from_config(cls, config: dict) -> 'FaceExtractor':
        """Create extractor from config.
        
        Args:
            config: Full config dict
            
        Returns:
            Configured FaceExtractor
        """
        dataset_cfg = config.get('dataset', {})
        
        return cls(
            output_root=Path(dataset_cfg.get('faces_dir', 'data/derived/faces')),
            margin=dataset_cfg.get('margin_factor', 0.3) + 1.0,  # Convert margin to expansion factor
            crop_size=dataset_cfg.get('crop_size', 256),
            min_confidence=dataset_cfg.get('min_face_confidence', 0.5),
        )


if __name__ == "__main__":
    # Test face detection
    print("Face extraction module loaded successfully!")
    print(f"BlazeFace version: {BLAZEFACE_VERSION}")
    print(f"Default margin: {DEFAULT_MARGIN}x")
    print(f"Default crop size: {DEFAULT_CROP_SIZE}px")
    print(f"JPEG quality: {DEFAULT_JPEG_QUALITY}")
