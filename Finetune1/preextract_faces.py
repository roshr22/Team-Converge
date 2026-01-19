"""Pre-extract all faces from videos for faster training.

This script extracts all faces in parallel before training starts,
dramatically reducing training time (15 min/epoch vs 10 hours).

Usage:
    python preextract_faces.py --ffpp_root /content/data/raw/ffpp/FaceForensics++_C23 \
        --manifest artifacts/manifests/train.csv \
        --cache_dir /content/cache/faces \
        --workers 8
"""

import os
import sys
import csv
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.face_extraction import (
    FaceDetector,
    decode_frame_ffmpeg,
    expand_bbox,
    crop_and_resize,
    DEFAULT_MARGIN,
    DEFAULT_CROP_SIZE,
    DEFAULT_JPEG_QUALITY,
)


@dataclass
class ExtractionTask:
    """A single face extraction task."""
    sample_id: str
    video_path: str
    timestamp: float
    split: str
    cache_path: Path


class FaceExtractor:
    """Thread-safe face extractor with caching."""
    
    def __init__(
        self,
        video_root: Path,
        cache_dir: Path,
        margin: float = DEFAULT_MARGIN,
        crop_size: int = DEFAULT_CROP_SIZE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    ):
        self.video_root = Path(video_root)
        self.cache_dir = Path(cache_dir)
        self.margin = margin
        self.crop_size = crop_size
        self.jpeg_quality = jpeg_quality
        
        # Thread-local detector
        self._local = threading.local()
        
        # Stats
        self._lock = threading.Lock()
        self.extracted = 0
        self.skipped = 0
        self.failed = 0
    
    def _get_detector(self) -> FaceDetector:
        """Get thread-local face detector."""
        if not hasattr(self._local, 'detector'):
            self._local.detector = FaceDetector()
        return self._local.detector
    
    def extract_one(self, task: ExtractionTask) -> Tuple[bool, str]:
        """Extract a single face.
        
        Returns:
            Tuple of (success, status_message)
        """
        # Skip if already cached
        if task.cache_path.exists():
            with self._lock:
                self.skipped += 1
            return True, "skipped"
        
        try:
            # Decode frame
            video_path = self.video_root / task.video_path
            frame = decode_frame_ffmpeg(video_path, task.timestamp)
            if frame is None:
                with self._lock:
                    self.failed += 1
                return False, "decode_failed"
            
            # Detect face
            detector = self._get_detector()
            bbox = detector.get_primary_face(frame)
            if bbox is None:
                with self._lock:
                    self.failed += 1
                return False, "no_face"
            
            # Crop and resize
            expanded = expand_bbox(bbox, margin=self.margin)
            crop = crop_and_resize(frame, expanded, target_size=self.crop_size)
            
            # Save to cache
            task.cache_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(task.cache_path, format='JPEG', quality=self.jpeg_quality)
            
            with self._lock:
                self.extracted += 1
            return True, "extracted"
            
        except Exception as e:
            with self._lock:
                self.failed += 1
            return False, str(e)[:50]
    
    def get_stats(self) -> dict:
        """Get extraction statistics."""
        with self._lock:
            total = self.extracted + self.skipped + self.failed
            return {
                "extracted": self.extracted,
                "skipped": self.skipped,
                "failed": self.failed,
                "total": total,
                "success_rate": (self.extracted + self.skipped) / total if total > 0 else 0,
            }


def load_manifest(manifest_path: Path) -> List[dict]:
    """Load sample manifest from CSV."""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def create_tasks(
    samples: List[dict],
    cache_dir: Path,
) -> List[ExtractionTask]:
    """Create extraction tasks from samples."""
    tasks = []
    for s in samples:
        cache_path = cache_dir / s['split'] / f"{s['sample_id']}.jpg"
        tasks.append(ExtractionTask(
            sample_id=s['sample_id'],
            video_path=s['video_path'],
            timestamp=float(s['timestamp']),
            split=s['split'],
            cache_path=cache_path,
        ))
    return tasks


def run_extraction(
    extractor: FaceExtractor,
    tasks: List[ExtractionTask],
    workers: int = 8,
    log_interval: int = 100,
) -> dict:
    """Run parallel extraction with progress logging.
    
    Args:
        extractor: FaceExtractor instance
        tasks: List of extraction tasks
        workers: Number of parallel workers
        log_interval: Log progress every N completions
        
    Returns:
        Final statistics dict
    """
    total = len(tasks)
    completed = 0
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"PRE-EXTRACTING {total} FACES")
    print(f"Workers: {workers}")
    print(f"{'='*60}\n")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {executor.submit(extractor.extract_one, task): task for task in tasks}
        
        # Process completions
        for future in as_completed(futures):
            completed += 1
            
            if completed % log_interval == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total - completed) / rate if rate > 0 else 0
                
                stats = extractor.get_stats()
                print(
                    f"[{completed}/{total}] "
                    f"extracted={stats['extracted']} "
                    f"skipped={stats['skipped']} "
                    f"failed={stats['failed']} "
                    f"| {rate:.1f}/s | ETA: {remaining/60:.1f}min"
                )
    
    elapsed = time.time() - start_time
    stats = extractor.get_stats()
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Extracted: {stats['extracted']}")
    print(f"Skipped (cached): {stats['skipped']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Pre-extract faces for faster training")
    parser.add_argument("--ffpp_root", type=str, required=True, 
                        help="Path to FF++ root directory")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to manifest CSV (train.csv, val.csv, or samples_master.csv)")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Directory to save cached face crops")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help=f"Face crop margin factor (default: {DEFAULT_MARGIN})")
    parser.add_argument("--crop_size", type=int, default=DEFAULT_CROP_SIZE,
                        help=f"Face crop size (default: {DEFAULT_CROP_SIZE})")
    
    args = parser.parse_args()
    
    # Load manifest
    print(f"Loading manifest from {args.manifest}...")
    samples = load_manifest(Path(args.manifest))
    print(f"Loaded {len(samples)} samples")
    
    # Create tasks
    cache_dir = Path(args.cache_dir)
    tasks = create_tasks(samples, cache_dir)
    
    # Create extractor
    extractor = FaceExtractor(
        video_root=Path(args.ffpp_root),
        cache_dir=cache_dir,
        margin=args.margin,
        crop_size=args.crop_size,
    )
    
    # Run extraction
    stats = run_extraction(extractor, tasks, workers=args.workers)
    
    # Report
    if stats['failed'] > 0:
        print(f"\nWARNING: {stats['failed']} samples failed to extract.")
        print("Training will use fallback samples for these.")


if __name__ == "__main__":
    main()
