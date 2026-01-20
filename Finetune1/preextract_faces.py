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

# Import face_extraction directly from file to avoid loading 'utils' package
# which imports torch/etc causing multiprocessing issues on Windows
import importlib.util
spec = importlib.util.spec_from_file_location(
    "face_extraction", 
    Path(__file__).parent / "utils" / "face_extraction.py"
)
face_extraction = importlib.util.module_from_spec(spec)
sys.modules["face_extraction"] = face_extraction
spec.loader.exec_module(face_extraction)

from face_extraction import (
    FaceDetector,
    FaceExtractor as FaceExtractorBase, # Renamed to avoid current file conflict if needed
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
    """Run parallel extraction with ProcessPoolExecutor for max speed.
    
    Args:
        extractor: FaceExtractor instance
        tasks: List of extraction tasks
        workers: Number of parallel workers
        log_interval: Log progress every N completions
        
    Returns:
        Final statistics dict
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Use max cores if workers > logical cores
    max_cores = multiprocessing.cpu_count()
    workers = min(workers, max_cores)
    
    total = len(tasks)
    
    # Filter out already done tasks first (fast check)
    print("Checking for existing files...")
    todo_tasks = []
    skipped = 0
    for t in tasks:
        if t.cache_path.exists():
            skipped += 1
        else:
            todo_tasks.append(t)
            
    print(f"Skipping {skipped} already cached files.")
    print(f"Processing remaining {len(todo_tasks)} files with {workers} workers (ProcessPool)...")
    
    if not todo_tasks:
        return {
            "extracted": 0,
            "skipped": skipped,
            "failed": 0,
            "total": total,
            "success_rate": 1.0
        }

    completed = 0
    failed = 0
    extracted = 0
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"PRE-EXTRACTING {len(todo_tasks)} FACES")
    print(f"Workers: {workers} (ProcessPool)")
    print(f"{'='*60}\n")
    
    # Use ProcessPoolExecutor to bypass GIL
    # NOTE: We can't pickle the extractor with its locks/local state easily
    # So we'll instantiate it inside the worker process if needed, 
    # but for simplicity we pass the config and let the worker rebuild it
    # OR we use a top-level function.
    
    # Since extract_one is an instance method and FaceExtractor has state (locks),
    # pickling will fail or be slow. 
    # BETTER APPROACH: Use a static/global worker function and simple arguments.
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit tasks using the global helper function defined below
        # We need to pass the arguments needed to reconstruct the extractor state or pass paths
        futures = []
        for task in todo_tasks:
            args = (
                str(extractor.video_root),
                str(extractor.cache_dir), 
                extractor.margin,
                extractor.crop_size,
                extractor.jpeg_quality,
                task
            )
            futures.append(executor.submit(_worker_extract, args))
            
        # Process completions
        for i, future in enumerate(as_completed(futures)):
            try:
                success, msg = future.result()
                if success:
                    extracted += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                
            completed += 1
            total_done = skipped + completed
            
            if completed % log_interval == 0 or completed == len(todo_tasks):
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining_items = len(todo_tasks) - completed
                remaining_time = remaining_items / rate if rate > 0 else 0
                
                print(
                    f"[{total_done}/{total}] "
                    f"new_extracted={extracted} "
                    f"failed={failed} "
                    f"| {rate:.1f} fps | ETA: {remaining_time/60:.1f}min"
                )
    
    elapsed = time.time() - start_time
    stats = extractor.get_stats() # This won't reflect process pool updates
    
    # Construct stats manually since processes didn't update the main object
    final_stats = {
        "extracted": extracted,
        "skipped": skipped,
        "failed": failed,
        "total": total,
        "success_rate": (extracted + skipped) / total if total > 0 else 0,
    }
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Extracted: {final_stats['extracted']}")
    print(f"Skipped (cached): {final_stats['skipped']}")
    print(f"Failed: {final_stats['failed']}")
    print(f"Success rate: {final_stats['success_rate']*100:.1f}%")
    print(f"{'='*60}\n")
    
    return final_stats


def _worker_extract(args):
    """Worker function for ProcessPoolExecutor.
    Needs to be outside class to be picklable.
    """
    video_root, cache_dir, margin, crop_size, jpeg_quality, task = args
    
    # Re-instantiate a lightweight extractor for just this task
    # We don't need locks since we are in a separate process
    try:
        # Import safely inside worker to ensure clean state and avoid torch (via utils package)
        import importlib.util
        import sys
        from pathlib import Path
        
        # If not already imported/patched in this worker process
        if "face_extraction" not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                "face_extraction", 
                Path(__file__).parent / "utils" / "face_extraction.py"
            )
            face_extraction = importlib.util.module_from_spec(spec)
            sys.modules["face_extraction"] = face_extraction
            spec.loader.exec_module(face_extraction)
            
        from face_extraction import (
            FaceDetector, decode_frame_ffmpeg, expand_bbox, crop_and_resize
        )
        
        # Check cache again just in case
        if task.cache_path.exists():
            return True, "skipped"
            
        # Decode
        video_path = Path(video_root) / task.video_path
        frame = decode_frame_ffmpeg(video_path, task.timestamp)
        if frame is None:
            return False, "decode_failed"
            
        # Detect
        # Cache detector process-locally
        if not hasattr(_worker_extract, 'detector'):
            _worker_extract.detector = FaceDetector()
            
        detector = _worker_extract.detector
        bbox = detector.get_primary_face(frame)
        
        if bbox is None:
            return False, "no_face"
            
        # Crop
        expanded = expand_bbox(bbox, margin=margin)
        crop = crop_and_resize(frame, expanded, target_size=crop_size)
        
        # Save
        task.cache_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(task.cache_path, format='JPEG', quality=jpeg_quality)
        
        return True, "extracted"
        
    except Exception as e:
        return False, str(e)



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
