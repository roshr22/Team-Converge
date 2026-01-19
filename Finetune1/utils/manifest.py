"""Sample-level manifest generation and validation.

Creates the sample manifest that the dataloader reads:
- samples_master.csv: One row per face crop/frame
- train.csv, val.csv, test.csv: Split-filtered manifests
- Comprehensive leakage checks
"""

import csv
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from .indexing import VideoRecord, load_master_index
from .sampling import FrameSampler, compute_uniform_timestamps, get_video_info
from .face_extraction import generate_sample_id


@dataclass
class SampleRecord:
    """A single sample (face crop or frame) record."""
    sample_id: str
    dataset: str
    split: str
    label: int
    method: str
    group_id: str
    video_id: str
    video_path: str  # Relative path to source video (for streaming extraction)
    timestamp: float
    filepath: str  # Path to stored crop, or empty if streaming


def generate_samples_from_videos(
    video_records: List[VideoRecord],
    sampler: FrameSampler,
    ffpp_root: Path,
    faces_root: Optional[Path] = None,
    streaming: bool = True,
) -> List[SampleRecord]:
    """Generate sample records from video index.
    
    Args:
        video_records: List of VideoRecord from master index
        sampler: Configured FrameSampler
        ffpp_root: Root directory of FF++ videos
        faces_root: Root for stored crops (if not streaming)
        streaming: If True, use video_path+timestamp; if False, use crop filepath
        
    Returns:
        List of SampleRecord
    """
    samples = []
    ffpp_root = Path(ffpp_root)
    
    # Reset sampler for clean group caps
    sampler.reset_epoch()
    
    for video in video_records:
        video_path = ffpp_root / video.video_path
        
        if not video_path.exists():
            continue
        
        # Get video duration
        try:
            duration, _, _ = get_video_info(video_path)
            if duration <= 0:
                continue
        except Exception:
            continue
        
        # Compute K for this split
        k = sampler.get_k_for_split(video.split)
        
        # Check group cap
        current = sampler._group_counts.get(video.group_id, 0)
        remaining = sampler.max_per_group - current
        if remaining <= 0:
            continue
        
        k = min(k, remaining)
        
        # Compute timestamps
        timestamps = compute_uniform_timestamps(
            duration,
            k,
            epsilon=sampler.epsilon,
            min_gap=sampler.min_gap,
        )
        
        for ts in timestamps:
            sample_id = generate_sample_id()
            
            # Filepath depends on mode
            if streaming:
                filepath = ""  # Will use video_path + timestamp
            else:
                filepath = f"{video.split}/{sample_id}.jpg"
            
            samples.append(SampleRecord(
                sample_id=sample_id,
                dataset=video.dataset,
                split=video.split,
                label=video.label,
                method=video.method,
                group_id=video.group_id,
                video_id=video.video_id,
                video_path=video.video_path,  # Store for streaming extraction
                timestamp=ts,
                filepath=filepath,
            ))
        
        # Update group count
        sampler._group_counts[video.group_id] = current + len(timestamps)
    
    return samples


def save_samples_csv(samples: List[SampleRecord], output_path: Path) -> None:
    """Save samples to CSV.
    
    Args:
        samples: List of SampleRecord
        output_path: Output CSV path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'sample_id', 'dataset', 'split', 'label', 'method',
        'group_id', 'video_id', 'video_path', 'timestamp', 'filepath'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(asdict(s))
    
    print(f"[MANIFEST] Saved {len(samples)} samples to {output_path}")


def load_samples_csv(csv_path: Path) -> List[SampleRecord]:
    """Load samples from CSV.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of SampleRecord
    """
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(SampleRecord(
                sample_id=row['sample_id'],
                dataset=row['dataset'],
                split=row['split'],
                label=int(row['label']),
                method=row['method'],
                group_id=row['group_id'],
                video_id=row['video_id'],
                video_path=row['video_path'],
                timestamp=float(row['timestamp']),
                filepath=row['filepath'],
            ))
    return samples


def create_split_manifests(
    samples: List[SampleRecord],
    output_dir: Path,
) -> Dict[str, Path]:
    """Create filtered manifests by split.
    
    Args:
        samples: All samples
        output_dir: Directory to save split manifests
        
    Returns:
        Dict mapping split name to manifest path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_samples = defaultdict(list)
    for s in samples:
        split_samples[s.split].append(s)
    
    paths = {}
    for split in ['train', 'val', 'test']:
        if split in split_samples:
            path = output_dir / f"{split}.csv"
            save_samples_csv(split_samples[split], path)
            paths[split] = path
    
    return paths


def validate_manifest(samples: List[SampleRecord]) -> Tuple[bool, List[str]]:
    """Validate sample manifest for leakage and balance.
    
    Checks:
    1. No group_id overlaps across splits
    2. No video_id overlaps across splits
    3. Class balance per split
    
    Args:
        samples: List of SampleRecord
        
    Returns:
        Tuple of (all_valid, list of issues)
    """
    issues = []
    
    # Collect group_ids and video_ids per split
    group_to_splits = defaultdict(set)
    video_to_splits = defaultdict(set)
    
    for s in samples:
        group_to_splits[s.group_id].add(s.split)
        video_to_splits[s.video_id].add(s.split)
    
    print("\n" + "=" * 70)
    print("MANIFEST VALIDATION")
    print("=" * 70)
    
    # 1. Group ID overlap check
    print("\n--- Group ID Overlap Check ---")
    leaked_groups = [(gid, splits) for gid, splits in group_to_splits.items() if len(splits) > 1]
    if leaked_groups:
        print(f"  ERROR: {len(leaked_groups)} group_ids in multiple splits!")
        for gid, splits in leaked_groups[:5]:
            issues.append(f"Group {gid} in splits: {splits}")
            print(f"    {gid}: {splits}")
    else:
        print(f"  OK: {len(group_to_splits)} groups, no overlap")
    
    # 2. Video ID overlap check
    print("\n--- Video ID Overlap Check ---")
    leaked_videos = [(vid, splits) for vid, splits in video_to_splits.items() if len(splits) > 1]
    if leaked_videos:
        print(f"  ERROR: {len(leaked_videos)} video_ids in multiple splits!")
        for vid, splits in leaked_videos[:5]:
            issues.append(f"Video {vid} in splits: {splits}")
            print(f"    {vid}: {splits}")
    else:
        print(f"  OK: {len(video_to_splits)} videos, no overlap")
    
    # 3. Class balance per split
    print("\n--- Class Balance by Split ---")
    split_counts = defaultdict(lambda: {'real': 0, 'fake': 0, 'total': 0})
    
    for s in samples:
        split_counts[s.split]['total'] += 1
        if s.label == 0:
            split_counts[s.split]['real'] += 1
        else:
            split_counts[s.split]['fake'] += 1
    
    print(f"{'Split':<8} {'Real':>8} {'Fake':>8} {'Total':>8} {'Fake%':>8}")
    print("-" * 50)
    
    for split in ['train', 'val', 'test']:
        if split in split_counts:
            c = split_counts[split]
            fake_pct = 100 * c['fake'] / c['total'] if c['total'] > 0 else 0
            print(f"{split:<8} {c['real']:>8} {c['fake']:>8} {c['total']:>8} {fake_pct:>7.1f}%")
            
            # Check balance (fake should be ~85% for FF++ with 6 methods vs 1 original)
            if fake_pct < 50 or fake_pct > 95:
                issues.append(f"Split {split} has unusual balance: {fake_pct:.1f}% fake")
    
    # 4. Method coverage
    print("\n--- Method Coverage by Split ---")
    split_methods = defaultdict(set)
    for s in samples:
        split_methods[s.split].add(s.method)
    
    all_methods = set(s.method for s in samples)
    for split in ['train', 'val', 'test']:
        missing = all_methods - split_methods.get(split, set())
        if missing and split in ['train', 'val']:
            issues.append(f"Split {split} missing methods: {missing}")
            print(f"  {split}: MISSING {missing}")
        else:
            print(f"  {split}: {len(split_methods.get(split, set()))} methods OK")
    
    # Summary
    print("\n" + "=" * 70)
    total = len(samples)
    print(f"TOTAL SAMPLES: {total}")
    print(f"ISSUES: {len(issues)}")
    print("=" * 70 + "\n")
    
    return len(issues) == 0, issues


def build_sample_manifest(
    videos_csv: Path,
    ffpp_root: Path,
    output_dir: Path,
    sampler: FrameSampler,
    streaming: bool = True,
) -> Tuple[List[SampleRecord], Dict[str, Path]]:
    """Build complete sample manifest from video index.
    
    Args:
        videos_csv: Path to videos_master.csv
        ffpp_root: Root of FF++ videos
        output_dir: Output directory for manifests
        sampler: Configured FrameSampler
        streaming: If True, use streaming mode
        
    Returns:
        Tuple of (all samples, dict of split manifest paths)
    """
    # Load video index
    print("[MANIFEST] Loading video index...")
    videos = load_master_index(videos_csv)
    print(f"  Loaded {len(videos)} videos")
    
    # Generate samples
    print("[MANIFEST] Generating samples...")
    samples = generate_samples_from_videos(
        videos,
        sampler,
        ffpp_root,
        streaming=streaming,
    )
    print(f"  Generated {len(samples)} samples")
    
    # Save master manifest
    output_dir = Path(output_dir)
    master_path = output_dir / "samples_master.csv"
    save_samples_csv(samples, master_path)
    
    # Create split manifests
    print("[MANIFEST] Creating split manifests...")
    split_paths = create_split_manifests(samples, output_dir)
    
    # Validate
    validate_manifest(samples)
    
    return samples, split_paths


if __name__ == "__main__":
    print("Manifest module loaded successfully!")
