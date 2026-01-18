"""Video-grouped splitting utilities.

Implements deterministic, hash-based splitting at the group_id level
to prevent data leakage between train/val/test sets.

Key concepts:
- group_id: Represents the underlying source video/identity unit
- Split rule: hash(group_id) mod 100 determines assignment
- Mapping: 0-79 train, 80-89 val, 90-99 test

For FaceForensics++:
  group_id = base filename shared across original and manipulations
  
For DFDC:
  group_id = original video ID (from metadata linking fakes to originals)
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


# Split boundaries (80/10/10)
TRAIN_UPPER = 80  # 0-79
VAL_UPPER = 90    # 80-89
# TEST: 90-99


@dataclass
class VideoInfo:
    """Information about a single video file."""
    path: Path
    group_id: str
    video_id: str
    label: int  # 0 = real, 1 = fake
    manipulation_type: Optional[str] = None  # For FF++: Deepfakes, Face2Face, etc.


def deterministic_split(group_id: str) -> str:
    """Assign a group_id to a split using deterministic hash.
    
    Uses MD5 hash mod 100 for consistent assignment.
    
    Args:
        group_id: The group identifier
        
    Returns:
        Split name: 'train', 'val', or 'test'
    """
    # Use MD5 for fast, consistent hashing
    hash_bytes = hashlib.md5(group_id.encode('utf-8')).digest()
    hash_int = int.from_bytes(hash_bytes[:4], 'little')
    bucket = hash_int % 100
    
    if bucket < TRAIN_UPPER:
        return 'train'
    elif bucket < VAL_UPPER:
        return 'val'
    else:
        return 'test'


def extract_ffpp_group_id(video_path: Path) -> str:
    """Extract group_id from FaceForensics++ video path.
    
    FF++ naming convention:
    - Original: {id}.mp4  (e.g., "000.mp4")
    - Manipulated: {source_id}_{target_id}.mp4  (e.g., "000_001.mp4")
    
    The group_id is the source_id (first number), which links:
    - The original video
    - All manipulated videos derived from it
    
    Args:
        video_path: Path to video file
        
    Returns:
        group_id string
    """
    stem = video_path.stem  # e.g., "000" or "000_001"
    
    # Check if it's a manipulated video (contains underscore)
    if '_' in stem:
        # Format: source_target -> take source
        source_id = stem.split('_')[0]
    else:
        # Original video - the ID itself is the group_id
        source_id = stem
    
    # Normalize to 3 digits
    source_id = source_id.zfill(3)
    
    return f"ffpp_{source_id}"


def extract_dfdc_group_id(video_path: Path, metadata: Optional[Dict] = None) -> str:
    """Extract group_id from DFDC video path using metadata.
    
    DFDC provides metadata.json linking fake videos to their originals.
    The group_id is the original video's ID.
    
    Args:
        video_path: Path to video file
        metadata: Optional pre-loaded metadata dict
        
    Returns:
        group_id string
    """
    video_name = video_path.name
    
    if metadata is None:
        # Without metadata, use the video name directly
        # This is less ideal but allows processing without metadata
        return f"dfdc_{video_path.stem}"
    
    # Check if this video has an "original" field in metadata
    video_info = metadata.get(video_name, {})
    
    if video_info.get('label') == 'REAL':
        # Real video - it IS the original
        return f"dfdc_{video_path.stem}"
    else:
        # Fake video - get its original
        original = video_info.get('original', video_path.stem)
        # Remove extension if present
        if '.' in original:
            original = original.rsplit('.', 1)[0]
        return f"dfdc_{original}"


def index_ffpp_videos(ffpp_root: Path) -> List[VideoInfo]:
    """Index all videos in FaceForensics++ directory.
    
    Expected structure:
    ffpp_root/
        original_sequences/
            youtube/
                c23/  (or c40)
                    videos/
                        000.mp4
                        ...
        manipulated_sequences/
            Deepfakes/
                c23/
                    videos/
                        000_001.mp4
                        ...
            Face2Face/
            FaceSwap/
            NeuralTextures/
    
    Args:
        ffpp_root: Root directory of FF++ dataset
        
    Returns:
        List of VideoInfo objects
    """
    videos = []
    ffpp_root = Path(ffpp_root)
    
    # Index original (real) videos
    original_dir = ffpp_root / "original_sequences" / "youtube"
    for compression in ["c23", "c40"]:
        video_dir = original_dir / compression / "videos"
        if video_dir.exists():
            for video_path in video_dir.glob("*.mp4"):
                videos.append(VideoInfo(
                    path=video_path,
                    group_id=extract_ffpp_group_id(video_path),
                    video_id=f"ffpp_orig_{video_path.stem}_{compression}",
                    label=0,  # Real
                    manipulation_type="original",
                ))
    
    # Index manipulated (fake) videos
    manipulation_types = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    manipulated_dir = ffpp_root / "manipulated_sequences"
    
    for manip_type in manipulation_types:
        for compression in ["c23", "c40"]:
            video_dir = manipulated_dir / manip_type / compression / "videos"
            if video_dir.exists():
                for video_path in video_dir.glob("*.mp4"):
                    videos.append(VideoInfo(
                        path=video_path,
                        group_id=extract_ffpp_group_id(video_path),
                        video_id=f"ffpp_{manip_type}_{video_path.stem}_{compression}",
                        label=1,  # Fake
                        manipulation_type=manip_type,
                    ))
    
    return videos


def index_dfdc_videos(dfdc_root: Path) -> List[VideoInfo]:
    """Index all videos in DFDC sample directory.
    
    Expected structure:
    dfdc_root/
        metadata.json
        *.mp4
    
    Args:
        dfdc_root: Root directory of DFDC sample
        
    Returns:
        List of VideoInfo objects
    """
    import json
    
    videos = []
    dfdc_root = Path(dfdc_root)
    
    # Load metadata if available
    metadata_path = dfdc_root / "metadata.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Index all videos
    for video_path in dfdc_root.glob("*.mp4"):
        video_name = video_path.name
        
        # Determine label from metadata or filename
        if metadata and video_name in metadata:
            label = 0 if metadata[video_name].get('label') == 'REAL' else 1
        else:
            # Fallback: assume FAKE if no metadata
            label = 1
        
        videos.append(VideoInfo(
            path=video_path,
            group_id=extract_dfdc_group_id(video_path, metadata),
            video_id=f"dfdc_{video_path.stem}",
            label=label,
            manipulation_type="dfdc",
        ))
    
    return videos


def assign_splits(videos: List[VideoInfo]) -> Dict[str, List[VideoInfo]]:
    """Assign videos to splits based on their group_id.
    
    Args:
        videos: List of VideoInfo objects
        
    Returns:
        Dict with 'train', 'val', 'test' keys containing video lists
    """
    splits = {'train': [], 'val': [], 'test': []}
    
    for video in videos:
        split = deterministic_split(video.group_id)
        splits[split].append(video)
    
    return splits


def save_group_lists(splits: Dict[str, List[VideoInfo]], output_dir: Path) -> None:
    """Save group_id lists to text files.
    
    Creates:
    - train_groups.txt
    - val_groups.txt
    - test_groups.txt
    
    Args:
        splits: Dict with split assignments
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, videos in splits.items():
        # Get unique group_ids
        group_ids = sorted(set(v.group_id for v in videos))
        
        output_path = output_dir / f"{split_name}_groups.txt"
        with open(output_path, 'w') as f:
            for gid in group_ids:
                f.write(f"{gid}\n")
        
        print(f"[SPLITS] Saved {len(group_ids)} groups to {output_path}")


def verify_no_leakage(splits_dir: Path) -> bool:
    """Verify no group_id appears in multiple split files.
    
    This is a HARD CHECK that must pass before training.
    
    Args:
        splits_dir: Directory containing *_groups.txt files
        
    Returns:
        True if no leakage, raises ValueError otherwise
    """
    splits_dir = Path(splits_dir)
    
    all_groups: Dict[str, str] = {}  # group_id -> first split found
    leaks = []
    
    for split_file in ['train_groups.txt', 'val_groups.txt', 'test_groups.txt']:
        file_path = splits_dir / split_file
        if not file_path.exists():
            continue
        
        split_name = split_file.replace('_groups.txt', '')
        
        with open(file_path, 'r') as f:
            for line in f:
                group_id = line.strip()
                if not group_id:
                    continue
                
                if group_id in all_groups:
                    leaks.append(f"{group_id}: found in both {all_groups[group_id]} and {split_name}")
                else:
                    all_groups[group_id] = split_name
    
    if leaks:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! The following group_ids appear in multiple splits:\n"
            + "\n".join(leaks[:10])
            + (f"\n... and {len(leaks) - 10} more" if len(leaks) > 10 else "")
        )
    
    print(f"[VERIFY] No leakage detected. {len(all_groups)} unique groups across splits.")
    return True


def load_split_groups(splits_dir: Path, split_name: str) -> Set[str]:
    """Load group_ids for a specific split.
    
    Args:
        splits_dir: Directory containing *_groups.txt files
        split_name: 'train', 'val', or 'test'
        
    Returns:
        Set of group_ids
    """
    file_path = Path(splits_dir) / f"{split_name}_groups.txt"
    
    if not file_path.exists():
        return set()
    
    with open(file_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def print_split_summary(splits: Dict[str, List[VideoInfo]]) -> None:
    """Print summary statistics of split assignments."""
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    
    for split_name in ['train', 'val', 'test']:
        videos = splits[split_name]
        groups = set(v.group_id for v in videos)
        real_count = sum(1 for v in videos if v.label == 0)
        fake_count = sum(1 for v in videos if v.label == 1)
        
        print(f"\n{split_name.upper()}:")
        print(f"  Groups: {len(groups)}")
        print(f"  Videos: {len(videos)} (Real: {real_count}, Fake: {fake_count})")
        
        # By manipulation type
        manip_counts = {}
        for v in videos:
            mt = v.manipulation_type or 'unknown'
            manip_counts[mt] = manip_counts.get(mt, 0) + 1
        
        for mt, count in sorted(manip_counts.items()):
            print(f"    - {mt}: {count}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Testing deterministic split function...")
    
    test_ids = ["ffpp_000", "ffpp_001", "ffpp_055", "dfdc_abc123", "dfdc_xyz789"]
    for gid in test_ids:
        split = deterministic_split(gid)
        hash_val = int.from_bytes(hashlib.md5(gid.encode()).digest()[:4], 'little') % 100
        print(f"  {gid} -> bucket {hash_val} -> {split}")
    
    print("\nVerification test...")
    # Create mock videos
    mock_videos = [
        VideoInfo(Path("test.mp4"), "ffpp_000", "vid1", 0),
        VideoInfo(Path("test2.mp4"), "ffpp_000", "vid2", 1),  # Same group
        VideoInfo(Path("test3.mp4"), "ffpp_001", "vid3", 0),
    ]
    
    splits = assign_splits(mock_videos)
    print_split_summary(splits)
