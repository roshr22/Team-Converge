"""Video-level master index builder.

Creates videos_master.csv with one row per video file containing:
- dataset: ffpp or dfdc_sample
- video_path: relative path to video
- label: 0=real, 1=fake
- method: manipulation method (original, Deepfakes, Face2Face, etc.)
- group_id: unique source identifier for splitting
- video_id: unique hash of dataset + video_path
- split: train/val/test based on group_id
"""

import csv
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from .splitting import deterministic_split


@dataclass
class VideoRecord:
    """A single video record in the master index."""
    dataset: str
    video_path: str
    label: int
    method: str
    group_id: str
    video_id: str
    split: str


def compute_video_id(dataset: str, video_path: str) -> str:
    """Compute unique video_id from dataset and path.
    
    Args:
        dataset: Dataset name (ffpp, dfdc_sample)
        video_path: Relative path to video
        
    Returns:
        Short hash string
    """
    key = f"{dataset}:{video_path}"
    hash_bytes = hashlib.sha256(key.encode('utf-8')).digest()
    return hash_bytes[:8].hex()


def extract_ffpp_group_id_from_filename(filename: str) -> str:
    """Extract group_id from FF++ video filename.
    
    FF++ naming conventions:
    - Original: {3-digit-id}.mp4 (e.g., "000.mp4", "123.mp4")
    - Standard manipulations: {source}_{target}.mp4 (e.g., "000_001.mp4")
    - DeepFakeDetection: {src}_{tgt}__{scene}__{code}.mp4 (e.g., "01_02__meeting__XYZ.mp4")
    
    The group_id is based on the source identity.
    
    Args:
        filename: Video filename (stem, no extension)
        
    Returns:
        group_id string
    """
    # Check for DeepFakeDetection format: XX_YY__scene__code
    dfd_match = re.match(r'^(\d+)_(\d+)__.*', filename)
    if dfd_match:
        # Use both source and target IDs for DFD
        source_id = dfd_match.group(1).zfill(2)
        target_id = dfd_match.group(2).zfill(2)
        return f"ffpp_dfd_{source_id}_{target_id}"
    
    # Check for standard manipulation format: XXX_YYY
    manip_match = re.match(r'^(\d+)_(\d+)$', filename)
    if manip_match:
        source_id = manip_match.group(1).zfill(3)
        return f"ffpp_{source_id}"
    
    # Original video format: XXX (just digits)
    if filename.isdigit():
        return f"ffpp_{filename.zfill(3)}"
    
    # Fallback: use full filename
    return f"ffpp_{filename}"


def index_ffpp_dataset(ffpp_root: Path) -> List[VideoRecord]:
    """Index all videos in FaceForensics++ dataset.
    
    Expected structure:
    ffpp_root/
        original/           <- REAL
        Deepfakes/          <- FAKE
        Face2Face/          <- FAKE
        FaceSwap/           <- FAKE
        FaceShifter/        <- FAKE
        NeuralTextures/     <- FAKE
        DeepFakeDetection/  <- FAKE
        csv/                <- metadata (skip)
    
    Args:
        ffpp_root: Root directory of FF++ dataset
        
    Returns:
        List of VideoRecord objects
    """
    ffpp_root = Path(ffpp_root)
    records = []
    
    # Define folder -> (label, method) mapping
    folder_mapping = {
        'original': (0, 'original'),
        'Deepfakes': (1, 'Deepfakes'),
        'Face2Face': (1, 'Face2Face'),
        'FaceSwap': (1, 'FaceSwap'),
        'FaceShifter': (1, 'FaceShifter'),
        'NeuralTextures': (1, 'NeuralTextures'),
        'DeepFakeDetection': (1, 'DeepFakeDetection'),
    }
    
    for folder_name, (label, method) in folder_mapping.items():
        folder_path = ffpp_root / folder_name
        if not folder_path.exists():
            print(f"  [SKIP] Folder not found: {folder_path}")
            continue
        
        # Find all video files
        video_extensions = ['.mp4', '.avi', '.mov']
        video_files = []
        for ext in video_extensions:
            video_files.extend(folder_path.glob(f'*{ext}'))
            video_files.extend(folder_path.glob(f'*{ext.upper()}'))
        
        for video_path in sorted(video_files):
            # Relative path from ffpp_root
            rel_path = video_path.relative_to(ffpp_root)
            
            # Extract group_id
            group_id = extract_ffpp_group_id_from_filename(video_path.stem)
            
            # Compute video_id
            video_id = compute_video_id('ffpp', str(rel_path))
            
            # Determine split
            split = deterministic_split(group_id)
            
            records.append(VideoRecord(
                dataset='ffpp',
                video_path=str(rel_path),
                label=label,
                method=method,
                group_id=group_id,
                video_id=video_id,
                split=split,
            ))
    
    return records


def index_dfdc_dataset(dfdc_root: Path) -> List[VideoRecord]:
    """Index all videos in DFDC sample dataset.
    
    Args:
        dfdc_root: Root directory of DFDC sample
        
    Returns:
        List of VideoRecord objects
    """
    import json
    
    dfdc_root = Path(dfdc_root)
    records = []
    
    # Load metadata if available
    metadata_path = dfdc_root / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    # Find all video files
    video_files = list(dfdc_root.glob('*.mp4'))
    
    for video_path in sorted(video_files):
        video_name = video_path.name
        rel_path = video_path.relative_to(dfdc_root)
        
        # Determine label and original from metadata
        video_info = metadata.get(video_name, {})
        label = 0 if video_info.get('label') == 'REAL' else 1
        
        # Group ID based on original video
        if label == 0:
            # Real video is its own group
            original = video_path.stem
        else:
            # Fake video - get original from metadata
            original = video_info.get('original', video_path.stem)
            if '.' in original:
                original = original.rsplit('.', 1)[0]
        
        group_id = f"dfdc_{original}"
        video_id = compute_video_id('dfdc_sample', str(rel_path))
        split = deterministic_split(group_id)
        
        records.append(VideoRecord(
            dataset='dfdc_sample',
            video_path=str(rel_path),
            label=label,
            method='dfdc',
            group_id=group_id,
            video_id=video_id,
            split=split,
        ))
    
    return records


def build_master_index(
    ffpp_root: Optional[Path] = None,
    dfdc_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> List[VideoRecord]:
    """Build the complete videos_master.csv index.
    
    Args:
        ffpp_root: Path to FF++ dataset root
        dfdc_root: Path to DFDC sample root
        output_path: Path to save CSV (optional)
        
    Returns:
        List of all VideoRecord objects
    """
    all_records = []
    
    if ffpp_root and Path(ffpp_root).exists():
        print(f"[INDEX] Indexing FF++ from {ffpp_root}...")
        ffpp_records = index_ffpp_dataset(ffpp_root)
        all_records.extend(ffpp_records)
        print(f"  Found {len(ffpp_records)} videos")
    
    if dfdc_root and Path(dfdc_root).exists():
        print(f"[INDEX] Indexing DFDC from {dfdc_root}...")
        dfdc_records = index_dfdc_dataset(dfdc_root)
        all_records.extend(dfdc_records)
        print(f"  Found {len(dfdc_records)} videos")
    
    if output_path:
        save_master_index(all_records, output_path)
    
    return all_records


def save_master_index(records: List[VideoRecord], output_path: Path) -> None:
    """Save master index to CSV.
    
    Args:
        records: List of VideoRecord objects
        output_path: Path to output CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ['dataset', 'video_path', 'label', 'method', 'group_id', 'video_id', 'split']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    
    print(f"[INDEX] Saved {len(records)} records to {output_path}")


def load_master_index(csv_path: Path) -> List[VideoRecord]:
    """Load master index from CSV.
    
    Args:
        csv_path: Path to videos_master.csv
        
    Returns:
        List of VideoRecord objects
    """
    records = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(VideoRecord(
                dataset=row['dataset'],
                video_path=row['video_path'],
                label=int(row['label']),
                method=row['method'],
                group_id=row['group_id'],
                video_id=row['video_id'],
                split=row['split'],
            ))
    return records


def validate_master_index(records: List[VideoRecord]) -> Tuple[bool, List[str]]:
    """Validate the master index.
    
    Checks:
    1. Print counts by dataset x method x label x split
    2. Confirm every method appears in train and val
    3. Confirm group_id uniqueness across splits
    
    Args:
        records: List of VideoRecord objects
        
    Returns:
        Tuple of (all_valid, list of issues)
    """
    issues = []
    
    # 1. Count by dataset x method x label x split
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    
    for r in records:
        counts[r.dataset][r.method][r.label][r.split] += 1
    
    print("\n" + "=" * 80)
    print("MASTER INDEX VALIDATION")
    print("=" * 80)
    
    print("\n--- Counts by Dataset x Method x Label x Split ---\n")
    print(f"{'Dataset':<12} {'Method':<18} {'Label':<6} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("-" * 80)
    
    for dataset in sorted(counts.keys()):
        for method in sorted(counts[dataset].keys()):
            for label in sorted(counts[dataset][method].keys()):
                split_counts = counts[dataset][method][label]
                train_n = split_counts['train']
                val_n = split_counts['val']
                test_n = split_counts['test']
                total = train_n + val_n + test_n
                label_str = 'real' if label == 0 else 'fake'
                print(f"{dataset:<12} {method:<18} {label_str:<6} {train_n:>6} {val_n:>6} {test_n:>6} {total:>7}")
    
    # 2. Check every method appears in train and val
    print("\n--- Method Coverage Check ---\n")
    
    methods_in_splits = defaultdict(set)
    for r in records:
        methods_in_splits[r.split].add(r.method)
    
    all_methods = set(r.method for r in records)
    
    for method in sorted(all_methods):
        in_train = method in methods_in_splits['train']
        in_val = method in methods_in_splits['val']
        in_test = method in methods_in_splits['test']
        
        status = "OK" if (in_train and in_val) else "WARN"
        if not in_train:
            issues.append(f"Method '{method}' missing from train split")
            status = "FAIL"
        if not in_val:
            issues.append(f"Method '{method}' missing from val split")
            status = "FAIL"
        
        print(f"  {method:<18} train={in_train} val={in_val} test={in_test} [{status}]")
    
    # 3. Check group_id uniqueness across splits
    print("\n--- Group ID Uniqueness Check ---\n")
    
    group_to_splits = defaultdict(set)
    for r in records:
        group_to_splits[r.group_id].add(r.split)
    
    leaked_groups = [(gid, splits) for gid, splits in group_to_splits.items() if len(splits) > 1]
    
    if leaked_groups:
        print(f"  ERROR: {len(leaked_groups)} group_ids appear in multiple splits!")
        for gid, splits in leaked_groups[:10]:
            issues.append(f"Group '{gid}' in splits: {splits}")
            print(f"    {gid}: {splits}")
        if len(leaked_groups) > 10:
            print(f"    ... and {len(leaked_groups) - 10} more")
    else:
        print(f"  OK: All {len(group_to_splits)} group_ids are unique to their splits")
    
    # Summary
    print("\n" + "=" * 80)
    total_records = len(records)
    total_groups = len(group_to_splits)
    real_count = sum(1 for r in records if r.label == 0)
    fake_count = sum(1 for r in records if r.label == 1)
    print(f"SUMMARY: {total_records} videos, {total_groups} groups, {real_count} real, {fake_count} fake")
    print(f"ISSUES: {len(issues)}")
    print("=" * 80 + "\n")
    
    return len(issues) == 0, issues


if __name__ == "__main__":
    # Test with local paths
    print("Master index module loaded successfully!")
