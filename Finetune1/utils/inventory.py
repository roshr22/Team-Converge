"""File inventory utilities for dataset integrity verification.

Creates CSV inventories with file paths, sizes, and SHA256 hashes
to detect incomplete downloads and silent corruption.
"""

import csv
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class FileInfo:
    """Information about a single file."""
    filepath: str
    size_bytes: int
    sha256: str


def compute_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file.
    
    Args:
        file_path: Path to file
        chunk_size: Bytes to read at a time
        
    Returns:
        Hex string of SHA256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def inventory_directory(
    root_dir: Path,
    extensions: Optional[List[str]] = None,
    skip_hash: bool = False,
    progress_callback=None,
) -> List[FileInfo]:
    """Create inventory of all files in a directory.
    
    Args:
        root_dir: Root directory to inventory
        extensions: List of extensions to include (e.g., ['.mp4', '.avi'])
                   If None, includes all files
        skip_hash: If True, skip SHA256 computation (faster)
        progress_callback: Optional function(current, total) for progress
        
    Returns:
        List of FileInfo objects
    """
    root_dir = Path(root_dir)
    
    # Collect all files first
    files = []
    for path in root_dir.rglob('*'):
        if path.is_file():
            if extensions is None or path.suffix.lower() in extensions:
                files.append(path)
    
    files.sort()
    total = len(files)
    
    # Build inventory
    inventory = []
    for i, file_path in enumerate(files):
        relative_path = file_path.relative_to(root_dir)
        size = file_path.stat().st_size
        
        if skip_hash:
            hash_val = ""
        else:
            hash_val = compute_sha256(file_path)
        
        inventory.append(FileInfo(
            filepath=str(relative_path),
            size_bytes=size,
            sha256=hash_val,
        ))
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return inventory


def save_inventory_csv(inventory: List[FileInfo], output_path: Path) -> None:
    """Save inventory to CSV file.
    
    Args:
        inventory: List of FileInfo objects
        output_path: Path to output CSV
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'size_bytes', 'sha256'])
        for info in inventory:
            writer.writerow([info.filepath, info.size_bytes, info.sha256])
    
    print(f"[INVENTORY] Saved {len(inventory)} files to {output_path}")


def load_inventory_csv(csv_path: Path) -> List[FileInfo]:
    """Load inventory from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of FileInfo objects
    """
    inventory = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inventory.append(FileInfo(
                filepath=row['filepath'],
                size_bytes=int(row['size_bytes']),
                sha256=row['sha256'],
            ))
    return inventory


def verify_inventory(
    root_dir: Path,
    csv_path: Path,
    check_hash: bool = True,
) -> Tuple[bool, List[str]]:
    """Verify files against saved inventory.
    
    Args:
        root_dir: Root directory of files
        csv_path: Path to inventory CSV
        check_hash: If True, verify SHA256 hashes
        
    Returns:
        Tuple of (all_ok, list of issues)
    """
    root_dir = Path(root_dir)
    inventory = load_inventory_csv(csv_path)
    
    issues = []
    
    for info in inventory:
        file_path = root_dir / info.filepath
        
        # Check existence
        if not file_path.exists():
            issues.append(f"MISSING: {info.filepath}")
            continue
        
        # Check size
        actual_size = file_path.stat().st_size
        if actual_size != info.size_bytes:
            issues.append(f"SIZE_MISMATCH: {info.filepath} (expected {info.size_bytes}, got {actual_size})")
            continue
        
        # Check hash
        if check_hash and info.sha256:
            actual_hash = compute_sha256(file_path)
            if actual_hash != info.sha256:
                issues.append(f"HASH_MISMATCH: {info.filepath}")
    
    return len(issues) == 0, issues


def create_ffpp_inventory(ffpp_root: Path, output_dir: Path, skip_hash: bool = False) -> Path:
    """Create inventory for FaceForensics++ dataset.
    
    Args:
        ffpp_root: Root directory of FF++ dataset
        output_dir: Directory to save inventory CSV
        skip_hash: If True, skip SHA256 computation
        
    Returns:
        Path to created CSV file
    """
    print(f"[INVENTORY] Scanning FF++ directory: {ffpp_root}")
    
    def progress(current, total):
        if current % 100 == 0 or current == total:
            print(f"  Processing {current}/{total} files...")
    
    start_time = time.time()
    
    inventory = inventory_directory(
        ffpp_root,
        extensions=['.mp4', '.avi', '.mov', '.json'],
        skip_hash=skip_hash,
        progress_callback=progress,
    )
    
    elapsed = time.time() - start_time
    
    output_path = Path(output_dir) / "ffpp_files.csv"
    save_inventory_csv(inventory, output_path)
    
    # Summary
    total_size = sum(f.size_bytes for f in inventory)
    print(f"[INVENTORY] Completed in {elapsed:.1f}s")
    print(f"[INVENTORY] Total: {len(inventory)} files, {total_size / 1e9:.2f} GB")
    
    return output_path


def create_dfdc_inventory(dfdc_root: Path, output_dir: Path, skip_hash: bool = False) -> Path:
    """Create inventory for DFDC sample dataset.
    
    Args:
        dfdc_root: Root directory of DFDC sample
        output_dir: Directory to save inventory CSV
        skip_hash: If True, skip SHA256 computation
        
    Returns:
        Path to created CSV file
    """
    print(f"[INVENTORY] Scanning DFDC directory: {dfdc_root}")
    
    def progress(current, total):
        if current % 50 == 0 or current == total:
            print(f"  Processing {current}/{total} files...")
    
    start_time = time.time()
    
    inventory = inventory_directory(
        dfdc_root,
        extensions=['.mp4', '.json'],
        skip_hash=skip_hash,
        progress_callback=progress,
    )
    
    elapsed = time.time() - start_time
    
    output_path = Path(output_dir) / "dfdc_files.csv"
    save_inventory_csv(inventory, output_path)
    
    total_size = sum(f.size_bytes for f in inventory)
    print(f"[INVENTORY] Completed in {elapsed:.1f}s")
    print(f"[INVENTORY] Total: {len(inventory)} files, {total_size / 1e9:.2f} GB")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    # Quick test with current directory
    print("Inventory module loaded successfully!")
    print("Usage: create_ffpp_inventory(ffpp_root, output_dir)")
