#!/usr/bin/env python
"""Copy FF++ dataset from Google Drive to Colab local disk.

Usage (Colab):
    !python copy_to_local.py --source /content/drive/MyDrive/data/raw/ffpp --dest /content/data/raw/ffpp

This significantly speeds up training by avoiding Drive I/O latency.
"""

import argparse
import shutil
import time
from pathlib import Path


def copy_with_progress(src: Path, dst: Path) -> tuple[int, int]:
    """Copy directory tree with progress tracking.
    
    Args:
        src: Source directory
        dst: Destination directory
        
    Returns:
        Tuple of (files_copied, total_bytes)
    """
    src = Path(src)
    dst = Path(dst)
    
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    
    # Count files first
    all_files = list(src.rglob("*"))
    files_only = [f for f in all_files if f.is_file()]
    total_files = len(files_only)
    
    print(f"[COPY] Source: {src}")
    print(f"[COPY] Destination: {dst}")
    print(f"[COPY] Files to copy: {total_files}")
    
    # Create destination
    dst.mkdir(parents=True, exist_ok=True)
    
    # Copy with progress
    copied = 0
    total_bytes = 0
    start_time = time.time()
    
    for i, file_path in enumerate(files_only):
        rel_path = file_path.relative_to(src)
        dest_path = dst / rel_path
        
        # Create parent dirs
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(file_path, dest_path)
        copied += 1
        total_bytes += file_path.stat().st_size
        
        # Progress every 100 files
        if (i + 1) % 100 == 0 or (i + 1) == total_files:
            elapsed = time.time() - start_time
            rate = total_bytes / (1024 * 1024 * elapsed) if elapsed > 0 else 0
            print(f"  [{i+1}/{total_files}] {total_bytes / 1e9:.2f} GB copied ({rate:.1f} MB/s)")
    
    elapsed = time.time() - start_time
    print(f"[COPY] Complete: {copied} files, {total_bytes / 1e9:.2f} GB in {elapsed:.1f}s")
    
    return copied, total_bytes


def verify_copy(src: Path, dst: Path) -> bool:
    """Verify that copy succeeded by comparing file counts.
    
    Args:
        src: Source directory
        dst: Destination directory
        
    Returns:
        True if verification passed
    """
    src_files = set(p.relative_to(src) for p in Path(src).rglob("*") if p.is_file())
    dst_files = set(p.relative_to(dst) for p in Path(dst).rglob("*") if p.is_file())
    
    missing = src_files - dst_files
    extra = dst_files - src_files
    
    if missing:
        print(f"[VERIFY] ERROR: {len(missing)} files missing from destination")
        for f in list(missing)[:5]:
            print(f"  - {f}")
        return False
    
    if extra:
        print(f"[VERIFY] WARNING: {len(extra)} extra files in destination")
    
    print(f"[VERIFY] OK: {len(dst_files)} files verified")
    return True


def main():
    parser = argparse.ArgumentParser(description="Copy FF++ from Drive to local disk")
    parser.add_argument("--source", required=True, help="Source directory (Drive)")
    parser.add_argument("--dest", required=True, help="Destination directory (local)")
    parser.add_argument("--skip-verify", action="store_true", help="Skip verification")
    parser.add_argument("--force", action="store_true", help="Overwrite existing destination")
    
    args = parser.parse_args()
    
    src = Path(args.source)
    dst = Path(args.dest)
    
    # Check if destination exists
    if dst.exists() and not args.force:
        existing_files = list(dst.rglob("*"))
        if existing_files:
            print(f"[COPY] Destination already exists with {len(existing_files)} files")
            print("[COPY] Use --force to overwrite or skip this step")
            return
    
    # Perform copy
    copy_with_progress(src, dst)
    
    # Verify
    if not args.skip_verify:
        verify_copy(src, dst)


if __name__ == "__main__":
    main()
