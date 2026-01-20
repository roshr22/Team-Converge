"""Diagnostic script for face pre-extraction pipeline issues.

This script tests each component of the face extraction pipeline
to identify the exact failure points.
"""

import os
import sys
import csv
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class DiagnosticResult:
    """Result of a diagnostic test."""
    test_name: str
    passed: bool
    message: str
    details: Optional[str] = None

class FaceExtractionDiagnostics:
    """Run comprehensive diagnostics on the face extraction pipeline."""
    
    def __init__(self, ffpp_root: str, manifest_path: str, cache_dir: str):
        self.ffpp_root = Path(ffpp_root)
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir)
        self.results: List[DiagnosticResult] = []
        
    def add_result(self, test_name: str, passed: bool, message: str, details: str = None):
        """Add a diagnostic result."""
        result = DiagnosticResult(test_name, passed, message, details)
        self.results.append(result)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {test_name}")
        print(f"   {message}")
        if details:
            for line in details.split('\n')[:5]:  # Limit details
                print(f"   > {line}")
        print()
        
    def test_ffpp_root_exists(self) -> bool:
        """Test 1: Check if FF++ root directory exists."""
        exists = self.ffpp_root.exists()
        is_dir = self.ffpp_root.is_dir() if exists else False
        
        if exists and is_dir:
            contents = list(self.ffpp_root.iterdir())[:10]
            self.add_result(
                "FF++ Root Directory",
                True,
                f"Directory exists: {self.ffpp_root}",
                f"Contents: {[c.name for c in contents]}"
            )
            return True
        else:
            self.add_result(
                "FF++ Root Directory",
                False,
                f"Directory does NOT exist or is not a directory: {self.ffpp_root}",
                "This is a critical path issue - videos cannot be found"
            )
            return False
            
    def test_manifest_exists(self) -> bool:
        """Test 2: Check if manifest file exists and is readable."""
        if not self.manifest_path.exists():
            self.add_result(
                "Manifest File",
                False,
                f"Manifest file does NOT exist: {self.manifest_path}"
            )
            return False
            
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if len(rows) == 0:
                self.add_result(
                    "Manifest File",
                    False,
                    "Manifest file is empty"
                )
                return False
                
            # Check required columns
            required_cols = ['sample_id', 'video_path', 'timestamp', 'split']
            first_row = rows[0]
            missing_cols = [c for c in required_cols if c not in first_row]
            
            if missing_cols:
                self.add_result(
                    "Manifest File",
                    False,
                    f"Missing required columns: {missing_cols}",
                    f"Available columns: {list(first_row.keys())}"
                )
                return False
                
            self.add_result(
                "Manifest File",
                True,
                f"Manifest has {len(rows)} samples with all required columns",
                f"Sample video_path: {first_row.get('video_path', 'N/A')}"
            )
            return True
            
        except Exception as e:
            self.add_result(
                "Manifest File",
                False,
                f"Error reading manifest: {e}"
            )
            return False
            
    def test_video_path_format(self) -> Tuple[bool, List[str]]:
        """Test 3: Check video path format and actual existence."""
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            self.add_result(
                "Video Path Format",
                False,
                f"Could not read manifest: {e}"
            )
            return False, []
            
        # Check a sample of video paths
        sample_size = min(20, len(rows))
        sample_rows = rows[:sample_size]
        
        issues = []
        paths_tested = []
        
        for row in sample_rows:
            video_path = row.get('video_path', '')
            
            # Issue 1: Check for Windows backslashes
            if '\\' in video_path:
                issues.append(f"Windows backslash in path: {video_path}")
                
            # Construct full path
            full_path_raw = self.ffpp_root / video_path
            
            # Also try with normalized path
            normalized_video_path = video_path.replace('\\', '/')
            full_path_normalized = self.ffpp_root / normalized_video_path
            
            paths_tested.append({
                'raw': str(full_path_raw),
                'normalized': str(full_path_normalized),
                'raw_exists': full_path_raw.exists(),
                'normalized_exists': full_path_normalized.exists(),
                'original_path': video_path
            })
            
        # Analyze results
        raw_exists_count = sum(1 for p in paths_tested if p['raw_exists'])
        normalized_exists_count = sum(1 for p in paths_tested if p['normalized_exists'])
        
        # Format issues for display
        path_issues = []
        for p in paths_tested[:5]:  # Show first 5
            path_issues.append(
                f"Path: {p['original_path']} -> raw_exists={p['raw_exists']}, normalized_exists={p['normalized_exists']}"
            )
            
        has_backslash_issue = any('\\' in row.get('video_path', '') for row in sample_rows)
        
        if raw_exists_count == 0 and normalized_exists_count == 0:
            # Neither format works - likely root path issue
            self.add_result(
                "Video Path Resolution",
                False,
                f"0/{sample_size} video files found with either path format",
                '\n'.join(path_issues + [f"\nFF++ root: {self.ffpp_root}"])
            )
            return False, paths_tested
        elif raw_exists_count < sample_size and has_backslash_issue:
            # Backslash issue
            self.add_result(
                "Video Path Resolution",
                False,
                f"Path separator issue: manifest uses backslashes but files use forward slashes",
                f"Raw exists: {raw_exists_count}/{sample_size}, Normalized exists: {normalized_exists_count}/{sample_size}\n" + 
                '\n'.join(path_issues[:3])
            )
            return False, paths_tested
        else:
            self.add_result(
                "Video Path Resolution",
                True,
                f"{raw_exists_count}/{sample_size} video files found",
                '\n'.join(path_issues[:3])
            )
            return True, paths_tested
            
    def test_ffmpeg_availability(self) -> bool:
        """Test 4: Check if ffmpeg is installed and accessible."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"
                self.add_result(
                    "FFmpeg Availability",
                    True,
                    "FFmpeg is installed and accessible",
                    version_line
                )
                return True
            else:
                self.add_result(
                    "FFmpeg Availability",
                    False,
                    "FFmpeg returned non-zero exit code",
                    result.stderr[:200] if result.stderr else "No error message"
                )
                return False
        except FileNotFoundError:
            self.add_result(
                "FFmpeg Availability",
                False,
                "FFmpeg is NOT installed or not in PATH",
                "Install ffmpeg: https://ffmpeg.org/download.html"
            )
            return False
        except Exception as e:
            self.add_result(
                "FFmpeg Availability",
                False,
                f"Error checking ffmpeg: {e}"
            )
            return False
            
    def test_ffprobe_availability(self) -> bool:
        """Test 5: Check if ffprobe is installed (needed for video dimensions)."""
        try:
            result = subprocess.run(
                ['ffprobe', '-version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"
                self.add_result(
                    "FFprobe Availability",
                    True,
                    "FFprobe is installed and accessible",
                    version_line
                )
                return True
            else:
                self.add_result(
                    "FFprobe Availability",
                    False,
                    "FFprobe returned non-zero exit code"
                )
                return False
        except FileNotFoundError:
            self.add_result(
                "FFprobe Availability",
                False,
                "FFprobe is NOT installed or not in PATH",
                "FFprobe typically comes with FFmpeg installation"
            )
            return False
        except Exception as e:
            self.add_result(
                "FFprobe Availability",
                False,
                f"Error checking ffprobe: {e}"
            )
            return False
            
    def test_mediapipe_import(self) -> bool:
        """Test 6: Check if MediaPipe can be imported."""
        try:
            import mediapipe as mp
            version = getattr(mp, '__version__', 'Unknown')
            self.add_result(
                "MediaPipe Import",
                True,
                f"MediaPipe {version} is installed and importable"
            )
            return True
        except ImportError as e:
            self.add_result(
                "MediaPipe Import",
                False,
                "MediaPipe is NOT installed",
                f"Install with: pip install mediapipe==0.10.9\nError: {e}"
            )
            return False
        except Exception as e:
            self.add_result(
                "MediaPipe Import",
                False,
                f"Error importing MediaPipe: {e}"
            )
            return False
            
    def test_face_detector_init(self) -> bool:
        """Test 7: Check if FaceDetector can be initialized."""
        try:
            from utils.face_extraction import FaceDetector
            detector = FaceDetector()
            # Force lazy initialization
            import numpy as np
            test_image = np.zeros((256, 256, 3), dtype=np.uint8)
            _ = detector.detect(test_image)
            detector.close()
            
            self.add_result(
                "FaceDetector Initialization",
                True,
                "FaceDetector initializes and runs without error"
            )
            return True
        except ImportError as e:
            self.add_result(
                "FaceDetector Initialization",
                False,
                f"Could not import FaceDetector: {e}"
            )
            return False
        except Exception as e:
            self.add_result(
                "FaceDetector Initialization",
                False,
                f"FaceDetector initialization failed: {e}"
            )
            return False
            
    def test_frame_extraction(self) -> bool:
        """Test 8: Test extracting a frame from an actual video."""
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            self.add_result(
                "Frame Extraction",
                False,
                f"Could not read manifest: {e}"
            )
            return False
            
        if not rows:
            self.add_result(
                "Frame Extraction",
                False,
                "No samples in manifest"
            )
            return False
            
        # Find a video that exists
        test_video = None
        test_timestamp = None
        for row in rows[:50]:
            video_path = row.get('video_path', '').replace('\\', '/')
            full_path = self.ffpp_root / video_path
            if full_path.exists():
                test_video = full_path
                test_timestamp = float(row.get('timestamp', 0.1))
                break
                
        if test_video is None:
            self.add_result(
                "Frame Extraction",
                False,
                "Could not find any existing video file to test",
                "Check Video Path Resolution test above"
            )
            return False
            
        # Try to extract a frame
        try:
            from utils.face_extraction import decode_frame_ffmpeg
            frame = decode_frame_ffmpeg(test_video, test_timestamp)
            
            if frame is None:
                self.add_result(
                    "Frame Extraction",
                    False,
                    f"decode_frame_ffmpeg returned None for {test_video}",
                    f"Timestamp: {test_timestamp}s"
                )
                return False
            else:
                self.add_result(
                    "Frame Extraction",
                    True,
                    f"Successfully extracted frame from {test_video.name}",
                    f"Frame shape: {frame.shape}, timestamp: {test_timestamp}s"
                )
                return True
        except Exception as e:
            self.add_result(
                "Frame Extraction",
                False,
                f"Frame extraction failed: {e}",
                f"Video: {test_video}, Timestamp: {test_timestamp}"
            )
            return False
            
    def test_cache_dir_writable(self) -> bool:
        """Test 9: Check if cache directory is writable."""
        try:
            # Try to create the cache directory if it doesn't exist
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Try to write a test file
            test_file = self.cache_dir / ".write_test"
            test_file.write_text("test")
            test_file.unlink()
            
            self.add_result(
                "Cache Directory Writable",
                True,
                f"Cache directory is writable: {self.cache_dir}"
            )
            return True
        except PermissionError:
            self.add_result(
                "Cache Directory Writable",
                False,
                f"Permission denied for cache directory: {self.cache_dir}"
            )
            return False
        except Exception as e:
            self.add_result(
                "Cache Directory Writable",
                False,
                f"Cache directory error: {e}"
            )
            return False
            
    def test_end_to_end_extraction(self) -> bool:
        """Test 10: Full end-to-end face extraction test."""
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            self.add_result(
                "End-to-End Extraction",
                False,
                f"Could not read manifest: {e}"
            )
            return False
            
        # Find a video that exists
        test_sample = None
        for row in rows[:50]:
            video_path = row.get('video_path', '').replace('\\', '/')
            full_path = self.ffpp_root / video_path
            if full_path.exists():
                test_sample = row
                break
                
        if test_sample is None:
            self.add_result(
                "End-to-End Extraction",
                False,
                "Could not find any existing video to test"
            )
            return False
            
        try:
            from utils.face_extraction import (
                FaceDetector, decode_frame_ffmpeg, expand_bbox, crop_and_resize
            )
            
            video_path = test_sample['video_path'].replace('\\', '/')
            full_path = self.ffpp_root / video_path
            timestamp = float(test_sample.get('timestamp', 0.1))
            
            # Step 1: Decode frame
            frame = decode_frame_ffmpeg(full_path, timestamp)
            if frame is None:
                self.add_result(
                    "End-to-End Extraction",
                    False,
                    "Frame decode step failed"
                )
                return False
                
            # Step 2: Detect face
            detector = FaceDetector()
            bbox = detector.get_primary_face(frame)
            detector.close()
            
            if bbox is None:
                self.add_result(
                    "End-to-End Extraction",
                    False,
                    "Face detection step failed - no face found in frame",
                    f"Video: {video_path}, Timestamp: {timestamp}s"
                )
                return False
                
            # Step 3: Expand bbox and crop
            expanded = expand_bbox(bbox, margin=1.3)
            crop = crop_and_resize(frame, expanded, target_size=256)
            
            # Step 4: Save test
            test_path = self.cache_dir / "_test_extraction.jpg"
            test_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(test_path, format='JPEG', quality=95)
            
            if test_path.exists():
                file_size = test_path.stat().st_size
                test_path.unlink()  # Clean up
                
                self.add_result(
                    "End-to-End Extraction",
                    True,
                    "Full extraction pipeline works correctly",
                    f"Extracted face from {video_path}, saved {file_size} bytes"
                )
                return True
            else:
                self.add_result(
                    "End-to-End Extraction",
                    False,
                    "File save step failed"
                )
                return False
                
        except Exception as e:
            import traceback
            self.add_result(
                "End-to-End Extraction",
                False,
                f"End-to-end test failed: {e}",
                traceback.format_exc()[:500]
            )
            return False
            
    def run_all_tests(self):
        """Run all diagnostic tests."""
        print("=" * 60)
        print("FACE PRE-EXTRACTION PIPELINE DIAGNOSTICS")
        print("=" * 60)
        print(f"\nFF++ Root: {self.ffpp_root}")
        print(f"Manifest: {self.manifest_path}")
        print(f"Cache Dir: {self.cache_dir}")
        print("\n" + "=" * 60 + "\n")
        
        # Run tests
        self.test_ffpp_root_exists()
        self.test_manifest_exists()
        self.test_video_path_format()
        self.test_ffmpeg_availability()
        self.test_ffprobe_availability()
        self.test_mediapipe_import()
        self.test_face_detector_init()
        self.test_frame_extraction()
        self.test_cache_dir_writable()
        self.test_end_to_end_extraction()
        
        # Summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        print(f"\nPassed: {passed}/{len(self.results)}")
        print(f"Failed: {failed}/{len(self.results)}")
        
        if failed > 0:
            print("\n" + "=" * 60)
            print("ISSUES FOUND:")
            print("=" * 60)
            for i, r in enumerate(self.results):
                if not r.passed:
                    print(f"\n{i+1}. {r.test_name}")
                    print(f"   Problem: {r.message}")
                    if r.details:
                        print(f"   Details: {r.details[:200]}...")
                        
        return self.results


def main():
    """Run diagnostics with default paths."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose face extraction pipeline issues")
    parser.add_argument("--ffpp_root", type=str, 
                       default="data/raw/ffpp/FaceForensics++_C23",
                       help="Path to FF++ root directory")
    parser.add_argument("--manifest", type=str,
                       default="artifacts/manifests/train.csv",
                       help="Path to manifest CSV")
    parser.add_argument("--cache_dir", type=str,
                       default="cache/faces",
                       help="Path to cache directory")
    
    args = parser.parse_args()
    
    # Make paths absolute if relative
    base_dir = Path(__file__).parent
    ffpp_root = Path(args.ffpp_root)
    if not ffpp_root.is_absolute():
        ffpp_root = base_dir / ffpp_root
        
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = base_dir / manifest
        
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = base_dir / cache_dir
    
    diagnostics = FaceExtractionDiagnostics(
        ffpp_root=str(ffpp_root),
        manifest_path=str(manifest),
        cache_dir=str(cache_dir)
    )
    
    diagnostics.run_all_tests()


if __name__ == "__main__":
    main()
