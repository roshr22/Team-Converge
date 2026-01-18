"""Frame sampling utilities for video datasets.

Implements Uniform-K sampling: exactly K frames uniformly distributed across video duration.
Uses ffmpeg for accurate timestamp-based extraction.
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FrameSample:
    """A single frame sample from a video."""
    video_path: str
    frame_index: int
    timestamp: float  # seconds


def get_video_info(video_path: Path) -> Tuple[float, int, float]:
    """Get video duration, frame count, and fps using ffprobe.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (duration_seconds, total_frames, fps)
    """
    video_path = Path(video_path)
    
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Get duration from format
        duration = float(info.get('format', {}).get('duration', 0))
        
        # Get video stream info
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if video_stream:
            # Parse frame rate (can be "30/1" or "29.97")
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den)
            else:
                fps = float(fps_str)
            
            # Get frame count
            nb_frames = video_stream.get('nb_frames')
            if nb_frames:
                total_frames = int(nb_frames)
            else:
                total_frames = int(duration * fps)
        else:
            fps = 30.0
            total_frames = int(duration * fps)
        
        return duration, total_frames, fps
        
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        # Fallback: assume 30fps and estimate
        return 0.0, 0, 30.0


def compute_uniform_timestamps(
    duration: float,
    k: int,
    epsilon: float = 0.1,
    min_gap: float = 0.5,
) -> List[float]:
    """Compute K uniformly distributed timestamps with jitter prevention.
    
    Args:
        duration: Video duration in seconds
        k: Number of frames to sample
        epsilon: Margin from start/end in seconds
        min_gap: Minimum gap between samples in seconds
        
    Returns:
        List of timestamps in seconds
    """
    if duration <= 0 or k <= 0:
        return []
    
    # Clamp epsilon to valid range
    epsilon = min(epsilon, duration / 4)
    
    # Compute effective duration
    effective_duration = duration - 2 * epsilon
    
    if effective_duration <= 0:
        # Very short video - just take middle
        return [duration / 2]
    
    if k == 1:
        return [duration / 2]
    
    # Generate uniformly spaced timestamps
    timestamps = np.linspace(epsilon, duration - epsilon, k).tolist()
    
    # Apply jitter prevention: ensure min_gap between consecutive samples
    if min_gap > 0 and len(timestamps) > 1:
        adjusted = [timestamps[0]]
        for t in timestamps[1:]:
            if t - adjusted[-1] < min_gap:
                # Jitter forward
                t = adjusted[-1] + min_gap
            if t < duration - epsilon / 2:
                adjusted.append(t)
        timestamps = adjusted
    
    return timestamps


def extract_frames_ffmpeg(
    video_path: Path,
    timestamps: List[float],
    output_dir: Optional[Path] = None,
) -> List[Tuple[float, bytes]]:
    """Extract frames at exact timestamps using ffmpeg.
    
    Args:
        video_path: Path to video file
        timestamps: List of timestamps in seconds
        output_dir: If provided, save frames as files
        
    Returns:
        List of (timestamp, frame_bytes) tuples
    """
    video_path = Path(video_path)
    frames = []
    
    for i, ts in enumerate(timestamps):
        cmd = [
            'ffmpeg',
            '-ss', f'{ts:.3f}',
            '-i', str(video_path),
            '-vframes', '1',
            '-f', 'image2pipe',
            '-vcodec', 'png',
            '-loglevel', 'error',
            '-'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            frame_bytes = result.stdout
            
            if output_dir:
                output_path = Path(output_dir) / f"frame_{i:04d}_{ts:.3f}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(frame_bytes)
            
            frames.append((ts, frame_bytes))
            
        except subprocess.CalledProcessError:
            # Skip failed frames
            pass
    
    return frames


def sample_video_frames(
    video_path: Path,
    k: int,
    epsilon: float = 0.1,
    min_gap: float = 0.5,
) -> List[FrameSample]:
    """Sample K frames from a video using uniform sampling.
    
    Args:
        video_path: Path to video file
        k: Number of frames to sample
        epsilon: Start/end margin in seconds
        min_gap: Minimum gap between samples
        
    Returns:
        List of FrameSample objects with timestamps
    """
    video_path = Path(video_path)
    
    # Get video info
    duration, total_frames, fps = get_video_info(video_path)
    
    if duration <= 0:
        return []
    
    # Compute timestamps
    timestamps = compute_uniform_timestamps(duration, k, epsilon, min_gap)
    
    # Create samples
    samples = []
    for i, ts in enumerate(timestamps):
        # Estimate frame index from timestamp
        frame_idx = int(ts * fps)
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        
        samples.append(FrameSample(
            video_path=str(video_path),
            frame_index=frame_idx,
            timestamp=ts,
        ))
    
    return samples


def apply_group_cap(
    samples: List[FrameSample],
    group_id: str,
    group_frame_counts: dict,
    max_per_group: int,
) -> List[FrameSample]:
    """Apply per-group frame cap.
    
    Args:
        samples: Frames to potentially filter
        group_id: Group identifier
        group_frame_counts: Running count dict (modified in place)
        max_per_group: Maximum frames per group
        
    Returns:
        Filtered list of FrameSample
    """
    current_count = group_frame_counts.get(group_id, 0)
    remaining = max_per_group - current_count
    
    if remaining <= 0:
        return []
    
    # Take up to remaining frames
    kept = samples[:remaining]
    group_frame_counts[group_id] = current_count + len(kept)
    
    return kept


class FrameSampler:
    """Configurable frame sampler for video datasets."""
    
    def __init__(
        self,
        frames_train_val: int = 10,
        frames_test: int = 20,
        epsilon: float = 0.1,
        min_gap: float = 0.5,
        max_per_video: int = 20,
        max_per_group: int = 100,
    ):
        """Initialize sampler with config.
        
        Args:
            frames_train_val: K for train/val splits
            frames_test: K for test split
            epsilon: Start/end margin
            min_gap: Minimum gap between samples
            max_per_video: Hard cap per video
            max_per_group: Cap per group per epoch
        """
        self.frames_train_val = frames_train_val
        self.frames_test = frames_test
        self.epsilon = epsilon
        self.min_gap = min_gap
        self.max_per_video = max_per_video
        self.max_per_group = max_per_group
        
        # Track group counts for capping
        self._group_counts = {}
    
    def reset_epoch(self):
        """Reset group counts for new epoch."""
        self._group_counts = {}
    
    def get_k_for_split(self, split: str) -> int:
        """Get number of frames K for a given split."""
        if split == 'test':
            return self.frames_test
        return self.frames_train_val
    
    def sample(
        self,
        video_path: Path,
        split: str,
        group_id: str,
    ) -> List[FrameSample]:
        """Sample frames from a video with all caps applied.
        
        Args:
            video_path: Path to video
            split: train/val/test
            group_id: Group identifier for capping
            
        Returns:
            List of FrameSample (may be empty if capped out)
        """
        k = self.get_k_for_split(split)
        k = min(k, self.max_per_video)
        
        # Check group cap
        current = self._group_counts.get(group_id, 0)
        remaining = self.max_per_group - current
        
        if remaining <= 0:
            return []
        
        k = min(k, remaining)
        
        # Sample frames
        samples = sample_video_frames(
            video_path,
            k,
            epsilon=self.epsilon,
            min_gap=self.min_gap,
        )
        
        # Update group count
        self._group_counts[group_id] = current + len(samples)
        
        return samples
    
    @classmethod
    def from_config(cls, config: dict) -> 'FrameSampler':
        """Create sampler from config dict.
        
        Args:
            config: Full config dict with dataset.sampling and dataset.caps
            
        Returns:
            Configured FrameSampler
        """
        dataset_cfg = config.get('dataset', {})
        sampling_cfg = dataset_cfg.get('sampling', {})
        caps_cfg = dataset_cfg.get('caps', {})
        
        return cls(
            frames_train_val=sampling_cfg.get('frames_train_val', 10),
            frames_test=sampling_cfg.get('frames_test', 20),
            epsilon=sampling_cfg.get('epsilon', 0.1),
            min_gap=sampling_cfg.get('min_jitter_gap', 0.5),
            max_per_video=caps_cfg.get('max_frames_per_video', 20),
            max_per_group=caps_cfg.get('max_frames_per_group', 100),
        )


if __name__ == "__main__":
    # Test uniform timestamp computation
    print("Testing uniform timestamp computation...")
    
    duration = 10.0  # 10 second video
    k = 10
    
    timestamps = compute_uniform_timestamps(duration, k, epsilon=0.1, min_gap=0.5)
    print(f"Duration: {duration}s, K: {k}")
    print(f"Timestamps: {[f'{t:.2f}' for t in timestamps]}")
    print(f"Gaps: {[f'{timestamps[i+1]-timestamps[i]:.2f}' for i in range(len(timestamps)-1)]}")
