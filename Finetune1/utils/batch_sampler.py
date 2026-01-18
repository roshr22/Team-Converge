"""Constrained batch sampler for FF++ dataset (Step 7).

Enforces:
1. Method mixing: each batch contains real + ≥1 fake method
2. Anti-correlation: at most 1 sample per video_id per batch (HARD)
3. Anti-correlation: at most 1 sample per group_id per batch (HARD, fallback to 2)
4. Epoch shuffling under constraints

This prevents the model from learning shortcuts based on batch composition.
"""

import random
from collections import defaultdict, Counter
from typing import List, Dict, Set, Iterator, Optional, Tuple
import logging

import torch
from torch.utils.data import Sampler


logger = logging.getLogger(__name__)


class ConstrainedBatchSampler(Sampler[List[int]]):
    """Batch sampler with method mixing and anti-correlation constraints.
    
    Constructs batches that:
    - Contain samples from multiple manipulation methods (real + fake)
    - Have at most 1 sample per video_id (HARD constraint)
    - Have at most 1 sample per group_id (HARD, relax to 2 as fallback)
    
    Args:
        samples: List of sample dicts or objects with 'method', 'video_id', 'group_id', 'label'
        batch_size: Target batch size
        require_method_mixing: If True, enforce real + fake in each batch
        min_fake_methods: Minimum number of distinct fake methods per batch
        max_per_video: Maximum samples per video_id per batch (HARD)
        max_per_group: Maximum samples per group_id per batch (HARD, fallback +1)
        shuffle: Whether to shuffle each epoch
        drop_last: Whether to drop incomplete last batch
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        samples: List[dict],
        batch_size: int = 24,
        require_method_mixing: bool = True,
        min_fake_methods: int = 1,
        max_per_video: int = 1,
        max_per_group: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.require_method_mixing = require_method_mixing
        self.min_fake_methods = min_fake_methods
        self.max_per_video = max_per_video
        self.max_per_group = max_per_group
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        
        self._epoch = 0
        self._rng = random.Random(seed)
        
        # Build index structures
        self._build_indices()
        
        # Track constraint violations for logging
        self.constraint_violations = {
            'method_mixing': 0,
            'min_fake_methods': 0,
            'video_collision': 0,
            'group_collision': 0,
            'group_relaxed': 0,  # Times we allowed max_per_group + 1
        }
        
        # Detailed tracking for audit
        self._relaxation_log: List[dict] = []
    
    def _build_indices(self):
        """Build index structures for efficient sampling."""
        self.method_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.video_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.group_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.real_indices: List[int] = []
        self.fake_indices: List[int] = []
        self.fake_method_indices: Dict[str, List[int]] = defaultdict(list)
        
        for i, sample in enumerate(self.samples):
            method = self._get_attr(sample, 'method')
            video_id = self._get_attr(sample, 'video_id')
            group_id = self._get_attr(sample, 'group_id')
            label = self._get_attr(sample, 'label')
            
            self.method_to_indices[method].append(i)
            self.video_to_indices[video_id].append(i)
            self.group_to_indices[group_id].append(i)
            
            if label == 0:
                self.real_indices.append(i)
            else:
                self.fake_indices.append(i)
                self.fake_method_indices[method].append(i)
        
        self.all_fake_methods = list(self.fake_method_indices.keys())
    
    def _get_attr(self, sample, key: str):
        """Get attribute from sample (works with dict or object)."""
        if isinstance(sample, dict):
            return sample[key]
        return getattr(sample, key)
    
    def _reset_epoch(self):
        """Reset state for new epoch."""
        self._rng = random.Random(self.seed + self._epoch)
        self.constraint_violations = {k: 0 for k in self.constraint_violations}
        self._relaxation_log = []
    
    def set_epoch(self, epoch: int):
        """Set random seed for this epoch."""
        self._epoch = epoch
    
    def get_relaxation_count(self) -> int:
        """Get number of times constraints were relaxed."""
        return self.constraint_violations.get('group_relaxed', 0)
    
    def get_constraint_violations(self) -> Dict[str, int]:
        """Get all constraint violations."""
        return dict(self.constraint_violations)
    
    def __len__(self) -> int:
        """Return number of batches."""
        if self.drop_last:
            return len(self.samples) // self.batch_size
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches with constraints."""
        self._reset_epoch()
        
        # Shuffle all indices
        all_indices = list(range(len(self.samples)))
        if self.shuffle:
            self._rng.shuffle(all_indices)
        
        # Track which indices have been used
        remaining = set(all_indices)
        batches = []
        
        while len(remaining) >= self.batch_size or (not self.drop_last and remaining):
            batch = self._construct_batch(remaining)
            if not batch:
                break
            batches.append(batch)
            remaining -= set(batch)
        
        # Log constraint violations if any
        total_violations = sum(self.constraint_violations.values())
        if total_violations > 0:
            logger.warning(
                f"Epoch {self._epoch}: {total_violations} constraint events "
                f"{self.constraint_violations}"
            )
        
        return iter(batches)
    
    def _construct_batch(self, remaining: Set[int]) -> List[int]:
        """Construct a single batch respecting constraints.
        
        Strategy:
        1. Start with a random real sample (if method mixing required)
        2. Add samples from different fake methods
        3. Fill remaining slots with any samples, respecting BOTH video_id AND group_id
        4. If still too small, relax group_id to allow max_per_group + 1
        """
        batch: List[int] = []
        batch_video_counts: Dict[str, int] = defaultdict(int)
        batch_group_counts: Dict[str, int] = defaultdict(int)
        batch_methods: Set[str] = set()
        batch_has_real = False
        batch_has_fake = False
        
        # Get candidates in shuffled order
        candidates = list(remaining)
        self._rng.shuffle(candidates)
        
        # Phase 1: Ensure method mixing (real + fake)
        if self.require_method_mixing:
            # First, try to add a real sample
            for idx in candidates:
                if idx in remaining and self._get_attr(self.samples[idx], 'label') == 0:
                    if self._can_add_strict(idx, batch_video_counts, batch_group_counts):
                        self._add_to_batch(idx, batch, batch_video_counts, batch_group_counts, batch_methods)
                        batch_has_real = True
                        break
            
            # Then, try to add samples from different fake methods
            fake_methods_added = 0
            for method in self._rng.sample(self.all_fake_methods, len(self.all_fake_methods)):
                method_indices = self.fake_method_indices[method]
                self._rng.shuffle(method_indices)
                for idx in method_indices:
                    if idx in remaining and idx not in batch:
                        if self._can_add_strict(idx, batch_video_counts, batch_group_counts):
                            self._add_to_batch(idx, batch, batch_video_counts, batch_group_counts, batch_methods)
                            fake_methods_added += 1
                            batch_has_fake = True
                            break
                
                if fake_methods_added >= self.min_fake_methods:
                    break
        
        # Phase 2: Fill remaining slots with STRICT constraints (video + group)
        for idx in candidates:
            if len(batch) >= self.batch_size:
                break
            
            if idx in remaining and idx not in batch:
                if self._can_add_strict(idx, batch_video_counts, batch_group_counts):
                    self._add_to_batch(idx, batch, batch_video_counts, batch_group_counts, batch_methods)
        
        # Phase 3: If batch is still too small, relax group constraint to max_per_group + 1
        if len(batch) < self.batch_size:
            for idx in candidates:
                if len(batch) >= self.batch_size:
                    break
                
                if idx in remaining and idx not in batch:
                    video_id = self._get_attr(self.samples[idx], 'video_id')
                    group_id = self._get_attr(self.samples[idx], 'group_id')
                    
                    # Still enforce video_id strictly
                    if batch_video_counts[video_id] >= self.max_per_video:
                        continue
                    
                    # Allow group_id up to max_per_group + 1
                    if batch_group_counts[group_id] < self.max_per_group + 1:
                        if batch_group_counts[group_id] >= self.max_per_group:
                            # Log relaxation
                            self.constraint_violations['group_relaxed'] += 1
                            self._relaxation_log.append({
                                'batch_idx': len(batch),
                                'group_id': group_id,
                                'current_count': batch_group_counts[group_id],
                            })
                        self._add_to_batch(idx, batch, batch_video_counts, batch_group_counts, batch_methods)
        
        # Log violations
        if self.require_method_mixing:
            if not batch_has_real and batch:
                self.constraint_violations['method_mixing'] += 1
            if not batch_has_fake and batch:
                self.constraint_violations['method_mixing'] += 1
        
        return batch
    
    def _can_add_strict(
        self, 
        idx: int, 
        batch_video_counts: Dict[str, int],
        batch_group_counts: Dict[str, int],
    ) -> bool:
        """Check if sample can be added with STRICT constraints (both video and group)."""
        sample = self.samples[idx]
        video_id = self._get_attr(sample, 'video_id')
        group_id = self._get_attr(sample, 'group_id')
        
        # HARD constraint: video_id
        if batch_video_counts[video_id] >= self.max_per_video:
            return False
        
        # HARD constraint: group_id (strict in Phase 1 and 2)
        if batch_group_counts[group_id] >= self.max_per_group:
            return False
        
        return True
    
    def _add_to_batch(
        self,
        idx: int,
        batch: List[int],
        batch_video_counts: Dict[str, int],
        batch_group_counts: Dict[str, int],
        batch_methods: Set[str],
    ):
        """Add sample to batch and update tracking."""
        sample = self.samples[idx]
        batch.append(idx)
        batch_video_counts[self._get_attr(sample, 'video_id')] += 1
        batch_group_counts[self._get_attr(sample, 'group_id')] += 1
        batch_methods.add(self._get_attr(sample, 'method'))
    
    @classmethod
    def from_dataset(
        cls,
        dataset,
        config: dict,
    ) -> 'ConstrainedBatchSampler':
        """Create sampler from dataset and config.
        
        Args:
            dataset: FFppDataset instance
            config: Full config dict
            
        Returns:
            Configured ConstrainedBatchSampler
        """
        training_cfg = config.get('training', {})
        batch_cfg = config.get('batch_sampling', {})
        
        # Build sample list with required attributes
        samples = []
        for i in range(len(dataset)):
            s = dataset.samples[i]
            samples.append({
                'method': s.method,
                'video_id': s.video_id,
                'group_id': s.group_id,
                'label': s.label,
            })
        
        return cls(
            samples=samples,
            batch_size=training_cfg.get('batch_size', 24),
            require_method_mixing=batch_cfg.get('require_method_mixing', True),
            min_fake_methods=batch_cfg.get('min_fake_methods', 1),
            max_per_video=batch_cfg.get('max_samples_per_video', 1),
            max_per_group=batch_cfg.get('max_samples_per_group', 1),
            shuffle=batch_cfg.get('shuffle_every_epoch', True),
            seed=config.get('seed', 42),
        )


def validate_batch_constraints(
    batch_samples: List[dict],
    require_mixing: bool = True,
    max_per_video: int = 1,
    max_per_group: int = 1,
) -> Tuple[bool, Dict[str, any]]:
    """Validate that a batch meets constraints and return detailed audit info.
    
    Args:
        batch_samples: List of sample dicts with method, video_id, group_id, label
        require_mixing: Whether method mixing is required
        max_per_video: Max samples per video_id
        max_per_group: Max samples per group_id
        
    Returns:
        Tuple of (valid, audit_dict)
    """
    violations = []
    
    # Method histogram
    method_counts = Counter(s.get('method', 'unknown') for s in batch_samples)
    
    # Label distribution
    labels = [s['label'] for s in batch_samples]
    has_real = 0 in labels
    has_fake = 1 in labels
    real_count = labels.count(0)
    fake_count = labels.count(1)
    
    if require_mixing and not (has_real and has_fake):
        violations.append("Missing real or fake samples")
    
    # Video ID analysis
    video_counts = Counter(s['video_id'] for s in batch_samples)
    video_duplicates = {vid: cnt for vid, cnt in video_counts.items() if cnt > max_per_video}
    unique_videos = len(video_counts)
    
    for vid, count in video_duplicates.items():
        violations.append(f"video_id {vid} appears {count} times (max {max_per_video})")
    
    # Group ID analysis
    group_counts = Counter(s['group_id'] for s in batch_samples)
    group_duplicates = {gid: cnt for gid, cnt in group_counts.items() if cnt > max_per_group}
    unique_groups = len(group_counts)
    
    for gid, count in group_duplicates.items():
        violations.append(f"group_id {gid} appears {count} times (max {max_per_group})")
    
    audit = {
        'valid': len(violations) == 0,
        'violations': violations,
        'method_histogram': dict(method_counts),
        'real_count': real_count,
        'fake_count': fake_count,
        'unique_videos': unique_videos,
        'video_duplicates': len(video_duplicates),
        'unique_groups': unique_groups,
        'group_duplicates': len(group_duplicates),
        'batch_size': len(batch_samples),
    }
    
    return len(violations) == 0, audit


def print_batch_audit(audit: dict, batch_idx: int):
    """Print a formatted batch audit report."""
    print(f"\n{'='*60}")
    print(f"BATCH {batch_idx} AUDIT")
    print(f"{'='*60}")
    print(f"Size: {audit['batch_size']}")
    print(f"Real/Fake: {audit['real_count']}/{audit['fake_count']}")
    print(f"Methods: {audit['method_histogram']}")
    print(f"Unique video_ids: {audit['unique_videos']} (duplicates: {audit['video_duplicates']})")
    print(f"Unique group_ids: {audit['unique_groups']} (duplicates: {audit['group_duplicates']})")
    print(f"Valid: {'✓' if audit['valid'] else '✗'}")
    if audit['violations']:
        print(f"Violations: {audit['violations']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Quick test
    print("ConstrainedBatchSampler module loaded successfully!")
    print("Constraints: video_id=HARD, group_id=HARD (fallback +1)")
