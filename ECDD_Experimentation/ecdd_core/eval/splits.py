"""Split utilities for Phase 6.

E6.1: source-based split
E6.2: time-based split

We provide a lightweight abstraction that works on metadata tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class SampleMeta:
    id: str
    label: int
    path: str
    source: str  # generator/device/platform/compression bucket
    timestamp: str  # sortable ISO-ish string


def source_based_split(samples: Sequence[SampleMeta], holdout_sources: List[str]) -> Tuple[List[SampleMeta], List[SampleMeta]]:
    train = [s for s in samples if s.source not in holdout_sources]
    test = [s for s in samples if s.source in holdout_sources]
    return train, test


def time_based_split(samples: Sequence[SampleMeta], split_timestamp: str) -> Tuple[List[SampleMeta], List[SampleMeta]]:
    train = [s for s in samples if s.timestamp < split_timestamp]
    test = [s for s in samples if s.timestamp >= split_timestamp]
    return train, test
