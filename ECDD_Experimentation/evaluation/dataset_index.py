"""Dataset index schema for ECDD evaluation.

Provides a schema for indexing datasets with:
- label (real/fake)
- source/generator family (celebv2, diffusion method name, etc.)
- compression level (or proxy bucket like 'high', 'medium', 'low')

This allows evaluation without checking large datasets into git.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime


@dataclass
class SampleEntry:
    """Single sample in the dataset index."""
    
    # Absolute or relative path to the image file
    path: str
    
    # Ground truth label: 0 = real, 1 = fake
    label: int
    
    # Source/generator family (e.g., "celebv2_real", "stable_diffusion", "midjourney")
    source: str
    
    # Compression level bucket: "high" (Q>80), "medium" (Q 50-80), "low" (Q<50), "unknown"
    compression: str = "unknown"
    
    # Optional: estimated JPEG quality (0-100)
    jpeg_quality: Optional[int] = None
    
    # Optional: original dataset split (train/val/test/calib)
    split: Optional[str] = None
    
    # Optional: additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SampleEntry":
        return cls(**d)


@dataclass
class DatasetIndex:
    """Index of a dataset for evaluation.
    
    Stores references to samples without storing the actual images,
    enabling evaluation on large external datasets.
    """
    
    # Unique name for this index
    name: str
    
    # Root directory for resolving relative paths
    root_dir: str
    
    # Version string for tracking index changes
    version: str = "1.0"
    
    # Creation timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # List of samples
    samples: List[SampleEntry] = field(default_factory=list)
    
    # Summary statistics (computed on load/save)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def add_sample(self, sample: SampleEntry) -> None:
        """Add a sample to the index."""
        self.samples.append(sample)
    
    def compute_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the index."""
        if not self.samples:
            self.stats = {"total": 0}
            return self.stats
        
        total = len(self.samples)
        labels = [s.label for s in self.samples]
        sources = [s.source for s in self.samples]
        compressions = [s.compression for s in self.samples]
        
        # Label distribution
        n_real = sum(1 for l in labels if l == 0)
        n_fake = sum(1 for l in labels if l == 1)
        
        # Source distribution
        source_counts: Dict[str, int] = {}
        for src in sources:
            source_counts[src] = source_counts.get(src, 0) + 1
        
        # Compression distribution
        compression_counts: Dict[str, int] = {}
        for comp in compressions:
            compression_counts[comp] = compression_counts.get(comp, 0) + 1
        
        # Source x Compression counts
        source_compression_counts: Dict[str, Dict[str, int]] = {}
        for s in self.samples:
            if s.source not in source_compression_counts:
                source_compression_counts[s.source] = {}
            src_comp = source_compression_counts[s.source]
            src_comp[s.compression] = src_comp.get(s.compression, 0) + 1
        
        self.stats = {
            "total": total,
            "n_real": n_real,
            "n_fake": n_fake,
            "real_ratio": n_real / total if total > 0 else 0,
            "sources": source_counts,
            "compressions": compression_counts,
            "source_compression": source_compression_counts,
        }
        return self.stats
    
    def filter_by_source(self, source: str) -> List[SampleEntry]:
        """Get samples from a specific source."""
        return [s for s in self.samples if s.source == source]
    
    def filter_by_compression(self, compression: str) -> List[SampleEntry]:
        """Get samples with a specific compression level."""
        return [s for s in self.samples if s.compression == compression]
    
    def filter_by_source_and_compression(self, source: str, compression: str) -> List[SampleEntry]:
        """Get samples matching both source and compression."""
        return [s for s in self.samples if s.source == source and s.compression == compression]
    
    def get_unique_sources(self) -> List[str]:
        """Get list of unique sources in the index."""
        return sorted(set(s.source for s in self.samples))
    
    def get_unique_compressions(self) -> List[str]:
        """Get list of unique compression levels in the index."""
        return sorted(set(s.compression for s in self.samples))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert index to dictionary for serialization."""
        self.compute_stats()
        return {
            "name": self.name,
            "root_dir": self.root_dir,
            "version": self.version,
            "created_at": self.created_at,
            "stats": self.stats,
            "samples": [s.to_dict() for s in self.samples],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetIndex":
        """Create index from dictionary."""
        samples = [SampleEntry.from_dict(s) for s in d.get("samples", [])]
        index = cls(
            name=d["name"],
            root_dir=d["root_dir"],
            version=d.get("version", "1.0"),
            created_at=d.get("created_at", datetime.now().isoformat()),
            samples=samples,
            stats=d.get("stats", {}),
        )
        if not index.stats:
            index.compute_stats()
        return index


def save_index(index: DatasetIndex, path: Path) -> None:
    """Save dataset index to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index.to_dict(), f, indent=2)


def load_index(path: Path) -> DatasetIndex:
    """Load dataset index from JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DatasetIndex.from_dict(data)


def create_index_from_directory(
    root_dir: Path,
    name: str,
    label_func=None,
    source_func=None,
    compression_func=None,
    extensions: List[str] = None,
) -> DatasetIndex:
    """Create a dataset index by scanning a directory.
    
    Args:
        root_dir: Root directory to scan
        name: Name for the index
        label_func: Function(path) -> int to determine label (0=real, 1=fake)
        source_func: Function(path) -> str to determine source family
        compression_func: Function(path) -> str to determine compression bucket
        extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png', '.webp'])
    
    Returns:
        DatasetIndex with discovered samples
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".webp"]
    
    root_dir = Path(root_dir)
    index = DatasetIndex(name=name, root_dir=str(root_dir))
    
    for ext in extensions:
        for path in root_dir.rglob(f"*{ext}"):
            rel_path = str(path.relative_to(root_dir))
            
            # Default: label from parent folder name containing 'real' or 'fake'
            if label_func:
                label = label_func(path)
            else:
                label = 0 if "real" in str(path).lower() else 1
            
            # Default: source from immediate parent folder
            if source_func:
                source = source_func(path)
            else:
                source = path.parent.name
            
            # Default: unknown compression
            if compression_func:
                compression = compression_func(path)
            else:
                compression = "unknown"
            
            index.add_sample(SampleEntry(
                path=rel_path,
                label=label,
                source=source,
                compression=compression,
            ))
    
    index.compute_stats()
    return index
