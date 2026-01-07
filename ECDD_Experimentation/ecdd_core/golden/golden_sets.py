"""Golden sets definitions.

The experimentation spec requires a golden dataset:
- 30 face images (balanced real/fake)
- 20 out-of-scope (OOD) images
- 30 edge cases (EXIF, alpha, low JPEG, small faces, multi-face, blur drift, color profiles)

This module defines a manifest format and helper utilities.

NOTE: We do not create/modify the actual images here; we point at existing
ECDD_Experiment_Data assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional


GoldenSplit = Literal["faces", "ood", "edge_cases"]


@dataclass(frozen=True)
class GoldenItem:
    id: str
    split: GoldenSplit
    path: Path
    label: Optional[int]  # 0 real, 1 fake, None for OOD/edge-cases if unknown


def default_golden_manifest(base_dir: Path) -> List[GoldenItem]:
    """Create a default manifest from the current ECDD_Experiment_Data folder.

    This is a starting point. Teams should curate exact counts/labels.
    """
    items: List[GoldenItem] = []

    # OOD
    ood_dir = base_dir / "ood"
    if ood_dir.exists():
        for p in sorted(ood_dir.glob("*.png")):
            items.append(GoldenItem(id=p.stem, split="ood", path=p, label=None))

    # Edge cases
    ec_dir = base_dir / "edge_cases"
    if ec_dir.exists():
        for p in sorted(list(ec_dir.glob("*.png")) + list(ec_dir.glob("*.jpg")) + list(ec_dir.glob("*.webp"))):
            items.append(GoldenItem(id=p.stem, split="edge_cases", path=p, label=None))

    # Faces (real/fake)
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"
    if real_dir.exists():
        for p in sorted(list(real_dir.glob("*.jpg")) + list(real_dir.glob("*.png"))):
            items.append(GoldenItem(id=f"real_{p.stem}", split="faces", path=p, label=0))
    if fake_dir.exists():
        for p in sorted(list(fake_dir.glob("*.jpg")) + list(fake_dir.glob("*.png"))):
            items.append(GoldenItem(id=f"fake_{p.stem}", split="faces", path=p, label=1))

    return items
