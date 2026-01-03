"""Create train/val/test splits from existing dataset."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys


def create_splits(
    dataset_root="dataset",
    output_dir="data/splits",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42,
):
    """
    Create train/val/test splits from existing dataset.

    Current structure:
        dataset/
        ├── train/
        │   ├── real/ (326 images)
        │   └── fake/ (153 images)
        ├── test/
        │   ├── real/ (110 images)
        │   └── fake/ (389 images)
        └── samples/
            └── fake/ (5 images, ignored)

    Strategy:
        - Combine all images from train/ and test/ folders
        - Split into 70% train, 15% val, 15% test
        - Maintain stratified class distribution
        - Save as CSV files (path, label, class)

    Total: 978 images (436 real, 542 fake)
    Expected split:
        - train: ~684 images (~305 real, ~379 fake)
        - val: ~147 images (~65 real, ~82 fake)
        - test: ~147 images (~65 real, ~82 fake)
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {dataset_root}")
    print(f"Output directory: {output_dir}")

    # Collect all images
    all_samples = []
    image_count = {"real": 0, "fake": 0}

    for split_dir in ["train", "test"]:
        for class_name in ["real", "fake"]:
            img_dir = dataset_root / split_dir / class_name
            if not img_dir.exists():
                print(f"⚠ Directory not found: {img_dir}")
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                all_samples.append({
                    "path": str(img_path),
                    "label": 0 if class_name == "real" else 1,
                    "class": class_name,
                })
                image_count[class_name] += 1

    print(f"\n✓ Loaded {len(all_samples)} images")
    print(f"  Real images: {image_count['real']}")
    print(f"  Fake images: {image_count['fake']}")
    print(f"  Real ratio: {image_count['real'] / len(all_samples) * 100:.1f}%")
    print(f"  Fake ratio: {image_count['fake'] / len(all_samples) * 100:.1f}%")

    # Create DataFrame
    df = pd.DataFrame(all_samples)

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"],
        random_state=random_state,
    )

    # Second split: train vs val
    val_test_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_test_ratio,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    # Save splits to CSV
    train_csv = output_dir / "train.csv"
    val_csv = output_dir / "val.csv"
    test_csv = output_dir / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # Print statistics
    print(f"\n✓ Created splits:")
    print(f"  Train: {len(train_df)} images ({train_df['label'].sum()} fake)")
    print(f"  Val:   {len(val_df)} images ({val_df['label'].sum()} fake)")
    print(f"  Test:  {len(test_df)} images ({test_df['label'].sum()} fake)")

    print(f"\n✓ Saved splits:")
    print(f"  {train_csv}")
    print(f"  {val_csv}")
    print(f"  {test_csv}")

    # Verify class balance
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        real_count = len(split_df[split_df["label"] == 0])
        fake_count = len(split_df[split_df["label"] == 1])
        real_pct = real_count / len(split_df) * 100
        fake_pct = fake_count / len(split_df) * 100
        print(f"\n{name.upper()} distribution:")
        print(f"  Real: {real_count} ({real_pct:.1f}%)")
        print(f"  Fake: {fake_count} ({fake_pct:.1f}%)")


if __name__ == "__main__":
    # Run with default parameters
    create_splits()
