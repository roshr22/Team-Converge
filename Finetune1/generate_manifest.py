"""Generate sample manifest from video index."""
from pathlib import Path
from utils.config import load_config
from utils.sampling import FrameSampler
from utils.manifest import build_sample_manifest

# Load config
config = load_config()

# Create sampler from config
sampler = FrameSampler.from_config(config)

# Paths
videos_csv = Path('data/index/videos_master.csv')
ffpp_root = Path('data/raw/ffpp/FaceForensics++_C23')
output_dir = Path('artifacts/manifests')

# Build manifest
samples, split_paths = build_sample_manifest(
    videos_csv=videos_csv,
    ffpp_root=ffpp_root,
    output_dir=output_dir,
    sampler=sampler,
    streaming=True,  # Use video_path + timestamp (no stored crops yet)
)

print(f"\nManifest generation complete!")
print(f"Split manifests:")
for split, path in split_paths.items():
    print(f"  {split}: {path}")
