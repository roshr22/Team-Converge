"""Training utilities package."""
from .seed import set_global_seed, get_worker_init_fn, write_runlog
from .config import load_config, save_config
from .splitting import deterministic_split, VideoInfo
from .inventory import create_ffpp_inventory, verify_inventory, FileInfo
from .indexing import (
    build_master_index,
    load_master_index,
    validate_master_index,
    VideoRecord,
)
from .sampling import (
    FrameSampler,
    compute_uniform_timestamps,
    sample_video_frames,
    FrameSample,
)
from .face_extraction import (
    FaceExtractor,
    FaceDetector,
    FaceCrop,
    BoundingBox,
    generate_sample_id,
)
from .manifest import (
    build_sample_manifest,
    validate_manifest,
    load_samples_csv,
    save_samples_csv,
    SampleRecord,
)
from .dataset import FFppDataset
from .batch_sampler import ConstrainedBatchSampler, validate_batch_constraints
from .augmentations import DeploymentRealismAugmentation, ValidationTransform
from .staged_training import (
    FinetuneStage,
    StageConfig,
    LayerFreezer,
    CheckpointManager,
    CalibrationLogger,
)

__all__ = [
    "set_global_seed",
    "write_runlog",
    "load_config",
    "save_config",
    "deterministic_split",
    "VideoRecord",
    "FrameSampler",
    "FaceExtractor",
    "FaceDetector",
    "build_sample_manifest",
    "validate_manifest",
    "load_samples_csv",
    "SampleRecord",
    "FFppDataset",
    "ConstrainedBatchSampler",
    "DeploymentRealismAugmentation",
    "ValidationTransform",
    "LayerFreezer",
    "CheckpointManager",
    "CalibrationLogger",
    "FinetuneStage",
    "StageConfig",
]
