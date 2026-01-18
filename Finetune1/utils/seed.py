"""Seed management utilities for reproducible training."""
import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seed for Python, NumPy, PyTorch for reproducibility.
    
    Args:
        seed: Integer seed value
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Make CUDA deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[SEED] Global seed set to {seed}")


def get_worker_init_fn(seed: int):
    """Get worker init function for DataLoader reproducibility.
    
    Usage:
        DataLoader(..., worker_init_fn=get_worker_init_fn(seed))
    """
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return worker_init_fn


def write_runlog(output_dir: str, config: dict) -> str:
    """Write RUNLOG.txt to output directory.
    
    Args:
        output_dir: Directory to write RUNLOG.txt
        config: Full training config dict
        
    Returns:
        Path to RUNLOG.txt
    """
    import datetime
    import platform
    
    os.makedirs(output_dir, exist_ok=True)
    runlog_path = os.path.join(output_dir, "RUNLOG.txt")
    
    with open(runlog_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("TRAINING RUN LOG\n")
        f.write("=" * 60 + "\n\n")
        
        # Timestamp
        f.write(f"Started: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Platform: {platform.system()} {platform.release()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"PyTorch: {torch.__version__}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA Device: {torch.cuda.get_device_name(0)}\n")
        f.write("\n")
        
        # Key parameters
        f.write("-" * 60 + "\n")
        f.write("KEY PARAMETERS\n")
        f.write("-" * 60 + "\n")
        f.write(f"Seed: {config.get('seed', 'NOT SET')}\n")
        f.write(f"Experiment: {config.get('experiment_name', 'unnamed')}\n")
        f.write(f"Batch Size: {config.get('training', {}).get('batch_size', 'N/A')}\n")
        f.write(f"Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}\n")
        f.write(f"Max Epochs: {config.get('training', {}).get('max_epochs', 'N/A')}\n")
        f.write(f"Frozen Layers: {config.get('model', {}).get('freeze_layers', [])}\n")
        f.write("\n")
        
        # Full config dump
        f.write("-" * 60 + "\n")
        f.write("FULL CONFIG\n")
        f.write("-" * 60 + "\n")
        import yaml
        f.write(yaml.dump(config, default_flow_style=False, sort_keys=False))
        
    print(f"[RUNLOG] Written to {runlog_path}")
    return runlog_path


if __name__ == "__main__":
    # Test seed setting
    set_global_seed(42)
    
    # Test random outputs
    print(f"Python random: {random.random()}")
    print(f"NumPy random: {np.random.rand()}")
    print(f"PyTorch random: {torch.rand(1).item()}")
