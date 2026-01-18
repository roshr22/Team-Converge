"""Configuration loading utilities."""
import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load training configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolve output_dir with experiment_name
    if "output_dir" in config and "${experiment_name}" in config["output_dir"]:
        config["output_dir"] = config["output_dir"].replace(
            "${experiment_name}", config.get("experiment_name", "unnamed")
        )
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[CONFIG] Saved to {output_path}")


if __name__ == "__main__":
    # Test config loading
    config = load_config()
    print(f"Loaded config with seed={config['seed']}")
    print(f"Experiment: {config['experiment_name']}")
    print(f"Output dir: {config['output_dir']}")
