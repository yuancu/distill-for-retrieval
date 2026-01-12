"""Configuration management for distillation training."""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional


class TrainingConfig:
    """Training configuration loaded from YAML file."""

    def __init__(self, config_path: str):
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        self.config_path = Path(config_path)
        self.config_name = self.config_path.stem  # Filename without extension

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ['model', 'phase1', 'phase2', 'training', 'paths']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section '{section}' in config")

    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config['model']

    @property
    def phase1(self) -> Dict[str, Any]:
        """Get Phase 1 training configuration."""
        return self.config['phase1']

    @property
    def phase2(self) -> Dict[str, Any]:
        """Get Phase 2 training configuration."""
        return self.config['phase2']

    @property
    def training(self) -> Dict[str, Any]:
        """Get training control settings."""
        return self.config['training']

    @property
    def paths(self) -> Dict[str, Any]:
        """Get output paths configuration."""
        return self.config['paths']

    def get_model_name(self) -> str:
        """Get model name based on configuration.

        Returns:
            Model name like 'distilled-3584d' or 'distilled-768d-mrl'
        """
        use_projection = self.model['use_projection']
        if use_projection:
            output_dim = self.model['teacher_dim']
            return f"distilled-{output_dim}d"
        else:
            output_dim = self.model['student_dim']
            return f"distilled-{output_dim}d-mrl"

    def get_experiment_name(self) -> str:
        """Get experiment name combining model name and config name.

        Returns:
            Experiment name like 'distilled-3584d-default'
        """
        return f"{self.get_model_name()}-{self.config_name}"

    def get_checkpoint_dir(self) -> Path:
        """Get directory for saving checkpoints.

        Returns:
            Path to checkpoint directory
        """
        output_dir = Path(self.paths['output_dir'])
        return output_dir / self.get_experiment_name()

    def get_artifacts_dir(self) -> Path:
        """Get directory for saving artifacts.

        Returns:
            Path to artifacts directory
        """
        artifacts_dir = Path(self.paths['artifacts_dir'])
        return artifacts_dir / self.get_experiment_name()

    def get_precomputed_embeddings_dir(self, dataset_names: list = None, precision: str = 'fp16', dim: Optional[int] = None) -> Optional[Path]:
        """Get directory for pre-computed embeddings.

        The path format is: {base_dir}/{dataset1}_{dataset2}_..._{precision}_{dim}/
        For example: ./cache/corpus_embedding/msmarco_fp16_768/

        Args:
            dataset_names: List of dataset names (if None, extracts from phase1 config)
            precision: Precision string (default: 'fp16')
            dim: Optional dimension (if specified, adds dimension to folder name)

        Returns:
            Path to embeddings directory or None if precomputed_embeddings_dir not configured
        """
        if 'precomputed_embeddings_dir' not in self.config:
            return None

        base_dir = Path(self.config['precomputed_embeddings_dir'])

        # Extract dataset names from phase1 config if not provided
        if dataset_names is None:
            dataset_names = []
            if 'datasets' in self.phase1:
                for ds_config in self.phase1['datasets']:
                    dataset_names.append(ds_config['name'])

        # Build subdirectory name: dataset1_dataset2_..._precision[_dim]
        if dataset_names:
            subdir_name = '_'.join(dataset_names) + f'_{precision}'
            if dim is not None:
                subdir_name += f'_{dim}'
        else:
            subdir_name = f'default_{precision}'
            if dim is not None:
                subdir_name += f'_{dim}'

        return base_dir / subdir_name

    def save_config_copy(self, dest_dir: Path):
        """Save a copy of the config file to destination directory.

        Args:
            dest_dir: Destination directory
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"config_{self.config_name}.yaml"
        shutil.copy2(self.config_path, dest_file)

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary.

        Returns:
            Complete configuration dictionary
        """
        return self.config.copy()

    def __repr__(self) -> str:
        """String representation."""
        return f"TrainingConfig(config_name='{self.config_name}', experiment='{self.get_experiment_name()}')"


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load training configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        TrainingConfig object
    """
    if config_path is None:
        # Use default config
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"

    return TrainingConfig(config_path)


def list_available_configs() -> list:
    """List all available config files.

    Returns:
        List of config file paths
    """
    configs_dir = Path(__file__).parent.parent / "configs"
    if not configs_dir.exists():
        return []

    return sorted(configs_dir.glob("*.yaml"))
