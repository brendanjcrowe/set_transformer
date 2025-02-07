"""Training configuration for Set Transformer experiments.

This module defines the configuration classes and defaults for training experiments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Model parameters
    num_particles: int = 500
    dim_particles: int = 4
    num_encodings: int = 8
    dim_encoder: int = 2
    num_inds: int = 32
    dim_hidden: int = 128
    num_heads: int = 4
    use_layer_norm: bool = True

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    weight_decay: float = 0.0
    clip_grad_norm: float = 1.0

    # Scheduler parameters
    scheduler_type: str = "cosine"  # ["cosine", "step", "none"]
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Loss parameters
    loss_type: str = "hausdorff"  # ["hausdorff", "chamfer", "sinkhorn"]
    sinkhorn_blur: float = 0.5
    sinkhorn_scaling: float = 0.5

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Logging and checkpointing
    log_freq: int = 100  # Steps between logging
    eval_freq: int = 1000  # Steps between evaluation
    save_freq: int = 5000  # Steps between saving checkpoints
    keep_last_n_checkpoints: int = 5


@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""

    # Experiment identification
    experiment_name: str
    run_name: str

    # Directory structure
    base_dir: Path = Path("experiments")

    @property
    def experiment_dir(self) -> Path:
        """Get the experiment directory."""
        return self.base_dir / self.experiment_name

    @property
    def run_dir(self) -> Path:
        """Get the run directory."""
        return self.experiment_dir / self.run_name

    @property
    def checkpoint_dir(self) -> Path:
        """Get the checkpoint directory."""
        return self.run_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        """Get the log directory."""
        return self.run_dir / "logs"

    def create_directories(self) -> None:
        """Create all necessary directories."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


def get_default_experiment_config(
    experiment_name: str, run_name: str, base_dir: Optional[Union[str, Path]] = None
) -> ExperimentConfig:
    """Get default experiment configuration.

    Args:
        experiment_name: Name of the experiment
        run_name: Name of the specific run
        base_dir: Base directory for experiments (optional)

    Returns:
        ExperimentConfig: Default experiment configuration
    """
    config = ExperimentConfig(experiment_name=experiment_name, run_name=run_name)

    if base_dir is not None:
        config.base_dir = Path(base_dir)

    return config


def get_default_training_configs() -> List[TrainingConfig]:
    """Get default training configurations for different loss functions.

    Returns:
        List[TrainingConfig]: List of training configurations
    """
    configs = []

    # EMD Loss configuration
    emd_config = TrainingConfig(loss_type="emd", batch_size=32, learning_rate=1e-3)
    configs.append(emd_config)

    # Chamfer Loss configuration
    chamfer_config = TrainingConfig(
        loss_type="chamfer", batch_size=32, learning_rate=1e-3
    )
    configs.append(chamfer_config)

    # Sinkhorn Loss configuration
    sinkhorn_config = TrainingConfig(
        loss_type="sinkhorn",
        batch_size=32,
        learning_rate=1e-3,
        sinkhorn_blur=0.5,
        sinkhorn_scaling=0.5,
    )
    configs.append(sinkhorn_config)

    return configs
