"""Training configuration for Set Transformer experiments.

This module defines the configuration classes and defaults for training experiments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import torch


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    # Model selection
    model_type: Literal[
        "pf_st", "set_vae", "set_vqvae", "ds_ae", "ds_vae", "ds_vqvae",
        "cgf_arm_ae"
    ] = "pf_st"

    # Model parameters
    num_particles: int = 500
    dim_particles: int = 4
    num_encodings: int = 8
    dim_encoder: int = 2
    num_inds: int = 32
    dim_hidden: int = 128
    num_heads: int = 4
    use_layer_norm: bool = True
    # SAB blocks between the PMA and the output Linear of the encoder (model_type "pf_st";
    # SetTransformer.num_post_sab). Any count >= 0; 2 is the geometry every checkpoint
    # written before 2026-09-13 has, when the field did not exist and the constructor
    # default applied. Recorded in the checkpoint so the RL extractor builds the same shape.
    num_post_sab: int = 2

    # Weighted particle sets (option B: mass in the measure, not the metric).
    # `dim_particles` always means the COORDINATE dimension D of a particle.
    # With weighted_particles=True the encoder additionally reads a mass
    # channel, so its input is D+1 while the decoder still reconstructs D
    # coordinates, and the reconstruction loss compares the weighted target
    # measure against the uniform reconstruction. Requires model_type "pf_st"
    # and a loss that accepts weights (sinkhorn / hausdorff / emd).
    weighted_particles: bool = False

    # VAE-specific (ignored unless model_type == "set_vae")
    kl_weight: float = 1.0

    # VQ-VAE-specific (ignored unless model_type == "set_vqvae")
    codebook_size: int = 64
    commitment_weight: float = 0.25
    ema_decay: float = 0.99

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

    # Loss parameters.
    # Sinkhorn is the default: it is the only differentiable loss here that
    # accepts weighted measures, and unlike Chamfer it is an actual metric
    # between point distributions. ("hausdorff" was the old default and is
    # broken upstream in geomloss, which needs a kernel name it is never given.)
    # Trainable choices: "sinkhorn" (weighted or not), "chamfer" (unweighted
    # only). "emd" is the eval metric and has no gradient; "hausdorff" is
    # broken upstream. Trainer._setup_loss refuses both.
    loss_type: str = "sinkhorn"  # ["sinkhorn", "chamfer"]
    # Blur is in COORDINATE UNITS: the length scale below which the loss stops
    # telling points apart. Set it well under the smallest belief structure you
    # need resolved (e.g. a den of radius 0.4 needs blur << 0.4).
    sinkhorn_blur: float = 0.05
    sinkhorn_scaling: float = 0.5

    # Latent metric alignment (set_transformer.latent_alignment). Adds
    # align_lambda * (1 - pearson_r) between the batch's latent pairwise
    # (cosine) distances and the corresponding entries of a precomputed
    # pairwise debiased-Sinkhorn matrix over the dataset (emd_matrix.py --
    # weighted when the dataset is). 0.0 = off (the default; nothing else in
    # this block is read then). lambda is held at 0 for align_warmup_epochs and
    # ramped linearly over align_ramp_epochs: alignment must not dominate
    # before reconstruction has partially converged. The collaborator's MoG
    # operating point was 0.2 / 15 / 15 over 60 epochs; our encoders converge
    # (Ant-Tag, ~10 epochs) or collapse (Odd-Even, ~30) far sooner, so the
    # schedule is a per-domain hyperparameter, not a constant.
    # Requires the loaders to be built with get_data_loader(indexed=True).
    # Model selection (best_val_loss) stays on the reconstruction loss alone,
    # blind to alignment; val/align_r is logged separately.
    align_lambda: float = 0.0
    align_metric: str = "cosine"  # ["cosine", "euclidean"]
    align_warmup_epochs: int = 0
    align_ramp_epochs: int = 0
    emd_matrix_path: Optional[str] = None
    # Held-out latent<->EMD correlation is computed over at most this many
    # val rows (pairs grow quadratically: 2000 rows = 2M pairs).
    align_val_max_samples: int = 2000

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Reproducibility. None keeps the process's global RNG state (legacy).
    # Trainer seeds torch/numpy from this before building the model; the
    # train/val split is seeded separately by get_data_loader(seed=...).
    # Stored in every checkpoint via the pickled config.
    seed: Optional[int] = None

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

    # (No EMD configuration: EMD is the evaluation metric and is not
    # differentiable, so a Trainer built on it raises at _setup_loss.)

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
        sinkhorn_blur=0.05,
        sinkhorn_scaling=0.5,
    )
    configs.append(sinkhorn_config)

    return configs
