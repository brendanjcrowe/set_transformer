"""RETIRED 2026-09-06. Do not use. Superseded by experiments/ant_tag/3_train_st.py.

Kept as history only; nothing imports it and no checkpoint any RL run consumed
was produced by it (checked across both experiment directories and the
collaborator's exp/PaperResults branch). It is unsafe on the datasets this
repo now produces, for three independent reasons (domain_mds/PITFALLS.md
section 8, item 4):

1. HALF-WEIGHTED OBJECTIVE. It builds the loaders with the default
   load_weights=True, so a weighted .npz yields (particles, weights) and the
   Trainer scores the reconstruction against the WEIGHTED target measure --
   but it never sets TrainingConfig.weighted_particles, so the encoder gets
   bare coordinates and cannot see which particles carry the mass. On the
   Odd-Even exact-support filter every belief has the same 50 positions and
   only the weights differ, so the encoder sees one input and is asked for 50
   different targets. Training runs without error, and the checkpoint's
   config.weighted_particles=False then tells the RL side to drop the weight
   channel too, so nothing downstream detects it. With --loss_type chamfer
   the weighted path raises; with sinkhorn it trains quietly.
2. SET GEOMETRY FROM CLI DEFAULTS. --num_particles / --dim_particles default
   to 500 / 4 instead of being read from the dataset. A wrong value is silent:
   Sinkhorn compares sets of different sizes without complaint and the ISAB
   encoder accepts any set size (PITFALLS.md section 4).
3. NO FRAME RECORD. It does not record particle_scale / particle_centre in
   the checkpoint, so the RL-side frame check (st.py) cannot refuse a
   mismatched encoder.

experiments/ant_tag/3_train_st.py does all three correctly (weights and set
geometry from the dataset, contradicting flags refused, frame recorded), is
env-generic (it reads a .npz and never touches an env), and adds the latent
metric-alignment option. Use it for every domain:

    cd set_transformer/experiments/ant_tag
    WANDB_MODE=offline python3 3_train_st.py \
        --data_path ../odd_even/data/oe50_short_pf_dataset.npz \
        --num_encodings 8 --dim_encoder 8 --sinkhorn_blur 0.02 --seed 0

The code below is unchanged from the last live version and still runs.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from ..data.dataset import get_data_loader
from .config import TrainingConfig, get_default_training_configs
from .experiment import run_training_experiments


def get_loss_choices() -> List[str]:
    """Trainable loss functions.

    "emd" (the eval metric, no gradient) and "hausdorff" (broken upstream in
    geomloss) used to be listed here, and "emd" was the default: every run
    crashed on its first backward() and the runner still printed "All
    experiments completed!".
    """
    return ["chamfer", "sinkhorn"]


def create_experiment_name(base_name: str, loss_type: str) -> str:
    """Create formatted experiment name.

    Args:
        base_name: Base experiment name
        loss_type: Type of loss function

    Returns:
        str: Formatted experiment name as <base_name>_<loss_type>_YYYY-MM-DD_HH:mm:ss
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return f"{base_name}_{loss_type}_{timestamp}"


def main() -> None:
    """Run training experiments."""
    # Set multiprocessing start method to 'spawn' for CUDA support
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Train Set Transformer models")

    # Experiment settings
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="experiment",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiments",
        help="Base directory for experiments",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/numpy/data.npy",
        help="Path to the data file",
    )

    # Data settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )

    # Training settings
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seeds model init, batch order and the train/val split.",
    )

    # Loss function settings
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=get_loss_choices(),
        default="sinkhorn",
        help="Type of loss function to use",
    )
    parser.add_argument(
        "--sinkhorn_blur",
        type=float,
        default=0.05,
        help="Blur parameter for Sinkhorn loss",
    )
    parser.add_argument(
        "--sinkhorn_scaling",
        type=float,
        default=0.5,
        help="Scaling parameter for Sinkhorn loss",
    )

    # Model selection
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["pf_st", "set_vae", "set_vqvae", "ds_ae", "ds_vae", "ds_vqvae"],
        default="pf_st",
        help="Which model variant to train",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help="Weight on KL term (SetVAE only)",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=64,
        help="Number of codes in VQ-VAE codebook (SetVQVAE only)",
    )
    parser.add_argument(
        "--commitment_weight",
        type=float,
        default=0.25,
        help="Commitment loss weight (SetVQVAE only)",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
        help="EMA decay for VQ codebook updates (SetVQVAE only)",
    )

    # Model architecture settings
    parser.add_argument(
        "--num_particles", type=int, default=500, help="Number of particles"
    )
    parser.add_argument(
        "--dim_particles", type=int, default=4, help="Dimension of each particle"
    )
    parser.add_argument(
        "--num_encodings",
        type=int,
        default=8,
        help="Number of encodings from set transformer",
    )
    parser.add_argument(
        "--dim_encoder", type=int, default=2, help="Dimension of each encoding"
    )
    parser.add_argument(
        "--num_inds", type=int, default=32, help="Number of inducing points"
    )
    parser.add_argument(
        "--dim_hidden", type=int, default=128, help="Hidden dimension size"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--use_layer_norm",
        type=bool,
        default=True,
        help="Whether to use layer normalization",
    )

    # Learning rate scheduler settings
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["cosine", "step", "none"],
        default="cosine",
        help="Type of learning rate scheduler",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-6, help="Minimum learning rate for scheduler"
    )

    # Logging and checkpointing settings
    parser.add_argument(
        "--log_freq", type=int, default=100, help="Steps between logging"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=1000, help="Steps between evaluation"
    )
    parser.add_argument(
        "--save_freq", type=int, default=5000, help="Steps between saving checkpoints"
    )
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=5,
        help="Number of checkpoints to keep",
    )

    # Checkpoint settings
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Create formatted experiment name
    experiment_name = create_experiment_name(args.experiment_name, args.loss_type)

    # Get data loaders
    train_loader, val_loader, train_size, val_size = get_data_loader(
        batch_size=args.batch_size,
        data_path=args.data_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Create a single config from command line arguments
    config = TrainingConfig(
        # Model selection
        model_type=args.model_type,
        kl_weight=args.kl_weight,
        codebook_size=args.codebook_size,
        commitment_weight=args.commitment_weight,
        ema_decay=args.ema_decay,
        # Model parameters
        num_particles=args.num_particles,
        dim_particles=args.dim_particles,
        num_encodings=args.num_encodings,
        dim_encoder=args.dim_encoder,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        use_layer_norm=args.use_layer_norm,
        # Training parameters
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        # Scheduler parameters
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        # Loss parameters
        loss_type=args.loss_type,
        sinkhorn_blur=args.sinkhorn_blur,
        sinkhorn_scaling=args.sinkhorn_scaling,
        seed=args.seed,
        # Logging parameters
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
    )

    # Run experiments with single config
    run_training_experiments(
        experiment_name=experiment_name,  # Use the formatted experiment name
        train_loader=train_loader,
        val_loader=val_loader,
        configs=[config],  # Pass as single-item list
        base_dir=args.base_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
