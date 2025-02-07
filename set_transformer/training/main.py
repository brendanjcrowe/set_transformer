"""Main script for running Set Transformer training experiments."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.dataset import get_data_loader
from .config import TrainingConfig, get_default_training_configs
from .experiment import run_training_experiments


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(
        description="Run Set Transformer training experiments"
    )

    # Experiment settings
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="pf_reconstruction",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="experiments",
        help="Base directory for experiments",
    )

    # Data settings
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
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

    # Model settings
    parser.add_argument(
        "--num_particles", type=int, default=500, help="Number of particles"
    )
    parser.add_argument(
        "--dim_hidden", type=int, default=128, help="Hidden dimension size"
    )

    # Checkpoint settings
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Get data loaders
    train_loader, val_loader, train_size, val_size = get_data_loader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Get default configs and update with command line arguments
    configs = get_default_training_configs()
    for config in configs:
        config.batch_size = args.batch_size
        config.num_epochs = args.num_epochs
        config.learning_rate = args.learning_rate
        config.weight_decay = args.weight_decay
        config.num_particles = args.num_particles
        config.dim_hidden = args.dim_hidden

    # Run experiments
    run_training_experiments(
        experiment_name=args.experiment_name,
        train_loader=train_loader,
        val_loader=val_loader,
        configs=configs,
        base_dir=args.base_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
