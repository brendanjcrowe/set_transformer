"""Experiment runner for Set Transformer training.

This module provides functionality for running multiple training experiments
with different configurations.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig, TrainingConfig, get_default_training_configs
from .trainer import Trainer


class ExperimentRunner:
    """Runner for multiple training experiments."""

    def __init__(
        self,
        experiment_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        base_dir: Optional[Path] = None,
    ):
        """Initialize experiment runner.

        Args:
            experiment_name: Name of the experiment
            train_loader: Training data loader
            val_loader: Validation data loader
            base_dir: Base directory for experiments (optional)
        """
        self.experiment_name = experiment_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.base_dir = Path(base_dir) if base_dir else Path("experiments")

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(logging.INFO)

        # Create experiment directory if it doesn't exist
        exp_dir = self.base_dir / self.experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        fh = logging.FileHandler(
            exp_dir / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def run_experiments(
        self,
        configs: Optional[List[TrainingConfig]] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """Run multiple training experiments.

        Args:
            configs: List of training configurations. If None, uses default configs.
            resume_from: Path to checkpoint to resume from (optional)
        """
        if configs is None:
            configs = get_default_training_configs()

        self.logger.info(f"Starting experiments for {self.experiment_name}")
        self.logger.info(f"Number of configurations: {len(configs)}")

        for i, config in enumerate(configs):
            # Create run name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{config.loss_type}_{timestamp}"

            # Create experiment config
            exp_config = ExperimentConfig(
                experiment_name=self.experiment_name,
                run_name=run_name,
                base_dir=self.base_dir,
            )

            self.logger.info(f"\nStarting run {i+1}/{len(configs)}")
            self.logger.info(f"Run name: {run_name}")
            self.logger.info(f"Loss type: {config.loss_type}")

            # Initialize trainer
            trainer = Trainer(
                training_config=config,
                experiment_config=exp_config,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                logger=self.logger,
            )

            # Resume from checkpoint if specified
            if resume_from:
                trainer.load_checkpoint(Path(resume_from))
                self.logger.info(f"Resumed from checkpoint: {resume_from}")

            # Train
            try:
                trainer.train()
            except Exception as e:
                self.logger.error(f"Error during training: {str(e)}")
                continue

            self.logger.info(f"Completed run: {run_name}\n")

        self.logger.info("All experiments completed!")


def run_training_experiments(
    experiment_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    configs: Optional[List[TrainingConfig]] = None,
    base_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> None:
    """Run multiple training experiments.

    This is the main entry point for running experiments.

    Args:
        experiment_name: Name of the experiment
        train_loader: Training data loader
        val_loader: Validation data loader
        configs: List of training configurations (optional)
        base_dir: Base directory for experiments (optional)
        resume_from: Path to checkpoint to resume from (optional)
    """
    runner = ExperimentRunner(
        experiment_name=experiment_name,
        train_loader=train_loader,
        val_loader=val_loader,
        base_dir=Path(base_dir) if base_dir else None,
    )

    runner.run_experiments(configs=configs, resume_from=resume_from)
