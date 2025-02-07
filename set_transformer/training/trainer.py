"""Trainer class for Set Transformer experiments.

This module provides the Trainer class which handles model training, evaluation,
logging, and checkpointing.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..loss import ChamferDistanceLoss, HausdorffDistanceLoss, SinkhornLoss, EarthMoverDistanceLoss
from ..models import PFSetTransformer
from ..plots import visualize_particle_filter_reconstruction
from .config import ExperimentConfig, TrainingConfig


class Trainer:
    """Trainer class for Set Transformer experiments."""

    def __init__(
        self,
        training_config: TrainingConfig,
        experiment_config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: Optional[logging.Logger] = None,
    ):
        """Initialize trainer.

        Args:
            training_config: Training configuration
            experiment_config: Experiment configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            logger: Logger instance (optional)
        """
        self.config = training_config
        self.exp_config = experiment_config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Create directories
        self.exp_config.create_directories()

        # Setup logging
        self.logger = logger or self._setup_logger()
        self.writer = SummaryWriter(self.exp_config.log_dir)

        # Initialize model, optimizer, scheduler, and losses
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.train_loss = self._setup_loss()
        self.eval_loss = EarthMoverDistanceLoss()  # Always use EMD for evaluation

        # Initialize tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Initialize wandb
        self._setup_wandb()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(self.exp_config.run_name)
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.exp_config.log_dir / "training.log")
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

    def _setup_model(self) -> nn.Module:
        """Setup model.

        Returns:
            nn.Module: Initialized model
        """
        model = PFSetTransformer(
            num_particles=self.config.num_particles,
            dim_particles=self.config.dim_particles,
            num_encodings=self.config.num_encodings,
            dim_encoder=self.config.dim_encoder,
            num_inds=self.config.num_inds,
            dim_hidden=self.config.dim_hidden,
            num_heads=self.config.num_heads,
            ln=self.config.use_layer_norm,
        ).to(self.config.device)

        return model

    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer.

        Returns:
            optim.Optimizer: Initialized optimizer
        """
        return optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler.

        Returns:
            Optional[optim.lr_scheduler._LRScheduler]: Initialized scheduler
        """
        if self.config.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.num_epochs, eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        return None

    def _setup_loss(self) -> nn.Module:
        """Setup loss function.

        Returns:
            nn.Module: Initialized loss function
        """
        if self.config.loss_type == "hausdorff":
            return HausdorffDistanceLoss()
        elif self.config.loss_type == "chamfer":
            return ChamferDistanceLoss()
        elif self.config.loss_type == "sinkhorn":
            return SinkhornLoss(
                blur=self.config.sinkhorn_blur, scaling=self.config.sinkhorn_scaling
            )
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def _setup_wandb(self) -> None:
        """Setup Weights & Biases logging."""
        wandb.init(
            project="set-transformer",
            name=self.exp_config.run_name,
            config={
                "training": self.config.__dict__,
                "experiment": {
                    "name": self.exp_config.experiment_name,
                    "run": self.exp_config.run_name,
                },
            },
        )

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }

        # Save latest checkpoint
        latest_path = self.exp_config.checkpoint_dir / f"checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save numbered checkpoint
        numbered_path = (
            self.exp_config.checkpoint_dir / f"checkpoint_{self.global_step}.pt"
        )
        torch.save(checkpoint, numbered_path)

        # Save best checkpoint
        if is_best:
            best_path = self.exp_config.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)

        # Remove old checkpoints if needed
        self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last n."""
        checkpoints = sorted(
            [f for f in self.exp_config.checkpoint_dir.glob("checkpoint_[0-9]*.pt")],
            key=lambda x: int(x.stem.split("_")[1]),
        )

        if len(checkpoints) > self.config.keep_last_n_checkpoints:
            for checkpoint in checkpoints[: -self.config.keep_last_n_checkpoints]:
                checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.config.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.train_loss(output, batch)

            # Backward pass
            loss.backward()
            if self.config.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip_grad_norm
                )
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_freq == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.global_step,
                )

            # Evaluation
            if self.global_step % self.config.eval_freq == 0:
                val_loss = self.evaluate()
                self.writer.add_scalar("val/loss", val_loss, self.global_step)
                wandb.log({"val/loss": val_loss}, step=self.global_step)

                # Save visualization
                self._save_visualization(batch, output)

                # Save checkpoint if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)

            # Regular checkpoint saving
            if self.global_step % self.config.save_freq == 0:
                self.save_checkpoint()

        epoch_loss = total_loss / num_batches
        return epoch_loss

    def evaluate(self) -> float:
        """Evaluate the model.

        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.config.device)
                output = self.model(batch)
                loss = self.eval_loss(output, batch)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _save_visualization(
        self, input_batch: torch.Tensor, output_batch: torch.Tensor
    ) -> None:
        """Save visualization of reconstruction.

        Args:
            input_batch: Input particle set
            output_batch: Reconstructed particle set
        """
        # Take first example from batch
        input_particles = input_batch[0].cpu().numpy()
        output_particles = output_batch[0].cpu().numpy()

        # Create visualization
        fig = visualize_particle_filter_reconstruction(
            input_particles,
            output_particles,
            title=f"Reconstruction (Step {self.global_step})",
        )

        # Log to tensorboard and wandb
        self.writer.add_figure("reconstruction", fig, self.global_step)
        wandb.log({"reconstruction": wandb.Image(fig)}, step=self.global_step)

        # Save to file
        fig.savefig(self.exp_config.log_dir / f"reconstruction_{self.global_step}.png")

    def train(self) -> None:
        """Train the model for the specified number of epochs."""
        self.logger.info("Starting training...")
        self.logger.info(f"Training config: {self.config}")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            epoch_loss = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch}/{self.config.num_epochs} - " f"Loss: {epoch_loss:.4f}"
            )

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log epoch metrics
            self.writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
            wandb.log({"train/epoch_loss": epoch_loss, "epoch": epoch})

        self.logger.info("Training completed!")
        wandb.finish()
