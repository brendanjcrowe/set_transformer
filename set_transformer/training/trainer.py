"""Trainer class for Set Transformer experiments.

This module provides the Trainer class which handles model training, evaluation,
logging, and checkpointing.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

# Configure matplotlib to use 'Agg' backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import wandb

from ..loss import (
    ChamferDistanceLoss,
    EarthMoverDistanceLoss,
    HausdorffLoss,
    SinkhornLoss,
)
from ..models import (
    DeepSetAE,
    DeepSetVAE,
    DeepSetVQVAE,
    PFSetTransformer,
    SetVAE,
    SetVQVAE,
)
from ..plots import visualize_particle_filter_reconstruction
from .config import ExperimentConfig, TrainingConfig
from ..data.dataset import IndexedDataset, POMDPDataset
from ..emd_matrix import load_matrix
from ..latent_alignment import (
    LambdaRamp,
    PearsonAlignmentLoss,
    flatten_upper_triangle,
    latent_pairwise_distances,
    pearson_r,
)


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

        # Seed before the model is built so its init is reproducible and
        # recorded (the config, seed included, goes into every checkpoint).
        seed = getattr(self.config, "seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))

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
        self.best_epoch = 0

        # Latent metric alignment (off unless config.align_lambda > 0).
        self._setup_alignment()

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
        common = dict(
            num_particles=self.config.num_particles,
            dim_particles=self.config.dim_particles,
            num_encodings=self.config.num_encodings,
            dim_encoder=self.config.dim_encoder,
            num_inds=self.config.num_inds,
            dim_hidden=self.config.dim_hidden,
            num_heads=self.config.num_heads,
            ln=self.config.use_layer_norm,
        )
        if self.config.weighted_particles:
            if self.config.model_type != "pf_st":
                raise ValueError(
                    "weighted_particles=True is implemented for "
                    "model_type='pf_st' only; the other architectures tie the "
                    f"decoder output dim to the encoder input dim (got "
                    f"model_type={self.config.model_type!r})"
                )
            # Encoder reads [coords..., mass]; decoder still emits coords only.
            common["dim_particles"] = self.config.dim_particles + 1
            common["dim_output_particles"] = self.config.dim_particles
        if self.config.model_type == "pf_st":
            model = PFSetTransformer(**common)
        elif self.config.model_type == "set_vae":
            model = SetVAE(**common)
        elif self.config.model_type == "set_vqvae":
            model = SetVQVAE(
                codebook_size=self.config.codebook_size,
                commitment_weight=self.config.commitment_weight,
                ema_decay=self.config.ema_decay,
                **common,
            )
        elif self.config.model_type == "ds_ae":
            model = DeepSetAE(**common)
        elif self.config.model_type == "ds_vae":
            model = DeepSetVAE(**common)
        elif self.config.model_type == "ds_vqvae":
            model = DeepSetVQVAE(
                codebook_size=self.config.codebook_size,
                commitment_weight=self.config.commitment_weight,
                ema_decay=self.config.ema_decay,
                **common,
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")
        return model.to(self.config.device)

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
        if self.config.loss_type == "emd":
            # EarthMoverDistanceLoss goes through POT in numpy and returns a
            # tensor with no grad_fn; loss.backward() raises on the first
            # step. It is the EVAL metric (self.eval_loss), never trainable.
            raise ValueError(
                "loss_type='emd' is not differentiable and cannot be trained "
                "on (it is the evaluation metric). Use 'sinkhorn' (weighted "
                "or unweighted) or 'chamfer' (unweighted only)."
            )
        elif self.config.loss_type == "hausdorff":
            # geomloss raises KeyError: None inside its kernel table for this
            # loss; it has never trained here. Refuse up front rather than
            # crash after the data has loaded.
            raise ValueError(
                "loss_type='hausdorff' is broken upstream in geomloss "
                "(KeyError: None). Use 'sinkhorn'."
            )
        elif self.config.loss_type == "chamfer":
            return ChamferDistanceLoss()
        elif self.config.loss_type == "sinkhorn":
            # sinkhorn_scaling used to be collected on two CLIs, stored in the
            # config and written into every checkpoint without ever reaching
            # geomloss. It is forwarded now.
            return SinkhornLoss(
                blur=self.config.sinkhorn_blur,
                scaling=self.config.sinkhorn_scaling,
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
            "best_epoch": self.best_epoch,
            "config": self.config,
            # The coordinate frame the encoder was trained in, so the RL
            # side can check it applies the same (x - centre) / scale.
            "particle_scale": getattr(self._base_dataset, "particle_scale", None),
            "particle_centre": getattr(self._base_dataset, "particle_centre", None),
            "alignment": self._alignment_record(),
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
        # save_checkpoint pickles the TrainingConfig dataclass under "config";
        # torch >= 2.6 defaults to weights_only=True and refuses it.
        checkpoint = torch.load(
            checkpoint_path, map_location=self.config.device, weights_only=False)

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]

    # ------------------------------------------------------------------
    # Latent metric alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_dataset(dataset):
        """(innermost dataset, indexed?) through Subset / IndexedDataset layers."""
        indexed = False
        while True:
            if isinstance(dataset, IndexedDataset):
                indexed = True
            if isinstance(dataset, POMDPDataset):
                break
            inner = getattr(dataset, "dataset", None)
            if inner is None:
                break
            dataset = inner
        return dataset, indexed

    def _setup_alignment(self) -> None:
        """Wire the alignment term up from config, or leave it off.

        On: loads the pairwise-EMD matrix (memmap; its size must match the
        base dataset exactly), builds the Pearson loss and the lambda ramp,
        and fixes a held-out set of val rows for val/align_r.
        """
        base, indexed = self._unwrap_dataset(self.train_loader.dataset)
        self._base_dataset = base
        self._indexed_loader = indexed
        self.align_loss = None
        self.emd_matrix = None
        self._lambda_ramp = None
        self._val_align_indices = None
        self._val_align_pairs = None

        lam = float(getattr(self.config, "align_lambda", 0.0) or 0.0)
        if lam <= 0.0:
            return
        path = getattr(self.config, "emd_matrix_path", None)
        if not path:
            raise ValueError(
                "align_lambda > 0 requires emd_matrix_path (the pairwise EMD "
                "matrix over the dataset; see 2b_precompute_emd.py)")
        if not indexed:
            raise ValueError(
                "align_lambda > 0 requires the loaders to carry dataset row "
                "indices: build them with get_data_loader(indexed=True)")
        if not isinstance(base, POMDPDataset):
            raise ValueError(
                f"alignment needs a POMDPDataset at the base of the loader, "
                f"got {type(base).__name__}")
        n = len(base)
        self.emd_matrix = load_matrix(Path(path), n, mmap=True)
        self.align_loss = PearsonAlignmentLoss(self.config.align_metric)
        self._lambda_ramp = LambdaRamp(
            lam, self.config.align_warmup_epochs, self.config.align_ramp_epochs)

        val_dataset = self.val_loader.dataset
        val_idx = np.asarray(getattr(val_dataset, "indices", np.arange(n)), dtype=np.int64)
        val_idx = np.sort(val_idx)[: int(self.config.align_val_max_samples)]
        if len(val_idx) >= 2:
            sub = np.array(self.emd_matrix[np.ix_(val_idx, val_idx)], dtype=np.float32)
            self._val_align_indices = val_idx
            self._val_align_pairs = flatten_upper_triangle(torch.from_numpy(sub))
        self.logger.info(
            f"Latent alignment ON: lambda={lam} ({self.config.align_metric}), "
            f"warmup {self.config.align_warmup_epochs} ep, ramp "
            f"{self.config.align_ramp_epochs} ep, matrix {path} over {n} rows, "
            f"val_r over {len(val_idx)} held-out rows")

    def _align_lambda(self, epoch: int) -> float:
        return 0.0 if self._lambda_ramp is None else float(self._lambda_ramp(epoch))

    def _alignment_record(self) -> Optional[dict]:
        """What a checkpoint records about alignment (None when off)."""
        if self.align_loss is None:
            return None
        return {
            "lambda": float(self.config.align_lambda),
            "metric": self.config.align_metric,
            "warmup_epochs": int(self.config.align_warmup_epochs),
            "ramp_epochs": int(self.config.align_ramp_epochs),
            "emd_matrix_path": str(self.config.emd_matrix_path),
            "lambda_at_best_epoch": self._align_lambda(self.best_epoch),
        }

    def _forward_with_latent(self, model_input, need_latent: bool):
        """``(recon, aux, latent)`` from one encoder pass.

        Without alignment this is plain ``model(x)`` (latent None). With it,
        the autoencoders decode an explicit ``encode()`` -- identical to
        ``forward`` for PFSetTransformer / DeepSetAE -- and the VAEs expose
        their posterior mean while the decoder still sees the sample. The
        VQ-VAEs quantize the code, so aligning their pre-quantization latent
        would optimize something the decoder never sees; refused.
        """
        if not need_latent:
            recon, aux = self._split_output(self.model(model_input))
            return recon, aux, None
        if self.config.model_type in ("pf_st", "ds_ae"):
            latent = self.model.encode(model_input)
            return self.model.decoder(latent), {}, latent
        if self.config.model_type in ("set_vae", "ds_vae"):
            recon, aux = self._split_output(self.model(model_input))
            return recon, aux, aux["mu"]
        raise ValueError(
            f"latent alignment is not supported for model_type="
            f"{self.config.model_type!r}")

    def _validation_alignment_r(self) -> float:
        """Pearson r between latent and EMD distances over the fixed val rows."""
        if self._val_align_indices is None:
            return float("nan")
        self.model.eval()
        codes = []
        base = self._base_dataset
        with torch.no_grad():
            for s in range(0, len(self._val_align_indices), 256):
                rows = self._val_align_indices[s:s + 256]
                samples = [base[int(i)] for i in rows]
                if base.is_weighted:
                    particles = torch.stack([p for p, _ in samples])
                    weights = torch.stack([w for _, w in samples])
                else:
                    particles, weights = torch.stack(samples), None
                particles = particles.to(self.config.device)
                weights = None if weights is None else weights.to(self.config.device)
                _, _, z = self._forward_with_latent(
                    self._model_input(particles, weights), need_latent=True)
                codes.append(z.reshape(z.shape[0], -1))
        z = torch.cat(codes, dim=0)
        r = pearson_r(latent_pairwise_distances(z, self.config.align_metric),
                      self._val_align_pairs.to(z.device))
        return float("nan") if r is None else float(r)

    def _split_batch(self, batch):
        """Return (particles, weights, indices) on the training device.

        The loader yields a bare tensor (unweighted), a (particles, weights)
        tuple (weighted), or -- when built with get_data_loader(indexed=True)
        for the alignment term -- either of those wrapped as (sample, idx).
        `indices` are BASE dataset rows (IndexedDataset wraps the base dataset
        before the split), which is how the EMD matrix is indexed; None when
        the loader is not indexed. `weights` is None for unweighted sets.
        """
        indices = None
        sample = batch
        if self._indexed_loader:
            sample, indices = batch
            indices = np.asarray(indices.cpu().numpy(), dtype=np.int64)
        if isinstance(sample, (tuple, list)):
            particles, weights = sample
            return (particles.to(self.config.device),
                    weights.to(self.config.device), indices)
        return sample.to(self.config.device), None, indices

    def _model_input(self, particles, weights):
        """Build the encoder input.

        In weighted mode the mass is appended as one extra input channel,
        scaled by the particle count so a uniform belief feeds 1.0 rather than
        1/N. That keeps the channel on the same order as normalized
        coordinates. The mass is an INPUT to the encoder only; it never appears
        in the reconstruction target, where it acts as the measure's weights.
        """
        if weights is None or not self.config.weighted_particles:
            return particles
        # Sanitize exactly as the RL feature extractor does, so the encoder
        # sees the same channel at pretraining time and at RL time.
        clean = torch.clamp(
            torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0), min=0.0
        )
        clean = clean / (clean.sum(dim=-1, keepdim=True) + 1e-8)
        scaled = (clean * clean.shape[-1]).unsqueeze(-1)
        return torch.cat([particles, scaled], dim=-1)

    def _split_output(self, output):
        """Return (recon, aux_dict) regardless of whether the model returned a
        plain tensor or a dict of components."""
        if isinstance(output, dict):
            recon = output["recon"]
            aux = {k: v for k, v in output.items() if k != "recon"}
        else:
            recon, aux = output, {}
        return recon, aux

    def _compose_loss(self, recon, target, aux, target_weights=None,
                      latent=None, batch_indices=None, align_lambda=0.0):
        """Compose total loss from reconstruction + weighted auxiliary terms.

        `target_weights` turns the reconstruction term into a comparison of
        MEASURES: the weighted empirical belief sum_i w_i delta(x_i) against
        the uniform reconstruction. It is passed straight through to the loss,
        so a loss with no weighted formulation (Chamfer) raises rather than
        quietly optimizing something else.

        `latent` + `batch_indices` (+ `align_lambda`) add the latent metric-
        alignment term: 1 - pearson_r between the batch's latent pairwise
        distances and the matching entries of the precomputed EMD matrix.
        The term is added even at lambda 0 during warmup so its value is
        logged; only its weight is 0. evaluate() passes no latent, so the
        VALIDATION loss stays reconstruction-only and model selection is
        blind to alignment (val/align_r is reported separately).

        Returns (total_loss, components_dict) where components is per-term
        scalar floats for logging.
        """
        if target_weights is None:
            recon_loss = self.train_loss(recon, target)
        else:
            recon_loss = self.train_loss(
                recon, target, None, target_weights
            )
        total = recon_loss
        components = {"recon": recon_loss.item()}
        if (self.align_loss is not None and latent is not None
                and batch_indices is not None):
            sub = np.array(self.emd_matrix[np.ix_(batch_indices, batch_indices)],
                           dtype=np.float32)
            align_term, r = self.align_loss(latent, torch.from_numpy(sub).to(latent.device))
            total = total + align_lambda * align_term
            components["align"] = float(align_term.detach())
            components["align_lambda"] = float(align_lambda)
            if r is not None:
                components["align_r"] = float(r.detach())
        if "kl" in aux:
            total = total + self.config.kl_weight * aux["kl"]
            components["kl"] = aux["kl"].item()
        if "commitment_loss" in aux:
            total = total + self.config.commitment_weight * aux["commitment_loss"]
            components["commitment_loss"] = aux["commitment_loss"].item()
        if "perplexity" in aux:
            components["perplexity"] = aux["perplexity"].item()
        return total, components

    def train_epoch(self) -> float:
        """Train for one epoch.

        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        align_lambda = self._align_lambda(self.current_epoch)
        for batch in self.train_loader:
            particles, weights, indices = self._split_batch(batch)

            # Forward pass
            self.optimizer.zero_grad()
            recon, aux, latent = self._forward_with_latent(
                self._model_input(particles, weights),
                need_latent=self.align_loss is not None)
            loss, components = self._compose_loss(
                recon, particles, aux, target_weights=weights,
                latent=latent, batch_indices=indices, align_lambda=align_lambda,
            )

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
                # Calculate running averages
                running_loss = total_loss / num_batches

                # Log to tensorboard
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar(
                    "train/running_loss", running_loss, self.global_step
                )

                # Log to wandb with more metrics
                wandb_payload = {
                    "train/step_loss": loss.item(),
                    "train/running_loss": running_loss,
                    "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                    "training/step": self.global_step,
                    "training/epoch": self.current_epoch,
                }
                for name, value in components.items():
                    wandb_payload[f"train/{name}"] = value
                    self.writer.add_scalar(f"train/{name}", value, self.global_step)
                wandb.log(wandb_payload, step=self.global_step)

            # Evaluation
            if self.global_step % self.config.eval_freq == 0:
                val_loss, val_metrics = self.evaluate()

                # Log to tensorboard
                self.writer.add_scalar("val/loss", val_loss, self.global_step)

                # Log to wandb with evaluation metrics
                if "align_r" in val_metrics:
                    self.writer.add_scalar("val/align_r", val_metrics["align_r"], self.global_step)
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/earth_mover_distance": val_metrics["emd"],
                        **({"val/align_r": val_metrics["align_r"]}
                           if "align_r" in val_metrics else {}),
                        "val/step": self.global_step,
                        "val/epoch": self.current_epoch,
                    },
                    step=self.global_step,
                )

                # Save visualization
                # Coordinates only: the plot compares reconstructed
                # points against the input points, not their mass.
                self._save_visualization(particles, recon)

                # Save checkpoint if best
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = self.current_epoch
                    self.save_checkpoint(is_best=True)
                    wandb.log(
                        {"val/best_loss": val_loss},
                        step=self.global_step,
                    )

            # Regular checkpoint saving
            if self.global_step % self.config.save_freq == 0:
                self.save_checkpoint()

        epoch_loss = total_loss / num_batches
        return epoch_loss

    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model.

        Returns:
            Tuple[float, Dict[str, float]]: Average validation loss and metrics dictionary
        """
        self.model.eval()
        total_loss = 0
        total_emd = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                particles, weights, _indices = self._split_batch(batch)
                output = self.model(self._model_input(particles, weights))
                recon, aux = self._split_output(output)

                # Calculate validation loss (using training loss)
                loss, _ = self._compose_loss(
                    recon, particles, aux, target_weights=weights
                )
                total_loss += loss.item()

                # Calculate Earth Mover Distance on reconstruction (always).
                # EMD is weight-aware, so the metric stays comparable with the
                # training objective in weighted mode.
                emd = self.eval_loss(recon, particles, None, weights)
                total_emd += emd.item()

                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_emd = total_emd / num_batches

        metrics = {
            "emd": avg_emd,
        }
        if self.align_loss is not None:
            metrics["align_r"] = self._validation_alignment_r()

        return avg_loss, metrics

    def _save_visualization(
        self, input_batch: torch.Tensor, output_batch: torch.Tensor
    ) -> None:
        """Save visualization of reconstruction.

        Args:
            input_batch: Input particle set
            output_batch: Reconstructed particle set
        """
        # Take first example from batch and detach from computation graph
        input_particles = input_batch[0].detach().cpu().numpy()
        output_particles = output_batch[0].detach().cpu().numpy()

        # The 4D visualization requires (N, 4) particles; for other dims
        # fall back to a simple 2D scatter comparison.
        dim = input_particles.shape[-1]
        if dim == 4:
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
                2, 3, figsize=(18, 10)
            )
            visualize_particle_filter_reconstruction(
                input_particles,
                output_particles,
                ax=(ax1, ax2, ax3, ax4, ax5, ax6),
                title=f"Reconstruction (Step {self.global_step})",
            )
        else:
            # Generic 2-column scatter for arbitrary dim (plot first 2 dims)
            fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(12, 5))
            d0, d1 = 0, min(1, dim - 1)
            ax_in.scatter(
                input_particles[:, d0], input_particles[:, d1], s=4, alpha=0.6
            )
            ax_in.set_title("Input")
            ax_out.scatter(
                output_particles[:, d0], output_particles[:, d1], s=4, alpha=0.6
            )
            ax_out.set_title("Reconstruction")
            fig.suptitle(f"Step {self.global_step}")

        # Log to tensorboard and wandb
        self.writer.add_figure("reconstruction", fig, self.global_step)
        wandb.log({"reconstruction": wandb.Image(fig)}, step=self.global_step)

        # Save to file
        fig.savefig(self.exp_config.log_dir / f"reconstruction_{self.global_step}.png")

        # Close the figure to free memory
        plt.close(fig)

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
            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "epoch": epoch,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                },
                step=self.global_step,
            )

        # Model selection is by reconstruction val loss, which is blind to
        # alignment. If the best epoch predates the end of the lambda ramp,
        # checkpoint_best.pt is a partially- or un-aligned encoder wearing an
        # "aligned" label -- nothing downstream can detect that, so say it.
        if self.align_loss is not None:
            lam_best = self._align_lambda(self.best_epoch)
            if lam_best < float(self.config.align_lambda):
                self.logger.warning(
                    f"best-by-val-loss epoch {self.best_epoch} has align_lambda="
                    f"{lam_best:.3f} < target {self.config.align_lambda}; "
                    "checkpoint_best.pt is NOT fully aligned. Train longer, lower "
                    "the alignment weight, or shorten the warmup/ramp.")

        # Always leave a checkpoint behind. Checkpoints are otherwise written
        # only at save_freq / eval_freq boundaries, so a run with fewer total
        # steps than save_freq (the default is 5000) reported "Training
        # completed!" and exited having saved NOTHING -- and the next pipeline
        # stage takes a checkpoint path as input.
        self.save_checkpoint()
        self.logger.info(
            f"Final checkpoint saved to {self.exp_config.checkpoint_dir}")

        self.logger.info("Training completed!")
        wandb.finish()
