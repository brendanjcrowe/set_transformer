"""Shared config, model registry, and training loop for the MoG encoder study.

Four methods — {Set Transformer, DeepSet} x {AE, VAE} — trained on random
mixture-of-Gaussians point sets. Every method shares the identical encoder shape,
decoder, Chamfer training loss, and cosine schedule; only the encoder architecture and
bottleneck (deterministic vs. variational) differ, so any gap is attributable to those.

The training loop here is purpose-built (rather than reusing ``training.trainer.Trainer``)
because the deliverable is a per-epoch validation-EMD *learning curve* over seeds, which
the step-based Trainer only exposes through tensorboard/wandb.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from set_transformer.emd_matrix import load_matrix
from set_transformer.latent_alignment import flatten_upper_triangle
from set_transformer.training.autoencoder import (
    AlignConfig,
    alignment_correlation,
    encode_all,
    train_autoencoder,
)
from set_transformer.loss import (
    ChamferDistanceLoss,
    EarthMoverDistanceLoss,
    SinkhornLoss,
)
from set_transformer.models import DeepSetAE, DeepSetVAE, PFSetTransformer, SetVAE

DATA_DIR = Path("experiments/mog/data")
RUNS_DIR = Path("experiments/mog/runs")
FIG_DIR = Path("experiments/mog/figures")

# Architecture shared by all four methods (matches the synthetic_sets study).
ARCH = dict(
    num_particles=100,
    dim_particles=2,
    num_encodings=8,
    dim_encoder=16,
    num_inds=32,
    dim_hidden=128,
    num_heads=4,
    ln=True,
)

# Training knobs shared by all runs.
NUM_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
CLIP_GRAD_NORM = 1.0
KL_WEIGHT = 1e-3  # VAE only; same operating point as the synthetic_sets sweep.
SINKHORN_BLUR = 0.05  # geomloss default; blur=0.5 is too coarse to learn 2D [-3,3] data.


def build_train_loss(loss_type: str) -> nn.Module:
    """Differentiable training objective. Eval is always exact EMD, independent of this."""
    if loss_type == "chamfer":
        return ChamferDistanceLoss(reduction="mean")
    if loss_type == "sinkhorn":
        return SinkhornLoss(p=2, blur=SINKHORN_BLUR, reduction="mean")
    raise ValueError(f"unknown loss_type: {loss_type}")


@dataclass(frozen=True)
class MethodSpec:
    label: str          # display label, e.g. "ST-VAE"
    model_type: str     # for the checkpoint config / builder
    builder: Callable[..., nn.Module]
    is_vae: bool


METHOD_REGISTRY: Dict[str, MethodSpec] = {
    "st_ae": MethodSpec("ST-AE", "pf_st", PFSetTransformer, False),
    "st_vae": MethodSpec("ST-VAE", "set_vae", SetVAE, True),
    "ds_ae": MethodSpec("DS-AE", "ds_ae", DeepSetAE, False),
    "ds_vae": MethodSpec("DS-VAE", "ds_vae", DeepSetVAE, True),
}

METHOD_ORDER: List[str] = ["st_ae", "st_vae", "ds_ae", "ds_vae"]
METHOD_COLORS: Dict[str, str] = {
    "st_ae": "tab:red",
    "st_vae": "tab:orange",
    "ds_ae": "tab:blue",
    "ds_vae": "tab:cyan",
}


#: Backwards-compatible alias; the implementation is shared with the benchmark pipeline.
load_emd_matrix = load_matrix


def emd_pairs_from_matrix(matrix: np.ndarray) -> torch.Tensor:
    """Flatten a square EMD matrix to the strict-upper-triangle pair vector."""
    return flatten_upper_triangle(torch.from_numpy(np.array(matrix, dtype=np.float32)))


def build_model(method: str) -> nn.Module:
    return METHOD_REGISTRY[method].builder(**ARCH)


def evaluate(model, val_loader, chamfer, emd, device) -> Tuple[float, float]:
    """Return (mean val Chamfer+KL loss, mean val EMD) over the val split."""
    model.eval()
    total_loss, total_emd, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, aux = _split_output(model(batch))
            loss = chamfer(recon, batch)
            if "kl" in aux:
                loss = loss + KL_WEIGHT * aux["kl"]
            total_loss += loss.item()
            total_emd += emd(recon, batch).item()
            n += 1
    return total_loss / n, total_emd / n


def train_one(
    method: str,
    train_loader,
    val_loader,
    seed: int,
    device: str,
    num_epochs: int = NUM_EPOCHS,
    loss_type: str = "chamfer",
    align: Optional[AlignConfig] = None,
    emd_matrix: Optional[np.ndarray] = None,
    val_points: Optional[torch.Tensor] = None,
    val_emd_pairs: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, Dict[str, np.ndarray], float]:
    """Train one (method, seed). Returns (best_model_state, history, best_val_emd).

    Thin wrapper over :func:`set_transformer.training.autoencoder.train_autoencoder`,
    which is shared with the benchmark's pretraining pipeline so an encoder pretrained
    for the RL sweep is trained by exactly the code whose numbers are reported here.
    This function only supplies the study's method registry, seeding and hyperparameters.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = METHOD_REGISTRY[method]
    model = build_model(method).to(device)
    return train_autoencoder(
        model, train_loader, val_loader, device, num_epochs,
        lr=LEARNING_RATE, clip_grad_norm=CLIP_GRAD_NORM,
        loss_type=loss_type, sinkhorn_blur=SINKHORN_BLUR,
        is_vae=spec.is_vae, kl_weight=KL_WEIGHT,
        align=align, emd_matrix=emd_matrix,
        val_points=val_points, val_emd_pairs=val_emd_pairs,
    )
