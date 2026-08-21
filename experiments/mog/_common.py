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

from set_transformer.latent_alignment import (
    LambdaRamp,
    PearsonAlignmentLoss,
    flatten_upper_triangle,
    latent_pairwise_distances,
    pearson_r,
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


def load_emd_matrix(path: Path, n: int, mmap: bool = False) -> np.ndarray:
    """Load a precomputed pairwise-EMD matrix written by ``8_precompute_emd_matrix.py``.

    Stored as a raw float32 memmap of shape (n, n) — ``n`` comes from the point split so
    the two cannot silently disagree.
    """
    arr = np.memmap(path, dtype=np.float32, mode="r", shape=(n, n))
    return arr if mmap else np.asarray(arr)


def emd_pairs_from_matrix(matrix: np.ndarray) -> torch.Tensor:
    """Flatten a square EMD matrix to the strict-upper-triangle pair vector."""
    return flatten_upper_triangle(torch.from_numpy(np.array(matrix, dtype=np.float32)))


@dataclass(frozen=True)
class AlignConfig:
    """Latent metric-alignment settings (see ``latent_alignment_spec.md``).

    ``lam`` is held at 0 for ``warmup_epochs`` and then ramped linearly over
    ``ramp_epochs``: alignment must not dominate before reconstruction has partially
    converged, or the encoder can settle into a degenerate embedding that satisfies the
    distance correlation while reconstructing nothing.
    """

    lam: float = 0.1
    metric: str = "cosine"
    warmup_epochs: int = 15
    ramp_epochs: int = 15

    def schedule(self) -> LambdaRamp:
        return LambdaRamp(self.lam, self.warmup_epochs, self.ramp_epochs)


def build_model(method: str) -> nn.Module:
    return METHOD_REGISTRY[method].builder(**ARCH)


def _split_output(output):
    if isinstance(output, dict):
        return output["recon"], {k: v for k, v in output.items() if k != "recon"}
    return output, {}


def forward_with_latent(model, batch, is_vae: bool):
    """Return ``(recon, aux, latent)`` from a single encoder pass.

    For the VAEs the latent is the posterior mean (the decoder still sees the
    reparameterised sample, exactly as in the unaligned runs).
    """
    if is_vae:
        recon, aux = _split_output(model(batch))
        return recon, aux, aux["mu"]
    z = model.encode(batch)
    return model.decoder(z), {}, z


def encode_all(model, points: torch.Tensor, device: str, is_vae: bool,
               batch_size: int = 256) -> torch.Tensor:
    """Latent codes for every row of ``points``, flattened to ``(N, latent_dim)``."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(points), batch_size):
            z = model.encode(points[s:s + batch_size].to(device))
            out.append(z.reshape(z.shape[0], -1))
    return torch.cat(out, dim=0)


def alignment_correlation(model, points: torch.Tensor, emd_pairs: torch.Tensor,
                          metric: str, device: str, is_vae: bool) -> float:
    """Pearson r between latent and EMD distances over a FIXED held-out pair set.

    ``emd_pairs`` must already be the flattened strict upper triangle of the EMD matrix
    for exactly these points, so both vectors share a pair ordering.
    """
    z = encode_all(model, points, device, is_vae)
    d_latent = latent_pairwise_distances(z, metric)
    r = pearson_r(d_latent, emd_pairs.to(d_latent.device))
    return float("nan") if r is None else float(r)


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

    ``history`` holds per-epoch arrays: ``epoch``, ``train_loss``, ``train_recon``,
    ``val_emd`` and — when aligning — ``train_align``, ``train_r``, ``val_r``,
    ``align_lambda``. The returned state dict is the best-by-val-EMD snapshot
    (deep-copied to CPU). ``loss_type`` selects the differentiable training objective
    (chamfer|sinkhorn); model selection and the reported ``val_emd`` always use exact EMD.

    Passing ``align`` adds the Pearson latent metric-alignment term. It requires
    ``train_loader`` to yield ``(batch, indices)`` (wrap the dataset in ``IndexedDataset``)
    and ``emd_matrix`` to be the precomputed pairwise EMD matrix over the *training* split,
    indexed by those same indices. ``val_points`` / ``val_emd_pairs`` enable the held-out
    correlation metric and are otherwise optional.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = METHOD_REGISTRY[method]
    model = build_model(method).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)
    chamfer = build_train_loss(loss_type)
    emd = EarthMoverDistanceLoss()

    align_loss = PearsonAlignmentLoss(align.metric) if align is not None else None
    lam_at = align.schedule() if align is not None else None
    if align is not None and emd_matrix is None:
        raise ValueError("align requires emd_matrix (the precomputed training EMD matrix)")

    keys = ["epoch", "train_loss", "train_recon", "val_emd"]
    if align is not None:
        keys += ["train_align", "train_r", "val_r", "align_lambda"]
    hist: Dict[str, list] = {k: [] for k in keys}
    best_emd = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(num_epochs):
        model.train()
        lam = lam_at(epoch) if lam_at is not None else 0.0
        running, running_recon, running_align, running_r, n_r, nb = 0.0, 0.0, 0.0, 0.0, 0, 0
        for item in train_loader:
            idx = None
            if align is not None:
                batch, idx = item
                idx = idx.numpy()
            else:
                batch = item
            batch = batch.to(device)
            opt.zero_grad()

            if align is None:
                recon, aux = _split_output(model(batch))
            else:
                recon, aux, z = forward_with_latent(model, batch, spec.is_vae)

            recon_loss = chamfer(recon, batch)
            loss = recon_loss
            if "kl" in aux:
                loss = loss + KL_WEIGHT * aux["kl"]

            if align is not None:
                target = torch.from_numpy(np.ascontiguousarray(emd_matrix[np.ix_(idx, idx)]))
                a_loss, r = align_loss(z, target.to(device))
                loss = loss + lam * a_loss
                running_align += float(a_loss.detach())
                if r is not None:
                    running_r += float(r.detach())
                    n_r += 1

            loss.backward()
            if CLIP_GRAD_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            opt.step()
            running += loss.item()
            running_recon += float(recon_loss.detach())
            nb += 1
        sched.step()

        _, val_emd = evaluate(model, val_loader, chamfer, emd, device)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(running / nb)
        hist["train_recon"].append(running_recon / nb)
        hist["val_emd"].append(val_emd)
        if align is not None:
            hist["train_align"].append(running_align / nb)
            hist["train_r"].append(running_r / n_r if n_r else float("nan"))
            hist["align_lambda"].append(lam)
            hist["val_r"].append(
                alignment_correlation(model, val_points, val_emd_pairs, align.metric,
                                      device, spec.is_vae)
                if val_points is not None and val_emd_pairs is not None else float("nan")
            )
        if val_emd < best_emd:
            best_emd = val_emd
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history = {k: np.asarray(v, dtype=np.float32) for k, v in hist.items()}
    return best_state, history, best_emd
