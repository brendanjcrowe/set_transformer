"""Reusable set-autoencoder training loop, with optional latent metric alignment.

Extracted from the MoG encoder study (``experiments/mog/_common.py``) so the benchmark's
pretraining pipeline and that study run the *same* code: an encoder pretrained for the RL
sweep should be trained exactly like the one whose alignment numbers were reported.

The differentiable reconstruction objective is selectable (Chamfer / Sinkhorn), while
model selection always uses exact EMD — the metric the results are reported in — so the
choice of surrogate cannot flatter a run's headline number.

Passing an :class:`AlignConfig` adds the Pearson latent metric-alignment term from
:mod:`set_transformer.latent_alignment`: it correlates each batch's latent pairwise
distances against the corresponding precomputed EMD distances, so that clouds close in
EMD land close in latent space. That requires the loader to yield ``(batch, indices)``
(wrap the dataset in :class:`~set_transformer.data.dataset.IndexedDataset`) and a
precomputed matrix over the *training* split indexed by those same indices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from set_transformer.latent_alignment import (
    LambdaRamp,
    PearsonAlignmentLoss,
    latent_pairwise_distances,
    pearson_r,
)
from set_transformer.loss import (
    ChamferDistanceLoss,
    EarthMoverDistanceLoss,
    SinkhornLoss,
)

#: Sinkhorn blur for the reconstruction objective. Matches
#: :data:`set_transformer.emd_matrix.DEFAULT_BLUR` so the training objective and the
#: alignment target measure the same geometry. (blur=0.5 is too coarse to learn on.)
SINKHORN_BLUR = 0.05

#: The alignment loss correlates over a batch's N(N-1)/2 pairs, so a small batch gives a
#: noisy or degenerate correlation. 16 samples = 120 pairs is the floor.
MIN_ALIGN_BATCH = 16


def build_train_loss(loss_type: str, blur: float = SINKHORN_BLUR) -> nn.Module:
    """Differentiable training objective. Eval is always exact EMD, independent of this."""
    if loss_type == "chamfer":
        return ChamferDistanceLoss(reduction="mean")
    if loss_type == "sinkhorn":
        return SinkhornLoss(p=2, blur=blur, reduction="mean")
    raise ValueError(f"unknown loss_type: {loss_type}")


@dataclass(frozen=True)
class AlignConfig:
    """Latent metric-alignment settings.

    ``lam`` is held at 0 for ``warmup_epochs`` and then ramped linearly over
    ``ramp_epochs``: alignment must not dominate before reconstruction has partially
    converged, or the encoder can settle into a degenerate embedding that satisfies the
    distance correlation while reconstructing nothing. The defaults are the operating
    point validated on the MoG sets (r 0.674 -> 0.993 at no reconstruction cost).
    """

    lam: float = 0.2
    metric: str = "cosine"
    warmup_epochs: int = 15
    ramp_epochs: int = 15

    def schedule(self) -> LambdaRamp:
        return LambdaRamp(self.lam, self.warmup_epochs, self.ramp_epochs)


def split_output(output):
    """Normalize a model's forward() to ``(recon, aux)`` across plain AEs and VAEs."""
    if isinstance(output, dict):
        return output["recon"], {k: v for k, v in output.items() if k != "recon"}
    return output, {}


def forward_with_latent(model, batch, is_vae: bool):
    """``(recon, aux, latent)`` from a single encoder pass.

    For VAEs the latent is the posterior mean, while the decoder still sees the
    reparameterised sample — so adding alignment does not alter the generative path.
    """
    if is_vae:
        recon, aux = split_output(model(batch))
        return recon, aux, aux["mu"]
    z = model.encode(batch)
    return model.decoder(z), {}, z


def encode_all(model, points: torch.Tensor, device: str, batch_size: int = 256) -> torch.Tensor:
    """Latent codes for every row of ``points``, flattened to ``(N, latent_dim)``."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(points), batch_size):
            z = model.encode(points[s:s + batch_size].to(device))
            out.append(z.reshape(z.shape[0], -1))
    return torch.cat(out, dim=0)


def alignment_correlation(model, points: torch.Tensor, emd_pairs: torch.Tensor,
                          metric: str, device: str) -> float:
    """Pearson r between latent and EMD distances over a FIXED held-out pair set.

    ``emd_pairs`` must be the flattened strict upper triangle of the EMD matrix for
    exactly these points, so both vectors share a pair ordering.
    """
    z = encode_all(model, points, device)
    r = pearson_r(latent_pairwise_distances(z, metric), emd_pairs.to(z.device))
    return float("nan") if r is None else float(r)


def evaluate(model, val_loader, recon_loss, emd, device, kl_weight: float = 0.0):
    """Mean (val loss, val EMD) over the val split."""
    model.eval()
    total_loss, total_emd, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            recon, aux = split_output(model(batch))
            loss = recon_loss(recon, batch)
            if "kl" in aux:
                loss = loss + kl_weight * aux["kl"]
            total_loss += loss.item()
            total_emd += emd(recon, batch).item()
            n += 1
    return total_loss / n, total_emd / n


def train_autoencoder(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    num_epochs: int,
    lr: float = 1e-3,
    clip_grad_norm: float = 1.0,
    loss_type: str = "sinkhorn",
    sinkhorn_blur: float = SINKHORN_BLUR,
    is_vae: bool = False,
    kl_weight: float = 1e-3,
    align: Optional[AlignConfig] = None,
    emd_matrix: Optional[np.ndarray] = None,
    val_points: Optional[torch.Tensor] = None,
    val_emd_pairs: Optional[torch.Tensor] = None,
    progress_every: int = 0,
) -> Tuple[dict, Dict[str, np.ndarray], float]:
    """Train ``model``; return ``(best_state_dict, history, best_val_emd)``.

    ``history`` holds per-epoch arrays: ``epoch``, ``train_loss``, ``train_recon``,
    ``val_emd``, plus — when aligning — ``train_align``, ``train_r``, ``val_r`` and
    ``align_lambda``; ``best_epoch`` is a scalar. The returned state dict is the
    best-by-val-EMD snapshot on CPU.

    With ``align`` set, ``train_loader`` must yield ``(batch, indices)`` and ``emd_matrix``
    must cover the training split. ``val_points`` / ``val_emd_pairs`` enable the held-out
    correlation metric and are otherwise optional.
    """
    if align is not None and emd_matrix is None:
        raise ValueError("align requires emd_matrix (the precomputed training EMD matrix)")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-6)
    recon_loss = build_train_loss(loss_type, sinkhorn_blur)
    emd = EarthMoverDistanceLoss()
    align_loss = PearsonAlignmentLoss(align.metric) if align is not None else None
    lam_at = align.schedule() if align is not None else None

    keys = ["epoch", "train_loss", "train_recon", "val_emd"]
    if align is not None:
        keys += ["train_align", "train_r", "val_r", "align_lambda"]
    hist: Dict[str, list] = {k: [] for k in keys}

    best_emd = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(num_epochs):
        model.train()
        lam = lam_at(epoch) if lam_at is not None else 0.0
        running = running_recon = running_align = running_r = 0.0
        n_r = nb = 0

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
                recon, aux = split_output(model(batch))
            else:
                recon, aux, z = forward_with_latent(model, batch, is_vae)

            recon_val = recon_loss(recon, batch)
            loss = recon_val
            if "kl" in aux:
                loss = loss + kl_weight * aux["kl"]

            if align is not None:
                target = torch.from_numpy(np.array(emd_matrix[np.ix_(idx, idx)],
                                                   dtype=np.float32))
                a_loss, r = align_loss(z, target.to(device))
                loss = loss + lam * a_loss
                running_align += float(a_loss.detach())
                if r is not None:
                    running_r += float(r.detach())
                    n_r += 1

            loss.backward()
            if clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            opt.step()
            running += loss.item()
            running_recon += float(recon_val.detach())
            nb += 1
        sched.step()

        _, val_emd = evaluate(model, val_loader, recon_loss, emd, device, kl_weight)
        hist["epoch"].append(epoch)
        hist["train_loss"].append(running / nb)
        hist["train_recon"].append(running_recon / nb)
        hist["val_emd"].append(val_emd)
        if align is not None:
            hist["train_align"].append(running_align / nb)
            hist["train_r"].append(running_r / n_r if n_r else float("nan"))
            hist["align_lambda"].append(lam)
            hist["val_r"].append(
                alignment_correlation(model, val_points, val_emd_pairs, align.metric, device)
                if val_points is not None and val_emd_pairs is not None else float("nan")
            )
        if val_emd < best_emd:
            best_emd = val_emd
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if progress_every and (epoch + 1) % progress_every == 0:
            extra = f" r={hist['val_r'][-1]:.3f} lam={lam:.3f}" if align is not None else ""
            print(f"  epoch {epoch + 1}/{num_epochs}: val_emd {val_emd:.4f}"
                  f" (best {best_emd:.4f}){extra}", flush=True)

    history = {k: np.asarray(v, dtype=np.float32) for k, v in hist.items()}
    history["best_epoch"] = np.asarray(best_epoch, dtype=np.int64)

    # Model selection is by val EMD, which is blind to alignment. If the best epoch lands
    # before the lambda ramp finishes, the "aligned" checkpoint would in fact be a
    # partially- or un-aligned encoder -- a silent mislabel that would quietly void the
    # comparison it exists to support. Loud, because nothing downstream can detect it.
    if align is not None and lam_at(best_epoch) < align.lam:
        print(f"  WARNING: best-by-val-EMD epoch {best_epoch} has lambda="
              f"{lam_at(best_epoch):.3f} < target {align.lam}; this checkpoint is not "
              f"fully aligned. Train longer, lower the alignment weight, or shorten the "
              f"warmup/ramp.", flush=True)

    return best_state, history, best_emd
