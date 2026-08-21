"""Latent metric-alignment losses.

Goal: make an autoencoder's latent codes carry the *pairwise geometry* of the inputs —
two point clouds that are close in EMD should land close in latent space, and far ones
far apart. Reconstruction alone does not guarantee this: it constrains each code
individually, never the relation between codes.

Phase 1 implements only the **Pearson** objective (see ``latent_alignment_spec.md``):
correlate the batch's latent pairwise distances against the corresponding precomputed
(debiased Sinkhorn) EMD distances, and minimise ``1 - r``.

Pearson's centring/normalisation is load-bearing. It kills two degenerate shortcuts an
un-normalised score would reward: driving every similarity toward a constant to exploit a
nonzero-mean target, and inflating similarity magnitudes to game a raw dot product. It
also makes the term scale-invariant, so its gradient magnitude is decoupled from the
reconstruction loss's units and ``lambda`` stays tunable in a sane range. Do not swap in
an un-normalised elementwise-product variant.

Note what the objective deliberately does *not* pin: absolute latent spread. Pearson (and
the Phase-2 Spearman fallback) constrain only relational structure; reconstruction is
meant to own the latent scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "flatten_upper_triangle",
    "latent_pairwise_distances",
    "pearson_r",
    "PearsonAlignmentLoss",
    "LambdaRamp",
]

LATENT_METRICS = ("cosine", "euclidean")


def flatten_upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
    """Flatten the strict upper triangle of a square matrix to a 1-D vector.

    Both the latent distances and the EMD targets go through this same function, so the
    two vectors are guaranteed to be in the same pair order (the diagonal, which is 0 by
    construction on both sides and would only inflate ``r``, is excluded).
    """
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {tuple(matrix.shape)}")
    n = matrix.shape[0]
    iu = torch.triu_indices(n, n, offset=1, device=matrix.device)
    return matrix[iu[0], iu[1]]


def latent_pairwise_distances(z: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """Pairwise distances between latent codes, as a flat vector over unordered pairs.

    Args:
        z: latent codes, ``(B, ...)``. Everything past the batch dim is flattened, so the
            ``(B, num_encodings, dim_encoder)`` bottleneck this repo uses is handled as a
            single ``B x (num_encodings * dim_encoder)`` vector.
        metric: ``"cosine"`` (1 - cosine similarity) or ``"euclidean"``.

    Returns:
        ``(B * (B - 1) / 2,)`` distances.
    """
    if metric not in LATENT_METRICS:
        raise ValueError(f"metric must be one of {LATENT_METRICS}, got {metric!r}")
    flat = z.reshape(z.shape[0], -1)
    if metric == "cosine":
        normed = F.normalize(flat, dim=1, eps=1e-8)
        dist = 1.0 - normed @ normed.t()
    else:
        dist = torch.cdist(flat, flat, p=2)
    return flatten_upper_triangle(dist)


def pearson_r(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> Optional[torch.Tensor]:
    """Pearson correlation between two 1-D vectors, or ``None`` if either is constant.

    Returning ``None`` rather than a clamped value makes the degenerate case (early
    training, or a batch too small to have spread) explicit at the call site instead of
    silently contributing a meaningless gradient.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.numel() < 2:
        return None
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = a_c.norm() * b_c.norm()
    if not torch.isfinite(denom) or denom.item() < eps:
        return None
    return (a_c * b_c).sum() / denom


@dataclass(frozen=True)
class LambdaRamp:
    """Warmup-then-linear-ramp schedule for the alignment weight.

    Alignment must not dominate before reconstruction has partially converged: an encoder
    that is still producing noise can satisfy the distance correlation with a degenerate
    embedding and never recover. So ``lambda`` is held at 0 for ``warmup_epochs``, then
    ramped linearly to ``target`` over ``ramp_epochs``.
    """

    target: float
    warmup_epochs: int = 0
    ramp_epochs: int = 0

    def __post_init__(self) -> None:
        if self.warmup_epochs < 0 or self.ramp_epochs < 0:
            raise ValueError("warmup_epochs and ramp_epochs must be non-negative")

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs == 0:
            return self.target
        progress = (epoch - self.warmup_epochs + 1) / self.ramp_epochs
        return self.target * min(1.0, progress)


class PearsonAlignmentLoss(nn.Module):
    """``L_align = 1 - r`` between latent pairwise distances and target EMD distances.

    Args:
        metric: latent distance, ``"cosine"`` or ``"euclidean"``.
        eps: variance floor below which the batch's term is skipped.
    """

    def __init__(self, metric: str = "cosine", eps: float = 1e-8) -> None:
        super().__init__()
        if metric not in LATENT_METRICS:
            raise ValueError(f"metric must be one of {LATENT_METRICS}, got {metric!r}")
        self.metric = metric
        self.eps = eps

    def forward(
        self, z: torch.Tensor, d_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            z: latent codes ``(B, ...)``.
            d_target: precomputed EMD distances for the batch, either the flat
                ``(B*(B-1)/2,)`` pair vector or the full ``(B, B)`` submatrix.

        Returns:
            ``(loss, r)``. On a degenerate batch (no variance in either vector) ``r`` is
            ``None`` and ``loss`` is a graph-connected zero, so the caller can add it
            unconditionally without perturbing the gradient.
        """
        d_latent = latent_pairwise_distances(z, self.metric)
        if d_target.dim() == 2:
            d_target = flatten_upper_triangle(d_target)
        d_target = d_target.to(device=d_latent.device, dtype=d_latent.dtype)

        r = pearson_r(d_latent, d_target, eps=self.eps)
        if r is None:
            return d_latent.sum() * 0.0, None
        return 1.0 - r, r

    def extra_repr(self) -> str:
        return f"metric={self.metric}"
