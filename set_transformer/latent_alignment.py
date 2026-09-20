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

Since 2026-09-19 the target distances can also be computed ONLINE, per batch, instead of
read from a precomputed matrix (the block "Online targets" below): the divergence between
two clouds is a function of the inputs alone, so nothing about the loss changes, only where
the numbers come from and how many pairs are used (:func:`sample_pairs`). The matrix path
is untouched and stays the default of the reconstruction objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "flatten_upper_triangle",
    "latent_pairwise_distances",
    "pearson_r",
    "PearsonAlignmentLoss",
    "LambdaRamp",
    # online targets (2026-09-19)
    "PAIRS_ALL",
    "DEFAULT_PAIRS",
    "DEFAULT_VAL_PAIRS",
    "parse_pairs",
    "all_pairs",
    "sample_pairs",
    "latent_pair_distances",
    "sinkhorn_pair_targets",
    "OnlineAlignment",
    "add_alignment_arguments",
    "resolve_sinkhorn_blur",
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
        self, z: torch.Tensor, d_target: torch.Tensor,
        pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Args:
            z: latent codes ``(B, ...)``.
            d_target: precomputed EMD distances for the batch, either the flat
                ``(B*(B-1)/2,)`` pair vector or the full ``(B, B)`` submatrix; with
                ``pairs``, the ``(P,)`` target for exactly those pairs.
            pairs: ``(a, b)`` index vectors (2026-09-19, online targets): correlate over
                these pairs only, the latent distances computed for them by
                :func:`latent_pair_distances`. ``None`` = every pair, as before.

        Returns:
            ``(loss, r)``. On a degenerate batch (no variance in either vector) ``r`` is
            ``None`` and ``loss`` is a graph-connected zero, so the caller can add it
            unconditionally without perturbing the gradient.
        """
        if pairs is None:
            d_latent = latent_pairwise_distances(z, self.metric)
            if d_target.dim() == 2:
                d_target = flatten_upper_triangle(d_target)
        else:
            d_latent = latent_pair_distances(z, pairs, self.metric)
            if d_target.dim() != 1:
                raise ValueError(f"with pairs, d_target must be a flat (P,) vector, "
                                 f"got shape {tuple(d_target.shape)}")
        d_target = d_target.to(device=d_latent.device, dtype=d_latent.dtype)

        r = pearson_r(d_latent, d_target, eps=self.eps)
        if r is None:
            return d_latent.sum() * 0.0, None
        return 1.0 - r, r

    def extra_repr(self) -> str:
        return f"metric={self.metric}"


# ---------------------------------------------------------------------------
# Online targets (2026-09-19): the same divergence, computed per batch
# ---------------------------------------------------------------------------
#
# The target distance between two clouds is a function of the INPUTS alone, so it can be
# computed at batch time under no_grad instead of read from a precomputed matrix. That
# removes the matrix's O(N^2) build and 4 N^2 bytes, the sidecar checks, the indexed loaders
# and the row cap (PITFALLS 13.4), and lets the term run inside any loop that holds a batch's
# clouds and a latent -- the supervised objectives included. What it costs is one Sinkhorn
# evaluation per pair per step, so the number of pairs is a budget (`sample_pairs`), not
# the full B(B-1)/2: at batch 256 every pair is 32,640 Sinkhorn calls per step (~1 s on a
# shared RTX 6000 Ada), one random partner per row is 256 (~5 ms, the reconstruction loss's
# own cost). The Pearson objective needs spread across pairs, not completeness.

#: The budget value that means every unordered pair of the batch.
PAIRS_ALL = "all"
#: Default pairs per batch: every pair at batch 64 (the recorded Ant-Tag aligned recipe), a
#: 1/16 sample at batch 256 (the hunt b256 recipe).
DEFAULT_PAIRS = 2016
#: Default size of the fixed held-out pair sample behind ``val/align_r``.
DEFAULT_VAL_PAIRS = 20000


def parse_pairs(value) -> Union[int, str]:
    """``"all"`` or a positive integer budget (the ``type=`` of ``--align_pairs``)."""
    if isinstance(value, str) and value.strip().lower() == PAIRS_ALL:
        return PAIRS_ALL
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"align_pairs must be 'all' or a positive integer, got {value!r}") from None
    if n <= 0:
        raise ValueError(f"align_pairs must be 'all' or a positive integer, got {value!r}")
    return n


def all_pairs(batch_size: int, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Every unordered pair ``(a < b)`` of a batch, in :func:`flatten_upper_triangle` order."""
    iu = torch.triu_indices(batch_size, batch_size, offset=1, device=device)
    return iu[0], iu[1]


def sample_pairs(batch_size: int, budget: Union[int, str],
                 generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(a, b)`` index vectors of unordered pairs (``a < b``) over one batch, on the CPU.

    ``budget`` ``"all"``, or at least ``B (B - 1) / 2``: every pair, in triangle order (the
    matrix path's pairs). Otherwise ``k = ceil(budget / B)`` random permutations of the batch,
    each row paired with its image (fixed points dropped, duplicates removed), cut to
    ``budget`` pairs at random. Every row is then in about ``k`` pairs, so every code receives
    an alignment gradient each step; drawing ``budget`` pairs uniformly could leave rows out.
    ``generator`` is the sampler's own stream, so the loader's shuffle and everything seeded
    from torch's global generator stay untouched. Fewer than ``budget`` pairs come back only
    when the de-duplication removes some (a budget close to B(B-1)/2 on a small batch).
    """
    n_all = batch_size * (batch_size - 1) // 2
    if budget == PAIRS_ALL or int(budget) >= n_all:
        return all_pairs(batch_size)
    budget = int(budget)
    k = -(-budget // max(batch_size, 1))
    rows = torch.arange(batch_size)
    chunks = []
    for _ in range(k):
        perm = torch.randperm(batch_size, generator=generator)
        keep = perm != rows
        lo = torch.minimum(rows[keep], perm[keep])
        hi = torch.maximum(rows[keep], perm[keep])
        chunks.append(lo * batch_size + hi)
    codes = torch.unique(torch.cat(chunks))
    if len(codes) > budget:
        codes = codes[torch.randperm(len(codes), generator=generator)[:budget]]
    return codes // batch_size, codes % batch_size


def latent_pair_distances(z: torch.Tensor, pairs: Tuple[torch.Tensor, torch.Tensor],
                          metric: str = "cosine") -> torch.Tensor:
    """The latent distance of each listed pair: :func:`latent_pairwise_distances` restricted
    to ``pairs`` (same metric, same numbers, ``(P,)`` instead of the full triangle)."""
    if metric not in LATENT_METRICS:
        raise ValueError(f"metric must be one of {LATENT_METRICS}, got {metric!r}")
    a, b = pairs
    flat = z.reshape(z.shape[0], -1)
    if metric == "cosine":
        normed = F.normalize(flat, dim=1, eps=1e-8)
        return 1.0 - (normed[a] * normed[b]).sum(dim=1)
    return (flat[a] - flat[b]).norm(dim=1)


def sinkhorn_pair_targets(particles: torch.Tensor, weights: Optional[torch.Tensor], sinkhorn,
                          pairs: Tuple[torch.Tensor, torch.Tensor], chunk: int = 4096) -> torch.Tensor:
    """The debiased Sinkhorn divergence between the two clouds of every listed pair, under
    ``no_grad``: the matrix path's entries (:mod:`set_transformer.emd_matrix`), computed for
    these pairs only.

    ``sinkhorn`` is a geomloss ``SamplesLoss("sinkhorn", ...)`` with its default
    ``debias=True``, built with the blur and scaling of the frame ``particles`` are in (the
    reconstruction loss's own settings). ``weights`` ``None`` = uniform clouds (the
    two-argument call); otherwise the PF weights, normalised by the training loss's rule
    (:meth:`~set_transformer.loss.SampleLoss._as_measure_weights`: mass in the measure, never in
    the ground metric; raises on NaN or zero mass), cast to the particles' dtype (Odd-Even keeps
    its posteriors in float64).
    """
    from .loss import SampleLoss   # local: loss.py imports POT, which this module does not need
    a, b = pairs
    pts = particles.detach()
    a = a.to(pts.device)
    b = b.to(pts.device)
    w = None
    if weights is not None:
        w = SampleLoss._as_measure_weights(
            weights.detach().to(device=pts.device, dtype=pts.dtype), pts)
    out = torch.empty(len(a), device=pts.device, dtype=pts.dtype)
    with torch.no_grad():
        for s in range(0, len(a), chunk):
            ia, ib = a[s:s + chunk], b[s:s + chunk]
            if w is None:
                out[s:s + chunk] = sinkhorn(pts[ia], pts[ib])
            else:
                out[s:s + chunk] = sinkhorn(w[ia], pts[ia], w[ib], pts[ib])
    return out


class OnlineAlignment:
    """The alignment term with per-batch targets: sample the pairs, compute their debiased
    Sinkhorn divergences from the batch's own clouds, correlate the latent distances of the
    same pairs (``1 - pearson_r``, :class:`PearsonAlignmentLoss`).

    One instance per training run. ``pairs`` is the per-batch budget (:func:`sample_pairs`);
    the sampler draws from its own generator seeded with ``seed``, and the held-out sample
    behind the validation correlation (:meth:`fixed_pairs`) from ``seed + 1``, drawn once so
    the metric is comparable across epochs. ``blur`` / ``scaling`` are in the frame the clouds
    are handed in (the reconstruction objective's loader frame; the task objectives divide the
    raw cloud by the dataset's scale first, as their extractor does).
    """

    def __init__(self, *, blur: float, scaling: float = 0.5, p: int = 2, metric: str = "cosine",
                 pairs: Union[int, str] = DEFAULT_PAIRS, seed: int = 0, chunk: int = 4096) -> None:
        from geomloss import SamplesLoss   # local, as in loss.py's users
        if float(blur) <= 0 or float(scaling) <= 0:
            raise ValueError("blur and scaling must be positive")
        self.blur, self.scaling, self.p = float(blur), float(scaling), int(p)
        self.metric = metric
        self.pairs = parse_pairs(pairs)
        self.seed = int(seed)
        self.chunk = int(chunk)
        self.sinkhorn = SamplesLoss("sinkhorn", p=self.p, blur=self.blur, scaling=self.scaling)
        self.loss = PearsonAlignmentLoss(metric)
        self.generator = torch.Generator().manual_seed(self.seed)

    def targets(self, particles: torch.Tensor, weights: Optional[torch.Tensor],
                pairs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return sinkhorn_pair_targets(particles, weights, self.sinkhorn, pairs, chunk=self.chunk)

    def term(self, latent: torch.Tensor, particles: torch.Tensor,
             weights: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """``(loss, r, n_pairs)`` for one batch. ``loss`` is graph-connected to ``latent``; the
        targets carry no gradient."""
        a, b = sample_pairs(int(latent.shape[0]), self.pairs, self.generator)
        pairs = (a.to(latent.device), b.to(latent.device))
        d_target = self.targets(particles, weights, pairs)
        loss, r = self.loss(latent, d_target, pairs=pairs)
        return loss, r, int(len(a))

    def fixed_pairs(self, n_rows: int, budget: Union[int, str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """A pair sample over ``n_rows`` held-out rows, drawn once from ``seed + 1``."""
        return sample_pairs(int(n_rows), budget, torch.Generator().manual_seed(self.seed + 1))

    def correlation(self, latent: torch.Tensor, pairs: Tuple[torch.Tensor, torch.Tensor],
                    d_target: torch.Tensor) -> float:
        """Pearson r of the latent distances of ``pairs`` against ``d_target`` (NaN if degenerate)."""
        a, b = pairs
        r = pearson_r(latent_pair_distances(latent, (a.to(latent.device), b.to(latent.device)), self.metric),
                      d_target.to(device=latent.device, dtype=latent.dtype))
        return float("nan") if r is None else float(r)

    def record(self) -> dict:
        """What a checkpoint records about the target (beside lambda and the ramp)."""
        return {"target": "online", "metric": self.metric, "pairs": self.pairs,
                "blur": self.blur, "scaling": self.scaling, "p": self.p, "pair_seed": self.seed}


def add_alignment_arguments(target, *, budget_only: bool = False, sinkhorn: bool = False,
                            sinkhorn_scaling: float = 0.5) -> None:
    """The alignment flags, spelled once for every objective that carries the term.

    ``target`` is a parser or an argument group. ``budget_only`` adds the two online-target
    flags (``--align_pairs`` / ``--align_val_pairs``) beside an existing ``--align_lambda``
    group (the reconstruction objective's). With ``sinkhorn`` the target's geometry flags are
    added too (the supervised objectives have no Sinkhorn flags of their own), spelled as the
    reconstruction objective spells them. ``--sinkhorn_blur`` defaults to None and is filled by
    :func:`resolve_sinkhorn_blur` with the DOMAIN's default (``Domain.default_sinkhorn_blur``), so
    a command that omits it gets the value the domain's recipes use, not geomloss's 0.05.
    """
    if not budget_only:
        target.add_argument(
            "--align_lambda", type=float, default=0.0,
            help="Weight of the latent metric-alignment term, 1 - pearson_r between the batch's "
                 "latent pairwise distances and the debiased Sinkhorn divergences between the same "
                 "clouds, computed online per batch (set_transformer.latent_alignment). 0 (default) "
                 "= off: nothing in this group is read.")
        target.add_argument("--align_metric", type=str, default="cosine", choices=list(LATENT_METRICS))
        target.add_argument("--align_warmup_epochs", type=int, default=0, help="Epochs with lambda held at 0.")
        target.add_argument("--align_ramp_epochs", type=int, default=0,
                            help="Epochs over which lambda ramps linearly to its target.")
    target.add_argument(
        "--align_pairs", type=parse_pairs, default=DEFAULT_PAIRS,
        help="Online targets: pairs per batch, 'all' or a budget (default %(default)s = every pair "
             "at batch 64). A budget is filled with ceil(P / B) random permutations of the batch, "
             "so every row is in about that many pairs each step.")
    target.add_argument(
        "--align_val_pairs", type=int, default=DEFAULT_VAL_PAIRS,
        help="Online targets: size of the fixed held-out pair sample behind val/align_r, drawn once.")
    if sinkhorn:
        target.add_argument(
            "--sinkhorn_blur", type=float, default=None,
            help="Sinkhorn eps of the alignment target, a length in the encoder's NORMALISED "
                 "coordinates (the cloud divided by the dataset's scale); the reconstruction "
                 "objective's own flag. Default: the domain's default_sinkhorn_blur (0.02 hunt / "
                 "Odd-Even / msearch, 0.01 Ant-Tag). Read only with --align_lambda > 0.")
        target.add_argument("--sinkhorn_scaling", type=float, default=sinkhorn_scaling,
                            help="Epsilon-scaling decay of the alignment target's Sinkhorn.")


#: geomloss's own default, the value the reconstruction flag carried before 2026-09-19 and the
#: fallback for a domain that declares none.
PACKAGE_SINKHORN_BLUR = 0.05


def resolve_sinkhorn_blur(args, domain=None, *, fallback: float = PACKAGE_SINKHORN_BLUR) -> str:
    """Fill ``args.sinkhorn_blur`` when the command omitted it, from the domain's
    ``default_sinkhorn_blur``; return where the value came from (``"given"``,
    ``"domain default"`` or ``"package default"``), also stored as ``args.sinkhorn_blur_source``
    so the run record says it. Idempotent."""
    source = getattr(args, "sinkhorn_blur_source", None)
    if getattr(args, "sinkhorn_blur", None) is not None and source is None:
        source = "given"
    elif getattr(args, "sinkhorn_blur", None) is None:
        default = getattr(domain, "default_sinkhorn_blur", None) if domain is not None else None
        args.sinkhorn_blur = float(default) if default is not None else float(fallback)
        source = "domain default" if default is not None else "package default"
    args.sinkhorn_blur_source = source
    return source
