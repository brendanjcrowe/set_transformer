"""Pairwise debiased-Sinkhorn distance matrices over a set of (weighted) point clouds.

The reference geometry the latent metric-alignment loss aligns against
(:mod:`set_transformer.latent_alignment`). The target is a deterministic function of the
input clouds, so it is computed once, offline, and read as a lookup during training.

Metric: the debiased Sinkhorn divergence between the measures
    alpha_i = sum_k w_ik delta(x_ik)
    S(alpha, beta) = OT_eps(alpha, beta) - 0.5 * OT_eps(alpha, alpha) - 0.5 * OT_eps(beta, beta)
Raw entropic OT carries an entropy bias that does not vanish on identical inputs; the
debiased form behaves like a proper divergence (S(alpha, alpha) = 0). One fixed ``blur``
(= eps) and ``scaling`` are used for a whole matrix so every entry is mutually consistent
-- and they must equal the reconstruction loss's, so the alignment target and the
training objective measure the same geometry.

**Weights: mass in the measure, never in the metric.** With ``weights=None`` every cloud
is uniform and geomloss is called in its two-argument form -- the collaborator's original
path, bit-for-bit. With per-particle weights the four-argument form
``SamplesLoss(w_a, x_a, w_b, x_b)`` is used and each cloud's self-term is computed with
its OWN weights, because the entropic bias of ``OT_eps(alpha, alpha)`` depends on the
measure: two clouds on the same support with different weights have different biases, and
skipping the correction would make the matrix reflect weight entropy rather than geometry.
Weights are normalized by the exact-sum rule the training loss uses
(:meth:`~set_transformer.loss.SampleLoss._as_measure_weights`), which raises on zero, NaN or
infinite mass. The weighted form is what makes the matrix non-degenerate on the Odd-Even
exact-support filter, where every cloud has the SAME support and only the weights differ
(an unweighted matrix there is identically zero).

Speed note: ``geomloss(debias=True)`` recomputes both self-terms for *every pair*, i.e. 3x
the work. Here the N self-terms are computed once up front and subtracted manually, so the
O(N^2) part runs a single un-debiased Sinkhorn per pair -- ~2x faster overall, and
numerically identical (checked by :func:`verify_against_geomloss`, which agrees to <1e-3).
Weights add no per-pair cost: the cost matrix is still ``n x n`` particles.

The matrix is written as a raw float32 memmap of shape (N, N): symmetric, zero diagonal,
and resumable at row-block granularity, since a full run is O(dataset^2) Sinkhorn calls
(~1 h for 20k clouds on an RTX 4070 Ti; 4 N^2 bytes on disk).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from geomloss import SamplesLoss

from set_transformer.loss import SampleLoss

__all__ = [
    "DEFAULT_BLUR",
    "DEFAULT_SCALING",
    "compute_matrix",
    "dataset_sha256",
    "load_matrix",
    "matrix_stats",
    "sidecar_path",
    "read_sidecar",
    "verify_against_geomloss",
    "write_sidecar",
]


def _prepare_weights(pts: torch.Tensor, weights) -> Optional[torch.Tensor]:
    """Normalized ``(N, n)`` masses on ``pts``'s device, or None for uniform clouds."""
    if weights is None:
        return None
    w = torch.as_tensor(np.asarray(weights) if not torch.is_tensor(weights) else weights)
    w = w.to(device=pts.device, dtype=pts.dtype)
    if w.shape != pts.shape[:2]:
        raise ValueError(
            f"weights must have shape (N, num_particles) = {tuple(pts.shape[:2])}, "
            f"got {tuple(w.shape)}")
    # Exact normalization; raises on zero / NaN / inf mass. Same contract as the
    # reconstruction loss, so the two sides agree on what a weight means.
    return SampleLoss._as_measure_weights(w, pts)


def _ot_pairs(ot: SamplesLoss, pts_a, pts_b, w_a, w_b) -> torch.Tensor:
    """``OT_eps`` between matched rows of two batches, weighted or uniform."""
    if w_a is None:
        return ot(pts_a, pts_b)
    return ot(w_a, pts_a, w_b, pts_b)


def _self_terms(pts: torch.Tensor, ot: SamplesLoss, chunk: int,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """OT_eps(alpha_i, alpha_i) for every sample -- the debiasing correction."""
    out = torch.empty(len(pts), device=pts.device)
    with torch.no_grad():
        for s in range(0, len(pts), chunk):
            block = pts[s:s + chunk]
            w = None if weights is None else weights[s:s + chunk]
            out[s:s + chunk] = _ot_pairs(ot, block, block.clone(), w, w)
    return out


def _cross_block(
    pts: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    ot: SamplesLoss,
    self_terms: torch.Tensor,
    pair_chunk: int,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Debiased Sinkhorn for every (row, col) pair; returns ``(len(rows), len(cols))``."""
    ri = rows.repeat_interleave(len(cols))
    ci = cols.repeat(len(rows))
    vals = torch.empty(len(ri), device=pts.device)
    with torch.no_grad():
        for s in range(0, len(ri), pair_chunk):
            a, b = ri[s:s + pair_chunk], ci[s:s + pair_chunk]
            w_a = None if weights is None else weights[a]
            w_b = None if weights is None else weights[b]
            vals[s:s + pair_chunk] = (
                _ot_pairs(ot, pts[a], pts[b], w_a, w_b)
                - 0.5 * self_terms[a] - 0.5 * self_terms[b]
            )
    return vals.view(len(rows), len(cols))


def verify_against_geomloss(pts, ot, self_terms, blur, p, n_pairs=64, weights=None,
                            scaling: float = 0.5) -> float:
    """Max |manual debias - geomloss(debias=True)| over a sample of pairs."""
    ref = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, debias=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randint(0, len(pts), (n_pairs,), generator=g).to(pts.device)
    b = torch.randint(0, len(pts), (n_pairs,), generator=g).to(pts.device)
    w_a = None if weights is None else weights[a]
    w_b = None if weights is None else weights[b]
    with torch.no_grad():
        mine = _ot_pairs(ot, pts[a], pts[b], w_a, w_b) - 0.5 * self_terms[a] - 0.5 * self_terms[b]
        theirs = _ot_pairs(ref, pts[a], pts[b], w_a, w_b)
    return float((mine - theirs).abs().max())


#: Sinkhorn eps. 0.05 is the repo-wide default for 2-D data on this scale (blur=0.5 is too
#: coarse to learn on); must match the reconstruction loss's blur so the alignment target
#: and the training objective measure the same geometry. Blur is a length in the
#: coordinates the matrix is computed in -- i.e. AFTER the dataset's
#: (x - particle_centre) / particle_scale mapping, the same frame the trainer sees.
DEFAULT_BLUR = 0.05
#: Epsilon-scaling decay; geomloss's and SinkhornLoss's default.
DEFAULT_SCALING = 0.5


def compute_matrix(
    points: np.ndarray,
    out_path: Path,
    blur: float = DEFAULT_BLUR,
    p: int = 2,
    block: int = 128,
    pair_chunk: int = 4096,
    device: str | None = None,
    resume: bool = True,
    verify: bool = True,
    weights: Optional[np.ndarray] = None,
    scaling: float = DEFAULT_SCALING,
) -> np.memmap:
    """Full pairwise debiased-Sinkhorn matrix over ``points`` -> ``out_path`` memmap.

    Args:
        points: ``(N, num_particles, dim)`` point clouds, already in the frame the
            trainer will see (``POMDPDataset.data``).
        out_path: raw float32 memmap destination; a ``.progress.json`` beside it makes the
            run resumable at row-block granularity.
        weights: optional ``(N, num_particles)`` per-particle masses. None = uniform
            clouds (two-argument geomloss call, the original unweighted path).
        scaling: epsilon-scaling decay; must match the reconstruction loss's.
        resume: continue a partial run for the same ``(N, block)`` if progress exists.
        verify: check the manual debiasing against ``geomloss(debias=True)`` on a sample.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(out_path)
    n = len(points)
    pts = torch.as_tensor(np.asarray(points) if not torch.is_tensor(points) else points)
    pts = pts.float().to(device)
    w = _prepare_weights(pts, weights)
    ot = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling, debias=False)

    progress_path = out_path.with_suffix(".progress.json")
    start_block = 0
    mode = "w+"
    if resume and out_path.exists() and progress_path.exists():
        state = json.loads(progress_path.read_text())
        same = (state.get("n") == n and state.get("block") == block
                and state.get("weighted", False) == (w is not None)
                and state.get("blur") == blur and state.get("scaling", DEFAULT_SCALING) == scaling)
        if same:
            start_block, mode = state["next_row_block"], "r+"
            print(f"resuming at row block {start_block}", flush=True)
        else:
            print("progress file does not match (n/block/weighted/blur/scaling); "
                  "starting over", flush=True)

    matrix = np.memmap(out_path, dtype=np.float32, mode=mode, shape=(n, n))
    if mode == "w+":
        matrix[:] = 0.0

    print(f"computing self terms ({'weighted' if w is not None else 'uniform'}) ...",
          flush=True)
    self_terms = _self_terms(pts, ot, pair_chunk, w)
    print(f"  OT_eps(x,x): mean {self_terms.mean():.5f}  max {self_terms.max():.5f}", flush=True)
    if verify:
        err = verify_against_geomloss(pts, ot, self_terms, blur, p, weights=w, scaling=scaling)
        print(f"  max |manual debias - geomloss(debias=True)| = {err:.2e}", flush=True)

    row_blocks = list(range(0, n, block))
    total_pairs = max(n * (n - 1) // 2, 1)
    t0 = time.time()

    def _pairs_in_row_block(i0: int) -> int:
        """Unordered pairs a row block owns: its cross columns plus its own triangle."""
        h = min(i0 + block, n) - i0
        return h * (n - i0 - h) + h * (h - 1) // 2

    done_pairs = sum(_pairs_in_row_block(i0) for i0 in row_blocks[:start_block])
    for bi in range(start_block, len(row_blocks)):
        i0 = row_blocks[bi]
        i1 = min(i0 + block, n)
        rows = torch.arange(i0, i1, device=device)
        for j0 in range(i0, n, block):
            j1 = min(j0 + block, n)
            cols = torch.arange(j0, j1, device=device)
            vals = _cross_block(pts, rows, cols, ot, self_terms, pair_chunk, w).cpu().numpy()
            if j0 == i0:  # diagonal block: keep only the strict upper triangle
                vals = np.triu(vals, k=1)
                matrix[i0:i1, j0:j1] = vals
                matrix[i0:i1, j0:j1] += vals.T
            else:
                matrix[i0:i1, j0:j1] = vals
                matrix[j0:j1, i0:i1] = vals.T
        matrix.flush()
        progress_path.write_text(json.dumps(
            {"n": n, "block": block, "next_row_block": bi + 1, "blur": blur, "p": p,
             "scaling": scaling, "weighted": w is not None}))
        done_pairs += _pairs_in_row_block(i0)
        frac = done_pairs / total_pairs
        elapsed = time.time() - t0
        eta = elapsed / max(frac, 1e-9) * (1 - frac) if bi > start_block else float("nan")
        print(f"  row block {bi + 1}/{len(row_blocks)}  {100 * frac:5.1f}%  "
              f"elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m", flush=True)

    np.fill_diagonal(matrix, 0.0)
    matrix.flush()
    return matrix


def load_matrix(path: Path, n: int, mmap: bool = False) -> np.ndarray:
    """Load a matrix written by :func:`compute_matrix`.

    ``n`` comes from the point set the caller holds, so a matrix and the clouds it
    describes cannot silently disagree about their size: ``np.memmap`` would accept a
    file LARGER than ``4 n^2`` bytes without complaint, so the size is checked exactly.
    """
    path = Path(path)
    expected = 4 * n * n
    actual = path.stat().st_size
    if actual != expected:
        side = n_from_size(actual)
        raise ValueError(
            f"{path} holds {actual} bytes, but a float32 matrix over n={n} clouds needs "
            f"{expected}. The file describes {side if side else 'a non-square number of'} "
            "clouds. Recompute it for this dataset, or load the dataset with the same "
            "max_samples the matrix was built with (the .json sidecar records n_samples)."
        )
    arr = np.memmap(path, dtype=np.float32, mode="r", shape=(n, n))
    return arr if mmap else np.asarray(arr)


def n_from_size(nbytes: int) -> Optional[int]:
    """The side of a square float32 matrix of ``nbytes``, or None if not square."""
    side = int(round((nbytes / 4) ** 0.5))
    return side if 4 * side * side == nbytes else None


def matrix_stats(matrix: np.ndarray, max_rows: int = 4000) -> dict:
    """Summary of the off-diagonal distribution -- the sanity check that the geometry is
    spread rather than collapsed into a narrow band (which would make Pearson alignment
    meaningless). Sampled over the leading ``max_rows`` for large matrices."""
    sub = np.asarray(matrix[:max_rows, :max_rows], dtype=np.float64)
    vals = sub[np.triu_indices(len(sub), k=1)]
    return {
        "offdiag_mean": float(vals.mean()),
        "offdiag_std": float(vals.std()),
        "offdiag_min": float(vals.min()),
        "offdiag_max": float(vals.max()),
        "diag_absmax": float(np.abs(np.diag(sub)).max()),
        "stats_sampled_rows": int(len(sub)),
    }


def dataset_sha256(path: Path, chunk: int = 1 << 22) -> str:
    """SHA-256 of a dataset file, so a sidecar can name the exact data it was built from."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sidecar_path(out_path: Path) -> Path:
    return Path(out_path).with_suffix(".json")


def write_sidecar(out_path: Path, source: str, n: int, blur: float, p: int,
                  stats: dict | None = None, *, weighted: bool = False,
                  scaling: float = DEFAULT_SCALING, extra: dict | None = None) -> None:
    """Record the metric settings next to the matrix.

    The matrix itself is large and regenerable (and gitignored); this small JSON is the
    committed provenance record saying which metric a checkpoint was aligned against.
    ``extra`` carries the dataset frame (``particle_scale``, ``particle_centre``,
    ``data_sha256``) so ``3_train_st.py`` can refuse a matrix built from different data.
    """
    sidecar_path(out_path).write_text(json.dumps({
        "source": str(source),
        "n_samples": int(n),
        "metric": "debiased_sinkhorn",
        "weighted": bool(weighted),
        "p": p,
        "blur": blur,
        "scaling": scaling,
        "dtype": "float32",
        "shape": [int(n), int(n)],
        **(extra or {}),
        **(stats or {}),
    }, indent=2))


def read_sidecar(out_path: Path) -> dict:
    path = sidecar_path(out_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found beside the EMD matrix. compute_matrix() writes the matrix; "
            "the precompute script writes this provenance sidecar. Re-run it.")
    return json.loads(path.read_text())
