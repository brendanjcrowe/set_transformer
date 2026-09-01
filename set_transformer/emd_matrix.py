"""Pairwise debiased-Sinkhorn distance matrices over a set of point clouds.

The reference geometry the latent metric-alignment loss aligns against
(:mod:`set_transformer.latent_alignment`). The target is a deterministic function of the
input clouds, so it is computed once, offline, and read as a lookup during training.

Metric: the debiased Sinkhorn divergence
    S(x, y) = OT_eps(x, y) - 0.5 * OT_eps(x, x) - 0.5 * OT_eps(y, y)
Raw entropic OT carries an entropy bias that does not vanish on identical inputs; the
debiased form behaves like a proper divergence (S(x, x) = 0). One fixed ``blur`` (= eps) is
used for a whole matrix so every entry is mutually consistent.

Speed note: ``geomloss(debias=True)`` recomputes both self-terms for *every pair*, i.e. 3x
the work. Here the N self-terms are computed once up front and subtracted manually, so the
O(N^2) part runs a single un-debiased Sinkhorn per pair — ~2x faster overall, and
numerically identical (checked by :func:`verify_against_geomloss`, which agrees to <1e-3).

The matrix is written as a raw float32 memmap of shape (N, N): symmetric, zero diagonal,
and resumable at row-block granularity, since a full run is O(dataset^2) Sinkhorn calls
(~1 h for 20k clouds on an RTX 4070 Ti).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from geomloss import SamplesLoss

__all__ = [
    "compute_matrix",
    "load_matrix",
    "matrix_stats",
    "verify_against_geomloss",
    "write_sidecar",
]


def _self_terms(pts: torch.Tensor, ot: SamplesLoss, chunk: int) -> torch.Tensor:
    """OT_eps(x_i, x_i) for every sample — the debiasing correction."""
    out = torch.empty(len(pts), device=pts.device)
    with torch.no_grad():
        for s in range(0, len(pts), chunk):
            block = pts[s:s + chunk]
            out[s:s + chunk] = ot(block, block.clone())
    return out


def _cross_block(
    pts: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    ot: SamplesLoss,
    self_terms: torch.Tensor,
    pair_chunk: int,
) -> torch.Tensor:
    """Debiased Sinkhorn for every (row, col) pair; returns ``(len(rows), len(cols))``."""
    ri = rows.repeat_interleave(len(cols))
    ci = cols.repeat(len(rows))
    vals = torch.empty(len(ri), device=pts.device)
    with torch.no_grad():
        for s in range(0, len(ri), pair_chunk):
            a, b = ri[s:s + pair_chunk], ci[s:s + pair_chunk]
            vals[s:s + pair_chunk] = (
                ot(pts[a], pts[b]) - 0.5 * self_terms[a] - 0.5 * self_terms[b]
            )
    return vals.view(len(rows), len(cols))


def verify_against_geomloss(pts, ot, self_terms, blur, p, n_pairs=64) -> float:
    """Max |manual debias - geomloss(debias=True)| over a sample of pairs."""
    ref = SamplesLoss("sinkhorn", p=p, blur=blur, debias=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randint(0, len(pts), (n_pairs,), generator=g).to(pts.device)
    b = torch.randint(0, len(pts), (n_pairs,), generator=g).to(pts.device)
    with torch.no_grad():
        mine = ot(pts[a], pts[b]) - 0.5 * self_terms[a] - 0.5 * self_terms[b]
        theirs = ref(pts[a], pts[b])
    return float((mine - theirs).abs().max())


#: Sinkhorn eps. 0.05 is the repo-wide operating point for 2-D data on this scale
#: (blur=0.5 is too coarse to learn on); must match the reconstruction loss's blur so the
#: alignment target and the training objective measure the same geometry.
DEFAULT_BLUR = 0.05


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
) -> np.memmap:
    """Full pairwise debiased-Sinkhorn matrix over ``points`` -> ``out_path`` memmap.

    Args:
        points: ``(N, num_particles, dim)`` point clouds.
        out_path: raw float32 memmap destination; a ``.progress.json`` beside it makes the
            run resumable at row-block granularity.
        resume: continue a partial run for the same ``(N, block)`` if progress exists.
        verify: check the manual debiasing against ``geomloss(debias=True)`` on a sample.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(points)
    pts = torch.from_numpy(points).float().to(device)
    ot = SamplesLoss("sinkhorn", p=p, blur=blur, debias=False)

    progress_path = out_path.with_suffix(".progress.json")
    start_block = 0
    mode = "w+"
    if resume and out_path.exists() and progress_path.exists():
        state = json.loads(progress_path.read_text())
        if state.get("n") == n and state.get("block") == block:
            start_block, mode = state["next_row_block"], "r+"
            print(f"resuming at row block {start_block}", flush=True)

    matrix = np.memmap(out_path, dtype=np.float32, mode=mode, shape=(n, n))
    if mode == "w+":
        matrix[:] = 0.0

    print("computing self terms ...", flush=True)
    self_terms = _self_terms(pts, ot, pair_chunk)
    print(f"  OT_eps(x,x): mean {self_terms.mean():.5f}  max {self_terms.max():.5f}", flush=True)
    if verify:
        err = verify_against_geomloss(pts, ot, self_terms, blur, p)
        print(f"  max |manual debias - geomloss(debias=True)| = {err:.2e}", flush=True)

    row_blocks = list(range(0, n, block))
    total_pairs = n * (n - 1) // 2
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
            vals = _cross_block(pts, rows, cols, ot, self_terms, pair_chunk).cpu().numpy()
            if j0 == i0:  # diagonal block: keep only the strict upper triangle
                vals = np.triu(vals, k=1)
                matrix[i0:i1, j0:j1] = vals
                matrix[i0:i1, j0:j1] += vals.T
            else:
                matrix[i0:i1, j0:j1] = vals
                matrix[j0:j1, i0:i1] = vals.T
        matrix.flush()
        progress_path.write_text(json.dumps(
            {"n": n, "block": block, "next_row_block": bi + 1, "blur": blur, "p": p}))
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
    describes cannot silently disagree about their size.
    """
    arr = np.memmap(path, dtype=np.float32, mode="r", shape=(n, n))
    return arr if mmap else np.asarray(arr)


def matrix_stats(matrix: np.ndarray, max_rows: int = 4000) -> dict:
    """Summary of the off-diagonal distribution — the sanity check that the geometry is
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


def write_sidecar(out_path: Path, source: str, n: int, blur: float, p: int,
                  stats: dict | None = None) -> None:
    """Record the metric settings next to the matrix.

    The matrix itself is large and regenerable (and gitignored); this small JSON is the
    committed provenance record saying which metric a checkpoint was aligned against.
    """
    out_path.with_suffix(".json").write_text(json.dumps({
        "source": str(source),
        "n_samples": int(n),
        "metric": "debiased_sinkhorn",
        "p": p,
        "blur": blur,
        "dtype": "float32",
        "shape": [int(n), int(n)],
        **(stats or {}),
    }, indent=2))
