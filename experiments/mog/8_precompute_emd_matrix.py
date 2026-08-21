"""Precompute the full pairwise debiased-Sinkhorn distance matrix over a MoG split.

The latent metric-alignment loss needs a *fixed reference geometry* over the training
set. Recomputing it inside the training loop would be both slow and redundant (the target
is a deterministic function of the input clouds), so it is computed once here and read as
a lookup during training.

Metric: the debiased Sinkhorn divergence
    S(x, y) = OT_eps(x, y) - 0.5 * OT_eps(x, x) - 0.5 * OT_eps(y, y)
Raw entropic OT carries an entropy bias that does not vanish on identical inputs; the
debiased form behaves like a proper divergence (S(x, x) = 0). One fixed ``blur`` (= eps)
is used for the whole matrix so every entry is mutually consistent.

Speed note: ``geomloss(debias=True)`` recomputes both self-terms for *every pair*, i.e.
3x the work. Here the N self-terms are computed once up front and subtracted manually, so
the O(N^2) part runs a single un-debiased Sinkhorn per pair — ~2x faster overall, and
numerically identical (verified by ``--verify``).

    python experiments/mog/8_precompute_emd_matrix.py --split both

Writes ``<data_dir>/emd_<split>.npy`` (float32, N x N, symmetric, zero diagonal) plus a
``.json`` sidecar recording the metric settings, and a histogram for the sanity check.
Resumable at row-block granularity: re-running picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from geomloss import SamplesLoss

from _common import DATA_DIR, SINKHORN_BLUR

FIG_DIR_DEFAULT = Path("experiments/mog/figures_align")


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


def compute_matrix(
    points: np.ndarray,
    out_path: Path,
    blur: float,
    p: int,
    block: int,
    pair_chunk: int,
    device: str,
    resume: bool,
    verify: bool,
) -> np.memmap:
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


def write_sidecar(out_path: Path, points_path: Path, n: int, blur: float, p: int,
                  stats: dict) -> None:
    out_path.with_suffix(".json").write_text(json.dumps({
        "source": str(points_path),
        "n_samples": n,
        "metric": "debiased_sinkhorn",
        "p": p,
        "blur": blur,
        "dtype": "float32",
        "shape": [n, n],
        **stats,
    }, indent=2))


def plot_histogram(matrix: np.ndarray, fig_path: Path, title: str, max_rows: int = 4000) -> dict:
    """Sanity check: off-diagonal values should be spread, not a narrow band."""
    sub = np.asarray(matrix[:max_rows, :max_rows], dtype=np.float64)
    iu = np.triu_indices(len(sub), k=1)
    vals = sub[iu]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=120, color="tab:purple", alpha=0.85)
    ax.set_xlabel("debiased Sinkhorn distance")
    ax.set_ylabel("pair count")
    ax.set_title(title)
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return {
        "offdiag_mean": float(vals.mean()),
        "offdiag_std": float(vals.std()),
        "offdiag_min": float(vals.min()),
        "offdiag_max": float(vals.max()),
        "diag_absmax": float(np.abs(np.diag(sub)).max()),
        "hist_sampled_rows": int(len(sub)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("experiments/mog/data_varsep"))
    ap.add_argument("--split", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--blur", type=float, default=SINKHORN_BLUR,
                    help="Sinkhorn eps, in ground-distance units; must match the recon loss")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--block", type=int, default=128, help="row/col block size")
    ap.add_argument("--pair_chunk", type=int, default=4096, help="pairs per geomloss call")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="truncate the split to the first K samples (0 = all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fig_dir", type=Path, default=FIG_DIR_DEFAULT)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--verify", action="store_true", default=True)
    args = ap.parse_args()

    splits = ["train", "eval"] if args.split == "both" else [args.split]
    for split in splits:
        points_path = args.data_dir / f"{split}.points.npy"
        points = np.load(points_path)
        if args.max_samples:
            points = points[:args.max_samples]
        out_path = args.data_dir / f"emd_{split}.npy"
        print(f"\n=== {split}: {points.shape} -> {out_path} "
              f"({points.shape[0] ** 2 * 4 / 1e9:.2f} GB) ===", flush=True)
        t0 = time.time()
        matrix = compute_matrix(
            points, out_path, args.blur, args.p, args.block, args.pair_chunk,
            args.device, resume=not args.no_resume, verify=args.verify,
        )
        stats = plot_histogram(
            matrix, args.fig_dir / f"emd_matrix_hist_{split}.png",
            f"pairwise debiased Sinkhorn ({split}, blur={args.blur})",
        )
        stats["wall_seconds"] = round(time.time() - t0, 1)
        write_sidecar(out_path, points_path, len(points), args.blur, args.p, stats)
        print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
