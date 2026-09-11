"""Precompute the pairwise debiased-Sinkhorn distance matrix over a MoG split.

Thin CLI over :mod:`set_transformer.emd_matrix`, which holds the actual computation and
is shared with the benchmark's pretraining pipeline. Only the paths, the histogram figure
and the split handling are specific to this study.

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
from set_transformer.emd_matrix import (
    compute_matrix,
    matrix_stats,
    verify_against_geomloss,
    write_sidecar,
)

from _common import DATA_DIR, SINKHORN_BLUR

FIG_DIR_DEFAULT = Path("experiments/mog/figures_align")


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
        plot_histogram(
            matrix, args.fig_dir / f"emd_matrix_hist_{split}.png",
            f"pairwise debiased Sinkhorn ({split}, blur={args.blur})",
        )
        stats = matrix_stats(matrix)
        stats["wall_seconds"] = round(time.time() - t0, 1)
        write_sidecar(out_path, points_path, len(points), args.blur, args.p, stats)
        print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
