"""Precompute the pairwise EMD matrix for an env's PF dataset.

Thin CLI over :mod:`set_transformer.emd_matrix` (shared with the MoG study's
``8_precompute_emd_matrix.py``). Only needed for the *aligned* pretraining arm — the
unaligned arm trains on reconstruction alone and can skip this step.

Cost is O(N^2) Sinkhorn calls: ~1 h for 20k clouds on an RTX 4070 Ti, and the matrix is
4*N^2 bytes (1.6 GB at N=20k). Both scale quadratically, so ``--max_samples`` is the knob
to reach for before the dataset gets large. Resumable at row-block granularity.

    python experiments/benchmark/pretrain/2_precompute_emd.py --env odd_even

Writes ``<data_dir>/<env>/emd_{train,eval}.npy`` plus a committed ``.json`` sidecar
recording the metric settings — the provenance for whichever checkpoint was aligned
against it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from set_transformer.emd_matrix import (
    DEFAULT_BLUR,
    compute_matrix,
    matrix_stats,
    write_sidecar,
)

DEFAULT_DATA = Path("experiments/benchmark/pretrain/data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", choices=["train", "eval", "both"], default="both")
    ap.add_argument("--blur", type=float, default=DEFAULT_BLUR,
                    help="Sinkhorn eps; must match the pretraining reconstruction loss")
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--pair_chunk", type=int, default=4096)
    ap.add_argument("--max_samples", type=int, default=0,
                    help="truncate the split to the first K clouds (0 = all); cost is "
                         "quadratic, so this is the knob for large datasets")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--no_verify", action="store_true")
    args = ap.parse_args()

    env_dir = args.data_dir / args.env
    splits = ["train", "eval"] if args.split == "both" else [args.split]
    for split in splits:
        points_path = env_dir / f"{split}.points.npy"
        if not points_path.exists():
            raise SystemExit(f"{points_path} not found — run 1_collect_pf_dataset.py first")
        points = np.load(points_path)
        if args.max_samples:
            points = points[:args.max_samples]
        out_path = env_dir / f"emd_{split}.npy"
        print(f"\n=== {args.env}/{split}: {points.shape} -> {out_path} "
              f"({len(points) ** 2 * 4 / 1e9:.2f} GB) ===", flush=True)
        t0 = time.time()
        matrix = compute_matrix(
            points, out_path, args.blur, args.p, args.block, args.pair_chunk,
            args.device, resume=not args.no_resume, verify=not args.no_verify,
        )
        stats = matrix_stats(matrix)
        stats["wall_seconds"] = round(time.time() - t0, 1)
        write_sidecar(out_path, points_path, len(points), args.blur, args.p, stats)
        print(json.dumps(stats, indent=2), flush=True)
        if stats["offdiag_std"] < 1e-6:
            print("WARNING: pairwise distances are nearly constant — the alignment "
                  "target carries no signal on this env.", flush=True)


if __name__ == "__main__":
    main()
