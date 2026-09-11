"""Generate the variable-separability MoG splits.

Each set draws n ~ Uniform[1,10] components AND a separability s ~ Uniform[0,1]: s=0 is the
original unbiased overlapping mix, s=1 is maximally separated tight blobs within the fixed
[-3,3] domain. Written to a *separate* data dir so the baseline (overlapping-only) data,
runs, and figures are never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from set_transformer.data.mixture_of_gaussians import generate_mog_dataset_separability

DEFAULT_OUT = Path("experiments/mog/data_varsep")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_samples", type=int, default=20000)
    p.add_argument("--eval_samples", type=int, default=2000)
    p.add_argument("--num_particles", type=int, default=100)
    p.add_argument("--n_min", type=int, default=1)
    p.add_argument("--n_max", type=int, default=10)
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--eval_seed", type=int, default=1)
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tr_pts, tr_n, tr_s = generate_mog_dataset_separability(
        args.train_samples, args.num_particles, args.train_seed, args.n_min, args.n_max)
    ev_pts, ev_n, ev_s = generate_mog_dataset_separability(
        args.eval_samples, args.num_particles, args.eval_seed, args.n_min, args.n_max)

    np.savez(args.out_dir / "train.npz", points=tr_pts, n_components=tr_n, separability=tr_s)
    np.save(args.out_dir / "train.points.npy", tr_pts)
    np.savez(args.out_dir / "eval.npz", points=ev_pts, n_components=ev_n, separability=ev_s)
    np.save(args.out_dir / "eval.points.npy", ev_pts)
    print(f"train {tr_pts.shape}, eval {ev_pts.shape} -> {args.out_dir}")
    print(f"separability (train): min {tr_s.min():.2f} mean {tr_s.mean():.2f} max {tr_s.max():.2f}")


if __name__ == "__main__":
    main()
