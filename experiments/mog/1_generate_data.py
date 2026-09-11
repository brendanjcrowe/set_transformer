"""Generate train + eval mixture-of-Gaussians splits for the MoG encoder study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from set_transformer.data.mixture_of_gaussians import generate_mog_dataset

from _common import DATA_DIR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_samples", type=int, default=20000)
    p.add_argument("--eval_samples", type=int, default=2000)
    p.add_argument("--num_particles", type=int, default=100)
    p.add_argument("--n_min", type=int, default=1)
    p.add_argument("--n_max", type=int, default=10)
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--eval_seed", type=int, default=1)
    p.add_argument("--out_dir", type=Path, default=DATA_DIR)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_pts, train_n = generate_mog_dataset(
        args.train_samples, args.num_particles, args.train_seed, args.n_min, args.n_max
    )
    eval_pts, eval_n = generate_mog_dataset(
        args.eval_samples, args.num_particles, args.eval_seed, args.n_min, args.n_max
    )
    np.savez(args.out_dir / "train.npz", points=train_pts, n_components=train_n)
    np.save(args.out_dir / "train.points.npy", train_pts)
    np.savez(args.out_dir / "eval.npz", points=eval_pts, n_components=eval_n)
    np.save(args.out_dir / "eval.points.npy", eval_pts)
    print(f"train {train_pts.shape}, eval {eval_pts.shape} -> {args.out_dir}")
    print(f"n_components histogram (eval): "
          f"{dict(zip(*np.unique(eval_n, return_counts=True)))}")


if __name__ == "__main__":
    main()
