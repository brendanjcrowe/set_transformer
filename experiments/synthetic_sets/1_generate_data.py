"""Generate train + eval synthetic point-set splits and a distribution gallery figure (F3).

Defaults are sized for the smoke pipeline; override via CLI for full runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from set_transformer.data.synthetic_sets import (
    FAMILY_NAMES,
    generate_dataset,
    sample_set,
)

DEFAULT_OUT_DIR = Path("experiments/synthetic_sets/data")
DEFAULT_FIG_DIR = Path("experiments/synthetic_sets/figures")


def dump_gallery(num_per_family: int, num_particles: int, out_path: Path) -> None:
    rng = np.random.default_rng(42)
    n_rows = len(FAMILY_NAMES)

    # Sample once up front so we can compute one global square bounding box.
    sets = [[sample_set(fam, num_particles, rng) for _ in range(num_per_family)]
            for fam in FAMILY_NAMES]
    stacked = np.concatenate([p.reshape(-1, 2) for row in sets for p in row], axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 + 0.4
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    fig, axes = plt.subplots(
        n_rows,
        num_per_family,
        figsize=(num_per_family * 2.0, n_rows * 2.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    for row, fam in enumerate(FAMILY_NAMES):
        for col in range(num_per_family):
            pts = sets[row][col]
            ax = axes[row, col]
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.7)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(axis="both", which="both", labelsize=7)
            if row == 0:
                ax.set_title(f"sample {col + 1}", fontsize=10, pad=4)
            if col == 0:
                ax.set_ylabel(fam, fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("x", fontsize=9)
    axes[-1, 0].set_ylabel(f"{FAMILY_NAMES[-1]}\n(y)", fontsize=10)

    fig.suptitle("F3 — synthetic distribution gallery", fontsize=14)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.08, right=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=20000)
    parser.add_argument("--eval_samples", type=int, default=2000)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--eval_seed", type=int, default=1)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig_dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--gallery_per_family", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_pts, train_ids = generate_dataset(
        args.train_samples, args.num_particles, seed=args.train_seed
    )
    eval_pts, eval_ids = generate_dataset(
        args.eval_samples, args.num_particles, seed=args.eval_seed
    )

    np.savez(args.out_dir / "train.npz", points=train_pts, family_ids=train_ids)
    np.save(args.out_dir / "train.points.npy", train_pts)
    np.savez(args.out_dir / "eval.npz", points=eval_pts, family_ids=eval_ids)
    np.save(args.out_dir / "eval.points.npy", eval_pts)
    print(f"Wrote train {train_pts.shape} and eval {eval_pts.shape} to {args.out_dir}")

    dump_gallery(
        args.gallery_per_family,
        args.num_particles,
        args.fig_dir / "F3_distribution_gallery.png",
    )


if __name__ == "__main__":
    main()
