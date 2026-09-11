"""Separability stress test: reconstruct well-separated MoG inputs with the frozen
best-of-seed models (no retraining), for n = 2..6 Gaussians.

The training/eval distribution has overlapping components; here every component is a
tight, well-separated blob (enforced minimum inter-mode distance), so the figure shows
whether each encoder preserves distinct modes or smears them together.

    python experiments/mog/4_inference_separable.py
    -> figures/reconstruction_separable.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

from set_transformer.data.mixture_of_gaussians import sample_mog_separable

from _common import ARCH, FIG_DIR, RUNS_DIR

# Reuse the model loader + grid renderer from the plotting script.
plot_mod = __import__("3_plot")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=Path, default=RUNS_DIR)
    p.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    p.add_argument("--n_min", type=int, default=2)
    p.add_argument("--n_max", type=int, default=6)
    p.add_argument("--min_sep", type=float, default=2.0)
    # Cap the mean domain to the training range [-3, 3] so this isolates separability
    # from any scale shift; requires a small enough min_sep to pack n_max blobs.
    p.add_argument("--box_half", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    examples = {
        n: sample_mog_separable(
            ARCH["num_particles"], n, rng,
            min_sep=args.min_sep, std_range=(0.1, 0.25), box_half=args.box_half,
        )
        for n in range(args.n_min, args.n_max + 1)
    }

    models = plot_mod._best_run_per_method(args.runs_dir)
    if not models:
        raise SystemExit(f"No trained models under {args.runs_dir}")

    plot_mod.render_reconstruction_grid(
        models,
        examples,
        args.fig_dir / "reconstruction_separable.png",
        "MoG reconstruction — separable components (frozen best models), by #Gaussians",
    )


if __name__ == "__main__":
    main()
