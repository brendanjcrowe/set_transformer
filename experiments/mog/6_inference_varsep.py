"""Reconstruct high-separability inputs with the variable-separability-trained models.

Answers "can the encoder learn separable sets when the training distribution contains
them?" — generates maximally separable inputs (separability=1) for n=2..6 in the fixed
[-3,3] domain and reconstructs with each method's best-of-seed model from the
variable-separability sweep.

    python experiments/mog/6_inference_varsep.py
    -> figures_varsep/reconstruction_sep1.png
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

from set_transformer.data.mixture_of_gaussians import sample_mog_with_separability

from _common import ARCH

plot_mod = __import__("3_plot")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=Path, default=Path("experiments/mog/runs_varsep"))
    p.add_argument("--fig_dir", type=Path, default=Path("experiments/mog/figures_varsep"))
    p.add_argument("--n_min", type=int, default=2)
    p.add_argument("--n_max", type=int, default=6)
    p.add_argument("--separability", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    examples = {
        n: sample_mog_with_separability(ARCH["num_particles"], n, rng, args.separability)
        for n in range(args.n_min, args.n_max + 1)
    }

    models = plot_mod._best_run_per_method(args.runs_dir)
    if not models:
        raise SystemExit(f"No trained models under {args.runs_dir}")

    tag = f"{args.separability:.2f}".rstrip("0").rstrip(".")
    plot_mod.render_reconstruction_grid(
        models, examples,
        args.fig_dir / f"reconstruction_sep{tag}.png",
        f"MoG reconstruction — separability={args.separability:g} inputs "
        f"(variable-separability-trained models), by #Gaussians",
    )


if __name__ == "__main__":
    main()
