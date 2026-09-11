"""Figures for the MoG encoder study.

    curves_val_emd.png   validation EMD vs epoch, mean +/- 95% bootstrap CI over seeds,
                         one line per method (the learning curve averaged over runs)
    reconstruction.png   rows n=1..n_max, cols = Input + one per method's best-of-seeds
                         model; one example set per component count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import (
    ARCH,
    DATA_DIR,
    FIG_DIR,
    METHOD_COLORS,
    METHOD_ORDER,
    METHOD_REGISTRY,
    RUNS_DIR,
    build_model,
)


def _bootstrap_band(values, n_boot=2000, alpha=0.05, seed=0):
    """values: (n_seeds, n_epochs) -> (mean, lo, hi) each (n_epochs,)."""
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    mean = values.mean(axis=0)
    if n < 2:
        return mean, mean, mean
    boot = np.stack([values[rng.integers(0, n, n)].mean(axis=0) for _ in range(n_boot)])
    return mean, np.quantile(boot, alpha / 2, axis=0), np.quantile(boot, 1 - alpha / 2, axis=0)


def load_histories(runs_dir: Path):
    """method -> (n_seeds, n_epochs) val_emd array (seeds truncated to common length)."""
    out = {}
    for method in METHOD_ORDER:
        curves, epochs = [], None
        for run in sorted((runs_dir / method).glob("seed*")):
            h = np.load(run / "history.npz")
            curves.append(h["val_emd"])
            epochs = h["epoch"]
        if curves:
            L = min(len(c) for c in curves)
            out[method] = (epochs[:L], np.stack([c[:L] for c in curves]))
    return out


def plot_curves(histories, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in METHOD_ORDER:
        if method not in histories:
            continue
        epochs, values = histories[method]
        mean, lo, hi = _bootstrap_band(values)
        color = METHOD_COLORS[method]
        label = f"{METHOD_REGISTRY[method].label} (n={values.shape[0]})"
        ax.plot(epochs, mean, label=label, color=color, lw=1.9)
        if values.shape[0] > 1:
            ax.fill_between(epochs, lo, hi, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel("validation EMD (lower is better)", fontsize=11)
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=10, frameon=False)
    fig.suptitle("MoG reconstruction — learning curves (mean ± 95% CI over seeds)", fontsize=12)
    _save(fig, out_path)


def _best_run_per_method(runs_dir: Path):
    """method -> loaded eval-mode model from its lowest-val-EMD seed."""
    models = {}
    for method in METHOD_ORDER:
        runs = sorted((runs_dir / method).glob("seed*"))
        if not runs:
            continue
        ckpts = [torch.load(r / "checkpoint_best.pt", map_location="cpu", weights_only=False)
                 for r in runs]
        best = min(ckpts, key=lambda c: c["best_val_emd"])
        model = build_model(method)
        model.load_state_dict(best["state_dict"])
        model.eval()
        models[method] = (model, best["seed"], best["best_val_emd"])
    return models


def _recon(model, x):
    # Follow the model's device rather than assuming CPU: callers may hand us a model
    # already placed on the GPU for other work in the same script.
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model(x.to(device))
    return (out["recon"] if isinstance(out, dict) else out).cpu()


def _examples_per_n(eval_pts, eval_n, n_max):
    """One example set per component count 1..n_max (first eval sample with that n)."""
    ex = {}
    for n in range(1, n_max + 1):
        idx = np.where(eval_n == n)[0]
        if len(idx):
            ex[n] = eval_pts[idx[0]]
    return ex


def _square_bbox(arrays, pad=0.4):
    stacked = np.concatenate([a.reshape(-1, 2) for a in arrays], axis=0)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    c = (lo + hi) / 2
    half = max(hi - lo) / 2 + pad
    return (c[0] - half, c[0] + half), (c[1] - half, c[1] + half)


def plot_reconstruction(models, eval_pts, eval_n, out_path: Path, n_max=10):
    examples = _examples_per_n(eval_pts, eval_n, n_max)
    render_reconstruction_grid(
        models, examples, out_path,
        "MoG reconstruction — best model per method, by #Gaussians",
    )


def render_reconstruction_grid(models, examples, out_path: Path, title: str):
    """Grid: rows = the n values in ``examples`` (dict {n: (num_particles,2)}),
    cols = Input + one per method's loaded best model."""
    columns = [("Input", None)] + [
        (METHOD_REGISTRY[m].label, models[m][0]) for m in METHOD_ORDER if m in models
    ]
    render_model_grid(columns, examples, out_path, title)


def render_model_grid(columns, examples, out_path: Path, title: str,
                      row_label=lambda k: f"n={k}"):
    """Generic point-set grid.

    Args:
        columns: list of ``(label, model_or_None)``. ``None`` plots the raw input, any
            model plots its reconstruction of that input. Arbitrary column sets (e.g.
            the same method aligned vs. unaligned) are the reason this is separate from
            ``render_reconstruction_grid``.
        examples: ``{row_key: (num_particles, 2)}``.
    """
    rows = sorted(examples)

    # Reconstruct every cell first so the bounding box covers inputs + recons.
    cells = {}
    for k, pts in examples.items():
        x = torch.from_numpy(np.asarray(pts, dtype=np.float32)).unsqueeze(0)
        for label, model in columns:
            cells[(label, k)] = pts if model is None else _recon(model, x)[0].numpy()
    xlim, ylim = _square_bbox(list(cells.values()))

    fig, axes = plt.subplots(
        len(rows), len(columns),
        figsize=(len(columns) * 1.9, len(rows) * 1.9),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    axes = np.atleast_2d(axes)
    for r, k in enumerate(rows):
        for c, (label, _) in enumerate(columns):
            ax = axes[r, c]
            pts = cells[(label, k)]
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.7)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(label, fontsize=11, pad=4)
            if c == 0:
                ax.set_ylabel(row_label(k), fontsize=10)
    # Two-line column headers need extra headroom or they collide with the suptitle.
    two_line = any("\n" in label for label, _ in columns)
    fig.suptitle(title, fontsize=13, y=1.02 if two_line else 0.995)
    fig.subplots_adjust(top=0.90 if two_line else 0.95, bottom=0.04, left=0.06, right=0.99)
    _save(fig, out_path)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=Path, default=RUNS_DIR)
    p.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    p.add_argument("--data_dir", type=Path, default=DATA_DIR)
    p.add_argument("--n_max", type=int, default=10)
    args = p.parse_args()

    histories = load_histories(args.runs_dir)
    if not histories:
        raise SystemExit(f"No run histories under {args.runs_dir}")
    plot_curves(histories, args.fig_dir / "curves_val_emd.png")

    eval_npz = np.load(args.data_dir / "eval.npz")
    models = _best_run_per_method(args.runs_dir)
    for m, (_, seed, emd) in models.items():
        print(f"  best {METHOD_REGISTRY[m].label}: seed{seed} (val EMD {emd:.4f})")
    plot_reconstruction(models, eval_npz["points"], eval_npz["n_components"],
                        args.fig_dir / "reconstruction.png", args.n_max)


if __name__ == "__main__":
    main()
