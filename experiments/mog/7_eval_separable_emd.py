"""Quantify "can the encoder learn separable sets?": reconstruction EMD on a held-out
maximally-separable eval set (separability=1, n=2..n_max), for the baseline (overlapping-
trained) models vs the variable-separability-trained models.

If training on separable data helps, the varsep models should score markedly lower EMD on
separable inputs than the baseline models. Writes a grouped bar chart + CSV.

    python experiments/mog/7_eval_separable_emd.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from set_transformer.data.mixture_of_gaussians import sample_mog_with_separability
from set_transformer.loss import EarthMoverDistanceLoss

from _common import ARCH, METHOD_COLORS, METHOD_ORDER, METHOD_REGISTRY, build_model

plot_mod = __import__("3_plot")


def make_separable_eval(n_samples, n_min, n_max, separability, seed):
    rng = np.random.default_rng(seed)
    ns = rng.integers(n_min, n_max + 1, size=n_samples)
    pts = np.stack([sample_mog_with_separability(ARCH["num_particles"], int(n), rng, separability)
                    for n in ns])
    return pts.astype(np.float32)


def _load_best(runs_dir: Path, method: str):
    runs = sorted((runs_dir / method).glob("seed*"))
    if not runs:
        return None
    ckpts = [torch.load(r / "checkpoint_best.pt", map_location="cpu", weights_only=False) for r in runs]
    best = min(ckpts, key=lambda c: c["best_val_emd"])
    model = build_model(method)
    model.load_state_dict(best["state_dict"])
    model.eval()
    return model


def eval_emd(model, pts, device, batch_size=64):
    emd = EarthMoverDistanceLoss(reduction="none")
    vals = []
    x_all = torch.from_numpy(pts)
    with torch.no_grad():
        for s in range(0, len(x_all), batch_size):
            x = x_all[s:s + batch_size].to(device)
            out = model(x)
            recon = out["recon"] if isinstance(out, dict) else out
            vals.append(emd(recon, x).cpu().numpy())
    return np.concatenate(vals)


def _boot_ci(x, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_runs", type=Path, default=Path("experiments/mog/runs"))
    p.add_argument("--varsep_runs", type=Path, default=Path("experiments/mog/runs_varsep"))
    p.add_argument("--baseline_label", default="baseline (overlap-trained)")
    p.add_argument("--varsep_label", default="varsep-trained")
    p.add_argument("--fig_dir", type=Path, default=Path("experiments/mog/figures_varsep"))
    p.add_argument("--out_name", default="separable_emd")
    p.add_argument("--n_samples", type=int, default=600)
    p.add_argument("--n_min", type=int, default=2)
    p.add_argument("--n_max", type=int, default=6)
    p.add_argument("--separability", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    pts = make_separable_eval(args.n_samples, args.n_min, args.n_max, args.separability, args.seed)
    sources = {args.baseline_label: args.baseline_runs,
               args.varsep_label: args.varsep_runs}

    rows = []
    summary = {}  # method -> {source: (mean, lo, hi)}
    for method in METHOD_ORDER:
        summary[method] = {}
        for src_label, runs_dir in sources.items():
            model = _load_best(runs_dir, method)
            if model is None:
                continue
            vals = eval_emd(model.to(args.device), pts, args.device)
            mean, lo, hi = _boot_ci(vals)
            summary[method][src_label] = (mean, lo, hi)
            rows.append([METHOD_REGISTRY[method].label, src_label, mean, lo, hi, len(vals)])
            print(f"{METHOD_REGISTRY[method].label:8s} {src_label:26s} "
                  f"EMD {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    with (args.fig_dir / f"{args.out_name}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "source", "emd_mean", "ci_lo", "ci_hi", "n"])
        w.writerows(rows)

    # Grouped bars: x = method, two bars (baseline vs varsep).
    src_labels = list(sources)
    methods = [m for m in METHOD_ORDER if summary[m]]
    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, src in enumerate(src_labels):
        means = [summary[m][src][0] for m in methods]
        lo = [summary[m][src][0] - summary[m][src][1] for m in methods]
        hi = [summary[m][src][2] - summary[m][src][0] for m in methods]
        hatch = None if i == 0 else "//"
        ax.bar(x + (i - 0.5) * width, means, width, yerr=[lo, hi], capsize=3,
               label=src, hatch=hatch,
               color=[METHOD_COLORS[m] for m in methods], edgecolor="black",
               alpha=0.65 if i == 0 else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_REGISTRY[m].label for m in methods])
    ax.set_ylabel("EMD on separable inputs (lower is better)", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, frameon=False, title="training")
    fig.suptitle(f"Reconstruction of separability={args.separability:g} inputs: "
                 f"{args.baseline_label} vs {args.varsep_label} (95% CI)", fontsize=11)
    out = args.fig_dir / f"{args.out_name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
