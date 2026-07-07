"""6-way comparison figures: Set Transformer × {Vanilla, VAE, VQ-VAE} versus
DeepSet × {Vanilla, VAE, VQ-VAE}.

Produces:
    F1_all_reconstruction.png — input + one column per model, one row per family
    F2_all_metrics.png        — per-family EMD/Chamfer bars for all six models
    F2_all_metrics.csv        — raw metric numbers with 95% bootstrap CIs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from set_transformer.data.synthetic_sets import FAMILY_NAMES

from _common import DATA_DIR
from importlib import import_module

# Reuse the existing loader + per-sample Chamfer/EMD helpers.
compare_mod = import_module("3_compare_reconstruction")

FIG_DIR = Path("experiments/synthetic_sets/figures")

# Six distinguishable colors, grouped so ST family reads as darker and DS as
# paler siblings.
MODEL_STYLES = [
    ("ST-Vanilla", "tab:blue"),
    ("ST-VAE", "tab:orange"),
    ("ST-VQ-VAE", "tab:green"),
    ("DS-Vanilla", "tab:cyan"),
    ("DS-VAE", "tab:red"),
    ("DS-VQ-VAE", "tab:olive"),
]


def parse_models(spec: str, device: str) -> "Dict[str, torch.nn.Module]":
    """Parse 'Label:path,Label:path,...' → ordered dict of loaded models."""
    models = {}
    for pair in spec.split(","):
        label, path = pair.split(":", 1)
        m, _ = compare_mod.load_model(Path(path), device)
        models[label] = m
    return models


def _pick_examples(eval_pts, eval_ids):
    ex = {}
    for i, fam in enumerate(FAMILY_NAMES):
        idx = np.where(eval_ids == i)[0]
        if len(idx) > 0:
            ex[fam] = eval_pts[idx[0]]
    return ex


def _per_sample_chamfer(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(pred, target)
    return d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)


def _bootstrap_mean_ci(x, n_bootstrap=500, alpha=0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_bootstrap)])
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def compute_metrics(models, eval_pts, eval_ids, device, batch_size=64):
    from set_transformer.loss import EarthMoverDistanceLoss

    emd = EarthMoverDistanceLoss(reduction="none")
    metric_fns = {"EMD": emd, "Chamfer": _per_sample_chamfer}
    per_sample = {m: {k: {f: [] for f in FAMILY_NAMES} for k in metric_fns} for m in models}

    pts = torch.from_numpy(eval_pts)
    for start in range(0, len(pts), batch_size):
        end = min(start + batch_size, len(pts))
        x = pts[start:end].to(device)
        ids = eval_ids[start:end]
        for label, model in models.items():
            recon = compare_mod.model_recon(model, x)
            for mname, fn in metric_fns.items():
                vals = fn(recon, x).detach().cpu().numpy()
                for i, fid in enumerate(ids):
                    per_sample[label][mname][FAMILY_NAMES[fid]].append(float(vals[i]))

    rng = np.random.default_rng(0)
    summary, arrays = {}, {}
    for label in models:
        summary[label] = {}
        arrays[label] = {}
        for mname in metric_fns:
            summary[label][mname] = {}
            arrays[label][mname] = {}
            for fam in FAMILY_NAMES:
                arr = np.asarray(per_sample[label][mname][fam], dtype=np.float32)
                arrays[label][mname][fam] = arr
                summary[label][mname][fam] = _bootstrap_mean_ci(arr, rng=rng)
    return arrays, summary


def _global_square_bbox(arrays, pad=0.4):
    stacked = np.concatenate([a.reshape(-1, 2) for a in arrays], axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 + pad
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_recon_grid(examples, recons, model_labels, out_path):
    column_names = ["Input"] + model_labels
    rows = list(examples.keys())
    all_arrays = list(examples.values())
    for m in model_labels:
        all_arrays.extend(recons[m].values())
    xlim, ylim = _global_square_bbox(all_arrays)

    fig, axes = plt.subplots(
        len(rows), len(column_names),
        figsize=(len(column_names) * 1.9, len(rows) * 1.9),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if len(rows) == 1:
        axes = axes[np.newaxis, :]
    for r, fam in enumerate(rows):
        cells = [examples[fam]] + [recons[m][fam] for m in model_labels]
        for c, (pts, cname) in enumerate(zip(cells, column_names)):
            ax = axes[r, c]
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.7)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(axis="both", which="both", labelsize=6)
            if r == 0:
                ax.set_title(cname, fontsize=10, pad=4)
            if c == 0:
                ax.set_ylabel(fam, fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("x", fontsize=9)
    axes[-1, 0].set_ylabel(f"{rows[-1]}\n(y)", fontsize=10)

    fig.suptitle("F1 (all encoders) — reconstruction grid", fontsize=14)
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.06, right=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_metrics(summary, model_labels, out_path):
    metric_names = ["EMD", "Chamfer"]
    n_fam = len(FAMILY_NAMES)
    fig, axes = plt.subplots(
        len(metric_names), 1,
        figsize=(max(11, n_fam * 1.3), 7),
        sharex=True,
    )
    if len(metric_names) == 1:
        axes = [axes]
    width = 0.8 / len(model_labels)
    x = np.arange(n_fam)
    color_map = dict(MODEL_STYLES)
    for ax, metric in zip(axes, metric_names):
        ax.set_title(metric, fontsize=11, pad=4)
        for m_i, label in enumerate(model_labels):
            means, lo, hi = [], [], []
            for fam in FAMILY_NAMES:
                mu, l, h = summary[label][metric][fam]
                means.append(mu)
                lo.append(mu - l)
                hi.append(h - mu)
            offset = (m_i - (len(model_labels) - 1) / 2) * width
            ax.bar(
                x + offset, means, width=width,
                yerr=[lo, hi], label=label,
                color=color_map.get(label, None),
                capsize=1.5,
            )
        ax.set_ylabel(f"{metric} (lower is better)", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(FAMILY_NAMES, rotation=30, ha="right", fontsize=9)
    axes[-1].set_xlabel("Distribution family", fontsize=10)
    fig.suptitle(
        "F2 (all encoders) — per-family reconstruction metrics (95% bootstrap CI)",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.92, bottom=0.14, left=0.07, right=0.99, hspace=0.25)
    fig.savefig(out_path, dpi=120)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"Wrote {out_path}")


def dump_csv(summary, arrays, out_path):
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "family", "mean", "ci_lo", "ci_hi", "n"])
        for label in summary:
            for metric in summary[label]:
                for fam in FAMILY_NAMES:
                    mu, lo, hi = summary[label][metric][fam]
                    n = len(arrays[label][metric][fam])
                    w.writerow([label, metric, fam, mu, lo, hi, n])
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated 'Label:ckpt_path' pairs; order defines column/bar order.",
    )
    parser.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    eval_npz = np.load(DATA_DIR / "eval.npz")
    eval_pts, eval_ids = eval_npz["points"], eval_npz["family_ids"]
    models = parse_models(args.models, args.device)
    model_labels = list(models.keys())

    arrays, summary = compute_metrics(models, eval_pts, eval_ids, args.device)
    plot_metrics(summary, model_labels, args.fig_dir / "F2_all_metrics.png")
    dump_csv(summary, arrays, args.fig_dir / "F2_all_metrics.csv")

    examples = _pick_examples(eval_pts, eval_ids)
    recons = {label: {} for label in models}
    with torch.no_grad():
        for fam, pts in examples.items():
            x = torch.from_numpy(pts).unsqueeze(0).to(args.device)
            for label, model in models.items():
                recons[label][fam] = compare_mod.model_recon(model, x)[0].cpu().numpy()
    plot_recon_grid(examples, recons, model_labels, args.fig_dir / "F1_all_reconstruction.png")


if __name__ == "__main__":
    main()
