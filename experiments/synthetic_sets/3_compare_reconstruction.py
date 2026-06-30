"""Headline figures F1 + F2 for the VAE / VQ-VAE comparison.

F1 — per-family reconstruction grid (input vs vanilla vs VAE-mean vs VAE-sample vs VQ-VAE)
F2 — per-family metrics chart (EMD + Chamfer with 95% bootstrap CIs)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from set_transformer.data.synthetic_sets import FAMILY_NAMES
from set_transformer.loss import EarthMoverDistanceLoss
from set_transformer.models import PFSetTransformer, SetVAE, SetVQVAE

from _common import DATA_DIR

FIG_DIR = Path("experiments/synthetic_sets/figures")
MODEL_COLORS = {"Vanilla": "tab:blue", "VAE": "tab:orange", "VQ-VAE": "tab:green"}


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    common = dict(
        num_particles=cfg.num_particles,
        dim_particles=cfg.dim_particles,
        num_encodings=cfg.num_encodings,
        dim_encoder=cfg.dim_encoder,
        num_inds=cfg.num_inds,
        dim_hidden=cfg.dim_hidden,
        num_heads=cfg.num_heads,
        ln=cfg.use_layer_norm,
    )
    if cfg.model_type == "pf_st":
        model = PFSetTransformer(**common)
    elif cfg.model_type == "set_vae":
        model = SetVAE(**common)
    elif cfg.model_type == "set_vqvae":
        model = SetVQVAE(
            codebook_size=cfg.codebook_size,
            commitment_weight=cfg.commitment_weight,
            ema_decay=cfg.ema_decay,
            **common,
        )
    else:
        raise ValueError(f"unknown model_type: {cfg.model_type}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, cfg


def model_recon(model, x: torch.Tensor, mode: str = "default") -> torch.Tensor:
    """Return reconstruction tensor (B, N, D), regardless of model output shape."""
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        if mode == "vae_sample" and "mu" in out:
            std = (0.5 * out["logvar"]).exp()
            z = out["mu"] + std * torch.randn_like(std)
            return model.decoder(z)
        return out["recon"]
    return out


def pick_examples(
    eval_pts: np.ndarray, eval_ids: np.ndarray
) -> Dict[str, np.ndarray]:
    """For each family, return one example point set (deterministic by family)."""
    examples = {}
    for fam_idx, fam in enumerate(FAMILY_NAMES):
        idx = np.where(eval_ids == fam_idx)[0]
        if len(idx) == 0:
            continue
        examples[fam] = eval_pts[idx[0]]
    return examples


def _global_square_bbox(arrays, pad: float = 0.4):
    """Return (xlim, ylim) of a square bounding box containing every array."""
    stacked = np.concatenate([a.reshape(-1, 2) for a in arrays], axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 + pad
    return (cx - half, cx + half), (cy - half, cy + half)


def plot_reconstruction_grid(
    examples: Dict[str, np.ndarray],
    recons: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    column_names = ["Input", "Vanilla", "VAE (mean)", "VAE (sample)", "VQ-VAE"]
    rows = list(examples.keys())

    # One global square bounding box so every cell has identical extent and
    # uniform visual size under set_aspect("equal").
    all_arrays = list(examples.values())
    for col in recons.values():
        all_arrays.extend(col.values())
    xlim, ylim = _global_square_bbox(all_arrays)

    fig, axes = plt.subplots(
        len(rows),
        len(column_names),
        figsize=(len(column_names) * 2.0, len(rows) * 2.0),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if len(rows) == 1:
        axes = axes[np.newaxis, :]
    for r, fam in enumerate(rows):
        cells = [
            examples[fam],
            recons["Vanilla"][fam],
            recons["VAE (mean)"][fam],
            recons["VAE (sample)"][fam],
            recons["VQ-VAE"][fam],
        ]
        for c, (pts, col_name) in enumerate(zip(cells, column_names)):
            ax = axes[r, c]
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.7)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(axis="both", which="both", labelsize=7)
            if r == 0:
                ax.set_title(col_name, fontsize=11, pad=4)
            if c == 0:
                ax.set_ylabel(fam, fontsize=10)

    # Outer axis labels (only on the outside edges of the grid).
    for ax in axes[-1, :]:
        ax.set_xlabel("x", fontsize=9)
    # ylabel on column 0 already shows family; add y-axis hint on the bottom-left.
    axes[-1, 0].set_ylabel(f"{rows[-1]}\n(y)", fontsize=10)

    fig.suptitle("F1 — reconstruction grid (per family)", fontsize=14)
    # Manually reserve room for suptitle so column titles never collide.
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.98)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print(f"Wrote {out_path}")


def _bootstrap_mean_ci(
    x: np.ndarray, n_bootstrap: int = 500, alpha: float = 0.05, rng=None
):
    if rng is None:
        rng = np.random.default_rng(0)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=len(x))
        means.append(x[idx].mean())
    means = np.asarray(means)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(
        np.quantile(means, 1 - alpha / 2)
    )


def compute_metrics(
    models: Dict[str, torch.nn.Module],
    eval_pts: np.ndarray,
    eval_ids: np.ndarray,
    device: str,
    batch_size: int = 32,
) -> Tuple[
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]],
]:
    """Returns per_sample (model -> metric -> family -> array of values) and
    summary (model -> metric -> family -> (mean, lo, hi)).
    """
    emd = EarthMoverDistanceLoss(reduction="none")

    def per_sample_chamfer(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred, target: (B, N, D); returns per-sample symmetric Chamfer (B,).
        d = torch.cdist(pred, target)
        return d.min(dim=2)[0].mean(dim=1) + d.min(dim=1)[0].mean(dim=1)

    metric_fns = {"EMD": emd, "Chamfer": per_sample_chamfer}
    per_sample: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        m: {k: {f: [] for f in FAMILY_NAMES} for k in metric_fns} for m in models
    }

    pts = torch.from_numpy(eval_pts)
    for start in range(0, len(pts), batch_size):
        end = min(start + batch_size, len(pts))
        x = pts[start:end].to(device)
        ids = eval_ids[start:end]
        for model_name, model in models.items():
            recon = model_recon(model, x)
            for metric_name, fn in metric_fns.items():
                vals = fn(recon, x).detach().cpu().numpy()
                for i, fid in enumerate(ids):
                    per_sample[model_name][metric_name][FAMILY_NAMES[fid]].append(
                        float(vals[i])
                    )

    rng = np.random.default_rng(0)
    summary: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]] = {}
    arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for model_name in models:
        summary[model_name] = {}
        arrays[model_name] = {}
        for metric_name in metric_fns:
            summary[model_name][metric_name] = {}
            arrays[model_name][metric_name] = {}
            for fam in FAMILY_NAMES:
                arr = np.asarray(
                    per_sample[model_name][metric_name][fam], dtype=np.float32
                )
                arrays[model_name][metric_name][fam] = arr
                summary[model_name][metric_name][fam] = _bootstrap_mean_ci(arr, rng=rng)
    return arrays, summary


def plot_metrics(
    summary: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]],
    out_path: Path,
) -> None:
    metric_names = ["EMD", "Chamfer"]
    model_names = list(summary.keys())
    n_fam = len(FAMILY_NAMES)
    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(max(9, n_fam * 1.1), 6.5),
        constrained_layout=True,
        sharex=True,
    )
    if len(metric_names) == 1:
        axes = [axes]
    width = 0.8 / len(model_names)
    x = np.arange(n_fam)
    for ax, metric in zip(axes, metric_names):
        ax.set_title(metric, fontsize=11, pad=4)
        for m_i, model in enumerate(model_names):
            means, lo, hi = [], [], []
            for fam in FAMILY_NAMES:
                mu, l, h = summary[model][metric][fam]
                means.append(mu)
                lo.append(mu - l)
                hi.append(h - mu)
            offset = (m_i - (len(model_names) - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width=width,
                yerr=[lo, hi],
                label=model,
                color=MODEL_COLORS.get(model, None),
                capsize=2,
            )
        ax.set_ylabel(f"{metric} (lower is better)", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=9, frameon=False)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(FAMILY_NAMES, rotation=30, ha="right", fontsize=9)
    axes[-1].set_xlabel("Distribution family", fontsize=10)
    fig.suptitle(
        "F2 — per-family reconstruction metrics (95% bootstrap CI)",
        fontsize=13,
        y=1.02,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def dump_metrics_csv(
    summary: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]],
    arrays: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "family", "mean", "ci_lo", "ci_hi", "n"])
        for model in summary:
            for metric in summary[model]:
                for fam in FAMILY_NAMES:
                    mu, lo, hi = summary[model][metric][fam]
                    n = len(arrays[model][metric][fam])
                    w.writerow([model, metric, fam, mu, lo, hi, n])
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=Path, required=True)
    parser.add_argument("--vae_ckpt", type=Path, required=True)
    parser.add_argument("--vqvae_ckpt", type=Path, required=True)
    parser.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    eval_npz = np.load(DATA_DIR / "eval.npz")
    eval_pts, eval_ids = eval_npz["points"], eval_npz["family_ids"]

    baseline, _ = load_model(args.baseline_ckpt, args.device)
    vae, _ = load_model(args.vae_ckpt, args.device)
    vqvae, _ = load_model(args.vqvae_ckpt, args.device)
    models = {"Vanilla": baseline, "VAE": vae, "VQ-VAE": vqvae}

    # F2 — metrics across the entire eval split.
    arrays, summary = compute_metrics(models, eval_pts, eval_ids, args.device)
    plot_metrics(summary, args.fig_dir / "F2_metrics.png")
    dump_metrics_csv(summary, arrays, args.fig_dir / "F2_metrics.csv")

    # F1 — one example per family, all models side-by-side.
    examples = pick_examples(eval_pts, eval_ids)
    recons: Dict[str, Dict[str, np.ndarray]] = {
        "Vanilla": {},
        "VAE (mean)": {},
        "VAE (sample)": {},
        "VQ-VAE": {},
    }
    with torch.no_grad():
        for fam, pts in examples.items():
            x = torch.from_numpy(pts).unsqueeze(0).to(args.device)
            recons["Vanilla"][fam] = model_recon(baseline, x)[0].cpu().numpy()
            recons["VAE (mean)"][fam] = model_recon(vae, x)[0].cpu().numpy()
            recons["VAE (sample)"][fam] = (
                model_recon(vae, x, mode="vae_sample")[0].cpu().numpy()
            )
            recons["VQ-VAE"][fam] = model_recon(vqvae, x)[0].cpu().numpy()
    plot_reconstruction_grid(examples, recons, args.fig_dir / "F1_reconstruction.png")


if __name__ == "__main__":
    main()
