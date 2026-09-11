"""Evaluate latent metric-alignment: does the latent geometry mirror the EMD geometry?

Compares alignment-trained runs against their unaligned counterparts on a held-out split,
scoring every (arm, method, seed) on:

    pearson_r     linear agreement between latent and EMD pairwise distances (the
                  training objective's own metric, measured out-of-sample)
    spearman_rho  monotone agreement — a curved-but-monotone scatter scores well here and
                  badly on Pearson, which is exactly the Phase-2 trigger condition
    knn_overlap   fraction of each cloud's k EMD-nearest neighbours that are also among
                  its k latent-nearest neighbours (does the *local* structure survive)
    val_emd       reconstruction quality, to price what alignment costs

    python experiments/mog/9_eval_alignment.py \
        --aligned_runs experiments/mog/runs_align \
        --baseline_runs experiments/mog/runs_varsep_sinkhorn

Writes figures + a tidy CSV to --fig_dir.
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

from set_transformer.latent_alignment import latent_pairwise_distances, pearson_r

from _common import (
    METHOD_COLORS,
    METHOD_REGISTRY,
    build_model,
    emd_pairs_from_matrix,
    encode_all,
    load_emd_matrix,
)

FIG_DIR_DEFAULT = Path("experiments/mog/figures_align")
ARM_STYLE = {"unaligned": ("//", 0.55), "aligned": ("", 1.0)}


def _iter_runs(runs_dir: Path, method: str):
    for run in sorted((runs_dir / method).glob("seed*")):
        ckpt_path = run / "checkpoint_best.pt"
        if ckpt_path.exists():
            yield run, torch.load(ckpt_path, map_location="cpu", weights_only=False)


def _load_model(ckpt, method: str, device: str):
    model = build_model(method)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def spearman_rho(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of the ranks. Ties are vanishingly unlikely on float
    distances, so plain argsort ranks are used rather than average ranks."""
    def ranks(x):
        order = torch.argsort(x)
        out = torch.empty_like(x)
        out[order] = torch.arange(len(x), device=x.device, dtype=x.dtype)
        return out
    r = pearson_r(ranks(a), ranks(b))
    return float("nan") if r is None else float(r)


def knn_overlap(d_latent_sq: torch.Tensor, d_emd_sq: torch.Tensor, k: int) -> float:
    """Mean |kNN(latent) ∩ kNN(emd)| / k over samples (self excluded)."""
    big = torch.finfo(d_latent_sq.dtype).max
    a = d_latent_sq.clone().fill_diagonal_(big)
    b = d_emd_sq.clone().fill_diagonal_(big)
    ka = a.topk(k, dim=1, largest=False).indices
    kb = b.topk(k, dim=1, largest=False).indices
    hits = 0
    for i in range(len(ka)):
        hits += len(set(ka[i].tolist()) & set(kb[i].tolist()))
    return hits / (len(ka) * k)


def score_run(model, points: torch.Tensor, emd_sq: torch.Tensor, emd_pairs: torch.Tensor,
              metric: str, device: str, is_vae: bool, k: int):
    z = encode_all(model, points, device)
    d_latent_sq = _full_latent_matrix(z, metric)
    d_latent = latent_pairwise_distances(z, metric)
    r = pearson_r(d_latent, emd_pairs.to(d_latent.device))
    return {
        "pearson_r": float("nan") if r is None else float(r),
        "spearman_rho": spearman_rho(d_latent, emd_pairs.to(d_latent.device)),
        f"knn_overlap@{k}": knn_overlap(d_latent_sq, emd_sq.to(device), k),
    }, d_latent.cpu().numpy()


def _full_latent_matrix(z: torch.Tensor, metric: str) -> torch.Tensor:
    flat = z.reshape(z.shape[0], -1)
    if metric == "cosine":
        normed = torch.nn.functional.normalize(flat, dim=1, eps=1e-8)
        return 1.0 - normed @ normed.t()
    return torch.cdist(flat, flat, p=2)


def _boot_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return float(x.mean()), float(x.mean()), float(x.mean())
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def plot_scatter(scatters, emd_pairs_np, fig_path: Path, max_points: int = 200_000):
    """d_latent vs d_emd on held-out pairs. Linear cloud = working; monotone-but-curved =
    Phase 2 (Spearman); shapeless blob = lambda too small or a bug."""
    if not scatters:
        return
    rng = np.random.default_rng(0)
    sel = rng.choice(len(emd_pairs_np), size=min(max_points, len(emd_pairs_np)),
                     replace=False)
    x = emd_pairs_np[sel]
    ncol = len(scatters)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4.0), squeeze=False)
    for ax, ((arm, method), d) in zip(axes[0], sorted(scatters.items())):
        ax.hexbin(x, d[sel], gridsize=70, bins="log", cmap="viridis")
        r = np.corrcoef(x, d[sel])[0, 1]
        ax.set_title(f"{METHOD_REGISTRY[method].label} — {arm}\nr = {r:.3f}")
        ax.set_xlabel("EMD (debiased Sinkhorn)")
        ax.set_ylabel("latent distance")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_summary(rows, metrics, fig_path: Path):
    """Grouped bars: aligned vs unaligned per method, one panel per metric."""
    methods = sorted({r["method"] for r in rows}, key=lambda m: list(METHOD_REGISTRY).index(m))
    arms = ["unaligned", "aligned"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2), squeeze=False)
    width = 0.36
    for ax, metric in zip(axes[0], metrics):
        for ai, arm in enumerate(arms):
            xs, ys, los, his = [], [], [], []
            for mi, method in enumerate(methods):
                vals = [r[metric] for r in rows
                        if r["method"] == method and r["arm"] == arm and np.isfinite(r[metric])]
                if not vals:
                    continue
                mean, lo, hi = _boot_ci(vals)
                xs.append(mi + (ai - 0.5) * width)
                ys.append(mean)
                los.append(mean - lo)
                his.append(hi - mean)
            if not xs:
                continue
            hatch, alpha = ARM_STYLE[arm]
            ax.bar(xs, ys, width=width, yerr=[los, his], capsize=3, hatch=hatch,
                   alpha=alpha, label=arm,
                   color=[METHOD_COLORS[methods[int(round(x))]] for x in xs])
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_REGISTRY[m].label for m in methods])
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(loc="upper right")
        ax.margins(y=0.28)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_curves(runs_dir: Path, methods, fig_path: Path):
    """Held-out correlation and val EMD across epochs, with the lambda ramp overlaid."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plotted = False
    for method in methods:
        curves_r, curves_emd, lam = [], [], None
        for run, _ in _iter_runs(runs_dir, method):
            h = np.load(run / "history.npz")
            if "val_r" not in h:
                continue
            curves_r.append(h["val_r"])
            curves_emd.append(h["val_emd"])
            lam = h["align_lambda"] if "align_lambda" in h else None
        if not curves_r:
            continue
        plotted = True
        L = min(len(c) for c in curves_r)
        ep = np.arange(L)
        for ax, curves, ylabel in ((axes[0], curves_r, "held-out Pearson r"),
                                   (axes[1], curves_emd, "val EMD")):
            arr = np.stack([c[:L] for c in curves])
            m = arr.mean(0)
            ax.plot(ep, m, color=METHOD_COLORS[method], label=METHOD_REGISTRY[method].label)
            if len(arr) > 1:
                sd = arr.std(0) / np.sqrt(len(arr))
                ax.fill_between(ep, m - 1.96 * sd, m + 1.96 * sd,
                                color=METHOD_COLORS[method], alpha=0.2)
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabel)
        if lam is not None and lam.max() > 0:
            ax2 = axes[0].twinx()
            ax2.plot(ep, lam[:L], color="grey", ls=":", lw=1)
            ax2.set_ylabel("lambda", color="grey")
    if not plotted:
        plt.close(fig)
        return
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_runs", type=Path, default=Path("experiments/mog/runs_align"))
    ap.add_argument("--baseline_runs", type=Path,
                    default=Path("experiments/mog/runs_varsep_sinkhorn"))
    ap.add_argument("--data_dir", type=Path, default=Path("experiments/mog/data_varsep"))
    ap.add_argument("--methods", nargs="+", default=["st_ae", "ds_ae"])
    ap.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine",
                    help="latent distance used for scoring; match the training config")
    ap.add_argument("--k", type=int, default=10, help="k for the kNN-overlap metric")
    ap.add_argument("--max_eval", type=int, default=2000,
                    help="cap on held-out clouds scored (must be <= the eval EMD matrix)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fig_dir", type=Path, default=FIG_DIR_DEFAULT)
    args = ap.parse_args()

    eval_pts = np.load(args.data_dir / "eval.points.npy")
    emd_full = load_emd_matrix(args.data_dir / "emd_eval.npy", len(eval_pts))
    n = min(args.max_eval, len(eval_pts))
    points = torch.from_numpy(eval_pts[:n]).float()
    emd_sq = torch.from_numpy(np.array(emd_full[:n, :n], dtype=np.float32))
    emd_pairs = emd_pairs_from_matrix(emd_sq.numpy()).to(args.device)
    emd_pairs_np = emd_pairs.cpu().numpy()

    rows, scatters = [], {}
    for arm, runs_dir in (("unaligned", args.baseline_runs), ("aligned", args.aligned_runs)):
        if not runs_dir.exists():
            print(f"skipping {arm}: {runs_dir} not found", flush=True)
            continue
        for method in args.methods:
            best_val, best_d = float("inf"), None
            for run, ckpt in _iter_runs(runs_dir, method):
                model = _load_model(ckpt, method, args.device)
                scores, d_latent = score_run(
                    model, points, emd_sq, emd_pairs, args.metric, args.device,
                    METHOD_REGISTRY[method].is_vae, args.k)
                scores.update(arm=arm, method=method, seed=ckpt.get("seed", -1),
                              val_emd=float(ckpt["best_val_emd"]))
                rows.append(scores)
                print(f"{arm:10s} {method:6s} seed{scores['seed']}: "
                      f"r={scores['pearson_r']:.3f} rho={scores['spearman_rho']:.3f} "
                      f"knn@{args.k}={scores[f'knn_overlap@{args.k}']:.3f} "
                      f"val_emd={scores['val_emd']:.4f}", flush=True)
                if scores["val_emd"] < best_val:
                    best_val, best_d = scores["val_emd"], d_latent
            if best_d is not None:
                scatters[(arm, method)] = best_d

    if not rows:
        raise SystemExit("no runs found — train something first")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["pearson_r", "spearman_rho", f"knn_overlap@{args.k}", "val_emd"]
    with open(args.fig_dir / "alignment_scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "method", "seed"] + metrics)
        w.writeheader()
        w.writerows(rows)

    with open(args.fig_dir / "alignment_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "method", "metric", "mean", "ci_lo", "ci_hi", "n_seeds"])
        for arm in ("unaligned", "aligned"):
            for method in args.methods:
                vals = [r for r in rows if r["arm"] == arm and r["method"] == method]
                for metric in metrics:
                    xs = [v[metric] for v in vals if np.isfinite(v[metric])]
                    if xs:
                        w.writerow([arm, method, metric, *(f"{v:.5f}" for v in _boot_ci(xs)),
                                    len(xs)])

    plot_scatter(scatters, emd_pairs_np, args.fig_dir / "alignment_scatter.png")
    plot_summary(rows, metrics, args.fig_dir / "alignment_summary.png")
    plot_curves(args.aligned_runs, args.methods, args.fig_dir / "alignment_curves.png")
    print(f"\nwrote figures + CSVs to {args.fig_dir}", flush=True)


if __name__ == "__main__":
    main()
