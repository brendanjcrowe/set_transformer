"""Qualitative inference figures: alignment-trained vs. unaligned models.

Two things worth seeing, and they answer different questions.

1. **Reconstruction grids** (same format as `3_plot.py` / `6_inference_varsep.py`, with
   the columns re-cut as aligned vs. unaligned per method). The expected result is that
   the two arms look the *same* — held-out EMD is unchanged to four decimals — which is
   the whole claim: the alignment term is free. A visible degradation here would mean the
   scalar metric was hiding a qualitative cost.

2. **Latent nearest-neighbour retrieval**, which is where the arms actually differ.
   Alignment's purpose is "similar clouds get similar codes", and that is a statement
   about neighbourhoods, not about any single reconstruction. For a query cloud this
   shows the true EMD-nearest neighbours next to whatever each arm's latent space
   retrieves, so a wrong neighbour is visible as a cloud that plainly does not match.

    python experiments/mog/10_inference_alignment.py
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

from set_transformer.data.mixture_of_gaussians import sample_mog_with_separability

from _common import ARCH, METHOD_REGISTRY, build_model, encode_all, load_emd_matrix

plot_mod = __import__("3_plot")

ARMS = ("unaligned", "aligned")


def load_best(runs_dir: Path, method: str, device: str = "cpu"):
    """Lowest-val-EMD seed for a method, or None if that arm has no runs."""
    ckpts = [torch.load(r / "checkpoint_best.pt", map_location="cpu", weights_only=False)
             for r in sorted((runs_dir / method).glob("seed*"))
             if (r / "checkpoint_best.pt").exists()]
    if not ckpts:
        return None
    best = min(ckpts, key=lambda c: c["best_val_emd"])
    model = build_model(method)
    model.load_state_dict(best["state_dict"])
    return model.to(device).eval(), best["seed"], best["best_val_emd"]


def build_columns(models, methods):
    """[(label, model)] ordered method-major so the two arms sit side by side."""
    cols = [("Input", None)]
    for method in methods:
        for arm in ARMS:
            entry = models.get((arm, method))
            if entry is not None:
                cols.append((f"{METHOD_REGISTRY[method].label}\n{arm}", entry[0]))
    return cols


def _latent_neighbours(model, points, device, is_vae, metric, query_idx, k):
    z = encode_all(model, points, device)
    flat = z.reshape(len(z), -1)
    if metric == "cosine":
        normed = torch.nn.functional.normalize(flat, dim=1, eps=1e-8)
        d = 1.0 - normed @ normed.t()
    else:
        d = torch.cdist(flat, flat, p=2)
    d.fill_diagonal_(torch.finfo(d.dtype).max)
    return d[query_idx].topk(k, largest=False).indices.cpu().numpy()


def retrieval_quality(models, points_np, emd_sq, method, k, metric, device):
    """Mean true EMD of the k latent-nearest neighbours, over every held-out cloud.

    The oracle (the k EMD-nearest neighbours) is the floor; the gap to it is what
    alignment is supposed to close. More interpretable than a rank correlation: it is in
    the same units as the reconstruction metric.
    """
    points = torch.from_numpy(points_np).float()
    is_vae = METHOD_REGISTRY[method].is_vae
    emd = emd_sq.copy()
    np.fill_diagonal(emd, np.inf)
    all_idx = torch.arange(len(points_np))

    oracle = np.sort(emd, axis=1)[:, :k].mean()
    out = {"oracle": float(oracle)}
    for arm in ARMS:
        entry = models.get((arm, method))
        if entry is None:
            continue
        nn = _latent_neighbours(entry[0], points, device, is_vae, metric, all_idx, k)
        out[arm] = float(np.take_along_axis(emd, nn, axis=1).mean())
    return out


def plot_retrieval(models, points_np, emd_sq, method, query_idx, k, metric, device,
                   out_path: Path):
    """Rows = queries. Cols = query, true EMD kNN, then each arm's latent kNN."""
    points = torch.from_numpy(points_np).float()
    is_vae = METHOD_REGISTRY[method].is_vae

    emd = emd_sq.copy()
    np.fill_diagonal(emd, np.inf)
    true_nn = np.argsort(emd[query_idx], axis=1)[:, :k]

    arm_nn = {}
    for arm in ARMS:
        entry = models.get((arm, method))
        if entry is not None:
            arm_nn[arm] = _latent_neighbours(entry[0], points, device, is_vae, metric,
                                             torch.as_tensor(query_idx), k)
    if not arm_nn:
        return

    col_labels = ["query"] + [f"EMD #{i + 1}" for i in range(k)]
    for arm in arm_nn:
        col_labels += [f"{arm} #{i + 1}" for i in range(k)]

    n_rows, n_cols = len(query_idx), len(col_labels)
    fig, axes = matplotlib.pyplot.subplots(
        n_rows, n_cols, figsize=(n_cols * 1.55, n_rows * 1.7),
        sharex=True, sharey=True, gridspec_kw={"hspace": 0.06, "wspace": 0.06})
    axes = np.atleast_2d(axes)

    lim = plot_mod._square_bbox([points_np])
    for r, q in enumerate(query_idx):
        idxs = [q] + list(true_nn[r])
        colors = ["black"] + ["tab:grey"] * k
        for arm in arm_nn:
            idxs += list(arm_nn[arm][r])
            colors += [("tab:blue" if arm == "unaligned" else "tab:red")] * k
        for c, (i, col) in enumerate(zip(idxs, colors)):
            ax = axes[r, c]
            ax.scatter(points_np[i][:, 0], points_np[i][:, 1], s=5, alpha=0.7, color=col)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(lim[0])
            ax.set_ylim(lim[1])
            ax.set_xticks([])
            ax.set_yticks([])
            if c > 0:
                # EMD from the query — the number that says whether the retrieval is good.
                ax.set_xlabel(f"{emd_sq[q, i]:.1f}", fontsize=7, labelpad=1)
            if r == 0:
                ax.set_title(col_labels[c], fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(f"q{q}", fontsize=9)
    fig.suptitle(
        f"{METHOD_REGISTRY[method].label} — latent nearest neighbours vs. true EMD "
        f"neighbours (numbers = EMD to query)", fontsize=12)
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.04, right=0.99)
    plot_mod._save(fig, out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aligned_runs", type=Path,
                   default=Path("experiments/mog/runs_align_sinkhorn"))
    p.add_argument("--baseline_runs", type=Path,
                   default=Path("experiments/mog/runs_varsep_sinkhorn"))
    p.add_argument("--data_dir", type=Path, default=Path("experiments/mog/data_varsep"))
    p.add_argument("--fig_dir", type=Path, default=Path("experiments/mog/figures_align"))
    p.add_argument("--methods", nargs="+", default=["st_ae", "ds_ae"])
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--n_max", type=int, default=10)
    p.add_argument("--sep_n_min", type=int, default=2)
    p.add_argument("--sep_n_max", type=int, default=6)
    p.add_argument("--separability", type=float, default=1.0)
    p.add_argument("--k", type=int, default=3, help="neighbours shown per retrieval row")
    p.add_argument("--n_queries", type=int, default=5)
    p.add_argument("--retrieval_pool", type=int, default=2000,
                   help="held-out clouds searched (must be <= the eval EMD matrix)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    models = {}
    for arm, runs_dir in (("unaligned", args.baseline_runs), ("aligned", args.aligned_runs)):
        for method in args.methods:
            entry = load_best(runs_dir, method, args.device) if runs_dir.exists() else None
            if entry is None:
                print(f"skipping {arm}/{method}: no runs under {runs_dir}", flush=True)
                continue
            models[(arm, method)] = entry
            print(f"  {arm:10s} {METHOD_REGISTRY[method].label}: seed{entry[1]} "
                  f"(val EMD {entry[2]:.4f})", flush=True)
    if not models:
        raise SystemExit("no trained models found")

    columns = build_columns(models, args.methods)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    # (1a) held-out eval sets, one per component count
    eval_npz = np.load(args.data_dir / "eval.npz")
    examples = plot_mod._examples_per_n(
        eval_npz["points"], eval_npz["n_components"], args.n_max)
    plot_mod.render_model_grid(
        columns, examples, args.fig_dir / "reconstruction_align_vs_unaligned.png",
        "MoG reconstruction — aligned vs. unaligned, by #Gaussians (held-out)")

    # (1b) maximally separable inputs — the stress case from the varsep round
    rng = np.random.default_rng(args.seed)
    sep_examples = {
        n: sample_mog_with_separability(ARCH["num_particles"], n, rng, args.separability)
        for n in range(args.sep_n_min, args.sep_n_max + 1)
    }
    plot_mod.render_model_grid(
        columns, sep_examples,
        args.fig_dir / f"reconstruction_sep{args.separability:g}_align_vs_unaligned.png",
        f"MoG reconstruction — separability={args.separability:g} inputs, "
        f"aligned vs. unaligned, by #Gaussians")

    # (2) latent retrieval — where the arms actually differ
    pool = min(args.retrieval_pool, len(eval_npz["points"]))
    points_np = np.asarray(eval_npz["points"][:pool], dtype=np.float32)
    emd_sq = np.array(
        load_emd_matrix(args.data_dir / "emd_eval.npy", len(eval_npz["points"]))[:pool, :pool],
        dtype=np.float32)
    query_idx = np.random.default_rng(args.seed).choice(pool, args.n_queries, replace=False)
    lines = ["method,k,oracle_mean_emd,unaligned_mean_emd,aligned_mean_emd"]
    for method in args.methods:
        plot_retrieval(models, points_np, emd_sq, method, query_idx, args.k,
                       args.metric, args.device,
                       args.fig_dir / f"retrieval_{method}.png")
        q = retrieval_quality(models, points_np, emd_sq, method, args.k,
                              args.metric, args.device)
        print(f"  retrieval@{args.k} mean EMD — {METHOD_REGISTRY[method].label}: "
              + "  ".join(f"{k}={v:.3f}" for k, v in q.items()), flush=True)
        lines.append(f"{method},{args.k},{q.get('oracle', float('nan')):.4f},"
                     f"{q.get('unaligned', float('nan')):.4f},"
                     f"{q.get('aligned', float('nan')):.4f}")
    (args.fig_dir / "retrieval_quality.csv").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
