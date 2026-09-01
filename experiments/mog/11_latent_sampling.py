"""Sample latent codes OUT OF DISTRIBUTION, decode them, and see what geometry survives.

Design: four sampled codes per model, arranged as two PAIRS.
  * pair A = (A1, A2) — deliberately placed CLOSE together in latent space
  * pair B = (B1, B2) — also close to each other, but the two pairs are placed as far
    apart from one another as the sampled pool allows

If the latent space carries the input geometry, the decoded sets should mirror that
structure: A1~A2 and B1~B2 should look like near-duplicates, while anything A-vs-B should
look plainly unrelated.

**Sampling is deliberately out of distribution.** Each latent coordinate is drawn
uniformly and independently between the low and high values that coordinate typically
takes over the encoded held-out split. Real codes occupy a thin, correlated manifold
inside that axis-aligned box, so independent uniform draws mostly land in the box's empty
interior — this probes what the decoder does off-manifold, which is a strictly harder test
than resampling the encoded distribution. Every run prints how far off-manifold the draws
actually landed (``OOD xN``, the median distance from a draw to the nearest real code in
units of the real codes' own nearest-neighbour distance), so the OOD claim is measured
rather than assumed.

The two anchors are the most distant pair in the pool. Each partner is then CONSTRUCTED at
a set distance from its anchor rather than searched for, because uniform draws in a 128-d
box are never close to one another. That distance is expressed in each model's own units —
the median nearest-neighbour distance between real codes — since raw latent scales are not
comparable across models.

    python experiments/mog/11_latent_sampling.py

Writes `latent_samples_<tag>.png` (rows = models, cols = A1 A2 B1 B2) plus
`latent_samples_<tag>.csv` with the 6 unique pair distances and their correlation.
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

from set_transformer.latent_alignment import pearson_r
from set_transformer.loss import EarthMoverDistanceLoss

from _common import ARCH, METHOD_REGISTRY, encode_all

plot_mod = __import__("3_plot")
infer_mod = __import__("10_inference_alignment")

CELL_LABELS = ["A1", "A2", "B1", "B2"]


def latent_bounds(z: torch.Tensor, percentile: float):
    """Per-dimension (low, high) of the encoded codes.

    ``percentile`` trims each end so a single outlier code cannot blow the box out; 0
    gives the exact observed min/max.
    """
    if percentile <= 0:
        return z.min(0).values, z.max(0).values
    q = torch.tensor([percentile / 100.0, 1.0 - percentile / 100.0], dtype=z.dtype)
    lo_hi = torch.quantile(z, q, dim=0)
    return lo_hi[0], lo_hi[1]


def sample_uniform_box(low, high, n, generator):
    """Independent uniform draws per coordinate — the OOD sampler."""
    u = torch.rand(n, len(low), generator=generator, dtype=low.dtype)
    return low.unsqueeze(0) + u * (high - low).unsqueeze(0)


def offmanifold_report(pool: torch.Tensor, z: torch.Tensor, metric: str):
    """How far the sampled pool sits from real codes, in units of the real codes' own
    nearest-neighbour distance. >> 1 means the draws are genuinely off-manifold."""
    d_real = pairwise(z, metric)
    d_real.fill_diagonal_(float("inf"))
    real_nn = d_real.min(1).values.median()
    d_pool = (1.0 - torch.nn.functional.normalize(pool, dim=1, eps=1e-8)
              @ torch.nn.functional.normalize(z, dim=1, eps=1e-8).t()
              ) if metric == "cosine" else torch.cdist(pool, z, p=2)
    pool_nn = d_pool.min(1).values.median()
    return float(pool_nn), float(real_nn), float(pool_nn / max(real_nn, 1e-12))


def pairwise(z: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "cosine":
        normed = torch.nn.functional.normalize(z, dim=1, eps=1e-8)
        return 1.0 - normed @ normed.t()
    return torch.cdist(z, z, p=2)


def perturb_to_distance(a: torch.Tensor, target: float, metric: str, generator):
    """A second code at exactly ``target`` distance from ``a``.

    Searching the uniform pool for a near neighbour does not work: independent uniform
    draws in a 128-d box are all far apart (nothing lands within ~0.3 cosine), so the
    "similar" pair would not actually be similar. Constructing the partner instead makes
    the within-pair separation exact and lets it be set to a meaningful scale — by
    default the distance between neighbouring *real* codes. A small perturbation of an
    OOD point is still OOD, so this does not smuggle the sample back on-manifold.
    """
    u = torch.randn(len(a), generator=generator, dtype=a.dtype)
    if metric == "euclidean":
        return a + target * u / u.norm()
    # cosine: rotate `a` by theta = arccos(1 - target) toward an orthogonal direction,
    # preserving its norm (which cosine distance ignores anyway).
    a_hat = a / a.norm()
    u = u - (u @ a_hat) * a_hat
    u = u / u.norm()
    theta = float(np.arccos(np.clip(1.0 - target, -1.0, 1.0)))
    return a.norm() * (a_hat * np.cos(theta) + u * np.sin(theta))


def pick_four(pool: torch.Tensor, metric: str, target: float, generator):
    """(A1, A2, B1, B2): two tight pairs placed as far apart from each other as possible.

    Anchors are the most distant pair in the pool, which makes "drastically different"
    concrete rather than a hand-picked threshold; partners are constructed at exactly
    ``target`` distance from their anchor.
    """
    d = pairwise(pool, metric)
    a1, b1 = np.unravel_index(int(torch.argmax(d)), (len(pool), len(pool)))
    anchors = [pool[int(a1)], pool[int(b1)]]
    return torch.stack([
        anchors[0], perturb_to_distance(anchors[0], target, metric, generator),
        anchors[1], perturb_to_distance(anchors[1], target, metric, generator),
    ])


def decode(model, z_flat: torch.Tensor) -> np.ndarray:
    """Decode flat latents back to point sets, reshaped to the bottleneck's own shape."""
    device = next(model.parameters()).device
    z = z_flat.reshape(len(z_flat), ARCH["num_encodings"], ARCH["dim_encoder"])
    with torch.no_grad():
        return model.decoder(z.to(device)).cpu().numpy()


def analyse_model(model, points, device, is_vae, metric, sim_scale, pool_size,
                  bounds_percentile, seed):
    z = encode_all(model, points, device).cpu().double()
    low, high = latent_bounds(z, bounds_percentile)
    g = torch.Generator().manual_seed(seed)
    pool = sample_uniform_box(low, high, pool_size, g)
    pool_nn, real_nn, ratio = offmanifold_report(pool, z, metric)
    # "Similar" is defined in each model's own units: the distance between neighbouring
    # real codes, so the pair is as alike as two adjacent points of real data.
    target = sim_scale * real_nn
    chosen = pick_four(pool, metric, target, g)

    sets = decode(model, chosen.float())
    d_lat = pairwise(chosen, metric)

    emd = EarthMoverDistanceLoss(reduction="none")
    t = torch.from_numpy(sets).float()
    d_emd = torch.zeros(4, 4)
    for i in range(4):
        for j in range(i + 1, 4):
            v = float(emd(t[i:i + 1], t[j:j + 1]))
            d_emd[i, j] = d_emd[j, i] = v

    iu = torch.triu_indices(4, 4, offset=1).unbind()
    r = pearson_r(d_lat[iu].float(), d_emd[iu])
    return {
        "sets": sets,
        "d_latent": d_lat.numpy(),
        "d_emd": d_emd.numpy(),
        "pearson_r": float("nan") if r is None else float(r),
        "latent_dim": z.shape[1],
        "sim_target": float(target),
        "pool_nn": pool_nn,
        "real_nn": real_nn,
        "ood_ratio": ratio,
    }


def plot_grid(results, order, labels, out_path: Path, metric: str, bounds_percentile):
    rows = [k for k in order if k in results]
    fig, axes = plt.subplots(len(rows), 4, figsize=(4 * 2.0, len(rows) * 2.15),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.08, "wspace": 0.06})
    axes = np.atleast_2d(axes)
    xlim, ylim = plot_mod._square_bbox([r["sets"] for r in results.values()])

    for r, key in enumerate(rows):
        res = results[key]
        for c in range(4):
            ax = axes[r, c]
            pts = res["sets"][c]
            ax.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.75,
                       color="tab:green" if c < 2 else "tab:purple")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(CELL_LABELS[c], fontsize=10, pad=3)
            if c == 0:
                ax.set_ylabel(labels[key], fontsize=9)
        dl, de = res["d_latent"], res["d_emd"]
        within_l = (dl[0, 1] + dl[2, 3]) / 2
        cross_l = dl[:2, 2:].mean()
        within_e = (de[0, 1] + de[2, 3]) / 2
        cross_e = de[:2, 2:].mean()
        axes[r, 3].text(
            1.04, 0.5,
            f"latent  in {within_l:.3f} / out {cross_l:.3f}\n"
            f"set EMD in {within_e:.2f} / out {cross_e:.2f}\n"
            f"r = {res['pearson_r']:.3f}   OOD x{res['ood_ratio']:.1f}",
            transform=axes[r, 3].transAxes, fontsize=8, va="center", ha="left",
            family="monospace")

    trim = "min/max" if bounds_percentile <= 0 else f"{bounds_percentile:g}-{100 - bounds_percentile:g}%"
    fig.suptitle(
        f"Decoded OOD latent samples (uniform per-coordinate over the {trim} range of "
        f"encoded codes)\nA1,A2 share a latent neighbourhood; B1,B2 share another, far "
        f"away. {metric} distance; 'in' = within-pair, 'out' = across-pair",
        fontsize=11)
    fig.subplots_adjust(top=0.88, bottom=0.04, left=0.10, right=0.70)
    plot_mod._save(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_runs", type=Path,
                    default=Path("experiments/mog/runs_align_sinkhorn"))
    ap.add_argument("--baseline_runs", type=Path,
                    default=Path("experiments/mog/runs_varsep_sinkhorn"))
    ap.add_argument("--data_dir", type=Path, default=Path("experiments/mog/data_varsep"))
    ap.add_argument("--fig_dir", type=Path, default=Path("experiments/mog/figures_align"))
    ap.add_argument("--mode", choices=["arms", "methods"], default="arms",
                    help="'arms' = {ST-AE, DS-AE} x {unaligned, aligned}; "
                         "'methods' = the four unaligned MoG methods (ST/DS x AE/VAE)")
    ap.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--sim_scale", type=float, default=1.0,
                    help="within-pair latent distance, in multiples of the median "
                         "nearest-neighbour distance between real encoded codes")
    ap.add_argument("--bounds_percentile", type=float, default=1.0,
                    help="per-dimension range trim; 0 = exact observed min/max")
    ap.add_argument("--pool_size", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    eval_pts = np.load(args.data_dir / "eval.points.npy")
    points = torch.from_numpy(eval_pts).float()

    specs = []
    if args.mode == "arms":
        for method in ("st_ae", "ds_ae"):
            for arm, runs in (("unaligned", args.baseline_runs),
                              ("aligned", args.aligned_runs)):
                specs.append((f"{method}|{arm}",
                              f"{METHOD_REGISTRY[method].label}\n{arm}", method, runs))
    else:
        for method in ("st_ae", "st_vae", "ds_ae", "ds_vae"):
            specs.append((method, METHOD_REGISTRY[method].label, method,
                          args.baseline_runs))

    results, labels, order = {}, {}, []
    for key, label, method, runs in specs:
        entry = infer_mod.load_best(runs, method, args.device) if runs.exists() else None
        if entry is None:
            print(f"skipping {key}: no runs under {runs}", flush=True)
            continue
        model, seed, val_emd = entry
        res = analyse_model(model, points, args.device, METHOD_REGISTRY[method].is_vae,
                            args.metric, args.sim_scale, args.pool_size,
                            args.bounds_percentile, args.seed)
        results[key], labels[key] = res, label
        order.append(key)
        print(f"{label.replace(chr(10), ' '):22s} seed{seed} val_emd={val_emd:.4f}  "
              f"OOD x{res['ood_ratio']:.1f} (pool_nn {res['pool_nn']:.3f} vs "
              f"real_nn {res['real_nn']:.3f})\n"
              f"    d_lat A={res['d_latent'][0,1]:.3f} B={res['d_latent'][2,3]:.3f} "
              f"cross={res['d_latent'][:2,2:].mean():.3f}   "
              f"EMD A={res['d_emd'][0,1]:.2f} B={res['d_emd'][2,3]:.2f} "
              f"cross={res['d_emd'][:2,2:].mean():.2f}   r={res['pearson_r']:.3f}",
              flush=True)

    if not results:
        raise SystemExit("no models found")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_grid(results, order, labels,
              args.fig_dir / f"latent_samples_{args.mode}.png", args.metric,
              args.bounds_percentile)

    with open(args.fig_dir / f"latent_samples_{args.mode}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "pair_i", "pair_j", "latent_distance", "decoded_set_emd",
                    "pearson_r_over_6_pairs", "ood_ratio"])
        for key in order:
            res = results[key]
            for i in range(4):
                for j in range(i + 1, 4):
                    w.writerow([key, CELL_LABELS[i], CELL_LABELS[j],
                                f"{res['d_latent'][i, j]:.5f}",
                                f"{res['d_emd'][i, j]:.5f}",
                                f"{res['pearson_r']:.5f}",
                                f"{res['ood_ratio']:.3f}"])
    print(f"\nwrote figure + CSV to {args.fig_dir}", flush=True)


if __name__ == "__main__":
    main()
