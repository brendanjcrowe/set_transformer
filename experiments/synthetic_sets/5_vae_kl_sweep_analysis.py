"""Analysis for the VAE KL-weight sweep.

For each checkpoint in the sweep, compute:
- mean EMD on the eval set (reconstruction quality)
- mean KL-per-sample achieved at convergence
- perplexity of the empirical mu-distribution (rough proxy for how much the
  posterior collapsed toward the prior)

Produces experiments/synthetic_sets/figures/F5_kl_sweep.png with two panels:
    left: EMD vs kl_weight (log-x); baseline shown as a horizontal reference
    right: mean KL vs kl_weight (log-x)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from set_transformer.loss import EarthMoverDistanceLoss

from _common import DATA_DIR
from importlib import import_module

compare_mod = import_module("3_compare_reconstruction")

FIG_DIR = Path("experiments/synthetic_sets/figures")


def eval_model(model, eval_pts, device, batch_size=64):
    emd_fn = EarthMoverDistanceLoss(reduction="none")
    all_emd = []
    all_kl = []
    pts = torch.from_numpy(eval_pts)
    with torch.no_grad():
        for start in range(0, len(pts), batch_size):
            x = pts[start : start + batch_size].to(device)
            out = model(x)
            recon = out["recon"] if isinstance(out, dict) else out
            all_emd.append(emd_fn(recon, x).cpu().numpy())
            if isinstance(out, dict) and "kl" in out:
                # kl is already batch-mean-scalar; expand to len(batch).
                all_kl.append(np.full(len(x), out["kl"].item()))
    emd = np.concatenate(all_emd)
    kl = np.concatenate(all_kl) if all_kl else None
    return emd, kl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=Path, required=True)
    parser.add_argument(
        "--vae_ckpts",
        type=str,
        required=True,
        help="Comma-separated 'kl_weight:path' pairs, e.g. '1e-4:.../best.pt,1e-3:.../best.pt'",
    )
    parser.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    eval_pts = np.load(DATA_DIR / "eval.npz")["points"]

    # Baseline reference.
    baseline, _ = compare_mod.load_model(args.baseline_ckpt, args.device)
    baseline_emd, _ = eval_model(baseline, eval_pts, args.device)
    baseline_mean = float(baseline_emd.mean())

    entries = []
    for pair in args.vae_ckpts.split(","):
        kl_str, path = pair.split(":", 1)
        kl_w = float(kl_str)
        model, _ = compare_mod.load_model(Path(path), args.device)
        emd, kl = eval_model(model, eval_pts, args.device)
        entries.append(
            {
                "kl_weight": kl_w,
                "emd_mean": float(emd.mean()),
                "emd_ci_lo": float(np.quantile(emd, 0.025)),
                "emd_ci_hi": float(np.quantile(emd, 0.975)),
                "kl_mean": float(kl.mean()) if kl is not None else float("nan"),
            }
        )
    entries.sort(key=lambda e: e["kl_weight"])

    kls = [e["kl_weight"] for e in entries]
    emd_means = [e["emd_mean"] for e in entries]
    emd_lo = [m - e["emd_ci_lo"] for m, e in zip(emd_means, entries)]
    emd_hi = [e["emd_ci_hi"] - m for m, e in zip(emd_means, entries)]
    kl_vals = [e["kl_mean"] for e in entries]

    fig, (ax_emd, ax_kl) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_emd.errorbar(
        kls, emd_means, yerr=[emd_lo, emd_hi],
        marker="o", capsize=4, color="tab:orange", label="VAE",
    )
    ax_emd.axhline(baseline_mean, color="tab:blue", linestyle="--", label=f"Vanilla ({baseline_mean:.3f})")
    ax_emd.set_xscale("log")
    ax_emd.set_xlabel("kl_weight", fontsize=10)
    ax_emd.set_ylabel("Mean EMD on eval (lower is better)", fontsize=10)
    ax_emd.set_title("Reconstruction quality", fontsize=11, pad=4)
    ax_emd.grid(True, linestyle=":", alpha=0.4)
    ax_emd.legend(loc="best", fontsize=9)

    ax_kl.plot(kls, kl_vals, marker="o", color="tab:orange")
    ax_kl.set_xscale("log")
    ax_kl.set_yscale("log")
    ax_kl.set_xlabel("kl_weight", fontsize=10)
    ax_kl.set_ylabel("Mean KL(q(z|x) || N(0,I))", fontsize=10)
    ax_kl.set_title("Posterior mass", fontsize=11, pad=4)
    ax_kl.grid(True, which="both", linestyle=":", alpha=0.4)

    fig.suptitle("F5 — VAE KL-weight sweep (higher weight → tighter prior, worse recon)", fontsize=13)
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.08, right=0.98, wspace=0.28)
    out_path = args.fig_dir / "F5_kl_sweep.png"
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # CSV of raw numbers.
    csv_path = args.fig_dir / "F5_kl_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["kl_weight", "emd_mean", "emd_ci_lo", "emd_ci_hi", "kl_mean"]
        )
        w.writeheader()
        for e in entries:
            w.writerow(e)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
