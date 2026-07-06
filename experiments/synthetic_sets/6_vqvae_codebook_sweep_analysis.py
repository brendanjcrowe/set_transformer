"""Analysis for the VQ-VAE codebook-size sweep.

For each checkpoint, compute:
- mean EMD on the eval set
- perplexity of the code-usage distribution across the eval set (higher =
  codes used more uniformly; max = codebook_size)
- fraction of codes that appear at least once (dead-code diagnostic)

Produces F6_codebook_sweep.png with two panels:
    left:  EMD vs codebook_size, baseline as horizontal reference
    right: perplexity vs codebook_size, with a dashed y=K "uniform-usage" line
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


def eval_vqvae(model, eval_pts, device, batch_size=64):
    emd_fn = EarthMoverDistanceLoss(reduction="none")
    all_emd = []
    all_indices = []
    pts = torch.from_numpy(eval_pts)
    with torch.no_grad():
        for start in range(0, len(pts), batch_size):
            x = pts[start : start + batch_size].to(device)
            out = model(x)
            all_emd.append(emd_fn(out["recon"], x).cpu().numpy())
            all_indices.append(out["indices"].cpu().numpy().reshape(-1))
    emd = np.concatenate(all_emd)
    indices = np.concatenate(all_indices)
    K = model.quantizer.num_codes
    counts = np.bincount(indices, minlength=K).astype(np.float64)
    probs = counts / max(1, counts.sum())
    perplexity = float(np.exp(-(probs * np.log(probs + 1e-12)).sum()))
    frac_used = float((counts > 0).mean())
    return emd, perplexity, frac_used, K


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_ckpt", type=Path, required=True)
    parser.add_argument(
        "--vqvae_ckpts",
        type=str,
        required=True,
        help="Comma-separated 'codebook_size:path' pairs",
    )
    parser.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    eval_pts = np.load(DATA_DIR / "eval.npz")["points"]

    baseline, _ = compare_mod.load_model(args.baseline_ckpt, args.device)
    baseline_emd, _, _, _ = eval_vqvae(baseline, eval_pts, args.device) if False else (None, None, None, None)
    from set_transformer.loss import EarthMoverDistanceLoss as _EMD
    emd_fn = _EMD(reduction="none")
    with torch.no_grad():
        all_bl = []
        pts = torch.from_numpy(eval_pts)
        for start in range(0, len(pts), 64):
            x = pts[start : start + 64].to(args.device)
            out = baseline(x)
            recon = out["recon"] if isinstance(out, dict) else out
            all_bl.append(emd_fn(recon, x).cpu().numpy())
        baseline_mean = float(np.concatenate(all_bl).mean())

    entries = []
    for pair in args.vqvae_ckpts.split(","):
        K_str, path = pair.split(":", 1)
        K = int(K_str)
        model, _ = compare_mod.load_model(Path(path), args.device)
        emd, ppl, frac_used, K_actual = eval_vqvae(model, eval_pts, args.device)
        entries.append(
            {
                "codebook_size": K,
                "emd_mean": float(emd.mean()),
                "emd_ci_lo": float(np.quantile(emd, 0.025)),
                "emd_ci_hi": float(np.quantile(emd, 0.975)),
                "perplexity": ppl,
                "frac_codes_used": frac_used,
            }
        )
    entries.sort(key=lambda e: e["codebook_size"])

    Ks = [e["codebook_size"] for e in entries]
    emd_means = [e["emd_mean"] for e in entries]
    emd_lo = [m - e["emd_ci_lo"] for m, e in zip(emd_means, entries)]
    emd_hi = [e["emd_ci_hi"] - m for m, e in zip(emd_means, entries)]
    ppls = [e["perplexity"] for e in entries]
    frac_used = [e["frac_codes_used"] for e in entries]

    fig, (ax_emd, ax_ppl) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_emd.errorbar(
        Ks, emd_means, yerr=[emd_lo, emd_hi],
        marker="o", capsize=4, color="tab:green", label="VQ-VAE",
    )
    ax_emd.axhline(baseline_mean, color="tab:blue", linestyle="--", label=f"Vanilla ({baseline_mean:.3f})")
    ax_emd.set_xscale("log", base=2)
    ax_emd.set_xticks(Ks)
    ax_emd.set_xticklabels([str(k) for k in Ks])
    ax_emd.set_xlabel("codebook_size (K)", fontsize=10)
    ax_emd.set_ylabel("Mean EMD on eval (lower is better)", fontsize=10)
    ax_emd.set_title("Reconstruction quality", fontsize=11, pad=4)
    ax_emd.grid(True, linestyle=":", alpha=0.4)
    ax_emd.legend(loc="best", fontsize=9)

    ax_ppl.plot(Ks, ppls, marker="o", color="tab:green", label="perplexity")
    ax_ppl.plot(Ks, Ks, color="grey", linestyle="--", label="ideal (K)")
    for k, ppl, fu in zip(Ks, ppls, frac_used):
        ax_ppl.annotate(
            f"{100 * fu:.0f}% codes used",
            (k, ppl),
            textcoords="offset points",
            xytext=(0, 8),
            fontsize=8,
            ha="center",
            color="tab:green",
        )
    ax_ppl.set_xscale("log", base=2)
    ax_ppl.set_yscale("log", base=2)
    ax_ppl.set_xticks(Ks)
    ax_ppl.set_xticklabels([str(k) for k in Ks])
    ax_ppl.set_xlabel("codebook_size (K)", fontsize=10)
    ax_ppl.set_ylabel("Perplexity of code usage", fontsize=10)
    ax_ppl.set_title("Codebook health", fontsize=11, pad=4)
    ax_ppl.grid(True, which="both", linestyle=":", alpha=0.4)
    ax_ppl.legend(loc="best", fontsize=9)

    fig.suptitle(
        "F6 — VQ-VAE codebook-size sweep (larger K → more expressive; watch for collapse)",
        fontsize=13,
    )
    fig.subplots_adjust(top=0.86, bottom=0.15, left=0.08, right=0.98, wspace=0.28)
    out_path = args.fig_dir / "F6_codebook_sweep.png"
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")

    csv_path = args.fig_dir / "F6_codebook_sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "codebook_size",
                "emd_mean",
                "emd_ci_lo",
                "emd_ci_hi",
                "perplexity",
                "frac_codes_used",
            ],
        )
        w.writeheader()
        for e in entries:
            w.writerow(e)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
