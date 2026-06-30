"""VQ-VAE codebook probes (figure F7).

Produces three artifacts under experiments/synthetic_sets/figures/:
    F7a_code_usage.png — code-usage histogram + perplexity
    F7b_code_family_heatmap.png — per-code distribution over distribution families
    F7c_decoder_intervention.png — replace each (slot, code) and visualize recon
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from set_transformer.data.synthetic_sets import FAMILY_NAMES

from _common import DATA_DIR

# Reuse the loader from 3_compare_reconstruction so the model dispatch stays
# in one place.
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

compare_mod = import_module("3_compare_reconstruction")

FIG_DIR = Path("experiments/synthetic_sets/figures")


def code_usage_histogram(
    model, eval_pts: np.ndarray, device: str, batch_size: int = 64
) -> np.ndarray:
    counts = np.zeros(model.quantizer.num_codes, dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(eval_pts), batch_size):
            x = torch.from_numpy(eval_pts[start : start + batch_size]).to(device)
            out = model(x)
            idx = out["indices"].cpu().numpy().reshape(-1)
            np.add.at(counts, idx, 1)
    return counts


def code_family_heatmap(
    model, eval_pts: np.ndarray, eval_ids: np.ndarray, device: str, batch_size: int = 64
) -> np.ndarray:
    """Returns (K, F): for each code, conditional distribution over families.

    P(family | code k) measured as fraction of point-sets containing code k that
    belong to each family.
    """
    K = model.quantizer.num_codes
    F = len(FAMILY_NAMES)
    sets_per_code = np.zeros((K, F), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(eval_pts), batch_size):
            end = min(start + batch_size, len(eval_pts))
            x = torch.from_numpy(eval_pts[start:end]).to(device)
            ids = eval_ids[start:end]
            out = model(x)
            indices = out["indices"].cpu().numpy()  # (B, num_encodings)
            for b in range(indices.shape[0]):
                family = int(ids[b])
                for k in np.unique(indices[b]):
                    sets_per_code[k, family] += 1
    row_sums = sets_per_code.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return sets_per_code / row_sums


def decoder_intervention(
    model, neutral_set: np.ndarray, device: str, max_codes_to_show: int = 32
) -> np.ndarray:
    """Build a (num_encodings, K_shown) grid of reconstructions where the (slot,
    code) cell decodes the encoding of `neutral_set` with slot `i` replaced by
    code `k`.
    Returns array of shape (num_encodings, K_shown, num_particles, 2).
    """
    with torch.no_grad():
        x = torch.from_numpy(neutral_set).unsqueeze(0).to(device)
        z_e = model.set_transformer(x)
        q = model.quantizer(z_e)
        z_q = q["z_q"]  # (1, num_encodings, D)
        num_encodings = z_q.shape[1]
        K = model.quantizer.num_codes
        K_shown = min(K, max_codes_to_show)

        # Probe the first cell to learn the decoder's output shape.
        first = model.decoder(z_q.clone())
        n_out, d_out = first.shape[1], first.shape[2]
        recons = np.empty((num_encodings, K_shown, n_out, d_out), dtype=np.float32)
        for i in range(num_encodings):
            for k in range(K_shown):
                z_mod = z_q.clone()
                z_mod[0, i] = model.quantizer.embedding[k]
                recon = model.decoder(z_mod)
                recons[i, k] = recon[0].cpu().numpy()
    return recons


def plot_code_usage(counts: np.ndarray, out_path: Path) -> None:
    probs = counts / max(1, counts.sum())
    perplexity = float(np.exp(-(probs * np.log(probs + 1e-12)).sum()))
    fig, ax = plt.subplots(
        figsize=(max(6, len(counts) * 0.18), 4.0),
        constrained_layout=True,
    )
    ax.bar(np.arange(len(counts)), counts, color="tab:green")
    ax.set_xlabel("Codebook index (k)", fontsize=10)
    ax.set_ylabel("Slot-assignment count\n(over eval set)", fontsize=10)
    ax.set_title(
        f"F7a — code usage histogram\n"
        f"perplexity = {perplexity:.2f}    (K = {len(counts)} codes, max possible = {len(counts)})",
        fontsize=11,
        pad=6,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_code_family_heatmap(matrix: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(
        figsize=(
            max(7, matrix.shape[1] * 0.9),
            max(6, matrix.shape[0] * 0.22) + 0.6,
        ),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(FAMILY_NAMES)))
    ax.set_xticklabels(FAMILY_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(np.arange(matrix.shape[0]), fontsize=7)
    ax.set_xlabel("Distribution family", fontsize=10)
    ax.set_ylabel("Codebook index (k)", fontsize=10)
    ax.set_title(
        "F7b — P(family | code k appears in encoding)", fontsize=11, pad=6
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Conditional probability", fontsize=9)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_decoder_intervention(
    neutral_set: np.ndarray, recons: np.ndarray, out_path: Path
) -> None:
    num_encodings, K_shown = recons.shape[:2]

    # Global square bbox over neutral input + every decoded reconstruction.
    all_arrays = [neutral_set, recons.reshape(-1, 2)]
    stacked = np.concatenate(all_arrays, axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2 + 0.3
    xlim = (cx - half, cx + half)
    ylim = (cy - half, cy + half)

    fig, axes = plt.subplots(
        num_encodings,
        K_shown,
        figsize=(K_shown * 1.1, num_encodings * 1.1),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )
    if num_encodings == 1:
        axes = axes[np.newaxis, :]
    for i in range(num_encodings):
        for k in range(K_shown):
            ax = axes[i, k]
            ax.scatter(recons[i, k, :, 0], recons[i, k, :, 1], s=2, alpha=0.7)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
            if i == 0:
                ax.set_title(f"k={k}", fontsize=8, pad=3)
            if k == 0:
                ax.set_ylabel(f"slot {i}", fontsize=9)

    fig.suptitle(
        "F7c — decoder-side intervention grid", fontsize=13,
    )
    fig.supxlabel("Codebook index (k) substituted into slot", fontsize=10)
    fig.supylabel("Encoding slot (i) being replaced", fontsize=10)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.07, right=0.98)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vqvae_ckpt", type=Path, required=True)
    parser.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    parser.add_argument("--max_codes_shown", type=int, default=32)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    eval_npz = np.load(DATA_DIR / "eval.npz")
    eval_pts, eval_ids = eval_npz["points"], eval_npz["family_ids"]

    model, cfg = compare_mod.load_model(args.vqvae_ckpt, args.device)
    assert cfg.model_type == "set_vqvae", f"checkpoint is {cfg.model_type}, not VQ-VAE"

    counts = code_usage_histogram(model, eval_pts, args.device)
    plot_code_usage(counts, args.fig_dir / "F7a_code_usage.png")

    matrix = code_family_heatmap(model, eval_pts, eval_ids, args.device)
    plot_code_family_heatmap(matrix, args.fig_dir / "F7b_code_family_heatmap.png")
    np.save(args.fig_dir / "F7b_code_family_heatmap.npy", matrix)

    # Use the first eval example whose family is "iso_gaussian" as the neutral set.
    iso_idx = np.where(eval_ids == 0)[0]
    neutral = eval_pts[iso_idx[0]] if len(iso_idx) > 0 else eval_pts[0]
    recons = decoder_intervention(
        model, neutral, args.device, max_codes_to_show=args.max_codes_shown
    )
    plot_decoder_intervention(
        neutral, recons, args.fig_dir / "F7c_decoder_intervention.png"
    )


if __name__ == "__main__":
    main()
