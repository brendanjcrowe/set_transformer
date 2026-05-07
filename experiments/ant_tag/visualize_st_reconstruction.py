"""
Visualize Set Transformer reconstruction quality.

Loads a trained PFSetTransformer checkpoint and a particle filter dataset,
then plots original vs reconstructed particle sets side-by-side for
a selection of samples (collapsed, partial, diffuse).

Usage:
    python visualize_st_reconstruction.py \
        --checkpoint experiments/ant_tag_st_v2/chamfer_*/checkpoints/checkpoint_best.pt \
        --data_path data/ant_tag_pf_dataset_v2.npy \
        --num_samples 12
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from set_transformer.models import PFSetTransformer


def load_model(checkpoint_path: str, device: str = "cpu",
               num_particles: int = 100, dim_particles: int = 2,
               num_encodings: int = 8, dim_encoder: int = 2,
               num_inds: int = 32, dim_hidden: int = 128,
               num_heads: int = 4, ln: bool = True) -> PFSetTransformer:
    model = PFSetTransformer(
        num_particles=num_particles, dim_particles=dim_particles,
        num_encodings=num_encodings, dim_encoder=dim_encoder,
        num_inds=num_inds, dim_hidden=dim_hidden,
        num_heads=num_heads, ln=ln,
    )
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    else:
        state_dict = loaded
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def select_diverse_samples(data: np.ndarray, num_samples: int) -> np.ndarray:
    """Select samples spanning collapsed, partial, and diffuse distributions."""
    spreads = data.std(axis=1).mean(axis=1)
    sorted_idx = np.argsort(spreads)

    n = len(sorted_idx)
    # Pick evenly from the spread distribution
    pick_indices = np.linspace(0, n - 1, num_samples, dtype=int)
    selected = sorted_idx[pick_indices]
    return selected


def visualize(model: PFSetTransformer, data: np.ndarray, indices: np.ndarray,
              device: str, output_path: str):
    num_samples = len(indices)
    cols = min(num_samples, 4)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, sample_idx in enumerate(indices):
        row, col = divmod(i, cols)
        ax = axes[row, col]

        original = data[sample_idx]  # [100, 2]
        spread = original.std(axis=0).mean()

        # Reconstruct
        input_tensor = torch.tensor(original, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed = model(input_tensor).squeeze(0).cpu().numpy()

        # Also get the latent encoding
        with torch.no_grad():
            latent = model.set_transformer(input_tensor).squeeze(0).cpu().numpy()

        # Plot
        ax.scatter(original[:, 0], original[:, 1], c='blue', alpha=0.4, s=15, label='Original')
        ax.scatter(reconstructed[:, 0], reconstructed[:, 1], c='red', alpha=0.4, s=15, label='Reconstructed')
        ax.set_xlim(-11, 11)
        ax.set_ylim(-11, 11)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_title(f'Sample {sample_idx} (spread={spread:.2f})\nLatent: {latent.flatten()[:6]}...', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(num_samples, rows * cols):
        row, col = divmod(i, cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Set Transformer: Original vs Reconstructed Particle Sets', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


def compute_reconstruction_stats(model: PFSetTransformer, data: np.ndarray,
                                  device: str, num_eval: int = 500):
    """Compute reconstruction error statistics across the dataset."""
    indices = np.random.choice(len(data), min(num_eval, len(data)), replace=False)
    chamfer_dists = []

    for idx in indices:
        original = data[idx]
        input_tensor = torch.tensor(original, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            reconstructed = model(input_tensor).squeeze(0).cpu().numpy()

        # Chamfer distance
        from scipy.spatial.distance import cdist
        dist_matrix = cdist(original, reconstructed)
        forward = dist_matrix.min(axis=1).mean()
        backward = dist_matrix.min(axis=0).mean()
        chamfer_dists.append(forward + backward)

    chamfer_dists = np.array(chamfer_dists)
    spreads = data[indices].std(axis=1).mean(axis=1)

    print(f"\nReconstruction stats ({num_eval} samples):")
    print(f"  Chamfer distance: mean={chamfer_dists.mean():.4f}, std={chamfer_dists.std():.4f}")
    print(f"  Chamfer by spread bucket:")
    for lo, hi, label in [(0, 0.5, 'collapsed'), (0.5, 3, 'partial'), (3, 100, 'diffuse')]:
        mask = (spreads >= lo) & (spreads < hi)
        if mask.sum() > 0:
            print(f"    {label} (n={mask.sum()}): chamfer={chamfer_dists[mask].mean():.4f}")
        else:
            print(f"    {label}: no samples")


def main():
    parser = argparse.ArgumentParser(description="Visualize ST reconstruction")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--output", type=str, default="st_reconstruction_viz.png")
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--dim_particles", type=int, default=2)
    parser.add_argument("--num_encodings", type=int, default=8)
    parser.add_argument("--dim_encoder", type=int, default=2)
    parser.add_argument("--num_inds", type=int, default=32)
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.checkpoint}")
    model = load_model(
        args.checkpoint, device=device,
        num_particles=args.num_particles, dim_particles=args.dim_particles,
        num_encodings=args.num_encodings, dim_encoder=args.dim_encoder,
        num_inds=args.num_inds, dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
    )
    latent_dim = args.num_encodings * args.dim_encoder
    print(f"Latent space: {args.num_encodings} encodings x {args.dim_encoder}D = {latent_dim}-dim")

    print(f"Loading data from {args.data_path}")
    data = np.load(args.data_path)
    print(f"Data shape: {data.shape}")

    indices = select_diverse_samples(data, args.num_samples)
    visualize(model, data, indices, device, args.output)
    compute_reconstruction_stats(model, data, device)


if __name__ == "__main__":
    main()
