"""Visualization script for particle data.

This module provides functions for visualizing particle sets in both 2D and 4D,
with statistical analysis and distribution visualization.
"""

import os
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from torch.utils.data import DataLoader

from .data.dataset import get_data_loader


def to_numpy(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert input to numpy array.

    Args:
        tensor (Union[np.ndarray, torch.Tensor]): Input array or tensor.

    Returns:
        np.ndarray: Numpy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def draw_ellipse(
    mean: np.ndarray, cov: np.ndarray, ax: Axes, n_std: float = 2.0, **kwargs
) -> None:
    """Draw a covariance ellipse.

    Args:
        mean (np.ndarray): Mean vector (2,)
        cov (np.ndarray): Covariance matrix (2, 2)
        ax (Axes): Matplotlib axes
        n_std (float, optional): Number of standard deviations. Defaults to 2.0.
        **kwargs: Additional arguments passed to Ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

    scale_x = np.sqrt(eigenvals[0]) * n_std
    scale_y = np.sqrt(eigenvals[1]) * n_std

    from matplotlib.patches import Ellipse

    ellipse = Ellipse(
        xy=mean, width=2 * scale_x, height=2 * scale_y, angle=angle, **kwargs
    )
    ax.add_patch(ellipse)


def visualize_particles_2d(
    particles: Union[np.ndarray, torch.Tensor],
    title: Optional[str] = None,
    alpha: float = 0.6,
    **kwargs,
) -> None:
    """Visualize 2D particles with statistical analysis.

    Args:
        particles (Union[np.ndarray, torch.Tensor]): Particles of shape (N, 2)
        title (Optional[str], optional): Plot title. Defaults to None.
        alpha (float, optional): Alpha value for scatter plot. Defaults to 0.6.
        **kwargs: Additional arguments passed to scatter plot.
    """
    particles = to_numpy(particles)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot particles
    ax.scatter(particles[:, 0], particles[:, 1], alpha=alpha, **kwargs)

    # Compute and plot Gaussian estimate
    mean = np.mean(particles, axis=0)
    cov = np.cov(particles.T)
    draw_ellipse(mean, cov, ax, fc="none", ec="r", lw=2)

    # Add statistics
    stats = f"μ=[{mean[0]:.2f}, {mean[1]:.2f}]\n"
    stats += f"σ=[{np.sqrt(cov[0,0]):.2f}, {np.sqrt(cov[1,1]):.2f}]"
    ax.text(
        0.05,
        0.95,
        stats,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Customize plot
    ax.set_title(title or "2D Particle Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def compute_dataset_statistics(
    data_path: str,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute statistics over the entire dataset.

    Args:
        data_path (str): Path to the data file
        batch_size (int, optional): Batch size for data loading. Defaults to 32.
        device (str, optional): Device to load data on. Defaults to "cuda" if available.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - All particles data
            - Global means
            - Global standard deviations
            - Histogram bins for each dimension
    """
    # Load data
    train_loader, _, _, _ = get_data_loader(
        batch_size=batch_size, data_path=data_path, device=device
    )

    # Collect all particles
    all_particles = []
    for batch in train_loader:
        all_particles.append(to_numpy(batch.reshape(-1, batch.shape[-1])))
    all_particles = np.concatenate(all_particles, axis=0)

    # Compute global statistics
    global_means = np.mean(all_particles, axis=0)
    global_stds = np.std(all_particles, axis=0)

    # Compute histogram bins for each dimension
    hist_bins = []
    for i in range(all_particles.shape[1]):
        hist_range = (np.min(all_particles[:, i]), np.max(all_particles[:, i]))
        bins = np.linspace(hist_range[0], hist_range[1], 50)
        hist_bins.append(bins)

    return all_particles, global_means, global_stds, hist_bins


def visualize_particles_4d(
    particles: Union[np.ndarray, torch.Tensor],
    title: Optional[str] = None,
    alpha: float = 0.6,
    save_path: Optional[str] = None,
    global_stats: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = None,
    **kwargs,
) -> None:
    """Visualize 4D particles with statistical analysis.

    Args:
        particles (Union[np.ndarray, torch.Tensor]): Particles of shape (N, 4)
        title (Optional[str], optional): Plot title. Defaults to None.
        alpha (float, optional): Alpha value for scatter plot. Defaults to 0.6.
        save_path (Optional[str], optional): Path to save the plot. If None, displays plot.
        global_stats (Optional[Tuple], optional): Global statistics from the dataset.
            Contains (all_particles, means, stds, hist_bins). If provided, uses these
            for histogram visualization.
        **kwargs: Additional arguments passed to scatter plot.
    """
    particles = to_numpy(particles)

    # Create figure with two subplots for dimensions (0,1) and (2,3)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot first two dimensions
    particles_2d_first = particles[:, :2]
    ax1.scatter(
        particles_2d_first[:, 0], particles_2d_first[:, 1], alpha=alpha, **kwargs
    )

    # Compute and plot Gaussian estimate for first two dimensions
    mean1 = np.mean(particles_2d_first, axis=0)
    cov1 = np.cov(particles_2d_first.T)
    draw_ellipse(mean1, cov1, ax1, fc="none", ec="r", lw=2)

    # Add statistics for first two dimensions
    stats1 = f"μ=[{mean1[0]:.2f}, {mean1[1]:.2f}]\n"
    stats1 += f"σ=[{np.sqrt(cov1[0,0]):.2f}, {np.sqrt(cov1[1,1]):.2f}]"
    ax1.text(
        0.05,
        0.95,
        stats1,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Plot second two dimensions
    particles_2d_second = particles[:, 2:]
    ax2.scatter(
        particles_2d_second[:, 0], particles_2d_second[:, 1], alpha=alpha, **kwargs
    )

    # Compute and plot Gaussian estimate for second two dimensions
    mean2 = np.mean(particles_2d_second, axis=0)
    cov2 = np.cov(particles_2d_second.T)
    draw_ellipse(mean2, cov2, ax2, fc="none", ec="r", lw=2)

    # Add statistics for second two dimensions
    stats2 = f"μ=[{mean2[0]:.2f}, {mean2[1]:.2f}]\n"
    stats2 += f"σ=[{np.sqrt(cov2[0,0]):.2f}, {np.sqrt(cov2[1,1]):.2f}]"
    ax2.text(
        0.05,
        0.95,
        stats2,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Plot histograms for each dimension
    for i, ax in enumerate([ax3, ax4]):
        dim_data = particles[:, i * 2 : (i + 1) * 2]
        if global_stats is not None:
            all_particles, means, stds, hist_bins = global_stats
            # Plot global distribution with lower alpha
            ax.hist(
                all_particles[:, i * 2],
                bins=hist_bins[i * 2],
                alpha=0.3,
                color="gray",
                label=f"Global Dim {i*2}",
            )
            ax.hist(
                all_particles[:, i * 2 + 1],
                bins=hist_bins[i * 2 + 1],
                alpha=0.3,
                color="lightgray",
                label=f"Global Dim {i*2+1}",
            )
            # Plot sample distribution on top
            ax.hist(
                dim_data[:, 0],
                bins=hist_bins[i * 2],
                alpha=0.7,
                label=f"Sample Dim {i*2}",
            )
            ax.hist(
                dim_data[:, 1],
                bins=hist_bins[i * 2 + 1],
                alpha=0.7,
                label=f"Sample Dim {i*2+1}",
            )
            # Add global statistics
            stats = (
                f"Global μ=[{means[i*2]:.2f}, {means[i*2+1]:.2f}]\n"
                f"Global σ=[{stds[i*2]:.2f}, {stds[i*2+1]:.2f}]"
            )
            ax.text(
                0.95,
                0.95,
                stats,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.8),
            )
        else:
            ax.hist(dim_data[:, 0], bins=30, alpha=0.5, label=f"Dim {i*2}")
            ax.hist(dim_data[:, 1], bins=30, alpha=0.5, label=f"Dim {i*2+1}")
        ax.legend()
        ax.set_title(f"Distribution of Dimensions {i*2}-{i*2+1}")
        ax.grid(True, alpha=0.3)

    # Customize plots
    ax1.set_title("Dimensions 0-1 Scatter")
    ax2.set_title("Dimensions 2-3 Scatter")
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, y=1.02, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def visualize_dataset_samples(
    data_path: str,
    output_dir: str = "visualizations",
    num_samples: int = 5,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Visualize samples from the dataset.

    Args:
        data_path (str): Path to the data file
        output_dir (str): Directory to save visualizations
        num_samples (int, optional): Number of samples to visualize. Defaults to 5.
        batch_size (int, optional): Batch size for data loading. Defaults to 32.
        device (str, optional): Device to load data on. Defaults to "cuda" if available.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Compute global statistics first
    print("Computing global statistics...")
    global_stats = compute_dataset_statistics(data_path, batch_size, device)

    # Load data for visualization
    train_loader, _, _, _ = get_data_loader(
        batch_size=batch_size, data_path=data_path, device=device
    )

    # Get first batch
    batch = next(iter(train_loader))

    # Visualize specified number of samples
    print("Generating visualizations...")
    for i in range(min(num_samples, len(batch))):
        sample = batch[i]
        save_path = os.path.join(output_dir, f"sample_{i+1}.png")
        visualize_particles_4d(
            sample,
            title=f"Sample {i+1} from Training Data",
            s=50,  # marker size
            alpha=0.6,
            color="blue",
            save_path=save_path,
            global_stats=global_stats,
        )
        print(f"Saved visualization to {save_path}")


def visualize_dataset_statistics(
    data_path: str,
    output_dir: str = "visualizations",
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Visualize statistics of the entire dataset.

    Args:
        data_path (str): Path to the data file
        output_dir (str): Directory to save visualizations
        batch_size (int, optional): Batch size for data loading. Defaults to 32.
        device (str, optional): Device to load data on. Defaults to "cuda" if available.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get global statistics
    all_particles, means, stds, _ = compute_dataset_statistics(
        data_path, batch_size, device
    )

    # Create figure for histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Dataset Statistics", y=1.02, fontsize=14)

    # Plot histograms for each dimension
    for i, ax in enumerate(axes.flat):
        ax.hist(all_particles[:, i], bins=50, alpha=0.7)
        ax.set_title(f"Distribution of Dimension {i}")
        ax.grid(True, alpha=0.3)

        # Add statistics
        stats = f"μ={means[i]:.2f}\nσ={stds[i]:.2f}"
        ax.text(
            0.95,
            0.95,
            stats,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "dataset_statistics.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved dataset statistics to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize particle data")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for data loading"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load data on",
    )

    args = parser.parse_args()

    print("Visualizing individual samples...")
    visualize_dataset_samples(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("\nVisualizing dataset statistics...")
    visualize_dataset_statistics(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
    )
