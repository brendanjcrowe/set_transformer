"""Visualization utilities for mixture models and point clouds.

This module provides functions for visualizing 2D point clouds, mixture components,
and confidence ellipses. It supports both labeled and unlabeled data visualization.
"""

from typing import Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse
from scipy.stats import chi2


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


def scatter(
    X: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create scatter plot of points, optionally with labels.

    Args:
        X (Union[np.ndarray, torch.Tensor]): Points to plot (Nx2).
        labels (Optional[Union[np.ndarray, torch.Tensor]], optional): Labels for points.
            Defaults to None.
        ax (Optional[Axes], optional): Matplotlib axes to plot on. Defaults to None.
        **kwargs: Additional arguments passed to scatter.

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]: Unique labels and colors if labels provided,
            otherwise (None, None).

    Raises:
        ValueError: If X does not have shape (N, 2) or if labels length doesn't match X.
    """
    if not isinstance(X, np.ndarray):
        X = to_numpy(X)
    if X.shape[1] != 2:
        raise ValueError(f"X must have shape (N, 2), got {X.shape}")

    if labels is not None:
        if not isinstance(labels, np.ndarray):
            labels = to_numpy(labels)
        if len(labels) != len(X):
            raise ValueError(
                f"Labels length {len(labels)} must match X length {len(X)}"
            )

    ax = ax or plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if labels is None:
        # Use default color if not specified in kwargs
        color = kwargs.pop("color", "k")
        facecolor = kwargs.pop("facecolor", color)
        edgecolor = kwargs.pop("edgecolor", [0.2, 0.2, 0.2])
        ax.scatter(X[:, 0], X[:, 1], facecolor=facecolor, edgecolor=edgecolor, **kwargs)
        return None, None
    else:
        ulabels = np.sort(np.unique(labels))
        colors = cm.rainbow(np.linspace(0, 1, len(ulabels)))
        for l, c in zip(ulabels, colors):
            mask = labels == l
            ax.scatter(X[mask, 0], X[mask, 1], color=c, edgecolor=c * 0.6, **kwargs)
        return ulabels, colors


def draw_ellipse(
    pos: Union[np.ndarray, torch.Tensor],
    cov: Union[np.ndarray, torch.Tensor],
    ax: Optional[Axes] = None,
    alpha: Optional[float] = None,
    **kwargs,
) -> None:
    """Draw confidence ellipses for a 2D Gaussian distribution.

    Draws multiple ellipses corresponding to different confidence levels
    (1 to 5 standard deviations).

    Args:
        pos (Union[np.ndarray, torch.Tensor]): Mean of the Gaussian (2D)
        cov (Union[np.ndarray, torch.Tensor]): Covariance matrix (2x2)
        ax (Optional[Axes], optional): Matplotlib axes to plot on. Defaults to None.
        alpha (Optional[float], optional): Override the default alpha scaling. Defaults to None.
        **kwargs: Additional arguments passed to Ellipse
    """
    if not isinstance(pos, np.ndarray):
        pos = to_numpy(pos)
    if not isinstance(cov, np.ndarray):
        cov = to_numpy(cov)
    ax = ax or plt.gca()
    U, s, Vt = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    width, height = 2 * np.sqrt(s)
    for nsig in range(1, 6):
        ellipse = Ellipse(
            pos,
            nsig * width,
            nsig * height,
            angle=angle,
            alpha=alpha if alpha is not None else 0.5 / nsig,
            **{k: v for k, v in kwargs.items() if k != "alpha"},
        )
        ax.add_patch(ellipse)


def scatter_mog(
    X: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    mu: Union[np.ndarray, torch.Tensor],
    cov: Union[np.ndarray, torch.Tensor],
    ax: Optional[Axes] = None,
    **kwargs,
) -> None:
    """Visualize mixture of Gaussians.

    Args:
        X (Union[np.ndarray, torch.Tensor]): Points to plot (Nx2).
        labels (Union[np.ndarray, torch.Tensor]): Labels for points.
        mu (Union[np.ndarray, torch.Tensor]): Means of Gaussians (Kx2).
        cov (Union[np.ndarray, torch.Tensor]): Covariance matrices (Kx2x2).
        ax (Optional[Axes], optional): Matplotlib axes to plot on. Defaults to None.
        **kwargs: Additional arguments passed to scatter.
    """
    if not isinstance(X, np.ndarray):
        X = to_numpy(X)
    if not isinstance(labels, np.ndarray):
        labels = to_numpy(labels)
    if not isinstance(mu, np.ndarray):
        mu = to_numpy(mu)
    if not isinstance(cov, np.ndarray):
        cov = to_numpy(cov)

    ax = ax or plt.gca()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(labels))))
    for i, l in enumerate(np.unique(labels)):
        mask = labels == l
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i : i + 1], **kwargs)
        draw_ellipse(mu[l], cov[l], ax=ax, fc=colors[i])


def visualize_particle_filter_reconstruction(
    original_particles: Union[np.ndarray, torch.Tensor],
    reconstructed_particles: Union[np.ndarray, torch.Tensor],
    ax: Optional[Tuple[Axes, Axes, Axes, Axes, Axes, Axes]] = None,
    title: Optional[str] = None,
    alpha: float = 0.6,
    **kwargs,
) -> None:
    """Visualize original and reconstructed particle filters with Gaussian estimates.

    Args:
        original_particles (Union[np.ndarray, torch.Tensor]): Original particles (N, 4)
        reconstructed_particles (Union[np.ndarray, torch.Tensor]): Reconstructed particles (N, 4)
        ax (Optional[Tuple[Axes, Axes, Axes, Axes, Axes, Axes]], optional): Tuple of matplotlib axes for original,
            reconstructed, and overlay plots for both dimension pairs. If None, creates new figure. Defaults to None.
        title (Optional[str], optional): Title for the plot. Defaults to None.
        alpha (float, optional): Alpha value for particle transparency. Defaults to 0.6.
        **kwargs: Additional arguments passed to scatter plots.

    Raises:
        ValueError: If particles don't have shape (N, 4) or if particle counts don't match.
    """
    if not isinstance(original_particles, np.ndarray):
        original_particles = to_numpy(original_particles)
    if not isinstance(reconstructed_particles, np.ndarray):
        reconstructed_particles = to_numpy(reconstructed_particles)

    # Validate dimensions
    if original_particles.shape[1] != 4:
        raise ValueError(
            f"Original particles must have shape (N, 4), got {original_particles.shape}"
        )
    if reconstructed_particles.shape[1] != 4:
        raise ValueError(
            f"Reconstructed particles must have shape (N, 4), got {reconstructed_particles.shape}"
        )

    # Validate matching particle counts
    if original_particles.shape[0] != reconstructed_particles.shape[0]:
        raise ValueError(
            f"Original particles count ({original_particles.shape[0]}) must match "
            f"reconstructed particles count ({reconstructed_particles.shape[0]})"
        )

    # Create figure if axes not provided
    if ax is None:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
    else:
        ax1, ax2, ax3, ax4, ax5, ax6 = ax

    # Process first two dimensions (0,1)
    original_particles_2d_first = original_particles[:, :2]
    reconstructed_particles_2d_first = reconstructed_particles[:, :2]

    # Process second two dimensions (2,3)
    original_particles_2d_second = original_particles[:, 2:]
    reconstructed_particles_2d_second = reconstructed_particles[:, 2:]

    # Get axis limits for first two dimensions
    all_particles_first = np.vstack(
        [original_particles_2d_first, reconstructed_particles_2d_first]
    )
    min_x1, max_x1 = all_particles_first[:, 0].min(), all_particles_first[:, 0].max()
    min_y1, max_y1 = all_particles_first[:, 1].min(), all_particles_first[:, 1].max()
    # Add 10% padding
    range_x1 = max_x1 - min_x1
    range_y1 = max_y1 - min_y1
    min_x1 -= range_x1 * 0.1
    max_x1 += range_x1 * 0.1
    min_y1 -= range_y1 * 0.1
    max_y1 += range_y1 * 0.1

    # Get axis limits for second two dimensions
    all_particles_second = np.vstack(
        [original_particles_2d_second, reconstructed_particles_2d_second]
    )
    min_x2, max_x2 = all_particles_second[:, 0].min(), all_particles_second[:, 0].max()
    min_y2, max_y2 = all_particles_second[:, 1].min(), all_particles_second[:, 1].max()
    # Add 10% padding
    range_x2 = max_x2 - min_x2
    range_y2 = max_y2 - min_y2
    min_x2 -= range_x2 * 0.1
    max_x2 += range_x2 * 0.1
    min_y2 -= range_y2 * 0.1
    max_y2 += range_y2 * 0.1

    # First row: dimensions 0,1
    # Plot 1: Original particles (dims 0,1)
    scatter(original_particles_2d_first, ax=ax1, alpha=alpha, **kwargs)
    ax1.set_title("Original Particles (dims 0,1)")
    ax1.set_xlim(min_x1, max_x1)
    ax1.set_ylim(min_y1, max_y1)

    # Compute and plot Gaussian estimate for original
    orig_mean = np.mean(original_particles_2d_first, axis=0)
    orig_cov = np.cov(original_particles_2d_first.T)
    draw_ellipse(orig_mean, orig_cov, ax=ax1, fc="none", ec="r", lw=2)

    # Add sample statistics for original
    orig_stats = f"μ=[{orig_mean[0]:.2f}, {orig_mean[1]:.2f}]\n"
    orig_stats += f"σ=[{np.sqrt(orig_cov[0,0]):.2f}, {np.sqrt(orig_cov[1,1]):.2f}]"
    ax1.text(
        0.05,
        0.95,
        orig_stats,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Plot 2: Reconstructed particles (dims 0,1)
    scatter(reconstructed_particles_2d_first, ax=ax2, alpha=alpha, **kwargs)
    ax2.set_title("Reconstructed Particles (dims 0,1)")
    ax2.set_xlim(min_x1, max_x1)
    ax2.set_ylim(min_y1, max_y1)

    # Compute and plot Gaussian estimate for reconstruction
    recon_mean = np.mean(reconstructed_particles_2d_first, axis=0)
    recon_cov = np.cov(reconstructed_particles_2d_first.T)
    draw_ellipse(recon_mean, recon_cov, ax=ax2, fc="none", ec="r", lw=2)

    # Add sample statistics for reconstruction
    recon_stats = f"μ=[{recon_mean[0]:.2f}, {recon_mean[1]:.2f}]\n"
    recon_stats += f"σ=[{np.sqrt(recon_cov[0,0]):.2f}, {np.sqrt(recon_cov[1,1]):.2f}]"
    ax2.text(
        0.05,
        0.95,
        recon_stats,
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Plot 3: Overlay (dims 0,1)
    scatter(
        original_particles_2d_first,
        ax=ax3,
        alpha=0.3,  # Lower opacity for overlay
        color="blue",
        label="Original",
        **{
            k: v for k, v in kwargs.items() if k != "alpha"
        },  # Pass all kwargs except alpha
    )
    scatter(
        reconstructed_particles_2d_first,
        ax=ax3,
        alpha=0.3,  # Lower opacity for overlay
        color="red",
        label="Reconstructed",
        **{
            k: v for k, v in kwargs.items() if k != "alpha"
        },  # Pass all kwargs except alpha
    )
    ax3.set_title("Overlay (dims 0,1)")
    ax3.set_xlim(min_x1, max_x1)
    ax3.set_ylim(min_y1, max_y1)

    # Plot both Gaussian estimates with higher opacity
    draw_ellipse(orig_mean, orig_cov, ax=ax3, fc="none", ec="blue", lw=2, alpha=0.6)
    draw_ellipse(recon_mean, recon_cov, ax=ax3, fc="none", ec="red", lw=2, alpha=0.6)

    # Add legend
    ax3.legend()

    # Second row: dimensions 2,3
    # Plot 4: Original particles (dims 2,3)
    scatter(original_particles_2d_second, ax=ax4, alpha=alpha, **kwargs)
    ax4.set_title("Original Particles (dims 2,3)")
    ax4.set_xlim(min_x2, max_x2)
    ax4.set_ylim(min_y2, max_y2)

    # Compute and plot Gaussian estimate for original
    orig_mean = np.mean(original_particles_2d_second, axis=0)
    orig_cov = np.cov(original_particles_2d_second.T)
    draw_ellipse(orig_mean, orig_cov, ax=ax4, fc="none", ec="r", lw=2)

    # Add sample statistics for original
    orig_stats = f"μ=[{orig_mean[0]:.2f}, {orig_mean[1]:.2f}]\n"
    orig_stats += f"σ=[{np.sqrt(orig_cov[0,0]):.2f}, {np.sqrt(orig_cov[1,1]):.2f}]"
    ax4.text(
        0.05,
        0.95,
        orig_stats,
        transform=ax4.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Plot 5: Reconstructed particles (dims 2,3)
    scatter(reconstructed_particles_2d_second, ax=ax5, alpha=alpha, **kwargs)
    ax5.set_title("Reconstructed Particles (dims 2,3)")
    ax5.set_xlim(min_x2, max_x2)
    ax5.set_ylim(min_y2, max_y2)

    # Compute and plot Gaussian estimate for reconstruction
    recon_mean = np.mean(reconstructed_particles_2d_second, axis=0)
    recon_cov = np.cov(reconstructed_particles_2d_second.T)
    draw_ellipse(recon_mean, recon_cov, ax=ax5, fc="none", ec="r", lw=2)

    # Add sample statistics for reconstruction
    recon_stats = f"μ=[{recon_mean[0]:.2f}, {recon_mean[1]:.2f}]\n"
    recon_stats += f"σ=[{np.sqrt(recon_cov[0,0]):.2f}, {np.sqrt(recon_cov[1,1]):.2f}]"
    ax5.text(
        0.05,
        0.95,
        recon_stats,
        transform=ax5.transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # Plot 6: Overlay (dims 2,3)
    scatter(
        original_particles_2d_second,
        ax=ax6,
        alpha=0.3,  # Lower opacity for overlay
        color="blue",
        label="Original",
        **{
            k: v for k, v in kwargs.items() if k != "alpha"
        },  # Pass all kwargs except alpha
    )
    scatter(
        reconstructed_particles_2d_second,
        ax=ax6,
        alpha=0.3,  # Lower opacity for overlay
        color="red",
        label="Reconstructed",
        **{
            k: v for k, v in kwargs.items() if k != "alpha"
        },  # Pass all kwargs except alpha
    )
    ax6.set_title("Overlay (dims 2,3)")
    ax6.set_xlim(min_x2, max_x2)
    ax6.set_ylim(min_y2, max_y2)

    # Plot both Gaussian estimates with higher opacity
    draw_ellipse(orig_mean, orig_cov, ax=ax6, fc="none", ec="blue", lw=2, alpha=0.6)
    draw_ellipse(recon_mean, recon_cov, ax=ax6, fc="none", ec="red", lw=2, alpha=0.6)

    # Add legend
    ax6.legend()

    # Add overall title if provided
    if title:
        plt.suptitle(title)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.95)  # Make room for suptitle


if __name__ == "__main__":
    """Example usage of particle filter reconstruction visualization.

    This example:
    1. Generates synthetic 4D particle data from a mixture of 3 Gaussians
    2. Creates "reconstructed" particles by adding noise and applying transformations
    3. Visualizes the original and reconstructed particles in two dimension pairs
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate particles from a mixture of 3 Gaussians
    N = 200  # Number of particles

    # Define 4D mixture components
    means = [
        np.array([1.0, 1.0, -0.5, 0.5]),
        np.array([-1.0, -1.0, 0.5, -0.5]),
        np.array([1.0, -1.0, 0.0, 0.0]),
    ]

    # Create 4x4 covariance matrices
    covs = [
        np.array(
            [
                [0.3, 0.1, 0.05, 0.02],
                [0.1, 0.2, 0.02, 0.03],
                [0.05, 0.02, 0.25, 0.08],
                [0.02, 0.03, 0.08, 0.2],
            ]
        ),
        np.array(
            [
                [0.2, -0.1, 0.03, -0.02],
                [-0.1, 0.3, -0.02, 0.04],
                [0.03, -0.02, 0.2, -0.05],
                [-0.02, 0.04, -0.05, 0.25],
            ]
        ),
        np.array(
            [
                [0.2, 0.05, 0.02, 0.01],
                [0.05, 0.2, 0.01, 0.02],
                [0.02, 0.01, 0.15, 0.03],
                [0.01, 0.02, 0.03, 0.15],
            ]
        ),
    ]
    weights = [0.4, 0.3, 0.3]  # Mixture weights

    # Generate particles
    original_particles = []
    for mean, cov, weight in zip(means, covs, weights):
        n = int(N * weight)
        particles = np.random.multivariate_normal(mean, cov, n)
        original_particles.append(particles)
    original_particles = np.vstack(original_particles)

    # Create "reconstructed" particles by:
    # 1. Adding noise
    # 2. Applying a transformation
    noise = np.random.normal(0, 0.1, original_particles.shape)

    # Create a 4D rotation-like transformation matrix
    theta1, theta2 = np.pi / 12, np.pi / 18  # Different rotation angles
    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)

    rotation = np.array(
        [[c1, -s1, 0, 0], [s1, c1, 0, 0], [0, 0, c2, -s2], [0, 0, s2, c2]]
    )

    translation = np.array([0.2, -0.1, 0.15, -0.05])

    reconstructed_particles = np.dot(original_particles + noise, rotation) + translation

    # Create visualization
    visualize_particle_filter_reconstruction(
        original_particles,
        reconstructed_particles,
        title="4D Particle Filter Reconstruction",
        s=50,  # Larger markers
        alpha=0.6,  # Transparency
        marker="o",  # Circle markers
    )

    plt.show()
