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
    ax: Optional[Tuple[Axes, Axes, Axes]] = None,
    title: Optional[str] = None,
    alpha: float = 0.6,
    **kwargs,
) -> None:
    """Visualize original and reconstructed particle filters with Gaussian estimates.

    Args:
        original_particles (Union[np.ndarray, torch.Tensor]): Original particles (Nx2)
        reconstructed_particles (Union[np.ndarray, torch.Tensor]): Reconstructed particles (Nx2)
        ax (Optional[Tuple[Axes, Axes, Axes]], optional): Tuple of matplotlib axes for original,
            reconstructed, and overlay plots. If None, creates new figure. Defaults to None.
        title (Optional[str], optional): Title for the plot. Defaults to None.
        alpha (float, optional): Alpha value for particle transparency. Defaults to 0.6.
        **kwargs: Additional arguments passed to scatter plots.

    Raises:
        ValueError: If particles don't have shape (N, 2) or if particle counts don't match.
    """
    if not isinstance(original_particles, np.ndarray):
        original_particles = to_numpy(original_particles)
    if not isinstance(reconstructed_particles, np.ndarray):
        reconstructed_particles = to_numpy(reconstructed_particles)

    # Validate dimensions
    if original_particles.shape[1] != 2:
        raise ValueError(
            f"Original particles must have shape (N, 2), got {original_particles.shape}"
        )
    if reconstructed_particles.shape[1] != 2:
        raise ValueError(
            f"Reconstructed particles must have shape (N, 2), got {reconstructed_particles.shape}"
        )

    # Validate matching particle counts
    if original_particles.shape[0] != reconstructed_particles.shape[0]:
        raise ValueError(
            f"Original particles count ({original_particles.shape[0]}) must match "
            f"reconstructed particles count ({reconstructed_particles.shape[0]})"
        )

    # Create figure if axes not provided
    if ax is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    else:
        ax1, ax2, ax3 = ax

    # Get axis limits that encompass all particles
    all_particles = np.vstack([original_particles, reconstructed_particles])
    min_x, max_x = all_particles[:, 0].min(), all_particles[:, 0].max()
    min_y, max_y = all_particles[:, 1].min(), all_particles[:, 1].max()
    # Add 10% padding
    range_x = max_x - min_x
    range_y = max_y - min_y
    min_x -= range_x * 0.1
    max_x += range_x * 0.1
    min_y -= range_y * 0.1
    max_y += range_y * 0.1

    # Plot 1: Original particles
    scatter(original_particles, ax=ax1, alpha=alpha, **kwargs)
    ax1.set_title("Original Particles")
    ax1.set_xlim(min_x, max_x)
    ax1.set_ylim(min_y, max_y)

    # Compute and plot Gaussian estimate for original
    orig_mean = np.mean(original_particles, axis=0)
    orig_cov = np.cov(original_particles.T)
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

    # Plot 2: Reconstructed particles
    scatter(reconstructed_particles, ax=ax2, alpha=alpha, **kwargs)
    ax2.set_title("Reconstructed Particles")
    ax2.set_xlim(min_x, max_x)
    ax2.set_ylim(min_y, max_y)

    # Compute and plot Gaussian estimate for reconstruction
    recon_mean = np.mean(reconstructed_particles, axis=0)
    recon_cov = np.cov(reconstructed_particles.T)
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

    # Plot 3: Overlay
    scatter(
        original_particles,
        ax=ax3,
        alpha=0.3,  # Lower opacity for overlay
        color="blue",
        label="Original",
        **{
            k: v for k, v in kwargs.items() if k != "alpha"
        },  # Pass all kwargs except alpha
    )
    scatter(
        reconstructed_particles,
        ax=ax3,
        alpha=0.3,  # Lower opacity for overlay
        color="red",
        label="Reconstructed",
        **{
            k: v for k, v in kwargs.items() if k != "alpha"
        },  # Pass all kwargs except alpha
    )
    ax3.set_title("Overlay")
    ax3.set_xlim(min_x, max_x)
    ax3.set_ylim(min_y, max_y)

    # Plot both Gaussian estimates with higher opacity
    draw_ellipse(orig_mean, orig_cov, ax=ax3, fc="none", ec="blue", lw=2, alpha=0.6)
    draw_ellipse(recon_mean, recon_cov, ax=ax3, fc="none", ec="red", lw=2, alpha=0.6)

    # Add legend
    ax3.legend()

    # Add overall title if provided
    if title:
        plt.suptitle(title)


if __name__ == "__main__":
    """Example usage of particle filter reconstruction visualization.

    This example:
    1. Generates synthetic particle data from a mixture of 3 Gaussians
    2. Creates "reconstructed" particles by adding noise and applying a transformation
    3. Visualizes the original and reconstructed particles side by side with overlay
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate particles from a mixture of 3 Gaussians
    N = 200  # Number of particles

    # Define mixture components
    means = [np.array([1.0, 1.0]), np.array([-1.0, -1.0]), np.array([1.0, -1.0])]
    covs = [
        np.array([[0.3, 0.1], [0.1, 0.2]]),
        np.array([[0.2, -0.1], [-0.1, 0.3]]),
        np.array([[0.2, 0.05], [0.05, 0.2]]),
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
    # 2. Applying a small rotation
    # 3. Adding a small translation
    noise = np.random.normal(0, 0.1, original_particles.shape)
    theta = np.pi / 12  # 15-degree rotation
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    translation = np.array([0.2, -0.1])

    reconstructed_particles = np.dot(original_particles + noise, rotation) + translation

    # Create visualization
    visualize_particle_filter_reconstruction(
        original_particles,
        reconstructed_particles,
        title="Particle Filter Reconstruction",
        s=50,  # Larger markers
        alpha=0.6,  # Transparency
        marker="o",  # Circle markers
    )

    plt.tight_layout()
    plt.show()
