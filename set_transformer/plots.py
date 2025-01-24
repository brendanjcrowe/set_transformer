"""Visualization utilities for mixture models and point clouds.

This module provides functions for visualizing 2D point clouds, mixture components,
and confidence ellipses. It supports both labeled and unlabeled data visualization.
"""

from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from matplotlib.axes import Axes
import torch


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
    **kwargs
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
            raise ValueError(f"Labels length {len(labels)} must match X length {len(X)}")

    ax = ax or plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], facecolor='k',
                edgecolor=[0.2, 0.2, 0.2], **kwargs)
        return None, None
    else:
        ulabels = np.sort(np.unique(labels))
        colors = cm.rainbow(np.linspace(0, 1, len(ulabels)))
        for l, c in zip(ulabels, colors):
            mask = labels == l
            ax.scatter(X[mask, 0], X[mask, 1], color=c,
                    edgecolor=c*0.6, **kwargs)
        return ulabels, colors


def draw_ellipse(
    pos: Union[np.ndarray, torch.Tensor],
    cov: Union[np.ndarray, torch.Tensor],
    ax: Optional[Axes] = None,
    **kwargs
) -> None:
    """Draw confidence ellipses for a 2D Gaussian distribution.

    Draws multiple ellipses corresponding to different confidence levels
    (1 to 5 standard deviations).

    Args:
        pos (Union[np.ndarray, torch.Tensor]): Mean of the Gaussian (2D)
        cov (Union[np.ndarray, torch.Tensor]): Covariance matrix (2x2)
        ax (Optional[Axes], optional): Matplotlib axes to plot on. Defaults to None.
        **kwargs: Additional arguments passed to Ellipse
    """
    if not isinstance(pos, np.ndarray):
        pos = to_numpy(pos)
    if not isinstance(cov, np.ndarray):
        cov = to_numpy(cov)
    ax = ax or plt.gca()
    U, s, Vt = np.linalg.svd(cov)
    angle = np.degrees(np.arctan2(U[1,0], U[0,0]))
    width, height = 2 * np.sqrt(s)
    for nsig in range(1, 6):
        ellipse = Ellipse(pos, nsig*width, nsig*height, angle=angle, alpha=0.5/nsig, **kwargs)
        ax.add_patch(ellipse)


def scatter_mog(
    X: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    mu: Union[np.ndarray, torch.Tensor],
    cov: Union[np.ndarray, torch.Tensor],
    ax: Optional[Axes] = None,
    **kwargs
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
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i:i+1], **kwargs)
        draw_ellipse(mu[l], cov[l], ax=ax, fc=colors[i])
