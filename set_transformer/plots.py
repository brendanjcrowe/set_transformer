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


def scatter(
    X: npt.NDArray, 
    labels: Optional[npt.NDArray] = None, 
    ax: Optional[Axes] = None, 
    colors: Optional[npt.NDArray] = None, 
    **kwargs
) -> Optional[Tuple[npt.NDArray, npt.NDArray]]:
    """Plot scatter points with optional color coding by labels.

    Args:
        X (npt.NDArray): Points to plot of shape (N, 2)
        labels (Optional[npt.NDArray], optional): Labels for coloring points. Defaults to None.
        ax (Optional[Axes], optional): Matplotlib axes to plot on. Defaults to None.
        colors (Optional[npt.NDArray], optional): Colors for each label. Defaults to None.
        **kwargs: Additional arguments passed to plt.scatter

    Returns:
        Optional[Tuple[npt.NDArray, npt.NDArray]]: If labels provided, returns:
            - Unique labels
            - Colors used for each label
    """
    ax = ax or plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    if labels is None:
        ax.scatter(X[:,0], X[:,1], facecolor='k',
                edgecolor=[0.2, 0.2, 0.2], **kwargs)
        return None
    else:
        ulabels = np.sort(np.unique(labels))
        colors = cm.rainbow(np.linspace(0, 1, len(ulabels))) \
                if colors is None else colors
        for (l, c) in zip(ulabels, colors):
            ax.scatter(X[labels==l,0], X[labels==l,1], color=c,
                    edgecolor=c*0.6, **kwargs)
        return ulabels, colors


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array.

    Detaches the tensor from the computation graph and moves it to CPU before
    converting to numpy.

    Args:
        tensor (torch.Tensor): PyTorch tensor to convert

    Returns:
        np.ndarray: Numpy array containing the same data
    """
    return tensor.detach().cpu().numpy()


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
        ax.add_patch(Ellipse(pos, nsig*width, nsig*height, angle,
            alpha=0.5/nsig, **kwargs))


def scatter_mog(
    X: npt.NDArray, 
    labels: npt.NDArray, 
    mu: npt.NDArray, 
    cov: npt.NDArray, 
    ax: Optional[Axes] = None, 
    colors: Optional[npt.NDArray] = None
) -> None:
    """Visualize a mixture of Gaussians with data points.

    Plots data points colored by their cluster assignments and confidence
    ellipses for each Gaussian component.

    Args:
        X (npt.NDArray): Data points of shape (N, 2)
        labels (npt.NDArray): Cluster assignments for each point
        mu (npt.NDArray): Means of Gaussian components
        cov (npt.NDArray): Covariance matrices of Gaussian components
        ax (Optional[Axes], optional): Matplotlib axes to plot on. Defaults to None.
        colors (Optional[npt.NDArray], optional): Colors for each component. Defaults to None.
    """
    ax = ax or plt.gca()
    ulabels, colors = scatter(X, labels=labels, ax=ax, colors=colors, zorder=10)
    for i, l in enumerate(ulabels):
        draw_ellipse(mu[l], cov[l], ax=ax, fc=colors[i])
