"""Mixture of Multivariate Normal Distributions.

This module implements mixture models of multivariate normal distributions,
providing functionality for sampling, computing log probabilities, and
visualizing the distributions.
"""

from typing import Optional, Tuple, Union

import torch
from torch.distributions import Dirichlet, Categorical
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from plots import scatter_mog


class MultivariateNormal:
    """Base class for multivariate normal distributions.
    
    This abstract class defines the interface for multivariate normal distributions
    that can be used in mixture models.

    Args:
        dim (int): Dimension of the distribution
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def sample(
        self, 
        B: int, 
        K: int, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Sample from the distribution.

        Args:
            B (int): Batch size
            K (int): Number of components
            labels (torch.Tensor): Component labels

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError

    def log_prob(
        self, 
        X: torch.Tensor, 
        params: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute log probability of samples.

        Args:
            X (torch.Tensor): Input samples
            params (Tuple[torch.Tensor, torch.Tensor]): Distribution parameters

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError

    def stats(
        self, 
        params: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute distribution statistics.

        Args:
            params (Tuple[torch.Tensor, torch.Tensor]): Distribution parameters

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError

    def parse(
        self, 
        raw: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Parse raw outputs into distribution parameters.

        Args:
            raw (torch.Tensor): Raw network outputs

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError


class MixtureOfMVNs:
    """Mixture of Multivariate Normal Distributions.
    
    This class implements a mixture model where each component is a multivariate
    normal distribution. It provides functionality for sampling, computing log
    probabilities, and visualization.

    Args:
        mvn (MultivariateNormal): The base multivariate normal distribution class
    """

    def __init__(self, mvn: MultivariateNormal) -> None:
        self.mvn = mvn

    def sample(
        self, 
        B: int, 
        N: int, 
        K: int, 
        return_gt: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Sample from the mixture model.

        Args:
            B (int): Batch size
            N (int): Number of samples per batch
            K (int): Number of mixture components
            return_gt (bool, optional): Whether to return ground truth values. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
                If return_gt is False:
                    - Samples X of shape (B, N, dim)
                If return_gt is True:
                    - Samples X of shape (B, N, dim)
                    - Labels of shape (B, N)
                    - Mixture weights pi of shape (B, K)
                    - Distribution parameters
        """
        device = 'cpu' if not torch.cuda.is_available() \
                else torch.cuda.current_device()
        pi = Dirichlet(torch.ones(K)).sample(torch.Size([B])).to(device)
        labels = Categorical(probs=pi).sample(torch.Size([N])).to(device)
        labels = labels.transpose(0,1).contiguous()

        X, params = self.mvn.sample(B, K, labels)
        if return_gt:
            return X, labels, pi, params
        else:
            return X

    def log_prob(
        self, 
        X: torch.Tensor, 
        pi: torch.Tensor, 
        params: Tuple[torch.Tensor, torch.Tensor], 
        return_labels: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute log probability of samples under the mixture model.

        Args:
            X (torch.Tensor): Input samples of shape (B, N, dim)
            pi (torch.Tensor): Mixture weights of shape (B, K)
            params (Tuple[torch.Tensor, torch.Tensor]): Distribution parameters
            return_labels (bool, optional): Whether to return predicted labels. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                If return_labels is False:
                    - Log probability
                If return_labels is True:
                    - Log probability
                    - Predicted labels of shape (B, N)
        """
        ll = self.mvn.log_prob(X, params)
        ll = ll + (pi + 1e-10).log().unsqueeze(-2)
        if return_labels:
            labels = ll.argmax(-1)
            return ll.logsumexp(-1).mean(), labels
        else:
            return ll.logsumexp(-1).mean()

    def plot(
        self, 
        X: torch.Tensor, 
        labels: torch.Tensor, 
        params: Tuple[torch.Tensor, torch.Tensor], 
        axes: np.ndarray
    ) -> None:
        """Plot the mixture model components and samples.

        Args:
            X (torch.Tensor): Samples of shape (B, N, dim)
            labels (torch.Tensor): Component labels of shape (B, N)
            params (Tuple[torch.Tensor, torch.Tensor]): Distribution parameters
            axes (np.ndarray): Array of matplotlib axes for plotting
        """
        mu, cov = self.mvn.stats(params)
        for i, ax in enumerate(axes.flatten()):
            scatter_mog(X[i].cpu().data.numpy(),
                    labels[i].cpu().data.numpy(),
                    mu[i].cpu().data.numpy(),
                    cov[i].cpu().data.numpy(),
                    ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.subplots_adjust(hspace=0.1, wspace=0.1)

    def parse(
        self, 
        raw: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Parse raw network outputs into mixture model parameters.

        Args:
            raw (torch.Tensor): Raw network outputs

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Parsed parameters
        """
        return self.mvn.parse(raw)
