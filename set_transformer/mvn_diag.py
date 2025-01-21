"""Multivariate Normal Distribution with Diagonal Covariance.

This module implements a multivariate normal distribution with diagonal covariance matrix,
which is a special case of the general multivariate normal distribution where all
off-diagonal elements of the covariance matrix are zero.
"""

from typing import Tuple, Union

import torch
import torch.nn.functional as F
import math

from mixture_of_mvns import MultivariateNormal


class MultivariateNormalDiag(MultivariateNormal):
    """Multivariate Normal Distribution with Diagonal Covariance Matrix.

    This class implements a multivariate normal distribution where the covariance
    matrix is diagonal, meaning all variables are independent. This is a simpler
    and more computationally efficient version of the general multivariate normal.

    Args:
        dim (int): Dimension of the distribution
    """

    def __init__(self, dim: int) -> None:
        super(MultivariateNormalDiag, self).__init__(dim)

    def sample(
        self, 
        B: int, 
        K: int, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples from the diagonal multivariate normal distribution.

        Args:
            B (int): Batch size
            K (int): Number of components
            labels (torch.Tensor): Component labels of shape (B, N)

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: A tuple containing:
                - Samples of shape (B, N, dim)
                - Parameters (mu, sigma) where:
                    - mu has shape (B, K, dim)
                    - sigma has shape (B, K, dim)
        """
        N = labels.shape[-1]
        device = labels.device
        mu = -4 + 8*torch.rand(B, K, self.dim).to(device)
        sigma = 0.3*torch.ones(B, K, self.dim).to(device)
        eps = torch.randn(B, N, self.dim).to(device)

        rlabels = labels.unsqueeze(-1).repeat(1, 1, self.dim)
        X = torch.gather(mu, 1, rlabels) + \
                eps * torch.gather(sigma, 1, rlabels)
        return X, (mu, sigma)

    def log_prob(
        self, 
        X: torch.Tensor, 
        params: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute log probability of samples under the distribution.

        Args:
            X (torch.Tensor): Samples of shape (B, N, dim)
            params (Tuple[torch.Tensor, torch.Tensor]): Distribution parameters (mu, sigma)
                where mu and sigma have shapes (B, K, dim)

        Returns:
            torch.Tensor: Log probabilities of shape (B, N, K)
        """
        mu, sigma = params
        dim = self.dim
        X = X.unsqueeze(2)
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)
        diff = X - mu
        ll = -0.5*math.log(2*math.pi) - sigma.log() - 0.5*(diff.pow(2)/sigma.pow(2))
        return ll.sum(-1)

    def stats(
        self, 
        params: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and covariance matrix from distribution parameters.

        Args:
            params (Tuple[torch.Tensor, torch.Tensor]): Distribution parameters (mu, sigma)
                where mu and sigma have shapes (B, K, dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Mean of shape (B, K, dim)
                - Covariance matrix of shape (B, K, dim, dim)
        """
        mu, sigma = params
        I = torch.eye(self.dim)[(None,)*(len(sigma.shape)-1)].to(sigma.device)
        cov = sigma.pow(2).unsqueeze(-1) * I
        return mu, cov

    def parse(
        self, 
        raw: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Parse raw network outputs into distribution parameters.

        Args:
            raw (torch.Tensor): Raw network outputs of shape (..., 1 + 2*dim)
                where the first element is used for mixture weights (pi),
                the next dim elements for mean (mu),
                and the last dim elements for standard deviation (sigma)

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: A tuple containing:
                - Mixture weights (pi) of shape (..., K)
                - Parameters (mu, sigma) where both have shape (..., dim)
        """
        pi = torch.softmax(raw[...,0], -1)
        mu = raw[...,1:1+self.dim]
        sigma = F.softplus(raw[...,1+self.dim:])
        return pi, (mu, sigma)
