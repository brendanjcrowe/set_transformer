"""Loss functions for set-based models.

This module provides various loss functions specifically designed for comparing sets,
including Chamfer Distance and Sinkhorn (Optimal Transport) losses.
"""

from typing import Callable, Dict, Literal

import torch
import torch.nn as nn
from geomloss import SamplesLoss

reduction_map: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": torch.mean,
    "sum": torch.sum,
    "max": torch.max,
    "min": torch.min,
}


class ChamferDistanceLoss(nn.Module):
    """Chamfer Distance Loss for comparing two point sets.

    The Chamfer Distance measures the average distance between each point in one set
    to its nearest neighbor in the other set, and vice versa. It is bidirectional
    and invariant to the order of points within each set.

    Args:
        reduction (str): Reduction method for the loss. One of: "mean", "sum", "max", "min".
        near_neighbor_alg (str): Algorithm for nearest neighbor search. Currently only "pairwise" is supported.

    Raises:
        TypeError: If reduction is not a string.
        ValueError: If reduction is not one of the supported methods.
    """

    def __init__(
        self, 
        reduction: Literal["mean", "sum", "max", "min"] = "mean", 
        near_neighbor_alg: Literal["pairwise"] = "pairwise"
    ) -> None:
        super(ChamferDistanceLoss, self).__init__()
        if not isinstance(reduction, str):
            raise TypeError(f"reduction must be of type str, got type {type(reduction)}")
        if reduction not in list(reduction_map.keys()):
            raise ValueError(f"reduction must be one of: {list(reduction_map.keys())}, got {reduction}")

        self.reduction = reduction_map[reduction]
        self.near_neighbor_alg = None  # TODO Implement KDTrees as alternative algorithm

    def forward(self, predicted_set: torch.Tensor, target_set: torch.Tensor) -> torch.Tensor:
        """Compute the Chamfer Distance between predicted and target sets.

        Args:
            predicted_set (torch.Tensor): Predicted set of points of shape (batch_size, num_points, dim)
            target_set (torch.Tensor): Target set of points of shape (batch_size, num_points, dim)

        Returns:
            torch.Tensor: Computed Chamfer Distance after applying the reduction method
        """
        pairwise_distance = torch.cdist(predicted_set, target_set)
        forward_distance = torch.min(pairwise_distance, dim=1)[0]
        backward_distance = torch.min(pairwise_distance, dim=0)[0]
        return self.reduction(forward_distance) + self.reduction(backward_distance)


class SinkhornLoss(nn.Module):
    """Sinkhorn (Optimal Transport) Loss for comparing two point sets.

    This loss computes the optimal transport distance between two point sets using
    the Sinkhorn algorithm. It provides a differentiable approximation of the
    Earth Mover's Distance (Wasserstein distance).

    Args:
        p (int): Power for the ground cost (e.g., 2 for squared Euclidean).
        blur (float): Regularization parameter for the Sinkhorn algorithm.
        reduction (str): Reduction method for the loss. One of: "mean", "sum", "max", "min".

    Raises:
        TypeError: If reduction is not a string.
        ValueError: If reduction is not one of the supported methods.
    """

    def __init__(
        self, 
        p: int = 2, 
        blur: float = 0.5, 
        reduction: Literal["mean", "sum", "max", "min"] = "mean"
    ) -> None:
        super(SinkhornLoss, self).__init__()
        if not isinstance(reduction, str):
            raise TypeError(f"reduction must be of type str, got type {type(reduction)}")
        if reduction not in list(reduction_map.keys()):
            raise ValueError(f"reduction must be one of: {list(reduction_map.keys())}, got {reduction}")
        self.reduction = reduction_map[reduction]
        self.sinkhorn = SamplesLoss(loss="sinkhorn", p=p, blur=blur)

    def forward(self, predicted_set: torch.Tensor, target_set: torch.Tensor) -> torch.Tensor:
        """Compute the Sinkhorn distance between predicted and target sets.

        Args:
            predicted_set (torch.Tensor): Predicted set of points of shape (batch_size, num_points, dim)
            target_set (torch.Tensor): Target set of points of shape (batch_size, num_points, dim)

        Returns:
            torch.Tensor: Computed Sinkhorn distance after applying the reduction method
        """
        loss = self.sinkhorn(predicted_set, target_set)
        return self.reduction(loss)

    def __str__(self) -> str:
        return "Sinkhorn Loss"