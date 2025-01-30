"""Loss functions for set-based models.

This module provides various loss functions specifically designed for comparing sets,
including Chamfer Distance and Sinkhorn (Optimal Transport) losses.
"""

from typing import Callable, Dict, Literal, Optional

import numpy as np
import ot
import torch
import torch.nn as nn
from geomloss import SamplesLoss

reduction_map: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": torch.mean,
    "sum": torch.sum,
    "max": torch.max,
    "min": torch.min,
    "none": lambda x: x,
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
        reduction: Literal["mean", "sum", "max", "min", "none"] = "mean",
        near_neighbor_alg: Literal["pairwise"] = "pairwise",
    ) -> None:
        """
        Initialize the ChamferDistanceLoss class.

        Args:
            reduction (str): Reduction method for the loss. One of: "mean", "sum", "max", "min".
            near_neighbor_alg (str): Algorithm for nearest neighbor search. Currently only "pairwise" is supported.

        Raises:
            TypeError: If reduction is not a string.
            ValueError: If reduction is not one of the supported methods.
        """
        super(ChamferDistanceLoss, self).__init__()
        if reduction is None:
            reduction = "none"
        if not isinstance(reduction, str):
            raise TypeError(
                f"reduction must be of type str, got type {type(reduction)}"
            )
        if reduction not in list(reduction_map.keys()):
            raise ValueError(
                f"reduction must be one of: {list(reduction_map.keys())}, got {reduction}"
            )

        self.reduction = reduction_map[reduction]
        self.near_neighbor_alg = None  # TODO Implement KDTrees as alternative algorithm

    def forward(
        self, predicted_set: torch.Tensor, target_set: torch.Tensor
    ) -> torch.Tensor:
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


class SampleLoss(nn.Module):
    """Sample Loss for comparing two point sets.

    This loss computes the average distance between each point in one set to its nearest neighbor in the other set.
    """

    def __init__(
        self,
        p: int = 2,
        blur: float = 0.5,
        reduction: Literal["mean", "sum", "max", "min", "none"] = "mean",
    ) -> None:
        super(SampleLoss, self).__init__()
        if reduction is None:
            reduction = "none"
        if not isinstance(reduction, str):
            raise TypeError(
                f"reduction must be of type str, got type {type(reduction)}"
            )
        if reduction not in list(reduction_map.keys()):
            raise ValueError(
                f"reduction must be one of: {list(reduction_map.keys())}, got {reduction}"
            )
        self.reduction = reduction_map[reduction]
        self.loss_function = None

    def forward(
        self, predicted_set: torch.Tensor, target_set: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_function is None:
            raise ValueError("Loss function not set")
        loss = self.loss_function(predicted_set, target_set)
        return self.reduction(loss)


class SinkhornLoss(SampleLoss):
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
        reduction: Literal["mean", "sum", "max", "min", "none"] = "mean",
    ) -> None:
        """
        Initialize the SinkhornLoss class.

        Args:
            p (int): Power for the ground cost (e.g., 2 for squared Euclidean).
            blur (float): Regularization parameter for the Sinkhorn algorithm.
            reduction (str): Reduction method for the loss. One of: "mean", "sum", "max", "min".

        Raises:
            TypeError: If reduction is not a string.
            ValueError: If reduction is not one of the supported methods.
        """
        super(SinkhornLoss, self).__init__(p=p, blur=blur, reduction=reduction)
        self.loss_function = SamplesLoss(loss="sinkhorn", p=p, blur=blur)

    def __str__(self) -> str:
        """
        Return a string representation of the SinkhornLoss class.

        Returns:
            str: String representation of the SinkhornLoss class.
        """
        return "Sinkhorn Loss"


class HausdorffLoss(SampleLoss):
    """Hausdorff Loss for comparing two point sets.

    This loss computes the Hausdorff distance between two point sets.
    """

    def __init__(
        self,
        p: int = 2,
        blur: float = 0.5,
        reduction: Literal["mean", "sum", "max", "min", "none"] = "mean",
    ) -> None:
        """
        Initialize the HausdorffLoss class.

        Args:
            p (int): Power for the ground cost (e.g., 2 for squared Euclidean).
            blur (float): Regularization parameter for the Sinkhorn algorithm.
            reduction (str): Reduction method for the loss. One of: "mean", "sum", "max", "min".
        """
        super(HausdorffLoss, self).__init__(p=p, blur=blur, reduction=reduction)
        self.loss_function = SamplesLoss(loss="hausdorff", p=p, blur=blur)

    def __str__(self) -> str:
        """
        Return a string representation of the HausdorffLoss class.

        Returns:
            str: String representation of the HausdorffLoss class.
        """
        return "Hausdorff Loss"


class EarthMoverDistanceLoss(object):
    """Earth Mover Distance Loss for comparing two point sets.

    This loss computes the Earth Mover's Distance (Wasserstein distance) between two point sets.

    Note: This is a non-differentiable loss function. Cannot be used with backpropagation.
    """

    def __init__(self) -> None:
        """
        Initialize the EarthMoverDistanceLoss class.
        """
        pass

    def forward(
        self,
        predicted_set: torch.Tensor,
        target_set: torch.Tensor,
        predicted_weights: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Earth Mover Distance between predicted and target sets.

        Args:
            predicted_set (torch.Tensor): Predicted set of points of shape (batch_size, num_points, dim)
            target_set (torch.Tensor): Target set of points of shape (batch_size, num_points, dim)
            predicted_weights (torch.Tensor): Weights for the predicted set of shape (batch_size, num_points)
            target_weights (torch.Tensor): Weights for the target set of shape (batch_size, num_points)
        """
        X, Y = predicted_set.cpu().numpy(), target_set.cpu().numpy()
        if predicted_weights is None:
            predicted_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            predicted_weights = predicted_weights.cpu().numpy()
        if target_weights is None:
            target_weights = np.ones(Y.shape[0]) / Y.shape[0]
        else:
            target_weights = target_weights.cpu().numpy()
        cost_matrix = np.linalg.norm(X[:, None] - Y[None, :], axis=2)
        loss = ot.emd2(predicted_weights, target_weights, cost_matrix)
        return self.reduction(loss)

    def __str__(self) -> str:
        """
        Return a string representation of the EarthMoverDistanceLoss class.

        Returns:
            str: String representation of the EarthMoverDistanceLoss class.
        """
        return "Earth Mover Distance Loss"
