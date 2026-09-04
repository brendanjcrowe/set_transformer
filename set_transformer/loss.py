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
        self,
        predicted_set: torch.Tensor,
        target_set: torch.Tensor,
        predicted_weights: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the Chamfer Distance between predicted and target sets.

        Raises:
            NotImplementedError: If per-point weights are supplied. Chamfer
                distance is built from nearest-neighbour distances, which carry
                no notion of mass: there is no weighting of them that yields a
                metric between measures. Weighting only the forward term is a
                heuristic, and silently ignoring the weights would mean training
                on a different objective than the caller asked for. Use
                SinkhornLoss (or EarthMoverDistanceLoss for evaluation) for
                weighted particle sets.
        """
        if predicted_weights is not None or target_weights is not None:
            raise NotImplementedError(
                "ChamferDistanceLoss does not support weighted point sets. "
                "Use SinkhornLoss for a weighted (measure-to-measure) "
                "objective, or resample the particles by weight so the sets "
                "are uniform."
            )
        return self._unweighted_forward(predicted_set, target_set)

    def _unweighted_forward(
        self, predicted_set: torch.Tensor, target_set: torch.Tensor
    ) -> torch.Tensor:
        """Chamfer distance between two uniform point sets.

        Args:
            predicted_set (torch.Tensor): Predicted set of points of shape (batch_size, num_points, dim),
                or (num_points, dim) for a single unbatched set.
            target_set (torch.Tensor): Target set of points of the same shape.

        Returns:
            torch.Tensor: Computed Chamfer Distance after applying the reduction method

        Note:
            The two reductions are taken over the LAST two axes of the pairwise
            distance matrix, never over the batch axis. `cdist` returns
            [..., num_predicted, num_target], so the predicted->target term is
            a min over dim=-1 and the target->predicted term a min over dim=-2.
            An earlier version used dim=1 and dim=0, which is correct only for
            an unbatched [num_points, dim] input; on a batched input dim=0 min'd
            across BATCH ELEMENTS, mixing unrelated sets into every gradient and
            leaving the loss non-zero (0.85 on 4x100x2) for a perfect
            reconstruction. Negative axes keep both input ranks correct.
        """
        pairwise_distance = torch.cdist(predicted_set, target_set)
        forward_distance = torch.min(pairwise_distance, dim=-1)[0]
        backward_distance = torch.min(pairwise_distance, dim=-2)[0]
        return self.reduction(forward_distance) + self.reduction(backward_distance)


class SampleLoss(nn.Module):
    """Sample Loss for comparing two point sets.

    This loss computes the average distance between each point in one set to its nearest neighbor in the other set.
    """

    def __init__(
        self,
        p: int = 2,
        blur: float = 0.05,
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

    @staticmethod
    def _as_measure_weights(
        weights: Optional[torch.Tensor], points: torch.Tensor
    ) -> torch.Tensor:
        """Return a normalized, non-negative mass vector for `points`.

        `None` means "uniform", i.e. the ordinary unweighted point set. Any
        supplied vector is clamped non-negative and renormalized per set, since
        an optimal-transport problem is only defined between two measures of
        equal total mass.

        Non-finite entries and zero-total-mass sets raise. Neither can come
        from a working particle filter, whose weights are normalized and
        epsilon-guarded, so they mean the data is broken. Substituting uniform
        would silently train on a belief the data does not contain, and simply
        dividing by (sum + eps) is worse still: for a set whose weights are all
        around 1e-30 the epsilon dominates the true sum and the "normalized"
        measure ends up with total mass ~1e-17, which geomloss happily turns
        into a number in the thousands.
        """
        if weights is None:
            n = points.shape[-2]
            return points.new_full(points.shape[:-1], 1.0 / n)
        if not bool(torch.isfinite(weights).all()):
            raise ValueError(
                "Particle weights contain NaN or inf; an optimal-transport "
                "loss has no meaning for them"
            )
        weights = torch.clamp(weights, min=0.0)
        total = weights.sum(dim=-1, keepdim=True)
        if bool((total <= 0).any()):
            raise ValueError(
                "Particle weights sum to zero for at least one set, so there "
                "is no probability measure to transport. Check the particle "
                "filter, and check for float32 underflow if the dataset was "
                "written from float64 weights."
            )
        return weights / total

    def forward(
        self,
        predicted_set: torch.Tensor,
        target_set: torch.Tensor,
        predicted_weights: Optional[torch.Tensor] = None,
        target_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compare two point sets, optionally as weighted measures.

        With both weight arguments left as None this is the ordinary
        set-to-set loss and geomloss is called in its two-argument form, so
        existing behavior is bit-for-bit unchanged.

        Supplying weights switches to geomloss's four-argument form,
        ``loss(alpha, x, beta, y)``, which compares the measures
        ``sum_i alpha_i delta(x_i)`` and ``sum_j beta_j delta(y_j)``. The mass
        then lives in the MEASURE and never in the ground metric: two particles
        at the same coordinates cost nothing to transport between, whatever
        their weights. Appending a weight as an extra coordinate would instead
        add mass differences into the distance, mixing probability units into a
        spatial metric.

        Args:
            predicted_set: Points of shape (batch, num_points, dim).
            target_set: Points of shape (batch, num_points, dim).
            predicted_weights: Optional per-point mass, shape (batch, num_points).
            target_weights: Optional per-point mass, shape (batch, num_points).
        """
        if self.loss_function is None:
            raise ValueError("Loss function not set")
        if predicted_weights is None and target_weights is None:
            loss = self.loss_function(predicted_set, target_set)
        else:
            loss = self.loss_function(
                self._as_measure_weights(predicted_weights, predicted_set),
                predicted_set,
                self._as_measure_weights(target_weights, target_set),
                target_set,
            )
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
        blur: float = 0.05,
        reduction: Literal["mean", "sum", "max", "min", "none"] = "mean",
        scaling: float = 0.5,
    ) -> None:
        """
        Initialize the SinkhornLoss class.

        Args:
            p (int): Power for the ground cost (e.g., 2 for squared Euclidean).
            blur (float): Sinkhorn regularization, in the SAME UNITS as the
                point coordinates. It is the length scale below which the loss
                stops distinguishing points, so it must be well under the
                smallest structure the reconstruction should resolve. The
                default 0.05 matches geomloss's own default; the previous 0.5
                erased any belief structure finer than half a coordinate unit.
            reduction (str): Reduction method for the loss. One of: "mean", "sum", "max", "min".
            scaling (float): Geometric decay of the epsilon-scaling schedule,
                between 0 and 1. Closer to 1 is more accurate and slower.

        Raises:
            TypeError: If reduction is not a string.
            ValueError: If reduction is not one of the supported methods.
        """
        super(SinkhornLoss, self).__init__(p=p, blur=blur, reduction=reduction)
        self.loss_function = SamplesLoss(
            loss="sinkhorn", p=p, blur=blur, scaling=scaling
        )

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
        blur: float = 0.05,
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


class EarthMoverDistanceLoss(nn.Module):
    """Earth Mover Distance Loss for comparing two point sets.

    This loss computes the Earth Mover's Distance (Wasserstein distance) between two point sets.

    Note: This is a non-differentiable loss function. Cannot be used with backpropagation.

    Args:
        reduction (str): Reduction method for the loss. One of: "mean", "sum", "max", "min".

    Raises:
        TypeError: If reduction is not a string.
        ValueError: If reduction is not one of the supported methods.
    """

    def __init__(
        self,
        reduction: Literal["mean", "sum", "max", "min", "none"] = "mean",
    ) -> None:
        """
        Initialize the EarthMoverDistanceLoss class.

        Args:
            reduction (str): Reduction method for the loss. One of: "mean", "sum", "max", "min".

        Raises:
            TypeError: If reduction is not a string.
            ValueError: If reduction is not one of the supported methods.
        """
        super(EarthMoverDistanceLoss, self).__init__()
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

        Returns:
            torch.Tensor: Computed Earth Mover Distance after applying the reduction method
        """
        batch_size = predicted_set.size(0)
        losses = []

        for i in range(batch_size):
            X = predicted_set[i].detach().cpu().numpy()
            Y = target_set[i].detach().cpu().numpy()

            # Normalized through the same contract as the Sinkhorn path, so
            # the eval metric and the training loss cannot disagree about what
            # the weights mean. POT's emd2 additionally requires both marginals
            # to sum to the same total, which raw PF weights need not do after
            # a float32 round trip.
            p_weights = SampleLoss._as_measure_weights(
                predicted_weights[i:i + 1] if predicted_weights is not None else None,
                predicted_set[i:i + 1],
            )[0].detach().cpu().numpy().astype(np.float64)
            t_weights = SampleLoss._as_measure_weights(
                target_weights[i:i + 1] if target_weights is not None else None,
                target_set[i:i + 1],
            )[0].detach().cpu().numpy().astype(np.float64)

            cost_matrix = np.linalg.norm(X[:, None] - Y[None, :], axis=2)
            loss = ot.emd2(p_weights, t_weights, cost_matrix)
            losses.append(loss)

        losses = torch.tensor(losses, device=predicted_set.device)
        return self.reduction(losses)

    def __str__(self) -> str:
        """
        Return a string representation of the EarthMoverDistanceLoss class.

        Returns:
            str: String representation of the EarthMoverDistanceLoss class.
        """
        return "Earth Mover Distance Loss"
