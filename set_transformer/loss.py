
import torch
import torch.nn as nn
from geomloss import SamplesLoss

reduction_map = {
    "mean": torch.mean,
    "sum": torch.sum,
    "max": torch.max,
    "min": torch.min,
}

class ChamferDistanceLoss(nn.Module):
    def __init__(self, reduction: str = "mean", near_neighbor_alg: str = "pairwise") -> None:
        super(ChamferDistanceLoss, self).__init__()
        if not isinstance(reduction, str):
            raise TypeError(f"reduction must be of type str, got type {type(reduction)}")
        if reduction not in list(reduction_map.keys()):
            raise ValueError(f"reduction must be one of: {list(reduction_map.keys())}, got {reduction}")

        self.reduction = reduction_map[reduction]
        self.near_neighbor_alg = None # TODO Implement KDTrees as alternative algorithm


    def forward(self, predicted_set: torch.Tensor, target_set: torch.Tensor) -> torch.Tensor:
        pairwise_distance = torch.cdist(predicted_set, target_set)
        forward_distance = torch.min(pairwise_distance, dim=1)[0]
        backward_distance = torch.min(pairwise_distance, dim=0)[0]
        return self.reduction(forward_distance) + self.reduction(backward_distance)

class SinkhornLoss(nn.Module):
    def __init__(self, p: int = 2, blur: float = 0.5, reduction: str = "mean") -> None:
        super(SinkhornLoss, self).__init__()
        if not isinstance(reduction, str):
            raise TypeError(f"reduction must be of type str, got type {type(reduction)}")
        if reduction not in list(reduction_map.keys()):
            raise ValueError(f"reduction must be one of: {list(reduction_map.keys())}, got {reduction}")
        self.reduction = reduction_map[reduction]
        self.sinkhorn = SamplesLoss(loss="sinkhorn", p=p, blur=blur)


    def forward(self, predicted_set: torch.Tensor, target_set: torch.Tensor):
        loss = self.sinkhorn(predicted_set, target_set)
        return self.reduction(loss)

    def __str__(self):
        return "Sinkhorn Loss"