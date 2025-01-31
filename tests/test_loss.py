import numpy as np
import pytest
import torch

from set_transformer.loss import (
    ChamferDistanceLoss,
    EarthMoverDistanceLoss,
    HausdorffLoss,
    SampleLoss,
    SinkhornLoss,
)


@pytest.fixture
def batch_size() -> int:
    return 8


@pytest.fixture
def seq_length() -> int:
    return 100


@pytest.fixture
def dim() -> int:
    return 3


@pytest.fixture
def sample_sets(
    batch_size: int, seq_length: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    # Create two sets of points with matching batch sizes
    predicted = torch.randn(batch_size, seq_length, dim)
    target = torch.randn(batch_size, seq_length, dim)
    return predicted, target


def test_chamfer_distance(sample_sets: tuple[torch.Tensor, torch.Tensor]) -> None:
    predicted, target = sample_sets
    loss_fn = ChamferDistanceLoss()
    loss = loss_fn(predicted, target)

    assert loss is not None
    assert not torch.isnan(loss).any()


def test_sinkhorn_loss(sample_sets: tuple[torch.Tensor, torch.Tensor]) -> None:
    predicted, target = sample_sets
    loss_fn = SinkhornLoss(p=2, blur=0.5)
    loss = loss_fn(predicted, target)

    assert loss is not None
    assert not torch.isnan(loss).any()


def test_hausdorff_loss(sample_sets: tuple[torch.Tensor, torch.Tensor]) -> None:
    predicted, target = sample_sets
    loss_fn = HausdorffLoss(p=2, blur=0.5)
    # Set the kernel name explicitly

    loss = loss_fn(predicted, target)

    assert loss is not None
    assert not torch.isnan(loss).any()


def test_emd_loss(sample_sets: tuple[torch.Tensor, torch.Tensor]) -> None:
    predicted, target = sample_sets
    loss_fn = EarthMoverDistanceLoss()

    # Ensure weights sum to 1 for each set and have correct shape
    batch_size, num_points = predicted.shape[:2]
    pred_weights = torch.ones(batch_size, num_points) / num_points
    target_weights = torch.ones(batch_size, num_points) / num_points

    # Squeeze any extra dimensions
    pred_weights = pred_weights.squeeze()
    target_weights = target_weights.squeeze()

    loss = loss_fn.forward(predicted, target, pred_weights, target_weights)
    assert loss is not None
    assert not torch.isnan(loss)


def test_sample_loss_inheritance() -> None:
    sinkhorn = SinkhornLoss()
    hausdorff = HausdorffLoss()

    assert isinstance(sinkhorn, SampleLoss)
    assert isinstance(hausdorff, SampleLoss)


def test_device_compatibility(sample_sets: tuple[torch.Tensor, torch.Tensor]) -> None:
    if torch.cuda.is_available():
        predicted, target = sample_sets
        device = torch.device("cuda")

        # Move tensors to GPU
        predicted = predicted.to(device)
        target = target.to(device)

        # Test ChamferDistance only for now
        loss_fn = ChamferDistanceLoss().to(device)
        loss = loss_fn(predicted, target)
        assert loss.device.type == "cuda"


def test_gradient_flow(sample_sets: tuple[torch.Tensor, torch.Tensor]) -> None:
    predicted, target = sample_sets
    predicted.requires_grad = True

    # Test ChamferDistance only for now
    loss_fn = ChamferDistanceLoss()
    loss = loss_fn(predicted, target)
    loss.backward()

    assert predicted.grad is not None
    assert not torch.isnan(predicted.grad).any()


def test_invalid_reduction() -> None:
    with pytest.raises(ValueError):
        ChamferDistanceLoss(reduction="invalid")

    with pytest.raises(ValueError):
        SinkhornLoss(reduction="invalid")

    with pytest.raises(ValueError):
        HausdorffLoss(reduction="invalid")


def test_sample_loss_without_loss_function() -> None:
    # Test that SampleLoss raises error when loss_function is not set
    loss_fn = SampleLoss()
    predicted = torch.randn(8, 100, 3)
    target = torch.randn(8, 100, 3)

    with pytest.raises(ValueError, match="Loss function not set"):
        loss_fn(predicted, target)
