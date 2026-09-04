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


def test_chamfer_distance_is_zero_for_identical_sets(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """A set against itself must sit at the loss minimum, zero.

    Regression test: the reductions used to be taken over dim=1 and dim=0, so
    on a batched input the second one min'd across BATCH ELEMENTS. Identical
    sets then scored ~0.85 instead of 0. The tolerance absorbs cdist's
    matmul-based float32 error on the self-distance diagonal (~1e-4), not any
    real mismatch.
    """
    predicted, _ = sample_sets
    loss_fn = ChamferDistanceLoss()

    assert loss_fn(predicted, predicted).item() == pytest.approx(0.0, abs=1e-3)
    # Unbatched [num_points, dim] input must behave the same way.
    assert loss_fn(predicted[0], predicted[0]).item() == pytest.approx(0.0, abs=1e-3)


def test_chamfer_distance_does_not_mix_batch_elements(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Each set is compared only with its own partner.

    With reduction="mean" the loss over a batch equals the mean of the
    per-element losses. Any leakage across the batch axis breaks this.
    """
    predicted, target = sample_sets
    loss_fn = ChamferDistanceLoss()

    batched = loss_fn(predicted, target)
    per_element = torch.stack([
        loss_fn(predicted[i : i + 1], target[i : i + 1])
        for i in range(predicted.shape[0])
    ]).mean()

    assert batched.item() == pytest.approx(per_element.item(), rel=1e-5)


def test_chamfer_distance_known_value() -> None:
    """Two single-point sets a distance d apart score 2*d: d each direction."""
    loss_fn = ChamferDistanceLoss()
    a = torch.zeros(1, 1, 2)
    b = torch.tensor([[[2.5, 0.0]]])

    assert loss_fn(a, b).item() == pytest.approx(5.0, rel=1e-5)


def test_sinkhorn_loss(sample_sets: tuple[torch.Tensor, torch.Tensor]) -> None:
    predicted, target = sample_sets
    loss_fn = SinkhornLoss(p=2, blur=0.5)
    loss = loss_fn(predicted, target)

    assert loss is not None
    assert not torch.isnan(loss).any()


def test_sinkhorn_uniform_weights_match_unweighted_call(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Explicit uniform mass must reproduce the plain set-to-set loss.

    This pins the two geomloss call signatures together: passing weights
    switches to loss(alpha, x, beta, y), and if that path disagreed with the
    two-argument path on uniform input, every weighted number would be on a
    different scale from the unweighted baseline.
    """
    predicted, target = sample_sets
    loss_fn = SinkhornLoss(p=2, blur=0.05)
    n = predicted.shape[1]
    uniform = torch.full(predicted.shape[:2], 1.0 / n)

    unweighted = loss_fn(predicted, target)
    explicit = loss_fn(predicted, target, uniform, uniform)

    assert explicit.item() == pytest.approx(unweighted.item(), rel=1e-5)


def test_sinkhorn_weighted_self_comparison_is_zero(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """A weighted measure compared with itself is at the minimum."""
    predicted, _ = sample_sets
    weights = torch.rand(predicted.shape[:2])
    weights = weights / weights.sum(dim=1, keepdim=True)
    loss_fn = SinkhornLoss(p=2, blur=0.05)

    loss = loss_fn(predicted, predicted, weights, weights)

    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_sinkhorn_weights_change_the_objective(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Concentrating the target mass must move the loss.

    Mass lives in the measure, so re-weighting the same points is a different
    optimal-transport problem. A loss that ignored the weights would return an
    identical number here.
    """
    predicted, target = sample_sets
    loss_fn = SinkhornLoss(p=2, blur=0.05)

    concentrated = torch.zeros(target.shape[:2])
    concentrated[:, 0] = 1.0   # all mass on one target point

    assert loss_fn(predicted, target).item() != pytest.approx(
        loss_fn(predicted, target, None, concentrated).item(), rel=1e-3
    )


def test_sinkhorn_weight_normalization_is_scale_invariant(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Uniformly tiny weights must equal uniform weights, not blow up.

    Regression test for normalizing by (sum + eps): for weights around 1e-30
    the epsilon dominates the true sum, the "normalized" measure ends up with
    total mass ~1e-17, and the loss came back in the thousands.
    """
    predicted, target = sample_sets
    loss_fn = SinkhornLoss(p=2, blur=0.05)
    n = target.shape[1]

    uniform = loss_fn(predicted, target, None, torch.full(target.shape[:2], 1.0 / n))
    tiny = loss_fn(predicted, target, None, torch.full(target.shape[:2], 1e-30))

    assert tiny.item() == pytest.approx(uniform.item(), rel=1e-5)


def test_sinkhorn_rejects_massless_and_nonfinite_weights(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """No probability measure means no transport problem — raise, don't guess."""
    predicted, target = sample_sets
    loss_fn = SinkhornLoss(p=2, blur=0.05)

    with pytest.raises(ValueError):
        loss_fn(predicted, target, None, torch.zeros(target.shape[:2]))
    with pytest.raises(ValueError):
        loss_fn(predicted, target, None,
                torch.full(target.shape[:2], float("nan")))

    # One bad set inside an otherwise valid batch must still be caught.
    weights = torch.full(target.shape[:2], 1.0 / target.shape[1])
    weights[0] = 0.0
    with pytest.raises(ValueError):
        loss_fn(predicted, target, None, weights)


def test_emd_matches_sinkhorn_weight_contract(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """The eval metric normalizes weights the same way the train loss does."""
    predicted, target = sample_sets
    n = target.shape[1]
    emd = EarthMoverDistanceLoss()

    unweighted = emd(predicted, target)
    explicit = emd(predicted, target, None, torch.full(target.shape[:2], 1.0 / n))
    unnormalized = emd(predicted, target, None, torch.full(target.shape[:2], 7.0))

    assert explicit.item() == pytest.approx(unweighted.item(), rel=1e-5)
    assert unnormalized.item() == pytest.approx(unweighted.item(), rel=1e-5)


def test_chamfer_rejects_weights(
    sample_sets: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Chamfer has no weighted formulation, so it must refuse, not ignore."""
    predicted, target = sample_sets
    weights = torch.rand(target.shape[:2])
    weights = weights / weights.sum(dim=1, keepdim=True)

    with pytest.raises(NotImplementedError):
        ChamferDistanceLoss()(predicted, target, None, weights)


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
