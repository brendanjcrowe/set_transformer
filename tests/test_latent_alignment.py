"""Tests for the Pearson latent metric-alignment loss.

Covers the pieces the training loop depends on being exactly right: the pair ordering
shared by the latent and target vectors, the correlation itself, the degenerate-batch
guard, gradient flow, and the lambda ramp.
"""

import numpy as np
import pytest
import torch

from set_transformer.latent_alignment import (
    LambdaRamp,
    PearsonAlignmentLoss,
    flatten_upper_triangle,
    latent_pairwise_distances,
    pearson_r,
)


def _symmetric(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(n, n, generator=g)
    m = m + m.t()
    m.fill_diagonal_(0.0)
    return m


def test_flatten_upper_triangle_order_and_length():
    m = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    flat = flatten_upper_triangle(m)
    assert flat.tolist() == [1.0, 2.0, 3.0]  # (0,1), (0,2), (1,2)
    n = 7
    assert len(flatten_upper_triangle(_symmetric(n))) == n * (n - 1) // 2


def test_flatten_upper_triangle_rejects_non_square():
    with pytest.raises(ValueError):
        flatten_upper_triangle(torch.zeros(3, 4))


def test_latent_and_target_share_pair_ordering():
    # The loss is only meaningful if entry k of d_latent and entry k of d_target refer to
    # the same pair. Both go through flatten_upper_triangle, so a target built from a
    # matrix of known pair ids must come back in the same order as the latent distances.
    z = torch.randn(5, 3)
    d_latent = latent_pairwise_distances(z, "euclidean")
    full = torch.cdist(z, z)
    assert torch.allclose(d_latent, flatten_upper_triangle(full), atol=1e-6)


def test_pearson_r_matches_numpy():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(200, generator=g)
    b = 0.7 * a + 0.3 * torch.randn(200, generator=g)
    expected = np.corrcoef(a.numpy(), b.numpy())[0, 1]
    assert pearson_r(a, b).item() == pytest.approx(expected, abs=1e-5)


def test_pearson_r_none_on_constant_vector():
    a = torch.ones(50)
    b = torch.randn(50)
    assert pearson_r(a, b) is None
    assert pearson_r(b, a) is None


def test_pearson_r_is_affine_invariant():
    # Pearson's scale invariance is what decouples the alignment gradient scale from the
    # reconstruction loss's units, so lambda stays tunable. Guard it.
    g = torch.Generator().manual_seed(2)
    a = torch.randn(100, generator=g)
    b = torch.randn(100, generator=g)
    base = pearson_r(a, b)
    assert pearson_r(a, 3.5 * b + 12.0).item() == pytest.approx(base.item(), abs=1e-5)


def test_loss_is_zero_on_perfectly_aligned_latents():
    # Euclidean distances between points on a line are an exact multiple of the 1-D
    # target distances, so r == 1 and the loss vanishes.
    t = torch.tensor([0.0, 1.0, 2.5, 4.0, 7.0, 9.0]).unsqueeze(1)
    target = flatten_upper_triangle(torch.cdist(t, t))
    loss, r = PearsonAlignmentLoss("euclidean")(2.0 * t, target)
    assert r.item() == pytest.approx(1.0, abs=1e-5)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_loss_is_two_on_anti_aligned_latents():
    t = torch.tensor([0.0, 1.0, 2.5, 4.0, 7.0, 9.0]).unsqueeze(1)
    d = flatten_upper_triangle(torch.cdist(t, t))
    loss, r = PearsonAlignmentLoss("euclidean")(t, -d)
    assert r.item() == pytest.approx(-1.0, abs=1e-5)
    assert loss.item() == pytest.approx(2.0, abs=1e-5)


def test_matrix_and_flat_targets_agree():
    z = torch.randn(9, 4)
    m = _symmetric(9, seed=3)
    loss_m, r_m = PearsonAlignmentLoss()(z, m)
    loss_f, r_f = PearsonAlignmentLoss()(z, flatten_upper_triangle(m))
    assert loss_m.item() == pytest.approx(loss_f.item(), abs=1e-6)
    assert r_m.item() == pytest.approx(r_f.item(), abs=1e-6)


def test_multi_dim_latent_is_flattened():
    # The repo's bottleneck is (B, num_encodings, dim_encoder); it must be treated as one
    # vector per sample, not per encoding.
    z = torch.randn(8, 4, 5)
    d = latent_pairwise_distances(z, "cosine")
    d_flat = latent_pairwise_distances(z.reshape(8, -1), "cosine")
    assert torch.allclose(d, d_flat, atol=1e-6)
    assert len(d) == 8 * 7 // 2


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_gradient_flows_to_latents(metric):
    z = torch.randn(12, 6, requires_grad=True)
    loss, _ = PearsonAlignmentLoss(metric)(z, _symmetric(12, seed=4))
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert z.grad.abs().sum() > 0


def test_degenerate_batch_is_skipped_but_stays_differentiable():
    # Identical latents => zero-variance distances. The term must contribute nothing
    # rather than a nan, and must still be safe to call .backward() on.
    z = torch.ones(10, 4, requires_grad=True)
    loss, r = PearsonAlignmentLoss("euclidean")(z, _symmetric(10, seed=5))
    assert r is None
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(z.grad).all()


def test_permuting_the_batch_leaves_the_correlation_unchanged():
    z = torch.randn(10, 5)
    m = _symmetric(10, seed=6)
    _, r = PearsonAlignmentLoss()(z, m)
    perm = torch.randperm(10, generator=torch.Generator().manual_seed(7))
    _, r_perm = PearsonAlignmentLoss()(z[perm], m[perm][:, perm])
    assert r_perm.item() == pytest.approx(r.item(), abs=1e-5)


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        PearsonAlignmentLoss("manhattan")
    with pytest.raises(ValueError):
        latent_pairwise_distances(torch.randn(4, 2), "manhattan")


def test_lambda_ramp_holds_then_ramps_then_saturates():
    ramp = LambdaRamp(target=0.5, warmup_epochs=3, ramp_epochs=4)
    assert [ramp(e) for e in range(3)] == [0.0, 0.0, 0.0]
    assert ramp(3) == pytest.approx(0.125)
    assert ramp(6) == pytest.approx(0.5)
    assert ramp(50) == pytest.approx(0.5)


def test_lambda_ramp_without_ramp_is_a_step():
    ramp = LambdaRamp(target=1.0, warmup_epochs=2, ramp_epochs=0)
    assert [ramp(e) for e in range(4)] == [0.0, 0.0, 1.0, 1.0]


def test_lambda_ramp_rejects_negative_epochs():
    with pytest.raises(ValueError):
        LambdaRamp(target=1.0, warmup_epochs=-1)
