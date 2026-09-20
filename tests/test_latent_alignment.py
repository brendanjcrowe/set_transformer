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


# ---------------------------------------------------------------------------------------
# Online targets (2026-09-19): the pair sampler, the pair-subset distances, the per-batch
# Sinkhorn targets against the precomputed matrix, and the loss on a pair subset.
# ---------------------------------------------------------------------------------------

from set_transformer.latent_alignment import (  # noqa: E402
    OnlineAlignment,
    all_pairs,
    latent_pair_distances,
    parse_pairs,
    sample_pairs,
    sinkhorn_pair_targets,
)


def test_parse_pairs_accepts_all_or_a_positive_budget():
    assert parse_pairs("all") == "all" and parse_pairs(" ALL ") == "all"
    assert parse_pairs("2016") == 2016 and parse_pairs(64) == 64
    for bad in ("0", "-3", "some", None):
        with pytest.raises(ValueError, match="align_pairs"):
            parse_pairs(bad)


def test_sample_pairs_all_is_the_upper_triangle_in_order():
    """'all' (or a budget at or above B(B-1)/2) gives the matrix path's pairs in its order, so a
    target read with these indices equals flatten_upper_triangle of the matrix."""
    a, b = sample_pairs(5, "all")
    ia, ib = all_pairs(5)
    assert torch.equal(a, ia) and torch.equal(b, ib)
    a2, b2 = sample_pairs(5, 10)
    assert torch.equal(a2, ia) and torch.equal(b2, ib)
    m = _symmetric(5)
    assert torch.equal(m[a, b], flatten_upper_triangle(m))


def test_sample_pairs_budget_covers_every_row_without_self_or_duplicate_pairs():
    """A budget is filled with ceil(P / B) permutations of the batch, so every row is in some
    pair; no (i, i) pairs, no duplicates, a < b, at most the budget."""
    B, budget = 64, 256                       # 4 permutations, nothing cut
    a, b = sample_pairs(B, budget, torch.Generator().manual_seed(3))
    assert 0.9 * budget <= len(a) <= budget and torch.all(a < b)
    assert len(torch.unique(a * B + b)) == len(a)
    assert set(a.tolist()) | set(b.tolist()) == set(range(B))
    a, b = sample_pairs(B, 100, torch.Generator().manual_seed(3))     # cut to the budget
    assert len(a) == 100 and torch.all(a < b) and len(torch.unique(a * B + b)) == 100


def test_sample_pairs_uses_its_own_generator_and_leaves_the_global_stream_alone():
    torch.manual_seed(0)
    before = torch.rand(3)
    torch.manual_seed(0)
    sample_pairs(32, 40, torch.Generator().manual_seed(1))
    assert torch.equal(before, torch.rand(3))
    x = sample_pairs(32, 40, torch.Generator().manual_seed(1))
    y = sample_pairs(32, 40, torch.Generator().manual_seed(1))
    assert torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_latent_pair_distances_equal_the_triangle_subset(metric):
    z = torch.randn(9, 3, 4, generator=torch.Generator().manual_seed(0))
    full = latent_pairwise_distances(z, metric)
    a, b = all_pairs(9)
    torch.testing.assert_close(latent_pair_distances(z, (a, b), metric), full, atol=1e-6, rtol=1e-5)
    sub = torch.tensor([0, 5, 17, 35])
    torch.testing.assert_close(latent_pair_distances(z, (a[sub], b[sub]), metric), full[sub],
                               atol=1e-6, rtol=1e-5)


def test_online_targets_equal_the_precomputed_matrix_entries(tmp_path):
    """The whole point: the same divergence as emd_matrix.compute_matrix (weighted, debiased),
    computed for the listed pairs only; uniform weights equal the unweighted call."""
    from geomloss import SamplesLoss
    from set_transformer.emd_matrix import compute_matrix
    g = torch.Generator().manual_seed(0)
    pts = torch.rand(16, 10, 2, generator=g) * 2 - 1
    w = torch.rand(16, 10, generator=g) ** 2
    w = w / w.sum(dim=1, keepdim=True)
    matrix = compute_matrix(pts.numpy(), tmp_path / "m.npy", blur=0.05, block=8, verify=False,
                            device="cpu", weights=w.numpy())
    reference = flatten_upper_triangle(torch.from_numpy(np.array(matrix)))
    ot = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5)
    a, b = all_pairs(16)
    # the matrix debiases by hand (self terms once per cloud), geomloss per pair: <1e-3 apart
    # by emd_matrix's own verify_against_geomloss bound (1.2e-4 measured here)
    torch.testing.assert_close(sinkhorn_pair_targets(pts, w, ot, (a, b)), reference, atol=1e-3, rtol=0)
    sub = torch.tensor([3, 40, 77, 119])
    torch.testing.assert_close(sinkhorn_pair_targets(pts, w, ot, (a[sub], b[sub]), chunk=3),
                               reference[sub], atol=1e-3, rtol=0)
    uniform = torch.full((16, 10), 0.1)
    torch.testing.assert_close(sinkhorn_pair_targets(pts, None, ot, (a, b)),
                               sinkhorn_pair_targets(pts, uniform, ot, (a, b)), atol=1e-5, rtol=0)
    # the weight rule is the loss's own: zero mass is refused, not silently made uniform
    with pytest.raises(ValueError, match="sum to zero"):
        sinkhorn_pair_targets(pts, torch.zeros(16, 10), ot, (a, b))


def test_pearson_loss_on_a_pair_subset_matches_the_flat_form():
    z = torch.randn(8, 4, generator=torch.Generator().manual_seed(1))
    m = _symmetric(8)
    loss_full, r_full = PearsonAlignmentLoss()(z, m)
    a, b = all_pairs(8)
    loss_pairs, r_pairs = PearsonAlignmentLoss()(z, m[a, b], pairs=(a, b))
    assert loss_pairs.item() == pytest.approx(loss_full.item(), abs=1e-6)
    assert r_pairs.item() == pytest.approx(r_full.item(), abs=1e-6)
    with pytest.raises(ValueError, match="flat"):
        PearsonAlignmentLoss()(z, m, pairs=(a, b))


def test_online_alignment_term_is_differentiable_and_records_its_settings():
    oa = OnlineAlignment(blur=0.05, pairs=20, seed=0)
    g = torch.Generator().manual_seed(2)
    pts = torch.rand(8, 10, 2, generator=g)
    w = torch.rand(8, 10, generator=g)
    z = torch.randn(8, 4, generator=g).requires_grad_(True)
    loss, r, n = oa.term(z, pts, w)
    assert 0 < n <= 20 and r is not None
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert oa.record() == {"target": "online", "metric": "cosine", "pairs": 20, "blur": 0.05,
                           "scaling": 0.5, "p": 2, "pair_seed": 0}
    p1, p2 = oa.fixed_pairs(30, 50), oa.fixed_pairs(30, 50)      # drawn from seed + 1, every time
    assert torch.equal(p1[0], p2[0]) and torch.equal(p1[1], p2[1]) and len(p1[0]) == 50
    # Odd-Even keeps its posteriors in float64: accepted on float32 particles
    loss64, _, _ = oa.term(z.detach(), pts, w.double())
    assert torch.isfinite(loss64)
    assert np.isfinite(oa.correlation(z.detach(), p1 if False else oa.fixed_pairs(8, 10),
                                      oa.targets(pts, w, oa.fixed_pairs(8, 10))))
    with pytest.raises(ValueError, match="positive"):
        OnlineAlignment(blur=0.0)


def test_resolve_sinkhorn_blur_fills_the_domain_default_only_when_omitted():
    """2026-09-19: a command that omits --sinkhorn_blur gets the DOMAIN's default (Domain.default_sinkhorn_blur),
    the package's 0.05 for a domain without one; a given value is kept; the source is recorded; idempotent."""
    from types import SimpleNamespace
    from set_transformer.latent_alignment import PACKAGE_SINKHORN_BLUR, resolve_sinkhorn_blur
    dom = SimpleNamespace(default_sinkhorn_blur=0.02)
    a = SimpleNamespace(sinkhorn_blur=None)
    assert resolve_sinkhorn_blur(a, dom) == "domain default" and a.sinkhorn_blur == 0.02
    assert resolve_sinkhorn_blur(a, dom) == "domain default" and a.sinkhorn_blur == 0.02      # idempotent
    b = SimpleNamespace(sinkhorn_blur=0.01)
    assert resolve_sinkhorn_blur(b, dom) == "given" and b.sinkhorn_blur == 0.01 and b.sinkhorn_blur_source == "given"
    c = SimpleNamespace(sinkhorn_blur=None)
    assert resolve_sinkhorn_blur(c, None) == "package default" and c.sinkhorn_blur == PACKAGE_SINKHORN_BLUR == 0.05
    d = SimpleNamespace(sinkhorn_blur=None)
    assert resolve_sinkhorn_blur(d, SimpleNamespace()) == "package default"                  # a domain without the field
    # the four pretraining domains declare the recipes' recorded values; Car-Flag keeps the fallback
    from set_transformer.rl import domains
    assert {n: domains.get(n).default_sinkhorn_blur for n in ("hunt", "odd_even", "msearch", "ant_tag", "car_flag")} == \
        {"hunt": 0.02, "odd_even": 0.02, "msearch": 0.02, "ant_tag": 0.01, "car_flag": 0.05}
