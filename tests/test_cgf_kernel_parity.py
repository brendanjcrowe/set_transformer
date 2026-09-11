"""One CGF kernel, two adapters: the arm extractor and the benchmark encoder must agree.

`rl/feature_extractors/cgf.py::WeightedCGFFeaturesExtractor` (the experiments/{ant_tag,
odd_even} arms) and `models/cgf_encoder.py::CGFEncoder` (the benchmark, via
`statistical.py::CGFExtractor`) compute K(t) = log sum_i w_i exp(t . x_i) through the
same module-level functions (`init_t_values`, `cgf_log_mgf`) since 2026-09-11; before
that the benchmark carried a July 2026 copy. They still differ in contract on purpose --
bottleneck, head, weight handling, pretraining objective, checkpoint layout -- and
PITFALLS.md sec. 11 has the table.

These tests are the guard-rail for that table: the parity tests fail if either adapter
stops routing through the shared kernel or init; the contract tests fail if someone
makes the two look interchangeable without updating the record.
"""

import contextlib
import io

import gymnasium as gym
import numpy as np
import pytest
import torch

from set_transformer.models.cgf_encoder import CGFEncoder
from set_transformer.rl.feature_extractors.cgf import WeightedCGFFeaturesExtractor
from set_transformer.rl.feature_extractors.gaussian import WeightedGaussianFeaturesExtractor
from set_transformer.rl.feature_extractors.statistical import (
    CGFExtractor,
    GaussianExtractor,
)

B, N, D, NUM_T = 4, 100, 2, 64
SCALE = 4.5          # Ant-Tag arena half-width; both sides divide particles by it
T_CLAMP = 2.0        # the legacy bound both forwards apply


def _space(obs_dim=3, with_weights=True):
    d = {
        "obs": gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
        "particles": gym.spaces.Box(-2 * SCALE, 2 * SCALE, (N, D), np.float32),
    }
    if with_weights:
        d["weights"] = gym.spaces.Box(0.0, 1.0, (N,), np.float32)
    return gym.spaces.Dict(d)


def _batch(seed=0):
    """Particles at arena scale, weights skewed so half the set is near-dead (ESS ~ 50)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, N, D, generator=g) * 3.0
    w = torch.rand(B, N, generator=g)
    w[:, N // 2:] *= 0.01
    w = w / w.sum(dim=1, keepdim=True)
    return {"obs": torch.zeros(B, 3), "particles": x, "weights": w}


def _benchmark_kernel(**kw):
    """CGFEncoder with the 64 -> 16 projection disabled so its output IS the raw K."""
    enc = CGFEncoder(dim_input=D, num_outputs=8, dim_output=8, num_t=NUM_T,
                     t_clamp=T_CLAMP, particle_scale=SCALE, t_init_mode="spread", **kw)
    assert isinstance(enc.cgf_proj, torch.nn.Identity)
    return enc


def _arm_kernel(**kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=NUM_T, arena_scale=SCALE, t_param="clamp",
            t_clamp=T_CLAMP, t_init_mode="spread", feature_mode="K",
            feature_norm="none", **kw)


def _particle_features_with(self, batch):
    self._particle_weights = batch.get("weights")
    return self._particle_features(batch["particles"])


GaussianExtractor._particle_features_with = _particle_features_with


# --- parity: the shared kernel ------------------------------------------------------

def test_spread_init_is_the_same_array_on_both_sides():
    """8 directions x geomspace(0.25, 2.8, 8), same row order. Not a coincidence to keep:
    PITFALLS sec. 9 item 2 documents what clamp 2.0 does to the norm-2.8 ring, and that
    analysis holds for both sides only while the arrays match."""
    torch.manual_seed(0)
    his, mine = _benchmark_kernel(), _arm_kernel()
    assert torch.equal(his.t_values.detach(), mine.t_values.detach())


def test_weighted_k_agrees_to_float32_precision():
    torch.manual_seed(0)
    his, mine = _benchmark_kernel(), _arm_kernel()
    with torch.no_grad():
        mine.t_values.copy_(his.t_values)
    batch = _batch()
    k_his = his(batch["particles"], batch["weights"]).reshape(B, -1)
    k_mine = mine._raw_cgf(batch)
    assert k_his.shape == k_mine.shape == (B, NUM_T)
    assert torch.allclose(k_his, k_mine, atol=1e-5, rtol=0.0), \
        (k_his - k_mine).abs().max().item()


def test_benchmark_unweighted_path_equals_arm_uniform_weights():
    """His `weights=None` branch is logsumexp - log n; ours has no such branch and must be
    handed uniform weights. The two must be the same number, or 'unweighted' on his side
    and 'uniform' on ours stop meaning the same thing."""
    torch.manual_seed(0)
    his, mine = _benchmark_kernel(), _arm_kernel()
    with torch.no_grad():
        mine.t_values.copy_(his.t_values)
    batch = _batch()
    uniform = torch.full((B, N), 1.0 / N)
    k_none = his(batch["particles"], None).reshape(B, -1)
    k_mine = mine._raw_cgf({**batch, "weights": uniform})
    assert torch.allclose(k_none, k_mine, atol=1e-5, rtol=0.0)


def test_both_forwards_clamp_t_identically():
    """Four of the 64 spread probes exceed the clamp on one axis (norm-2.8 ring, axis
    aligned). Both sides must flatten them the same way, or K differs on exactly those
    columns while the stored t_values still look identical."""
    torch.manual_seed(0)
    his, mine = _benchmark_kernel(), _arm_kernel()
    his_eff = torch.clamp(his.t_values.detach(), -T_CLAMP, T_CLAMP)
    assert torch.equal(his_eff, mine.effective_t().detach())
    assert int((his_eff != his.t_values.detach()).any(dim=1).sum()) == 4


# --- contract: where they differ, on purpose ----------------------------------------

def test_the_contracts_differ_where_pitfalls_says_they_do():
    """Pins the divergences so nobody assumes drop-in. If this fails because the two were
    deliberately unified, update PITFALLS sec. 11 and delete the assertion, not the doc."""
    torch.manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        his = CGFExtractor(_space(with_weights=False), particle_scale=SCALE)
    mine = _arm_kernel()
    # bottleneck: 64 -> 16 learned projection vs 64 raw values
    assert his._particle_stat_dim() == 16 and set(his.encoder.state_dict()) == {
        "t_values", "cgf_proj.weight", "cgf_proj.bias"}
    assert mine._raw_cgf(_batch()).shape[1] == NUM_T and set(mine.state_dict()) == {"t_values"}
    # head: his goes through obs-MLP + concat + Linear; ours concatenates raw to obs
    assert his.features_dim == 128
    assert mine.features_dim == 3 + NUM_T
    # weights: optional on his side, required on ours
    batch = _batch()
    his({"obs": batch["obs"], "particles": batch["particles"]})
    with pytest.raises(KeyError):
        mine({"obs": batch["obs"], "particles": batch["particles"]})


def test_both_gaussians_read_weights_and_agree():
    """Since 2026-09-11 the benchmark GaussianExtractor computes the statistic through the
    arm's `weighted_mean_cov`, so on a weighted env both summarise the same measure. The
    feature layouts differ ([mean, tril(cov)] vs [mean, var, offdiag]) and the arm divides
    by arena_scale, so compare the moments, not the vectors."""
    torch.manual_seed(0)
    batch = _batch()
    uniform = {**batch, "weights": torch.full((B, N), 1.0 / N)}
    with contextlib.redirect_stdout(io.StringIO()):
        his = GaussianExtractor(_space())
        mine = WeightedGaussianFeaturesExtractor(_space(), arena_scale=1.0)
    # both react to the weights ...
    assert not torch.allclose(his._particle_features_with(batch), his._particle_features_with(uniform), atol=1e-4)
    assert not torch.allclose(mine(batch), mine(uniform), atol=1e-4)
    # ... and to the same moments: mean and the diagonal of the covariance
    d = D
    his_stat = his._particle_features_with(batch)
    mine_stat = mine(batch)[:, 3:]                     # strip the 3 obs dims
    assert torch.allclose(his_stat[:, :d], mine_stat[:, :d], atol=1e-5)          # mean
    tril = torch.tril_indices(d, d)
    diag_cols = [i for i, (r, c) in enumerate(zip(tril[0].tolist(), tril[1].tolist())) if r == c]
    assert torch.allclose(his_stat[:, d:][:, diag_cols], mine_stat[:, d:2 * d], atol=1e-5)  # var
