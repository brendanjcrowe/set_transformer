"""The 2026-09-05 opt-in modes of WeightedCGFFeaturesExtractor.

Legacy behaviour (t_param="clamp", feature_mode="K", feature_norm="none") is
pinned numerically by test_ant_tag_shared_pieces_regression.py. These tests
cover what that gate cannot see: the tanh bound, K'(t), the feature
standardisers, and that the wide-t path is exact where the old exp-sum-log
path was not.
"""

import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from set_transformer.rl.feature_extractors.cgf import (
    RunningFeatureNorm,
    WeightedCGFFeaturesExtractor,
)

N_STATES = 50
CENTRE, SCALE = 25.5, 24.5


def _space(particle_dim=1):
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (N_STATES, particle_dim), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (N_STATES,), np.float32),
    })


def _odd_even_batch():
    """Three exact-support beliefs in the RL arm's centred coordinates."""
    states = torch.arange(1, N_STATES + 1, dtype=torch.float32)
    particles = (states - CENTRE).reshape(1, N_STATES, 1).repeat(3, 1, 1)
    w = torch.zeros(3, N_STATES)
    w[0, 24], w[0, 22] = 0.85, 0.15          # mode 25, some mass on 23
    w[1, 22], w[1, 24] = 0.85, 0.15          # mode 23, some mass on 25
    w[2] = 1.0 / N_STATES                    # uniform
    return {"obs": torch.zeros(3, 1), "particles": particles, "weights": w}


def _reference_k_and_kgrad(batch, t):
    """float64 K and K' straight from the definition."""
    x = (batch["particles"][:, :, 0].double() / SCALE)          # [B, N]
    w = batch["weights"].double()
    a = x[:, :, None] * t.double()[None, None, :] + torch.log(w)[:, :, None]
    k = torch.logsumexp(a, dim=1)
    tilt = torch.softmax(a, dim=1)
    kgrad = (tilt * x[:, :, None]).sum(dim=1)
    return k, kgrad


# --------------------------------------------------------------- t_param=tanh

def test_tanh_mode_reproduces_the_requested_init_and_stays_inside_the_bound():
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0)
    t = ext.effective_t().detach()
    assert isinstance(ext.raw_t, torch.nn.Parameter)
    assert not hasattr(ext, "t_values")
    expected = torch.tensor(np.geomspace(0.25, 40.0, 32), dtype=torch.float32)
    torch.testing.assert_close(t[:32, 0], expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(t[32:, 0], -expected, rtol=1e-4, atol=1e-4)
    assert float(t.abs().max()) < 50.0
    # Push the pre-activation to absurd values: t stays strictly inside.
    with torch.no_grad():
        ext.raw_t.fill_(1e6)
        assert float(ext.effective_t().abs().max()) <= 50.0


def test_tanh_mode_has_gradient_where_clamp_mode_has_none():
    batch = _odd_even_batch()
    clamp = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=8, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="clamp", t_clamp=2.0)
    tanh = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=8, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=2.0, t_init_max=1.9)
    with torch.no_grad():
        clamp.t_values.fill_(3.0)          # beyond the clamp: frozen forever
        tanh.raw_t.fill_(1.0)               # tanh(1) = 0.76 of the bound
    clamp(batch)[:, 1:].sum().backward()
    tanh(batch)[:, 1:].sum().backward()
    assert float(clamp.t_values.grad.abs().max()) == 0.0
    assert float(tanh.raw_t.grad.abs().max()) > 0.0


def test_tanh_mode_rejects_an_init_at_or_beyond_the_bound():
    with pytest.raises(ValueError, match="flat region"):
        WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=8, arena_scale=SCALE, t_init_mode="spread_1d",
            t_param="tanh", t_bound=10.0, t_init_max=10.0)
    with pytest.raises(ValueError, match="positive t_bound"):
        WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=8, arena_scale=SCALE, t_param="tanh")


def test_tanh_mode_frozen_registers_a_buffer():
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=8, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0, t_frozen=True)
    assert "raw_t" in dict(ext.named_buffers())
    assert len(list(ext.parameters())) == 0


def test_legacy_defaults_are_unchanged():
    ext = WeightedCGFFeaturesExtractor(_space(), num_cgf_features=8, arena_scale=SCALE,
                                       t_init_mode="spread_1d")
    assert ext.t_param == "clamp" and ext.feature_mode == "K"
    assert isinstance(ext.feature_norm, torch.nn.Identity)
    assert isinstance(ext.t_values, torch.nn.Parameter)
    torch.testing.assert_close(ext.effective_t(),
                               torch.clamp(ext.t_values, -2.0, 2.0))
    assert ext.features_dim == 1 + 8


# --------------------------------------------------------------- feature_mode

@pytest.mark.parametrize("mode,width", [("K", 64), ("K_grad", 64), ("both", 128)])
def test_feature_mode_widths(mode, width):
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_mode=mode)
    out = ext(_odd_even_batch())
    assert out.shape == (3, 1 + width)
    assert ext.features_dim == 1 + width


def test_feature_mode_width_scales_with_particle_dim():
    ext = WeightedCGFFeaturesExtractor(
        _space(particle_dim=2), num_cgf_features=16, arena_scale=7.0,
        t_init_mode="spread", feature_mode="both")
    assert ext.features_dim == 1 + 16 * 3


def test_k_and_kgrad_match_a_float64_reference_at_wide_t():
    """The regime the old exp-sum-log path got wrong by ~27 at |t| = 50."""
    batch = _odd_even_batch()
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="clamp", t_clamp=50.0, feature_mode="both")
    grid = torch.linspace(-50.0, 50.0, 64)
    with torch.no_grad():
        ext.t_values.copy_(grid.reshape(-1, 1))
        out = ext(batch)[:, 1:]
    k_ref, kgrad_ref = _reference_k_and_kgrad(batch, grid)
    torch.testing.assert_close(out[:, :64].double(), k_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out[:, 64:].double(), kgrad_ref, rtol=1e-5, atol=1e-5)
    # The two neighbour-mode beliefs are now told apart at t = +50 ...
    assert abs(float(out[0, 63] - out[1, 63])) > 1.0
    # ... and K'(+50) sits at the top of the support, K'(-50) near the
    # bottom. Not exactly AT it: at t = -50 the 0.85 on state 25 is tilted
    # down by exp(-50 * 0.0816) = 0.017, which still leaves 8.8% of the
    # tilted mass on 25 and pulls K' 0.007 above the atom at 23.
    x25, x23 = (25 - CENTRE) / SCALE, (23 - CENTRE) / SCALE
    assert abs(float(out[0, 64 + 63]) - x25) < 1e-3
    assert 0.0 < float(out[0, 64]) - x23 < 1e-2


def test_kgrad_at_t_zero_is_the_posterior_mean():
    batch = _odd_even_batch()
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=4, arena_scale=SCALE, t_init_mode="random",
        t_init_scale=0.0, feature_mode="K_grad")
    with torch.no_grad():
        out = ext(batch)[:, 1:]
    x = batch["particles"][:, :, 0] / SCALE
    mean = (batch["weights"] * x).sum(dim=1)
    for j in range(4):
        torch.testing.assert_close(out[:, j], mean, rtol=1e-5, atol=1e-6)


def test_dead_rows_stay_finite_in_every_mode_with_finite_gradients():
    batch = _odd_even_batch()
    batch["weights"][2] = 0.0
    for mode in ("K", "K_grad", "both"):
        ext = WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=8, arena_scale=SCALE, t_init_mode="spread_1d",
            t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_mode=mode)
        out = ext(batch)
        assert torch.isfinite(out).all()
        if mode == "K":
            assert torch.allclose(out[2, 1:], torch.full((8,), math.log(1e-8)))
        out[:, 1:].sum().backward()
        assert torch.isfinite(ext.raw_t.grad).all()


# --------------------------------------------------------------- feature_norm

def test_running_norm_standardises_per_feature_and_is_stable_in_eval():
    torch.manual_seed(0)
    norm = RunningFeatureNorm(4, momentum=0.1)
    x = torch.randn(256, 4) * torch.tensor([1.0, 10.0, 0.1, 30.0]) + torch.tensor([0.0, 5.0, -2.0, 40.0])
    norm.train()
    y = norm(x)                                   # first call copies batch stats
    assert torch.allclose(y.mean(dim=0), torch.zeros(4), atol=1e-4)
    assert torch.allclose(y.std(dim=0, unbiased=False), torch.ones(4), atol=1e-3)
    norm.eval()
    y1, y2 = norm(x * 3), norm(x * 3)             # no update, deterministic
    torch.testing.assert_close(y1, y2)
    assert int(norm.num_updates) == 1


def test_running_norm_keeps_the_posterior_mean_where_layernorm_loses_it():
    """On near-rank-1 features LayerNorm divides the mean out; running
    per-feature z-scoring keeps it. This is why 'running' is the default."""
    t = torch.linspace(-2.0, 2.0, 16)
    mean = torch.tensor([-0.8, -0.2, 0.3, 0.9])
    feats = mean[:, None] * t[None, :]                            # exactly rank 1
    ln = torch.nn.LayerNorm(16, elementwise_affine=False)(feats)
    # Same normalised vector for every sample up to sign: |mean| is gone.
    assert torch.allclose(ln[0].abs(), ln[3].abs(), atol=1e-4)
    assert torch.allclose(ln[1].abs(), ln[2].abs(), atol=1e-4)
    rn = RunningFeatureNorm(16)
    rn.train()
    out = rn(feats + 0.01 * t[None, :] ** 2)                      # plus a small variance term
    # The standardised features are still an affine image of the mean.
    corr = torch.corrcoef(torch.stack([out[:, 3], mean]))[0, 1]
    assert float(corr.abs()) > 0.999


def test_feature_norm_is_applied_to_the_cgf_block_only():
    batch = _odd_even_batch()
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=8, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_norm="running")
    ext.train()
    batch["obs"] = torch.tensor([[0.1], [0.5], [0.9]])
    out = ext(batch)
    torch.testing.assert_close(out[:, :1], batch["obs"])      # passthrough untouched
    assert abs(float(out[:, 1:].mean())) < 1e-4                # block standardised


def test_kwargs_round_trip_the_way_sb3_reloads_them():
    kwargs = dict(num_cgf_features=16, arena_scale=SCALE, t_init_mode="spread_1d",
                  t_param="tanh", t_bound=50.0, t_init_max=40.0,
                  feature_mode="both", feature_norm="running", exp_arg_clamp=20.0)
    a = WeightedCGFFeaturesExtractor(_space(), **kwargs)
    b = WeightedCGFFeaturesExtractor(_space(), **kwargs)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(a(_odd_even_batch()), b(_odd_even_batch()))
