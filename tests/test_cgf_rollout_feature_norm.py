"""RunningFeatureNorm's two update modes and the pre-norm feature hook.

PITFALLS.md section 8 item 5: under PPO the 'minibatch' mode re-estimates
the statistics 1280 times inside one update, so the stored and recomputed
log-probs are standardised differently. The 'rollout' mode never updates
itself; RolloutFeatureNormCallback (tested end to end in
test_odd_even_pipeline.py) sets the statistics once per cycle.
"""

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


def _space():
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (N_STATES, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (N_STATES,), np.float32),
    })


def _batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = torch.arange(1, N_STATES + 1, dtype=torch.float32)
    particles = (states - CENTRE).reshape(1, N_STATES, 1).repeat(n, 1, 1)
    w = torch.softmax(torch.randn(n, N_STATES, generator=g) * 3, dim=-1)
    return {"obs": torch.rand(n, 1, generator=g), "particles": particles, "weights": w}


def test_rollout_mode_never_updates_itself_in_training_mode():
    norm = RunningFeatureNorm(4, update="rollout").train()
    x = torch.randn(64, 4) * 5 + 3
    before = (norm.running_mean.clone(), norm.running_var.clone())
    y = norm(x)
    torch.testing.assert_close(norm.running_mean, before[0])
    torch.testing.assert_close(norm.running_var, before[1])
    assert int(norm.num_updates) == 0
    # Output uses the (untouched) running statistics, here (0, 1).
    torch.testing.assert_close(y, x / torch.sqrt(torch.tensor(1.0 + norm.eps)))


def test_minibatch_mode_is_the_legacy_lerp():
    torch.manual_seed(0)
    norm = RunningFeatureNorm(3, momentum=0.1).train()
    x1, x2 = torch.randn(32, 3) + 2, torch.randn(32, 3) * 4
    norm(x1)                                           # first call copies
    torch.testing.assert_close(norm.running_mean, x1.mean(0))
    torch.testing.assert_close(norm.running_var, x1.var(0, unbiased=False))
    y = norm(x2)                                       # then lerps ...
    expect_mean = x1.mean(0).lerp(x2.mean(0), 0.1)
    expect_var = x1.var(0, unbiased=False).lerp(x2.var(0, unbiased=False), 0.1)
    torch.testing.assert_close(norm.running_mean, expect_mean)
    torch.testing.assert_close(norm.running_var, expect_var)
    # ... and normalises with the UPDATED statistics (the item-5 behaviour).
    torch.testing.assert_close(y, (x2 - expect_mean) / torch.sqrt(expect_var + norm.eps))
    assert int(norm.num_updates) == 2


def test_set_statistics_copies_first_then_honours_momentum():
    norm = RunningFeatureNorm(2, update="rollout")
    norm.set_statistics(torch.tensor([1.0, 2.0]), torch.tensor([4.0, 9.0]), momentum=0.5)
    torch.testing.assert_close(norm.running_mean, torch.tensor([1.0, 2.0]))   # first: copy
    norm.set_statistics(torch.tensor([3.0, 2.0]), torch.tensor([4.0, 1.0]), momentum=0.5)
    torch.testing.assert_close(norm.running_mean, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(norm.running_var, torch.tensor([4.0, 5.0]))
    norm.set_statistics(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))   # momentum 1: adopt
    torch.testing.assert_close(norm.running_mean, torch.zeros(2))
    assert int(norm.num_updates) == 3


def test_eval_mode_never_updates_in_either_mode():
    for update in RunningFeatureNorm.UPDATE_MODES:
        norm = RunningFeatureNorm(3, update=update).eval()
        norm(torch.randn(16, 3) + 10)
        assert int(norm.num_updates) == 0, update


def test_unknown_update_mode_is_refused():
    with pytest.raises(ValueError, match="update must be"):
        RunningFeatureNorm(3, update="epoch")


def test_raw_cgf_features_is_what_the_norm_standardises():
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=8, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_norm="running",
        readout_hidden=16, readout_depth=1).eval()
    batch = _batch()
    with torch.no_grad():
        raw = ext.raw_cgf_features(batch)
        ext.feature_norm.set_statistics(raw.mean(0), raw.var(0, unbiased=False))
        out = ext(batch)
        expect = ext.readout(ext.feature_norm(raw))
    assert raw.shape == (6, 8)
    torch.testing.assert_close(out[:, :1], batch["obs"])
    torch.testing.assert_close(out[:, 1:], expect)
    # Standardised block is on scale once the statistics come from these rows.
    z = ext.feature_norm(raw)
    assert abs(float(z.mean())) < 1e-4 and abs(float(z.std(0, unbiased=False).mean()) - 1) < 1e-2


def test_raw_and_full_forward_agree_on_the_legacy_path():
    """Splitting _forward must not change the no-norm, no-readout numbers."""
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=8, arena_scale=SCALE, t_init_mode="spread_1d").eval()
    batch = _batch()
    with torch.no_grad():
        torch.testing.assert_close(ext(batch)[:, 1:], ext.raw_cgf_features(batch))
