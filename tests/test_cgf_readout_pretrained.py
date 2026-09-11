"""The CGF readout MLP, parameter matching, freezing and pretrained loading
(2026-09-05), the pieces that make a size- and supervision-matched CGF-vs-ST
comparison possible on Odd-Even. Legacy behaviour (no readout) is pinned by
test_ant_tag_shared_pieces_regression.py; test_cgf_extractor_modes.py covers
the tanh / K_grad / norm modes.
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from set_transformer.rl.feature_extractors.cgf import (
    WeightedCGFFeaturesExtractor,
    cgf_raw_dim,
    matched_readout_hidden,
    non_readout_param_count,
    readout_param_count,
    x_embed_param_count,
)

N_STATES, SCALE = 50, 24.5
SMALL_ST_PARAMS = 109_448          # the 2-SAB, dim_hidden 64, 16-inducing-point encoder


def _space():
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (N_STATES, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (N_STATES,), np.float32),
    })


def _batch(b=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    states = torch.arange(1, N_STATES + 1, dtype=torch.float32) - 25.5
    return {"obs": torch.rand(b, 1, generator=g),
            "particles": states.reshape(1, N_STATES, 1).repeat(b, 1, 1),
            "weights": torch.softmax(3 * torch.randn(b, N_STATES, generator=g), dim=-1)}


def _matched(mode, t_frozen=False, **kw):
    raw = cgf_raw_dim(64, 1, mode)
    fixed = non_readout_param_count(64, 1, mode, t_frozen, "running")
    hidden, total = matched_readout_hidden(SMALL_ST_PARAMS, raw, 2, 64, fixed)
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0, t_frozen=t_frozen,
        feature_mode=mode, feature_norm="running",
        readout_hidden=hidden, readout_depth=2, **kw)
    return ext, hidden, total


# ------------------------------------------------------------------ sizing

@pytest.mark.parametrize("mode", ["K", "K_grad", "both"])
@pytest.mark.parametrize("t_frozen", [False, True])
def test_matched_readout_lands_within_half_a_percent_of_the_target(mode, t_frozen):
    ext, hidden, total = _matched(mode, t_frozen)
    assert ext.encoder_parameter_count() == total
    # 'both' has a 128-wide first layer, so one step in hidden moves ~700 params.
    assert abs(total - SMALL_ST_PARAMS) / SMALL_ST_PARAMS < 0.005
    # K and K' have the same raw width on 1-D particles, so the same hidden.
    if mode in ("K", "K_grad"):
        assert hidden == 272


def test_readout_param_count_is_exact():
    ext, hidden, _ = _matched("both")
    mlp = sum(p.numel() for p in ext.readout.parameters())
    assert mlp == readout_param_count(128, hidden, 2, 64)
    assert ext.encoder_parameter_count() == mlp + 64      # + learned t


def test_features_dim_is_obs_plus_readout_dim_and_last_layer_is_linear():
    ext, _, _ = _matched("both")
    assert ext.features_dim == 1 + 64
    assert isinstance(ext.readout[-1], torch.nn.Linear)
    assert ext(_batch()).shape == (6, 65)


def test_readout_dim_defaults_to_64_whatever_the_probe_count():
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=256, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_mode="both",
        feature_norm="running", readout_hidden=16, readout_depth=1)
    assert ext.readout_dim == 64 and ext.features_dim == 65
    assert ext(_batch()).shape == (6, 65)


def test_readout_flags_must_agree_and_readout_dim_needs_a_readout():
    with pytest.raises(ValueError, match="both be > 0"):
        WeightedCGFFeaturesExtractor(_space(), arena_scale=SCALE, readout_hidden=32)
    with pytest.raises(ValueError, match="needs a readout"):
        WeightedCGFFeaturesExtractor(_space(), arena_scale=SCALE, readout_dim=16)


def test_depth_zero_is_bit_identical_to_the_extractor_without_readout_kwargs():
    kw = dict(num_cgf_features=16, arena_scale=SCALE, t_init_mode="spread_1d",
              t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_norm="none")
    a = WeightedCGFFeaturesExtractor(_space(), **kw)
    b = WeightedCGFFeaturesExtractor(_space(), readout_hidden=0, readout_depth=0, **kw)
    assert list(a.state_dict()) == list(b.state_dict()) == ["raw_t"]
    torch.testing.assert_close(a(_batch()), b(_batch()))


# ------------------------------------------------------------------ freezing

def test_freeze_covers_t_norm_statistics_and_readout():
    ext, _, _ = _matched("K_grad")
    ext.train()
    ext(_batch(seed=1))                                   # first norm update
    ext.freeze_encoder()
    before = {k: v.clone() for k, v in ext.state_dict().items()}
    assert ext.encoder_parameter_count(trainable_only=True) == 0
    assert all(not p.requires_grad for p in ext.parameters())

    ext.train(True)                                       # what SB3 does before an update
    assert not ext.training, "a frozen encoder must stay in eval mode"
    out = ext(_batch(seed=2))
    assert not out.requires_grad
    after = ext.state_dict()
    for k in before:
        torch.testing.assert_close(after[k], before[k], rtol=0, atol=0)
    assert int(ext.feature_norm.num_updates) == 1


def test_t_frozen_alone_leaves_the_readout_trainable():
    ext, _, _ = _matched("K", t_frozen=True)
    assert "raw_t" in dict(ext.named_buffers())
    assert ext.encoder_parameter_count(trainable_only=True) == \
        sum(p.numel() for p in ext.readout.parameters())


# ------------------------------------------------------------------ checkpoints

def _save(ext, path, **extra_config):
    torch.save({"model_state_dict": {k: v.clone() for k, v in ext.state_dict().items()},
                "config": {**ext._cgf_geometry, **extra_config}}, path)


def test_checkpoint_round_trip_reproduces_features_and_freezes(tmp_path):
    ext, hidden, _ = _matched("both")
    ext.train(); ext(_batch(seed=3)); ext.eval()
    path = tmp_path / "cgf.pt"
    _save(ext, path)

    loaded = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_mode="both",
        feature_norm="running", readout_hidden=hidden, readout_depth=2,
        pretrained_cgf_model_path=str(path), cgf_frozen=True)
    torch.testing.assert_close(loaded(_batch(seed=4)), ext(_batch(seed=4)))
    assert loaded.encoder_parameter_count(trainable_only=True) == 0
    for k, v in ext.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[k], v, rtol=0, atol=0)


def test_learned_t_checkpoint_loads_into_a_t_frozen_extractor(tmp_path):
    """The frozen RL arm: t learned under the supervised objective, then fixed.
    The state_dict key is the same (raw_t) whether Parameter or buffer."""
    ext, hidden, _ = _matched("K")
    path = tmp_path / "cgf.pt"; _save(ext, path)
    frozen = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
        t_param="tanh", t_bound=50.0, t_init_max=40.0, t_frozen=True, feature_mode="K",
        feature_norm="running", readout_hidden=hidden, readout_depth=2,
        pretrained_cgf_model_path=str(path))
    torch.testing.assert_close(frozen.effective_t(), ext.effective_t())


@pytest.mark.parametrize("field,value", [("feature_mode", "K_grad"), ("t_bound", 25.0),
                                         ("readout_hidden", 100), ("arena_scale", 4.5)])
def test_geometry_mismatch_is_refused_before_the_state_dict_load(tmp_path, field, value):
    ext, hidden, _ = _matched("K")
    path = tmp_path / "cgf.pt"
    _save(ext, path, **{field: value})
    with pytest.raises(RuntimeError, match=f"{field}: checkpoint="):
        WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
            t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_mode="K",
            feature_norm="running", readout_hidden=hidden, readout_depth=2,
            pretrained_cgf_model_path=str(path))


def test_shape_mismatch_the_config_cannot_see_is_still_refused(tmp_path):
    ext, hidden, _ = _matched("K")
    path = tmp_path / "cgf.pt"
    state = {k: v.clone() for k, v in ext.state_dict().items()}
    torch.save({"model_state_dict": state}, path)        # no config at all
    with pytest.raises(RuntimeError, match="Could not load"):
        WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
            t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_mode="K",
            feature_norm="running", readout_hidden=hidden + 1, readout_depth=2,
            pretrained_cgf_model_path=str(path))


def test_frozen_without_path_freezes_but_does_not_raise():
    """SB3 rebuilds the extractor from stored kwargs when a saved policy is
    loaded, with the pretraining path scrubbed; that must construct."""
    ext = WeightedCGFFeaturesExtractor(
        _space(), arena_scale=SCALE, t_init_mode="spread_1d", t_param="tanh",
        t_bound=50.0, t_init_max=40.0, feature_norm="running",
        readout_hidden=8, readout_depth=1, cgf_frozen=True)
    assert ext.encoder_parameter_count(trainable_only=True) == 0


def test_kwargs_round_trip_the_way_sb3_reloads_them():
    ext, hidden, _ = _matched("both")
    kwargs = dict(num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread_1d",
                  t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_mode="both",
                  feature_norm="running", readout_hidden=hidden, readout_depth=2,
                  readout_dim=None, pretrained_cgf_model_path=None, cgf_frozen=True)
    ext.train(); ext(_batch(seed=3)); ext.eval()          # settle the running norm first
    rebuilt = WeightedCGFFeaturesExtractor(_space(), **kwargs)
    rebuilt.load_state_dict(ext.state_dict())
    torch.testing.assert_close(rebuilt(_batch(seed=5)), ext(_batch(seed=5)))


# ------------------------------------------------------------------ learned particle embedding (idea 3)

def _embedded(mode="K", d=4, hidden=64, depth=1):
    raw = cgf_raw_dim(64, d, mode)
    fixed = non_readout_param_count(64, 1, mode, False, "running", d, hidden, depth)
    h, total = matched_readout_hidden(SMALL_ST_PARAMS, raw, 2, 64, fixed)
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="linspace_first_dim",
        t_init_scale=10.0, t_param="tanh", t_bound=50.0, feature_mode=mode,
        feature_norm="running", readout_hidden=h, readout_depth=2,
        x_embed_dim=d, x_embed_hidden=hidden, x_embed_depth=depth)
    return ext, total


@pytest.mark.parametrize("mode,d", [("K", 4), ("K", 16), ("K_grad", 4), ("both", 4)])
def test_embedding_arm_counts_are_exact_and_matched(mode, d):
    ext, total = _embedded(mode, d)
    assert ext.encoder_parameter_count() == total
    assert abs(total - SMALL_ST_PARAMS) / SMALL_ST_PARAMS < 0.005
    assert sum(p.numel() for p in ext.x_embed.parameters()) == x_embed_param_count(1, 64, 1, d)
    assert ext.effective_t().shape == (64, d)
    assert ext.features_dim == 65 and ext(_batch()).shape == (6, 65)


def test_embedding_off_is_the_plain_cgf():
    kw = dict(num_cgf_features=16, arena_scale=SCALE, t_init_mode="spread_1d",
              t_param="tanh", t_bound=50.0, t_init_max=40.0, feature_norm="none")
    a = WeightedCGFFeaturesExtractor(_space(), **kw)
    b = WeightedCGFFeaturesExtractor(_space(), x_embed_dim=0, **kw)
    assert isinstance(b.x_embed, torch.nn.Identity)
    assert list(a.state_dict()) == list(b.state_dict())
    torch.testing.assert_close(a(_batch()), b(_batch()))


def test_embedding_is_frozen_and_checkpointed_with_the_rest(tmp_path):
    ext, _ = _embedded("K", 4)
    ext.train(); ext(_batch(seed=1)); ext.eval()
    path = tmp_path / "cgf_embed.pt"; _save(ext, path)
    loaded = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="linspace_first_dim",
        t_init_scale=10.0, t_param="tanh", t_bound=50.0, feature_mode="K",
        feature_norm="running", readout_hidden=ext.readout_hidden, readout_depth=2,
        x_embed_dim=4, x_embed_hidden=64, x_embed_depth=1,
        pretrained_cgf_model_path=str(path), cgf_frozen=True)
    torch.testing.assert_close(loaded(_batch(seed=2)), ext(_batch(seed=2)))
    assert all(not p.requires_grad for p in loaded.x_embed.parameters())
    with pytest.raises(RuntimeError, match="x_embed_dim: checkpoint="):
        WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="linspace_first_dim",
            t_init_scale=10.0, t_param="tanh", t_bound=50.0, feature_mode="K",
            feature_norm="running", readout_hidden=ext.readout_hidden, readout_depth=2,
            x_embed_dim=8, x_embed_hidden=64, x_embed_depth=1,
            pretrained_cgf_model_path=str(path))
