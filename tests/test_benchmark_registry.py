"""Tests for the benchmark registry and env/wrapper wiring.

Only constructs the Odd-Even env (no MuJoCo); Ant-Tag is checked via its spec only.
Skipped if the ``[rl]`` extra is missing.
"""

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from set_transformer.rl.benchmark.registry import (
    ENV_REGISTRY,
    METHOD_REGISTRY,
    build_extractor_kwargs,
    get_env_spec,
    get_method_spec,
)
from set_transformer.rl.wrappers.shaping import PotentialBasedShapingWrapper


def test_registries_populated():
    assert {"ant_tag", "car_flag", "odd_even"} <= set(ENV_REGISTRY)
    assert {"gaussian", "kmoments", "cgf", "st_frozen", "st_finetune", "st_scratch"} <= set(METHOD_REGISTRY)


def test_car_flag_spec_is_unshaped_and_maskless():
    # Hidden-side env: runs sparse first (no potential), and direction is observed
    # legitimately so nothing is masked and no mapper is needed.
    spec = get_env_spec("car_flag")
    assert spec.potential_fn is None
    assert spec.obs_mask_indices is None
    assert spec.pf_mapper is None
    assert spec.success_fn is not None and spec.success_criterion


def test_unknown_lookups_raise():
    with pytest.raises(KeyError, match="Unknown env"):
        get_env_spec("nope")
    with pytest.raises(KeyError, match="Unknown method"):
        get_method_spec("nope")


def test_st_frozen_requires_path():
    with pytest.raises(ValueError, match="pretrained"):
        build_extractor_kwargs(get_method_spec("st_frozen"), 128, [64, 64])


def test_st_arch_and_path_injected_for_st_only():
    st_arch = dict(num_encodings=4, dim_encoder=2, num_inds=8, dim_hidden=32, num_heads=2, ln=True)
    kw = build_extractor_kwargs(
        get_method_spec("st_finetune"), 128, [64, 64],
        pretrained_model_path="/tmp/x.pt", st_arch=st_arch,
    )
    assert kw["pretrained_model_path"] == "/tmp/x.pt"
    assert kw["num_encodings"] == 4
    # Non-ST methods ignore ST arch / path.
    kw2 = build_extractor_kwargs(
        get_method_spec("gaussian"), 128, [64, 64],
        pretrained_model_path="/tmp/x.pt", st_arch=st_arch,
    )
    assert "pretrained_model_path" not in kw2
    assert "num_encodings" not in kw2


def test_shaping_policy_per_env():
    # Ant-Tag shapes (true-state potential); Odd-Even does not (dense native reward).
    assert get_env_spec("ant_tag").potential_fn is not None
    assert get_env_spec("odd_even").potential_fn is None


def test_odd_even_env_stack_produces_dict_obs():
    from set_transformer.rl.wrappers.particle_filter import PFDictObservationWrapper

    spec = get_env_spec("odd_even")
    env = spec.make_base_env(seed=0)
    env = PFDictObservationWrapper(
        env=env,
        particle_filter_class=spec.particle_filter_class,
        particle_filter_kwargs=spec.particle_filter_kwargs,
        num_particles=spec.num_particles,
        pf_interaction_mapper=spec.pf_mapper,
        obs_mask_indices=spec.obs_mask_indices,
    )
    obs, _ = env.reset()
    assert set(obs.keys()) == {"obs", "particles"}
    assert obs["particles"].shape == (spec.num_particles, 1)
    # Odd-Even is unshaped: no PotentialBasedShapingWrapper needed.
    assert spec.potential_fn is None
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert "particles" in obs


def test_shaping_wrapper_reports_true_reward_when_applied():
    """When a potential is present, shaped step exposes the true reward in info."""
    spec = get_env_spec("odd_even")
    base = spec.make_base_env(seed=1)
    # Attach a trivial potential to exercise the wrapper on a non-MuJoCo env.
    wrapped = PotentialBasedShapingWrapper(base, potential_fn=lambda e: 0.0, gamma=spec.gamma)
    wrapped.reset()
    _, shaped, _, _, info = wrapped.step(wrapped.action_space.sample())
    assert info["true_reward"] == pytest.approx(shaped)  # zero potential => no change
