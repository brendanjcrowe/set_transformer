"""Tests for the env-specific PF interaction mappers and PF-kwargs bridges.

The Ant-Tag mapper is pure numpy (no MuJoCo needed); the env-derived PF kwargs are
checked against a stub env, plus a MuJoCo-gated check against the real Ant-Tag.

2026-09-12: the functions under test are the ones the Ant-Tag arm scripts use
(``set_transformer.rl.domains.ant_tag``); the stripped copy in ``rl/mappers.py`` was
retired. The arm version also reads ``tag_radius`` off the env and forwards the live
visibility radius (and the evasion / target-speed / den knobs when the env has them), so
the mapper is called with the env and the kwargs carry one more key.
"""

import numpy as np
import pytest

from set_transformer.rl.domains.ant_tag import (
    ant_tag_pf_interaction_mapper,
    get_ant_tag_pf_kwargs,
)


def _obs(ant_xy, target_xy):
    """A 31-D Ant-Tag-shaped observation: ant xy up front, target xy in the last two."""
    obs = np.zeros(31, dtype=np.float32)
    obs[:2] = ant_xy
    obs[-2:] = target_xy
    return obs


def test_predict_uses_previous_ant_position_update_uses_current():
    # The target moved in response to where the ant WAS, so the belief must be propagated
    # against the pre-step position; visibility this step is decided by the current one.
    previous = _obs([1.0, 2.0], [0.0, 0.0])
    current = _obs([1.5, 2.5], [3.0, 4.0])

    args = ant_tag_pf_interaction_mapper(
        base_env_obs=current,
        base_env_info={},
        unwrapped_env=_StubAntTagEnv(),
        previous_base_env_obs=previous,
    )

    assert args["predict_args"]["ant_current_pos_from_obs"] == pytest.approx([1.0, 2.0])
    assert args["update_args"]["ant_current_pos_from_obs"] == pytest.approx([1.5, 2.5])
    assert args["update_args"]["observed_target_pos"] == pytest.approx([3.0, 4.0])


def test_first_step_falls_back_to_current_position():
    current = _obs([1.5, 2.5], [3.0, 4.0])
    args = ant_tag_pf_interaction_mapper(base_env_obs=current, base_env_info={},
                                         unwrapped_env=_StubAntTagEnv())
    assert args["predict_args"]["ant_current_pos_from_obs"] == pytest.approx([1.5, 2.5])


def test_zero_target_reads_as_not_visible():
    # The env zeros obs[-2:] when the target is out of visual range.
    args = ant_tag_pf_interaction_mapper(
        base_env_obs=_obs([1.0, 1.0], [0.0, 0.0]), base_env_info={},
        unwrapped_env=_StubAntTagEnv(),
    )
    assert np.all(np.isnan(args["update_args"]["observed_target_pos"]))


def test_mapper_does_not_alias_the_observation():
    current = _obs([1.5, 2.5], [3.0, 4.0])
    args = ant_tag_pf_interaction_mapper(base_env_obs=current, base_env_info={},
                                         unwrapped_env=_StubAntTagEnv())
    current[:2] = 99.0
    assert args["update_args"]["ant_current_pos_from_obs"] == pytest.approx([1.5, 2.5])


class _StubAntTagEnv:
    cage_max_x = 7.0
    cage_max_y = 7.0
    target_step = 0.25
    visible_radius = 2.0
    min_distance = 4.0
    tag_radius = 1.0

    @property
    def unwrapped(self):
        return self


def test_pf_kwargs_read_off_the_env():
    kwargs = get_ant_tag_pf_kwargs(_StubAntTagEnv())
    assert kwargs == {
        "arena_limits": (-7.0, 7.0),
        "target_step": 0.25,
        "visibility_radius": 2.0,
        "min_initial_distance": 4.0,
        "tag_radius": 1.0,
    }


def test_non_square_arena_rejected():
    env = _StubAntTagEnv()
    env.cage_max_y = 14.0
    with pytest.raises(ValueError, match="square arena"):
        get_ant_tag_pf_kwargs(env)


def test_registry_ant_tag_pf_kwargs_match_the_live_env():
    """The registry's env-derived hook must reproduce the real Ant-Tag's own constants."""
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    pytest.importorskip("stable_baselines3")

    from set_transformer.rl.benchmark.registry import get_env_spec

    spec = get_env_spec("ant_tag")
    assert spec.pf_kwargs_from_env is get_ant_tag_pf_kwargs

    env = spec.make_base_env(seed=0)
    try:
        kwargs = spec.pf_kwargs_from_env(env)
        u = env.unwrapped
        assert kwargs["arena_limits"] == (-u.cage_max_x, u.cage_max_x)
        assert kwargs["target_step"] == u.target_step
        assert kwargs["visibility_radius"] == u.visible_radius
        assert kwargs["min_initial_distance"] == u.min_distance
        assert kwargs["tag_radius"] == u.tag_radius
    finally:
        env.close()


def test_ant_tag_prior_respects_the_env_reset_distance():
    """t=0 particles must sit at least ``min_distance`` from the ant, as the env does."""
    pytest.importorskip("gymnasium")
    pytest.importorskip("mujoco")
    pytest.importorskip("stable_baselines3")

    from set_transformer.rl.benchmark.registry import get_env_spec
    from set_transformer.rl.wrappers.particle_filter import PFDictObservationWrapper

    spec = get_env_spec("ant_tag")
    base = spec.make_base_env(seed=0)
    try:
        env = PFDictObservationWrapper(
            env=base,
            particle_filter_class=spec.particle_filter_class,
            particle_filter_kwargs=spec.pf_kwargs_from_env(base),
            num_particles=spec.num_particles,
            pf_interaction_mapper=spec.pf_mapper,
            obs_mask_indices=spec.obs_mask_indices,
        )
        obs, _ = env.reset(seed=0)
        particles = env.particle_filter.particles
        distances = np.linalg.norm(particles - obs["obs"][:2], axis=1)
        assert distances.min() > base.unwrapped.min_distance
        assert np.abs(particles).max() <= base.unwrapped.cage_max_x
    finally:
        base.close()
