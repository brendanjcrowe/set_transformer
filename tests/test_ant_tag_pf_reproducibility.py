import importlib
import sys
from pathlib import Path

import numpy as np

from set_transformer.rl.particle_filters.ant_tag import (
    CounterweightedDenAntTagParticleFilter,
)


def _make_pf(seed):
    return CounterweightedDenAntTagParticleFilter(
        num_particles=100,
        initial_env_obs=np.zeros(31, dtype=np.float32),
        arena_limits=(-7.0, 7.0),
        visibility_radius=1.0,
        tag_radius=0.6,
        rng_seed=seed,
    )


def test_filter_owned_rng_reproduces_all_particle_draws():
    a = _make_pf(12345)
    b = _make_pf(12345)
    c = _make_pf(54321)

    np.testing.assert_array_equal(a.particles, b.particles)
    assert not np.array_equal(a.particles, c.particles)

    diag = np.array([1.0, 1.0]) / np.sqrt(2.0)
    predict_kwargs = {
        "ant_current_pos_from_obs": np.zeros(2),
        "cden_heavy_pos": 2.4 * diag,
        "cden_light_pos": -6.75 * diag,
        "cden_w_heavy": 6.75 / (2.4 + 6.75),
        "cden_r": 0.4,
        "cden_spooked": False,
    }
    action = np.zeros(8, dtype=np.float32)

    for _ in range(10):
        a.predict(action, **predict_kwargs)
        b.predict(action, **predict_kwargs)
        np.testing.assert_array_equal(a.particles, b.particles)
        np.testing.assert_array_equal(a.weights, b.weights)


def test_wrapped_worker_seed_reproduces_multiple_episode_stream():
    ant_tag_dir = Path(__file__).resolve().parents[1] / "experiments" / "ant_tag"
    sys.path.insert(0, str(ant_tag_dir))
    train = importlib.import_module("4_train_rl_cgf")

    kwargs = dict(
        num_particles=100,
        rank=2,
        seed=17,
        distance_coeff=0.0,
        entropy_coeff=0.0,
        tag_bonus_coeff=0.0,
        initial_visibility_radius=1.0,
        apply_reward_shaping=False,
        env_id="pdomains-ant-tag-cdens-terminal-v0",
        particle_filter_class=CounterweightedDenAntTagParticleFilter,
        target_speed_scale=0.0,
    )
    env_a = train.make_ant_tag_cgf_env(**kwargs)()
    env_b = train.make_ant_tag_cgf_env(**kwargs)()
    action = np.zeros(env_a.action_space.shape, dtype=np.float32)

    try:
        # Both independently constructed workers advance through the same
        # deterministic, episode-varying PF stream.
        for reset_kwargs in ({"seed": 901}, {}):
            obs_a, _ = env_a.reset(**reset_kwargs)
            obs_b, _ = env_b.reset(**reset_kwargs)
            for key in obs_a:
                np.testing.assert_array_equal(obs_a[key], obs_b[key])

            for _ in range(8):
                obs_a, reward_a, term_a, trunc_a, info_a = env_a.step(action)
                obs_b, reward_b, term_b, trunc_b, info_b = env_b.step(action)
                for key in obs_a:
                    np.testing.assert_array_equal(obs_a[key], obs_b[key])
                assert reward_a == reward_b
                assert term_a == term_b
                assert trunc_a == trunc_b
                assert info_a["is_success"] == info_b["is_success"]
    finally:
        env_a.close()
        env_b.close()
