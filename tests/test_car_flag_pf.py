"""Tests for the Car-Flag particle filter, success detector, and belief potential."""

import numpy as np
import pytest

from set_transformer.rl.particle_filters.car_flag import (
    DIRECTION_OBS_INDEX,
    CarFlagParticleFilter,
)


def make_pf(n=100, **kw):
    # initial_env_obs is unused by this PF's prior, but the base API requires it.
    return CarFlagParticleFilter(num_particles=n, initial_env_obs=np.zeros(3), **kw)


def obs(position=0.0, velocity=0.0, direction=0.0):
    o = np.zeros(3, dtype=np.float32)
    o[0], o[1], o[DIRECTION_OBS_INDEX] = position, velocity, direction
    return o


# --- Prior ---------------------------------------------------------------------

def test_prior_is_uniform_binary():
    pf = make_pf(100)
    assert pf.particles.shape == (100, 1)
    assert pf.particle_dim == 1
    vals = set(np.unique(pf.particles.round(6)).tolist())
    assert vals == {-1.0, 1.0}
    assert pf.particles.sum() == 0  # exact half/half split
    assert pf.heaven_right_prob() == pytest.approx(0.5)
    np.testing.assert_allclose(pf.weights, np.ones(100) / 100)


# --- predict is a no-op (static latent) ----------------------------------------

def test_predict_does_not_move_particles():
    pf = make_pf(50)
    before = pf.particles.copy()
    pf.predict(np.array([0.5]))
    np.testing.assert_array_equal(pf.particles, before)


# --- update on the priest reading ----------------------------------------------

def test_direction_zero_is_uninformative():
    pf = make_pf(100)
    w0 = pf.weights.copy()
    pf.update(obs(position=0.0, direction=0.0))
    np.testing.assert_array_equal(pf.weights, w0)
    assert pf.heaven_right_prob() == pytest.approx(0.5)


def test_priest_reveal_collapses_belief_to_right():
    pf = make_pf(100)
    pf.update(obs(position=0.5, direction=1.0))  # heaven revealed on the right
    assert pf.heaven_right_prob() == pytest.approx(1.0)
    assert np.all(pf.particles > 0)  # resampled to a delta at +1


def test_priest_reveal_collapses_belief_to_left():
    pf = make_pf(100)
    pf.update(obs(position=0.5, direction=-1.0))
    assert pf.heaven_right_prob() == pytest.approx(0.0)
    assert np.all(pf.particles < 0)


def test_belief_persists_after_reveal_then_zero_readings():
    """Once revealed, subsequent uninformative steps must not un-collapse the belief."""
    pf = make_pf(100)
    pf.update(obs(position=0.5, direction=1.0))
    for _ in range(5):
        pf.predict(np.array([0.0]))
        pf.update(obs(direction=0.0))
    assert pf.heaven_right_prob() == pytest.approx(1.0)


def test_weights_stay_normalized_and_finite():
    pf = make_pf(64)
    for d in (0.0, -1.0, 0.0):
        pf.update(obs(direction=d))
        assert np.all(np.isfinite(pf.weights))
        assert pf.weights.sum() == pytest.approx(1.0)


# --- success detector ----------------------------------------------------------

@pytest.mark.parametrize("terminal,length,expected", [
    (0.0, 40, True),    # reached heaven-right (terminal 0)
    (1.0, 20, True),    # reached heaven-left (terminal +1)
    (-5.0, 30, False),  # reached hell
    (-1.0, 160, False), # timed out (last step is a normal -1)
])
def test_car_flag_success(terminal, length, expected):
    from set_transformer.rl.benchmark.envs import car_flag_success
    # Reconstruct the episode return the env would produce: (length-1) steps of -1 + terminal.
    episode_return = -(length - 1) + terminal
    assert car_flag_success(episode_return, length) is expected


# --- belief potential ----------------------------------------------------------

def test_belief_potential_prefers_confident_alignment():
    """After a right-reveal, the potential should rise as the car nears the +1 flag."""
    from types import SimpleNamespace

    from set_transformer.rl.benchmark.envs import car_flag_belief_potential

    pf = make_pf(100)
    pf.update(obs(position=0.5, direction=1.0))  # belief: heaven on the right (+1)

    def fake_env(position):
        # find_particle_filter walks .env; expose the PF and unwrapped .state.
        return SimpleNamespace(
            particle_filter=pf,
            unwrapped=SimpleNamespace(state=np.array([position, 0.0, 0.0])),
        )

    near = car_flag_belief_potential(fake_env(0.9))
    far = car_flag_belief_potential(fake_env(-0.9))
    assert near > far
    assert near == pytest.approx(-abs(0.9 - 1.0))  # delta belief => plain distance to +1
