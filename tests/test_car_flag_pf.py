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
    (1.0, 40, True),    # reached heaven
    (1.0, 159, True),   # reached heaven slowly (step penalty must not flip the verdict)
    (-1.0, 30, False),  # reached hell
    (-1.0, 5, False),   # reached hell fast
])
def test_car_flag_success(terminal, length, expected):
    from set_transformer.rl.benchmark.envs import CAR_FLAG_STEP_PENALTY, car_flag_success
    # Reconstruct the return the env produces: (length-1) penalty steps + the terminal.
    episode_return = CAR_FLAG_STEP_PENALTY * (length - 1) + terminal
    assert car_flag_success(episode_return, length) is expected


def test_car_flag_success_rejects_timeout():
    """A 160-step timeout has no terminal payout and must not count as success."""
    from set_transformer.rl.benchmark.envs import CAR_FLAG_STEP_PENALTY, car_flag_success

    assert car_flag_success(CAR_FLAG_STEP_PENALTY * 160, 160) is False


# --- corrected reward wrapper ---------------------------------------------------

def test_reward_wrapper_pays_by_outcome():
    """Heaven/hell are decided by which flag the car reaches, not by which side is which."""
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("pdomains")

    from set_transformer.rl.benchmark.envs import (
        CAR_FLAG_HEAVEN_REWARD,
        CAR_FLAG_HELL_REWARD,
        CAR_FLAG_STEP_PENALTY,
        make_car_flag_base_env,
    )

    for seed in range(6):  # spans both heaven sides
        env = make_car_flag_base_env(seed=seed)
        env.reset(seed=seed)
        heaven = float(env.unwrapped.heaven_position)
        rewards = []
        while True:
            obs, r, term, trunc, info = env.step([1.0])  # always drive right
            rewards.append(r)
            if term or trunc:
                break
        assert all(r == CAR_FLAG_STEP_PENALTY for r in rewards[:-1])
        if term:
            # Driving right reaches the +1 flag: heaven iff heaven is the right flag.
            expected = CAR_FLAG_HEAVEN_REWARD if heaven > 0 else CAR_FLAG_HELL_REWARD
            assert rewards[-1] == expected
            assert info["reached_heaven"] is (heaven > 0)
        env.close()


def test_information_now_pays():
    """The whole point of the correction: using the priest must beat gambling."""
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("pdomains")

    from set_transformer.rl.benchmark.envs import make_car_flag_base_env

    def episode_return(policy, seed):
        env = make_car_flag_base_env(seed=seed)
        obs, _ = env.reset(seed=seed)
        total, revealed = 0.0, 0.0
        while True:
            if obs[2] != 0.0:
                revealed = obs[2]
            obs, r, term, trunc, _ = env.step([policy(revealed)])
            total += r
            if term or trunc:
                env.close()
                return total

    gambler = np.mean([episode_return(lambda d: 1.0, s) for s in range(40)])
    informed = np.mean([
        episode_return(lambda d: float(np.sign(d)) if d != 0.0 else 1.0, s)
        for s in range(40)
    ])
    assert informed > gambler


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
