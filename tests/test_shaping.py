"""Tests for potential-based reward shaping.

Skipped automatically if the ``[rl]`` extra (gymnasium) is not installed.
"""

import pytest

gym = pytest.importorskip("gymnasium")

import numpy as np

from set_transformer.rl.wrappers.shaping import (
    PotentialBasedShapingWrapper,
    find_particle_filter,
    pf_belief_expected_distance_potential,
)


class _LineEnv(gym.Env):
    """1-D walk toward a goal. State = position; true reward -1/step, 0 at goal.

    ``terminate_at_goal`` toggles whether reaching the goal is a true termination
    (absorbing) or the episode simply runs to a fixed horizon.
    """

    def __init__(self, goal=5.0, horizon=8, terminate_at_goal=True):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)  # 0 stay, 1 step toward goal
        self.goal = goal
        self.horizon = horizon
        self.terminate_at_goal = terminate_at_goal

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0.0
        self.t = 0
        return np.array([self.pos], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        if action == 1:
            self.pos += 1.0
        at_goal = self.pos >= self.goal
        reward = 0.0 if at_goal else -1.0
        terminated = bool(at_goal and self.terminate_at_goal)
        truncated = self.t >= self.horizon
        return np.array([self.pos], dtype=np.float32), reward, terminated, truncated, {}


def _pos_potential(env):
    """Phi(s) = -|goal - pos|  (true-state potential)."""
    return -abs(env.unwrapped.goal - env.unwrapped.pos)


def test_shaping_is_reported_in_info():
    env = PotentialBasedShapingWrapper(_LineEnv(), _pos_potential, gamma=0.9)
    env.reset()
    obs, shaped, term, trunc, info = env.step(1)
    assert info["shaping_reward"] == pytest.approx(shaped - info["true_reward"])
    assert "potential" in info


def test_telescoping_sum_terminated():
    """Sum of discounted shaping over an episode telescopes to -Phi(s_0) when the
    goal is an absorbing terminal state (Phi(s_T) := 0)."""
    gamma = 0.9
    env = PotentialBasedShapingWrapper(
        _LineEnv(goal=3.0, horizon=10, terminate_at_goal=True), _pos_potential, gamma
    )
    _, info = env.reset()
    phi0 = info["potential"]
    discounted = 0.0
    g = 1.0
    steps = 0
    terminated = False
    while True:
        _, _, terminated, truncated, info = env.step(1)  # always step toward goal
        discounted += g * info["shaping_reward"]
        g *= gamma
        steps += 1
        if terminated or truncated:
            break
    assert terminated, "expected to reach the absorbing goal"
    # sum_t gamma^t F_t = -Phi(s_0) exactly (Phi(s_T)=0).
    assert discounted == pytest.approx(-phi0, abs=1e-6)


def test_terminal_potential_is_zero():
    env = PotentialBasedShapingWrapper(
        _LineEnv(goal=1.0, horizon=5, terminate_at_goal=True), _pos_potential, gamma=0.9
    )
    env.reset()
    _, _, terminated, _, info = env.step(1)  # reaches goal immediately
    assert terminated
    assert info["potential"] == 0.0


def test_truncation_keeps_potential():
    """On truncation (time limit, non-absorbing) Phi(s') is NOT zeroed."""
    env = PotentialBasedShapingWrapper(
        _LineEnv(goal=100.0, horizon=1, terminate_at_goal=True), _pos_potential, gamma=0.9
    )
    env.reset()
    _, _, terminated, truncated, info = env.step(0)  # stay; hits horizon
    assert truncated and not terminated
    assert info["potential"] != 0.0


def test_true_reward_preserved():
    env = PotentialBasedShapingWrapper(_LineEnv(), _pos_potential, gamma=0.99)
    env.reset()
    _, _, _, _, info = env.step(0)
    assert info["true_reward"] == -1.0


# --- belief potential ----------------------------------------------------------


class _FakePF:
    def __init__(self, particles, weights=None):
        self._p = np.asarray(particles, dtype=np.float64)
        self._w = None if weights is None else np.asarray(weights, dtype=np.float64)

    @property
    def particles(self):
        return self._p

    @property
    def weights(self):
        return self._w


class _PFHolder(gym.Wrapper):
    """Stand-in for PFDictObservationWrapper: exposes a live ``particle_filter``."""

    def __init__(self, env, pf):
        super().__init__(env)
        self.particle_filter = pf


def test_find_particle_filter_walks_chain():
    pf = _FakePF([[0.0, 0.0]])
    env = _PFHolder(_LineEnv(), pf)
    # An extra wrapper on top must not hide the PF.
    top = PotentialBasedShapingWrapper(env, lambda e: 0.0, gamma=0.9)
    assert find_particle_filter(top) is pf


def test_belief_potential_expected_distance():
    # particles at distance 3 and 5 from agent at origin; uniform weights -> mean 4.
    pf = _FakePF([[3.0, 0.0], [0.0, 5.0]])
    env = _PFHolder(_LineEnv(), pf)
    pot = pf_belief_expected_distance_potential(get_agent_pos=lambda e: np.zeros(2))
    assert pot(env) == pytest.approx(-4.0)


def test_belief_potential_uses_weights():
    pf = _FakePF([[3.0, 0.0], [0.0, 5.0]], weights=[3.0, 1.0])
    env = _PFHolder(_LineEnv(), pf)
    pot = pf_belief_expected_distance_potential(get_agent_pos=lambda e: np.zeros(2))
    # weighted mean distance = (3*3 + 1*5) / 4 = 3.5
    assert pot(env) == pytest.approx(-3.5)


def test_find_particle_filter_missing_raises():
    env = PotentialBasedShapingWrapper(_LineEnv(), lambda e: 0.0, gamma=0.9)
    with pytest.raises(AttributeError, match="particle_filter"):
        find_particle_filter(env)
