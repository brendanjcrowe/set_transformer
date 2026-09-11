"""A dead belief (all-zero, NaN or inf PF weights) must never reach the policy.

PITFALLS.md section 8, item 8. The three encoders disagree about a dead row:
CGF in K mode emits a sentinel, CGF in K' mode the tilted mean of a UNIFORM
belief, and the Gaussian extractor mean 0 / covariance 0 -- a delta at the
arena centre. In two of three modes a filter failure therefore looks like a
real belief. PFDictWithWeightsObservationWrapper is the one place every
domain's belief passes through, so the guard lives there: raise by default,
or substitute uniform weights and flag it when asked to.

The filter here is a stub whose weights are scripted per step, so the test
exercises the wrapper alone. Neither real filter family can produce a dead
row today (Odd-Even raises itself; Ant-Tag floors at 1e-300), which is the
premise of the "raise" default and is asserted at the end.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

from set_transformer.rl.wrappers.particle_filter import (  # noqa: E402
    PFDictWithWeightsObservationWrapper,
)

N = 8


class _Env(gym.Env):
    observation_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, np.float32), {}

    def step(self, action):
        return np.zeros(2, np.float32), 0.0, False, False, {}


class _ScriptedFilter:
    """weights come from SCRIPT[step]; step 0 is the value at reset."""
    SCRIPT = []
    particle_dim = 1

    def __init__(self, num_particles, initial_env_obs=None, rng_seed=None, **_):
        self.n = num_particles
        self.k = 0
        self.particles = np.linspace(-1.0, 1.0, num_particles)[:, None]

    @property
    def weights(self):
        w = self.SCRIPT[min(self.k, len(self.SCRIPT) - 1)]
        return np.full(self.n, 1.0 / self.n) if w is None else np.asarray(w, dtype=np.float64)

    def predict(self, action, **_):
        pass

    def update(self, obs, **_):
        self.k += 1


def _wrap(script, **kwargs):
    class F(_ScriptedFilter):
        SCRIPT = script
    return PFDictWithWeightsObservationWrapper(
        _Env(), F, {}, num_particles=N, particle_filter_seed=0, **kwargs)


def test_live_weights_pass_through_unchanged():
    live = np.exp(-np.arange(N)); live /= live.sum()
    env = _wrap([None, live])
    obs, _ = env.reset()
    np.testing.assert_allclose(obs["weights"], np.full(N, 1 / N))
    obs = env.step(0)[0]
    np.testing.assert_allclose(obs["weights"], live.astype(np.float32))
    assert env.dead_belief_count == 0


@pytest.mark.parametrize("dead,label", [
    (np.zeros(N), "zero mass"),
    (np.r_[np.nan, np.ones(N - 1) / (N - 1)], "NaN"),
    (np.r_[np.inf, np.ones(N - 1) / (N - 1)], "inf"),
])
def test_dead_belief_raises_by_default_and_names_the_filter(dead, label):
    env = _wrap([None, dead])
    env.reset()
    with pytest.raises(RuntimeError, match="F produced a dead belief") as excinfo:
        env.step(0)
    assert "on_dead_belief='uniform'" in str(excinfo.value), label
    assert env.dead_belief_count == 1


def test_dead_belief_at_reset_is_caught_too():
    env = _wrap([np.zeros(N)])
    with pytest.raises(RuntimeError, match="dead belief"):
        env.reset()


def test_uniform_mode_substitutes_flags_and_counts(capsys):
    live = np.exp(-np.arange(N)); live /= live.sum()
    env = _wrap([None, np.zeros(N), live, np.full(N, np.nan)], on_dead_belief="uniform")
    env.reset()
    obs, _, _, _, info = env.step(0)                         # dead -> uniform
    np.testing.assert_allclose(obs["weights"], np.full(N, 1 / N, np.float32))
    assert info.get("pf_dead_belief") is True
    obs, _, _, _, info = env.step(0)                         # live again
    np.testing.assert_allclose(obs["weights"], live.astype(np.float32))
    assert "pf_dead_belief" not in info
    env.step(0)                                              # dead (NaN)
    assert env.dead_belief_count == 2
    out = capsys.readouterr().out
    assert out.count("substituting uniform weights") == 1, "warn once, count always"


def test_rejects_an_unknown_policy():
    with pytest.raises(ValueError, match="on_dead_belief"):
        _wrap([None], on_dead_belief="ignore")


def test_the_real_filters_cannot_produce_a_dead_row():
    """The premise of the 'raise' default: total refutation comes out as
    uniform (Ant-Tag: += 1e-300 before normalising) or as an exception from
    the filter itself (Odd-Even), never as a zero / NaN row."""
    from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter
    from set_transformer.rl.particle_filters.odd_even import (
        OddEvenExactSupportParticleFilter,
    )

    pf = AntTagParticleFilter(num_particles=N, initial_env_obs=np.zeros(31, np.float32),
                              rng_seed=0)
    pf.update(np.array([1e6, 1e6]), np.zeros(2))          # observation nowhere near any particle
    assert np.all(np.isfinite(pf.weights)) and pf.weights.sum() > 0
    np.testing.assert_allclose(pf.weights, np.full(N, 1 / N))

    # Odd-Even: an even observation zeroes every odd state. Building the filter
    # on an odd observation and then feeding it an even one leaves NO mass; the
    # filter raises instead of emitting a zero row.
    pf = OddEvenExactSupportParticleFilter(
        num_particles=10, initial_env_obs=np.array([3.0], np.float32), n_dist_size=10, rng_seed=0)
    assert pf.weights[1::2].sum() == 0.0                    # even states are dead ...
    with pytest.raises(ValueError, match="total probability"):
        pf.update(np.array([4.0], np.float32))              # ... so an even obs kills them all
