"""The spread-gain shaping term (PFRewardShapingWrapper, 2026-09-08).

Pins: the reward schedule parser accepts 3-, 4- and 5-field entries and pads
missing trailing fields with 0 (so every existing schedule keeps its
meaning); the wrapper adds gain_coeff * (spread_{t-1} - spread_t) with the
collector's weighted-spread measure, resets its baseline on reset(), and with
gain_coeff 0 reproduces the historical reward bit for bit; the curriculum's
3-argument set_reward_coeffs path still works and the 4-argument path reaches
the wrapper through _CurriculumRouter (the SubprocVecEnv env_method route).
"""
import importlib
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
_ANT_TAG_DIR = _ST_ROOT / "experiments" / "ant_tag"
for _p in (str(_ST_ROOT), str(_ANT_TAG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco", reason="4_train_rl_frozen imports the Ant-Tag env module")


@pytest.fixture(scope="module")
def mods():
    pytest.importorskip("pdomains")
    return importlib.import_module("4_train_rl_frozen"), importlib.import_module("4_train_rl_cgf"), \
        importlib.import_module("4_train_rl_st")


class _FakePF:
    def __init__(self, particles, weights=None):
        self.particles = np.asarray(particles, dtype=float)
        n = len(self.particles)
        self.weights = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)


class _StubBeliefEnv(gym.Env):
    """Minimal stand-in for the PF wrapper stack: exposes .particle_filter and a dict obs."""
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {"obs": gym.spaces.Box(-np.inf, np.inf, (31,), np.float32)})
        self.action_space = gym.spaces.Box(-1, 1, (8,), np.float32)
        self.particle_filter = _FakePF(np.random.default_rng(0).uniform(-4.5, 4.5, (100, 2)))
        self.terminate_next = False

    def reset(self, **kwargs):
        return {"obs": np.zeros(31, np.float32)}, {}

    def step(self, action):
        return {"obs": np.zeros(31, np.float32)}, -1.0, self.terminate_next, False, {}


def _spread(pf):
    return float(np.sqrt(np.average((pf.particles - np.average(pf.particles, weights=pf.weights, axis=0)) ** 2,
                                    weights=pf.weights, axis=0)).mean())


@pytest.mark.parametrize("entry,expected", [
    ("0:1:2", (0.0, 1.0, 2.0, 0.0, 0.0)),
    ("0.5:0.15:2:50", (0.5, 0.15, 2.0, 50.0, 0.0)),
    ("0.5:0.15:2:50:10", (0.5, 0.15, 2.0, 50.0, 10.0)),
])
def test_parser_pads_to_five_fields(mods, entry, expected):
    _, cgf, st = mods
    assert cgf._parse_reward_schedule(entry) == [expected]
    assert st._parse_reward_schedule(entry) == [expected]


def test_parser_rejects_two_or_six_fields(mods):
    _, cgf, _ = mods
    for bad in ("0:1", "0:1:2:3:4:5"):
        with pytest.raises(ValueError):
            cgf._parse_reward_schedule(bad)


def test_gain_zero_reproduces_historical_reward(mods):
    frozen, _, _ = mods
    env = frozen.PFRewardShapingWrapper(_StubBeliefEnv(), distance_coeff=1.0, entropy_coeff=2.0,
                                        tag_bonus_coeff=50.0)
    assert env.gain_coeff == 0.0
    env.reset()
    pf = env.env.particle_filter
    pf.particles[:] *= 0.1        # belief collapses: a big spread gain, must be worth 0
    _, r, _, _, info = env.step(env.action_space.sample())
    # the historical formula, rebuilt explicitly
    dist = float(np.linalg.norm(np.zeros(2) - np.average(pf.particles, weights=pf.weights, axis=0)))
    entropy = -float(np.sum(pf.weights * np.log(pf.weights)))
    assert r == pytest.approx(-1.0 - dist - 2.0 * entropy)
    assert info["gain_reward"] == 0.0 and info["pf_spread_gain"] > 0


def test_gain_pays_for_shrinking_spread_and_resets_baseline(mods):
    frozen, _, _ = mods
    env = frozen.PFRewardShapingWrapper(_StubBeliefEnv(), distance_coeff=0.0, entropy_coeff=0.0,
                                        tag_bonus_coeff=0.0, gain_coeff=10.0)
    pf = env.env.particle_filter
    env.reset()
    s0 = _spread(pf)
    pf.particles[:] *= 0.5                       # spread halves
    _, r, _, _, info = env.step(env.action_space.sample())
    s1 = _spread(pf)
    assert info["pf_spread"] == pytest.approx(s1)
    assert info["pf_spread_gain"] == pytest.approx(s0 - s1)
    assert r == pytest.approx(-1.0 + 10.0 * (s0 - s1))
    # no change -> no gain; growth -> negative gain
    _, r, _, _, _ = env.step(env.action_space.sample())
    assert r == pytest.approx(-1.0)
    pf.particles[:] *= 3.0
    _, r, _, _, info = env.step(env.action_space.sample())
    assert r < -1.0 and info["pf_spread_gain"] < 0
    # reset() re-baselines: the first step after reset does not pay for the previous episode's spread
    env.reset()
    _, r, _, _, info = env.step(env.action_space.sample())
    assert info["pf_spread_gain"] == pytest.approx(0.0) and r == pytest.approx(-1.0)


def test_set_reward_coeffs_three_and_four_args_via_router(mods):
    frozen, _, _ = mods
    shaping = frozen.PFRewardShapingWrapper(_StubBeliefEnv(), gain_coeff=3.0)
    router = frozen._CurriculumRouter(shaping)
    router.set_reward_coeffs(0.5, 1.0, 20.0)                # legacy 3-arg path: gain untouched
    assert (shaping.distance_coeff, shaping.entropy_coeff, shaping.tag_bonus_coeff, shaping.gain_coeff) \
        == (0.5, 1.0, 20.0, 3.0)
    router.set_reward_coeffs(0.15, 2.0, 50.0, 10.0)         # what CurriculumCallback now sends
    assert shaping.gain_coeff == 10.0


def test_curriculum_callback_interpolates_gain(mods):
    frozen, cgf, _ = mods
    cb = frozen.CurriculumCallback(
        total_timesteps=100,
        schedule=[(0.0, 100.0), (1.0, 1.5)],
        reward_schedule=cgf._parse_reward_schedule("0:1:2:0:0,0.5:0.15:2:50:10,1:0.15:2:50:10"),
        evasion_schedule=[(0.0, 0.0), (1.0, 1.0)],
        verbose=0,
    )
    vals = cb._interpolate_schedule(cb.reward_schedule, 0.25)
    assert len(vals) == 4
    assert vals[3] == pytest.approx(5.0)          # halfway up the 0 -> 10 ramp
    assert vals[2] == pytest.approx(25.0)
