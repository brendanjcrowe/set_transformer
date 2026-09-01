"""Tests for the pre-flight env probe.

The probe's whole value is catching a reward spec where information does not pay, so the
key test drives it against the *stock* Car-Flag reward (the known-broken case) and asserts
it flags it — and against the corrected reward and asserts it passes.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "benchmark"))

import probe_env  # noqa: E402

from set_transformer.rl.benchmark.registry import get_env_spec  # noqa: E402


def _run_all(env_spec, n_episodes=60, seed=0, probe_name=None):
    # Follow the env under test; a spec built with `replace()` keeps the same name.
    probe = probe_env.PROBE_REGISTRY[probe_name or env_spec.name]
    return {
        name: probe_env.run_policy(env_spec, probe, policy, n_episodes, seed)
        for name, policy in probe.policies.items()
    }


def test_probe_registered_for_car_flag():
    probe = probe_env.PROBE_REGISTRY["car_flag"]
    assert {"oracle", "informed", "gambler", "staller"} <= set(probe.policies)
    assert probe.latent_fn is not None


def test_probe_ranks_policies_correctly_on_corrected_reward():
    """With the corrected reward: oracle > informed > gambler > staller."""
    results = _run_all(get_env_spec("car_flag"))
    mean = {k: v[0].mean() for k, v in results.items()}
    assert mean["oracle"] > mean["informed"] > mean["gambler"] > mean["staller"]

    # The informed policy must actually solve the task; the gambler is a coin flip.
    assert results["informed"][2].mean() == pytest.approx(1.0)
    assert 0.2 < results["gambler"][2].mean() < 0.8
    # Information gathering shows up as a longer episode (the priest detour).
    assert results["informed"][1].mean() > results["gambler"][1].mean()


def test_probe_detects_the_broken_stock_reward():
    """Regression guard: on stock Car-Flag, informed must score WORSE than gambler.

    This is the failure the probe exists to catch — the stock -1/step reward makes the
    priest detour cost far more than the information is worth, so RL correctly learns to
    ignore the belief and the env cannot discriminate encoders.
    """
    def make_stock_env(seed=0, **kwargs):
        env = gym.make("pdomains-car-flag-v0")  # deliberately NOT CarFlagRewardWrapper
        env.reset(seed=seed)
        return env

    # success_fn is tied to the corrected reward, so drop it for this stock-reward spec.
    stock_spec = replace(get_env_spec("car_flag"), make_base_env=make_stock_env, success_fn=None)
    results = _run_all(stock_spec)
    assert results["informed"][0].mean() < results["gambler"][0].mean()


def test_probe_policies_emit_valid_actions():
    env_spec = get_env_spec("car_flag")
    env = env_spec.make_base_env(seed=0)
    obs, _ = env.reset(seed=0)
    ctx = {"latent": 1.0}
    for name, policy in probe_env.PROBE_REGISTRY["car_flag"].policies.items():
        action = policy(obs, ctx)
        assert env.action_space.contains(np.asarray(action, dtype=np.float32)), name
    env.close()


# --- Odd-Even probe (2026-08-24) ----------------------------------------------------

def test_odd_even_probe_ranks_policies_correctly():
    """oracle > informed > gambler > staller under the parity-gated reward.

    informed plays the posterior MODE, gambler rounds the posterior MEAN. On a comb the
    mean lands between the teeth -- on a state of the opposite parity, which the gated
    reward floors -- so this ordering is the whole reason the env can discriminate
    belief encoders at all.
    """
    results = _run_all(get_env_spec("odd_even"), n_episodes=60)
    mean = {k: v[0].mean() for k, v in results.items()}
    assert mean["oracle"] > mean["informed"] > mean["gambler"] > mean["staller"]
    # The gap must be large, not marginal: tracking the mean eats the wrong-parity floor.
    assert mean["informed"] - mean["gambler"] > 100


def test_odd_even_stock_reward_makes_the_mean_sufficient():
    """Regression guard: under the STOCK -squared_error reward the belief mean is a
    sufficient statistic, so gambler must match informed and the env cannot discriminate.

    This is the defect the parity gate exists to fix, and the reason Odd-Even could not
    have been swept as shipped.
    """
    from set_transformer.rl.benchmark import envs

    def make_stock_env(seed=0, **kwargs):
        return envs.make_odd_even_base_env(seed=seed, parity_gated_reward=False, **kwargs)

    stock = replace(get_env_spec("odd_even"), make_base_env=make_stock_env,
                    success_fn=None)
    results = _run_all(stock, n_episodes=60, probe_name="odd_even")
    informed, gambler = results["informed"][0].mean(), results["gambler"][0].mean()
    # Sharper than "they tie": round(mean) IS the Bayes action for squared error, so the
    # mean-tracker must be at least as good as the mode-tracker. Representing the comb
    # does not merely fail to pay under the stock reward -- it actively costs.
    assert gambler >= informed, (informed, gambler)


def test_odd_even_probe_policies_emit_valid_actions():
    env_spec = get_env_spec("odd_even")
    env = env_spec.make_base_env(seed=0)
    obs, _ = env.reset(seed=0)
    ctx = {"latent": int(env.unwrapped.true_state)}
    for name, policy in probe_env.PROBE_REGISTRY["odd_even"].policies.items():
        assert env.action_space.contains(policy(obs, ctx)), name
    env.close()
