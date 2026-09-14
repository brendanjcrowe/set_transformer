"""The ``car_flag`` domain on the shared harness (plan section 10, batch 10.2, 2026-09-14).

Pins: the registry (two reward variants on one env id, the binary filter, cap 160 from the
registration, ``shaped`` the default); the THREE reachable beliefs through the full wrapper
stack (50/50 at reset, collapsed to the revealed side inside the priest zone, still collapsed
after leaving it); the two rewards on a drive-right episode and the outcome info on both
variants; the timeout outcome; the evaluation report on hand-built episodes (hell is NOT a
success even though it ends early; the generic rule would count it); ``--dry_run`` from a
foreign cwd writes under the output root only, with the tanh-3 CGF defaults; the Domain record
is complete; a 128-step smoke train.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
pdomains = pytest.importorskip("pdomains", reason="envs are registered by pdomains")

import gymnasium as gym  # noqa: E402

from set_transformer.rl import domains  # noqa: E402
from set_transformer.rl import run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.benchmark.envs import (  # noqa: E402
    CAR_FLAG_HEAVEN_REWARD,
    CAR_FLAG_HELL_REWARD,
    CAR_FLAG_STEP_PENALTY,
)
from set_transformer.rl.domains import car_flag  # noqa: E402
from set_transformer.rl.domains.base import Domain  # noqa: E402
from set_transformer.rl.eval_true_reward import Episode  # noqa: E402
from set_transformer.rl.particle_filters.car_flag import CarFlagParticleFilter  # noqa: E402

CAR_FLAG = car_flag.CAR_FLAG


def _env(variant: str, *, seed: int = 0, num_particles: int = 100, training: bool = True):
    return CAR_FLAG.make_env(variant, num_particles=num_particles,
                             particle_filter_class=CAR_FLAG.resolve(variant).particle_filter,
                             seed=seed, rank=0, monitor_dir=None, training=training, options={})()


def _drive(env, action: float, until=None, max_steps: int = 200):
    """Step with a constant action until ``until(obs, info)`` is true or the episode ends.
    Returns (obs, reward, terminated, truncated, info, n_steps)."""
    for k in range(1, max_steps + 1):
        obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        if terminated or truncated or (until is not None and until(obs, info)):
            return obs, reward, terminated, truncated, info, k
    raise AssertionError("episode did not end")


# --------------------------------------------------------------------------
# Registry and record
# --------------------------------------------------------------------------

def test_registry_two_reward_variants_on_one_env_and_the_cap_from_the_registration():
    assert sorted(CAR_FLAG.variants) == ["shaped", "stock"]
    assert CAR_FLAG.default_variant == "shaped"
    for name in CAR_FLAG.variants:
        v = CAR_FLAG.resolve(name)
        assert v.env_id == "pdomains-car-flag-v0" and v.particle_filter is CarFlagParticleFilter
        assert v.reward == name
        assert CAR_FLAG.episode_cap(name) == 160 == gym.spec(v.env_id).max_episode_steps
        assert CAR_FLAG.run_subdir("st", name) == f"car_flag_st_{name}"
    with pytest.raises(ValueError, match="shaped"):
        CAR_FLAG.resolve("car_flag_shaped")
    assert domains.get("car_flag") is CAR_FLAG
    assert "car_flag" in domains.DOMAIN_NAMES
    assert domains.domain_of_variant("shaped") is CAR_FLAG


def test_domain_record_is_complete_and_declares_no_collector():
    assert isinstance(CAR_FLAG, Domain)
    assert CAR_FLAG.name == "car_flag" and CAR_FLAG.particle_dim == 1
    assert CAR_FLAG.default_num_particles("shaped") == 100
    assert CAR_FLAG.default_arena_scale("shaped") == 1.0
    assert CAR_FLAG.default_total_timesteps == 1_000_000
    assert CAR_FLAG.collection is None
    assert CAR_FLAG.pretraining.objectives == {} and CAR_FLAG.pretraining.default_objective is None
    assert CAR_FLAG.evaluation.reseed_per_episode is True
    assert CAR_FLAG.evaluation.default_n_episodes == 300
    args = argparse.Namespace(variant="shaped")
    assert CAR_FLAG.schedules(args) == ()
    assert CAR_FLAG.particle_filter(args) is CarFlagParticleFilter
    extras = CAR_FLAG.run_config_extras(args)
    assert extras["episode_cap"] == 160 and extras["reward"] == "shaped"
    assert extras["reward_constants"] == dict(step=-0.01, heaven=1.0, hell=-1.0)
    assert "reward_constants" not in CAR_FLAG.run_config_extras(argparse.Namespace(variant="stock"))
    cgf = CAR_FLAG.encoder_defaults["cgf"]
    assert cgf["t_param"] == "tanh" and cgf["t_bound"] == 3.0 and cgf["t_init_mode"] == "spread_1d"
    assert cgf["t_init_max_default"](argparse.Namespace(t_param="tanh", t_bound=3.0)) == pytest.approx(2.4)
    assert cgf["t_init_max_default"](argparse.Namespace(t_param="clamp", t_bound=None)) is None


# --------------------------------------------------------------------------
# The three reachable beliefs, through the full stack
# --------------------------------------------------------------------------

def test_the_three_reachable_beliefs_through_the_wrapper_stack():
    env = _env("shaped", seed=3)
    obs, info = env.reset(seed=3)
    assert set(obs) == {"obs", "particles", "weights"}
    assert obs["obs"].shape == (3,) and obs["particles"].shape == (100, 1) and obs["weights"].shape == (100,)
    # 1: the 50/50 prior -- half the particles at +1, half at -1, uniform weights
    assert set(np.unique(obs["particles"]).tolist()) == {-1.0, 1.0}
    assert obs["particles"].sum() == 0
    np.testing.assert_allclose(obs["weights"], 1 / 100)
    assert obs["obs"][2] == 0.0                       # outside the priest zone
    heaven = float(env.unwrapped.heaven_position)
    # 2: drive right into the priest zone: the reading fires and the belief collapses
    obs, _, term, trunc, _, k = _drive(env, 1.0, until=lambda o, i: o["obs"][2] != 0.0)
    assert not term and not trunc and k > 1
    assert obs["obs"][2] == heaven and 0.3 <= obs["obs"][0] <= 0.7
    assert np.all(obs["particles"] == heaven)
    np.testing.assert_allclose(obs["weights"], 1 / 100)
    # 3: drive on; the reading goes silent again but the belief stays collapsed
    obs, _, term, trunc, info, _ = _drive(env, 1.0 if heaven > 0 else -1.0,
                                          until=lambda o, i: o["obs"][2] == 0.0)
    assert obs["obs"][2] == 0.0 or term
    assert np.all(obs["particles"] == heaven)
    env.close()


@pytest.mark.parametrize("variant", ["shaped", "stock"])
def test_rewards_and_outcome_info_on_a_drive_right_episode(variant):
    seen = set()
    for seed in range(8):                              # spans both heaven sides
        env = _env(variant, seed=seed)
        env.reset(seed=seed)
        heaven = float(env.unwrapped.heaven_position)
        rewards, infos = [], []
        while True:
            _, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
            rewards.append(r)
            infos.append(info)
            if term or trunc:
                break
        assert term and not trunc
        reached = heaven > 0
        seen.add(reached)
        # the outcome wrapper speaks on both variants
        assert infos[-1]["reached_heaven"] is reached and infos[-1]["is_success"] is reached
        assert infos[-1]["outcome"] == ("heaven" if reached else "hell")
        assert all("outcome" not in i for i in infos[:-1])
        if variant == "shaped":
            assert all(r == CAR_FLAG_STEP_PENALTY for r in rewards[:-1])
            assert rewards[-1] == (CAR_FLAG_HEAVEN_REWARD if reached else CAR_FLAG_HELL_REWARD)
        else:
            assert all(r == -1 for r in rewards[:-1])
            # stock: heaven on the right pays 0, on the left +1; hell -5
            assert rewards[-1] == (0.0 if reached else -5.0)
        env.close()
    assert seen == {True, False}


def test_a_timeout_is_an_outcome_too():
    env = _env("shaped", seed=1)
    env.reset(seed=1)
    _, _, term, trunc, info, k = _drive(env, 0.0)   # never leaves the start region
    assert trunc and not term and k == 160
    assert info["outcome"] == "timeout" and info["is_success"] is False and "reached_heaven" not in info
    env.close()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _episode(n_steps: int, final: dict, terminal_reward: float | None) -> Episode:
    ep = Episode()
    for i in range(n_steps):
        last = i == n_steps - 1
        ep.rewards.append(terminal_reward if (last and terminal_reward is not None) else CAR_FLAG_STEP_PENALTY)
        ep.infos.append(dict(final) if last else {})
    return ep


def test_report_counts_heaven_hell_timeout_and_hell_is_not_a_success(capsys):
    cap = 160
    episodes = [
        _episode(70, dict(outcome="heaven", reached_heaven=True, is_success=True), 1.0),
        _episode(40, dict(outcome="hell", reached_heaven=False, is_success=False), -1.0),
        _episode(160, dict(outcome="timeout", is_success=False), None),
        _episode(90, dict(outcome="heaven", reached_heaven=True, is_success=True), 1.0),
        _episode(12, {}, None),                    # cut by --max_steps: no outcome -> timeout
    ]
    # without the outcome wrapper's key the generic rule would call the hell episode a success
    assert Episode.is_success(_episode(40, {}, -1.0), cap) is True
    assert episodes[1].is_success(cap) is False
    args = argparse.Namespace(deterministic=True)
    report = car_flag._eval_report(episodes, None, args, CAR_FLAG.resolve("shaped"), cap)
    assert report["reward"] == "shaped"
    assert report["successes"] == 2 and report["success_rate"] == pytest.approx(0.4)
    assert report["outcome_counts"] == {"heaven": 2, "hell": 1, "timeout": 2}
    assert report["outcome_hell"] == pytest.approx(0.2)
    assert report["heaven_mean_length"] == pytest.approx(80.0)
    assert report["mean_length"] == pytest.approx((70 + 40 + 160 + 90 + 12) / 5)
    assert [e["outcome"] for e in report["episodes"]] == ["heaven", "hell", "timeout", "heaven", "timeout"]
    assert [e["success"] for e in report["episodes"]] == [True, False, False, True, False]
    out = capsys.readouterr().out
    assert "Success (heaven): 2/5 (40.0%)" in out
    assert "heaven 2, hell 1, timeout 2" in out
    json.dumps(report)   # what the eval script writes


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------

def test_dry_run_from_a_foreign_cwd_writes_under_the_root_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    root = tmp_path / "root"
    train_mod.main(["--domain", "car_flag", "--encoder", "cgf", "--feature_mode", "K_grad",
                    "--output_root", str(root), "--dry_run"])
    [record] = list(root.glob("car_flag/shaped/rl/cgf/*_seed0/run_config.json"))
    config = json.loads(record.read_text())
    assert config["env_id"] == "pdomains-car-flag-v0"
    assert config["particle_filter_class"] == "CarFlagParticleFilter"
    assert config["num_particles"] == 100 and config["arena_scale"] == 1.0
    assert config["t_param"] == "tanh" and config["t_bound"] == 3.0
    assert config["t_init_max"] == pytest.approx(2.4) and config["t_init_mode"] == "spread_1d"
    assert config["feature_norm"] == "none"
    assert config["episode_cap"] == 160 and config["reward"] == "shaped"
    assert config["reward_constants"] == dict(step=-0.01, heaven=1.0, hell=-1.0)
    assert config["total_timesteps"] == 1_000_000
    assert list(elsewhere.iterdir()) == []
    assert "Car-Flag variant 'shaped': reward shaped" in capsys.readouterr().out
    # the stock variant records its reward too
    train_mod.main(["--domain", "car_flag", "--encoder", "st", "--variant", "stock",
                    "--output_root", str(root), "--dry_run"])
    [record] = list(root.glob("car_flag/stock/rl/st/*_seed0/run_config.json"))
    assert json.loads(record.read_text())["reward"] == "stock"


@pytest.mark.slow
def test_smoke_train_shaped_st_saves_and_reloads(monkeypatch, tmp_path, capsys):
    from stable_baselines3 import PPO
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    model_path = tmp_path / "models" / "st_agent.zip"
    model = train_mod.main(
        ["--variant", "shaped", "--total_timesteps", "128", "--n_envs", "2",
         "--ppo_n_steps", "32", "--batch_size", "32", "--n_epochs", "1", "--device", "cpu",
         "--eval_freq", "64", "--n_eval_episodes", "1", "--save_freq", "64",
         "--output_root", str(tmp_path / "root"),
         "--log_dir", str(tmp_path / "logs") + "/", "--model_save_path", str(model_path)],
        domain="car_flag", encoder="st")
    out = capsys.readouterr().out
    assert "ST geometry: num_inds=16 dim_hidden=64 num_post_sab=2 (default)" in out
    assert model_path.exists()
    assert run_records.read_run_status(str(model_path))["status"] == "completed"
    assert (tmp_path / "models" / "checkpoints" / "car_flag_st_64_steps.zip").exists()
    assert list((tmp_path / "root").glob("car_flag/shaped/rl/st/*/run_config.json"))
    PPO.load(str(model_path), device="cpu")
