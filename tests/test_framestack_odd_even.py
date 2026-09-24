"""Odd-Even ``--policy_obs raw`` and the framestack arm on it (2026-09-23).

Pins ``change_mds/framestack_oddeven_raw_obs_2026-09-23.md``:

* ``policy_obs="step_index"`` (the default) is byte-identical to the wrapper as it was before
  the option existed (a verbatim copy of that class is the reference below), at the wrapper
  and through the whole belief env;
* ``policy_obs="raw"`` emits ``[step / cap, o_1 / n, ..., o_m / n]``, float32, with the right
  shape, and the particle filter still reads ``info["observations"]`` (its belief is the same
  array for array as under ``step_index`` on the same seed);
* the framestack refusal on odd_even depends on ``--policy_obs``; the domain's framestack
  padding default is ``zeros``;
* a short real run trains through the raw, zero-padded, 30-stacked env; a standalone eval
  with ONLY ``--policy_obs raw`` rebuilds that env off the zip, and the same eval without the
  flag is refused with a clear error.
"""
from __future__ import annotations

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
pytest.importorskip("pdomains", reason="envs are registered by pdomains")

import gymnasium as gym  # noqa: E402

from set_transformer.rl import eval_true_reward as eval_mod  # noqa: E402
from set_transformer.rl import run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.domains import odd_even as oe  # noqa: E402
from set_transformer.rl.wrappers.obs_history import (  # noqa: E402
    checkpoint_obs_history_spec,
    checkpoint_obs_width,
)

VARIANT = "oe50_short"


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    """PITFALLS.md 13.36: the file sets the variable itself."""
    monkeypatch.setenv("WANDB_MODE", "offline")


class _LegacyStepIndexObservationWrapper(gym.ObservationWrapper):
    """VERBATIM copy of ``StepIndexObservationWrapper`` at set_transformer 5718297 (before
    ``policy_obs`` existed): the reference the default path must match byte for byte."""

    def __init__(self, env: gym.Env, episode_cap: int):
        super().__init__(env)
        if episode_cap <= 0:
            raise ValueError(f"episode_cap must be positive, got {episode_cap}")
        self.episode_cap = int(episode_cap)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step_count = 0

    def observation(self, observation):
        return np.array([self._step_count / self.episode_cap],
                        dtype=np.float32)

    def reset(self, **kwargs):
        self._step_count = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        return (self.observation(obs), reward, terminated, truncated, info)


def _raw_env():
    import pdomains  # noqa: F401,PLC0415
    return gym.make(oe.resolve(VARIANT).env_id)


def _roll(env, seed, steps=12):
    """Reset on ``seed`` and step with a fixed action sequence; return every (obs, info)."""
    out = [env.reset(seed=seed)]
    for t in range(steps):
        obs, _r, term, trunc, info = env.step(t % 7)
        out.append((obs, info))
        if term or trunc:
            break
    return out


# --------------------------------------------------------------------------
# 1. step_index is the old wrapper, byte for byte
# --------------------------------------------------------------------------

@pytest.mark.parametrize("explicit", [False, True])
def test_step_index_is_byte_identical_to_the_wrapper_before_the_option(explicit):
    cap = oe.episode_cap(VARIANT)
    legacy = _LegacyStepIndexObservationWrapper(_raw_env(), episode_cap=cap)
    new = (oe.StepIndexObservationWrapper(_raw_env(), episode_cap=cap, policy_obs="step_index")
           if explicit else oe.StepIndexObservationWrapper(_raw_env(), episode_cap=cap))
    assert new.observation_space == legacy.observation_space
    for (a, ia), (b, ib) in zip(_roll(legacy, 3, 40), _roll(new, 3, 40)):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        assert np.array_equal(ia["observations"], ib["observations"])


def test_the_belief_env_default_and_explicit_step_index_are_the_same_env():
    a = oe.make_odd_even_belief_env(num_particles=50, variant=VARIANT, seed=4)()
    b = oe.make_odd_even_belief_env(num_particles=50, variant=VARIANT, seed=4,
                                    policy_obs="step_index")()
    assert a.observation_space == b.observation_space
    for (oa, _), (ob, _) in zip(_roll(a, 9), _roll(b, 9)):
        for key in ("obs", "particles", "weights"):
            assert oa[key].tobytes() == ob[key].tobytes(), key


def test_an_unknown_policy_obs_is_refused():
    with pytest.raises(ValueError, match="unknown policy_obs"):
        oe.make_odd_even_belief_env(num_particles=50, variant=VARIANT, policy_obs="belief")


# --------------------------------------------------------------------------
# 2. raw: [step / cap, o / n], and the filter is untouched
# --------------------------------------------------------------------------

def test_raw_emits_the_step_index_then_the_observations_over_n():
    cap, n = oe.episode_cap(VARIANT), oe.resolve(VARIANT).n_dist_size
    env = oe.StepIndexObservationWrapper(_raw_env(), episode_cap=cap, policy_obs="raw")
    m = env.env.observation_space.shape[0]
    assert m == 1
    space = env.observation_space
    assert space.shape == (1 + m,) and space.dtype == np.float32
    assert np.all(space.low == 0.0) and np.all(space.high == 1.0)
    for t, (obs, info) in enumerate(_roll(env, 11, 40)):
        assert obs.dtype == np.float32 and obs.shape == (1 + m,)
        assert obs[0] == np.float32(t / cap)
        expected = (np.asarray(info["observations"], dtype=np.float64) / n).astype(np.float32)
        assert np.array_equal(obs[1:], expected)
        assert np.all(obs[1:] > 0.0) and np.all(obs[1:] <= 1.0)
        assert space.contains(obs)


def test_raw_under_the_full_belief_env_and_the_filter_still_reads_info():
    n = oe.resolve(VARIANT).n_dist_size
    step_env = oe.make_odd_even_belief_env(num_particles=n, variant=VARIANT, seed=2)()
    raw_env = oe.make_odd_even_belief_env(num_particles=n, variant=VARIANT, seed=2,
                                          policy_obs="raw")()
    assert raw_env.observation_space["obs"].shape == (2,)
    assert raw_env.observation_space["particles"] == step_env.observation_space["particles"]
    assert raw_env.observation_space["weights"] == step_env.observation_space["weights"]

    moved = False
    for (so, si), (ro, ri) in zip(_roll(step_env, 5), _roll(raw_env, 5)):
        # The belief is the same, array for array: the filter is fed from info["observations"]
        # under both choices, never from the policy's obs key.
        assert so["particles"].tobytes() == ro["particles"].tobytes()
        assert so["weights"].tobytes() == ro["weights"].tobytes()
        assert np.array_equal(si["observations"], ri["observations"])
        # The raw obs key carries exactly what the filter was fed, over n.
        assert ro["obs"][0] == so["obs"][0]
        assert np.array_equal(ro["obs"][1:], (ri["observations"] / n).astype(np.float32))
        moved = moved or not np.allclose(ro["weights"], ro["weights"][0])
    assert moved, "the filter never left the uniform prior: it is starved of evidence"


# --------------------------------------------------------------------------
# 3. The command line: refusal, record, defaults
# --------------------------------------------------------------------------

@pytest.fixture
def _quiet_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    return tmp_path


def _dry(tmp_path, *extra, encoder="framestack"):
    root = tmp_path / "root"
    train_mod.main(["--variant", VARIANT, "--output_root", str(root), "--dry_run",
                    "--run_tag", "t", *extra], domain="odd_even", encoder=encoder)
    [config] = list(root.rglob("run_config.json"))
    return json.loads(config.read_text())


def test_the_framestack_refusal_depends_on_policy_obs(_quiet_runs, capsys):
    with pytest.raises(SystemExit):
        _dry(_quiet_runs, "--n_stack", "5")
    assert "--policy_obs raw" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _dry(_quiet_runs, "--n_stack", "5", "--policy_obs", "step_index")
    config = _dry(_quiet_runs, "--n_stack", "30", "--policy_obs", "raw")
    assert config["policy_obs"] == "raw" and config["n_stack"] == 30
    # The domain's framestack default is zero padding.
    assert config["stack_padding"] == "zeros"


def test_every_odd_even_record_carries_policy_obs_and_it_defaults_to_step_index(_quiet_runs):
    config = _dry(_quiet_runs, encoder="gaussian")
    assert config["policy_obs"] == "step_index"
    assert "stack_padding" not in config and "n_stack" not in config


def test_stack_padding_can_be_overridden_and_defaults_to_reset_frame_elsewhere(_quiet_runs):
    config = _dry(_quiet_runs, "--n_stack", "4", "--policy_obs", "raw",
                  "--stack_padding", "reset_frame")
    assert config["stack_padding"] == "reset_frame"
    parser = train_mod.build_parser(train_mod._domains.get("hunt"),
                                    train_mod._encoders.get("framestack"))
    assert parser.parse_args(["--variant", "cluster_hunt"]).stack_padding == "reset_frame"


# --------------------------------------------------------------------------
# 4. End to end: train (raw, zeros, the whole episode), eval with only --policy_obs raw
# --------------------------------------------------------------------------

def test_train_then_eval_reads_k_and_padding_off_the_zip_and_refuses_the_wrong_policy_obs(
        _quiet_runs, capsys):
    tmp_path = _quiet_runs
    cap = oe.episode_cap(VARIANT)
    model_path = tmp_path / "models" / "framestack_agent.zip"
    model = train_mod.main(
        ["--variant", VARIANT, "--n_stack", str(cap), "--policy_obs", "raw",
         "--total_timesteps", "128", "--n_envs", "1", "--ppo_n_steps", "64",
         "--batch_size", "64", "--n_epochs", "1", "--device", "cpu", "--eval_freq", "100000",
         "--save_freq", "100000", "--output_root", str(tmp_path / "root"),
         "--log_dir", str(tmp_path / "logs") + "/", "--model_save_path", str(model_path)],
        domain="odd_even", encoder="framestack")
    capsys.readouterr()
    assert model.observation_space["obs"].shape == (cap * 2,)
    extractor = model.policy.features_extractor
    assert (extractor.n_stack, extractor.padding) == (cap, "zeros")
    assert checkpoint_obs_history_spec(str(model_path)) == (cap, "zeros")
    assert checkpoint_obs_width(str(model_path)) == cap * 2
    vecnorm = str(model_path.parent / "vecnormalize.pkl")

    episodes = eval_mod.main(
        ["--variant", VARIANT, "--model_path", str(model_path), "--vecnormalize_path", vecnorm,
         "--policy_obs", "raw", "--n_episodes", "2", "--seed", "0", "--no_summary"],
        domain="odd_even")
    out = capsys.readouterr().out
    assert (f"Frame stacking: the checkpoint was trained on {cap} stacked base observations "
            "(padding: zeros)") in out
    assert f"obs width {cap * 2} = {cap} frames x 2" in out
    assert len(episodes) == 2

    # Without --policy_obs raw the env's frames are 1 wide: refused, with the flag named.
    with pytest.raises(SystemExit):
        eval_mod.main(["--variant", VARIANT, "--model_path", str(model_path),
                       "--vecnormalize_path", vecnorm, "--n_episodes", "2", "--seed", "0",
                       "--no_summary"], domain="odd_even")
    err = capsys.readouterr().err
    assert f"{cap * 2} wide ({cap} frame(s) of 2)" in err
    assert "--policy_obs raw" in err
