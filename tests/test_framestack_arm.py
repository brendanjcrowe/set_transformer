"""The ``framestack`` arm: the no-belief control (2026-09-22).

Pins the three pieces of ``change_mds/framestack_arm_2026-09-22.md`` (design:
``change_mds/framestack_arm_plan_2026-09-22.md``):

* the wrapper -- the ``"obs"`` key is the last k frames newest last, ``particles`` and
  ``weights`` pass through, the deque pads on reset and clears on the next one, and
  ``with_obs_history`` returns the thunk UNCHANGED at k <= 1 (the identity that makes every
  other arm bit-identical) and a cloudpickle-safe callable above it;
* the registry -- ``framestack`` is analytic, carries ``n_stack`` into the extractor kwargs,
  and ``Encoder.obs_history`` is 1 for every OTHER encoder;
* the doors -- a short real run on hunt/cluster_hunt trains through a 5-stacked env, and a
  standalone eval of that checkpoint WITH NO ``--n_stack`` on the command line rebuilds the
  same 5-stacked env from the zip.
"""
from __future__ import annotations

import argparse
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

import cloudpickle  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from set_transformer.rl import encoders  # noqa: E402
from set_transformer.rl import eval_true_reward as eval_mod  # noqa: E402
from set_transformer.rl import run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.feature_extractors.framestack import (  # noqa: E402
    FrameStackFeaturesExtractor,
)
from set_transformer.rl.wrappers.obs_history import (  # noqa: E402
    ObsHistoryDictWrapper,
    checkpoint_obs_history,
    with_obs_history,
)


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    """Every run in this file is offline (PITFALLS.md 13.36: an exported WANDB_MODE in the
    shell hides a missing fixture, so the file sets it itself)."""
    monkeypatch.setenv("WANDB_MODE", "offline")


# --------------------------------------------------------------------------
# A fake 3-key Dict env, the shape every domain's belief env emits
# --------------------------------------------------------------------------

D = 3
N = 4


class _FakeDictEnv(gym.Env):
    """Emits ``{"obs" [D], "particles" [N, 2], "weights" [N]}``; ``obs`` counts up so a
    stacked observation is readable by eye."""

    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(D,), dtype=np.float32),
            "particles": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N, 2), dtype=np.float32),
            "weights": gym.spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Discrete(2)
        self.t = 0
        self.particles = np.arange(N * 2, dtype=np.float32).reshape(N, 2)
        self.weights = np.full(N, 1.0 / N, dtype=np.float32)

    def _obs(self):
        return {"obs": np.full(D, float(self.t), dtype=np.float32),
                "particles": self.particles, "weights": self.weights}

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, False, False, {}

    def set_marker(self, value):
        """A domain-style setter, to pin that the wrapper forwards it (hunt's collection
        calls ``env.set_n_active`` on the OUTERMOST env)."""
        self.marker = value
        return value


# --------------------------------------------------------------------------
# 1. with_obs_history: the identity, and the picklable wrap
# --------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 0, -3])
def test_with_obs_history_returns_the_same_thunk_object_when_there_is_no_history(k):
    thunk = _FakeDictEnv
    assert with_obs_history(thunk, k) is thunk


def test_with_obs_history_round_trips_through_cloudpickle():
    wrapped = with_obs_history(_FakeDictEnv, 5)
    assert wrapped is not _FakeDictEnv
    revived = cloudpickle.loads(cloudpickle.dumps(wrapped))
    env = revived()
    assert isinstance(env, ObsHistoryDictWrapper) and env.n_stack == 5
    assert env.observation_space["obs"].shape == (5 * D,)


# --------------------------------------------------------------------------
# 2. Wrapper semantics
# --------------------------------------------------------------------------

def test_the_space_stacks_only_the_obs_key():
    env = ObsHistoryDictWrapper(_FakeDictEnv(), 5)
    space = env.observation_space
    assert space["obs"].shape == (5 * D,)
    assert space["obs"].dtype == np.float32
    assert np.all(space["obs"].low == -1.0) and np.all(space["obs"].high == 1.0)
    assert space["particles"].shape == (N, 2) and space["weights"].shape == (N,)


def test_reset_pads_with_the_reset_frame_and_step_slides_newest_last():
    env = ObsHistoryDictWrapper(_FakeDictEnv(), 5)
    obs, _ = env.reset()
    # Every slot holds the reset frame (t = 0), so the width is right from the first step.
    assert obs["obs"].shape == (5 * D,)
    assert np.array_equal(obs["obs"], np.zeros(5 * D, dtype=np.float32))
    for expected in ([0, 0, 0, 0, 1], [0, 0, 0, 1, 2], [0, 0, 1, 2, 3],
                     [0, 1, 2, 3, 4], [1, 2, 3, 4, 5]):
        obs, _r, _te, _tr, _i = env.step(0)
        assert np.array_equal(obs["obs"],
                              np.repeat(np.array(expected, dtype=np.float32), D))
    # Newest last: the final frame is always the current observation.
    assert np.array_equal(obs["obs"][-D:], np.full(D, 5.0, dtype=np.float32))


def test_particles_and_weights_are_the_envs_own_arrays():
    inner = _FakeDictEnv()
    env = ObsHistoryDictWrapper(inner, 5)
    obs, _ = env.reset()
    assert obs["particles"] is inner.particles and obs["weights"] is inner.weights
    obs, *_ = env.step(0)
    assert np.array_equal(obs["particles"], inner.particles)
    assert np.array_equal(obs["weights"], inner.weights)


def test_the_second_reset_clears_the_history():
    env = ObsHistoryDictWrapper(_FakeDictEnv(), 5)
    env.reset()
    for _ in range(4):
        env.step(0)
    obs, _ = env.reset()
    assert np.array_equal(obs["obs"], np.zeros(5 * D, dtype=np.float32))


def test_k_equal_one_is_the_identity_on_the_space_and_the_arrays():
    plain = _FakeDictEnv()
    env = ObsHistoryDictWrapper(_FakeDictEnv(), 1)
    assert env.observation_space["obs"] == plain.observation_space["obs"]
    a, _ = plain.reset()
    b, _ = env.reset()
    assert np.array_equal(a["obs"], b["obs"])
    for _ in range(3):
        a, *_ = plain.step(0)
        b, *_ = env.step(0)
        assert np.array_equal(a["obs"], b["obs"])


def test_a_domain_setter_on_the_outermost_env_still_reaches_the_env():
    """gymnasium 1.2.3's Wrapper has no ``__getattr__``; the collection loop calls domain
    setters on the object it holds (hunt: ``env.set_n_active``)."""
    inner = _FakeDictEnv()
    env = ObsHistoryDictWrapper(inner, 5)
    assert env.set_marker(7) == 7 and inner.marker == 7
    assert env.get_wrapper_attr("set_marker") is not None
    with pytest.raises(AttributeError):
        env.no_such_attribute


# --------------------------------------------------------------------------
# 3. checkpoint_obs_history: best effort, defaulting to 1
# --------------------------------------------------------------------------

def test_checkpoint_obs_history_defaults_to_one(tmp_path):
    assert checkpoint_obs_history(None) == 1
    assert checkpoint_obs_history(str(tmp_path / "missing.zip")) == 1
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip")
    assert checkpoint_obs_history(str(junk)) == 1


# --------------------------------------------------------------------------
# 4. The registry entry
# --------------------------------------------------------------------------

def test_framestack_is_analytic_and_carries_n_stack_into_the_extractor_kwargs():
    enc = encoders.get("framestack")
    assert enc.learned is False and enc.pretrained_dest is None
    assert enc.extractor_class is FrameStackFeaturesExtractor
    args = argparse.Namespace(n_stack=5)
    assert enc.extractor_kwargs(args) == {"n_stack": 5}
    assert enc.obs_history(args) == 5
    assert enc.callbacks({}, {}) == []


def test_obs_history_is_one_for_every_other_encoder():
    args = argparse.Namespace(n_stack=5)
    for name, enc in encoders.ENCODERS.items():
        if name == "framestack":
            continue
        assert enc.obs_history(args) == 1, name


def test_n_stack_above_one_is_refused_on_odd_even(monkeypatch, tmp_path):
    monkeypatch.setattr(run_records, "output_root", lambda *a, **k: tmp_path / "runs")
    with pytest.raises(SystemExit):
        train_mod.main(["--variant", "oe50_short", "--n_stack", "5", "--dry_run"],
                       domain="odd_even", encoder="framestack")
    # k == 1 is accepted: the arm is then the base observation alone.
    assert train_mod.main(["--variant", "oe50_short", "--n_stack", "1", "--dry_run"],
                          domain="odd_even", encoder="framestack") is None


# --------------------------------------------------------------------------
# 5. The extractor
# --------------------------------------------------------------------------

def _space(width: int) -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(width,), dtype=np.float32),
        "particles": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(N, 2), dtype=np.float32),
        "weights": gym.spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32),
    })


def test_the_extractor_refuses_a_width_that_is_not_a_multiple_of_n_stack():
    with pytest.raises(ValueError, match="not a multiple"):
        FrameStackFeaturesExtractor(_space(11), n_stack=5)


def test_the_extractor_passes_the_stacked_obs_through_as_a_fresh_tensor():
    extractor = FrameStackFeaturesExtractor(_space(10), n_stack=5)
    assert extractor.features_dim == 10
    assert extractor._geometry == dict(encoder="framestack", n_stack=5, frame_dim=2)
    obs = {"obs": torch.arange(20, dtype=torch.float32).reshape(2, 10),
           "particles": torch.zeros(2, N, 2), "weights": torch.zeros(2, N)}
    out = extractor(obs)
    assert torch.equal(out, obs["obs"])
    # A fresh tensor, not a view into the rollout buffer's batch.
    assert out is not obs["obs"] and out.data_ptr() != obs["obs"].data_ptr()


# --------------------------------------------------------------------------
# 6. End to end: train through a stacked env, then evaluate with NO --n_stack
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_smoke_train_then_eval_derives_the_frame_count_from_the_zip(monkeypatch, tmp_path,
                                                                   capsys):
    pytest.importorskip("pdomains.hunt", reason="needs the pomdp-domains hunt envs")
    from stable_baselines3 import PPO

    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    model_path = tmp_path / "models" / "framestack_agent.zip"
    model = train_mod.main(
        # The hunt recipe's PPO block, shrunk to a few hundred steps.
        ["--variant", "cluster_hunt", "--n_stack", "5", "--total_timesteps", "256",
         "--n_envs", "1", "--ppo_n_steps", "64", "--batch_size", "64", "--n_epochs", "1",
         "--ent_coef", "0.005", "--separate_extractors", "--no_vec_normalize",
         "--device", "cpu", "--eval_freq", "128", "--n_eval_episodes", "1",
         "--save_freq", "1000", "--output_root", str(tmp_path / "root"),
         "--log_dir", str(tmp_path / "logs") + "/", "--model_save_path", str(model_path)],
        domain="hunt", encoder="framestack")
    capsys.readouterr()

    # Hunt's base observation is the 2-D agent position, so 5 frames are 10 entries; the
    # belief keys are untouched.
    base_dim = 2
    assert model.observation_space["obs"].shape == (5 * base_dim,)
    assert model.observation_space["particles"].shape == (100, 2)
    assert model.policy.features_extractor.n_stack == 5
    assert model_path.exists()

    # The eval command carries NO --n_stack: k comes off the zip (plan section 6c). SB3's
    # check_for_correct_spaces would refuse the load if the env were built one frame wide.
    assert checkpoint_obs_history(str(model_path)) == 5
    episodes = eval_mod.main(
        ["--variant", "cluster_hunt", "--model_path", str(model_path),
         "--n_episodes", "2", "--seed", "0"], domain="hunt")
    out = capsys.readouterr().out
    assert "Frame stacking: the checkpoint was trained on 5 stacked base observations" in out
    assert len(episodes) == 2
    PPO.load(str(model_path), device="cpu")


def test_attribute_forwarding_does_not_recurse_before_env_is_set():
    """``__getattr__`` runs during unpickling and copying, before ``self.env`` exists. Without
    the ``env`` guard, ``hasattr(w, "x")`` on such an object raised ``RecursionError``
    (measured 2026-09-22). Both lookups must fail with a plain AttributeError instead."""
    w = ObsHistoryDictWrapper.__new__(ObsHistoryDictWrapper)
    with pytest.raises(AttributeError):
        _ = w.env
    assert not hasattr(w, "some_public_name")
    assert not hasattr(w, "_some_private_name")
