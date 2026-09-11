"""Forking an Ant-Tag ST run from a mid-run checkpoint (4_train_rl_st.py --resume_from, 2026-09-08).

Pins the three facts the second-half reward ablation relies on:

1. SB3 ``learn(total_timesteps=remaining, reset_num_timesteps=False)`` keeps the
   loaded step counter, so the progress-based LR schedule resumes at the
   checkpoint's progress instead of restarting at the initial rate.
2. ``CurriculumCallback`` applies the schedule at the resumed progress when
   training starts (``_on_training_start``), so the first env steps of a fork
   are rewarded under the second-half coefficients, not the progress-0 ones.
3. ``_default_resume_vecnormalize`` finds the VecNormalize snapshot that
   ``CheckpointCallback(save_vecnormalize=True)`` writes beside a checkpoint.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

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
    return importlib.import_module("4_train_rl_frozen"), importlib.import_module("4_train_rl_st")


class _TinyEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1, 1, (3,), np.float32)
        self.action_space = gym.spaces.Box(-1, 1, (1,), np.float32)
        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(3, np.float32), {}

    def step(self, action):
        self.t += 1
        return np.zeros(3, np.float32), 0.0, self.t >= 5, False, {}


def test_resume_continues_step_counter_and_lr_schedule(tmp_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([_TinyEnv])
    lr = lambda pr: 1e-3 * pr  # noqa: E731 - the anneal 4_train_rl_st uses
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, learning_rate=lr, seed=0, verbose=0)
    model.learn(total_timesteps=32)
    assert model.num_timesteps == 32
    model.save(tmp_path / "ckpt.zip")

    resumed = PPO.load(tmp_path / "ckpt.zip", env=DummyVecEnv([_TinyEnv]),
                       custom_objects={"learning_rate": lr, "lr_schedule": lr})
    assert resumed.num_timesteps == 32
    # SB3 refreshes _current_progress_remaining AFTER on_rollout_end and right
    # before train(), so record it inside the update (train() also writes the
    # scheduled LR into the optimizer first thing).
    seen = []
    orig_train = resumed.train

    def train_and_record():
        orig_train()
        seen.append((resumed._current_progress_remaining, resumed.policy.optimizer.param_groups[0]["lr"]))
    resumed.train = train_and_record
    resumed.learn(total_timesteps=64 - resumed.num_timesteps, reset_num_timesteps=False)
    assert resumed.num_timesteps == 64
    # First update after resume happens at 40 of a 64-step horizon: progress_remaining 0.375,
    # not 0.875 (a restarted counter) and not 0.75 (remaining-only horizon).
    pr, lr_seen = seen[0]
    assert pr == pytest.approx(1 - 40 / 64)
    assert lr_seen == pytest.approx(1e-3 * pr)
    assert seen[-1][0] == pytest.approx(0.0)


def test_curriculum_applies_resumed_progress_on_training_start(mods):
    frozen, _ = mods
    cb = frozen.CurriculumCallback(
        total_timesteps=100,
        schedule=[(0.0, 100.0), (0.4, 1.5), (1.0, 1.5)],
        reward_schedule=[(0.0, 1.0, 2.0, 0.0, 0.0), (0.5, 0.0, 2.0, 50.0, 0.0), (1.0, 0.0, 2.0, 50.0, 0.0)],
        evasion_schedule=[(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)],
        verbose=0,
    )
    calls = []

    class _Venv:  # SubprocVecEnv stand-in: records env_method calls
        num_envs = 1

        def env_method(self, name, *args):
            calls.append((name, args))

    # What BaseCallback.init_callback / on_training_start set before _on_training_start runs.
    venv = _Venv()
    cb.model = SimpleNamespace(num_timesteps=50, get_env=lambda: venv)
    try:
        cb.training_env = venv
    except AttributeError:  # property on newer SB3: served by model.get_env()
        pass
    cb.num_timesteps = 50
    cb._on_training_start()
    by_name = dict(calls)
    assert by_name["set_curriculum_radius"] == (1.5,)
    assert by_name["set_reward_coeffs"] == (0.0, 2.0, 50.0, 0.0)
    assert by_name["set_evasion_scale"] == (1.0,)


@pytest.mark.parametrize("zip_name,expected", [
    ("ant_tag_st_3000000_steps.zip", "ant_tag_st_vecnormalize_3000000_steps.pkl"),
    ("st_agent.zip", "vecnormalize.pkl"),
])
def test_default_resume_vecnormalize(mods, zip_name, expected):
    _, st = mods
    assert st._default_resume_vecnormalize(f"/x/models/{zip_name}") == f"/x/models/{expected}"


def test_default_resume_vecnormalize_rejects_unknown(mods):
    _, st = mods
    with pytest.raises(ValueError):
        st._default_resume_vecnormalize("/x/models/best_model.zip")
