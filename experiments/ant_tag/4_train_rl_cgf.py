"""
RL training with a weighted CGF belief encoder for Ant-Tag.

This keeps the AntTag particle filter, curriculum, and reward-shaping pipeline
from 4_train_rl_frozen.py, but replaces the Set Transformer belief encoder with
a trainable CGF feature extractor:

    CGF_j = log(sum_i w_i * exp(<t_j, x_i>))

where x_i is the target-position particle and w_i is its particle-filter weight.
"""

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers pdomains-ant-tag-v0
import variants  # noqa: E402 - env/filter/subdir registry
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter
from set_transformer.rl.wrappers.particle_filter import _call_pf_interaction_mapper

# The belief-encoder pieces below now live in the shared package so a second
# domain can use them without importing an Ant-Tag script. They are re-exported
# here, unchanged and as the SAME objects, because SB3 pickles a policy's
# features-extractor CLASS into the saved zip by module path: loading an
# existing checkpoint runs
# getattr(import_module("4_train_rl_cgf"), "WeightedCGFFeaturesExtractor").
# 4_train_rl_st.py and 4_train_rl_gaussian.py also read all three off this
# module by name.
from set_transformer.rl.feature_extractors.cgf import (  # noqa: E402
    TNormLoggingCallback,
    WeightedCGFFeaturesExtractor,
)
from set_transformer.rl.wrappers.particle_filter import (  # noqa: E402
    PFDictWithWeightsObservationWrapper,
)


# Reuse the existing AntTag curriculum/PF utilities without touching ST code.
_train_rl_frozen = importlib.import_module("4_train_rl_frozen")
CurriculumCallback = _train_rl_frozen.CurriculumCallback
CurriculumVisibilityWrapper = _train_rl_frozen.CurriculumVisibilityWrapper
PFRewardShapingWrapper = _train_rl_frozen.PFRewardShapingWrapper
_CurriculumRouter = _train_rl_frozen._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_frozen.ant_tag_pf_interaction_mapper
get_ant_tag_pf_kwargs = _train_rl_frozen.get_ant_tag_pf_kwargs


def get_ant_tag_arena_scale(env_id: str = "pdomains-ant-tag-v0") -> float:
    """Derive the CGF particle-normalization scale from the live AntTag arena.

    Mirrors get_ant_tag_pf_kwargs's arena_limits derivation, so the CGF
    extractor's particle normalization always matches the PF's actual
    arena_limits instead of relying on a hardcoded default that would go
    stale if the env's arena size ever changes.
    """
    env = gym.make(env_id, rendering=False)
    try:
        unwrapped = env.unwrapped
        cage_max_x = float(unwrapped.cage_max_x)
        cage_max_y = float(unwrapped.cage_max_y)
        if not np.isclose(cage_max_x, cage_max_y):
            raise ValueError(
                "WeightedCGFFeaturesExtractor currently assumes a square arena, "
                f"but got cage_max_x={cage_max_x}, cage_max_y={cage_max_y}"
            )
        return cage_max_x
    finally:
        env.close()


def get_env_visible_radius(env_id: str = "pdomains-ant-tag-v0") -> float:
    """Derive the evaluation visibility radius from the live env.

    Mirror of get_ant_tag_arena_scale. Replaces the hardcoded 3.0 that used
    to be baked into every eval-env construction: every legacy env reports
    visible_radius == 3.0, so this is behavior-identical for them, while
    arena-scaled variants (CounterweightedDenAntTagEnv, 1.8) are evaluated at
    THEIR real POMDP difficulty instead of a stale constant.
    """
    env = gym.make(env_id, rendering=False)
    try:
        return float(env.unwrapped.visible_radius)
    finally:
        env.close()


def _make_vec_normalize(vec_env, training: bool, norm_reward: bool):
    """Normalize only base obs so PF weights remain valid probabilities."""
    try:
        return VecNormalize(
            vec_env,
            training=training,
            norm_obs=True,
            norm_reward=norm_reward,
            norm_obs_keys=["obs"],
        )
    except TypeError:
        print(
            "Warning: this SB3 VecNormalize lacks norm_obs_keys; disabling "
            "obs normalization to avoid corrupting PF weights."
        )
        return VecNormalize(
            vec_env,
            training=training,
            norm_obs=False,
            norm_reward=norm_reward,
        )


def make_ant_tag_cgf_env(
    num_particles: int,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    distance_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    tag_bonus_coeff: float = 0.0,
    initial_visibility_radius: float = 100.0,
    obs_mask_indices: list[int] | None = None,
    apply_reward_shaping: bool = True,
    env_id: str = "pdomains-ant-tag-v0",
    particle_filter_class: type = AntTagParticleFilter,
    target_speed_scale: float | None = None,
):
    """Return a callable that creates a weighted-CGF AntTag env.

    apply_reward_shaping=False skips PFRewardShapingWrapper entirely, so
    Monitor sees the env's true sparse reward (-1/step, 0-and-terminate on
    tag). Use this for the eval env: CurriculumCallback only ever updates
    reward coefficients on the training env, so a shaped eval env would
    report reward numbers stuck at their initial (dense) coefficients for
    the entire run, making EvalCallback's "best_model" selection
    meaningless. Eval envs already fix visibility at the real POMDP radius
    regardless of training progress; this applies that same principle to
    the reward too.

    env_id / particle_filter_class default to the original dumb-target
    AntTag env + its matching PF; pass "pdomains-ant-tag-smart-v0" +
    SmartAntTagParticleFilter to train on the smart-target variant instead,
    so belief propagation matches that env's true motion model.

    target_speed_scale=None (default) leaves the env at its own default
    (0.0 for SmartAntTagEnv: the target flees more OFTEN when cornered but
    never faster). Pass a float to override; only SmartAntTagEnv has this
    knob, so passing it with the base AntTag env is a hard error rather
    than a silently ignored no-op. The PF is told the live value each step
    by ant_tag_pf_interaction_mapper, so belief propagation always matches
    whatever the env is actually doing.
    """

    def _init():
        env_make_kwargs = {"rendering": False}
        if target_speed_scale is not None:
            env_make_kwargs["target_speed_scale"] = target_speed_scale
        try:
            env = gym.make(env_id, **env_make_kwargs)
        except TypeError as exc:
            if target_speed_scale is None or "target_speed_scale" not in str(exc):
                raise
            raise ValueError(
                f"--target_speed_scale was given ({target_speed_scale}) but env_id="
                f"{env_id!r} does not support it. Only SmartAntTagEnv "
                "('pdomains-ant-tag-smart-v0') has a cornered-speed knob; the base "
                "AntTag target always moves at a constant target_step. Either drop "
                "the flag or train on the smart env."
            ) from exc
        if target_speed_scale is not None:
            # Fail loudly if it didn't land: a silently-dropped kwarg here
            # would train against a different target speed than requested,
            # and the mapper would faithfully feed that wrong value to the PF.
            actual = getattr(env.unwrapped, "target_speed_scale", None)
            if actual is None or not np.isclose(actual, target_speed_scale):
                raise ValueError(
                    f"target_speed_scale={target_speed_scale} did not take effect on "
                    f"env_id={env_id!r} (env reports {actual!r}). Only "
                    "SmartAntTagEnv ('pdomains-ant-tag-smart-v0') supports this knob."
                )
        env.reset(seed=seed + rank)
        particle_filter_kwargs = get_ant_tag_pf_kwargs(env)

        env = CurriculumVisibilityWrapper(
            env,
            initial_visibility_radius=initial_visibility_radius,
        )
        env = PFDictWithWeightsObservationWrapper(
            env=env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs=particle_filter_kwargs,
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=obs_mask_indices,
            particle_filter_seed=seed + rank,
        )
        if apply_reward_shaping:
            env = PFRewardShapingWrapper(
                env,
                distance_coeff=distance_coeff,
                entropy_coeff=entropy_coeff,
                tag_bonus_coeff=tag_bonus_coeff,
            )

        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        else:
            env = Monitor(env)
        env = _CurriculumRouter(env)
        return env

    return _init


def _make_vec_env_from_fns(env_fns, n_envs: int):
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def _default_run_dir(seed: int, run_subdir: str = "ant_tag_cgf",
                      run_tag: str | None = None) -> str:
    """runs/<run_subdir>/<timestamp>_seed<seed>[_<run_tag>]/, so parallel runs
    with different seeds (and different env variants, via run_subdir) land in
    distinct, sortable, self-describing folders instead of overwriting a
    fixed sb3_ant_tag_cgf_logs/ path.

    run_tag is a free-form human label (e.g. "6M_vis0.2-0.5") for eyeballing
    `ls runs/<run_subdir>/` without opening any files. It is NOT the source
    of truth for what a run actually used — that's runs/<run_subdir>/<...>/
    run_config.json, written alongside it with every CLI arg."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{re.sub(r'[^A-Za-z0-9._-]', '_', run_tag)}" if run_tag else ""
    return os.path.join("runs", run_subdir, f"{timestamp}_seed{seed}{suffix}")


def _write_run_config(run_dir: str, **config) -> None:
    """Dump every CLI arg for this run to run_dir/run_config.json — the
    unambiguous source of truth for what a run used (total_timesteps,
    curriculum/reward/evasion schedules, env_id, etc.), since the run_tag in
    the directory name is just a human-readable hint, not a full record."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str, sort_keys=True)
    print(f"Run config saved to {path}")


def _git_provenance() -> dict:
    """Record WHICH CODE produced this run, for run_config.json.

    run_config.json pins every hyperparameter but not the source that read
    them, and that gap has already bitten this project: the
    ant_tag_cgf_cdens_terminal runs of 2026-08-26 finished at 17:30, and
    4_train_rl_cgf.py gained deterministic per-episode PF seeding at 18:39.
    Their run_config.json is byte-identical either side of that change, so
    nothing on disk says which behavior those checkpoints were trained with.

    HEAD alone would not close it — both submodules are routinely dirty — so
    the SHA-256 of `git diff HEAD` gives an uncommitted working tree a stable
    identity. Equal (head, diff_sha256) means the same code; a different
    diff_sha256 means something moved between two runs, even when both say
    "dirty". The porcelain status lines are kept as a human-readable hint of
    WHICH files were dirty.

    Never raises: a missing git, a detached worktree or a stripped checkout
    records an "error" string rather than killing a multi-hour training run.
    """
    repos = {
        "set_transformer": Path(__file__).resolve().parents[2],
        "pomdp-domains": Path(__file__).resolve().parents[3] / "pomdp-domains",
    }

    def _git(repo: Path, *args: str) -> str:
        return subprocess.run(
            ("git", "-C", str(repo)) + args,
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout

    provenance = {}
    for name, repo in repos.items():
        try:
            head = _git(repo, "rev-parse", "HEAD").strip()
            status = [line for line in
                      _git(repo, "status", "--porcelain").splitlines() if line]
            diff = _git(repo, "diff", "HEAD")
            provenance[name] = {
                "path": str(repo),
                "head": head,
                "dirty": bool(status),
                # Tracked-file modifications only; untracked content is not in
                # `git diff HEAD`, which is why the status lines are kept too.
                "diff_sha256": (hashlib.sha256(diff.encode()).hexdigest()
                                if diff else None),
                "status": status,
            }
        except Exception as exc:  # noqa: BLE001 - provenance must never abort a run
            provenance[name] = {"path": str(repo), "error": f"{type(exc).__name__}: {exc}"}
    return provenance


def train_ant_tag_cgf(
    algorithm: str = "PPO",
    total_timesteps: int = 3_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    ppo_n_steps: int = 2048,
    num_particles: int = 100,
    num_cgf_features: int = 64,
    arena_scale: float | None = None,
    t_init_mode: str = "linspace_all_dims",
    t_init_scale: float = 0.1,
    t_clamp: float = 2.0,
    exp_arg_clamp: float = 20.0,
    t_frozen: bool = False,
    device: str = "cuda:1",
    seed: int = 0,
    log_dir: str | None = None,
    model_save_path: str | None = None,
    eval_freq: int = 20_000,
    save_freq: int = 100_000,
    use_vec_normalize: bool = True,
    distance_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    curriculum_schedule: list[tuple[float, float]] | None = None,
    reward_schedule: list[tuple[float, ...]] | None = None,
    evasion_schedule: list[tuple[float, float]] | None = None,
    net_arch: list[int] | None = None,
    obs_mask_indices: list[int] | None = None,
    progress_bar: bool = False,
    target_speed_scale: float | None = None,
    env_id: str = "pdomains-ant-tag-v0",
    particle_filter_class: type = AntTagParticleFilter,
    lr_anneal: bool = False,
    target_kl: float | None = None,
    n_epochs: int = 10,
    n_eval_episodes: int = 20,
):
    resolved_arena_scale = (
        arena_scale if arena_scale is not None else get_ant_tag_arena_scale(env_id)
    )
    if log_dir is None or model_save_path is None:
        run_dir = _default_run_dir(seed)
        if log_dir is None:
            log_dir = os.path.join(run_dir, "logs") + "/"
        if model_save_path is None:
            model_save_path = os.path.join(run_dir, "models", "cgf_agent.zip")

    print(f"Training {algorithm} on AntTag with weighted CGF features")
    print(f"CGF features={num_cgf_features}, t_init={t_init_mode}, "
          f"t_frozen={t_frozen}")
    print(f"Arena scale={resolved_arena_scale}")
    print(f"SB3 device={device}")
    print(f"log_dir={log_dir}")
    print(f"model_save_path={model_save_path}")

    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.dirname(model_save_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    initial_vis = curriculum_schedule[0][1] if curriculum_schedule else 100.0
    env_kw = dict(
        num_particles=num_particles,
        distance_coeff=distance_coeff,
        entropy_coeff=entropy_coeff,
        tag_bonus_coeff=0.0,
        initial_visibility_radius=initial_vis,
        obs_mask_indices=obs_mask_indices,
        env_id=env_id,
        particle_filter_class=particle_filter_class,
        # In env_kw (not just the train envs) so the eval envs built from
        # dict(env_kw) below inherit the SAME target speed. Evaluating a
        # different target speed than we trained against would silently
        # misreport success rate.
        target_speed_scale=target_speed_scale,
    )

    env_fns = [
        make_ant_tag_cgf_env(
            **env_kw,
            rank=rank,
            seed=seed,
            monitor_dir=monitor_dir,
        )
        for rank in range(n_envs)
    ]
    vec_env = _make_vec_env_from_fns(env_fns, n_envs)
    if use_vec_normalize:
        vec_env = _make_vec_normalize(vec_env, training=True, norm_reward=True)

    # Eval envs — always evaluate at the env's real POMDP difficulty (its own
    # visible_radius) and on the true sparse reward (no shaping, so the metric
    # doesn't depend on where the training curriculum currently is).
    eval_env_kw = dict(env_kw)
    eval_env_kw["initial_visibility_radius"] = get_env_visible_radius(env_id)
    eval_env_kw["apply_reward_shaping"] = False
    eval_vec_env = DummyVecEnv([
        make_ant_tag_cgf_env(
            **eval_env_kw,
            rank=n_envs + 1,
            seed=seed,
        )
    ])
    if use_vec_normalize:
        eval_vec_env = _make_vec_normalize(
            eval_vec_env,
            training=False,
            norm_reward=False,
        )

    policy_kwargs = {
        "features_extractor_class": WeightedCGFFeaturesExtractor,
        "features_extractor_kwargs": dict(
            num_cgf_features=num_cgf_features,
            arena_scale=resolved_arena_scale,
            t_init_mode=t_init_mode,
            t_init_scale=t_init_scale,
            t_clamp=t_clamp,
            exp_arg_clamp=exp_arg_clamp,
            t_frozen=t_frozen,
        ),
    }
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    if algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=(lambda progress_remaining: learning_rate * progress_remaining) if lr_anneal else learning_rate,
            n_steps=ppo_n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            target_kl=target_kl,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=device,
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=policy_kwargs,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(model_dir, "checkpoints") if model_dir else "checkpoints",
        name_prefix="ant_tag_cgf",
    )
    eval_cb = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(model_dir, "best_model") if model_dir else "best_model",
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=n_eval_episodes,
    )
    t_norm_cb = TNormLoggingCallback()
    curriculum_cb = CurriculumCallback(
        total_timesteps=total_timesteps,
        schedule=curriculum_schedule,
        reward_schedule=reward_schedule,
        evasion_schedule=evasion_schedule,
        verbose=1,
    )

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_cb, eval_cb, curriculum_cb, t_norm_cb],
            progress_bar=progress_bar,
        )
    finally:
        model.save(model_save_path)
        if use_vec_normalize and isinstance(vec_env, VecNormalize):
            vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl") if model_dir else "vecnormalize.pkl"
            vec_env.save(vecnorm_path)
            print(f"VecNormalize saved to {vecnorm_path}")
        print(f"Model saved to {model_save_path}")
        vec_env.close()
        eval_vec_env.close()


def _parse_curriculum(curriculum: str) -> list[tuple[float, float]]:
    schedule = []
    for pair in curriculum.split(","):
        frac, radius = pair.strip().split(":")
        schedule.append((float(frac), float(radius)))
    return schedule


def _parse_reward_schedule(reward_schedule: str) -> list[tuple[float, ...]]:
    schedule = []
    for entry in reward_schedule.split(","):
        parts = [float(part) for part in entry.strip().split(":")]
        if len(parts) == 3:
            parts.append(0.0)
        if len(parts) != 4:
            raise ValueError("Each reward schedule entry must have 3 or 4 values")
        schedule.append(tuple(parts))
    return schedule


class _TeeStream:
    """Duplicates writes to multiple streams (e.g. the real stdout + a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


def _tee_stdout_stderr(log_path: str) -> None:
    """Mirror stdout/stderr into log_path, in addition to the console.

    Lets a background (nohup) run's console output land in a log file that
    lives next to that same run's TensorBoard/model output (both under
    log_dir), instead of depending on the caller to redirect stdout by hand
    into a path that has to be matched back up to a run directory later.
    """
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_file)
    sys.stderr = _TeeStream(sys.stderr, log_file)


def main(encoder: str = "cgf"):
    """Entry point. --variant selects env id, particle filter and run subdir
    together from variants.py, so the three cannot disagree."""
    parser = argparse.ArgumentParser(
        description="RL with weighted CGF particle-belief features on AntTag. "
                    "Use --variant to pick the env; --list_variants to see them."
    )
    variants.add_variant_argument(parser)
    parser.add_argument(
        "--run_subdir", type=str, default=None,
        help="Override the derived runs/ant_tag_cgf[_<variant>] "
             "directory. For a sweep that needs its own tree "
             "(e.g. ant_tag_cgf_cdens_hard_dist0), which the "
             "derived name cannot express.",
    )
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--total_timesteps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help=(
            "Optional short human-readable label appended to the run "
            "directory name (e.g. '6M_vis0.2-0.5'), so parallel runs are "
            "distinguishable by eye in `ls`. Not a substitute for "
            "run_config.json, which always records the full CLI args."
        ),
    )

    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--num_cgf_features", type=int, default=64)
    parser.add_argument(
        "--arena_scale",
        type=float,
        default=None,
        help=(
            "Particle normalization scale for CGF features. Defaults to the "
            "live env's actual arena half-width (cage_max_x), so it always "
            "tracks the PF's arena_limits instead of a stale hardcoded value."
        ),
    )
    parser.add_argument(
        "--t_init_mode",
        type=str,
        default="linspace_all_dims",
        choices=["linspace_all_dims", "linspace_first_dim", "random",
                 "spread"],
    )
    parser.add_argument(
        "--t_frozen",
        action="store_true",
        help=(
            "Register t_values as a BUFFER instead of a Parameter, so PPO "
            "cannot learn the projection directions. Combined with "
            "--t_init_mode spread this isolates representational CAPACITY "
            "(fixed, already-spread features) from the optimization dynamics "
            "of t_j growth."
        ),
    )
    parser.add_argument("--t_init_scale", type=float, default=0.1)
    parser.add_argument("--t_clamp", type=float, default=2.0)
    parser.add_argument("--exp_arg_clamp", type=float, default=20.0)
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Defaults to runs/ant_tag_cgf[_<variant>]/<timestamp>_seed<seed>/logs/",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help="Defaults to runs/ant_tag_cgf[_<variant>]/<timestamp>_seed<seed>/models/cgf_agent.zip",
    )
    parser.add_argument("--eval_freq", type=int, default=20_000)
    parser.add_argument("--save_freq", type=int, default=100_000)
    parser.add_argument("--no_vec_normalize", action="store_true")
    parser.add_argument(
        "--net_arch",
        type=str,
        default=None,
        help="Policy/value MLP sizes, e.g. '256,256'.",
    )

    parser.add_argument("--distance_coeff", type=float, default=1.0)
    parser.add_argument("--entropy_coeff", type=float, default=0.0)
    parser.add_argument(
        "--curriculum",
        type=str,
        default=None,
        help="Visibility curriculum 'frac:radius,...'. Defaults to the "
             "variant's own curriculum, else the base schedule.",
    )
    parser.add_argument(
        "--reward_schedule",
        type=str,
        default="0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50",
    )
    parser.add_argument(
        "--evasion_curriculum",
        type=str,
        default=None,
        help=(
            "Optional frac:scale schedule for SmartAntTagEnv.evasion_scale "
            "(0=dumb-target behavior, 1=full smart evasion). E.g. "
            "'0:0.2,0.5:1,1:1' ramps evasion strength up over the first "
            "half of training. Default: constant 1.0 (full strength "
            "throughout, i.e. no curriculum). No-op on envs without an "
            "evasion_scale knob (e.g. the base AntTag env)."
        ),
    )
    parser.add_argument(
        "--mask_target_obs",
        action="store_true",
        default=True,
        help="Zero out obs[-2:] for the agent. Default on.",
    )
    parser.add_argument(
        "--no_mask_target_obs",
        dest="mask_target_obs",
        action="store_false",
    )
    parser.add_argument(
        "--target_speed_scale",
        type=float,
        default=None,
        help=(
            "SmartAntTagEnv only. How much faster the target moves as it gets "
            "cornered: step = target_step * (1 + urgency * scale). Omit to use "
            "the env default (0.0 = constant speed; the target flees more often "
            "when cornered but never faster). 1.0 restores the old up-to-2x "
            "behavior. Applied to both the training and eval envs."
        ),
    )
    parser.add_argument(
        "--progress_bar",
        action="store_true",
        help="Enable SB3 progress bar. Requires stable-baselines3[extra].",
    )
    parser.add_argument(
        "--lr_anneal",
        action="store_true",
        help="Linearly anneal learning_rate to 0 over training (progress_remaining schedule).",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=None,
        help="PPO target_kl early-stop threshold per rollout (None = disabled).",
    )
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--n_eval_episodes", type=int, default=20)

    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    variant = variants.resolve(args.variant)
    env_id = variant.env_id
    particle_filter_class = variant.particle_filter
    run_subdir = args.run_subdir or variants.run_subdir(
        encoder, args.variant)
    args.curriculum = variants.resolve_schedule(
        args.variant, args.curriculum, "default_curriculum",
        "0:100,0.3:100,0.7:3,1:3")
    variants.warn_if_not_evading(
        args.variant, args.evasion_curriculum, args.target_speed_scale)
    args.evasion_curriculum = variants.resolve_schedule(
        args.variant, args.evasion_curriculum, "default_evasion_curriculum",
        None)

    net_arch = [int(x) for x in args.net_arch.split(",")] if args.net_arch else None
    obs_mask = [-2, -1] if args.mask_target_obs else None

    run_dir = _default_run_dir(args.seed, run_subdir, run_tag=args.run_tag)
    log_dir = args.log_dir or os.path.join(run_dir, "logs") + "/"
    model_save_path = args.model_save_path or os.path.join(run_dir, "models", "cgf_agent.zip")

    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, "stdout.log")
    _tee_stdout_stderr(stdout_log_path)
    print(f"Mirroring stdout/stderr to {stdout_log_path}")

    run_config = vars(args).copy()
    # The resolved run_subdir is passed explicitly below; drop the raw flag so
    # the two do not collide as duplicate keyword arguments. --list_variants
    # already returned by this point and is not part of the run's identity.
    run_config.pop("run_subdir", None)
    run_config.pop("list_variants", None)
    run_config.update(log_dir=log_dir, model_save_path=model_save_path)
    # env_id / particle_filter_class / run_subdir are derived from --variant,
    # but they are still written out: run_config.json stays a complete record
    # even if the registry entry is later edited.
    _write_run_config(
        run_dir,
        env_id=env_id,
        particle_filter_class=particle_filter_class.__name__,
        run_subdir=run_subdir,
        git=_git_provenance(),
        **run_config,
    )

    train_ant_tag_cgf(
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        num_particles=args.num_particles,
        num_cgf_features=args.num_cgf_features,
        arena_scale=args.arena_scale,
        t_init_mode=args.t_init_mode,
        t_init_scale=args.t_init_scale,
        t_clamp=args.t_clamp,
        exp_arg_clamp=args.exp_arg_clamp,
        t_frozen=args.t_frozen,
        device=args.device,
        seed=args.seed,
        log_dir=log_dir,
        model_save_path=model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_vec_normalize=not args.no_vec_normalize,
        distance_coeff=args.distance_coeff,
        entropy_coeff=args.entropy_coeff,
        curriculum_schedule=_parse_curriculum(args.curriculum),
        reward_schedule=_parse_reward_schedule(args.reward_schedule),
        evasion_schedule=_parse_curriculum(args.evasion_curriculum) if args.evasion_curriculum else None,
        net_arch=net_arch,
        obs_mask_indices=obs_mask,
        progress_bar=args.progress_bar,
        target_speed_scale=args.target_speed_scale,
        env_id=env_id,
        particle_filter_class=particle_filter_class,
        lr_anneal=args.lr_anneal,
        target_kl=args.target_kl,
        n_epochs=args.n_epochs,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()
