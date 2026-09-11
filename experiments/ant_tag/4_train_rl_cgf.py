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
    EncoderDriftLoggingCallback,
    RolloutFeatureNormCallback,
    TNormLoggingCallback,
    WeightedCGFFeaturesExtractor,
    cgf_raw_dim,
    matched_readout_hidden,
    non_readout_param_count,
    readout_param_count,
)
from set_transformer.rl.pretrained_encoder import reload_pretrained_cgf  # noqa: E402
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
    gain_coeff: float = 0.0,
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
                gain_coeff=gain_coeff,
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
    # Ported from the Odd-Even arm on 2026-09-10 (change_mds/
    # ant_tag_cgf_port_2026-09-10.md). Every default is the legacy value, so
    # a caller that does not pass them trains exactly what it trained before.
    t_param: str = "clamp",
    t_bound: float | None = None,
    t_init_max: float | None = None,
    feature_mode: str = "K",
    feature_norm: str = "none",
    running_norm_update: str = "rollout",
    readout_hidden: int = 0,
    readout_depth: int = 0,
    readout_dim: int | None = None,
    pretrained_cgf_model_path: str | None = None,
    cgf_frozen: bool = False,
    x_embed_dim: int = 0,
    x_embed_hidden: int = 64,
    x_embed_depth: int = 1,
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
          f"t_frozen={t_frozen}, t_param={t_param}, t_bound={t_bound}, "
          f"t_init_max={t_init_max}, feature_mode={feature_mode}, "
          f"feature_norm={feature_norm}, readout={readout_hidden}x{readout_depth}"
          f"->{readout_dim}, x_embed_dim={x_embed_dim}, "
          f"pretrained={pretrained_cgf_model_path!r}, cgf_frozen={cgf_frozen}")
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
            t_param=t_param,
            t_bound=t_bound if t_param != "clamp" else None,
            t_init_max=t_init_max,
            feature_mode=feature_mode,
            feature_norm=feature_norm,
            readout_hidden=readout_hidden,
            readout_depth=readout_depth,
            readout_dim=readout_dim,
            pretrained_cgf_model_path=pretrained_cgf_model_path,
            cgf_frozen=cgf_frozen,
            x_embed_dim=x_embed_dim,
            x_embed_hidden=x_embed_hidden,
            x_embed_depth=x_embed_depth,
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

    if pretrained_cgf_model_path:
        # SB3's _build re-initialises every Linear in the extractor (the
        # readout, the embedding) AFTER the constructor loaded them; reload,
        # re-freeze and assert max|delta| == 0 (PITFALLS.md section 1). Also
        # scrubs the absolute path from the saved policy_kwargs.
        reload_pretrained_cgf(model, pretrained_cgf_model_path, cgf_frozen)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=os.path.join(model_dir, "checkpoints") if model_dir else "checkpoints",
        name_prefix="ant_tag_cgf",
        # Also snapshot VecNormalize with every checkpoint: without it the
        # 100k-step checkpoints cannot be evaluated faithfully (the obs
        # normalization is otherwise written once, at the end), and a run
        # killed early is a total loss (PITFALLS.md section 7).
        save_vecnormalize=True,
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
    # cgf/t_norm_q*: does PPO grow ||t||; cgf/drift_*: relative movement of
    # every extractor tensor group since training start (exactly 0 for
    # --cgf_frozen). The running norm's per-cycle refresh only when that norm
    # is in use and unfrozen (PITFALLS.md section 8 item 5).
    encoder_cbs = [TNormLoggingCallback(), EncoderDriftLoggingCallback()]
    if (feature_norm == "running" and running_norm_update == "rollout"
            and not cgf_frozen):
        encoder_cbs.append(RolloutFeatureNormCallback())
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
            callback=[checkpoint_cb, eval_cb, curriculum_cb, *encoder_cbs],
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
        # frac:distance:entropy[:tag_bonus[:spread_gain]] -- missing trailing
        # fields are 0, so every pre-2026-09-08 schedule keeps its meaning.
        if len(parts) in (3, 4):
            parts.extend([0.0] * (5 - len(parts)))
        if len(parts) != 5:
            raise ValueError("Each reward schedule entry must have 3 to 5 values: "
                             "frac:distance:entropy[:tag_bonus[:spread_gain]]")
        schedule.append(tuple(parts))
    return schedule


def _resolve_reward_shaping(parser, args) -> None:
    """Reconcile --distance_coeff/--entropy_coeff with --reward_schedule, in place.

    CurriculumCallback writes the schedule's interpolated coefficients into
    every env on every step, so a coefficient flag given alongside a schedule
    used to govern the first n_envs steps only, while run_config.json recorded
    it as live. Nobody noticed because the default schedule's first waypoint
    equals the flag defaults. Two outcomes now:

    * ``--reward_schedule none`` (or empty): the flags (defaults 1.0 / 0.0)
      become a constant schedule, so they really do hold for the whole run.
    * a schedule plus an explicit flag: ``parser.error``. The user asked for
      two things that cannot both happen.

    In both cases ``args.distance_coeff`` / ``args.entropy_coeff`` are set to
    the values in force at progress 0 and ``args.reward_schedule`` to a
    parseable string, so ``run_config.json`` records what actually ran.
    Shared by the Gaussian and ST arms.
    """
    explicit = [f"--{name}" for name in ("distance_coeff", "entropy_coeff")
                if getattr(args, name) is not None]
    schedule_str = (args.reward_schedule or "").strip()
    if schedule_str.lower() in ("", "none"):
        distance = 1.0 if args.distance_coeff is None else float(args.distance_coeff)
        entropy = 0.0 if args.entropy_coeff is None else float(args.entropy_coeff)
        args.reward_schedule = f"0:{distance}:{entropy}:0,1:{distance}:{entropy}:0"
    else:
        if explicit:
            parser.error(
                f"{' and '.join(explicit)} cannot be combined with "
                "--reward_schedule: the schedule sets the shaping coefficients "
                "on every step, so the flag would govern the first rollout "
                "only. Either drop the flag and put the values in the schedule "
                "(entries are frac:distance:entropy[:tag_bonus]) or pass "
                "--reward_schedule none to run on constant coefficients."
            )
        first = _parse_reward_schedule(schedule_str)[0]
        distance, entropy = float(first[1]), float(first[2])
    args.distance_coeff = distance
    args.entropy_coeff = entropy


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


#: Ant-Tag particles are the target's (x, y): 2-D by construction of every
#: registered filter. Used to size the readout before any env exists; the
#: extractor's strict state_dict load is the backstop if this ever changes.
ANT_TAG_PARTICLE_DIM = 2

#: CGF geometry a pretrained checkpoint carries in its ``config``. With
#: --pretrained_cgf_model_path, a flag left at its default takes the
#: checkpoint's value and an explicit value that disagrees is an error (the
#: Odd-Even arm's rule).
CGF_GEOMETRY_FLAGS = ("num_cgf_features", "feature_mode", "t_param", "t_bound",
                      "t_clamp", "feature_norm", "readout_hidden", "readout_depth",
                      "readout_dim", "arena_scale", "t_init_mode", "t_init_max",
                      "x_embed_dim", "x_embed_hidden", "x_embed_depth")


def add_cgf_encoder_arguments(parser) -> None:
    """The CGF encoder flags ported from experiments/odd_even/4_train_rl_cgf.py
    (2026-09-10). Every default is the LEGACY behaviour -- clamp 2.0, K, no
    norm, no readout -- so a bare re-run of any recorded Ant-Tag CGF command
    still trains what it recorded. The new recipe is opted into per flag, or
    through a variant's registry defaults; see change_mds/
    ant_tag_cgf_port_2026-09-10.md for the reasoning and the sizing rule.
    """
    parser.add_argument(
        "--t_param", type=str, default="clamp", choices=["clamp", "tanh", "polar"],
        help="How the learned t is bounded. 'clamp' (default, legacy): hard "
             "torch.clamp at +-t_clamp, zero gradient beyond it -- on the "
             "'spread' init 4 of 64 probes already sit on the clamp. 'tanh': "
             "t = t_bound * tanh(raw_t), a smooth box (Odd-Even's mode). "
             "'polar': t = t_bound * sigmoid(a) * v/|v|, a smooth BALL with "
             "direction and magnitude learned separately -- the recommended "
             "2-D mode.")
    parser.add_argument(
        "--t_bound", type=float, default=None,
        help="tanh / polar only: the largest tilt. Default: "
             "variants.cgf_t_bound(variant) = 3 / (tag_radius / arena half-width) "
             "-- 9.0 on smart, 13.5 on smart_mid_slow_v15, 35 on cdens_terminal "
             "-- so t * tag_radius sits in the 2..4 window the CGF needs to "
             "resolve the tag disc. The legacy clamp 2.0 gives 0.17..0.67.")
    parser.add_argument(
        "--t_init_max", type=float, default=None,
        help="'spread' init only: the largest probe norm of the log-spaced "
             "grid (from 0.25). Default: unset in clamp mode, which keeps the "
             "extractor's legacy ceiling of 2.8 (4 axis probes then sit on the "
             "clamp -- the recorded behaviour); 0.8 * t_bound otherwise, so the "
             "init spans the range the bound allows without starting on its "
             "flat region. An explicit value above t_clamp in clamp mode is "
             "refused.")
    parser.add_argument(
        "--feature_mode", type=str, default="K", choices=["K", "K_grad", "both"],
        help="K (default, legacy): log-MGF at each probe, T features. K_grad: "
             "the tilted mean K'(t), T x 2 features, bounded by the particle "
             "range whatever t is (needs no standardisation); beat K on "
             "ClusterHunt, LeastMass and Odd-Even. both: concatenated.")
    parser.add_argument(
        "--feature_norm", type=str, default="none", choices=["none", "running", "layernorm"],
        help="Standardise the CGF block before the policy MLP. 'none' "
             "(default): raw, right for K_grad and for K at small t. "
             "'running': per-feature z-score, statistics fixed per PPO cycle "
             "(see --running_norm_update); the Odd-Even 2x2 found it COSTS "
             "from-scratch CGF 0.15-0.25, so use it only for K at wide t. "
             "'layernorm': across features per sample; divides the mean out "
             "of near-rank-1 features (RunningFeatureNorm docstring).")
    parser.add_argument(
        "--running_norm_update", type=str, default="rollout", choices=["rollout", "minibatch"],
        help="--feature_norm running only. 'rollout' (default): statistics "
             "held fixed for each collect + update cycle and refreshed from "
             "the rollout buffer between cycles (RolloutFeatureNormCallback), "
             "so stored and recomputed log-probs are standardised identically. "
             "'minibatch': the pre-fix lerp on every minibatch, A/B control "
             "only (PITFALLS.md section 8 item 5).")
    parser.add_argument(
        "--readout_hidden", type=int, default=0,
        help="Width of an MLP readout between the (normalised) CGF block and "
             "the policy. 0 (default) = none. Where a parameter-matched CGF "
             "arm keeps its budget; see --match_params.")
    parser.add_argument(
        "--readout_depth", type=int, default=0,
        help="Hidden layers of the readout. 0 = none. --match_params with "
             "depth 0 uses 2.")
    parser.add_argument(
        "--readout_dim", type=int, default=None,
        help="Readout output width; default 64 (the ST arm's 8 x 8) whatever "
             "the probe count, so the policy heads match across arms.")
    parser.add_argument(
        "--match_params", type=int, default=None,
        help="Pick --readout_hidden so the encoder total (learned t + norm "
             "affine + readout + embedding) lands closest to this count, e.g. "
             "the ST arm's. Printed and recorded as encoder_params.")
    parser.add_argument(
        "--pretrained_cgf_model_path", type=str, default=None,
        help="A CGF checkpoint (model_state_dict + config, as "
             "experiments/odd_even/3_pretrain_st_belief.py --encoder cgf "
             "writes). Loaded, RE-loaded after PPO construction and verified "
             "max|delta| == 0 (PITFALLS.md section 1). Geometry flags left at "
             "their defaults are taken from it; its arena_scale must equal "
             "the variant's.")
    parser.add_argument(
        "--cgf_frozen", action="store_true",
        help="Freeze the WHOLE pretrained encoder (t, norm statistics, "
             "readout, embedding); only PPO's heads learn -- the CGF twin of "
             "--st_frozen. Requires --pretrained_cgf_model_path.")
    parser.add_argument(
        "--x_embed_dim", type=int, default=0,
        help="Learned per-particle embedding phi: R^2 -> R^d before the CGF "
             "(t then lives in R^d). 0 = off. Makes the arm a learned "
             "Deep-Set-family encoder, not a parameter-free statistic.")
    parser.add_argument("--x_embed_hidden", type=int, default=64)
    parser.add_argument("--x_embed_depth", type=int, default=1)


def resolve_cgf_encoder_args(parser, args, env_id: str) -> None:
    """CLI > checkpoint > registry/default for the CGF encoder flags, in place.

    Order matters and is pinned by tests/test_ant_tag_cgf_port.py:

    1. ``arena_scale`` is resolved from the live env so run_config.json
       records a number, not None.
    2. With a checkpoint, geometry flags at their defaults take the
       checkpoint's values; explicit disagreements are errors. The
       checkpoint's arena_scale must equal the variant's -- the t values
       only mean anything in the frame they were fitted in.
    3. ``t_bound`` (tanh / polar): registry ``cgf_t_bound`` when None.
    4. ``t_init_max`` (spread): left None in clamp mode (the extractor's
       legacy 2.8 ceiling, recorded as null as every old run did), 0.8 *
       t_bound otherwise; an explicit value above t_clamp in clamp mode is
       refused (the extractor refuses it too).
    5. ``--match_params`` sizes the readout; ``encoder_params`` is recorded.
    """
    if args.arena_scale is None:
        args.arena_scale = get_ant_tag_arena_scale(env_id)
    if args.cgf_frozen and not args.pretrained_cgf_model_path:
        parser.error("--cgf_frozen without --pretrained_cgf_model_path would "
                     "freeze a RANDOM readout. Pass a checkpoint or drop the "
                     "flag (use --t_frozen alone to fix t).")

    from_ckpt = {}
    if args.pretrained_cgf_model_path:
        checkpoint = torch.load(args.pretrained_cgf_model_path,
                                map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        from_ckpt = {k: config[k] for k in CGF_GEOMETRY_FLAGS if k in config}
        for key, ckpt_value in from_ckpt.items():
            given = getattr(args, key)
            if key == "arena_scale":
                # Resolved above from the env; the checkpoint has to agree.
                if not np.isclose(float(given), float(ckpt_value), rtol=1e-6, atol=1e-9):
                    parser.error(
                        f"checkpoint {args.pretrained_cgf_model_path} was fitted at "
                        f"arena_scale={ckpt_value!r}, this variant's is {given!r}; the "
                        "t values are meaningless in another frame (PITFALLS.md "
                        "section 4).")
                continue
            if given == parser.get_default(key):
                setattr(args, key, ckpt_value)
            elif given != ckpt_value:
                parser.error(
                    f"--{key} {given!r} disagrees with the checkpoint's {key}="
                    f"{ckpt_value!r} ({args.pretrained_cgf_model_path}). Drop the "
                    "flag to take the checkpoint's geometry.")
        if args.match_params is not None:
            parser.error("--match_params sizes a NEW readout; with a pretrained "
                         "checkpoint the readout shape comes from the checkpoint.")

    if args.t_param == "clamp":
        args.t_bound = None
        # t_init_max stays None: the extractor then uses its built-in spread
        # ceiling of 2.8, WITH the 4 axis probes clamped to 2.0 -- the exact
        # behaviour every recorded Ant-Tag `spread` run had, and what a bare
        # re-run must reproduce. Only an explicit value is policed.
        if args.t_init_max is not None and args.t_init_max > args.t_clamp:
            parser.error(
                f"--t_init_max {args.t_init_max} exceeds --t_clamp {args.t_clamp}: "
                "every probe beyond the clamp would be flattened to +-t_clamp "
                "with zero gradient. Use --t_param polar (or tanh) with a "
                "--t_bound, or lower --t_init_max.")
    else:
        if args.t_bound is None:
            args.t_bound = variants.cgf_t_bound(args.variant)
            print(f"CGF t_bound from the registry: {args.t_bound:.3g} "
                  f"(= {variants.CGF_TILT_TARGET} / (tag_radius / arena half-width) "
                  f"for variant {args.variant!r})")
        if args.t_init_max is None and args.t_init_mode == "spread":
            args.t_init_max = 0.8 * float(args.t_bound)
        if args.t_init_max is not None and args.t_init_max >= args.t_bound:
            parser.error(f"--t_init_max {args.t_init_max} must be below "
                         f"--t_bound {args.t_bound}")

    particle_dim = ANT_TAG_PARTICLE_DIM
    t_dim = args.x_embed_dim if args.x_embed_dim > 0 else particle_dim
    raw_dim = cgf_raw_dim(args.num_cgf_features, t_dim, args.feature_mode)
    fixed = non_readout_param_count(
        args.num_cgf_features, particle_dim, args.feature_mode, args.t_frozen,
        args.feature_norm, args.x_embed_dim, args.x_embed_hidden, args.x_embed_depth)
    if args.t_param == "polar" and not args.t_frozen:
        fixed += args.num_cgf_features       # the extra magnitude scalar per probe
    if args.match_params is not None:
        if args.readout_depth <= 0:
            args.readout_depth = 2
        out_dim = (args.readout_dim if args.readout_dim is not None
                   else WeightedCGFFeaturesExtractor.DEFAULT_READOUT_DIM)
        args.readout_hidden, total = matched_readout_hidden(
            args.match_params, raw_dim, args.readout_depth, out_dim, fixed)
        print(f"CGF readout sized to match {args.match_params:,} params: "
              f"hidden={args.readout_hidden} depth={args.readout_depth} "
              f"-> encoder total {total:,}")
    else:
        out_dim = args.readout_dim if args.readout_dim is not None else (
            WeightedCGFFeaturesExtractor.DEFAULT_READOUT_DIM
            if args.readout_depth > 0 else raw_dim)
        total = fixed + readout_param_count(raw_dim, args.readout_hidden,
                                            args.readout_depth, out_dim)
    args.encoder_params = int(total)
    print(f"CGF encoder: t_param={args.t_param} t_bound={args.t_bound} "
          f"t_init_max={args.t_init_max} feature_mode={args.feature_mode} "
          f"feature_norm={args.feature_norm} raw block {raw_dim} -> "
          f"{args.encoder_params:,} encoder parameters")
    if from_ckpt:
        print("CGF geometry taken from checkpoint: "
              + ", ".join(f"{k}={getattr(args, k)!r}" for k in from_ckpt))


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
    parser.add_argument(
        "--exp_arg_clamp", type=float, default=20.0,
        help="DEPRECATED, no longer applied: the CGF is computed with "
             "logsumexp, which needs no clamp on the exponent. Accepted and "
             "recorded in run_config.json for compatibility only.")
    add_cgf_encoder_arguments(parser)
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

    parser.add_argument(
        "--distance_coeff", type=float, default=None,
        help="Constant PF-mean-distance shaping coefficient (default 1.0). "
             "Only honoured with --reward_schedule none: the schedule sets "
             "these coefficients on every step, so combining the two is an "
             "error rather than a silent override.")
    parser.add_argument(
        "--entropy_coeff", type=float, default=None,
        help="Constant PF belief-entropy shaping coefficient (default 0.0). "
             "NOT PPO's entropy bonus. Same rule as --distance_coeff.")
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
        default=None,
        help="Defaults to the variant's own recipe (variants.py default_reward_schedule), else "
             "'0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50' (PF-entropy 0 throughout). "
             "Shaping schedule 'frac:distance:entropy[:tag_bonus[:spread_gain]],...' (spread_gain pays gain*(spread_{t-1}-spread_t) of the belief's weighted std; 0 when omitted), "
             "interpolated over training progress and applied on every step. "
             "Pass 'none' to run on the constant --distance_coeff / "
             "--entropy_coeff values instead; giving both is an error.",
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
    # Before _resolve_reward_shaping: it reads an empty schedule as "use the
    # constant flags", so a None default must be resolved first.
    args.reward_schedule = variants.resolve_schedule(
        args.variant, args.reward_schedule, "default_reward_schedule",
        "0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50")
    _resolve_reward_shaping(parser, args)
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
    # Before run_config.json is written, so it records the numbers that ran
    # (the 2026-09-03 audit found arena_scale recorded as None).
    resolve_cgf_encoder_args(parser, args, env_id)

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
        t_param=args.t_param,
        t_bound=args.t_bound,
        t_init_max=args.t_init_max,
        feature_mode=args.feature_mode,
        feature_norm=args.feature_norm,
        running_norm_update=args.running_norm_update,
        readout_hidden=args.readout_hidden,
        readout_depth=args.readout_depth,
        readout_dim=args.readout_dim,
        pretrained_cgf_model_path=args.pretrained_cgf_model_path,
        cgf_frozen=args.cgf_frozen,
        x_embed_dim=args.x_embed_dim,
        x_embed_hidden=args.x_embed_hidden,
        x_embed_depth=args.x_embed_depth,
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
