"""
RL training with a weighted Gaussian (mean + covariance) belief encoder for Ant-Tag.

This keeps the AntTag particle filter, curriculum, and reward-shaping pipeline
from 4_train_rl_frozen.py / 4_train_rl_cgf.py, but replaces the belief encoder
with the lowest-order moments of the weighted particle distribution:

    mean_d      = sum_i w_i * x_i[d]
    cov_{d,d'}  = sum_i w_i * (x_i[d] - mean_d) * (x_i[d'] - mean_d')

for particle dimension d, d' in {0, ..., D-1} (D=2 for AntTag: target x, y).
The policy receives [mean, var_x, var_y, cov_xy] (5 features for D=2), analogous
to mean_var_encoding_odd_even_beliefmdp.py's [mean, var] for the 1D OddEven belief.

Unlike the CGF and Set Transformer encoders, this extractor has no learnable
parameters — it is a fixed, closed-form summary of the particle set.
"""

import argparse
import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers pdomains-ant-tag-v0


# Reuse the existing AntTag curriculum/PF/dict-obs utilities. The dict-obs env
# (particles + PF weights) built by 4_train_rl_cgf.py has no CGF-specific
# logic in it, so it is reused as-is for the Gaussian extractor too.
_train_rl_cgf = importlib.import_module("4_train_rl_cgf")
CurriculumCallback = _train_rl_cgf.CurriculumCallback
CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
PFRewardShapingWrapper = _train_rl_cgf.PFRewardShapingWrapper
_CurriculumRouter = _train_rl_cgf._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs
get_ant_tag_arena_scale = _train_rl_cgf.get_ant_tag_arena_scale
get_env_visible_radius = _train_rl_cgf.get_env_visible_radius
PFDictWithWeightsObservationWrapper = _train_rl_cgf.PFDictWithWeightsObservationWrapper
make_ant_tag_belief_env = _train_rl_cgf.make_ant_tag_cgf_env
_make_vec_normalize = _train_rl_cgf._make_vec_normalize
_make_vec_env_from_fns = _train_rl_cgf._make_vec_env_from_fns
AntTagParticleFilter = _train_rl_cgf.AntTagParticleFilter
_tee_stdout_stderr = _train_rl_cgf._tee_stdout_stderr


class WeightedGaussianFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor for weighted Gaussian (mean + covariance) particle features.

    Computes the weighted mean and covariance of the particle set (normalized
    by arena_scale, matching WeightedCGFFeaturesExtractor's particle scaling),
    then exposes [mean, var, off-diagonal covariance] as features. For
    particle_dim=D this is D + D + D*(D-1)/2 features (5 for D=2).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        arena_scale: float = 4.5,
    ):
        obs_dim = observation_space["obs"].shape[0]
        particle_dim = observation_space["particles"].shape[1]
        num_gaussian_features = particle_dim + particle_dim + particle_dim * (particle_dim - 1) // 2
        super().__init__(observation_space, features_dim=obs_dim + num_gaussian_features)

        self.particle_dim = particle_dim
        self.arena_scale = arena_scale

        triu_indices = torch.triu_indices(particle_dim, particle_dim, offset=1)
        self.register_buffer("triu_rows", triu_indices[0])
        self.register_buffer("triu_cols", triu_indices[1])

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        base_obs = obs_dict["obs"]
        particles = obs_dict["particles"] / self.arena_scale
        weights = obs_dict["weights"]

        particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        w = weights.unsqueeze(-1)  # [B, N, 1]
        mean = torch.sum(w * particles, dim=1)  # [B, D]

        centered = particles - mean.unsqueeze(1)  # [B, N, D]
        cov = torch.einsum("bni,bnj->bij", w * centered, centered)  # [B, D, D]
        cov = torch.nan_to_num(cov, nan=0.0)

        # Weighted variance can dip slightly below zero from floating-point
        # cancellation (mean subtracted then squared back out); clamp so the
        # policy never sees a negative "variance" feature.
        var = torch.clamp(torch.diagonal(cov, dim1=-2, dim2=-1), min=0.0)  # [B, D]
        off_diag_cov = cov[:, self.triu_rows, self.triu_cols]  # [B, D*(D-1)/2]

        gaussian_features = torch.cat([mean, var, off_diag_cov], dim=-1)
        return torch.cat([base_obs, gaussian_features], dim=-1)


def _default_run_dir(seed: int, run_subdir: str = "ant_tag_gaussian",
                      run_tag: str | None = None) -> str:
    """runs/<run_subdir>/<timestamp>_seed<seed>[_<run_tag>]/, so parallel runs
    with different seeds (and different env variants, via run_subdir) land in
    distinct, sortable, self-describing folders instead of overwriting a
    fixed sb3_ant_tag_gaussian_logs/ path.

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


def train_ant_tag_gaussian(
    algorithm: str = "PPO",
    total_timesteps: int = 3_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    ppo_n_steps: int = 2048,
    num_particles: int = 100,
    arena_scale: float | None = None,
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
            model_save_path = os.path.join(run_dir, "models", "gaussian_agent.zip")

    print(f"Training {algorithm} on AntTag with weighted Gaussian (mean+cov) features")
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
        # In env_kw so the eval envs built from dict(env_kw) below inherit
        # the SAME target speed as training.
        target_speed_scale=target_speed_scale,
    )

    env_fns = [
        make_ant_tag_belief_env(
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
        make_ant_tag_belief_env(
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
        "features_extractor_class": WeightedGaussianFeaturesExtractor,
        "features_extractor_kwargs": dict(
            arena_scale=resolved_arena_scale,
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
        name_prefix="ant_tag_gaussian",
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
            callback=[checkpoint_cb, eval_cb, curriculum_cb],
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


def main(
    env_id: str = "pdomains-ant-tag-v0",
    particle_filter_class: type = AntTagParticleFilter,
    run_subdir: str = "ant_tag_gaussian",
):
    parser = argparse.ArgumentParser(
        description=(
            "RL with weighted Gaussian (mean+covariance) particle-belief "
            f"features on AntTag (env_id={env_id})"
        )
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
    parser.add_argument(
        "--arena_scale",
        type=float,
        default=None,
        help=(
            "Particle normalization scale for Gaussian features. Defaults to "
            "the live env's actual arena half-width (cage_max_x), so it "
            "always tracks the PF's arena_limits instead of a stale "
            "hardcoded value."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda:1")

    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=f"Defaults to runs/{run_subdir}/<timestamp>_seed<seed>/logs/",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=None,
        help=f"Defaults to runs/{run_subdir}/<timestamp>_seed<seed>/models/gaussian_agent.zip",
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
        default="0:100,0.3:100,0.7:3,1:3",
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
    net_arch = [int(x) for x in args.net_arch.split(",")] if args.net_arch else None
    obs_mask = [-2, -1] if args.mask_target_obs else None

    run_dir = _default_run_dir(args.seed, run_subdir, run_tag=args.run_tag)
    log_dir = args.log_dir or os.path.join(run_dir, "logs") + "/"
    model_save_path = args.model_save_path or os.path.join(run_dir, "models", "gaussian_agent.zip")

    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, "stdout.log")
    _tee_stdout_stderr(stdout_log_path)
    print(f"Mirroring stdout/stderr to {stdout_log_path}")

    run_config = vars(args).copy()
    run_config.update(log_dir=log_dir, model_save_path=model_save_path)
    _write_run_config(
        run_dir,
        env_id=env_id,
        particle_filter_class=particle_filter_class.__name__,
        run_subdir=run_subdir,
        **run_config,
    )

    train_ant_tag_gaussian(
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        num_particles=args.num_particles,
        arena_scale=args.arena_scale,
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
