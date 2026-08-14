"""Unified benchmark trainer: one script for every (env, method, seed).

Fully registry-driven (``set_transformer.rl.benchmark.registry``). Builds
``base env -> PFDictObservationWrapper -> [PotentialBasedShapingWrapper] -> Monitor``,
plugs the chosen feature extractor into an SB3 ``MultiInputPolicy`` (PPO or SAC), trains,
and writes a standardized run record for the cross-machine results harness.

Examples
--------
    python experiments/benchmark/train.py --env ant_tag --method gaussian --seed 0
    python experiments/benchmark/train.py --env ant_tag --method st_frozen --seed 0 \
        --pretrained_model_path experiments/ant_tag_st/<run>/checkpoints/checkpoint_best.pt
    python experiments/benchmark/train.py --env odd_even --method cgf --seed 3

Evaluation always uses a **separate, unshaped** env so eval reward is the true task
reward; SB3's ``EvalCallback`` writes ``evaluations.npz`` (timesteps / returns / lengths)
into the run dir, from which curves, success rate, and the summary table are built.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# `set_transformer` is a namespace package and the editable install may point at a
# different checkout; ensure THIS repo's package root is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from set_transformer.rl.benchmark.callbacks import CGFTNormCallback
from set_transformer.rl.benchmark.registry import (
    build_extractor_kwargs,
    get_env_spec,
    get_method_spec,
)
from set_transformer.rl.wrappers.particle_filter import PFDictObservationWrapper
from set_transformer.rl.wrappers.shaping import PotentialBasedShapingWrapper

ALGOS = {"PPO": PPO, "SAC": SAC}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def resolve_pf_kwargs(env_spec, env) -> dict:
    """Particle-filter kwargs for ``env``: env-derived config, then registry overrides."""
    kwargs = dict(env_spec.pf_kwargs_from_env(env) if env_spec.pf_kwargs_from_env else {})
    kwargs.update(env_spec.particle_filter_kwargs)
    return kwargs


def make_env_thunk(env_spec, seed: int, for_eval: bool, pf_kwargs_record: dict | None = None):
    """Build one wrapped env: base -> PFDict -> [shaping if training] -> Monitor.

    ``pf_kwargs_record``, if given, receives the resolved PF kwargs so the caller can
    record them in the run's ``meta.json`` (the env only exists inside this closure).
    """

    def _init():
        env = env_spec.make_base_env(seed=seed, **env_spec.base_env_kwargs)
        pf_kwargs = resolve_pf_kwargs(env_spec, env)
        if pf_kwargs_record is not None:
            pf_kwargs_record.update(pf_kwargs)
        env = PFDictObservationWrapper(
            env=env,
            particle_filter_class=env_spec.particle_filter_class,
            particle_filter_kwargs=pf_kwargs,
            num_particles=env_spec.num_particles,
            pf_interaction_mapper=env_spec.pf_mapper,
            obs_mask_indices=env_spec.obs_mask_indices,
        )
        # Shaping only during training; eval sees the true (unshaped) reward.
        if not for_eval and env_spec.potential_fn is not None:
            env = PotentialBasedShapingWrapper(
                env, env_spec.potential_fn, gamma=env_spec.gamma
            )
        return Monitor(env)

    return _init


def build_model(algo_name, vec_env, extractor_class, extractor_kwargs, args, gamma):
    policy_kwargs = dict(
        features_extractor_class=extractor_class,
        features_extractor_kwargs=extractor_kwargs,
    )
    common = dict(
        policy="MultiInputPolicy",
        env=vec_env,
        gamma=gamma,
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.results_dir,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    if algo_name == "PPO":
        return PPO(n_steps=args.ppo_n_steps, batch_size=args.batch_size, **common)
    if algo_name == "SAC":
        if isinstance(vec_env.action_space, __import__("gymnasium").spaces.Discrete):
            raise ValueError("SAC does not support discrete action spaces (e.g. odd_even). Use PPO.")
        return SAC(batch_size=args.batch_size, **common)
    raise ValueError(f"Unknown algo {algo_name}")


def encoder_cost(features_extractor) -> dict:
    total = sum(p.numel() for p in features_extractor.parameters())
    trainable = sum(p.numel() for p in features_extractor.parameters() if p.requires_grad)
    stat_dim = None
    if hasattr(features_extractor, "_particle_stat_dim"):
        try:
            stat_dim = int(features_extractor._particle_stat_dim())
        except Exception:
            stat_dim = None
    return {
        "extractor_params_total": int(total),
        "extractor_params_trainable": int(trainable),
        "particle_stat_dim": stat_dim,
        "features_dim": int(features_extractor.features_dim),
    }


def final_metrics(run_dir: Path, success_fn) -> dict:
    npz = run_dir / "evaluations.npz"
    if not npz.exists():
        return {}
    data = np.load(npz)
    last_returns = data["results"][-1]  # [n_eval_episodes]
    last_lengths = data["ep_lengths"][-1]
    out = {
        "eval_return_mean": float(np.mean(last_returns)),
        "eval_return_std": float(np.std(last_returns)),
        "eval_length_mean": float(np.mean(last_lengths)),
        "n_eval_episodes": int(last_returns.shape[0]),
        "final_timestep": int(data["timesteps"][-1]),
    }
    if success_fn is not None:
        successes = [bool(success_fn(float(r), int(l))) for r, l in zip(last_returns, last_lengths)]
        out["success_rate"] = float(np.mean(successes))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", required=True, help="registered env name (e.g. ant_tag, odd_even)")
    p.add_argument("--method", required=True, help="registered method (gaussian, kmoments, cgf, st_frozen, ...)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--algo", choices=list(ALGOS), default=None, help="override env's default algo")
    p.add_argument("--total_timesteps", type=int, default=None, help="override env default")
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--ppo_n_steps", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--features_dim", type=int, default=128)
    p.add_argument("--obs_mlp_hidden_dims", type=int, nargs="+", default=[64, 64])
    # ST-only:
    p.add_argument("--pretrained_model_path", default=None)
    p.add_argument("--num_encodings", type=int, default=8)
    p.add_argument("--dim_encoder", type=int, default=2)
    p.add_argument("--num_inds", type=int, default=32)
    p.add_argument("--dim_hidden", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--no_ln", action="store_true")
    # infra:
    p.add_argument("--results_dir", default="results")
    p.add_argument("--eval_freq", type=int, default=20000, help="env steps between evals")
    p.add_argument("--n_eval_episodes", type=int, default=10)
    p.add_argument("--vec_normalize", action="store_true", help="normalize obs (opt-in)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    env_spec = get_env_spec(args.env)
    method_spec = get_method_spec(args.method)
    algo_name = args.algo or env_spec.default_algo
    total_timesteps = args.total_timesteps or env_spec.default_timesteps

    st_arch = dict(
        num_encodings=args.num_encodings, dim_encoder=args.dim_encoder,
        num_inds=args.num_inds, dim_hidden=args.dim_hidden,
        num_heads=args.num_heads, ln=not args.no_ln,
    )
    extractor_kwargs = build_extractor_kwargs(
        method_spec, args.features_dim, args.obs_mlp_hidden_dims,
        pretrained_model_path=args.pretrained_model_path, st_arch=st_arch,
    )

    run_dir = Path(args.results_dir) / args.env / args.method / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir = str(run_dir)  # tensorboard + eval logs land here

    # --- envs -----------------------------------------------------------------
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    train_env = vec_cls([make_env_thunk(env_spec, args.seed + i, for_eval=False)
                         for i in range(args.n_envs)])
    # Record the resolved PF config off the eval env: it is always a DummyVecEnv, so its
    # thunk runs in this process (SubprocVecEnv workers could not report back).
    pf_kwargs_record: dict = {}
    eval_env = DummyVecEnv([
        make_env_thunk(env_spec, args.seed + 10_000, for_eval=True,
                       pf_kwargs_record=pf_kwargs_record)
    ])
    if args.vec_normalize:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    model = build_model(algo_name, train_env, method_spec.extractor_class, extractor_kwargs, args, env_spec.gamma)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir),  # writes evaluations.npz here
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    # No-op unless the extractor is CGF: logs learned ||t_m|| to TensorBoard + dumps npz.
    callbacks = CallbackList(
        [eval_cb, CGFTNormCallback(run_dir, log_freq=args.eval_freq, verbose=1)]
    )

    cost = encoder_cost(model.policy.features_extractor)
    print(f"[benchmark] env={args.env} method={args.method} algo={algo_name} "
          f"seed={args.seed} steps={total_timesteps} | encoder {cost}")

    start = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    wall_clock = time.time() - start

    model.save(str(run_dir / "final_model"))
    if args.vec_normalize:
        train_env.save(str(run_dir / "vecnormalize.pkl"))

    meta = {
        "env": args.env,
        "method": args.method,
        "seed": args.seed,
        "algo": algo_name,
        "gamma": env_spec.gamma,
        "total_timesteps": total_timesteps,
        "num_particles": env_spec.num_particles,
        "particle_filter": env_spec.particle_filter_class.__name__,
        "particle_filter_kwargs": {
            k: list(v) if isinstance(v, tuple) else v for k, v in pf_kwargs_record.items()
        },
        "shaped": env_spec.potential_fn is not None,
        "vec_normalize": args.vec_normalize,
        "extractor_kwargs": {k: v for k, v in extractor_kwargs.items() if k != "obs_mlp_hidden_dims"},
        "max_ep_steps": env_spec.max_ep_steps,
        "success_criterion": env_spec.success_criterion,
        "git_commit": _git_commit(),
        "wall_clock_sec": wall_clock,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        **cost,
        **final_metrics(run_dir, env_spec.success_fn),
    }
    with open(run_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[benchmark] wrote {run_dir/'meta.json'} | "
          f"return={meta.get('eval_return_mean')} success={meta.get('success_rate')}")


if __name__ == "__main__":
    main()
