"""
Evaluate the best-trained OddEven PPO+ST policy with a custom 0/-1 reward.

Reward override (only at eval time, not env-internal):
    +0  if the agent's predicted_mean exactly matches the true mean
    -1  otherwise

Each episode is fixed-length (max_steps, default 100) and the agent makes one
guess per step, so the per-episode custom reward is in [-max_steps, 0] and
equals -(number of wrong guesses).

Usage:
    python training/eval_odd_even_pretrained.py \
        --model_path sb3_odd_even_pretrained_models/best_ppo_odd_even_pretrained/best_model.zip \
        --vecnormalize_path sb3_odd_even_pretrained_models/vecnormalize_odd_even_pretrained.pkl \
        --pretrained_st_model_path <path-to-st-checkpoint>.pt \
        --n_episodes 50
"""
import argparse
import os
import sys

_REPO_WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_WORKDIR not in sys.path:
    sys.path.insert(0, _REPO_WORKDIR)

import numpy as np
from pdomains.odd_even_pomdp import OddEvenPOMDPConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from train_odd_even_pretrained import make_pretrained_env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="sb3_odd_even_pretrained_models/best_ppo_odd_even_pretrained/best_model.zip")
    p.add_argument("--vecnormalize_path", type=str,
                   default="sb3_odd_even_pretrained_models/vecnormalize_odd_even_pretrained.pkl")
    p.add_argument("--pretrained_st_model_path", type=str, required=True,
                   help="Pretrained Set Transformer checkpoint (.pt) — same one used during PPO training")
    p.add_argument("--st_processor_device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--n_dist_size", type=int, default=10)
    p.add_argument("--std_dev", type=float, default=2.0)
    p.add_argument("--n_particles", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    args = p.parse_args()

    pomdp_config = OddEvenPOMDPConfig(
        n_dist_size=args.n_dist_size,
        mean=None,
        std_dev=args.std_dev,
        seed=None,
        n_particles=args.n_particles,
        true_particles=True,
        resample_proportion=0.5,
    )

    env_fn = make_pretrained_env(
        pomdp_config=pomdp_config,
        pretrained_st_model_path=args.pretrained_st_model_path,
        st_processor_device=args.st_processor_device,
        seed=args.seed,
        monitor_dir=None,
        max_steps=args.max_steps,
    )
    env = make_vec_env(env_fn, n_envs=1, seed=args.seed, vec_env_cls=DummyVecEnv)

    if args.vecnormalize_path and os.path.exists(args.vecnormalize_path):
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False  # we override reward anyway
        print(f"Loaded VecNormalize from {args.vecnormalize_path}")
    else:
        print(f"WARNING: no VecNormalize at {args.vecnormalize_path} — obs normalization missing")

    model = PPO.load(args.model_path, env=env)
    print(f"Loaded model from {args.model_path}")
    print(f"Eval reward: 0 if predicted_mean == true_mean, -1 otherwise")
    print(f"Episodes: {args.n_episodes}, max_steps/episode: {args.max_steps}, "
          f"deterministic={args.deterministic}\n")

    ep_rewards = []
    ep_correct_counts = []
    for ep in range(args.n_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        ep_correct = 0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _r, dones, infos = env.step(action)
            info = infos[0]
            correct = int(info["predicted_mean"]) == int(info["true_mean"])
            ep_r += 0.0 if correct else -1.0
            ep_correct += int(correct)
            ep_len += 1
            done = bool(dones[0])
        ep_rewards.append(ep_r)
        ep_correct_counts.append(ep_correct)

    ep_rewards = np.array(ep_rewards)
    ep_correct_counts = np.array(ep_correct_counts)
    accuracy = ep_correct_counts / args.max_steps  # fraction correct per episode

    print(f"=== Eval over {args.n_episodes} episodes ===")
    print(f"Custom reward (sum per episode, range [-{args.max_steps}, 0]):")
    print(f"  mean: {ep_rewards.mean():.2f} ± {ep_rewards.std():.2f}")
    print(f"  min / max: {ep_rewards.min():.0f} / {ep_rewards.max():.0f}")
    print(f"Per-step accuracy (correct guesses / step):")
    print(f"  mean: {accuracy.mean():.3f} ± {accuracy.std():.3f}")
    print(f"  min / max: {accuracy.min():.3f} / {accuracy.max():.3f}")


if __name__ == "__main__":
    main()
