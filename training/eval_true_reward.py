"""
Evaluate a finetune checkpoint on the REAL sparse tag reward.

Builds the eval env identically to the training eval env EXCEPT:
- vis_radius fixed at 3.0 (real POMDP)
- No PFRewardShapingWrapper (so Monitor captures the raw -1/step / 0-on-tag reward)

Usage:
    python training/eval_true_reward.py \
        --model_path sb3_ant_tag_finetune_v2_models/best_model/best_model.zip \
        --vecnormalize_path sb3_ant_tag_finetune_v2_models/vecnormalize.pkl \
        --n_episodes 50
"""
import argparse
import os
import sys

_REPO_WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_WORKDIR not in sys.path:
    sys.path.insert(0, _REPO_WORKDIR)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import pdomains  # noqa: F401
from particle_filters.ant_tag_pf import AntTagParticleFilter
from wrappers.particle_filter_wrappers import PFDictObservationWrapper

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from train_ant_tag_pretrained import (
    CurriculumVisibilityWrapper,
    _CurriculumRouter,
    ant_tag_pf_interaction_mapper,
)


def make_eval_env(num_particles: int, obs_mask_indices, seed: int):
    def _init():
        env = gym.make("pdomains-ant-tag-v0", rendering=False)
        env.reset(seed=seed)
        env = CurriculumVisibilityWrapper(env, initial_visibility_radius=3.0)
        env = PFDictObservationWrapper(
            env=env,
            particle_filter_class=AntTagParticleFilter,
            particle_filter_kwargs={},
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=obs_mask_indices,
        )
        # NOTE: no PFRewardShapingWrapper → Monitor sees true env reward
        env = Monitor(env)
        env = _CurriculumRouter(env)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--vecnormalize_path", type=str, default=None,
                   help="Path to vecnormalize.pkl saved during training")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--num_particles", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_mask", action="store_true")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    args = p.parse_args()

    obs_mask = None if args.no_mask else [-2, -1]

    env_fn = make_eval_env(args.num_particles, obs_mask, args.seed)
    env = DummyVecEnv([env_fn])

    if args.vecnormalize_path and os.path.exists(args.vecnormalize_path):
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False   # keep true reward
        print(f"Loaded VecNormalize from {args.vecnormalize_path}")
    else:
        print("No VecNormalize — evaluating without obs normalization (may be invalid if training used it)")

    model = PPO.load(args.model_path, env=env)
    print(f"Loaded model from {args.model_path}")

    rewards = []
    lengths = []
    tagged = 0
    for ep in range(args.n_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, dones, infos = env.step(action)
            ep_r += float(r[0])
            ep_len += 1
            done = bool(dones[0])
        rewards.append(ep_r)
        lengths.append(ep_len)
        # Tagged = episode ended before truncation. AntTag truncates at 400.
        if ep_len < 400:
            tagged += 1

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    print(f"\n=== Eval over {args.n_episodes} episodes (deterministic={args.deterministic}) ===")
    print(f"Mean reward : {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Min / Max   : {rewards.min():.1f} / {rewards.max():.1f}")
    print(f"Mean length : {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"Tag rate    : {tagged}/{args.n_episodes} ({100*tagged/args.n_episodes:.1f}%)")
    # Distribution of tag times (only on tagged episodes)
    if tagged > 0:
        tag_lens = lengths[lengths < 400]
        print(f"When tagged : mean_len={tag_lens.mean():.1f}, median={np.median(tag_lens):.1f}")


if __name__ == "__main__":
    main()
