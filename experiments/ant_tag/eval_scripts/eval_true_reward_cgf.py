"""
Evaluate a weighted-CGF checkpoint on the REAL sparse tag reward.

Mirrors eval_true_reward.py but uses PFDictWithWeightsObservationWrapper
(particles + PF weights), since WeightedCGFFeaturesExtractor requires both.

Usage:
    python experiments/ant_tag/eval_scripts/eval_true_reward_cgf.py \
        --model_path sb3_ant_tag_cgf_models_3M/best_model/best_model.zip \
        --vecnormalize_path sb3_ant_tag_cgf_models_3M/vecnormalize.pkl \
        --n_episodes 50
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The 4_train_rl_* pipeline scripts live one level up, in experiments/ant_tag/.
# Only this script's own directory is on sys.path by default, so add theirs.
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
if str(_ANT_TAG_DIR) not in sys.path:
    sys.path.insert(0, str(_ANT_TAG_DIR))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import pdomains  # noqa: F401
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter

# Sibling module name starts with a digit, so importlib is required.
_train_rl_cgf = importlib.import_module("4_train_rl_cgf")
CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
_CurriculumRouter = _train_rl_cgf._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs
PFDictWithWeightsObservationWrapper = _train_rl_cgf.PFDictWithWeightsObservationWrapper


def make_eval_env(num_particles: int, obs_mask_indices, seed: int,
                   env_id: str = "pdomains-ant-tag-v0",
                   particle_filter_class: type = AntTagParticleFilter):
    def _init():
        env = gym.make(env_id, rendering=False)
        env.reset(seed=seed)
        particle_filter_kwargs = get_ant_tag_pf_kwargs(env)
        env = CurriculumVisibilityWrapper(
            env,
            initial_visibility_radius=float(env.unwrapped.visible_radius))
        env = PFDictWithWeightsObservationWrapper(
            env=env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs=particle_filter_kwargs,
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=obs_mask_indices,
        )
        # NOTE: no PFRewardShapingWrapper → Monitor sees true env reward
        env = Monitor(env)
        env = _CurriculumRouter(env)
        return env
    return _init


def main(env_id: str = "pdomains-ant-tag-v0",
         particle_filter_class: type = AntTagParticleFilter):
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--vecnormalize_path", type=str, default=None,
                   help="Path to vecnormalize.pkl saved during training")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument("--num_particles", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_steps", type=int, default=400,
        help="Episode cap of the env under evaluation; an episode that\n             ends strictly before this many steps counts as a tag.\n             400 for the v0/smart/ghost AntTag variants, 200 for\n             pdomains-ant-tag-dens-v0.",
    )
    p.add_argument("--no_mask", action="store_true")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    args = p.parse_args()

    obs_mask = None if args.no_mask else [-2, -1]

    print(f"Env: {env_id}, particle filter: {particle_filter_class.__name__}")
    env_fn = make_eval_env(args.num_particles, obs_mask, args.seed,
                            env_id=env_id, particle_filter_class=particle_filter_class)
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
        # Tagged = episode ended before truncation.
        if ep_len < args.max_steps:
            tagged += 1

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    best_idx = int(np.argmax(rewards))

    print(f"\n=== Eval over {args.n_episodes} episodes (deterministic={args.deterministic}) ===")
    print(f"Success rate  : {tagged}/{args.n_episodes} ({100*tagged/args.n_episodes:.1f}%)")
    print(f"Mean reward   : {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Mean length   : {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"Median length : {np.median(lengths):.1f}")
    print(f"Best episode  : reward={rewards[best_idx]:.2f}, length={lengths[best_idx]}")
    # Distribution of tag times (only on tagged episodes)
    if tagged > 0:
        tag_lens = lengths[lengths < args.max_steps]
        print(f"When tagged   : mean_len={tag_lens.mean():.1f}, median_len={np.median(tag_lens):.1f}")


if __name__ == "__main__":
    main()
