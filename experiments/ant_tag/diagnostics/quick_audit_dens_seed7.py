"""Quick first-visit / interception audit for the Twin-Den seed7 checkpoints.

For each episode: was the target tagged WHILE STILL IN TRANSIT (before ever
settling in a den), or after the ant approached a specific den? And when the
ant does approach a den first, does it correlate with tight_den (informed)?
"""
import argparse
import importlib
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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401
from set_transformer.rl.particle_filters.ant_tag import TwinDenAntTagParticleFilter

_train_rl_cgf = importlib.import_module("4_train_rl_cgf")
CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
PFDictWithWeightsObservationWrapper = _train_rl_cgf.PFDictWithWeightsObservationWrapper
_CurriculumRouter = _train_rl_cgf._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs


def make_env(seed):
    def _init():
        env = gym.make("pdomains-ant-tag-dens-v0", rendering=False, target_speed_scale=0.0)
        env.reset(seed=seed)
        pf_kwargs = get_ant_tag_pf_kwargs(env)
        env = CurriculumVisibilityWrapper(env, initial_visibility_radius=3.0)
        env = PFDictWithWeightsObservationWrapper(
            env=env, particle_filter_class=TwinDenAntTagParticleFilter,
            particle_filter_kwargs=pf_kwargs, num_particles=100,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=[-2, -1],
        )
        return _CurriculumRouter(env)
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--vecnormalize_path", required=True)
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--commit_radius", type=float, default=1.6)
    args = p.parse_args()

    vec_env = DummyVecEnv([make_env(seed=123)])
    vec_env = VecNormalize.load(args.vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args.model_path, device="cpu")

    unwrapped = vec_env.envs[0].unwrapped
    dens = np.array([[-2.7, -2.7], [2.7, 2.7]])

    tagged_in_transit = 0
    tagged_after_den_visit = 0
    first_visit_correct = 0
    first_visit_total = 0
    lengths = []

    for ep in range(args.n_episodes):
        obs = vec_env.reset()
        tight_den = unwrapped.tight_den
        first_den_visited = None
        ep_len = 0
        tagged = False
        for t in range(200):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = vec_env.step(action)
            ep_len += 1
            ant_pos = unwrapped.data.qpos[:2]
            d = np.linalg.norm(dens - ant_pos, axis=1)
            if first_den_visited is None and np.min(d) < args.commit_radius:
                first_den_visited = int(np.argmin(d))
            if done[0]:
                tagged = ep_len < 200
                break
        lengths.append(ep_len)
        if tagged:
            if first_den_visited is None:
                tagged_in_transit += 1
            else:
                tagged_after_den_visit += 1
        if first_den_visited is not None:
            first_visit_total += 1
            if first_den_visited == tight_den:
                first_visit_correct += 1

    print(f"n_episodes={args.n_episodes}")
    print(f"tagged in transit (never got within {args.commit_radius} of a den before tag): {tagged_in_transit}")
    print(f"tagged after visiting a den first: {tagged_after_den_visit}")
    print(f"episodes where ant ever got within {args.commit_radius} of a den: {first_visit_total}")
    if first_visit_total:
        print(f"P(first den visited == tight_den) = {first_visit_correct/first_visit_total:.3f}  (chance=0.5)")
    print(f"mean episode length: {np.mean(lengths):.1f}")


if __name__ == "__main__":
    main()
