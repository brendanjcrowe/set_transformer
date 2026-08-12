"""
Probe the practical top chase speed (V_max) of the trained locomotion policy.

Loads the policy + VecNormalize saved by 1_train_locomotion.py, teleports the
ant to one corner of the arena and the target to the far corner, then runs
the policy for a fixed number of steps per episode, measuring per-step ant
displacement from env.unwrapped.data.qpos[:2]. This is the V_max ceiling used
by .claude/20260730_2337_smart_ant_tag_rl_training_fix_plan.md Phase 2.2.
"""

import argparse

import gymnasium as gym
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers pdomains-ant-tag-v0


class DenseRewardWrapper(gym.Wrapper):
    """Always reveals true target position in obs[-2:] (matches 1_train_locomotion.py)."""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._reveal(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._reveal(obs), reward, terminated, truncated, info

    def _reveal(self, obs):
        obs = obs.copy()
        obs[-2:] = self.env.unwrapped.get_target_pos()
        return obs


def make_env():
    def _init():
        env = gym.make("pdomains-ant-tag-v0", rendering=False)
        env = DenseRewardWrapper(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Probe trained locomotion policy's top chase speed")
    parser.add_argument("--model_path", type=str, default="models/ant_locomotion_policy.zip")
    parser.add_argument("--vecnormalize_path", type=str, default="models/locomotion_vecnorm.pkl")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--n_steps", type=int, default=150)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ant_start", type=float, nargs=2, default=[-4.0, -4.0])
    parser.add_argument("--target_start", type=float, nargs=2, default=[4.4, 4.4])
    args = parser.parse_args()

    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize.load(args.vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(args.model_path, device=args.device)

    ant_start = np.array(args.ant_start, dtype=np.float64)
    target_start = np.array(args.target_start, dtype=np.float64)

    all_displacements = []
    episode_tag_steps = []

    for ep in range(args.n_episodes):
        vec_env.reset()
        base_env = vec_env.envs[0].unwrapped

        base_env.data.qpos[:2] = ant_start
        base_env.data.mocap_pos[0][:2] = target_start
        base_env.data.mocap_pos[1][:2] = ant_start
        base_env.data.mocap_pos[2][:2] = ant_start
        mujoco.mj_step(base_env.model, base_env.data)

        raw_obs = base_env._get_obs(True)
        obs = vec_env.normalize_obs(raw_obs[None, :])

        tagged_at = None
        for t in range(args.n_steps):
            action, _ = model.predict(obs, deterministic=True)
            prev_qpos = base_env.data.qpos[:2].copy()
            obs, reward, done, info = vec_env.step(action)
            if done[0]:
                # DummyVecEnv auto-resets on done, so base_env.data.qpos here
                # reflects the NEXT episode's random reset, not the tag step's
                # true final position. Use the pre-reset qpos VecEnv stashes
                # in info["terminal_observation"] instead (obs, not qpos, but
                # good enough to know a tag happened here) and skip recording
                # a displacement for this corrupted transition.
                tagged_at = t + 1
                break
            cur_qpos = base_env.data.qpos[:2].copy()
            displacement = float(np.linalg.norm(cur_qpos - prev_qpos))
            all_displacements.append(displacement)
        episode_tag_steps.append(tagged_at)

    all_displacements = np.array(all_displacements)
    print(f"n_episodes={args.n_episodes}, n_steps_per_ep={args.n_steps}")
    print(f"n_steps_measured={len(all_displacements)}")
    print(f"episode tagged_at (None=never within n_steps): {episode_tag_steps}")
    print(f"displacement mean = {all_displacements.mean():.4f}")
    print(f"displacement p50  = {np.percentile(all_displacements, 50):.4f}")
    print(f"displacement p90  = {np.percentile(all_displacements, 90):.4f}")
    print(f"displacement max  = {all_displacements.max():.4f}")


if __name__ == "__main__":
    main()
