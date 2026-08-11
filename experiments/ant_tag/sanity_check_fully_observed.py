"""
Sanity check: PPO on fully-observed Ant-Tag with the env's native reward.

Reward: -1 per step, 0 on successful tag (terminated=True).
Observation: always reveals true target position in obs[-2:].

If the agent can't learn this, the env or obs space has a problem.
If it CAN learn this but fails through the PF/ST pipeline, the pipeline is lossy.
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import pdomains  # noqa: F401


class FullyObservableWrapper(gym.Wrapper):
    """Always reveals true target position in obs[-2:]. Dense distance reward."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_distance: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._reveal(obs)
        ant_pos = obs[:2]
        target_pos = self.env.unwrapped.get_target_pos()
        self._prev_distance = float(np.linalg.norm(ant_pos - target_pos))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._reveal(obs)
        ant_pos = obs[:2]
        target_pos = self.env.unwrapped.get_target_pos()
        curr_distance = float(np.linalg.norm(ant_pos - target_pos))

        # Dense reward: positive when getting closer
        dense_reward = self._prev_distance - curr_distance
        if terminated:
            dense_reward += 10.0

        self._prev_distance = curr_distance
        return obs, dense_reward, terminated, truncated, info

    def _reveal(self, obs):
        obs = obs.copy()
        obs[-2:] = self.env.unwrapped.get_target_pos()
        return obs


def make_env(env_id="pdomains-ant-tag-v0", seed=0, rank=0):
    def _init():
        env = gym.make(env_id, rendering=False)
        env = FullyObservableWrapper(env)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Sanity check: PPO on fully-observed Ant-Tag")
    parser.add_argument("--env_id", type=str, default="pdomains-ant-tag-v0")
    parser.add_argument("--total_timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="If set, outputs go to ./sanity_check_logs/<run_name>/ and "
             "./sanity_check_models/<run_name>/ instead of the flat defaults, "
             "so smart/dumb-env runs don't collide.",
    )
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    log_dir = args.log_dir or (
        os.path.join("./sanity_check_logs", args.run_name) + "/"
        if args.run_name else "./sanity_check_logs/"
    )
    save_path = args.save_path or (
        os.path.join("./sanity_check_models", args.run_name, "fully_obs_agent")
        if args.run_name else "./sanity_check_models/fully_obs_agent"
    )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vec_env = make_vec_env(
        make_env(env_id=args.env_id, seed=args.seed), n_envs=args.n_envs, seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    eval_vec_env = make_vec_env(make_env(env_id=args.env_id, seed=args.seed + 100), n_envs=1)
    eval_vec_env = VecNormalize(eval_vec_env, training=False, norm_obs=True, norm_reward=False)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=lambda progress_remaining: 3e-4 * progress_remaining,
        n_steps=2048, batch_size=64,
        verbose=1, tensorboard_log=log_dir, seed=args.seed, device=args.device,
    )

    eval_cb = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(os.path.dirname(save_path), "best"),
        log_path=log_dir,
        eval_freq=max(20_000 // args.n_envs, 1),
        deterministic=True, n_eval_episodes=20,
    )

    try:
        model.learn(total_timesteps=args.total_timesteps, callback=[eval_cb], progress_bar=True)
    finally:
        model.save(save_path)
        vec_env.save(os.path.join(os.path.dirname(save_path), "vecnormalize.pkl"))
        print(f"Model saved to {save_path}")
        vec_env.close()
        eval_vec_env.close()


if __name__ == "__main__":
    main()
