"""
Pre-train a locomotion policy for the Ant to move toward a target position.

Wraps the AntTag environment with:
  - Full observability (always reveals the true target position in obs[-2:])
  - Dense reward: prev_distance - curr_distance (reward for getting closer)

The trained policy can then be used during data collection: by placing the
particle filter mean in obs[-2:], the policy moves the ant toward the PF estimate.

Usage:
    python train_ant_locomotion.py --total_timesteps 1000000 --save_path models/ant_locomotion_policy
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

import pdomains  # noqa: F401 — registers pdomains-ant-tag-v0


class DenseRewardWrapper(gym.Wrapper):
    """
    Wraps AntTag to provide:
      1) Full observability — always puts true target position in obs[-2:]
      2) Dense reward — reward = (prev_distance - curr_distance) each step,
         plus a bonus on successful tag
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._prev_distance: float = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._make_fully_observable(obs)
        ant_pos = obs[:2]
        target_pos = self.env.unwrapped.get_target_pos()
        self._prev_distance = float(np.linalg.norm(ant_pos - target_pos))
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        obs = self._make_fully_observable(obs)

        ant_pos = obs[:2]
        target_pos = self.env.unwrapped.get_target_pos()
        curr_distance = float(np.linalg.norm(ant_pos - target_pos))

        # Dense reward: positive when getting closer
        dense_reward = self._prev_distance - curr_distance

        # Bonus for tagging (original env sets terminated=True on tag)
        if terminated:
            dense_reward += 10.0

        self._prev_distance = curr_distance
        return obs, dense_reward, terminated, truncated, info

    def _make_fully_observable(self, obs: np.ndarray) -> np.ndarray:
        """Always put the true target position in obs[-2:]."""
        obs = obs.copy()
        target_pos = self.env.unwrapped.get_target_pos()
        obs[-2:] = target_pos
        return obs


def make_locomotion_env(seed: int = 0, rank: int = 0):
    """Factory for creating the dense-reward AntTag env."""
    def _init():
        env = gym.make("pdomains-ant-tag-v0", rendering=False)
        env = DenseRewardWrapper(env)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train ant locomotion policy for target pursuit"
    )
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save_path", type=str, default="models/ant_locomotion_policy"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs/ant_locomotion/"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Training envs
    vec_env = make_vec_env(
        make_locomotion_env(seed=args.seed),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if args.n_envs > 1 else None,
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # Eval env
    eval_vec_env = make_vec_env(
        make_locomotion_env(seed=args.seed + 100),
        n_envs=1,
    )
    eval_vec_env = VecNormalize(
        eval_vec_env, training=False, norm_obs=True, norm_reward=False
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.join(os.path.dirname(args.save_path), "checkpoints"),
        name_prefix="ant_locomotion",
    )
    eval_cb = EvalCallback(
        eval_vec_env,
        best_model_save_path=os.path.join(
            os.path.dirname(args.save_path), "best_locomotion"
        ),
        log_path=args.log_dir,
        eval_freq=max(20_000 // args.n_envs, 1),
        deterministic=True,
        n_eval_episodes=5,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_cb, eval_cb],
            progress_bar=True,
        )
    finally:
        model.save(args.save_path)
        vec_env.save(
            os.path.join(os.path.dirname(args.save_path), "locomotion_vecnorm.pkl")
        )
        print(f"Model saved to {args.save_path}")
        print(
            f"VecNormalize saved to "
            f"{os.path.join(os.path.dirname(args.save_path), 'locomotion_vecnorm.pkl')}"
        )
        vec_env.close()
        eval_vec_env.close()


if __name__ == "__main__":
    main()
