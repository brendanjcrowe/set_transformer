"""
Training script for RL with pretrained Set Transformer on OddEvenPOMDP environment.

This script trains an RL agent using a pretrained Set Transformer as a feature processor,
similar to train_with_pretrained.py but specifically for the OddEvenPOMDP environment.
"""

import argparse
import os

import gymnasium as gym

# Use a non-interactive backend so plotting works in headless / non-GUI environments.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from features_extractors.set_transformer_pretrained_processor import (
    PretrainedSetTransformerProcessor,
)
from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


class NormalizedRewardCallback(BaseCallback):
    """
    Callback to log normalized reward metrics during training.
    Extracts reward_normalized from info dict and logs it.
    """
    def __init__(self, verbose: int = 0, plot_dir: str | None = None):
        super().__init__(verbose)
        self.normalized_rewards: list[float] = []
        self.raw_rewards: list[float] = []
        self.log_steps: list[int] = []
        self.log_means: list[float] = []
        # Explicit directory where plots will be saved
        self.plot_dir = plot_dir
        
    def _on_step(self) -> bool:
        # Extract normalized reward from info dict if available
        if self.locals.get('infos') is not None:
            for info in self.locals['infos']:
                if isinstance(info, dict):
                    if 'reward_normalized' in info:
                        self.normalized_rewards.append(info['reward_normalized'])
                    if 'reward' in self.locals:
                        # Raw reward might be in locals
                        pass
        
        # Log to tensorboard if available
        if len(self.normalized_rewards) > 0:
            # Calculate mean normalized reward over recent steps
            recent_normalized = self.normalized_rewards[-100:] if len(self.normalized_rewards) > 100 else self.normalized_rewards
            mean_normalized = sum(recent_normalized) / len(recent_normalized) if recent_normalized else 0.0
            
            # Log every 100 steps to avoid too much logging
            if self.num_timesteps % 100 == 0:
                self.logger.record('reward/normalized_mean', mean_normalized)
                if self.verbose > 0:
                    print(f"Step {self.num_timesteps}: Mean normalized reward (last 100): {mean_normalized:.4f}")
                # Store for plotting later
                self.log_steps.append(self.num_timesteps)
                self.log_means.append(mean_normalized)
                # Also save an intermediate plot so we have something even if training aborts
                self._save_plot(suffix="_latest")
        
        return True

    def _on_training_end(self) -> None:
        """Save a plot of normalized reward over time at the end of training."""
        if not self.log_steps:
            return
        self._save_plot(suffix="")

    def _save_plot(self, suffix: str = "") -> None:
        """Helper to save the normalized reward plot to disk."""
        try:
            # Decide where to save the plot
            if self.plot_dir is not None:
                plot_dir = self.plot_dir
            else:
                # Fallback to SB3 logger directory
                plot_dir = self.logger.get_dir() or "."
            os.makedirs(plot_dir, exist_ok=True)
            filename = f"normalized_reward{suffix}.png"
            plot_path = os.path.join(plot_dir, filename)

            plt.figure(figsize=(8, 4))
            plt.plot(self.log_steps, self.log_means, marker="o", linestyle="-")
            plt.xlabel("Timesteps")
            plt.ylabel("Mean normalized reward (last 100 steps)")
            plt.title("Normalized reward over time")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            if self.verbose > 0:
                print(f"Saved normalized reward plot to {plot_path}")
        except Exception as e:
            # Do not crash training if plotting fails
            if self.verbose > 0:
                print(f"Warning: failed to save normalized reward plot: {e}")


class OddEvenPOMDPGymAdapter(gym.Wrapper):
    """
    Wrapper for OddEvenPOMDP to handle max_steps truncation.
    Since POMDP is now gymnasium-compatible, we just wrap it to add max_steps handling.
    """
    def __init__(self, pomdp: OddEvenPOMDP, max_steps: int = 100):
        super().__init__(pomdp)
        self.max_steps = max_steps
        self.step_count = 0
    
    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        # Handle max_steps truncation
        if self.step_count >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


class OddEvenSTWrapper(gym.Wrapper):
    """
    Wrapper for OddEvenPOMDP that processes particles with a pretrained Set Transformer.
    
    The environment already provides particles as observations, so we just need to:
    1. Process particles through the Set Transformer to get features
    2. Use those features as the observation (or concatenate if needed)
    """
    
    def __init__(self, env: gym.Env, pretrained_st_processor: PretrainedSetTransformerProcessor):
        """
        Args:
            env: The OddEvenPOMDP environment (wrapped as gym.Env)
            pretrained_st_processor: The pretrained Set Transformer processor
        """
        super().__init__(env)
        self.st_processor = pretrained_st_processor
        
        # Get the feature dimension from the processor
        self.st_feature_dim = self.st_processor.st_output_dim
        
        # The new observation space is just the features from the Set Transformer
        # (or we could concatenate with something else if needed)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.st_feature_dim,),
            dtype=np.float32
        )
        
        print(f"OddEvenSTWrapper: Original obs shape: {env.observation_space.shape}")
        print(f"OddEvenSTWrapper: ST feature dim: {self.st_feature_dim}")
        print(f"OddEvenSTWrapper: New obs shape: {self.observation_space.shape}")
    
    def reset(self, **kwargs):
        """Reset the environment and process initial particles."""
        obs, info = self.env.reset(**kwargs)
        processed_obs = self._process_particles(obs)
        return processed_obs, info
    
    def step(self, action):
        """Step the environment and process particles."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_particles(obs)
        return processed_obs, reward, terminated, truncated, info
    
    def _process_particles(self, particles: np.ndarray) -> np.ndarray:
        """
        Process particles through the pretrained Set Transformer.
        
        Args:
            particles: Particle observation from the environment, shape [n_particles,]
        
        Returns:
            features: Processed features, shape [st_feature_dim,]
        """
        # Reshape particles to [n_particles, 1] since the ST expects [N, particle_dim]
        # The particles are 1D values, so we need to add a dimension
        particles_reshaped = particles.reshape(-1, 1).astype(np.float32)
        
        # Process through Set Transformer
        # The processor expects particles in the format it was trained on
        # Check if we need to add weights or if particles alone are sufficient
        if self.st_processor.dim_particle_input == 1:
            # Model expects just particle values
            st_features = self.st_processor.process_particles_numpy(particles_reshaped, None)
        elif self.st_processor.dim_particle_input == 2:
            # Model might expect [value, weight] - but we don't have weights from the env
            # So we'll just use particles and let the processor handle it
            # Or create uniform weights
            weights = np.ones(len(particles)) / len(particles)
            st_features = self.st_processor.process_particles_numpy(particles_reshaped, weights)
        else:
            # Fallback: try direct processing
            particle_set_torch = torch.tensor(particles_reshaped, dtype=torch.float32).unsqueeze(0)
            st_features = self.st_processor.process_particles(particle_set_torch).squeeze(0)
        
        return st_features.astype(np.float32)


def make_pretrained_env(
    pomdp_config: OddEvenPOMDPConfig,
    pretrained_st_model_path: str,
    st_processor_device: str,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    max_steps: int = 100
):
    """Utility function for multiprocessed env creation for training with pretrained ST."""
    def _init():
        # Create the POMDP
        pomdp = OddEvenPOMDP(config=pomdp_config)
        # Wrap it as a gym.Env using the adapter
        env = OddEvenPOMDPGymAdapter(pomdp, max_steps=max_steps)
        env.reset(seed=seed + rank)

        # Create the pretrained Set Transformer processor
        st_processor = PretrainedSetTransformerProcessor(
            model_path=pretrained_st_model_path, 
            device=st_processor_device
        )
        
        # Wrap with Set Transformer processor (no particle filter needed)
        env = OddEvenSTWrapper(env, pretrained_st_processor=st_processor)
        
        if monitor_dir:
            log_file = os.path.join(monitor_dir, str(rank))
            env = Monitor(env, log_file)
        else:
            env = Monitor(env)
        return env
    return _init


def train_odd_even_pretrained(
    pomdp_config: OddEvenPOMDPConfig,
    pretrained_st_model_path: str,
    st_processor_device: str = "auto",
    # SB3 Agent params
    algorithm: str = "PPO",
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    batch_size: int = 64,  # For PPO, this is n_steps * n_envs. For SAC, training batch_size.
    ppo_n_steps: int = 2048,
    policy_type: str = "MlpPolicy",  # Standard policy as ST is part of wrapper
    seed: int = 0,
    log_dir: str = "./sb3_odd_even_pretrained_logs/",
    model_save_path: str = "./sb3_odd_even_pretrained_models/pretrained_agent",
    eval_freq: int = 20000,
    save_freq: int = 100000,
    use_vec_normalize: bool = True,
    max_steps: int = 100
):
    """
    Main training loop for RL with a Pretrained Set Transformer as a feature processor
    on the OddEvenPOMDP environment.
    """
    print(f"Starting training for OddEvenPOMDP using pretrained ST: {pretrained_st_model_path}")
    print(f"POMDP config: n_dist_size={pomdp_config.n_dist_size}, std_dev={pomdp_config.std_dev}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    monitor_dir = os.path.join(log_dir, "gym_monitor")
    os.makedirs(monitor_dir, exist_ok=True)

    env_fn = make_pretrained_env(
        pomdp_config=pomdp_config,
        pretrained_st_model_path=pretrained_st_model_path,
        st_processor_device=st_processor_device,
        seed=seed,
        monitor_dir=monitor_dir,
        max_steps=max_steps
    )
    # Use SubprocVecEnv only when using multiple environments; for a single env,
    # DummyVecEnv (default) avoids multiprocessing-related crashes and simplifies debugging.
    vec_env_cls = SubprocVecEnv if n_envs > 1 else None
    vec_env = make_vec_env(env_fn, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls)

    if use_vec_normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Standard policy, as feature extraction is handled by the wrapper
    policy_kwargs = {} 

    if algorithm.upper() == "PPO":
        model = PPO(policy_type, vec_env, policy_kwargs=policy_kwargs, 
                    learning_rate=learning_rate, n_steps=ppo_n_steps, 
                    batch_size=batch_size, 
                    verbose=1, tensorboard_log=log_dir, seed=seed)
    elif algorithm.upper() == "SAC":
        model = SAC(policy_type, vec_env, policy_kwargs=policy_kwargs, 
                    learning_rate=learning_rate, batch_size=batch_size, 
                    verbose=1, tensorboard_log=log_dir, seed=seed, buffer_size=int(1e6))
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose PPO or SAC.")

    print(f"Policy architecture: {model.policy}")

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs, 
        save_path=os.path.join(os.path.dirname(model_save_path), f"{algorithm.lower()}_odd_even_pretrained_checkpoints"), 
        name_prefix="pretrained_model"
    )
    
    eval_env_fn = make_pretrained_env(
        pomdp_config=pomdp_config,
        pretrained_st_model_path=pretrained_st_model_path,
        st_processor_device=st_processor_device,
        rank=n_envs + 1,
        seed=seed,
        monitor_dir=None,
        max_steps=max_steps
    )
    # Single-process eval environment: no need for SubprocVecEnv here.
    eval_vec_env = make_vec_env(eval_env_fn, n_envs=1)
    if use_vec_normalize:
        eval_vec_env = VecNormalize(eval_vec_env, training=False, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(
        eval_vec_env, 
        best_model_save_path=os.path.join(os.path.dirname(model_save_path), f"best_{algorithm.lower()}_odd_even_pretrained"),
        log_path=log_dir, 
        eval_freq=max(eval_freq // n_envs, 1),
        deterministic=True, 
        render=False, 
        n_eval_episodes=5
    )
    
    # Add callback to log normalized reward and save plots
    plot_dir = os.path.join(log_dir, "plots")
    normalized_reward_callback = NormalizedRewardCallback(verbose=1, plot_dir=plot_dir)

    print(f"Starting training with {algorithm} using pretrained ST processor...")
    min_reward = -((pomdp_config.n_dist_size - 1) ** 2)
    print(f"Reward bounds: min={min_reward}, max=0.0 (normalized: 0.0 = worst, 1.0 = best)")
    try:
        model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback, normalized_reward_callback])
    finally:
        model.save(model_save_path)
        if use_vec_normalize and isinstance(vec_env, VecNormalize):
            vec_env.save(os.path.join(os.path.dirname(model_save_path), "vecnormalize_odd_even_pretrained.pkl"))
        print(f"Training finished. Model saved to {model_save_path}")
        vec_env.close()
        eval_vec_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Training with Pretrained Set Transformer on OddEvenPOMDP")
    parser.add_argument("--pretrained_st_model_path", type=str, required=True, 
                        help="Path to the pretrained Set Transformer .pth model file")
    parser.add_argument("--st_processor_device", type=str, default="auto", 
                        choices=["auto", "cpu", "cuda"], help="Device for ST processor")
    
    # POMDP config args
    parser.add_argument("--n_dist_size", type=int, default=10, 
                        help="Maximum number in range [1, n]")
    parser.add_argument("--mean", type=float, default=None, 
                        help="Mean of Gaussian (random if None)")
    parser.add_argument("--std_dev", type=float, default=2.0, 
                        help="Standard deviation of Gaussian")
    parser.add_argument("--pomdp_seed", type=int, default=None, 
                        help="Random seed for POMDP")
    parser.add_argument("--n_particles", type=int, default=100, 
                        help="Number of particles in POMDP")
    parser.add_argument("--max_steps", type=int, default=100, 
                        help="Maximum steps per episode")

    # SB3 Agent args
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--policy_type", type=str, default="MlpPolicy", 
                        help="SB3 policy type (e.g., MlpPolicy, CnnPolicy)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="./sb3_odd_even_pretrained_logs/")
    parser.add_argument("--model_save_path", type=str, 
                        default="./sb3_odd_even_pretrained_models/pretrained_agent.zip")
    parser.add_argument("--eval_freq", type=int, default=20000)
    parser.add_argument("--save_freq", type=int, default=100000)
    parser.add_argument("--no_vec_normalize", action="store_true")

    args = parser.parse_args()

    # Create POMDP config
    pomdp_config = OddEvenPOMDPConfig(
        n_dist_size=args.n_dist_size,
        mean=args.mean,
        std_dev=args.std_dev,
        seed=args.pomdp_seed,
        n_particles=args.n_particles,
        true_particles=True,
        resample_proportion=0.5
    )

    train_odd_even_pretrained(
        pomdp_config=pomdp_config,
        pretrained_st_model_path=args.pretrained_st_model_path,
        st_processor_device=args.st_processor_device,
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        policy_type=args.policy_type,
        seed=args.seed,
        log_dir=args.log_dir,
        model_save_path=args.model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_vec_normalize=not args.no_vec_normalize,
        max_steps=args.max_steps
    )