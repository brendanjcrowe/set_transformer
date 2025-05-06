"""
End-to-End Reinforcement Learning with Integrated Set Transformer

This script implements end-to-end training where the Set Transformer is integrated directly 
into the Stable Baselines 3 policy network. The particle filter beliefs are processed by
the Set Transformer, which is jointly trained with the RL policy.

Usage:
    python sb3_end_to_end.py --num_particles 100
"""

import argparse
import gymnasium as gym
import numpy as np
import os
import torch
import torch.nn as nn
from torch.distributions import Normal
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Import Set Transformer modules
import sys
sys.path.append('.')  # Add the current directory to path
try:
    from set_transformer.modules import ISAB, PMA, SAB
except ImportError:
    raise ImportError("Could not import Set Transformer modules. Make sure the set_transformer module is in your PYTHONPATH.")

# Import Stable Baselines 3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# Import our particle filter implementation
from particle_filter_ant_tag import AntTagParticleFilter

class SetTransformerNetwork(nn.Module):
    """
    Set Transformer network that processes sets of particles.
    This implementation follows the architecture from the Set Transformer paper.
    """
    def __init__(
        self,
        dim_input=3,    # x, y, weight
        dim_hidden=128,
        dim_output=64,
        num_heads=4,
        num_inds=32,
        num_outputs=1,
        ln=False
    ):
        super(SetTransformerNetwork, self).__init__()
        
        # Encoder: Maps each particle to a higher-dimensional space
        self.encoder = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        )
        
        # Decoder: Produces a fixed-size representation
        self.decoder = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output)
        )
        
        # Store dimensions for later use
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_outputs = num_outputs
        
    def forward(self, x):
        """
        Forward pass through the Set Transformer.
        
        Args:
            x: Input tensor of shape [batch_size, num_particles, dim_input]
            
        Returns:
            output: Tensor of shape [batch_size, dim_output]
        """
        # Encoder: Process each particle
        x = self.encoder(x)
        
        # Decoder: Aggregate information into fixed-size representation
        x = self.decoder(x)
        
        # Reshape to [batch_size, dim_output]
        x = x.view(-1, self.num_outputs * self.dim_output)
        
        return x

class CustomSetTransformerExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Stable Baselines 3 that integrates a Set Transformer
    to process particle filter beliefs.
    """
    def __init__(
        self,
        observation_space,
        set_transformer_dim=64,
        set_transformer_heads=4,
        set_transformer_inds=32,
        features_dim=128
    ):
        # Determine the shape of the observation space
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError("CustomSetTransformerExtractor expects a Dict observation space")
        
        super(CustomSetTransformerExtractor, self).__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        self.obs_dim = observation_space["obs"].shape[0]
        self.num_particles = observation_space["particles"].shape[0]
        self.particle_dim = observation_space["particles"].shape[1]
        
        # Create the Set Transformer
        self.set_transformer = SetTransformerNetwork(
            dim_input=self.particle_dim,
            dim_hidden=128,
            dim_output=set_transformer_dim,
            num_heads=set_transformer_heads,
            num_inds=set_transformer_inds
        )
        
        # Create MLP for original observations
        self.obs_net = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combine features from both sources
        self.combined_net = nn.Sequential(
            nn.Linear(64 + set_transformer_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        """
        Process both original observations and particle filter states
        
        Args:
            observations: Dict with 'obs' (environment observations) and 'particles' (particle states)
            
        Returns:
            features: Combined features for policy
        """
        # Process original observations
        obs_features = self.obs_net(observations["obs"])
        
        # Process particles with Set Transformer
        particles = observations["particles"]
        # Add batch dimension if needed
        if len(particles.shape) == 2:
            particles = particles.unsqueeze(0)
        particle_features = self.set_transformer(particles)
        
        # Combine features
        combined = torch.cat([obs_features, particle_features], dim=1)
        return self.combined_net(combined)

class ParticleFilterDictWrapper(gym.Wrapper):
    """
    Environment wrapper that maintains a particle filter and returns a dict observation
    with both environment observations and particle states.
    """
    def __init__(self, env, num_particles=100):
        super(ParticleFilterDictWrapper, self).__init__(env)
        self.num_particles = num_particles
        self.particle_filter = None
        
        # Define the new observation space as a Dict
        self.observation_space = gym.spaces.Dict({
            "obs": env.observation_space,
            "particles": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_particles, 3)  # x, y, weight
            )
        })
        
        print(f"Original observation space: {env.observation_space}")
        print(f"New observation space: {self.observation_space}")
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Initialize a new particle filter
        self.particle_filter = AntTagParticleFilter(self.num_particles, obs)
        
        # Process the initial state
        processed_obs = self._process_observation(obs)
        return processed_obs, info
    
    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update particle filter
        # First predict step
        self.particle_filter.predict(action)
        
        # Get the true opponent position (for evaluation only)
        true_opponent_pos = self.env.env.get_target_pos()
        
        # Check if opponent is visible
        ant_pos = np.array([next_obs[0], next_obs[1]])
        opponent_visible = np.linalg.norm(ant_pos - true_opponent_pos) <= 3.0
        
        # Get observed position (NaN if not visible)
        observed_opponent_pos = next_obs[13:15] if opponent_visible else np.array([np.nan, np.nan])
        
        # Update particle filter with observation
        self.particle_filter.update(observed_opponent_pos, ant_pos)
        
        # Process observation with particle filter
        processed_obs = self._process_observation(next_obs)
        
        return processed_obs, reward, terminated, truncated, info
    
    def _process_observation(self, obs):
        """
        Process observation with particle filter
        """
        # Get particles and weights
        particles = self.particle_filter.particles
        weights = self.particle_filter.weights
        
        # Combine particles and weights
        particle_features = np.column_stack((particles, weights.reshape(-1, 1)))
        
        # Create the dict observation
        dict_obs = {
            "obs": obs,
            "particles": particle_features.astype(np.float32)
        }
        
        return dict_obs

def make_ant_tag_env(num_particles=100):
    """Helper function to create an Ant-Tag environment with particle filter"""
    env = gym.make('pdomains-ant-tag-v0')
    env = ParticleFilterDictWrapper(env, num_particles)
    env = Monitor(env)
    return env

class ParticleFilterVisualizerCallback(BaseCallback):
    """Callback for visualizing particle filter during training"""
    
    def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=3, verbose=1):
        super(ParticleFilterVisualizerCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Run episodes with visualization
            self._visualize_episodes()
        return True
    
    def _visualize_episodes(self):
        """Run evaluation episodes with visualization"""
        returns = []
        
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Predict action
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Apply action
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                episode_reward += reward
                
                # Visualize particle filter
                self._visualize_step(obs, ax, ep)
                plt.pause(0.01)
                
                done = terminated or truncated
            
            returns.append(episode_reward)
            print(f"Episode {ep+1} reward: {episode_reward:.2f}")
        
        plt.close(fig)
        plt.ioff()
        
        mean_reward = np.mean(returns)
        
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            # Save best model
            self.model.save(f"models/sb3_best_model")
            print(f"New best model with mean reward: {mean_reward:.2f}")
        
        print(f"Mean episode reward: {mean_reward:.2f}")
    
    def _visualize_step(self, obs, ax, episode):
        """Visualize a single step"""
        # Access particle filter from environment
        pf = self.eval_env.particle_filter
        
        # Clear previous plot
        ax.clear()
        
        # Get particles and weights
        particles = pf.particles
        weights = pf.weights
        colors = plt.cm.viridis(weights / np.max(weights))
        
        # Plot particles
        ax.scatter(particles[:, 0], particles[:, 1], c=colors, s=15, alpha=0.6, label='Particles')
        
        # Plot ant position
        ant_pos = np.array([obs["obs"][0], obs["obs"][1]])
        ax.scatter(ant_pos[0], ant_pos[1], color='red', s=100, label='Ant')
        
        # Plot true opponent position
        true_pos = self.eval_env.env.env.get_target_pos()
        ax.scatter(true_pos[0], true_pos[1], color='green', s=100, label='True Opponent')
        
        # Plot estimated position
        est_pos = pf.estimate()
        ax.scatter(est_pos[0], est_pos[1], color='blue', s=100, label='Estimated')
        
        # Draw visibility circle
        visibility = Circle(ant_pos, 3.0, fill=False, linestyle='--', color='blue', alpha=0.5)
        ax.add_patch(visibility)
        
        # Set plot limits and labels
        ax.set_xlim([-15, 15])
        ax.set_ylim([-15, 15])
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Ant-Tag with Set Transformer - Episode {episode+1}')
        ax.legend()

def train_end_to_end(
    num_particles=100,
    total_timesteps=1000000,
    algorithm="PPO",
    n_envs=4,
    save_path="models/sb3_e2e_agent"
):
    """
    Train an RL agent with an integrated Set Transformer in an end-to-end manner
    
    Args:
        num_particles: Number of particles for the filter
        total_timesteps: Number of timesteps to train for
        algorithm: RL algorithm to use ("PPO" or "SAC")
        n_envs: Number of parallel environments
        save_path: Where to save the trained model
    """
    # Create a function to initialize environment with wrapper
    vec_env = make_vec_env(
        lambda: make_ant_tag_env(num_particles), 
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv
    )
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Create a single environment for evaluation
    eval_env = make_ant_tag_env(num_particles)
    
    # Create the policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomSetTransformerExtractor,
        features_extractor_kwargs=dict(
            features_dim=128,
            set_transformer_dim=64,
            set_transformer_heads=4,
            set_transformer_inds=32
        ),
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )
    
    # Create directory for saving models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.dirname(save_path),
        name_prefix="sb3_e2e_model"
    )
    
    # Visualization callback
    vis_callback = ParticleFilterVisualizerCallback(eval_env, eval_freq=50000)
    
    # Combined callbacks
    callbacks = [checkpoint_callback, vis_callback]
    
    # Create and train the agent
    if algorithm.upper() == "PPO":
        model = PPO(
            "MultiInputPolicy", 
            vec_env, 
            policy_kwargs=policy_kwargs,
            verbose=1, 
            tensorboard_log="./sb3_e2e_tensorboard/"
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MultiInputPolicy", 
            vec_env, 
            policy_kwargs=policy_kwargs,
            verbose=1, 
            tensorboard_log="./sb3_e2e_tensorboard/"
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    # Save the final model
    model.save(save_path)
    vec_env.save(f"{save_path}_vecnormalize.pkl")
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model

def evaluate_e2e_agent(
    model_path,
    vecnormalize_path=None,
    num_particles=100,
    num_episodes=5,
    render=True
):
    """
    Evaluate and visualize a trained end-to-end agent
    
    Args:
        model_path: Path to the trained model
        vecnormalize_path: Path to the saved VecNormalize stats
        num_particles: Number of particles for filter
        num_episodes: Number of episodes to evaluate
        render: Whether to render visualization
    """
    # Create environment
    env = make_ant_tag_env(num_particles)
    
    # Load normalization stats if provided
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        # Don't update stats at test time
        env.training = False
        # Don't normalize reward at test time
        env.norm_reward = False
    
    # Load the trained model
    if "sac" in model_path.lower():
        model = SAC.load(model_path)
    else:
        model = PPO.load(model_path)
    
    # Setup visualization if render is enabled
    if render:
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Run evaluation episodes
    returns = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from the agent
            action, _ = model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Visualize particle filter if rendering
            if render:
                # Access the particle filter from the wrapped environment
                pf = env.particle_filter
                
                # Clear previous plot
                ax.clear()
                
                # Plot particles
                particles = pf.particles
                weights = pf.weights
                colors = plt.cm.viridis(weights / np.max(weights))
                
                ax.scatter(particles[:, 0], particles[:, 1], c=colors, s=15, alpha=0.6, label='Particles')
                
                # Plot ant position
                ant_pos = np.array([obs["obs"][0], obs["obs"][1]])
                ax.scatter(ant_pos[0], ant_pos[1], color='red', s=100, label='Ant')
                
                # Plot true opponent position
                true_pos = env.env.env.get_target_pos()
                ax.scatter(true_pos[0], true_pos[1], color='green', s=100, label='True Opponent')
                
                # Plot estimated position
                est_pos = pf.estimate()
                ax.scatter(est_pos[0], est_pos[1], color='blue', s=100, label='Estimated')
                
                # Draw visibility circle
                visibility = Circle(ant_pos, 3.0, fill=False, linestyle='--', color='blue', alpha=0.5)
                ax.add_patch(visibility)
                
                # Set plot limits and labels
                ax.set_xlim([-15, 15])
                ax.set_ylim([-15, 15])
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_title(f'Ant-Tag Evaluation - Episode {episode+1}')
                ax.legend()
                
                plt.draw()
                plt.pause(0.01)
            
            done = terminated or truncated
        
        returns.append(episode_reward)
        print(f"Episode {episode+1}: reward = {episode_reward:.2f}")
    
    if render:
        plt.ioff()
        plt.close()
    
    # Print evaluation summary
    print(f"Average return: {np.mean(returns):.2f} +/- {np.std(returns):.2f}")
    
    return np.mean(returns)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end RL with integrated Set Transformer for particle filtering")
    parser.add_argument("--num_particles", type=int, default=100, help="Number of particles for the filter")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"], help="RL algorithm to use")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a trained model")
    parser.add_argument("--model_path", type=str, help="Path to the trained model (for evaluation)")
    parser.add_argument("--vecnorm_path", type=str, help="Path to the VecNormalize stats (for evaluation)")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments for training")
    parser.add_argument("--save_path", type=str, default="models/sb3_e2e_agent", help="Where to save the trained model")
    
    args = parser.parse_args()
    
    if args.train:
        # Train a new end-to-end model
        model = train_end_to_end(
            num_particles=args.num_particles,
            total_timesteps=args.total_timesteps,
            algorithm=args.algorithm,
            n_envs=args.n_envs,
            save_path=args.save_path
        )
    
    if args.evaluate:
        # Check if model path is provided
        if not args.model_path:
            raise ValueError("Model path must be provided for evaluation")
        
        # Evaluate the agent
        evaluate_e2e_agent(
            model_path=args.model_path,
            vecnormalize_path=args.vecnorm_path,
            num_particles=args.num_particles
        ) 