"""
Reinforcement Learning with Pretrained Set Transformer for Particle Filter Processing

This script uses a pretrained Set Transformer model as a state processor for the
particle beliefs coming from the Ant-Tag environment. The processed state is then
fed to a Stable Baselines 3 RL agent.

Usage:
    python sb3_pretrained_processor.py --set_transformer_model models/set_transformer.pth --num_particles 100
"""

import argparse
import gymnasium as gym
import numpy as np
import os
import torch
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Import Set Transformer model
import sys
sys.path.append('.')  # Add the current directory to path
try:
    from set_transformer.models import SetTransformer
except ImportError:
    raise ImportError("Could not import SetTransformer. Make sure the set_transformer module is in your PYTHONPATH.")

# Import Stable Baselines 3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# Import our particle filter implementation
from particle_filter_ant_tag import AntTagParticleFilter

class SetTransformerFeatureExtractor:
    """
    Feature extractor using a pretrained Set Transformer to process particle filter beliefs
    """
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Load the pretrained model
        print(f"Loading Set Transformer model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Load model architecture and weights
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()  # Set to evaluation mode
        
        # Get model dimensions
        self.dim_input = self.model.dim_input
        self.dim_output = self.model.dim_output
        
        print(f"Model loaded: input dim={self.dim_input}, output dim={self.dim_output}")
    
    def extract_features(self, particles, weights):
        """
        Process particles through the Set Transformer model to get a fixed-size representation
        
        Args:
            particles: Particle positions [num_particles, 2]
            weights: Particle weights [num_particles]
            
        Returns:
            features: Processed features from Set Transformer
        """
        # Combine particles and weights
        if self.dim_input == 2:
            # Model expects only x, y coordinates
            particle_set = torch.tensor(particles, dtype=torch.float32).to(self.device)
        else:
            # Model expects x, y, weight
            particle_features = np.column_stack((particles, weights.reshape(-1, 1)))
            particle_set = torch.tensor(particle_features, dtype=torch.float32).to(self.device)
        
        # Add batch dimension if needed
        if len(particle_set.shape) == 2:
            particle_set = particle_set.unsqueeze(0)
        
        # Forward pass through model
        with torch.no_grad():
            features = self.model(particle_set)
        
        # Remove batch dimension if present
        if features.shape[0] == 1:
            features = features.squeeze(0)
            
        return features.cpu().numpy()

class ParticleFilterWrapper(gym.Wrapper):
    """
    Environment wrapper that integrates a particle filter and Set Transformer feature extractor
    """
    def __init__(self, env, feature_extractor, num_particles=100):
        super(ParticleFilterWrapper, self).__init__(env)
        self.feature_extractor = feature_extractor
        self.num_particles = num_particles
        self.particle_filter = None
        
        # Define the new observation space
        # Original observation plus processed features from Set Transformer
        orig_obs_dim = env.observation_space.shape[0]
        feature_dim = self.feature_extractor.dim_output
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(orig_obs_dim + feature_dim,)
        )
        
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
        # First predict step (before getting observation)
        self.particle_filter.predict(action)
        
        # Get the true opponent position (for visualization and evaluation only)
        true_opponent_pos = self.env.env.get_target_pos()
        
        # Check if opponent is visible
        ant_pos = np.array([next_obs[0], next_obs[1]])
        opponent_visible = np.linalg.norm(ant_pos - true_opponent_pos) <= 3.0
        
        # Get observed position (NaN if not visible)
        observed_opponent_pos = next_obs[13:15] if opponent_visible else np.array([np.nan, np.nan])
        
        # Update particle filter with observation
        self.particle_filter.update(observed_opponent_pos, ant_pos)
        
        # Process observation with particle filter and Set Transformer
        processed_obs = self._process_observation(next_obs)
        
        return processed_obs, reward, terminated, truncated, info
    
    def _process_observation(self, obs):
        """
        Process observation with particle filter and Set Transformer
        """
        # Get particles and weights from the filter
        particles = self.particle_filter.particles
        weights = self.particle_filter.weights
        
        # Extract features using Set Transformer
        features = self.feature_extractor.extract_features(particles, weights)
        
        # Concatenate original observation with extracted features
        processed_obs = np.concatenate([obs, features])
        
        return processed_obs

class ParticleFilterVisualizeCallback(BaseCallback):
    """Callback for visualizing particle filter during training"""
    
    def __init__(self, eval_env, feature_extractor, eval_freq=1000, verbose=1):
        super(ParticleFilterVisualizeCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.feature_extractor = feature_extractor
        self.eval_freq = eval_freq
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            self._visualize_particle_filter()
        return True
    
    def _visualize_particle_filter(self):
        # Run one episode with visualization
        obs, _ = self.eval_env.reset()
        done = False
        
        plt.figure(figsize=(8, 8))
        
        while not done:
            # Get action from the agent
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step the environment
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            
            # Visualize particle filter
            self._plot_particles()
            plt.pause(0.1)
            
            done = terminated or truncated
        
        plt.close()
    
    def _plot_particles(self):
        # Access the particle filter from the wrapped environment
        pf = self.eval_env.particle_filter
        
        # Clear previous plot
        plt.clf()
        
        # Plot particles
        particles = pf.particles
        weights = pf.weights
        colors = plt.cm.viridis(weights / np.max(weights))
        
        plt.scatter(particles[:, 0], particles[:, 1], c=colors, s=15, alpha=0.6, label='Particles')
        
        # Plot ant position (from observation)
        ant_pos = np.array([self.eval_env.env.env.get_body_com("torso")[0], 
                            self.eval_env.env.env.get_body_com("torso")[1]])
        plt.scatter(ant_pos[0], ant_pos[1], color='red', s=100, label='Ant')
        
        # Plot true opponent position
        true_pos = self.eval_env.env.env.get_target_pos()
        plt.scatter(true_pos[0], true_pos[1], color='green', s=100, label='True Opponent')
        
        # Plot estimated position
        est_pos = pf.estimate()
        plt.scatter(est_pos[0], est_pos[1], color='blue', s=100, label='Estimated')
        
        # Draw visibility circle
        visibility = Circle(ant_pos, 3.0, fill=False, linestyle='--', color='blue', alpha=0.5)
        plt.gca().add_patch(visibility)
        
        # Set plot limits and labels
        plt.xlim([-15, 15])
        plt.ylim([-15, 15])
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f'Ant-Tag Particle Filter - Step {self.n_calls}')
        plt.legend()

def train_rl_agent(
    set_transformer_path,
    num_particles=100,
    total_timesteps=1000000,
    algorithm="PPO",
    n_envs=4,
    save_path="models/sb3_pf_agent"
):
    """
    Train an RL agent using pretrained Set Transformer as feature extractor
    
    Args:
        set_transformer_path: Path to pretrained Set Transformer model
        num_particles: Number of particles for filter
        total_timesteps: Number of timesteps to train for
        algorithm: RL algorithm to use ("PPO" or "SAC")
        n_envs: Number of parallel environments
        save_path: Where to save the trained model
    """
    # Load the Set Transformer model
    feature_extractor = SetTransformerFeatureExtractor(set_transformer_path)
    
    # Create a function to initialize environment with wrapper
    def make_env():
        env = gym.make('pdomains-ant-tag-v0')
        env = ParticleFilterWrapper(env, feature_extractor, num_particles)
        env = Monitor(env)
        return env
    
    # Create vectorized environment
    vec_env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # Create a single environment for evaluation
    eval_env = gym.make('pdomains-ant-tag-v0')
    eval_env = ParticleFilterWrapper(eval_env, feature_extractor, num_particles)
    
    # Create directory for saving models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.dirname(save_path),
        name_prefix="sb3_pf_model"
    )
    
    # Visualization callback (commented out as it slows down training)
    # viz_callback = ParticleFilterVisualizeCallback(eval_env, feature_extractor)
    
    # Create and train the agent
    if algorithm.upper() == "PPO":
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./sb3_pf_tensorboard/")
    elif algorithm.upper() == "SAC":
        model = SAC("MlpPolicy", vec_env, verbose=1, tensorboard_log="./sb3_pf_tensorboard/")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save(save_path)
    vec_env.save(f"{save_path}_vecnormalize.pkl")
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model

def evaluate_and_visualize(
    model_path,
    set_transformer_path,
    vecnormalize_path=None,
    num_episodes=5,
    num_particles=100,
    render=True
):
    """
    Evaluate a trained agent and visualize the particle filter
    
    Args:
        model_path: Path to the trained Stable Baselines 3 model
        set_transformer_path: Path to the pretrained Set Transformer
        vecnormalize_path: Path to the saved VecNormalize stats
        num_episodes: Number of episodes to evaluate
        num_particles: Number of particles for the filter
        render: Whether to render the environment
    """
    # Load the Set Transformer model
    feature_extractor = SetTransformerFeatureExtractor(set_transformer_path)
    
    # Create environment with wrapper
    env = gym.make('pdomains-ant-tag-v0', rendering=render)
    env = ParticleFilterWrapper(env, feature_extractor, num_particles)
    
    # Load normalization stats if provided
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        env = VecNormalize.load(vecnormalize_path, env)
        # Don't update stats at test time
        env.training = False
        # Don't normalize reward at test time
        env.norm_reward = False
    
    # Determine model type from extension
    if model_path.endswith(".zip"):
        if "sac" in model_path.lower():
            model = SAC.load(model_path)
        else:
            model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown model type for {model_path}")
    
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
                
                # Plot ant position (from observation)
                ant_pos = np.array([obs[0], obs[1]])
                ax.scatter(ant_pos[0], ant_pos[1], color='red', s=100, label='Ant')
                
                # Get true opponent position
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
                ax.set_title(f'Ant-Tag Particle Filter - Episode {episode+1}')
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
    parser = argparse.ArgumentParser(description="Train an RL agent with a pretrained Set Transformer for particle filtering")
    parser.add_argument("--set_transformer_model", type=str, required=True, help="Path to pretrained Set Transformer model")
    parser.add_argument("--num_particles", type=int, default=100, help="Number of particles for the filter")
    parser.add_argument("--algorithm", type=str, default="PPO", choices=["PPO", "SAC"], help="RL algorithm to use")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate a trained model")
    parser.add_argument("--model_path", type=str, help="Path to the trained model (for evaluation)")
    parser.add_argument("--vecnorm_path", type=str, help="Path to the VecNormalize stats (for evaluation)")
    parser.add_argument("--total_timesteps", type=int, default=1000000, help="Total timesteps for training")
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments for training")
    parser.add_argument("--save_path", type=str, default="models/sb3_pf_agent", help="Where to save the trained model")
    
    args = parser.parse_args()
    
    # Check if Set Transformer model exists
    if not os.path.exists(args.set_transformer_model):
        raise FileNotFoundError(f"Set Transformer model {args.set_transformer_model} not found")
    
    if args.train:
        # Train a new agent
        model = train_rl_agent(
            set_transformer_path=args.set_transformer_model,
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
        evaluate_and_visualize(
            model_path=args.model_path,
            set_transformer_path=args.set_transformer_model,
            vecnormalize_path=args.vecnorm_path,
            num_particles=args.num_particles
        ) 