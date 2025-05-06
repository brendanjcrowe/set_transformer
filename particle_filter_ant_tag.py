"""
Generate a dataset from the Ant-Tag POMDP environment with particle filtering.

This script creates a dataset from the Ant-Tag environment in the pomdp-domains
repository, using a particle filter to track the opponent's position even when 
not directly observable. The dataset is saved in a format compatible with the 
Set Transformer model.

Usage:
    python particle_filter_ant_tag.py --num_trajectories 50 --timesteps 100 --output_file ant_tag_pf_dataset.npy
"""

import argparse
import gymnasium as gym
import json
import numpy as np
import os
from tqdm import tqdm
import sys
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

# Import filterpy for particle filtering
from filterpy.monte_carlo import systematic_resample
from filterpy.monte_carlo import stratified_resample

def create_pf_dataset(num_trajectories, timesteps_per_trajectory, num_particles=100, 
                     seed=42, visualize=False, max_particles_to_store=None):
    """
    Create dataset from the Ant-Tag POMDP environment using a particle filter.
    
    Args:
        num_trajectories: Number of trajectories to collect
        timesteps_per_trajectory: Maximum timesteps per trajectory
        num_particles: Number of particles for the filter
        seed: Random seed for reproducibility
        visualize: Whether to visualize the particle filter
        max_particles_to_store: Maximum number of particles to store in the dataset
                              (None means store all particles)
        
    Returns:
        raw_data: Dictionary in the format for the Set Transformer
    """
    # Register and load the environment
    try:
        import pdomains
    except ImportError:
        print("pdomains package not found. Make sure you've installed the pomdp-domains repository.")
        sys.exit(1)
        
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize environment
    env = gym.make('pdomains-ant-tag-v0', rendering=False)
    env.seed(seed)
    
    raw_data = {}
    total_visible_error = 0
    total_hidden_error = 0
    total_visible_steps = 0
    total_hidden_steps = 0
    
    # Set up visualization if enabled
    if visualize:
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(8, 8))
        particle_scatter = None
        ant_scatter = None
        opponent_scatter = None
        estimate_scatter = None
        visibility_circle = None
        
    for traj_idx in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        traj_key = f"traj_{traj_idx}"
        raw_data[traj_key] = {}
        
        # Reset environment and get initial observation
        obs, _ = env.reset()
        
        # Initialize particle filter
        particle_filter = AntTagParticleFilter(num_particles, obs)
        
        for t in range(timesteps_per_trajectory):
            # Take a random action
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Update particle filter with new observation
            particle_filter.predict(action)
            
            # Get the true opponent position from the environment state
            # (This would not be available in a real POMDP scenario, but we use it for evaluation)
            true_opponent_pos = env.env.env.get_target_pos()
            
            # Check if opponent is visible (within observation radius)
            ant_pos = np.array([obs[0], obs[1]])
            opponent_visible = np.linalg.norm(ant_pos - true_opponent_pos) <= 3.0
            
            # If opponent is visible, use the observation; otherwise it's NaN
            observed_opponent_pos = next_obs[13:15] if opponent_visible else np.array([np.nan, np.nan])
            
            # Update particle filter with observation
            particle_filter.update(observed_opponent_pos, ant_pos)
            
            # Get estimated position from particle filter
            estimated_pos = particle_filter.estimate()
            
            # Calculate error for evaluation
            error = np.linalg.norm(estimated_pos - true_opponent_pos)
            if opponent_visible:
                total_visible_error += error
                total_visible_steps += 1
            else:
                total_hidden_error += error
                total_hidden_steps += 1
            
            # Store the data
            t_key = f"t_{t}"
            raw_data[traj_key][t_key] = {
                # Ant position and velocity (from observation)
                "p_0": {"x": float(obs[0]), "y": float(obs[1])},
                
                # Observed opponent position (NaN if not visible)
                "p_1": {"x": float(observed_opponent_pos[0]), "y": float(observed_opponent_pos[1])},
                
                # Particle filter estimate
                "p_2": {"x": float(estimated_pos[0]), "y": float(estimated_pos[1])},
                
                # Ground truth position (for evaluation)
                "p_3": {"x": float(true_opponent_pos[0]), "y": float(true_opponent_pos[1])}
            }
            
            # Store all particles and their weights (or a subset if max_particles_to_store is specified)
            all_particles = particle_filter.get_all_particles()
            
            # Limit the number of particles to store if specified
            if max_particles_to_store is not None:
                all_particles = all_particles[:max_particles_to_store]
                
            for i, (particle, weight) in enumerate(all_particles):
                particle_idx = i + 4  # Start from p_4
                raw_data[traj_key][t_key][f"p_{particle_idx}"] = {
                    "x": float(particle[0]), 
                    "y": float(particle[1]),
                    "weight": float(weight)
                }
            
            # Visualize if enabled
            if visualize and (traj_idx < 3 or traj_idx == num_trajectories-1):
                # Clear previous plots
                ax.clear()
                
                # Plot particles with color based on weight
                particles = particle_filter.particles
                weights = particle_filter.weights
                colors = plt.cm.viridis(weights / np.max(weights))
                
                ax.scatter(particles[:, 0], particles[:, 1], c=colors, s=15, alpha=0.6, label='Particles')
                
                # Plot the ant position
                ax.scatter(ant_pos[0], ant_pos[1], color='red', s=100, label='Ant')
                
                # Plot true opponent position
                ax.scatter(true_opponent_pos[0], true_opponent_pos[1], color='green', s=100, label='True Opponent')
                
                # Plot estimated position
                ax.scatter(estimated_pos[0], estimated_pos[1], color='blue', s=100, label='Estimated')
                
                # Draw visibility circle
                visibility = Circle(ant_pos, 3.0, fill=False, linestyle='--', color='blue', alpha=0.5)
                ax.add_patch(visibility)
                
                # Set axis limits and labels
                ax.set_xlim([-15, 15])
                ax.set_ylim([-15, 15])
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_title(f'Ant-Tag Particle Filter (Trajectory {traj_idx+1}, Step {t+1})')
                ax.legend()
                
                plt.draw()
                plt.pause(0.1)
            
            # Update for next step
            obs = next_obs
            
            if terminated or truncated:
                break
                
    # Calculate average errors
    avg_visible_error = total_visible_error / max(total_visible_steps, 1)
    avg_hidden_error = total_hidden_error / max(total_hidden_steps, 1)
    
    print(f"Particle Filter Evaluation:")
    print(f"  Average error when opponent visible: {avg_visible_error:.4f}")
    print(f"  Average error when opponent hidden: {avg_hidden_error:.4f}")
    print(f"  Visible steps: {total_visible_steps}, Hidden steps: {total_hidden_steps}")
    
    # Close visualization
    if visualize:
        plt.ioff()
        plt.close()
                
    return raw_data

class AntTagParticleFilter:
    """
    Particle filter implementation for the Ant-Tag environment using filterpy.
    
    This filter tracks the position of the opponent in the Ant-Tag environment,
    maintaining a belief state even when the opponent is not directly observable.
    """
    
    def __init__(self, num_particles, initial_obs, initial_std=5.0):
        """
        Initialize the particle filter.
        
        Args:
            num_particles: Number of particles to use
            initial_obs: Initial observation from the environment
            initial_std: Standard deviation for initial particle distribution
        """
        # Number of particles
        self.num_particles = num_particles
        
        # Initialize particles uniformly in the arena
        self.particles = np.random.uniform(-10, 10, (num_particles, 2))
        
        # Equal initial weights
        self.weights = np.ones(num_particles) / num_particles
        
        # Standard deviation of the observation noise
        self.obs_std = 0.1
        
        # Standard deviation of the process noise
        self.process_std = 0.2
        
        # Avoidance factor (how strongly the opponent avoids the ant)
        self.avoidance_factor = 0.8
        
        # Maximum avoidance distance
        self.max_avoidance_dist = 7.0
        
        # Visibility radius
        self.visibility_radius = 3.0
    
    def predict(self, action):
        """
        Predict step: Move particles according to the opponent's movement model.
        
        In Ant-Tag, the opponent tries to avoid the ant by moving away from it.
        
        Args:
            action: Action taken by the ant (not directly used, but could be in a more complex model)
        """
        # Add random motion to all particles (process noise)
        self.particles += np.random.normal(0, self.process_std, self.particles.shape)
        
        # Ensure particles stay within bounds (-10, 10)
        self.particles = np.clip(self.particles, -10, 10)
    
    def update(self, observed_position, ant_position):
        """
        Update step: Adjust particle weights based on the observation.
        
        Args:
            observed_position: Observed position of the opponent (NaN if not visible)
            ant_position: Current position of the ant
        """
        # If the opponent is visible (no NaN values)
        if not np.isnan(observed_position[0]):
            # Calculate likelihood for each particle
            dist = np.linalg.norm(self.particles - observed_position, axis=1)
            likelihood = np.exp(-0.5 * (dist / self.obs_std)**2)
            
            # Update weights
            self.weights *= likelihood
        
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            # If all weights are zero (degenerate case), reset weights
            self.weights = np.ones(self.num_particles) / self.num_particles
        
        # Check for effective sample size and resample if needed
        n_eff = 1.0 / np.sum(np.square(self.weights))
        
        if n_eff < self.num_particles / 2:
            # Use filterpy's systematic resampling
            indices = systematic_resample(self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
            
            # Add some jitter to avoid particle collapse
            self.particles += np.random.normal(0, self.process_std * 0.5, self.particles.shape)
        
        # Apply opponent's avoidance behavior
        # Calculate distance from each particle to ant
        for i in range(self.num_particles):
            particle_to_ant = ant_position - self.particles[i]
            distance = np.linalg.norm(particle_to_ant)
            
            # Apply avoidance only within a certain radius
            if distance < self.max_avoidance_dist:
                # Direction away from ant
                avoidance_dir = -particle_to_ant / max(distance, 1e-6)
                
                # Strength decreases with distance
                avoidance_strength = self.avoidance_factor * (1 - distance / self.max_avoidance_dist)
                
                # Apply avoidance movement
                self.particles[i] += avoidance_dir * avoidance_strength
        
        # Ensure particles stay within bounds after avoidance
        self.particles = np.clip(self.particles, -10, 10)
    
    def estimate(self):
        """
        Get the estimated position of the opponent (weighted average of particles).
        
        Returns:
            estimated_position: 2D array with x, y coordinates
        """
        return np.average(self.particles, weights=self.weights, axis=0)
    
    def get_all_particles(self):
        """
        Get all particles with their weights.
        
        Returns:
            all_particles: List of (particle, weight) tuples for all particles
        """
        all_particles = [(self.particles[i], self.weights[i]) for i in range(self.num_particles)]
        
        # Sort by weight for consistency (highest weight first)
        all_particles.sort(key=lambda x: x[1], reverse=True)
        
        return all_particles

def raw_to_numpy(raw_data):
    """
    Convert raw data to numpy array format expected by Set Transformer.
    
    Args:
        raw_data: Dictionary with trajectory data
        
    Returns:
        numpy_array: Array of shape [num_samples, num_particles, num_features]
    """
    if not raw_data:
        raise ValueError("Raw data is empty")
    
    # Collect all samples
    samples = []
    for traj in raw_data.values():
        for timestep in traj.values():
            # Convert timestep to sample
            sample = []
            for particle_idx in sorted(timestep.keys()):
                particle = timestep[particle_idx]
                # Extract features (x, y) and weight if available
                if "weight" in particle:
                    particle_features = [particle["x"], particle["y"], particle["weight"]]
                else:
                    particle_features = [particle["x"], particle["y"]]
                sample.append(particle_features)
            samples.append(sample)
    
    # Convert to numpy array
    array = np.array(samples, dtype=np.float32)
    
    # Expected shape: (num_samples, num_particles, num_features)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")
    
    return array

def save_data(data, output_file, raw_data):
    """
    Save the numpy array to a file.
    
    Args:
        data: Numpy array of shape [num_samples, num_particles, num_features]
        output_file: Path to save the data
        raw_data: Dictionary with raw trajectory data for reference
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save the array
    np.save(output_file, data)
    print(f"Dataset saved to {output_file}")
    
    # Save a small portion as a separate test set (20%)
    test_file = output_file.replace('.npy', '.eval.npy')
    train_file = output_file.replace('.npy', '.train.npy')
    
    # Split the data
    n_samples = len(data)
    n_test = int(0.2 * n_samples)
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    test_data = data[test_indices]
    train_data = data[train_indices]
    
    np.save(test_file, test_data)
    np.save(train_file, train_data)
    
    print(f"Test set ({len(test_data)} samples) saved to {test_file}")
    print(f"Training set ({len(train_data)} samples) saved to {train_file}")
    
    # Also save the raw data as JSON for reference
    raw_file = output_file.replace('.npy', '.json')
    with open(raw_file, 'w') as f:
        json.dump(raw_data, f)
    print(f"Raw data saved to {raw_file}")

def evaluate_pf_accuracy(raw_data):
    """
    Evaluate the accuracy of the particle filter estimates.
    
    Args:
        raw_data: Dictionary with trajectory data
        
    Returns:
        mse: Mean squared error of the estimates
    """
    errors = []
    visible_errors = []
    hidden_errors = []
    
    for traj in raw_data.values():
        for timestep in traj.values():
            # True position
            true_pos = np.array([timestep["p_3"]["x"], timestep["p_3"]["y"]])
            
            # Estimated position
            est_pos = np.array([timestep["p_2"]["x"], timestep["p_2"]["y"]])
            
            # Observed position (to check if opponent is visible)
            obs_pos = np.array([timestep["p_1"]["x"], timestep["p_1"]["y"]])
            
            # Calculate error
            error = np.linalg.norm(true_pos - est_pos)
            errors.append(error)
            
            # Check if opponent was visible
            if not np.isnan(obs_pos[0]):
                visible_errors.append(error)
            else:
                hidden_errors.append(error)
    
    # Calculate mean square errors
    mse = np.mean(np.square(errors))
    visible_mse = np.mean(np.square(visible_errors)) if visible_errors else 0
    hidden_mse = np.mean(np.square(hidden_errors)) if hidden_errors else 0
    
    print(f"Particle Filter MSE: {mse:.4f}")
    print(f"  When opponent visible: {visible_mse:.4f} (n={len(visible_errors)})")
    print(f"  When opponent hidden: {hidden_mse:.4f} (n={len(hidden_errors)})")
    
    return mse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset from the Ant-Tag POMDP environment with particle filtering")
    parser.add_argument("--num_trajectories", type=int, default=50, help="Number of trajectories to collect")
    parser.add_argument("--timesteps", type=int, default=100, help="Maximum timesteps per trajectory")
    parser.add_argument("--num_particles", type=int, default=100, help="Number of particles for the filter")
    parser.add_argument("--max_particles_to_store", type=int, default=None, help="Maximum number of particles to store (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="ant_tag_pf_dataset.npy", help="Output file path")
    parser.add_argument("--visualize", action="store_true", help="Visualize the particle filter")
    
    args = parser.parse_args()
    
    # Create dataset with particle filtering
    print(f"Generating dataset with {args.num_trajectories} trajectories...")
    print(f"Using {args.num_particles} particles for filtering")
    max_particles_msg = "all" if args.max_particles_to_store is None else str(args.max_particles_to_store)
    print(f"Storing {max_particles_msg} particles in the dataset")
    
    raw_data = create_pf_dataset(
        args.num_trajectories, 
        args.timesteps, 
        args.num_particles,
        args.seed,
        args.visualize,
        args.max_particles_to_store
    )
    
    # Convert to numpy
    print("Converting to numpy format...")
    numpy_data = raw_to_numpy(raw_data)
    print(f"Dataset shape: {numpy_data.shape}")
    
    # Evaluate particle filter accuracy
    print("Evaluating particle filter accuracy...")
    evaluate_pf_accuracy(raw_data)
    
    # Save the data
    save_data(numpy_data, args.output_file, raw_data) 