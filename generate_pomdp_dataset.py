"""
Generate a dataset from the Car-Flag POMDP environment.

This script creates a dataset from the Car-Flag environment in the pomdp-domains
repository, collecting trajectories and saving them in a format compatible with
the Set Transformer model.

Usage:
    python generate_pomdp_dataset.py --num_trajectories 1000 --timesteps 40 --output_file dataset.npy
"""

import argparse
import gymnasium as gym
import json
import numpy as np
import os
from tqdm import tqdm
import sys
import random

def create_raw_data(num_trajectories, timesteps_per_trajectory, seed=42):
    """
    Create raw data from the Car-Flag POMDP environment.
    
    Args:
        num_trajectories: Number of trajectories to collect
        timesteps_per_trajectory: Maximum timesteps per trajectory
        seed: Random seed for reproducibility
        
    Returns:
        raw_data: Dictionary in the format expected by raw_to_numpy
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
    
    env = gym.make('pdomains-car-flag-v0', rendering=False)
    env.seed(seed)
    
    raw_data = {}
    
    for traj_idx in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        traj_key = f"traj_{traj_idx}"
        raw_data[traj_key] = {}
        
        obs, _ = env.reset()
        
        for t in range(timesteps_per_trajectory):
            # Take a random action
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Store observation in the raw data format
            t_key = f"t_{t}"
            raw_data[traj_key][t_key] = {
                "p_0": {"x": float(obs[0]), "y": float(obs[1])},  # position, velocity
                "p_1": {"x": float(obs[2]), "y": 0.0}             # direction indicator
            }
            
            obs = next_obs
            
            if terminated or truncated:
                break
                
    return raw_data

def raw_to_numpy(raw_data):
    """
    Convert raw data to numpy array format expected by Set Transformer.
    
    Args:
        raw_data: Dictionary with trajectory data
        
    Returns:
        numpy_array: Array of shape [num_samples, num_particles, num_variables]
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
                particle_features = [particle["x"], particle["y"]]
                sample.append(particle_features)
            samples.append(sample)
    
    # Convert to numpy array
    array = np.array(samples, dtype=np.float32)
    
    # Expected shape: (num_samples, num_particles, num_variables)
    if array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {array.shape}")
    
    return array

def save_data(data, output_file, raw_data):
    """
    Save the numpy array to a file.
    
    Args:
        data: Numpy array of shape [num_samples, num_particles, num_variables]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset from the Car-Flag POMDP environment")
    parser.add_argument("--num_trajectories", type=int, default=100, help="Number of trajectories to collect")
    parser.add_argument("--timesteps", type=int, default=40, help="Maximum timesteps per trajectory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_file", type=str, default="car_flag_dataset.npy", help="Output file path")
    
    args = parser.parse_args()
    
    # Create raw data
    print(f"Generating dataset with {args.num_trajectories} trajectories...")
    raw_data = create_raw_data(args.num_trajectories, args.timesteps, args.seed)
    
    # Convert to numpy
    print("Converting to numpy format...")
    numpy_data = raw_to_numpy(raw_data)
    print(f"Dataset shape: {numpy_data.shape}")
    
    # Save the data
    save_data(numpy_data, args.output_file, raw_data) 