"""
Generate a dataset from the CarEnv POMDP environment with particle filtering.

This script creates a dataset from the CarEnv environment, using a
particle filter to track the direction of "heaven", which is not directly
observable. The dataset is saved in a format compatible with the
Set Transformer model.

Usage:
    python particle_filter_car_env.py --num_trajectories 50 \\
        --timesteps 100 --output_file car_env_pf_dataset.npy
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from filterpy.monte_carlo import systematic_resample
from tqdm import tqdm

# It is assumed that the script is run from the root of the project.
# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from environments.car_env import CarEnv


def create_pf_dataset(
    num_trajectories,
    timesteps_per_trajectory,
    num_particles=100,
    seed=42,
    max_particles_to_store=None,
):
    """
    Create dataset from the CarEnv POMDP environment.

    This uses a particle filter.

    Args:
        num_trajectories: Number of trajectories to collect
        timesteps_per_trajectory: Maximum timesteps per trajectory
        num_particles: Number of particles for the filter
        seed: Random seed for reproducibility
        max_particles_to_store: Maximum number of particles to store
                              (None means store all particles)

    Returns:
        raw_data: Dictionary in the format for the Set Transformer
    """
    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Initialize environment
    env = CarEnv(seed=seed)

    raw_data = {}
    total_error = 0
    total_steps = 0

    for traj_idx in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        traj_key = f"traj_{traj_idx}"
        raw_data[traj_key] = {}

        # Reset environment and get initial observation
        obs, _ = env.reset(seed=seed + traj_idx)

        # Initialize particle filter
        particle_filter = CarEnvParticleFilter(num_particles)

        for t in range(timesteps_per_trajectory):
            # Take a random action
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            # The revealed direction is the last element of the observation
            revealed_direction = next_obs[2]

            # PF predict step (is a no-op for this static hidden state)
            particle_filter.predict()

            # Particle filter update step
            particle_filter.update(revealed_direction)

            # Get estimated direction from particle filter
            estimated_direction = particle_filter.estimate()

            # Get ground truth direction
            true_direction = 1.0 if env.heaven_position > 0 else -1.0

            # Calculate error for evaluation
            error = abs(estimated_direction - true_direction)
            total_error += error
            total_steps += 1

            # Store the data
            t_key = f"t_{t}"
            raw_data[traj_key][t_key] = {
                # Car position (x) and velocity (y) with a dummy weight
                "p_0": {"x": float(obs[0]), "y": float(obs[1]), "weight": 1.0},
                # Revealed direction, embedded in x, with a dummy weight
                "p_1": {"x": float(revealed_direction), "y": 0.0, "weight": 1.0},
                # Particle filter estimate, embedded in x, with a dummy weight
                "p_2": {"x": float(estimated_direction), "y": 0.0, "weight": 1.0},
                # Ground truth direction, embedded in x, with a dummy weight
                "p_3": {"x": float(true_direction), "y": 0.0, "weight": 1.0},
            }

            # Store all particles and their weights
            all_particles = particle_filter.get_all_particles()

            # Limit the number of particles to store if specified
            if max_particles_to_store is not None:
                all_particles = all_particles[:max_particles_to_store]

            for i, (particle, weight) in enumerate(all_particles):
                particle_idx = i + 4  # Start from p_4
                raw_data[traj_key][t_key][f"p_{particle_idx}"] = {
                    "x": float(particle),
                    "y": 0.0,
                    "weight": float(weight),
                }

            # Update for next step
            obs = next_obs

            if terminated or truncated:
                break

    avg_error = total_error / max(total_steps, 1)

    print("Particle Filter Evaluation:")
    print(f"  Average absolute error of belief: {avg_error:.4f}")

    return raw_data


class CarEnvParticleFilter:
    """
    Particle filter for the CarEnv environment.

    This filter tracks the belief over the direction of "heaven", which can be
    either -1 (left) or 1 (right).
    """

    def __init__(self, num_particles):
        """
        Initialize the particle filter.

        Args:
            num_particles: Number of particles to use.
        """
        self.num_particles = num_particles

        # Initialize half particles to -1 and half to 1
        self.particles = np.ones(num_particles)
        self.particles[: num_particles // 2] = -1
        np.random.shuffle(self.particles)

        # Equal initial weights
        self.weights = np.ones(num_particles) / num_particles

        # Likelihood for a correct observation
        self.correct_likelihood = 0.99
        # Likelihood for an incorrect observation (to avoid zero weights)
        self.incorrect_likelihood = 1 - self.correct_likelihood

    def predict(self):
        """
        Predict step. The hidden state (heaven direction) is static.
        """
        pass

    def update(self, revealed_direction):
        """
        Update particle weights based on the observation.

        Args:
            revealed_direction: The direction of heaven (-1 or 1) if revealed,
                              otherwise 0.
        """
        # Only update weights if the direction was revealed
        if revealed_direction != 0:
            # Calculate likelihood for each particle
            likelihood = np.full_like(self.weights, self.incorrect_likelihood)
            likelihood[self.particles == revealed_direction] = self.correct_likelihood

            # Update weights
            self.weights *= likelihood

            # Normalize weights
            if np.sum(self.weights) > 0:
                self.weights /= np.sum(self.weights)
            else:
                # Reset weights in the degenerate case
                self.weights = np.ones(self.num_particles) / self.num_particles

            # Resample if effective sample size is too low
            n_eff = 1.0 / np.sum(np.square(self.weights))
            if n_eff < self.num_particles / 2:
                indices = systematic_resample(self.weights)
                self.particles = self.particles[indices]
                self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """
        Get the estimated direction (belief) of heaven.

        Returns:
            A float between -1 and 1 representing the belief.
            -1: confident heaven is left.
             1: confident heaven is right.
             0: uncertain.
        """
        return np.sum(self.particles * self.weights)

    def get_all_particles(self):
        """
        Get all particles with their weights.

        Returns:
            A list of (particle, weight) tuples.
        """
        return list(zip(self.particles, self.weights))


def _sort_key(p_key):
    return int(p_key.split("_")[1])


def raw_to_numpy(raw_data):
    """
    Convert raw data to numpy array format expected by Set Transformer.

    Args:
        raw_data: Dictionary with trajectory data

    Returns:
        numpy_array: Array of shape [num_samples, num_elements, num_features]
    """
    if not raw_data:
        raise ValueError("Raw data is empty")

    samples = []
    for traj in raw_data.values():
        for timestep in traj.values():
            sample = []
            # Sort by key to maintain order
            for p_key in sorted(timestep.keys(), key=_sort_key):
                element = timestep[p_key]
                # All elements are expected to have x, y, and weight
                features = [element["x"], element["y"], element["weight"]]
                sample.append(features)
            samples.append(sample)

    # Pad samples to the same length
    max_len = max(len(s) for s in samples)
    padded_samples = np.zeros((len(samples), max_len, 3), dtype=np.float32)

    for i, sample in enumerate(samples):
        padded_samples[i, : len(sample), :] = np.array(sample)

    return padded_samples


def save_data(data, output_file, raw_data):
    """
    Save the numpy array to a file and split into train/test sets.

    Args:
        data: Numpy array of shape [num_samples, num_elements, num_features]
        output_file: Path to save the data
        raw_data: Dictionary with raw trajectory data for reference
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(output_file, data)
    print(f"Dataset saved to {output_file}")

    # Split into training and evaluation sets (80/20)
    test_file = output_file.replace(".npy", ".eval.npy")
    train_file = output_file.replace(".npy", ".train.npy")

    n_samples = len(data)
    n_test = int(0.2 * n_samples)
    indices = np.random.permutation(n_samples)

    test_indices, train_indices = indices[:n_test], indices[n_test:]
    test_data, train_data = data[test_indices], data[train_indices]

    np.save(test_file, test_data)
    np.save(train_file, train_data)

    print(f"Test set ({len(test_data)} samples) saved to {test_file}")
    print(f"Training set ({len(train_data)} samples) saved to {train_file}")

    # Save raw data for reference
    raw_file = output_file.replace(".npy", ".json")
    with open(raw_file, "w") as f:
        json.dump(raw_data, f, indent=4)
    print(f"Raw data saved to {raw_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dataset from the CarEnv POMDP "
        "environment with particle filtering."
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=50,
        help="Number of trajectories to collect",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="Maximum timesteps per trajectory"
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=100,
        help="Number of particles for the filter",
    )
    parser.add_argument(
        "--max_particles_to_store",
        type=int,
        default=None,
        help="Maximum number of particles to store (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_file",
        type=str,
        default="car_env_pf_dataset.npy",
        help="Output file path",
    )

    args = parser.parse_args()

    print(f"Generating dataset with {args.num_trajectories} trajectories...")
    print(f"Using {args.num_particles} particles for filtering.")
    max_p_msg = (
        "all"
        if args.max_particles_to_store is None
        else str(args.max_particles_to_store)
    )
    print(f"Storing {max_p_msg} particles in the dataset.")

    raw_data = create_pf_dataset(
        args.num_trajectories,
        args.timesteps,
        args.num_particles,
        args.seed,
        args.max_particles_to_store,
    )

    print("Converting to numpy format...")
    numpy_data = raw_to_numpy(raw_data)
    print(f"Dataset shape: {numpy_data.shape}")

    save_data(numpy_data, args.output_file, raw_data)
