"""
Generate a dataset from the BoxEnv POMDP environment with particle filtering.

This script creates a dataset from the BoxEnv environment, using a
particle filter to track the configuration of the boxes, which is not directly
observable. The dataset is saved in a format compatible with the
Set Transformer model.

Usage:
    python particle_filter_box_env.py --num_trajectories 50 \\
        --timesteps 100 --output_file box_env_pf_dataset.npy
"""

import argparse
import json
import os
import random
import sys

import numpy as np
from filterpy.monte_carlo import systematic_resample
from tqdm import tqdm

from environments.box_env import BoxEnv

# It is assumed that the script is run from the root of the project.
# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def create_pf_dataset(
    num_trajectories,
    timesteps_per_trajectory,
    num_particles=100,
    seed=42,
    max_particles_to_store=None,
):
    """
    Create dataset from the BoxEnv POMDP environment.

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
    env = BoxEnv(seed=seed)

    raw_data = {}
    total_error = 0
    total_steps = 0

    for traj_idx in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        traj_key = f"traj_{traj_idx}"
        raw_data[traj_key] = {}

        # Reset environment
        env.reset(seed=seed + traj_idx)

        # Initialize particle filter
        particle_filter = BoxEnvParticleFilter(num_particles)

        for t in range(timesteps_per_trajectory):
            # Take a random action
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)

            # Observation is (gripper_x, finger_theta)
            gripper_x, finger_theta = next_obs[0], next_obs[1]

            # PF predict step (is a no-op for this static hidden state)
            particle_filter.predict()

            # Particle filter update step
            particle_filter.update(gripper_x, finger_theta)

            # Get estimated direction from particle filter
            estimated_direction = particle_filter.estimate()

            # Get ground truth direction
            true_direction = 1.0 if env.go_to_left else -1.0

            # Calculate error for evaluation
            # We care about the sign, so if estimate is >0 and truth is >0,
            # error is low.
            error = (estimated_direction - true_direction) ** 2
            total_error += error
            total_steps += 1

            # Store the data
            t_key = f"t_{t}"
            raw_data[traj_key][t_key] = {
                # Gripper position (x) and finger sensor (y)
                "p_0": {"x": float(gripper_x), "y": float(finger_theta), "weight": 1.0},
                # Particle filter estimate of goal direction
                "p_1": {"x": float(estimated_direction), "y": 0.0, "weight": 1.0},
                # Ground truth direction
                "p_2": {"x": float(true_direction), "y": 0.0, "weight": 1.0},
            }

            # Store all particles and their weights
            all_particles = particle_filter.get_all_particles()

            # Limit the number of particles to store if specified
            if max_particles_to_store is not None:
                all_particles = all_particles[:max_particles_to_store]

            for i, (particle, weight) in enumerate(all_particles):
                particle_idx = i + 3  # Start from p_3
                # Particle state: [left_box, right_box], 0=small, 1=big
                p_key = f"p_{particle_idx}"
                raw_data[traj_key][t_key][p_key] = {
                    "x": float(particle[0]),
                    "y": float(particle[1]),
                    "weight": float(weight),
                }

            if terminated or truncated:
                break

    avg_error = total_error / max(total_steps, 1)

    print("Particle Filter Evaluation:")
    print(f"  Average squared error of belief: {avg_error:.4f}")

    return raw_data


class BoxEnvParticleFilter:
    """
    Particle filter for the BoxEnv environment.

    This filter tracks the belief over the four possible box configurations.
    State representation: [left_box_type, right_box_type]
    where 0 = small, 1 = big.
    The four states are: [0,0], [0,1], [1,0], [1,1]
    """

    def __init__(self, num_particles):
        self.num_particles = num_particles

        # Initialize particles, 1/4 for each of the 4 states
        states = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.particles = np.array([states[i % 4] for i in range(num_particles)])
        np.random.shuffle(self.particles)

        self.weights = np.ones(num_particles) / num_particles

        # Observation model parameters
        self.theta_threshold = 0.5  # Assumed threshold to distinguish boxes
        self.box_1_x = 40.0 / 100.0  # Normalized box positions
        self.box_2_x = 60.0 / 100.0
        self.pos_threshold = 0.05  # Normalized distance
        self.correct_likelihood = 0.95
        self.incorrect_likelihood = (1 - self.correct_likelihood) / 3

    def predict(self):
        pass

    def update(self, gripper_x, finger_theta):
        if finger_theta == 0:
            return  # No new information

        observed_size = 1 if finger_theta > self.theta_threshold else 0

        # Check which box is being probed
        if abs(gripper_x - self.box_1_x) < self.pos_threshold:
            probed_box_idx = 0  # Probing the left box
        elif abs(gripper_x - self.box_2_x) < self.pos_threshold:
            probed_box_idx = 1  # Probing the right box
        else:
            return  # Not close enough to a box

        likelihood = np.full(self.num_particles, self.incorrect_likelihood)
        # Increase likelihood for particles matching the observation
        matches = self.particles[:, probed_box_idx] == observed_size
        likelihood[matches] = self.correct_likelihood

        self.weights *= likelihood

        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights.fill(1.0 / self.num_particles)

        # Resample
        neff = 1.0 / np.sum(np.square(self.weights))
        if neff < self.num_particles / 2:
            indices = systematic_resample(self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        """
        Estimate the goal direction (-1 for right, 1 for left).
        - Go left if state is [0,1] or [1,0].
        - Go right if state is [0,0] or [1,1].
        """
        prob_go_left = 0.0
        for i, p in enumerate(self.particles):
            # If sum is 1, it's a mixed pair (sb or bs), so go left
            if sum(p) == 1:
                prob_go_left += self.weights[i]

        # Convert probability to a value between -1 and 1
        # P(left) * 1 + P(right) * -1 = P(left) * 1 + (1-P(left)) * -1
        # = 2 * P(left) - 1
        return 2 * prob_go_left - 1

    def get_all_particles(self):
        return list(zip(self.particles.tolist(), self.weights))


def _sort_key(p_key):
    return int(p_key.split("_")[1])


def raw_to_numpy(raw_data):
    if not raw_data:
        raise ValueError("Raw data is empty")

    samples = []
    for traj in raw_data.values():
        for timestep in traj.values():
            sample = []
            for p_key in sorted(timestep.keys(), key=_sort_key):
                element = timestep[p_key]
                features = [element["x"], element["y"], element.get("weight", 1.0)]
                sample.append(features)
            samples.append(sample)

    max_len = max(len(s) for s in samples)
    padded_samples = np.zeros((len(samples), max_len, 3), dtype=np.float32)
    for i, sample in enumerate(samples):
        padded_samples[i, : len(sample), :] = np.array(sample)

    return padded_samples


def save_data(data, output_file, raw_data):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(output_file, data)
    print(f"Dataset saved to {output_file}")

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

    raw_file = output_file.replace(".npy", ".json")
    with open(raw_file, "w") as f:
        json.dump(raw_data, f, indent=4)
    print(f"Raw data saved to {raw_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a dataset from the BoxEnv POMDP environment."
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
        default="box_env_pf_dataset.npy",
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
