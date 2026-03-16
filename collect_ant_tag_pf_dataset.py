"""
Collect a particle filter dataset from the Ant-Tag POMDP environment.

Uses a mix of random and pursuit (locomotion policy heading toward PF mean)
episodes to produce diverse particle distributions. Outputs a .npy file
of shape [num_samples, num_particles, 2] compatible with POMDPDataset.

Usage:
    python collect_ant_tag_pf_dataset.py \
        --num_trajectories 200 --timesteps 200 --num_particles 100 \
        --pursuit_fraction 0.5 \
        --locomotion_policy_path models/ant_locomotion_policy.zip \
        --output_file data/ant_tag_pf_dataset.npy
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from tqdm import tqdm

import pdomains  # noqa: F401 — registers pdomains-ant-tag-v0

# Use the clean AntTagParticleFilter from the proper package location
from particle_filters.ant_tag_pf import AntTagParticleFilter


def collect_dataset(
    num_trajectories: int,
    timesteps_per_trajectory: int,
    num_particles: int,
    pursuit_fraction: float,
    locomotion_policy_path: str | None,
    locomotion_vecnorm_path: str | None,
    seed: int,
) -> np.ndarray:
    """
    Collect PF particle snapshots from the Ant-Tag environment.

    Returns:
        Array of shape [num_samples, num_particles, 2].
    """
    np.random.seed(seed)

    # Load locomotion policy if provided
    locomotion_policy = None
    vecnorm_stats = None
    if locomotion_policy_path and os.path.exists(locomotion_policy_path):
        locomotion_policy = PPO.load(locomotion_policy_path)
        print(f"Loaded locomotion policy from {locomotion_policy_path}")
        if locomotion_vecnorm_path and os.path.exists(locomotion_vecnorm_path):
            # Load the VecNormalize stats so we can manually normalize obs
            import pickle
            with open(locomotion_vecnorm_path, "rb") as f:
                vecnorm_stats = pickle.load(f)
            print(f"Loaded VecNormalize stats from {locomotion_vecnorm_path}")
    elif pursuit_fraction > 0:
        print(
            "WARNING: pursuit_fraction > 0 but no locomotion policy provided. "
            "Falling back to random actions for all trajectories."
        )
        pursuit_fraction = 0.0

    env = gym.make("pdomains-ant-tag-v0", rendering=False)

    all_snapshots = []
    pursuit_count = 0
    random_count = 0

    for traj_idx in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        obs, _ = env.reset()

        # Initialize particle filter
        pf = AntTagParticleFilter(
            num_particles=num_particles,
            initial_env_obs=obs,
        )

        use_pursuit = (locomotion_policy is not None) and (
            np.random.random() < pursuit_fraction
        )
        if use_pursuit:
            pursuit_count += 1
        else:
            random_count += 1

        for t in range(timesteps_per_trajectory):
            # Choose action
            if use_pursuit:
                action = _pursuit_action(
                    obs, pf, locomotion_policy, vecnorm_stats
                )
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)

            # Determine visibility using ground truth (available during collection)
            ant_pos = next_obs[:2].copy()
            true_target = env.unwrapped.get_target_pos()
            visible = float(np.linalg.norm(ant_pos - true_target)) <= 3.0

            if visible:
                observed_target = next_obs[-2:].copy()  # FIXED: was obs[13:15]
            else:
                observed_target = np.array([np.nan, np.nan])

            # PF predict + update
            pf.predict(action, ant_current_pos_from_obs=ant_pos)
            pf.update(
                observed_target_pos=observed_target,
                ant_current_pos_from_obs=ant_pos,
            )

            # Store particle snapshot [num_particles, 2]
            all_snapshots.append(pf.particles.copy())

            obs = next_obs
            if terminated or truncated:
                break

    env.close()

    print(f"Trajectories — pursuit: {pursuit_count}, random: {random_count}")
    print(f"Total snapshots: {len(all_snapshots)}")

    return np.array(all_snapshots, dtype=np.float32)


def _pursuit_action(
    obs: np.ndarray,
    pf: AntTagParticleFilter,
    locomotion_policy: PPO,
    vecnorm_stats=None,
) -> np.ndarray:
    """
    Get an action from the locomotion policy that moves toward the PF mean.

    The locomotion policy was trained with full observability — it expects
    obs[-2:] to contain the target position. We substitute the PF mean so
    the policy moves toward the belief estimate.
    """
    pf_mean = pf.estimate_opponent_pos()

    # Build the observation the locomotion policy expects
    policy_obs = obs.copy()
    policy_obs[-2:] = pf_mean  # target position ← PF mean

    # If VecNormalize stats are available, manually normalize
    if vecnorm_stats is not None:
        policy_obs = _normalize_obs(policy_obs, vecnorm_stats)

    action, _ = locomotion_policy.predict(policy_obs, deterministic=False)
    return action


def _normalize_obs(obs: np.ndarray, vecnorm: VecNormalize) -> np.ndarray:
    """Manually normalize an observation using saved VecNormalize stats."""
    obs_mean = vecnorm.obs_rms.mean
    obs_var = vecnorm.obs_rms.var
    clip = vecnorm.clip_obs
    normalized = (obs - obs_mean) / np.sqrt(obs_var + vecnorm.epsilon)
    return np.clip(normalized, -clip, clip).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Collect particle filter dataset for Ant-Tag"
    )
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--pursuit_fraction", type=float, default=0.5)
    parser.add_argument("--locomotion_policy_path", type=str, default=None)
    parser.add_argument("--locomotion_vecnorm_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_file", type=str, default="data/ant_tag_pf_dataset.npy"
    )
    args = parser.parse_args()

    # Auto-detect vecnorm path if not specified
    if (
        args.locomotion_vecnorm_path is None
        and args.locomotion_policy_path is not None
    ):
        candidate = os.path.join(
            os.path.dirname(args.locomotion_policy_path),
            "locomotion_vecnorm.pkl",
        )
        if os.path.exists(candidate):
            args.locomotion_vecnorm_path = candidate
            print(f"Auto-detected VecNormalize stats: {candidate}")

    data = collect_dataset(
        num_trajectories=args.num_trajectories,
        timesteps_per_trajectory=args.timesteps,
        num_particles=args.num_particles,
        pursuit_fraction=args.pursuit_fraction,
        locomotion_policy_path=args.locomotion_policy_path,
        locomotion_vecnorm_path=args.locomotion_vecnorm_path,
        seed=args.seed,
    )

    print(f"Dataset shape: {data.shape}")
    print(
        f"Value range: x=[{data[:,:,0].min():.2f}, {data[:,:,0].max():.2f}], "
        f"y=[{data[:,:,1].min():.2f}, {data[:,:,1].max():.2f}]"
    )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, data)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
