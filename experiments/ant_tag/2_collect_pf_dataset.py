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

from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter


def collect_dataset(
    num_trajectories: int,
    timesteps_per_trajectory: int,
    num_particles: int,
    pursuit_fraction: float,
    fully_observed_fraction: float,
    visibility_radius_range: tuple[float, float],
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
    fully_observed_count = 0

    for traj_idx in tqdm(range(num_trajectories), desc="Collecting trajectories"):
        obs, _ = env.reset()

        # Initialize particle filter
        pf = AntTagParticleFilter(
            num_particles=num_particles,
            initial_env_obs=obs,
        )

        # Decide trajectory type: fully_observed, pursuit, or random
        roll = np.random.random()
        if roll < fully_observed_fraction:
            traj_type = "fully_observed"
            fully_observed_count += 1
        elif (locomotion_policy is not None) and (
            roll < fully_observed_fraction + pursuit_fraction
        ):
            traj_type = "pursuit"
            pursuit_count += 1
        else:
            traj_type = "random"
            random_count += 1

        # For pursuit trajectories, sample a visibility radius from the range
        # to simulate different levels of observability (as in curriculum)
        traj_vis_radius = np.random.uniform(*visibility_radius_range)

        for t in range(timesteps_per_trajectory):
            # Choose action
            if traj_type in ("pursuit", "fully_observed"):
                if locomotion_policy is not None:
                    action = _pursuit_action(
                        obs, pf, locomotion_policy, vecnorm_stats
                    )
                else:
                    action = env.action_space.sample()
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)

            # Determine visibility using ground truth (available during collection)
            ant_pos = next_obs[:2].copy()
            true_target = env.unwrapped.get_target_pos()

            if traj_type == "fully_observed":
                # Always visible — particles collapse onto true target
                visible = True
            else:
                visible = float(np.linalg.norm(ant_pos - true_target)) <= traj_vis_radius

            if visible:
                observed_target = true_target.copy()
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

    print(f"Trajectories — fully_observed: {fully_observed_count}, pursuit: {pursuit_count}, random: {random_count}")
    print(f"Total snapshots (raw): {len(all_snapshots)}")

    data = np.array(all_snapshots, dtype=np.float32)
    data = _rebalance_by_spread(data)
    return data


def _rebalance_by_spread(
    data: np.ndarray,
    collapsed_frac: float = 0.30,
    intermediate_frac: float = 0.40,
    diffuse_frac: float = 0.30,
    collapsed_threshold: float = 0.5,
    diffuse_threshold: float = 4.0,
) -> np.ndarray:
    """Rebalance dataset so intermediate-spread samples are well represented.

    Downsamples over-represented buckets and upsamples (with replacement)
    under-represented ones to hit the target fractions.
    """
    spreads = data.std(axis=1).mean(axis=1)

    collapsed_mask = spreads < collapsed_threshold
    intermediate_mask = (spreads >= collapsed_threshold) & (spreads < diffuse_threshold)
    diffuse_mask = spreads >= diffuse_threshold

    collapsed_idx = np.where(collapsed_mask)[0]
    intermediate_idx = np.where(intermediate_mask)[0]
    diffuse_idx = np.where(diffuse_mask)[0]

    print(f"Pre-rebalance: collapsed={len(collapsed_idx)} ({len(collapsed_idx)/len(data)*100:.1f}%), "
          f"intermediate={len(intermediate_idx)} ({len(intermediate_idx)/len(data)*100:.1f}%), "
          f"diffuse={len(diffuse_idx)} ({len(diffuse_idx)/len(data)*100:.1f}%)")

    # Target size: use the total that keeps the dataset roughly the same size
    # as the largest bucket that needs no upsampling
    n_total = len(data)
    n_collapsed = int(n_total * collapsed_frac)
    n_intermediate = int(n_total * intermediate_frac)
    n_diffuse = int(n_total * diffuse_frac)

    rng = np.random.default_rng(42)

    # Sample each bucket (with replacement if upsampling needed)
    def _sample(idx, n_target):
        if len(idx) == 0:
            print(f"  WARNING: bucket is empty, cannot sample {n_target}")
            return np.array([], dtype=int)
        replace = len(idx) < n_target
        return rng.choice(idx, size=n_target, replace=replace)

    sampled_collapsed = _sample(collapsed_idx, n_collapsed)
    sampled_intermediate = _sample(intermediate_idx, n_intermediate)
    sampled_diffuse = _sample(diffuse_idx, n_diffuse)

    all_idx = np.concatenate([sampled_collapsed, sampled_intermediate, sampled_diffuse])
    rng.shuffle(all_idx)

    result = data[all_idx]
    spreads_out = result.std(axis=1).mean(axis=1)
    print(f"Post-rebalance: {len(result)} samples — "
          f"collapsed={int((spreads_out < collapsed_threshold).sum())} ({(spreads_out < collapsed_threshold).mean()*100:.1f}%), "
          f"intermediate={int(((spreads_out >= collapsed_threshold) & (spreads_out < diffuse_threshold)).sum())} "
          f"({((spreads_out >= collapsed_threshold) & (spreads_out < diffuse_threshold)).mean()*100:.1f}%), "
          f"diffuse={int((spreads_out >= diffuse_threshold).sum())} ({(spreads_out >= diffuse_threshold).mean()*100:.1f}%)")

    return result


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
    parser.add_argument(
        "--fully_observed_fraction", type=float, default=0.2,
        help="Fraction of trajectories with full visibility (particles collapse to target)"
    )
    parser.add_argument(
        "--visibility_radius_min", type=float, default=3.0,
        help="Min visibility radius for pursuit trajectories"
    )
    parser.add_argument(
        "--visibility_radius_max", type=float, default=15.0,
        help="Max visibility radius for pursuit trajectories"
    )
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
        fully_observed_fraction=args.fully_observed_fraction,
        visibility_radius_range=(args.visibility_radius_min, args.visibility_radius_max),
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
