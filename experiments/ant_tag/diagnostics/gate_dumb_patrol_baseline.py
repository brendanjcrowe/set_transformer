"""
Zero-belief baseline: how much of the observed CGF/Gaussian success rates on
SmartAntTag is just generous detection geometry (visible_radius=3.0,
tag_radius=1.5 relative to the 9x9 arena) rather than genuine belief-guided
search?

Drives the ant with the Phase-2 locomotion policy (trained to walk toward
whatever point is in obs[-2:]) along a FIXED, static perimeter patrol route
that never looks at the particle filter, the true target, or anything at all
about where the target is. Measures real tag success in the actual env.

If this scores meaningfully high, most of any policy's success in this task
is attributable to coverage + generous detection radius, not to how well an
encoder represents the belief. If it scores near zero, genuine belief-guided
search is doing real work.

Usage:
    python3 diagnostics/gate_dumb_patrol_baseline.py --n_episodes 100
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401


def make_env(target_speed_scale, seed, env_id="pdomains-ant-tag-smart-v0"):
    def _init():
        env = gym.make(
            env_id, rendering=False,
            target_speed_scale=target_speed_scale,
        )
        return env
    return _init


def perimeter_waypoints(cage_max: float, n: int, radius_frac: float = 1.0) -> np.ndarray:
    """n waypoints evenly spaced around a square loop at radius_frac * cage_max
    (1.0 = hug the outer wall; smaller = a tighter inner loop)."""
    margin = cage_max * radius_frac - 0.1
    perim_points = []
    side = np.linspace(-margin, margin, max(2, n // 4), endpoint=False)
    for x in side:
        perim_points.append([x, margin])
    for y in side:
        perim_points.append([margin, -y])
    for x in side:
        perim_points.append([-x, -margin])
    for y in side:
        perim_points.append([-margin, y])
    return np.array(perim_points[:n])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--locomotion_model", type=str, default="models/ant_locomotion_policy.zip")
    parser.add_argument("--locomotion_vecnorm", type=str, default="models/locomotion_vecnorm.pkl")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--n_waypoints", type=int, default=16)
    parser.add_argument("--steps_per_leg", type=int, default=25)
    parser.add_argument("--reach_threshold", type=float, default=0.6)
    parser.add_argument("--radius_frac", type=float, default=1.0,
                         help="Patrol loop radius as a fraction of cage_max (1.0=hug outer wall).")
    parser.add_argument("--target_speed_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max_steps", type=int, default=400,
        help="Episode cap of --env_id; also the success convention (an\n             episode ending strictly before this many steps is a tag).\n             400 for v0/smart/ghost, 200 for pdomains-ant-tag-dens-v0.",
    )
    parser.add_argument(
        "--env_id", type=str, default="pdomains-ant-tag-smart-v0",
        help="Which AntTag variant to patrol on, e.g. "
             "'pdomains-ant-tag-ghost-v0'. The patrol never touches the PF "
             "or info, so ping-emitting envs need no other changes.",
    )
    args = parser.parse_args()

    vec_env = DummyVecEnv([make_env(args.target_speed_scale, args.seed, args.env_id)])
    vec_env = VecNormalize.load(args.locomotion_vecnorm, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(args.locomotion_model, device="cpu")

    raw_env = vec_env.envs[0].unwrapped
    cage_max = float(raw_env.cage_max_x)
    waypoints = perimeter_waypoints(cage_max, args.n_waypoints, args.radius_frac)
    print(f"Patrol route ({len(waypoints)} waypoints):\n{np.round(waypoints, 2)}")

    tagged = 0
    lengths = []

    for ep in range(args.n_episodes):
        vec_env.reset()
        wp_idx = 0
        steps_on_leg = 0
        ep_len = 0

        # Build the first observation with the waypoint injected, bypassing
        # VecNormalize's own reset-returned obs (which reveals the TRUE
        # target — we deliberately overwrite it with our fixed waypoint).
        raw_obs = np.concatenate([raw_env.data.qpos, raw_env.data.qvel, waypoints[wp_idx]]).astype(np.float32)
        obs = vec_env.normalize_obs(raw_obs[None, :])

        for t in range(args.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            _, _, done, info = vec_env.step(action)
            ep_len += 1

            # done[0] fires on BOTH a real tag (terminated) and the
            # timeout (truncated) — the old VecEnv API doesn't distinguish
            # them. Use ep_len < max_steps (same convention as every eval
            # script in this investigation) to tell a tag from a timeout.
            if done[0]:
                break

            ant_pos = raw_env.data.qpos[:2]
            dist_to_wp = np.linalg.norm(ant_pos - waypoints[wp_idx])
            steps_on_leg += 1
            if dist_to_wp < args.reach_threshold or steps_on_leg >= args.steps_per_leg:
                wp_idx = (wp_idx + 1) % len(waypoints)
                steps_on_leg = 0

            raw_obs = np.concatenate([raw_env.data.qpos, raw_env.data.qvel, waypoints[wp_idx]]).astype(np.float32)
            obs = vec_env.normalize_obs(raw_obs[None, :])

        if ep_len < args.max_steps:
            tagged += 1
        lengths.append(ep_len)

    lengths = np.array(lengths)
    print(f"\n=== Zero-belief fixed-perimeter-patrol baseline over {args.n_episodes} episodes ===")
    print(f"Success rate  : {tagged}/{args.n_episodes} ({100*tagged/args.n_episodes:.1f}%)")
    print(f"Mean length   : {lengths.mean():.1f}")
    print(f"Median length : {np.median(lengths):.1f}")
    if tagged > 0:
        tag_lens = lengths[lengths < args.max_steps]
        print(f"When tagged   : mean_len={tag_lens.mean():.1f}, median_len={np.median(tag_lens):.1f}")


if __name__ == "__main__":
    main()
