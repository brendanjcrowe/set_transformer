"""
Is the SmartAntTag particle-filter belief actually multimodal in practice?

Drives the ant with a trained policy through real episodes (full predict +
update + resample cycle, exactly as during training/eval), detects every
visible->invisible transition, and snapshots the particle cloud at several
step-counts after the target is lost. For each snapshot, quantifies
multimodality without any extra dependencies (no sklearn) via:

  1. Distance-threshold clustering (union-find): how many spatially distinct
     clusters do the particles form?
  2. A between/within variance ratio (like a one-way ANOVA F-ratio) computed
     from those clusters: how much of the total spread is *between* clusters
     vs *within* them? A single Gaussian blob has a ratio near 0 (no
     meaningful between-cluster separation); genuine multimodality drives it
     up sharply.

Also saves a scatter-plot figure of representative snapshots for visual
inspection.

Usage:
    python3 analyze_belief_multimodality.py \
      --model_path runs/ant_tag_cgf_smart/<run>/models/best_model/best_model.zip \
      --vecnormalize_path runs/ant_tag_cgf_smart/<run>/models/vecnormalize.pkl \
      --n_episodes 30
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import pdomains  # noqa: F401
from set_transformer.rl.particle_filters.ant_tag import SmartAntTagParticleFilter

_train_rl_cgf = importlib.import_module("4_train_rl_cgf")
CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
PFDictWithWeightsObservationWrapper = _train_rl_cgf.PFDictWithWeightsObservationWrapper
_CurriculumRouter = _train_rl_cgf._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs


def make_env(num_particles, seed, target_speed_scale):
    def _init():
        env = gym.make(
            "pdomains-ant-tag-smart-v0", rendering=False,
            target_speed_scale=target_speed_scale,
        )
        env.reset(seed=seed)
        pf_kwargs = get_ant_tag_pf_kwargs(env)
        env = CurriculumVisibilityWrapper(env, initial_visibility_radius=3.0)
        env = PFDictWithWeightsObservationWrapper(
            env=env,
            particle_filter_class=SmartAntTagParticleFilter,
            particle_filter_kwargs=pf_kwargs,
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=[-2, -1],
        )
        env = _CurriculumRouter(env)
        return env
    return _init


def cluster_by_distance(points: np.ndarray, threshold: float) -> np.ndarray:
    """Union-find clustering: points within `threshold` of each other (single-
    link) share a cluster. Returns a label array. O(n^2), fine for n=100."""
    n = len(points)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] < threshold:
                union(i, j)

    roots = [find(i) for i in range(n)]
    unique_roots = {r: idx for idx, r in enumerate(sorted(set(roots)))}
    return np.array([unique_roots[r] for r in roots])


def weight_near_mean_fraction(points: np.ndarray, weights: np.ndarray, radius: float) -> float:
    """Fraction of total particle WEIGHT within `radius` of the weighted mean.

    Directly tests what a pursuit policy following "go to the reported mean"
    would actually find: if the belief is a tight blob, this is high (most
    mass is near the mean); if it's a wall-hugging/curved/multimodal shape,
    this is low (the mean sits in a low-density gap, most mass is elsewhere).
    """
    mean = np.average(points, weights=weights, axis=0)
    dists = np.linalg.norm(points - mean, axis=1)
    return float(weights[dists < radius].sum() / weights.sum())


def between_within_ratio(points: np.ndarray, weights: np.ndarray, labels: np.ndarray) -> float:
    """Weighted between-cluster / within-cluster variance ratio (both axes
    combined). ~0 for a single unimodal blob; large when clusters are well
    separated relative to their own spread."""
    overall_mean = np.average(points, weights=weights, axis=0)
    total_var = np.average(np.sum((points - overall_mean) ** 2, axis=1), weights=weights)

    within = 0.0
    between = 0.0
    for lbl in np.unique(labels):
        mask = labels == lbl
        w = weights[mask]
        w_sum = w.sum()
        if w_sum <= 0:
            continue
        cluster_mean = np.average(points[mask], weights=w, axis=0)
        cluster_var = np.average(np.sum((points[mask] - cluster_mean) ** 2, axis=1), weights=w) if mask.sum() > 1 else 0.0
        within += w_sum * cluster_var
        between += w_sum * np.sum((cluster_mean - overall_mean) ** 2)
    within /= weights.sum()
    between /= weights.sum()
    if within < 1e-8:
        return float("inf") if between > 1e-8 else 0.0
    return float(between / within)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vecnormalize_path", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=30)
    parser.add_argument("--target_speed_scale", type=float, default=0.0)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--cluster_threshold", type=float, default=0.2)
    parser.add_argument("--near_mean_radius", type=float, default=1.0)
    parser.add_argument("--snapshot_steps", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out_prefix", type=str, default="belief_multimodality")
    args = parser.parse_args()

    env_fn = make_env(args.num_particles, args.seed, args.target_speed_scale)
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize.load(args.vecnormalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(args.model_path, device="cpu")

    base_env = vec_env.envs[0]  # PFDictWithWeightsObservationWrapper (via _CurriculumRouter)
    pf_wrapper = base_env
    while not isinstance(pf_wrapper, PFDictWithWeightsObservationWrapper):
        pf_wrapper = pf_wrapper.env

    snapshots = {k: [] for k in args.snapshot_steps}  # steps_since_lost -> list of (particles, weights)
    fig_examples = []  # a few full trajectories of snapshots for plotting

    for ep in range(args.n_episodes):
        obs = vec_env.reset()
        unwrapped = pf_wrapper.unwrapped
        was_visible = True
        steps_since_lost = None
        ep_snapshots = {}
        for t in range(400):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            ant_pos = unwrapped.data.qpos[:2]
            target_pos = unwrapped.get_target_pos()
            dist = np.linalg.norm(ant_pos - target_pos)
            visible = dist < unwrapped.visible_radius

            if was_visible and not visible:
                steps_since_lost = 0
            elif not visible and steps_since_lost is not None:
                steps_since_lost += 1
            elif visible:
                steps_since_lost = None
            was_visible = visible

            if steps_since_lost is not None and steps_since_lost in snapshots:
                particles = pf_wrapper.particle_filter.particles.copy()
                weights = pf_wrapper.particle_filter.weights.copy()
                snapshots[steps_since_lost].append((particles, weights))
                ep_snapshots[steps_since_lost] = (particles.copy(), weights.copy(), ant_pos.copy(), target_pos.copy())

            if done[0]:
                break

        if len(ep_snapshots) >= 3 and len(fig_examples) < 4:
            fig_examples.append(ep_snapshots)

    print(f"Collected snapshots per steps-since-lost bucket: "
          f"{ {k: len(v) for k, v in snapshots.items()} }")

    print("\n=== Multimodality analysis (distance threshold = "
          f"{args.cluster_threshold}) ===")
    for k in args.snapshot_steps:
        entries = snapshots[k]
        if not entries:
            print(f"steps_since_lost={k}: no snapshots collected")
            continue
        n_clusters_list = []
        bw_ratios = []
        near_mean_fracs = []
        for particles, weights in entries:
            labels = cluster_by_distance(particles, args.cluster_threshold)
            n_clusters_list.append(len(np.unique(labels)))
            bw_ratios.append(between_within_ratio(particles, weights, labels))
            near_mean_fracs.append(weight_near_mean_fraction(particles, weights, args.near_mean_radius))
        n_clusters_arr = np.array(n_clusters_list)
        bw_arr = np.array(bw_ratios)
        near_mean_arr = np.array(near_mean_fracs)
        pct_multimodal = float((n_clusters_arr > 1).mean()) * 100
        print(
            f"steps_since_lost={k:3d}  n={len(entries):3d}  "
            f"clusters: mean={n_clusters_arr.mean():.2f} median={np.median(n_clusters_arr):.0f} "
            f"max={n_clusters_arr.max()}  "
            f"%snapshots with >1 cluster={pct_multimodal:.0f}%  "
            f"between/within ratio: mean={np.nanmean(bw_arr[np.isfinite(bw_arr)]):.2f} "
            f"median={np.nanmedian(bw_arr[np.isfinite(bw_arr)]):.2f}  "
            f"weight within {args.near_mean_radius} of mean: mean={near_mean_arr.mean()*100:.1f}% "
            f"median={np.median(near_mean_arr)*100:.1f}%"
        )

    # Visualization
    if fig_examples:
        n_rows = len(fig_examples)
        n_cols = len(args.snapshot_steps)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
        for row, ep_snap in enumerate(fig_examples):
            for col, k in enumerate(args.snapshot_steps):
                ax = axes[row][col]
                if k in ep_snap:
                    particles, weights, ant_pos, target_pos = ep_snap[k]
                    sizes = 200 * weights / (weights.max() + 1e-12)
                    ax.scatter(particles[:, 0], particles[:, 1], s=sizes, alpha=0.5, c="tab:blue")
                    ax.scatter(*ant_pos, marker="^", c="black", s=80, label="ant")
                    ax.scatter(*target_pos, marker="*", c="red", s=120, label="true target")

                    # Overlay what the Gaussian encoder sees: weighted mean + 2-sigma
                    # covariance ellipse, to show how well (or badly) that ellipse
                    # actually covers where the particle mass really is.
                    mean = np.average(particles, weights=weights, axis=0)
                    centered = particles - mean
                    cov = (weights[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(axis=0) / weights.sum()
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    eigvals = np.clip(eigvals, 1e-6, None)
                    angle = np.degrees(np.arctan2(eigvecs[1, np.argmax(eigvals)], eigvecs[0, np.argmax(eigvals)]))
                    width, height = 2 * 2 * np.sqrt(np.sort(eigvals)[::-1])  # 2-sigma, largest eig first
                    ellipse = matplotlib.patches.Ellipse(
                        mean, width, height, angle=angle,
                        fill=False, edgecolor="green", linewidth=2, label="Gaussian encoder's implied 2σ region",
                    )
                    ax.add_patch(ellipse)
                    ax.scatter(*mean, marker="x", c="green", s=100, label="Gaussian encoder's mean")

                    ax.set_xlim(-5, 5)
                    ax.set_ylim(-5, 5)
                    ax.set_aspect("equal")
                if row == 0:
                    ax.set_title(f"steps_since_lost={k}")
                if col == 0:
                    ax.set_ylabel(f"episode {row}")
        axes[0][0].legend(loc="upper left", fontsize=6)
        fig.tight_layout()
        out_path = f"{args.out_prefix}.png"
        fig.savefig(out_path, dpi=120)
        print(f"\nSaved visualization to {out_path}")


if __name__ == "__main__":
    main()
