"""
Collect a particle-filter dataset from an Ant-Tag POMDP environment.

Step 2 of the pipeline. Produces the belief snapshots that 3_train_st.py
pretrains the Set Transformer on.

GENERIC OVER ENV + FILTER. Nothing here is tied to one env variant. Pass
--variant (see variants.py) to select the env and its matching filter
together. The rollout reuses 4_train_rl_cgf.py's
make_ant_tag_cgf_env, so the belief distribution written to disk is produced
by exactly the same particle filter, interaction mapper and visibility
wrapper that will run at RL time. A hand-rolled predict/update loop (what
this script used to do) silently drifts from the RL-time filter and, worse,
cannot supply the per-episode live parameters that the den-based filters need
through the mapper.

WEIGHTS ARE STORED. Output is an .npz with

    particles      [num_samples, num_particles, dim]  float32, RAW env coords
    weights        [num_samples, num_particles]       float32, PF weights
    particle_scale scalar: the arena half-width the RL extractors divide by
    metadata       JSON string: env id, filter, CLI args, git provenance

3_train_st.py uses the weights by default, putting the mass in the
reconstruction loss's target measure rather than in the ground metric. The
legacy .npy (particles only) is still readable by the training script; it
simply trains unweighted.

Usage:
    python3 2_collect_pf_dataset.py --variant cdens_terminal \
        --num_trajectories 300 --timesteps 300 --num_particles 100 \
        --locomotion_policy_path models/ant_locomotion_policy.zip

    python3 2_collect_pf_dataset.py --list_variants
"""

import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
    
import pdomains  # noqa: F401,E402 - registers the pdomains-ant-tag-* envs
import variants  # noqa: E402 - env/filter registry
from set_transformer.rl import particle_filters as _pf_package  # noqa: E402
from set_transformer.rl.particle_filters import ant_tag as ant_tag_filters  # noqa: E402
from set_transformer.rl.particle_filters.base import BaseParticleFilter  # noqa: E402

# The RL pipeline's env factory. Sibling module name starts with a digit, so
# importlib is required.
_train_rl_cgf = importlib.import_module("4_train_rl_cgf")
make_ant_tag_belief_env = _train_rl_cgf.make_ant_tag_cgf_env
get_ant_tag_arena_scale = _train_rl_cgf.get_ant_tag_arena_scale
_git_provenance = _train_rl_cgf._git_provenance
AntTagParticleFilter = _train_rl_cgf.AntTagParticleFilter

#: Visibility radius that makes the target permanently visible. Any value far
#: past the arena diagonal works; CurriculumVisibilityWrapper compares a
#: distance against it.
_ALWAYS_VISIBLE_RADIUS = 1e6


def resolve_particle_filter(name: str) -> type:
    """Look up a particle filter class by name.

    Only names exported by set_transformer.rl.particle_filters.ant_tag are
    accepted, so a typo fails here with the list of valid options rather than
    at the first PF update.
    """
    def _is_concrete_filter(obj) -> bool:
        return (isinstance(obj, type) and issubclass(obj, BaseParticleFilter)
                and obj is not BaseParticleFilter)

    candidate = getattr(ant_tag_filters, name, None)
    if not _is_concrete_filter(candidate):
        available = sorted(
            attr for attr in dir(ant_tag_filters)
            if _is_concrete_filter(getattr(ant_tag_filters, attr))
        )
        raise ValueError(
            f"Unknown particle filter {name!r}. Available: {available}"
        )
    return candidate


def _find_particle_filter(env):
    """Walk the wrapper chain for the live particle filter instance."""
    current = env
    while current is not None:
        pf = getattr(current, "particle_filter", None)
        if pf is not None:
            return pf
        current = getattr(current, "env", None)
    raise RuntimeError("No particle_filter found in the wrapper chain")


def _normalize_obs(obs: np.ndarray, vecnorm: VecNormalize) -> np.ndarray:
    """Manually normalize an observation using saved VecNormalize stats."""
    obs_mean = vecnorm.obs_rms.mean
    obs_var = vecnorm.obs_rms.var
    clip = vecnorm.clip_obs
    normalized = (obs - obs_mean) / np.sqrt(obs_var + vecnorm.epsilon)
    return np.clip(normalized, -clip, clip).astype(np.float32)


def _pursuit_action(
    base_obs: np.ndarray,
    particle_filter,
    locomotion_policy: PPO,
    vecnorm_stats=None,
) -> np.ndarray:
    """Action from the locomotion policy, aimed at the PF's belief mean.

    The locomotion policy was trained fully observed and expects obs[-2:] to
    hold the target position, so the belief mean is substituted there. This is
    what makes the collected beliefs cover states an actual pursuer reaches,
    instead of only those a random walk stumbles into.
    """
    policy_obs = base_obs.copy()
    policy_obs[-2:] = particle_filter.estimate_opponent_pos()
    if vecnorm_stats is not None:
        policy_obs = _normalize_obs(policy_obs, vecnorm_stats)
    action, _ = locomotion_policy.predict(policy_obs, deterministic=False)
    return action


def _weighted_spread(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-sample belief spread: weighted std per coordinate, averaged.

    Weighted rather than plain, because the spread of a particle CLOUD and the
    spread of the BELIEF it represents diverge whenever the weights are far
    from uniform — which is exactly the regime the alarm/negative-information
    updates create.
    """
    w = np.clip(weights, 0.0, None)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    w3 = w[:, :, None]
    mean = (w3 * particles).sum(axis=1, keepdims=True)
    var = (w3 * (particles - mean) ** 2).sum(axis=1)
    return np.sqrt(np.clip(var, 0.0, None)).mean(axis=1)


def _rebalance_by_spread(
    particles: np.ndarray,
    weights: np.ndarray,
    collapsed_frac: float = 0.30,
    intermediate_frac: float = 0.40,
    diffuse_frac: float = 0.30,
    collapsed_threshold: float = 0.5,
    diffuse_threshold: float = 4.0,
    seed: int = 42,
    upsample: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebalance so intermediate-spread beliefs are well represented.

    Downsamples over-represented buckets. With ``upsample=True`` (the
    historical behaviour) under-represented buckets are upsampled WITH
    REPLACEMENT to their quota, so the output keeps its size -- at the cost of
    duplicated rows, which then land on both sides of the train/val split and
    make checkpoint selection optimistic (PITFALLS.md section 4; the
    cdens_terminal dataset duplicated its intermediate bucket 11x this way).
    With ``upsample=False`` an under-represented bucket keeps every row it has
    and nothing is duplicated, so the dataset shrinks instead: collect more
    raw snapshots (--max_snapshots) to compensate.

    Thresholds are in ENV COORDINATE UNITS and so are problem-specific; expose
    them on the CLI rather than assuming the Ant-Tag arena.
    """
    spreads = _weighted_spread(particles, weights)

    collapsed_idx = np.where(spreads < collapsed_threshold)[0]
    intermediate_idx = np.where(
        (spreads >= collapsed_threshold) & (spreads < diffuse_threshold))[0]
    diffuse_idx = np.where(spreads >= diffuse_threshold)[0]

    n_total = len(particles)
    print(f"Pre-rebalance: collapsed={len(collapsed_idx)} "
          f"({len(collapsed_idx)/n_total*100:.1f}%), "
          f"intermediate={len(intermediate_idx)} "
          f"({len(intermediate_idx)/n_total*100:.1f}%), "
          f"diffuse={len(diffuse_idx)} ({len(diffuse_idx)/n_total*100:.1f}%)")

    rng = np.random.default_rng(seed)

    buckets = [
        ("collapsed", collapsed_idx, collapsed_frac),
        ("intermediate", intermediate_idx, intermediate_frac),
        ("diffuse", diffuse_idx, diffuse_frac),
    ]
    # An empty bucket cannot be sampled from. Its quota is redistributed over
    # the buckets that do have data, in proportion to their own quotas, so the
    # output keeps its size. Dropping the quota instead would silently discard
    # a chunk of the dataset whenever the spread distribution is concentrated
    # — e.g. every sample intermediate gives back only intermediate_frac of
    # the data, with nothing but a one-line warning to say so.
    live = [b for b in buckets if len(b[1]) > 0]
    if not live:
        raise ValueError("No samples to rebalance")
    empty_frac = sum(frac for name, idx, frac in buckets if len(idx) == 0)
    if empty_frac > 0:
        empty_names = [name for name, idx, _ in buckets if len(idx) == 0]
        print(f"  WARNING: no samples in bucket(s) {empty_names}; their share "
              "is redistributed over the remaining buckets to preserve the "
              "dataset size. Consider adjusting --collapsed_threshold / "
              "--diffuse_threshold for this env's coordinate scale.")
    live_frac_total = sum(frac for _, _, frac in live)

    sampled = []
    for name, idx, frac in live:
        share = frac / live_frac_total if live_frac_total > 0 else 1.0 / len(live)
        n_target = int(round(n_total * share))
        if not upsample:
            n_target = min(n_target, len(idx))
        sampled.append(rng.choice(idx, size=n_target, replace=len(idx) < n_target))
    all_idx = np.concatenate(sampled)
    rng.shuffle(all_idx)

    particles, weights = particles[all_idx], weights[all_idx]
    spreads_out = _weighted_spread(particles, weights)
    print(f"Post-rebalance: {len(particles)} samples — "
          f"collapsed={int((spreads_out < collapsed_threshold).sum())}, "
          f"intermediate={int(((spreads_out >= collapsed_threshold) & (spreads_out < diffuse_threshold)).sum())}, "
          f"diffuse={int((spreads_out >= diffuse_threshold).sum())}")
    return particles, weights


def collect_dataset(
    env_id: str,
    particle_filter_class: type,
    num_trajectories: int,
    timesteps_per_trajectory: int,
    num_particles: int,
    pursuit_fraction: float,
    fully_observed_fraction: float,
    visibility_radius_range: tuple[float, float],
    locomotion_policy_path: str | None,
    locomotion_vecnorm_path: str | None,
    seed: int,
    evasion_scale: float = 1.0,
    target_speed_scale: float | None = None,
    max_snapshots: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the RL-time belief env and record every belief it produces.

    Returns:
        (particles [S, num_particles, dim], weights [S, num_particles]).
    """
    rng = np.random.default_rng(seed)

    locomotion_policy = None
    vecnorm_stats = None
    if locomotion_policy_path and os.path.exists(locomotion_policy_path):
        locomotion_policy = PPO.load(locomotion_policy_path)
        print(f"Loaded locomotion policy from {locomotion_policy_path}")
        if locomotion_vecnorm_path and os.path.exists(locomotion_vecnorm_path):
            with open(locomotion_vecnorm_path, "rb") as handle:
                vecnorm_stats = pickle.load(handle)
            print(f"Loaded VecNormalize stats from {locomotion_vecnorm_path}")
    elif pursuit_fraction > 0 or fully_observed_fraction > 0:
        print("WARNING: no locomotion policy provided; pursuit and "
              "fully-observed trajectories fall back to random actions.")

    # The same factory the RL scripts use: same PF, same interaction mapper,
    # same visibility wrapper. Reward shaping off — nothing here reads reward.
    env = make_ant_tag_belief_env(
        num_particles=num_particles,
        rank=0,
        seed=seed,
        monitor_dir=None,
        initial_visibility_radius=visibility_radius_range[1],
        obs_mask_indices=None,   # the pursuit policy needs the real base obs
        apply_reward_shaping=False,
        env_id=env_id,
        particle_filter_class=particle_filter_class,
        target_speed_scale=target_speed_scale,
    )()
    env.set_evasion_scale(evasion_scale)

    all_particles: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    counts = {"fully_observed": 0, "pursuit": 0, "random": 0}

    def _record(obs_dict):
        all_particles.append(np.asarray(obs_dict["particles"], dtype=np.float32))
        all_weights.append(np.asarray(obs_dict["weights"], dtype=np.float32))

    progress = tqdm(range(num_trajectories), desc="Collecting trajectories")
    for _ in progress:
        roll = rng.random()
        if roll < fully_observed_fraction:
            traj_type = "fully_observed"
        elif (locomotion_policy is not None
              and roll < fully_observed_fraction + pursuit_fraction):
            traj_type = "pursuit"
        else:
            traj_type = "random"
        counts[traj_type] += 1

        # Visibility is set through the SAME curriculum knob RL uses, so a
        # collected belief is always one the RL agent could actually hold.
        radius = (_ALWAYS_VISIBLE_RADIUS if traj_type == "fully_observed"
                  else float(rng.uniform(*visibility_radius_range)))
        env.set_curriculum_radius(radius)

        obs, _ = env.reset()
        _record(obs)   # the prior is a legitimate belief state

        for _step in range(timesteps_per_trajectory):
            if traj_type in ("pursuit", "fully_observed") and locomotion_policy is not None:
                action = _pursuit_action(
                    obs["obs"], _find_particle_filter(env),
                    locomotion_policy, vecnorm_stats,
                )
            else:
                action = env.action_space.sample()

            obs, _reward, terminated, truncated, _info = env.step(action)
            _record(obs)

            if terminated or truncated:
                break

        if max_snapshots is not None and len(all_particles) >= max_snapshots:
            print(f"\nReached --max_snapshots ({max_snapshots}); stopping early.")
            break
        progress.set_postfix(snapshots=len(all_particles))

    env.close()

    print(f"Trajectories — " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    print(f"Total snapshots (raw): {len(all_particles)}")

    particles = np.asarray(all_particles, dtype=np.float32)
    weights = np.asarray(all_weights, dtype=np.float32)
    if max_snapshots is not None:
        particles, weights = particles[:max_snapshots], weights[:max_snapshots]
    return particles, weights


def main():
    parser = argparse.ArgumentParser(
        description="Collect a particle-filter belief dataset. --variant "
                    "picks the env and its matching filter together."
    )
    variants.add_variant_argument(parser)
    parser.add_argument(
        "--env_id", type=str, default=None,
        help="Override the variant's env id. Rarely needed; the pairing with "
             "the particle filter is the thing that must not drift.",
    )
    parser.add_argument(
        "--particle_filter", type=str, default=None,
        help="Override the variant's particle filter, by class name from "
             "set_transformer.rl.particle_filters.ant_tag. It MUST match the "
             "env's target motion model, or the stored beliefs are wrong.",
    )
    parser.add_argument("--num_trajectories", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--pursuit_fraction", type=float, default=0.5)
    parser.add_argument(
        "--fully_observed_fraction", type=float, default=0.2,
        help="Fraction of trajectories with full visibility (particles "
             "collapse onto the target)",
    )
    parser.add_argument("--visibility_radius_min", type=float, default=3.0)
    parser.add_argument("--visibility_radius_max", type=float, default=15.0)
    parser.add_argument(
        "--evasion_scale", type=float, default=1.0,
        help="SmartAntTag-family evasion strength during collection "
             "(0=dumb target, 1=full). No-op on envs without the knob.",
    )
    parser.add_argument(
        "--target_speed_scale", type=float, default=None,
        help="SmartAntTagEnv only; omit to use the env default.",
    )
    parser.add_argument("--locomotion_policy_path", type=str, default=None)
    parser.add_argument("--locomotion_vecnorm_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_snapshots", type=int, default=None,
        help="Stop once this many belief snapshots are collected.",
    )

    parser.add_argument(
        "--no_rebalance", action="store_true",
        help="Skip spread rebalancing and keep the raw visit distribution.",
    )
    parser.add_argument(
        "--rebalance_no_upsample", action="store_true",
        help="Rebalance by downsampling only: never duplicate rows to fill an "
             "under-represented bucket (PITFALLS.md section 4). The dataset "
             "shrinks instead; raise --max_snapshots to compensate.",
    )
    parser.add_argument(
        "--collapsed_threshold", type=float, default=0.5,
        help="Belief spread below this counts as collapsed. ENV UNITS.",
    )
    parser.add_argument(
        "--diffuse_threshold", type=float, default=4.0,
        help="Belief spread above this counts as diffuse. ENV UNITS.",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Defaults to data/<variant>_pf_dataset.npz",
    )
    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    variant = variants.resolve(args.variant)
    resolved_env_id = args.env_id or variant.env_id
    resolved_pf = (resolve_particle_filter(args.particle_filter)
                   if args.particle_filter else variant.particle_filter)
    resolved_output = (args.output_file
                       or f"data/{args.variant}_pf_dataset.npz")
    variants.warn_if_override_contradicts(
        args.variant, env_id=args.env_id,
        particle_filter=resolve_particle_filter(args.particle_filter)
        if args.particle_filter else None,
    )

    if args.locomotion_vecnorm_path is None and args.locomotion_policy_path:
        candidate = os.path.join(
            os.path.dirname(args.locomotion_policy_path),
            "locomotion_vecnorm.pkl",
        )
        if os.path.exists(candidate):
            args.locomotion_vecnorm_path = candidate
            print(f"Auto-detected VecNormalize stats: {candidate}")

    print(f"Env: {resolved_env_id}, particle filter: {resolved_pf.__name__}")

    particles, weights = collect_dataset(
        env_id=resolved_env_id,
        particle_filter_class=resolved_pf,
        num_trajectories=args.num_trajectories,
        timesteps_per_trajectory=args.timesteps,
        num_particles=args.num_particles,
        pursuit_fraction=args.pursuit_fraction,
        fully_observed_fraction=args.fully_observed_fraction,
        visibility_radius_range=(args.visibility_radius_min,
                                 args.visibility_radius_max),
        locomotion_policy_path=args.locomotion_policy_path,
        locomotion_vecnorm_path=args.locomotion_vecnorm_path,
        seed=args.seed,
        evasion_scale=args.evasion_scale,
        target_speed_scale=args.target_speed_scale,
        max_snapshots=args.max_snapshots,
    )

    if not args.no_rebalance:
        particles, weights = _rebalance_by_spread(
            particles, weights,
            collapsed_threshold=args.collapsed_threshold,
            diffuse_threshold=args.diffuse_threshold,
            seed=args.seed,
            upsample=not args.rebalance_no_upsample,
        )

    print(f"Dataset: particles {particles.shape}, weights {weights.shape}")
    for dim in range(particles.shape[-1]):
        column = particles[:, :, dim]
        print(f"  dim {dim}: [{column.min():.2f}, {column.max():.2f}]")
    ess = 1.0 / np.clip((weights ** 2).sum(axis=1), 1e-12, None)
    print(f"  effective sample size: median {np.median(ess):.1f} "
          f"of {particles.shape[1]} particles "
          f"(min {ess.min():.1f}, max {ess.max():.1f})")

    # Particles are stored RAW, in env coordinates. The normalization the RL
    # feature extractors apply (divide by the arena half-width) is recorded
    # alongside them so 3_train_st.py can train the encoder on exactly the
    # inputs it will be handed at RL time. Without this the encoder would be
    # pretrained on coordinates several times larger than the ones it later
    # sees, and the pretrained weights would be near-useless.
    arena_scale = float(get_ant_tag_arena_scale(resolved_env_id))
    print(f"  arena scale (recorded for pretraining): {arena_scale}")

    metadata = {
        "variant": args.variant,
        "env_id": resolved_env_id,
        "particle_filter_class": resolved_pf.__name__,
        "num_particles": int(particles.shape[1]),
        "dim_particles": int(particles.shape[2]),
        "particle_scale": arena_scale,
        "args": vars(args),
        "git": _git_provenance(),
    }

    directory = os.path.dirname(resolved_output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if resolved_output.endswith(".npy"):
        raise ValueError(
            "Output must be .npz — a .npy cannot hold the weights alongside "
            "the particles. Legacy .npy datasets are still READABLE by "
            "3_train_st.py; they simply train unweighted."
        )
    np.savez(
        resolved_output,
        particles=particles,
        weights=weights,
        particle_scale=np.float32(arena_scale),
        metadata=json.dumps(metadata, default=str),
    )
    print(f"Saved to {resolved_output}")


if __name__ == "__main__":
    main()
