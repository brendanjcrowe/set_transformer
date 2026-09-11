"""
Evaluate a weighted-CGF checkpoint on the REAL sparse tag reward.

Mirrors eval_true_reward.py but uses PFDictWithWeightsObservationWrapper
(particles + PF weights), since WeightedCGFFeaturesExtractor requires both.

Usage:
    python experiments/ant_tag/eval_scripts/eval_true_reward_cgf.py \
        --model_path sb3_ant_tag_cgf_models_3M/best_model/best_model.zip \
        --vecnormalize_path sb3_ant_tag_cgf_models_3M/vecnormalize.pkl \
        --n_episodes 50
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The 4_train_rl_* pipeline scripts live one level up, in experiments/ant_tag/.
# Only this script's own directory is on sys.path by default, so add theirs.
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
if str(_ANT_TAG_DIR) not in sys.path:
    sys.path.insert(0, str(_ANT_TAG_DIR))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

import pdomains  # noqa: F401
import variants  # noqa: E402 - env/filter/cap registry
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter

# Sibling module name starts with a digit, so importlib is required.
_train_rl_cgf = importlib.import_module("4_train_rl_cgf")
CurriculumVisibilityWrapper = _train_rl_cgf.CurriculumVisibilityWrapper
_CurriculumRouter = _train_rl_cgf._CurriculumRouter
ant_tag_pf_interaction_mapper = _train_rl_cgf.ant_tag_pf_interaction_mapper
get_ant_tag_pf_kwargs = _train_rl_cgf.get_ant_tag_pf_kwargs
PFDictWithWeightsObservationWrapper = _train_rl_cgf.PFDictWithWeightsObservationWrapper


def make_eval_env(num_particles: int, obs_mask_indices, seed: int,
                   env_id: str = "pdomains-ant-tag-v0",
                   particle_filter_class: type = AntTagParticleFilter):
    def _init():
        env = gym.make(env_id, rendering=False)
        env.reset(seed=seed)
        particle_filter_kwargs = get_ant_tag_pf_kwargs(env)
        env = CurriculumVisibilityWrapper(
            env,
            initial_visibility_radius=float(env.unwrapped.visible_radius))
        env = PFDictWithWeightsObservationWrapper(
            env=env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs=particle_filter_kwargs,
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=obs_mask_indices,
        )
        # NOTE: no PFRewardShapingWrapper → Monitor sees true env reward
        env = Monitor(env)
        env = _CurriculumRouter(env)
        return env
    return _init


def _checkpoint_num_particles(model_path: str) -> int | None:
    """Particle-set size recorded in a saved policy's observation space.

    Returns None if it cannot be determined, in which case the caller's value
    (or the env default) stands and SB3 will complain on its own.
    """
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _, _ = load_from_zip_file(model_path, load_data=True,
                                        device="cpu", print_system_info=False)
        space = data["observation_space"]
        return int(space["particles"].shape[0])
    except Exception:  # noqa: BLE001 - a best-effort default, never fatal
        return None


def main():
    """--variant selects env id, particle filter AND the episode cap together."""
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the true sparse tag reward.")
    variants.add_variant_argument(p)
    # Not argparse-required: --list_variants must work without it,
    # and argparse enforces required= before any of our own code runs.
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--vecnormalize_path", type=str, default=None,
                   help="Path to vecnormalize.pkl saved during training")
    p.add_argument("--n_episodes", type=int, default=50)
    p.add_argument(
        "--num_particles", type=int, default=None,
        help="Particle count for the eval env. Defaults to the "
             "count baked into the checkpoint's observation space, "
             "which is the only value that can work.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max_steps", type=int, default=None,
        help="Episode cap of the env under evaluation; an episode ending "
             "strictly before this many steps counts as a tag. Defaults to "
             "the variant's registered max_episode_steps, which is the only "
             "correct value — a larger one counts every timeout as a tag.",
    )
    p.add_argument("--no_mask", action="store_true")
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", dest="deterministic", action="store_false")
    args = p.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    if args.model_path is None:
        p.error("--model_path is required")

    # The env's particle count must match the one the policy was trained with,
    # or SB3 rejects the observation space. Read it off the checkpoint instead
    # of defaulting to 100 and making the caller remember.
    trained_particles = _checkpoint_num_particles(args.model_path)
    if args.num_particles is None:
        args.num_particles = trained_particles
    elif trained_particles is not None and args.num_particles != trained_particles:
        p.error(
            f"--num_particles {args.num_particles} contradicts the checkpoint, "
            f"which was trained with {trained_particles}. Omit the flag."
        )

    variant = variants.resolve(args.variant)
    env_id = variant.env_id
    particle_filter_class = variant.particle_filter
    if args.max_steps is None:
        args.max_steps = variants.episode_cap(args.variant)

    obs_mask = None if args.no_mask else [-2, -1]

    print(f"Variant: {args.variant} | env: {env_id} | "
          f"filter: {particle_filter_class.__name__} | "
          f"episode cap: {args.max_steps}")
    env_fn = make_eval_env(args.num_particles, obs_mask, args.seed,
                            env_id=env_id, particle_filter_class=particle_filter_class)
    env = DummyVecEnv([env_fn])

    if args.vecnormalize_path and os.path.exists(args.vecnormalize_path):
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False   # keep true reward
        print(f"Loaded VecNormalize from {args.vecnormalize_path}")
    else:
        print("No VecNormalize — evaluating without obs normalization (may be invalid if training used it)")

    model = PPO.load(args.model_path, env=env)
    print(f"Loaded model from {args.model_path}")

    # Re-apply --seed AFTER the load. PPO.load restores the TRAINING seed from
    # the checkpoint and BaseAlgorithm.set_random_seed re-seeds the vec env
    # with it, so every evaluation of a given checkpoint replayed the same
    # episode set no matter what --seed said. Four "different" eval seeds
    # returned byte-identical results, which looks like a robust policy and is
    # actually one sample. Seeding here overrides that.
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    print(f"Eval episode seed: {args.seed} "
          f"(overriding the checkpoint's training seed)")

    rewards = []
    lengths = []
    successes = []
    tagged = 0
    for ep in range(args.n_episodes):
        obs = env.reset()
        done = False
        ep_r = 0.0
        ep_len = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, r, dones, infos = env.step(action)
            ep_r += float(r[0])
            ep_len += 1
            done = bool(dones[0])
        rewards.append(ep_r)
        lengths.append(ep_len)
        # A terminal hazard can end before max_steps without being a tag.
        success = bool(infos[0].get("is_success", ep_len < args.max_steps))
        successes.append(success)
        if success:
            tagged += 1

    rewards = np.array(rewards)
    lengths = np.array(lengths)

    best_idx = int(np.argmax(rewards))

    print(f"\n=== Eval over {args.n_episodes} episodes (deterministic={args.deterministic}) ===")
    print(f"Success rate  : {tagged}/{args.n_episodes} ({100*tagged/args.n_episodes:.1f}%)")
    print(f"Mean reward   : {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Mean length   : {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"Median length : {np.median(lengths):.1f}")
    print(f"Best episode  : reward={rewards[best_idx]:.2f}, length={lengths[best_idx]}")
    # Distribution of tag times (only on tagged episodes)
    if tagged > 0:
        tag_lens = lengths[np.asarray(successes, dtype=bool)]
        print(f"When tagged   : mean_len={tag_lens.mean():.1f}, median_len={np.median(tag_lens):.1f}")


if __name__ == "__main__":
    main()
