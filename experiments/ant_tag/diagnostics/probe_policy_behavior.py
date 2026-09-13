"""Roll out a trained Ant-Tag policy and describe WHAT it does, not just whether it tags.

Answers, per episode: does the ant move at all (speed, path length, net
displacement), does it ever get the target in view (first-seen step, steps in
view, how many times it acquires and loses it), how close it gets (min
distance), does it head toward the belief (fraction of steps that shrink the
distance to the PF mean), and how the episode ends. Episodes are then binned:

  tag              terminated with a tag
  chase_failure    target was in view at some point, never tagged
  search_failure   target never came into view
  (plus `stationary` if the ant's mean speed is under --stationary_speed)

Works for the CGF / Gaussian / ST belief policies (same dict-obs eval env as
eval_scripts/eval_true_reward_*.py) and for the fully-observed ceiling
policies from sanity_check_fully_observed.py (--fullyobs). Deterministic
actions. No reward shaping anywhere.

    python3 diagnostics/probe_policy_behavior.py --variant smart_hard \
        --encoder cgf --model_path runs/.../models/cgf_agent.zip \
        --vecnormalize_path runs/.../models/vecnormalize.pkl --n_episodes 50
    python3 diagnostics/probe_policy_behavior.py --variant smart_hard --fullyobs \
        --model_path sanity_check_models/smart_hard_fullyobs_seed1/fully_obs_agent.zip \
        --vecnormalize_path sanity_check_models/smart_hard_fullyobs_seed1/vecnormalize.pkl
    python3 diagnostics/probe_policy_behavior.py --variant smart_hard --random
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# diagnostics/ is one level below the ant_tag scripts: parents[3] is the
# set_transformer repo root, parents[1] the ant_tag directory (see CLAUDE.md).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_ANT_TAG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import gymnasium as gym  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

import variants  # noqa: E402

# SB3 unpickles the features-extractor class out of policy_kwargs, so the
# module that defines it (or imports it) must be loaded first. Importing all
# three arms is harmless and lets one script probe any checkpoint.
_cgf = importlib.import_module("4_train_rl_cgf")
importlib.import_module("4_train_rl_gaussian")
importlib.import_module("4_train_rl_st")


def _make_belief_env(env_id, particle_filter_class, num_particles, seed):
    """Same construction as set_transformer.rl.domains.ant_tag.make_eval_env."""
    def _init():
        env = gym.make(env_id, rendering=False)
        env.reset(seed=seed)
        pf_kwargs = _cgf.get_ant_tag_pf_kwargs(env)
        env = _cgf.CurriculumVisibilityWrapper(
            env, initial_visibility_radius=float(env.unwrapped.visible_radius))
        env = _cgf.PFDictWithWeightsObservationWrapper(
            env=env, particle_filter_class=particle_filter_class,
            particle_filter_kwargs=pf_kwargs, num_particles=num_particles,
            pf_interaction_mapper=_cgf.ant_tag_pf_interaction_mapper,
            obs_mask_indices=[-2, -1])
        return _cgf._CurriculumRouter(env)
    return _init


def _make_fullyobs_env(env_id, seed):
    sanity = importlib.import_module("sanity_check_fully_observed")
    def _init():
        env = gym.make(env_id, rendering=False)
        env.reset(seed=seed)
        return sanity.FullyObservableWrapper(env)
    return _init


def _unwrapped(vec_env):
    venv = vec_env.venv if isinstance(vec_env, VecNormalize) else vec_env
    return venv.envs[0].unwrapped


def _pf_mean(obs):
    if not isinstance(obs, dict) or "particles" not in obs:
        return None
    parts = np.asarray(obs["particles"])[0]
    w = np.asarray(obs["weights"])[0]
    w = w / max(float(w.sum()), 1e-12)
    return (parts * w[:, None]).sum(axis=0)


def _pf_spread(obs):
    """Weighted std per coordinate, averaged (the collector's belief-spread measure).
    > ~2.7 on the 9x9 cage means MORE spread than uniform: a hollow perimeter frame."""
    if not isinstance(obs, dict) or "particles" not in obs:
        return None
    parts = np.asarray(obs["particles"])[0]
    w = np.asarray(obs["weights"])[0]
    w = w / max(float(w.sum()), 1e-12)
    mu = (parts * w[:, None]).sum(axis=0)
    var = (w[:, None] * (parts - mu) ** 2).sum(axis=0)
    return float(np.sqrt(var).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    variants.add_variant_argument(ap, default="smart_hard")
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--vecnormalize_path", type=str, default=None)
    ap.add_argument("--encoder", choices=["cgf", "gaussian", "st"], default="cgf",
                    help="Only used for the label; all extractor modules are imported.")
    ap.add_argument("--fullyobs", action="store_true",
                    help="Checkpoint is from sanity_check_fully_observed.py (raw 31-D obs, "
                         "true target revealed every step).")
    ap.add_argument("--random", action="store_true", help="Uniform random actions, no model.")
    ap.add_argument("--num_particles", type=int, default=100)
    ap.add_argument("--n_episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stationary_speed", type=float, default=0.02,
                    help="Mean per-step displacement below which an episode counts as stationary.")
    ap.add_argument("--centre_radius", type=float, default=2.0,
                    help="ant within this distance of the cage centre counts as 'central' (cage half-width 4.5).")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()
    if args.list_variants:
        variants.print_variants(); return

    variant = variants.resolve(args.variant)
    cap = variants.episode_cap(args.variant)
    if args.fullyobs:
        env = DummyVecEnv([_make_fullyobs_env(variant.env_id, args.seed)])
    else:
        env = DummyVecEnv([_make_belief_env(variant.env_id, variant.particle_filter,
                                            args.num_particles, args.seed)])
    if args.vecnormalize_path:
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
    model = None
    if not args.random:
        model = PPO.load(args.model_path, env=env, device="cpu")
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    raw = _unwrapped(env)
    vis_r, tag_r = float(raw.visible_radius), float(raw.tag_radius)
    print(f"variant={args.variant} env={variant.env_id} cap={cap} visible={vis_r} tag={tag_r} "
          f"target_step={raw.target_step} | {'RANDOM' if args.random else args.model_path} "
          f"| {'fully observed' if args.fullyobs else 'belief (PF) obs'}")

    episodes = []
    for ep in range(args.n_episodes):
        obs = env.reset()
        ant_prev = raw.data.qpos[:2].copy()
        ant_start = ant_prev.copy()
        speeds, dists, seen, toward_belief = [], [], [], []
        centre_d, spreads, mean_d = [], [], []   # ant->cage centre, belief spread, ant->belief mean
        done, steps, tagged = False, 0, False
        while not done:
            if args.random:
                action = np.asarray([env.action_space.sample()])
            else:
                action, _ = model.predict(obs, deterministic=True)
            pf_mean_before = _pf_mean(obs)
            ant_before = raw.data.qpos[:2].copy()
            d_belief_before = (np.linalg.norm(ant_before - pf_mean_before)
                               if pf_mean_before is not None else None)
            centre_d.append(float(np.linalg.norm(ant_before)))
            if pf_mean_before is not None:
                spreads.append(_pf_spread(obs)); mean_d.append(float(d_belief_before))
            obs, r, dones, infos = env.step(action)
            steps += 1
            done = bool(dones[0])
            if done:
                # DummyVecEnv has already auto-reset the env: raw.data now
                # holds the NEXT episode's start, so nothing about this step
                # can be read off it. A termination before the cap is a tag.
                tagged = steps < cap
                break
            ant = raw.data.qpos[:2].copy()
            tgt = raw.get_target_pos()
            d = float(np.linalg.norm(ant - tgt))
            speeds.append(float(np.linalg.norm(ant - ant_prev)))
            dists.append(d)
            seen.append(d < vis_r)
            if d_belief_before is not None:
                toward_belief.append(np.linalg.norm(ant - pf_mean_before) < d_belief_before)
            ant_prev = ant
        speeds, dists, seen = np.array(speeds), np.array(dists), np.array(seen)
        if tagged:
            # the tagging step itself was not recorded; it ended inside tag_r
            dists = np.append(dists, min(float(dists.min()) if dists.size else tag_r, tag_r))
            seen = np.append(seen, True)
        # count acquisitions: transitions from not-seen to seen
        acq = int(np.sum(seen[1:] & ~seen[:-1]) + (1 if seen[0] else 0))
        first_seen = int(np.argmax(seen)) if seen.any() else None
        rec = dict(
            ep=ep, steps=steps, tagged=tagged,
            mean_speed=float(speeds.mean()), path_length=float(speeds.sum()),
            net_displacement=float(np.linalg.norm(ant_prev - ant_start)),
            start_dist=float(dists[0]), min_dist=float(dists.min()), final_dist=float(dists[-1]),
            steps_in_view=int(seen.sum()), first_seen_step=first_seen, acquisitions=acq,
            frac_steps_toward_belief=(float(np.mean(toward_belief)) if toward_belief else None),
            mean_dist_from_centre=float(np.mean(centre_d)),
            frac_steps_near_centre=float(np.mean(np.array(centre_d) < args.centre_radius)),
            frac_steps_frame_belief=(float(np.mean(np.array(spreads) > 2.7)) if spreads else None),
            mean_dist_to_belief_mean=(float(np.mean(mean_d)) if mean_d else None),
            # where is the ant while the belief is a hollow frame?
            frame_steps_mean_dist_from_centre=(float(np.mean(np.array(centre_d)[np.array(spreads) > 2.7]))
                                               if spreads and (np.array(spreads) > 2.7).any() else None),
        )
        if tagged:
            rec["outcome"] = "tag"
        elif seen.any():
            rec["outcome"] = "chase_failure"
        else:
            rec["outcome"] = "search_failure"
        if rec["mean_speed"] < args.stationary_speed:
            rec["outcome"] += "+stationary"
        episodes.append(rec)

    n = len(episodes)
    def col(k): return np.array([e[k] for e in episodes if e[k] is not None], dtype=float)
    outcomes = {}
    for e in episodes:
        outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1
    print(f"\n=== {n} episodes, deterministic, seed {args.seed} ===")
    print("outcomes:", ", ".join(f"{k} {v} ({100*v/n:.0f}%)" for k, v in sorted(outcomes.items())))
    print(f"mean speed (units/step): median {np.median(col('mean_speed')):.3f}  "
          f"[p10 {np.percentile(col('mean_speed'),10):.3f}, p90 {np.percentile(col('mean_speed'),90):.3f}]"
          f"   (reference: trained locomotion policy ~0.165; run --random for the flailing baseline)")
    print(f"path length: median {np.median(col('path_length')):.1f}   net displacement: median {np.median(col('net_displacement')):.2f}")
    print(f"start dist: median {np.median(col('start_dist')):.2f}   min dist reached: median {np.median(col('min_dist')):.2f} "
          f"[p10 {np.percentile(col('min_dist'),10):.2f}]   final dist: median {np.median(col('final_dist')):.2f}")
    sv = col("steps_in_view")
    print(f"steps with target in view (< {vis_r}): median {np.median(sv):.0f}, mean {sv.mean():.1f} of {cap}; "
          f"episodes with any sighting: {int((sv>0).sum())}/{n}; acquisitions per episode: mean {col('acquisitions').mean():.2f}")
    fs = col("first_seen_step")
    if fs.size:
        print(f"first sighting step: median {np.median(fs):.0f}")
    tb = col("frac_steps_toward_belief")
    if tb.size:
        print(f"fraction of steps that reduce distance to the PF mean: median {np.median(tb):.2f} (0.5 = no preference)")
    tags = [e for e in episodes if e["tagged"]]
    if tags:
        print(f"tagged episodes: {len(tags)}; length median {np.median([e['steps'] for e in tags]):.0f}; "
              f"steps in view before tag median {np.median([e['steps_in_view'] for e in tags]):.0f}")
    chase = [e for e in episodes if e["outcome"].startswith("chase_failure")]
    if chase:
        print(f"chase failures: {len(chase)}; min dist median {np.median([e['min_dist'] for e in chase]):.2f} "
              f"(tag radius {tag_r}); sightings per episode median {np.median([e['acquisitions'] for e in chase]):.0f}; "
              f"steps in view median {np.median([e['steps_in_view'] for e in chase]):.0f}")
    # --- mean-tracking / centre-hovering report, split by outcome ---------------
    def grp(key, eps):
        v = [e[key] for e in eps if e.get(key) is not None]
        return f"{np.median(v):.2f}" if v else "-"
    fails = [e for e in episodes if not e["tagged"]]
    print("\n--- where the ant spends its time (medians over episodes) ---")
    print(f"{'group':18s} {'n':>3s} {'dist from centre':>17s} {'frac steps <%.1f of centre' % args.centre_radius:>27s} "
          f"{'frac steps w/ frame belief':>27s} {'dist to belief mean':>20s} {'dist from centre DURING frame steps':>36s} {'frac steps toward mean':>23s}")
    for name, eps in (("all", episodes), ("tagged", tags), ("failed", fails)):
        if not eps: continue
        print(f"{name:18s} {len(eps):3d} {grp('mean_dist_from_centre', eps):>17s} {grp('frac_steps_near_centre', eps):>27s} "
              f"{grp('frac_steps_frame_belief', eps):>27s} {grp('mean_dist_to_belief_mean', eps):>20s} "
              f"{grp('frame_steps_mean_dist_from_centre', eps):>36s} {grp('frac_steps_toward_belief', eps):>23s}")
    print("(reference: a uniformly random position in the 9x9 cage is 2.6 from the centre on average, "
          f"and lies within {args.centre_radius} of it {100*np.pi*args.centre_radius**2/81:.0f}% of the time)")
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(dict(args=vars(args), env_id=variant.env_id, visible=vis_r, tag=tag_r,
                           episodes=episodes), f, indent=1)
        print("wrote", args.out_json)


if __name__ == "__main__":
    main()
