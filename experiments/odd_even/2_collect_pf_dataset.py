"""
Collect a particle-filter dataset from the Odd-Even POMDP.

Step 2 of the pipeline. Produces the belief snapshots that 3_train_st.py
pretrains the Set Transformer on, in exactly the .npz contract that script
already reads -- so step 3 needs no Odd-Even version at all:

    particles      [num_samples, num_particles, dim]  float32, RAW states
    weights        [num_samples, num_particles]        float32, PF weights
    particle_scale scalar: the half-width the RL extractors divide by
    metadata       JSON string: env id, filter, CLI args, git provenance

The rollout goes through `odd_even_belief_env.make_odd_even_belief_env`, the
same factory the RL arms and the evaluator use, so a stored belief is always
one the RL-time filter really produces. A hand-rolled predict/update loop
drifts from that filter silently.

WHAT THIS SCRIPT DOES *NOT* HAVE, and why. Its Ant-Tag counterpart carries a
locomotion policy, a pursuit-versus-random action mix, a visibility radius and
spread thresholds in arena units. None of it applies: on this env the action
is a PREDICTION, so it changes neither the hidden state nor the observation
stream. The belief trajectory is a function of the observations alone, and
random actions therefore give the correct, unbiased belief distribution. There
is no exploration policy to design and step 1 of the pipeline drops out.

WHAT REPLACES THE SPREAD REBALANCING. The belief here locks on by about step
21 of 50, so uniform sampling over an episode leaves roughly 60% of snapshots
one-hot -- the regime where every encoder is equivalent, and so the regime an
encoder comparison learns nothing from. Rebalancing is by STEP INDEX (or, with
--rebalance_by ess, by effective sample size), never by "spread in arena
units": on a 1-D integer state the spread of a near-one-hot belief and of a
two-mode belief can coincide, while the step index is exactly the axis along
which the belief sharpens.

Usage:
    python3 2_collect_pf_dataset.py --variant oe50 \
        --num_episodes 200 --timesteps 50 --num_particles 50

    python3 2_collect_pf_dataset.py --list_variants
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from tqdm import tqdm

import pdomains  # noqa: F401,E402 - registers the pdomains-odd-even-* envs
# The registry and the belief env live in the package (change 5.3a retired the by-path
# sibling loader that used to keep this `variants` apart from experiments/ant_tag/'s).
from set_transformer.rl.domains import odd_even as variants  # noqa: E402
from set_transformer.rl.domains.odd_even import make_odd_even_belief_env  # noqa: E402
from set_transformer.rl.run_records import git_provenance as _git_provenance  # noqa: E402

#: Steps below this still carry a genuinely BROAD belief. Measured at n=50:
#: the share of snapshots with effective sample size above 3 (of 50) is 0.90,
#: 0.95, 0.95 at steps 0, 1, 2 and collapses to 0.20, 0.05, 0.00 at steps
#: 3, 4, 5. The belief therefore sharpens several times faster than the
#: max(belief) > 0.9 criterion suggests, so the 'early' bucket has to be
#: narrow to hold anything an encoder can distinguish.
EARLY_STEP = 3

#: Step at which the exact posterior has locked on at n=50 (max(belief) > 0.9),
#: measured in domain_mds/oddeven.md. The transient/steady boundary everywhere
#: in this pipeline, and the default rebalancing boundary here.
COLLAPSE_STEP = 21


def _effective_sample_size(weights: np.ndarray) -> np.ndarray:
    """1 / sum(w^2) per snapshot. About 1.0 once the belief has locked on."""
    weights = np.asarray(weights, dtype=np.float64)
    return 1.0 / np.clip((weights ** 2).sum(axis=1), 1e-30, None)


def _rebalance(
    particles: np.ndarray,
    weights: np.ndarray,
    steps: np.ndarray,
    by: str = "step",
    early_step: int = EARLY_STEP,
    collapse_step: int = COLLAPSE_STEP,
    diffuse_ess: float = 5.0,
    collapsed_ess: float = 1.5,
    early_frac: float = 0.40,
    mid_frac: float = 0.35,
    late_frac: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebalance the snapshot mix across the belief's own sharpening axis.

    Three buckets, downsampled (never upsampled -- no row appears twice) to
    the target fractions. `by="step"` splits on the step index, which is the axis the
    belief sharpens along and needs no coordinate units. `by="ess"` splits on
    effective sample size, which measures the sharpening directly and so is
    the better choice when episodes are ragged.

    An empty bucket cannot be sampled from; its quota is redistributed over
    the buckets that do have data, in proportion to their own quotas, so the
    target fractions still describe the output. The output is smaller than
    the input whenever the buckets are not already in the target ratio, and
    says so.
    """
    if by == "step":
        early = np.where(steps < early_step)[0]
        mid = np.where((steps >= early_step) & (steps < collapse_step))[0]
        late = np.where(steps >= collapse_step)[0]
        labels = (f"steps <{early_step}", f"steps {early_step}-{collapse_step-1}",
                  f"steps >={collapse_step}")
    elif by == "ess":
        ess = _effective_sample_size(weights)
        early = np.where(ess >= diffuse_ess)[0]
        mid = np.where((ess < diffuse_ess) & (ess > collapsed_ess))[0]
        late = np.where(ess <= collapsed_ess)[0]
        labels = (f"ess >={diffuse_ess}",
                  f"ess {collapsed_ess}-{diffuse_ess}",
                  f"ess <={collapsed_ess}")
    else:
        raise ValueError(f"Unknown --rebalance_by: {by!r}; use step or ess")

    n_total = len(particles)
    buckets = [(labels[0], early, early_frac),
               (labels[1], mid, mid_frac),
               (labels[2], late, late_frac)]
    print("Pre-rebalance: " + ", ".join(
        f"{name}={len(idx)} ({len(idx)/max(n_total,1)*100:.1f}%)"
        for name, idx, _ in buckets))

    live = [b for b in buckets if len(b[1]) > 0]
    if not live:
        raise ValueError("No samples to rebalance")
    empty = [name for name, idx, _ in buckets if len(idx) == 0]
    if empty:
        print(f"  WARNING: no samples in bucket(s) {empty}; their share is "
              "redistributed over the remaining buckets. Consider a longer "
              "--timesteps or different --early_step / --collapse_step.")

    rng = np.random.default_rng(seed)
    live_frac_total = sum(frac for _, _, frac in live)
    shares = [(frac / live_frac_total if live_frac_total > 0
               else 1.0 / len(live)) for _, _, frac in live]
    # Never upsample. Sampling WITH replacement used to fill the early
    # bucket's quota with copies (measured 5.84x duplication on oe50_short);
    # 3_train_st.py then random-splits the rows, so identical snapshots
    # landed on both sides and made best_val_loss optimistic. Instead the
    # output size is set by the tightest bucket, so every target fraction is
    # met exactly with distinct rows. Collect more episodes for more data.
    n_out = int(min(len(idx) / share for (_n, idx, _f), share in zip(live, shares)))
    sampled = []
    for (_name, idx, _frac), share in zip(live, shares):
        n_target = min(len(idx), int(round(n_out * share)))
        sampled.append(rng.choice(idx, size=n_target, replace=False))
    all_idx = np.concatenate(sampled)
    rng.shuffle(all_idx)
    if len(all_idx) < n_total:
        print(f"  Rebalance keeps {len(all_idx)} of {n_total} snapshots "
              "(downsampled to the target fractions without duplication)")
    return particles[all_idx], weights[all_idx], steps[all_idx]


def collect_dataset(
    variant: str,
    num_episodes: int,
    timesteps: int,
    num_particles: int,
    seed: int,
    particle_filter_class: type | None = None,
    max_snapshots: int | None = None,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the RL-time belief env and record every belief it produces.

    Actions are drawn uniformly. On this env that is not a compromise: the
    action is a prediction, so it does not move the state or the observation
    stream, and the belief distribution collected under random actions is the
    same one any policy would induce.

    Returns:
        (particles [S, N, D] RAW states, weights [S, N], step_index [S]).
    """
    env = make_odd_even_belief_env(
        num_particles=num_particles,
        rank=0,
        seed=seed,
        variant=variant,
        particle_filter_class=particle_filter_class,
    )()
    centre = variants.state_centre(variant)

    all_particles: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_steps: list[int] = []

    def _record(obs_dict, step_index):
        # Stored RAW, undoing the env's centring. The file records BOTH
        # halves of the RL-side mapping -- particle_centre and particle_scale
        # -- and dataset.py applies (x - centre) / scale at load, so the
        # encoder is pretrained on exactly the inputs it will later be handed
        # (PITFALLS.md section 4). Until 2026-09-03 only the scale was
        # applied, and pretraining ran one whole normalized unit off.
        all_particles.append(
            np.asarray(obs_dict["particles"], dtype=np.float32) + np.float32(centre))
        all_weights.append(np.asarray(obs_dict["weights"], dtype=np.float32))
        all_steps.append(int(step_index))

    episodes = range(num_episodes)
    bar = tqdm(episodes, desc="Collecting episodes") if progress else episodes
    for episode in bar:
        # seed + episode: on this env the hidden state is drawn at reset, so
        # THE SEED IS THE EPISODE. A constant reset seed would collect one
        # episode num_episodes times (PITFALLS.md section 2).
        obs, _info = env.reset(seed=seed + episode)
        _record(obs, 0)   # b0 = P(s | o0) is a legitimate belief state

        for step in range(timesteps):
            obs, _r, terminated, truncated, _info = env.step(
                env.action_space.sample())
            _record(obs, step + 1)
            if terminated or truncated:
                break

        if max_snapshots is not None and len(all_particles) >= max_snapshots:
            print(f"\nReached --max_snapshots ({max_snapshots}); stopping.")
            break

    env.close()

    particles = np.asarray(all_particles, dtype=np.float32)
    weights = np.asarray(all_weights, dtype=np.float32)
    steps = np.asarray(all_steps, dtype=np.int64)
    if max_snapshots is not None:
        particles = particles[:max_snapshots]
        weights = weights[:max_snapshots]
        steps = steps[:max_snapshots]
    return particles, weights, steps


def collect_dataset_for_test(ns: int, num_episodes: int, timesteps: int,
                            num_particles: int, seed: int):
    """Collect a small dataset in-process, for the contract tests.

    Named and shaped for tests/test_odd_even_pomdp_contract.py, which calls
    exactly this signature. `ns` selects the variant by state range, so a
    test does not have to know the registry keys.

    Returns:
        (particles, weights, metadata_dict) -- the same three things the .npz
        carries, so a test checks the real contract and not a parallel one.
    """
    matches = [name for name, v in variants.VARIANTS.items()
               if v.n_dist_size == int(ns)]
    if not matches:
        raise ValueError(
            f"No registered variant with n_dist_size={ns}; have "
            + ", ".join(f"{n}(n={v.n_dist_size})"
                        for n, v in variants.VARIANTS.items()))
    # The shortest cap among the matches: a test wants the cheapest env whose
    # state range is the one it asked for.
    variant = min(matches, key=variants.episode_cap)

    particles, weights, steps = collect_dataset(
        variant=variant,
        num_episodes=num_episodes,
        timesteps=timesteps,
        num_particles=num_particles,
        seed=seed,
        progress=False,
    )
    particles, weights, steps = _rebalance(particles, weights, steps,
                                            by="step", seed=seed)
    metadata = _build_metadata(variant, particles, weights, steps, args=None)
    return particles, weights, metadata


def _build_metadata(variant: str, particles, weights, steps, args) -> dict:
    """The metadata JSON that travels in the .npz."""
    resolved = variants.resolve(variant)
    return {
        "variant": variant,
        "env_id": resolved.env_id,
        "particle_filter_class": resolved.particle_filter.__name__,
        "n_dist_size": resolved.n_dist_size,
        "episode_cap": variants.episode_cap(variant),
        "num_particles": int(particles.shape[1]),
        "dim_particles": int(particles.shape[2]),
        # The half-width of the state range. 3_train_st.py divides by this and
        # so do the RL extractors' arena_scale, which is the whole reason it
        # is recorded rather than recomputed.
        "particle_scale": variants.state_scale(variant),
        # The centre the RL env subtracts before that division. Recorded for
        # the same reason: the two halves of the mapping must not drift.
        "particle_centre": variants.state_centre(variant),
        "step_index_min": int(steps.min()) if len(steps) else None,
        "step_index_max": int(steps.max()) if len(steps) else None,
        "args": vars(args) if args is not None else None,
        "git": _git_provenance(),
    }


def _report_distribution(particles, weights, steps, collapse_step) -> None:
    """Print the achieved mix, so a bad dataset is visible before step 3."""
    ess = _effective_sample_size(weights)
    n = len(particles)
    print(f"Dataset: particles {particles.shape}, weights {weights.shape}")
    print(f"  state range: [{particles.min():.1f}, {particles.max():.1f}]")
    print(f"  effective sample size: median {np.median(ess):.2f} of "
          f"{particles.shape[1]} (min {ess.min():.2f}, max {ess.max():.2f})")
    print(f"  step index: min {steps.min()}, median {np.median(steps):.0f}, "
          f"max {steps.max()}")
    print(f"  pre-collapse snapshots (step < {collapse_step}): "
          f"{int((steps < collapse_step).sum())}/{n} "
          f"({(steps < collapse_step).mean()*100:.1f}%)")
    for threshold in (1.5, 3.0, 5.0, 10.0):
        share = float((ess > threshold).mean())
        print(f"  ess > {threshold:>4}: {share*100:5.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect an Odd-Even particle-filter dataset "
                    "(pipeline step 2). --variant picks the env and its "
                    "matching filter together; --list_variants shows them.")
    variants.add_variant_argument(parser)
    parser.add_argument(
        "--particle_filter", type=str, default=None,
        choices=sorted(variants.PARTICLE_FILTERS),
        help="Override the variant's filter. The bootstrap filter is a "
             "deliberate arm (a lossier belief on the same env), not a "
             "different domain.")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Steps per episode. Defaults to the variant's registered "
             "episode cap, so the transient and the steady state are both "
             "represented in their true proportions before rebalancing.")
    parser.add_argument(
        "--num_particles", type=int, default=None,
        help="Filter set size. Defaults to n_dist_size, which makes the "
             "exact-support filter's belief EXACT -- one particle per state.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_snapshots", type=int, default=None,
        help="Stop once this many snapshots are collected.")
    parser.add_argument(
        "--no_rebalance", action="store_true",
        help="Keep the raw visit distribution. At the default cap that is "
             "about 60%% one-hot beliefs, the regime where every encoder is "
             "equivalent.")
    parser.add_argument(
        "--rebalance_by", type=str, default="step", choices=["step", "ess"],
        help="Bucket on the step index (default) or on effective sample "
             "size. NOT on spread in coordinate units, which does not "
             "separate a near-one-hot belief from a two-mode one here.")
    parser.add_argument(
        "--early_step", type=int, default=EARLY_STEP,
        help=f"Steps below this are the 'early' bucket (default "
             f"{EARLY_STEP}). Measured, not guessed: at n=50 the share of "
             "snapshots with effective sample size above 3 of 50 is 0.90, "
             "0.95, 0.95 at steps 0-2 and then 0.20, 0.05, 0.00 at steps "
             "3-5. Everything the encoder could distinguish is in those "
             "first three steps.")
    parser.add_argument(
        "--collapse_step", type=int, default=COLLAPSE_STEP,
        help=f"Steps at or above this are 'late'. Default {COLLAPSE_STEP}, "
             "the measured step at which the n=50 posterior has locked on.")
    parser.add_argument("--diffuse_ess", type=float, default=5.0)
    parser.add_argument("--collapsed_ess", type=float, default=1.5)
    parser.add_argument("--early_frac", type=float, default=0.40)
    parser.add_argument("--mid_frac", type=float, default=0.35)
    parser.add_argument("--late_frac", type=float, default=0.25)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Defaults to data/<variant>_pf_dataset.npz")
    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    resolved = variants.resolve(args.variant)
    particle_filter_class = variants.resolve_particle_filter(
        args.variant, args.particle_filter)
    if args.timesteps is None:
        args.timesteps = variants.episode_cap(args.variant)
    if args.num_particles is None:
        args.num_particles = resolved.n_dist_size
    output = args.output or f"data/{args.variant}_pf_dataset.npz"
    if output.endswith(".npy"):
        raise ValueError(
            "Output must be .npz -- a .npy cannot hold the weights alongside "
            "the particles, and on this domain the weights ARE the belief "
            "(median effective sample size about 1 of 50).")

    print(f"Variant: {args.variant} | env: {resolved.env_id} | "
          f"filter: {particle_filter_class.__name__} | "
          f"n={resolved.n_dist_size} | cap={variants.episode_cap(args.variant)}")

    particles, weights, steps = collect_dataset(
        variant=args.variant,
        num_episodes=args.num_episodes,
        timesteps=args.timesteps,
        num_particles=args.num_particles,
        seed=args.seed,
        particle_filter_class=particle_filter_class,
        max_snapshots=args.max_snapshots,
    )
    print(f"\nRaw snapshots: {len(particles)}")
    _report_distribution(particles, weights, steps, args.collapse_step)

    if not args.no_rebalance:
        print("\nRebalancing...")
        particles, weights, steps = _rebalance(
            particles, weights, steps,
            by=args.rebalance_by,
            early_step=args.early_step,
            collapse_step=args.collapse_step,
            diffuse_ess=args.diffuse_ess,
            collapsed_ess=args.collapsed_ess,
            early_frac=args.early_frac,
            mid_frac=args.mid_frac,
            late_frac=args.late_frac,
            seed=args.seed,
        )
        print("\nPost-rebalance:")
        _report_distribution(particles, weights, steps, args.collapse_step)

    metadata = _build_metadata(args.variant, particles, weights, steps, args)
    print(f"\n  particle_scale (recorded for pretraining): "
          f"{metadata['particle_scale']}")
    print(f"  particle_centre: {metadata['particle_centre']}")

    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(
        output,
        particles=particles,
        weights=weights,
        particle_scale=np.float32(metadata["particle_scale"]),
        # Top-level so get_dataset() reads it without parsing the metadata.
        # dataset.py applies (x - centre) / scale, matching the RL wrapper.
        particle_centre=np.float32(metadata["particle_centre"]),
        # Per-row step index, so the transient/steady mix can be checked or
        # re-split downstream (only min/max used to be recorded).
        steps=steps.astype(np.int32),
        metadata=json.dumps(metadata, default=str),
    )
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
