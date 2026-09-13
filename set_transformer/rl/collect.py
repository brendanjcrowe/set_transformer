"""One dataset-collection command for every domain (pipeline step 2).

Batch 7.4 of the harness centralisation (``refactor_plans.md`` section 7, 2026-09-13)::

    python3 -m set_transformer.rl.collect --domain ant_tag --variant smart \\
        --locomotion_policy_path models/ant_locomotion_policy.zip
    python3 -m set_transformer.rl.collect --domain odd_even --variant oe50_short --num_episodes 4000

``experiments/<domain>/2_collect_pf_dataset.py`` are entry points of this command with the
domain fixed; every flag they took still works (``--num_trajectories`` and ``--output`` are
aliases of ``--num_episodes`` and ``--output_file``).

What this module owns, the same for every problem: the command line around the problem's
flags (``--domain``, ``--variant``, ``--num_episodes``, ``--timesteps``, ``--num_particles``,
``--seed``, ``--max_snapshots``, ``--no_rebalance``, ``--output_file`` / ``--output_root`` /
``--run_tag``); the rollout loop -- build the belief env exactly as the RL step does, play
episodes, store the particle cloud and its weights after the reset and after every step, stop
at ``--max_snapshots``; the call to the problem's rebalance; the ``.npz`` contract::

    particles      [S, N, D]  float32, RAW env coordinates
    weights        [S, N]     float32, the PF weights
    particle_scale scalar: the arena half-width the RL extractors divide by
    metadata       JSON: variant, env id, filter, set size, dimension, scale, CLI args, git
    (+ the problem's extra members: Odd-Even adds particle_centre and the per-row step index)

What the problem owns (``rl/domains/base.py::Collection``): how an episode is driven, how rows
are rebalanced, what extra facts and arrays the file carries.

WEIGHTS ARE STORED: pretraining puts the mass in the reconstruction loss's target measure
(``rl/pretrain_objectives/reconstruction.py``); a ``.npy`` cannot hold them and is refused.

PLACEMENT (decision 1 of plan section 7): ``--output_file`` as given, else
``<root>/<domain>/<variant>/data/<variant>_pf_dataset[_<run_tag>].npz`` with the root of the
shared run layout (``--output_root`` > ``$RL_BMDP_RUNS`` > the parent repo's ``runs/``); never
the current directory. Recorded datasets stay in ``experiments/<domain>/data/``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from tqdm import tqdm

from set_transformer.rl import domains as _domains
from set_transformer.rl import run_records
from set_transformer.rl.domains.base import Collection, Domain

NPY_REFUSAL = ("Output must be .npz -- a .npy cannot hold the weights alongside the particles. "
               "Legacy .npy datasets are still READABLE by the reconstruction pretraining; they "
               "simply train unweighted.")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def collection_of(domain: Domain, parser_for_errors=None) -> Collection:
    if domain.collection is None:
        message = f"domain {domain.name!r} declares no dataset collector (Domain.collection is None)"
        if parser_for_errors is not None:
            parser_for_errors.error(message)
        raise ValueError(message)
    return domain.collection


def build_parser(domain: Domain, *, prog: str | None = None,
                 selectors: bool = True) -> argparse.ArgumentParser:
    """The full command line for one domain: the shared flags (with the problem's own
    defaults where its script had them), then the problem's flags."""
    collection = collection_of(domain)
    d = collection.defaults
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f"Collect a particle-filter belief dataset from {domain.name} (pipeline step "
                    "2). --variant picks the env and its matching filter together; "
                    "--list_variants shows them.")
    if selectors:
        parser.add_argument("--domain", choices=sorted(_domains.DOMAIN_NAMES), default=domain.name)
    domain.add_variant_argument(parser)
    g = parser.add_argument_group("collection")
    g.add_argument("--num_episodes", "--num_trajectories", dest="num_episodes", type=int,
                   default=int(d.get("num_episodes", 200)),
                   help="Episodes (trajectories) to roll. --num_trajectories is the Ant-Tag "
                        "script's spelling of the same flag.")
    g.add_argument("--timesteps", type=int, default=d.get("timesteps"),
                   help="Steps per episode." + (
                       "" if d.get("timesteps") is not None else
                       " Default: the variant's registered episode cap, so the transient and the "
                       "steady state are both represented in their true proportions before "
                       "rebalancing."))
    g.add_argument("--num_particles", type=int, default=d.get("num_particles"),
                   help="Filter set size." + (
                       "" if d.get("num_particles") is not None else
                       " Default: the variant's state count, which makes the exact-support "
                       "filter's belief EXACT -- one particle per state."))
    g.add_argument("--seed", type=int, default=int(d.get("seed", 0)))
    g.add_argument("--max_snapshots", type=int, default=None,
                   help="Stop once this many belief snapshots are collected.")
    g.add_argument("--no_rebalance", action="store_true",
                   help="Skip rebalancing and keep the raw visit distribution.")
    p = parser.add_argument_group("placement")
    p.add_argument("--output_file", "--output", dest="output_file", type=str, default=None,
                   help="Where to write the .npz. Default: "
                        "<root>/<domain>/<variant>/data/<variant>_pf_dataset[_<run_tag>].npz "
                        "-- see --output_root. --output is the Odd-Even script's spelling.")
    p.add_argument("--run_tag", type=str, default="",
                   help="Appended to the default file name.")
    p.add_argument("--output_root", type=str, default=None,
                   help="Root of the shared run layout when --output_file is not given: "
                        "$RL_BMDP_RUNS, else <parent repo>/runs when this checkout is a "
                        "submodule, else <checkout>/runs.")
    collection.add_arguments(parser)
    return parser


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def collect_arrays(domain: Domain, args, options: dict, *, progress: bool = True):
    """Roll out the RL-time belief env and record every belief it produces.

    The loop the two collectors shared, written once: the problem prepares (policy, RNG),
    builds the env, and per episode says how to reset and how to act; after the reset and after
    every step the particle cloud and its weights are stored, with the problem's centre added
    back so the file holds RAW coordinates.

    Returns ``(particles [S, N, D] float32, weights [S, N] float32, steps [S] int64)``.
    """
    collection = collection_of(domain)
    state = collection.prepare(args, options)
    env = collection.make_env(args, options, state)
    centre = collection.particle_centre(args, options)

    all_particles: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    all_steps: list[int] = []

    def _record(obs_dict, step_index):
        particles = np.asarray(obs_dict["particles"], dtype=np.float32)
        if centre is not None:
            # Stored RAW, undoing the env's centring; the file records BOTH halves of the
            # RL-side mapping (particle_centre, particle_scale) and the dataset loader applies
            # (x - centre) / scale, so the encoder is pretrained on exactly the inputs it will
            # later be handed (PITFALLS.md section 4).
            particles = particles + np.float32(centre)
        all_particles.append(particles)
        all_weights.append(np.asarray(obs_dict["weights"], dtype=np.float32))
        all_steps.append(int(step_index))

    episodes = range(args.num_episodes)
    bar = tqdm(episodes, desc=collection.progress_desc) if progress else episodes
    for episode in bar:
        reset_kwargs, act = collection.begin_episode(args, options, state, env, episode)
        obs, _info = env.reset(**reset_kwargs)
        _record(obs, 0)   # the prior / b0 is a legitimate belief state

        for step in range(args.timesteps):
            obs, _reward, terminated, truncated, _info = env.step(act(obs))
            _record(obs, step + 1)
            if terminated or truncated:
                break

        if args.max_snapshots is not None and len(all_particles) >= args.max_snapshots:
            print(f"\nReached --max_snapshots ({args.max_snapshots}); stopping.")
            break
        if progress:
            bar.set_postfix(snapshots=len(all_particles))

    env.close()
    collection.finish(args, options, state, len(all_particles))

    particles = np.asarray(all_particles, dtype=np.float32)
    weights = np.asarray(all_weights, dtype=np.float32)
    steps = np.asarray(all_steps, dtype=np.int64)
    if args.max_snapshots is not None:
        particles = particles[:args.max_snapshots]
        weights = weights[:args.max_snapshots]
        steps = steps[:args.max_snapshots]
    return particles, weights, steps


def build_metadata(domain: Domain, args, options: dict, particles, weights, steps, *,
                   record_args: bool = True) -> dict:
    """The metadata JSON that travels in the .npz: the shared facts, the problem's extras, the
    CLI args (``None`` for an in-process test collection) and git provenance."""
    collection = collection_of(domain)
    metadata = {
        "variant": args.variant,
        "env_id": options["env_id"],
        "particle_filter_class": options["particle_filter_class"].__name__,
        "num_particles": int(particles.shape[1]),
        "dim_particles": int(particles.shape[2]),
        # The RL feature extractors divide by this; recorded rather than recomputed so
        # pretraining and RL cannot drift apart (PITFALLS.md section 4).
        "particle_scale": collection.particle_scale(args, options),
    }
    metadata.update(collection.metadata_extras(args, options, particles, weights, steps))
    metadata["args"] = vars(args) if record_args else None
    metadata["git"] = run_records.git_provenance()
    return metadata


def save_dataset(path: Path, particles, weights, metadata: dict, extra_arrays: dict) -> None:
    path = Path(path)
    if path.suffix == ".npy":
        raise ValueError(NPY_REFUSAL)
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        particles=particles,
        weights=weights,
        particle_scale=np.float32(metadata["particle_scale"]),
        **extra_arrays,
        metadata=json.dumps(metadata, default=str),
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None, *, domain: Domain | str | None = None,
         prog: str | None = None) -> Path | None:
    """Parse, resolve, collect, rebalance, write. ``domain`` given: an entry point's fixed
    choice; not given: ``--domain`` from ``argv``. Returns the written path (None after a
    listing)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    errors = argparse.ArgumentParser(prog=prog)
    if domain is None:
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument("--domain", choices=sorted(_domains.DOMAIN_NAMES), default=None)
        known, _ = pre.parse_known_args(argv)
        if known.domain is None:
            if "-h" in argv or "--help" in argv:
                known.domain = sorted(_domains.DOMAIN_NAMES)[0]
            else:
                errors.error("--domain is required (e.g. --domain ant_tag)")
        domain = known.domain
    domain = _domains.get(domain)
    collection = collection_of(domain, errors)

    parser = build_parser(domain, prog=prog)
    args = parser.parse_args(argv)
    if args.list_variants:
        domain.print_variants()
        return None

    options = collection.resolve_arguments(parser, args, domain)
    for required in ("env_id", "particle_filter_class"):
        if required not in options:
            raise RuntimeError(f"domain {domain.name!r}'s collection.resolve_arguments returned "
                               f"no {required!r}")
    for required in ("timesteps", "num_particles"):
        if getattr(args, required, None) is None:
            parser.error(f"--{required} is required for {domain.name} (the domain resolved no default)")
    if args.output_file:
        output = Path(args.output_file)
    else:
        output = run_records.dataset_path(domain.name, args.variant, tag=args.run_tag,
                                          root=args.output_root)
        print(f"Output root layout: {output} (domain {domain.name}, variant {args.variant})")
    if output.suffix == ".npy":
        # Refused before anything is collected.
        raise ValueError(NPY_REFUSAL)

    print(f"Collecting {args.num_episodes} episodes x up to {args.timesteps} steps, "
          f"{args.num_particles} particles, seed {args.seed}")
    particles, weights, steps = collect_arrays(domain, args, options, progress=True)
    collection.report(args, options, particles, weights, steps, "raw")
    if not args.no_rebalance:
        particles, weights, steps = collection.rebalance(args, options, particles, weights, steps)
    collection.report(args, options, particles, weights, steps, "final")

    metadata = build_metadata(domain, args, options, particles, weights, steps)
    extra = collection.extra_arrays(args, options, particles, weights, steps)
    save_dataset(output, particles, weights, metadata, extra)
    print(f"Saved to {output}")
    return output


if __name__ == "__main__":
    main()
