"""Evaluate a saved agent on its environment's OWN reward: one script for every domain and
encoder (change 5.1 of the harness centralisation, 2026-09-12).

    python -m set_transformer.rl.eval_true_reward --domain ant_tag --variant smart \\
        --model_path runs/ant_tag_cgf_smart/<run>/models/cgf_agent.zip \\
        --vecnormalize_path runs/ant_tag_cgf_smart/<run>/models/vecnormalize.pkl --n_episodes 100
    python -m set_transformer.rl.eval_true_reward --domain odd_even --variant oe50_short \\
        --baselines_only --n_episodes 400

The per-domain files under ``experiments/<domain>/eval_scripts/`` are entry points of this
module with the domain fixed; every flag they took still works.

What the script does, in order, and why each step is load-bearing (PITFALLS.md sections 2,
5 and 8):

1. **The particle count comes off the checkpoint.** The env's particle count must equal the
   one the policy was trained with or SB3 rejects the observation space, and a wrong default
   silently changes what the encoder reads. ``--num_particles`` may only confirm it.
2. **The episode cap comes from the gym registration** (``Domain.episode_cap``), never from
   a flag default: an eval cap of 400 against an env that truncates at 200 counted every
   timeout as a success. ``--max_steps`` is an override.
3. **The env is the domain's EVAL env** -- ``Domain.make_env(training=False)``, the same
   construction the trainer's EvalCallback uses: Ant-Tag at the env's real visibility radius
   and without reward shaping, so the reward is the env's own. VecNormalize, when given, is
   loaded with ``training=False`` (statistics frozen) and ``norm_reward=False`` (a normalised
   reward would be reported in units of a running standard deviation).
4. **The extractor class must be importable before the zip is unpickled.** SB3 pickles the
   features-extractor CLASS by module path; the recorded Ant-Tag zips name the numbered
   scripts (``4_train_rl_gaussian.WeightedGaussianFeaturesExtractor``), so
   ``experiments/<domain>/`` is put on ``sys.path`` before ``PPO.load``.
5. **``run_status.json`` is read.** A run that crashed leaves ``<encoder>_agent.zip`` shaped
   like a finished one; the status file says so and the script warns.
6. **``--seed`` is re-applied AFTER ``PPO.load``.** The load restores the TRAINING seed from
   the checkpoint and ``set_random_seed`` re-seeds the vec env with it, so every evaluation of
   a checkpoint replayed one episode set whatever ``--seed`` said; four "different" eval seeds
   once returned byte-identical results. Domains whose hidden state is drawn at reset
   (Odd-Even) additionally re-seed before EVERY episode (``Evaluation.reseed_per_episode``):
   there the seed IS the episode.
7. **The rollout records every step** (reward and ``info``), and the domain's
   :class:`~set_transformer.rl.domains.base.Evaluation` record turns that into the report:
   by default the success-rate report below (an episode is a success when its final ``info``
   says ``is_success``, else when it ended strictly before the cap -- a terminal hazard can
   end an episode early without a tag, which is why ``info`` is consulted first); Odd-Even
   supplies its transient / steady split against the Bayes oracle instead. The report's
   numbers, with what was evaluated and how, are also written as one JSON file under
   ``<output root>/<domain>/<variant>/eval/`` (change 5.2; ``--summary_path`` /
   ``--no_summary``), so a result is a record and not a line to grep out of a log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from set_transformer.rl import domains as _domains
from set_transformer.rl import run_records
from set_transformer.rl.domains.base import Domain


@dataclass
class Episode:
    """One evaluated episode: the env's reward and ``info`` at every step."""

    rewards: list[float] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return float(sum(self.rewards))

    @property
    def final_info(self) -> dict:
        return self.infos[-1] if self.infos else {}

    def is_success(self, cap: int) -> bool:
        """``is_success`` from the final ``info`` when the env reports it, else "ended before
        the cap" -- the rule every Ant-Tag eval used."""
        return bool(self.final_info.get("is_success", self.length < cap))


def checkpoint_num_particles(model_path: str) -> int | None:
    """Particle-set size recorded in a saved policy's observation space, or None if it cannot
    be read (the caller's value then stands and SB3 complains on its own)."""
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _params, _other = load_from_zip_file(
            model_path, load_data=True, device="cpu", print_system_info=False)
        return int(data["observation_space"]["particles"].shape[0])
    except Exception:  # noqa: BLE001 - a best-effort default, never fatal
        return None


def rollout(model, env, n_episodes: int, cap: int, *, seed: int, deterministic: bool = True,
            reseed_per_episode: bool = False) -> list[Episode]:
    """Roll ``model`` out for ``n_episodes`` on the (single) vec env and record every step.

    An episode ends when the env says so or when it reaches ``cap`` steps, whichever is
    first; with the cap from the registration the env truncates itself at exactly that step.
    """
    episodes = []
    for index in range(n_episodes):
        if reseed_per_episode:
            env.seed(seed + index)
        obs = env.reset()
        episode = Episode()
        while True:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, dones, infos = env.step(action)
            episode.rewards.append(float(reward[0]))
            episode.infos.append(dict(infos[0]))
            if bool(dones[0]) or episode.length >= cap:
                break
        episodes.append(episode)
    return episodes


def report_success_rate(episodes: Sequence[Episode], references, args, variant, cap: int) -> dict:
    """The success-rate report every Ant-Tag eval printed (the wave drivers grep its lines).
    Returns the same numbers as a dict for the JSON summary."""
    rewards = np.array([episode.total_reward for episode in episodes])
    lengths = np.array([episode.length for episode in episodes])
    successes = np.array([episode.is_success(cap) for episode in episodes], dtype=bool)
    n = len(episodes)
    tagged = int(successes.sum())
    best_idx = int(np.argmax(rewards))

    print(f"\n=== Eval over {n} episodes (deterministic={args.deterministic}) ===")
    print(f"Success rate  : {tagged}/{n} ({100 * tagged / n:.1f}%)")
    print(f"Mean reward   : {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Mean length   : {lengths.mean():.1f} ± {lengths.std():.1f}")
    print(f"Median length : {np.median(lengths):.1f}")
    print(f"Best episode  : reward={rewards[best_idx]:.2f}, length={lengths[best_idx]}")
    summary = dict(
        n_episodes=n, successes=tagged, success_rate=tagged / n,
        mean_reward=float(rewards.mean()), std_reward=float(rewards.std()),
        mean_length=float(lengths.mean()), std_length=float(lengths.std()),
        median_length=float(np.median(lengths)),
        best_episode=dict(reward=float(rewards[best_idx]), length=int(lengths[best_idx])),
        episodes=[dict(reward=float(r), length=int(l), success=bool(ok))
                  for r, l, ok in zip(rewards, lengths, successes)])
    if tagged > 0:
        tag_lens = lengths[successes]
        print(f"When tagged   : mean_len={tag_lens.mean():.1f}, "
              f"median_len={np.median(tag_lens):.1f}")
        summary["tagged_mean_length"] = float(tag_lens.mean())
        summary["tagged_median_length"] = float(np.median(tag_lens))
    return summary


def build_parser(domain: Domain, *, prog: str | None = None,
                 selector: bool = False) -> argparse.ArgumentParser:
    """The shared flags, the domain's variant flag, then the domain's own eval flags."""
    evaluation = domain.evaluation
    parser = argparse.ArgumentParser(
        prog=prog,
        description=f"Evaluate a saved {domain.name} agent on the env's own reward.")
    if selector:
        parser.add_argument("--domain", choices=sorted(_domains.DOMAIN_NAMES), default=domain.name)
    domain.add_variant_argument(parser, default=domain.default_variant)
    # Not argparse-required: --list_variants (and a domain's references-only mode) must work
    # without a checkpoint, and argparse enforces required= before any of our code runs.
    parser.add_argument("--model_path", type=str, default=None,
                        help="Saved agent zip (<encoder>_agent.zip, best_model.zip or a "
                             "checkpoint).")
    parser.add_argument("--vecnormalize_path", type=str, default=None,
                        help="vecnormalize.pkl saved during training. Loaded with "
                             "training=False and norm_reward=False, so the reported reward is "
                             "the env's own.")
    parser.add_argument("--n_episodes", type=int, default=evaluation.default_n_episodes)
    parser.add_argument(
        "--num_particles", type=int, default=None,
        help="Particle count for the eval env. Defaults to the count baked into the "
             "checkpoint's observation space, which is the only value that can work.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Eval episode seed, re-applied AFTER PPO.load (which restores "
                             "the training seed).")
    parser.add_argument(
        "--max_steps", type=int, default=None,
        help="Episode cap of the env under evaluation; an episode ending strictly before "
             "this many steps counts as a success. Defaults to the variant's registered "
             "max_episode_steps, which is the only correct value -- a larger one counts "
             "every timeout as a success.")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", dest="deterministic", action="store_false")
    parser.add_argument(
        "--summary_path", type=str, default=None,
        help="Where the JSON summary of this evaluation goes. Default: <output root>/"
             f"{domain.name}/<variant>/eval/<timestamp>_<run dir name>_<model stem>_seed<seed>"
             "_<n>ep.json (change 5.2); --no_summary writes none.")
    parser.add_argument("--no_summary", action="store_true", help="Print only; write no JSON.")
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="Root of the shared run layout for the summary: $RL_BMDP_RUNS, else <parent "
             "repo>/runs when this checkout is a submodule, else <checkout>/runs.")
    evaluation.add_arguments(parser)
    return parser


def summary_path(domain: Domain, args) -> Path:
    """`<root>/<domain>/<variant>/eval/<timestamp>_<run dir name>_<model stem>_seed<seed>_<n>ep.json`.
    The run-dir name is the checkpoint's grandparent (``<run>/models/<zip>``) or parent
    (``<run>/models/best_model/<zip>``), so a summary names the run it measured."""
    if args.summary_path:
        return Path(args.summary_path)
    model = Path(args.model_path).resolve()
    run_name = next((p.name for p in model.parents
                     if p.name not in ("models", "best_model", "checkpoints")), model.parent.name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{run_name}_{model.stem}_seed{args.seed}_{args.n_episodes}ep.json"
    return run_records.eval_dir(domain.name, args.variant, root=args.output_root) / name


def write_summary(path: Path, domain: Domain, args, variant, cap: int, report: dict,
                  references) -> None:
    """The evaluation as a record: what was evaluated, how, and the report's numbers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        domain=domain.name, variant=args.variant, env_id=variant.env_id,
        particle_filter=variant.particle_filter.__name__, episode_cap=cap,
        model_path=str(Path(args.model_path).resolve()),
        vecnormalize_path=(str(Path(args.vecnormalize_path).resolve())
                           if args.vecnormalize_path else None),
        num_particles=args.num_particles, seed=args.seed, n_episodes=args.n_episodes,
        deterministic=args.deterministic, timestamp=datetime.now().isoformat(timespec="seconds"),
        run_status=run_records.read_run_status(args.model_path),
        report=report, references=references)
    path.write_text(json.dumps(payload, indent=1, default=_jsonable))
    print(f"Summary written to {path}")


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _select_domain(argv: Sequence[str]) -> str | None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--domain", choices=sorted(_domains.DOMAIN_NAMES), default=None)
    known, _rest = pre.parse_known_args(argv)
    return known.domain


def _make_extractor_classes_importable(domain: Domain) -> None:
    """SB3 unpickles the features-extractor class by module path. The recorded Ant-Tag zips
    name the numbered scripts (``4_train_rl_gaussian``), which live in
    ``experiments/<domain>/``; appended (not prepended) so no flat-name import of this
    process is shadowed."""
    scripts = run_records.checkout_root() / "experiments" / domain.name
    if scripts.is_dir() and str(scripts) not in sys.path:
        sys.path.append(str(scripts))


def main(argv: Sequence[str] | None = None, *, domain: Domain | str | None = None,
         prog: str | None = None):
    """Parse, build the eval env, load, roll out, report. ``domain`` given: a per-domain entry
    point (``experiments/<domain>/eval_scripts/``); not given: ``--domain`` is read from
    ``argv`` first (the ``python -m`` entry). Returns the evaluated episodes, or None after
    ``--list_variants`` / a references-only run."""
    argv = list(sys.argv[1:] if argv is None else argv)
    selector = domain is None
    if selector:
        domain = _select_domain(argv)
        if domain is None:
            if "-h" in argv or "--help" in argv:
                domain = sorted(_domains.DOMAIN_NAMES)[0]
            else:
                argparse.ArgumentParser(prog=prog).error(
                    "--domain is required (e.g. --domain ant_tag --variant smart ...)")
    domain = _domains.get(domain)
    evaluation = domain.evaluation

    parser = build_parser(domain, prog=prog, selector=selector)
    args = parser.parse_args(argv)
    if args.list_variants:
        domain.print_variants()
        return None
    references_only = bool(evaluation.references_only(args))
    if args.model_path is None and not references_only:
        parser.error("--model_path is required")

    variant = domain.resolve(args.variant)
    cap = domain.episode_cap(args.variant) if args.max_steps is None else int(args.max_steps)
    print(f"Variant: {args.variant} | env: {variant.env_id} | "
          f"filter: {variant.particle_filter.__name__} | episode cap: {cap}")

    references = evaluation.references(args, variant)
    if references_only:
        return None

    # 1. The particle count, off the checkpoint.
    trained_particles = checkpoint_num_particles(args.model_path)
    if args.num_particles is None:
        args.num_particles = (trained_particles if trained_particles is not None
                              else domain.default_num_particles(args.variant))
    elif trained_particles is not None and args.num_particles != trained_particles:
        parser.error(
            f"--num_particles {args.num_particles} contradicts the checkpoint, "
            f"which was trained with {trained_particles}. Omit the flag.")

    # 3. The domain's eval env; VecNormalize frozen and without reward normalisation.
    env = DummyVecEnv([domain.make_env(
        args.variant, num_particles=args.num_particles,
        particle_filter_class=variant.particle_filter, seed=args.seed, rank=0,
        monitor_dir=None, training=False, options=evaluation.options(args))])
    if args.vecnormalize_path and os.path.exists(args.vecnormalize_path):
        env = VecNormalize.load(args.vecnormalize_path, env)
        env.training = False
        env.norm_reward = False
        print(f"Loaded VecNormalize from {args.vecnormalize_path}")
    else:
        print("No VecNormalize -- evaluating without obs normalization. If training used "
              "it, this is INVALID: the policy sees a different input distribution than it "
              "was trained on.")

    # 4./5. Load, with the extractor class importable, and read the run status.
    _make_extractor_classes_importable(domain)
    model = PPO.load(args.model_path, env=env)
    print(f"Loaded model from {args.model_path}")
    run_status = run_records.read_run_status(args.model_path)
    if run_status is None:
        print("  (no run_status.json beside it: a run from before it was written; check "
              "its stdout.log reached 'Model saved')")
    elif run_status.get("status") != "completed":
        print("  WARNING: run_status.json says this run "
              f"{run_status.get('status', '?').upper()} at step "
              f"{run_status.get('timesteps')} of {run_status.get('total_timesteps')} "
              f"({run_status.get('error')}). The saved agent is a crash artefact, not a "
              "trained policy (PITFALLS.md section 8 item 7).")

    # 6. --seed takes effect only if applied AFTER the load.
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    print(f"Eval episode seed: {args.seed} (overriding the checkpoint's training seed"
          + ("; re-seeded per episode)" if evaluation.reseed_per_episode else ")"))

    # 7. Roll out and report.
    episodes = rollout(model, env, args.n_episodes, cap, seed=args.seed,
                       deterministic=args.deterministic,
                       reseed_per_episode=evaluation.reseed_per_episode)
    report = evaluation.report or report_success_rate
    summary = report(episodes, references, args, variant, cap)
    if not args.no_summary:
        write_summary(summary_path(domain, args), domain, args, variant, cap, summary, references)
    env.close()
    return episodes


if __name__ == "__main__":
    main(prog=Path(__file__).name)
