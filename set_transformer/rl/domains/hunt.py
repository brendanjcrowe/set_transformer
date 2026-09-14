"""The ``hunt`` domain: Cluster-Hunt, least-mass and most-var on the shared RL harness.

Plan section 9 of the parent repo's ``refactor_plans.md`` (2026-09-13). The three tasks are
``pdomains.hunt`` envs (``pdomains-cluster-hunt-v0``, ``pdomains-least-mass-v0``,
``pdomains-most-var-v0``; moved there from ``src/hunt_tasks/env`` in batch 9.0). They differ
from Ant-Tag and Odd-Even in one structural way: **the env emits the belief itself.** Its
observation is a Dict ``{"agent": [2], "particles": [100, 2]}`` -- the agent's own position and
a cloud redrawn from the live clusters at every step -- and there is no hidden state for a
filter to track. So this module does three small things and otherwise leaves the harness alone:

* :class:`HuntAgentObsWrapper` turns the Dict into the 2-D ``agent`` Box, which is what the
  policy MLP reads beside the belief features (the ``"obs"`` key of the shared wrapper);
* the shared :class:`~set_transformer.rl.wrappers.particle_filter.PFDictWithWeightsObservationWrapper`
  is given :class:`~set_transformer.rl.particle_filters.hunt.EnvEmittedBeliefFilter`, a
  pass-through that copies ``env.particles - env.pos`` (agent-relative, raw arena units) with
  uniform weights after every step -- no inference (decision B / 9c-2);
* :class:`HuntScheduleWrapper` exposes the two curriculum setters the envs have
  (``set_n_active``, rounded as the recorded callbacks rounded; ``set_hit_radius``) so the
  shared :class:`~set_transformer.rl.curriculum.ScheduleCallback` drives them.

**Frame.** Every recorded hunt arm subtracted the agent position inside its extractor and saw
the env's scaled coordinates, i.e. ``(particles - pos) / 10``. Here the filter subtracts and the
extractors divide by ``arena_scale`` = :data:`ARENA_SCALE` = 10 (``pdomains.hunt.SCALE``), so the
tensor handed to the encoder is the same one (``tests/test_hunt_domain.py`` checks it against the
original extractor's computation).

**Schedules** (the recorded curricula): least-mass / most-var grow ``n_active`` 2 -> 5 over the
first 40 % of training (``minmass/train_rl.py::Curriculum(2, 5, 0.4)``); Cluster-Hunt grows
``n_active`` 1 -> 5 and shrinks ``hit_radius`` 1.6 -> 0.6 over the same 40 %
(``train.py::CurriculumCallback`` with ``--curriculum_start 1.6 --curriculum_frac 0.4
--curriculum_n_start 1``). The eval env (``training=False``) sits at the FINAL values, which are
also the envs' registered defaults; the recorded protocol evaluated on the env's own reward with
every cluster active.

**Evaluation.** The generic success rule ("``is_success`` in info, else ended before the cap") is
wrong on these tasks: on least-mass a WRONG hit also ends the episode early. :func:`_eval_report`
reads the env's own verdict -- ``info["solved"]`` with the correct / wrong / timeout split on the
pick-a-target tasks, the number of clusters collected on Cluster-Hunt -- over 300 episodes
(the record's fixed set), re-seeded per episode because the configuration is drawn at reset.

**Collection and pretraining** (batch 9.2): the recorded behaviour policy collects labelled
snapshots through the shared collector (``Collection.snapshot_extras``), and the ``task``
objective trains encoder + head on those labels (bodies moved from
``src/hunt_tasks/pretrain/{collect,heads,pretrain}.py``); see the two sections below.

The numbers this module produces are a NEW table under the record's protocol, not a
reproduction of ``domain_mds/{cluster_hunt,least_mass}.md``: the extractors are the harness
ones, the ST reads a (constant) weight channel, and the trainer is the shared PPO loop.
``src/hunt_tasks/`` stays untouched as the record (decision 9c-6).
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import pdomains  # noqa: F401 - registers the pdomains-* env ids
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from set_transformer.rl.curriculum import Schedule, ScheduleRouter, parse_curriculum
from set_transformer.rl.domains.base import (
    Collection,
    Domain,
    Evaluation,
    Objective,
    PretrainContext,
    PretrainResult,
    Pretraining,
)
from set_transformer.rl.particle_filters.hunt import EnvEmittedBeliefFilter
from set_transformer.rl.wrappers.particle_filter import PFDictWithWeightsObservationWrapper

#: ``pdomains.hunt.SCALE``: the arena half-width the envs divide by, and the ``arena_scale`` the
#: shared extractors divide the (agent-relative, raw) particles by. Written here as a number so
#: this module loads against any pdomains checkout; ``tests/test_hunt_domain.py`` pins it.
ARENA_SCALE = 10.0

#: The two curriculum setters (targets of the schedules below), reached through the router.
SCHEDULE_TARGETS = ("set_n_active", "set_hit_radius")


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One hunt env plus the pass-through filter, its task kind and its recorded schedules."""

    env_id: str
    particle_filter: type
    #: ``collect_all`` (Cluster-Hunt: visit every cluster) or ``pick_target`` (least-mass /
    #: most-var: one hit ends the episode, right or wrong). Selects the evaluation report.
    task: str
    #: What the belief looks like here and what the target is. Printed by --list_variants.
    notes: str = ""
    #: ``frac:n,...`` schedule for the number of live clusters (rounded to the nearest integer
    #: before it reaches the env, as the recorded callbacks did). None = no curriculum.
    default_n_active_curriculum: str | None = None
    #: ``frac:radius,...`` schedule for the hit radius. None = the env's registered radius.
    default_hit_radius_curriculum: str | None = None
    #: The recorded horizon for this task, used when --total_timesteps is left at the domain
    #: default (3M): Cluster-Hunt ran 1.5M.
    default_total_timesteps: int = 3_000_000


VARIANTS: dict[str, Variant] = {
    "cluster_hunt": Variant(
        env_id="pdomains-cluster-hunt-v0",
        particle_filter=EnvEmittedBeliefFilter,
        task="collect_all",
        notes=("Visit every live cluster. Cloud = 100 particles from the alive clusters "
               "(equal counts), redrawn after each collection. Recorded RL config: hit_radius "
               "0.6, min_sep 2.5, 60 steps, 1.5M steps (domain_mds/cluster_hunt.md)."),
        default_n_active_curriculum="0:1,0.4:5,1:5",
        default_hit_radius_curriculum="0:1.6,0.4:0.6,1:0.6",
        default_total_timesteps=1_500_000,
    ),
    "least_mass": Variant(
        env_id="pdomains-least-mass-v0",
        particle_filter=EnvEmittedBeliefFilter,
        task="pick_target",
        notes=("Go to the LIGHTEST cluster (fewest particles, mass_margin 0.06); a hit on any "
               "other cluster ends the episode as 'wrong'. Cloud redrawn every step. Recorded: "
               "3M steps, Curriculum(2, 5, 0.4) (domain_mds/least_mass.md)."),
        default_n_active_curriculum="0:2,0.4:5,1:5",
        default_total_timesteps=3_000_000,
    ),
    "most_var": Variant(
        env_id="pdomains-most-var-v0",
        particle_filter=EnvEmittedBeliefFilter,
        task="pick_target",
        notes=("Go to the WIDEST cluster (largest sigma, margin 0.2 on [0.30, 0.90]); counts "
               "equal, so mass carries no information. Same env class and protocol as "
               "least_mass; new in 2026-09 (plan 9c-1), no record yet."),
        default_n_active_curriculum="0:2,0.4:5,1:5",
        default_total_timesteps=3_000_000,
    ),
}


def resolve(name: str) -> Variant:
    """Look up a variant, listing the valid names on a typo."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"Unknown variant {name!r}. Available: {sorted(VARIANTS)}") from None


def episode_cap(name: str) -> int:
    """The variant's episode cap, read from its gym registration (60 for all three)."""
    spec = gym.spec(resolve(name).env_id)
    if spec.max_episode_steps is None:
        raise ValueError(f"{spec.id} registers no max_episode_steps; pass --max_steps")
    return int(spec.max_episode_steps)


def run_subdir(encoder: str, name: str) -> str:
    """The legacy-layout folder name for this (encoder, variant) pair."""
    resolve(name)
    return f"hunt_{encoder}_{name}"


def add_variant_argument(parser, default: str = "least_mass") -> None:
    parser.add_argument(
        "--variant", type=str, default=default, choices=sorted(VARIANTS),
        help=f"Hunt task (default: {default}). Selects env id, task kind, schedules, run "
             "subdirectory and episode cap together.")
    parser.add_argument("--list_variants", action="store_true",
                        help="Print the variant registry and exit.")


def print_variants() -> None:
    for name, variant in VARIANTS.items():
        try:
            cap = episode_cap(name)
        except Exception:  # noqa: BLE001 - listing must not fail on one bad entry
            cap = "?"
        print(f"{name:14s} {variant.env_id:30s} cap={cap:<5} task={variant.task:12s} "
              f"{variant.particle_filter.__name__}")
        if variant.notes:
            print(f"{'':14s}   {variant.notes}")


def _env_config(name: str):
    """The live env's config dataclass (``n_particles``, ``hit_radius``, ...) for a variant."""
    env = gym.make(resolve(name).env_id)
    try:
        return env.unwrapped.cfg
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class HuntAgentObsWrapper(gym.ObservationWrapper):
    """Dict ``{"agent", "particles"[, "oracle"]}`` -> the 2-D ``agent`` Box.

    The cloud stays on the unwrapped env (``env.unwrapped.particles``, RAW [0, 20]^2 units),
    where the pass-through filter reads it; the agent position the env emits is already
    centred and scaled to [-1, 1], as the recorded arms saw it. An env built with
    ``include_oracle`` would leak the privileged key into the base obs, so it is refused.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Dict) or "agent" not in space.spaces:
            raise TypeError(f"expected a Dict observation space with an 'agent' key, got {space}")
        if "oracle" in space.spaces:
            raise ValueError("the env was built with include_oracle=True; the privileged "
                             "oracle key must not reach the policy through the base obs")
        self.observation_space = space.spaces["agent"]

    def observation(self, observation):
        return np.asarray(observation["agent"], dtype=np.float32)


class HuntScheduleWrapper(gym.Wrapper):
    """The two curriculum setters, as the shared ScheduleCallback calls them.

    ``set_n_active`` receives an interpolated FLOAT from the callback and rounds it to the
    nearest integer before the env's own setter (which truncates) sees it -- what both recorded
    callbacks did (``int(round(n_start + p * (n_end - n_start)))``). ``set_hit_radius`` forwards.
    """

    def set_n_active(self, n: float) -> None:
        self.env.unwrapped.set_n_active(int(round(float(n))))

    def set_hit_radius(self, r: float) -> None:
        self.env.unwrapped.set_hit_radius(float(r))

    @property
    def n_active(self) -> int:
        return int(self.env.unwrapped.cfg.n_active)

    @property
    def hit_radius(self) -> float:
        return float(self.env.unwrapped.cfg.hit_radius)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def make_schedules(n_active_curriculum: str | None,
                   hit_radius_curriculum: str | None) -> tuple[Schedule, ...]:
    """The Schedule records for one run; ``None`` or ``"none"`` leaves a quantity unscheduled."""
    schedules = []
    if n_active_curriculum and n_active_curriculum.lower() != "none":
        schedules.append(Schedule("n_active", target="set_n_active",
                                  waypoints=tuple(parse_curriculum(n_active_curriculum)),
                                  labels=(("n_active", ".2f"),)))
    if hit_radius_curriculum and hit_radius_curriculum.lower() != "none":
        schedules.append(Schedule("hit_radius", target="set_hit_radius",
                                  waypoints=tuple(parse_curriculum(hit_radius_curriculum)),
                                  labels=(("hit_radius", ".3f"),)))
    return tuple(schedules)


def final_values(schedules) -> dict[str, float]:
    """``setter name -> value at progress 1`` -- what the eval env is set to."""
    return {schedule.target: schedule.values_at(1.0)[0] for schedule in schedules}


def variant_schedules(name: str) -> tuple[Schedule, ...]:
    variant = resolve(name)
    return make_schedules(variant.default_n_active_curriculum,
                          variant.default_hit_radius_curriculum)


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


def make_hunt_belief_env(
    num_particles: int,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    variant: str = "least_mass",
    env_id: str | None = None,
    particle_filter_class: type | None = None,
    schedule_values: dict[str, float] | None = None,
):
    """Return a callable that builds one hunt belief env.

    Stack, inside out: ``gym.make(env_id)`` (TimeLimit at the registered cap) ->
    :class:`HuntAgentObsWrapper` -> :class:`PFDictWithWeightsObservationWrapper` with the
    pass-through filter reading the unwrapped env -> :class:`HuntScheduleWrapper` -> Monitor ->
    :class:`ScheduleRouter` over the two setters. ``schedule_values`` (``setter -> value``) are
    applied once after construction: the eval env is built with the schedules' final values.
    """
    resolved = resolve(variant)
    env_id = env_id or resolved.env_id
    particle_filter_class = particle_filter_class or resolved.particle_filter
    schedule_values = dict(schedule_values or {})

    def _init():
        # Registered HERE too, not only at module import: this closure is cloudpickled by
        # value into a SubprocVecEnv child, which never runs this module's top-level import.
        import pdomains  # noqa: F401,PLC0415

        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        unwrapped = env.unwrapped
        env_n = int(unwrapped.cfg.n_particles)
        if env_n != num_particles:
            raise ValueError(
                f"{env_id} emits {env_n} particles but the run asks for num_particles="
                f"{num_particles}. The belief is the env's own cloud, so the two must agree; "
                f"pass --num_particles {env_n} or leave it to the domain default.")
        env = HuntAgentObsWrapper(env)
        env = PFDictWithWeightsObservationWrapper(
            env=env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs={"env": unwrapped},
            num_particles=num_particles,
            pf_interaction_mapper=None,
            obs_mask_indices=None,
            # Nothing in the pass-through is random; the seed is passed for uniformity with
            # the other domains (the wrapper derives one per episode).
            particle_filter_seed=seed + rank,
        )
        env = HuntScheduleWrapper(env)
        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        else:
            env = Monitor(env)
        env = ScheduleRouter(env, targets=SCHEDULE_TARGETS)
        for target, value in schedule_values.items():
            getattr(env, target)(value)
        return env

    return _init


def _make_vec_env_from_fns(env_fns, n_envs: int):
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def _make_vec_normalize(vec_env, training: bool, norm_reward: bool):
    """Normalise the base obs (the agent position) and the reward, never the PF weights."""
    return VecNormalize(vec_env, training=training, norm_obs=True, norm_reward=norm_reward,
                        norm_obs_keys=["obs"])


# ---------------------------------------------------------------------------
# The domain's flags, schedules and env hook
# ---------------------------------------------------------------------------


def _add_arguments(parser) -> None:
    parser.add_argument(
        "--n_active_curriculum", type=str, default=None,
        help="Schedule 'frac:n,...' for the number of live clusters, rounded to the nearest "
             "integer (the recorded rule). Default: the variant's (least_mass / most_var "
             "'0:2,0.4:5,1:5'; cluster_hunt '0:1,0.4:5,1:5'). 'none' disables it.")
    parser.add_argument(
        "--hit_radius_curriculum", type=str, default=None,
        help="Schedule 'frac:radius,...' for the hit radius. Default: the variant's "
             "(cluster_hunt '0:1.6,0.4:0.6,1:0.6'; none on least_mass / most_var, whose "
             "registered radius 0.6 stands). 'none' disables it.")


def _resolve_arguments(parser, args) -> dict:
    """CLI > variant for the two schedule strings; the variant's recorded horizon when
    --total_timesteps was left at the domain default. Returns the (empty) env options."""
    variant = resolve(args.variant)
    if args.n_active_curriculum is None:
        args.n_active_curriculum = variant.default_n_active_curriculum
    if args.hit_radius_curriculum is None:
        args.hit_radius_curriculum = variant.default_hit_radius_curriculum
    if args.total_timesteps == parser.get_default("total_timesteps") \
            and variant.default_total_timesteps != args.total_timesteps:
        args.total_timesteps = variant.default_total_timesteps
        print(f"--total_timesteps left at the domain default; using the {args.variant} "
              f"record's {args.total_timesteps:,}")
    make_schedules(args.n_active_curriculum, args.hit_radius_curriculum)   # parse now
    return {}


def _schedules(args):
    return make_schedules(args.n_active_curriculum, args.hit_radius_curriculum)


def _run_config_extras(args) -> dict:
    variant = resolve(args.variant)
    return dict(episode_cap=episode_cap(args.variant), task=variant.task,
                env_kwargs=dict(gym.spec(variant.env_id).kwargs))


def _make_env(variant: str, *, num_particles: int, particle_filter_class: type, seed: int,
              rank: int, monitor_dir: str | None, training: bool, options: dict):
    """One worker's env. The eval env (``training=False``) is set to the variant's schedules'
    FINAL values (every cluster active, the registered hit radius): the recorded protocol."""
    values = None if training else final_values(variant_schedules(variant))
    return make_hunt_belief_env(
        num_particles=num_particles, rank=rank, seed=seed, monitor_dir=monitor_dir,
        variant=variant, particle_filter_class=particle_filter_class, schedule_values=values)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _eval_report(episodes, references, args, variant, cap) -> dict:
    """The env's own verdict, not the generic early-termination rule.

    ``pick_target`` (least_mass, most_var): success = final ``info["solved"]``; the outcome
    split correct / wrong / timeout the record prints (a wrong hit ends the episode early and
    is NOT a success). ``collect_all`` (cluster_hunt): clusters collected out of ``n_active``,
    and the fraction of episodes that collected every one (the env's ``solved``).
    """
    n = len(episodes)
    rewards = np.array([episode.total_reward for episode in episodes], dtype=float)
    lengths = np.array([episode.length for episode in episodes], dtype=int)
    finals = [episode.final_info for episode in episodes]
    solved = np.array([bool(info.get("solved", False)) for info in finals], dtype=bool)
    summary = dict(
        task=variant.task, n_episodes=n, successes=int(solved.sum()),
        success_rate=float(solved.mean()) if n else float("nan"),
        mean_reward=float(rewards.mean()), std_reward=float(rewards.std()),
        mean_length=float(lengths.mean()), std_length=float(lengths.std()))
    print(f"\n=== Eval over {n} episodes (deterministic={args.deterministic}, "
          f"task={variant.task}) ===")
    if variant.task == "pick_target":
        # An episode the rollout cut at --max_steps before the env's own end has no
        # outcome; it is a timeout for the report's purposes.
        outcomes = [info.get("outcome", "timeout") for info in finals]
        counts = {k: int(sum(o == k for o in outcomes)) for k in ("correct", "wrong", "timeout")}
        print(f"Success (correct cluster): {counts['correct']}/{n} "
              f"({100 * summary['success_rate']:.1f}%)")
        print(f"Outcomes      : correct {counts['correct']}, wrong {counts['wrong']}, "
              f"timeout {counts['timeout']}")
        summary.update(outcome_counts=counts,
                       **{f"outcome_{k}": (v / n if n else float("nan"))
                          for k, v in counts.items()})
        per_episode = [dict(reward=float(r), length=int(l), success=bool(s), outcome=o)
                       for r, l, s, o in zip(rewards, lengths, solved, outcomes)]
    else:
        collected = np.array([int(info.get("episode_collected", info.get("n_collected_total", 0)))
                              for info in finals], dtype=int)
        active = np.array([int(info.get("n_active", 0)) for info in finals], dtype=int)
        fraction = collected / np.maximum(active, 1)
        print(f"Collected     : {collected.mean():.2f} of {active.mean():.1f} clusters "
              f"({100 * fraction.mean():.1f}%)")
        print(f"All collected : {int(solved.sum())}/{n} ({100 * summary['success_rate']:.1f}%)")
        summary.update(mean_collected=float(collected.mean()), mean_n_active=float(active.mean()),
                       fraction_collected=float(fraction.mean()))
        per_episode = [dict(reward=float(r), length=int(l), success=bool(s), collected=int(c))
                       for r, l, s, c in zip(rewards, lengths, solved, collected)]
    print(f"Mean reward   : {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Mean length   : {lengths.mean():.1f} ± {lengths.std():.1f} (cap {cap})")
    summary["episodes"] = per_episode
    return summary


# ---------------------------------------------------------------------------
# Dataset collection (batch 9.2, 2026-09-13). The shared loop is rl/collect.py; what is here is
# what only the hunt tasks know, moved from src/hunt_tasks/pretrain/collect.py of the parent repo:
# the behaviour policy (an OU random walk or noisy pursuit of a random cluster), the per-episode
# draw of the task size, and the LABELS each snapshot carries (cluster centres, alive flags,
# counts, widths, the target) through the Collection.snapshot_extras hook. Particles are stored as
# the pass-through filter emits them -- agent-relative, raw arena units, particle_scale 10, no
# centre -- so the generic reconstruction objective reads the file unchanged and the `task`
# objective below reads the labels. No rebalancing (the record had none).
#
# One difference from the record: the shared loop also records the state AFTER the terminal step,
# so a harness dataset has one more row per episode than pretrain/collect.py produced (which
# stopped before it); on Cluster-Hunt that last row has no live cluster, so its target is zero
# and target_index -1. tests/test_hunt_pretrain.py drops those rows before comparing.
# ---------------------------------------------------------------------------

#: The record's per-episode draw of how many clusters spawn: the RL curriculum covers 1 -> 5
#: (Cluster-Hunt) / 2 -> 5, so the data covers all of it, weighted toward the full task.
N_ACTIVE_CHOICES = {"collect_all": (1, 2, 3, 4, 5, 5, 5), "pick_target": (2, 3, 4, 5, 5, 5)}


def _behaviour(kind, rng, pos, goal, prev):
    """Return an action in [-1,1]^2.

    'walk'    : an OU random walk. Independent uniform actions would average out
                and leave the agent near where it started, so the walk is
                correlated in time and actually travels.
    'pursuit' : unit step toward `goal` plus noise.
    """
    if kind == "walk":
        a = 0.8 * prev + 0.6 * rng.standard_normal(2)
    else:
        d = goal - pos
        n = np.linalg.norm(d)
        a = (d / n if n > 1e-6 else rng.standard_normal(2)) + 0.35 * rng.standard_normal(2)
    return np.clip(a, -1.0, 1.0)


def _collect_add_arguments(parser) -> None:
    parser.add_argument("--pursuit_frac", type=float, default=0.6,
                        help="Fraction of episodes driven by noisy pursuit of a random cluster; "
                             "the rest are OU random walks (the record's 0.6).")
    parser.add_argument("--n_active_choices", type=str, default=None,
                        help="Comma list the number of live clusters is drawn from per episode. "
                             "Default: the record's (cluster_hunt 1,2,3,4,5,5,5; least_mass / "
                             "most_var 2,3,4,5,5,5).")


def _collect_resolve_arguments(parser, args, domain) -> dict:
    variant = resolve(args.variant)
    if args.timesteps is None:
        args.timesteps = episode_cap(args.variant)
    if args.num_particles is None:
        args.num_particles = int(_env_config(args.variant).n_particles)
    if args.n_active_choices:
        choices = tuple(int(x) for x in args.n_active_choices.split(","))
    else:
        choices = N_ACTIVE_CHOICES[variant.task]
    print(f"Variant: {args.variant} | env: {variant.env_id} | task: {variant.task} | "
          f"cap={args.timesteps} | n_active per episode from {choices}")
    return {"env_id": variant.env_id, "particle_filter_class": variant.particle_filter,
            "n_active_choices": choices, "task": variant.task}


def _collect_prepare(args, options):
    from types import SimpleNamespace   # noqa: PLC0415
    return SimpleNamespace(rng=np.random.default_rng(args.seed), kinds={"walk": 0, "pursuit": 0})


def _collect_make_env(args, options, state):
    return make_hunt_belief_env(num_particles=args.num_particles, rank=0, seed=args.seed,
                                variant=args.variant,
                                particle_filter_class=options["particle_filter_class"])()


def _collect_begin_episode(args, options, state, env, episode):
    """The record's draw order: task size, reset seed, behaviour kind; the goal is drawn on the
    first action (it needs the centres the reset produces); Cluster-Hunt re-draws the goal at
    5 % per step or when its cluster is gone."""
    rng = state.rng
    unwrapped = env.unwrapped
    env.set_n_active(int(rng.choice(options["n_active_choices"])))
    reset_seed = int(rng.integers(1 << 30))
    kind = "pursuit" if rng.random() < args.pursuit_frac else "walk"
    state.kinds[kind] += 1
    memo = {"prev": np.zeros(2), "goal": None}
    collect_all = options["task"] == "collect_all"

    def act(obs):
        if memo["goal"] is None:
            # Drawn on the first action: it needs the centres the reset produced. The record
            # drew it right after reset and then entered the loop, whose goal-switch check
            # below ran on the first state as well.
            n_live = int(unwrapped.n_active) if collect_all else int(unwrapped.k)
            memo["goal"] = unwrapped.centers[rng.integers(n_live)]
        if collect_all:
            live = np.flatnonzero(unwrapped.alive)
            if rng.random() < 0.05 or not unwrapped.alive[np.argmin(
                    np.linalg.norm(unwrapped.centers - memo["goal"], axis=-1))]:
                memo["goal"] = unwrapped.centers[rng.choice(live)]
        memo["prev"] = _behaviour(kind, rng, unwrapped.pos, memo["goal"], memo["prev"])
        return memo["prev"]

    return {"seed": reset_seed}, act


def _collect_snapshot_extras(args, options, state, env, obs, step_index) -> dict:
    """The labels of one snapshot, in the record's layout: K = n_clusters slots."""
    u = env.unwrapped
    K = int(u.cfg.n_clusters)
    centers = np.zeros((K, 2), np.float32)
    alive = np.zeros(K, np.float32)
    counts = np.zeros(K, np.float32)
    sigmas = np.zeros(K, np.float32)
    if options["task"] == "collect_all":
        # Every centre (the record labelled all five, dead ones included) with its alive flag.
        centers[:] = ((u.centers - u.pos) / ARENA_SCALE).astype(np.float32)
        alive[:] = u.alive.astype(np.float32)
        sigmas[:] = u.sigmas
        counts[:] = u._particle_counts()
        live = np.flatnonzero(u.alive)
        if len(live):
            d = np.linalg.norm(u.centers[live] - u.pos, axis=-1)
            j = int(live[np.argmin(d)])
            target = ((u.centers[j] - u.pos) / ARENA_SCALE).astype(np.float32)
        else:
            j, target = -1, np.zeros(2, np.float32)
    else:
        k = int(u.k)
        centers[:k] = (u.centers - u.pos) / ARENA_SCALE
        alive[:k] = 1.0
        counts[:k] = u.counts
        sigmas[:k] = u.sigmas
        j = int(u.target)
        target = ((u.centers[j] - u.pos) / ARENA_SCALE).astype(np.float32)
    return dict(agent=np.asarray(obs["obs"], np.float32), centers=centers, alive=alive,
                counts=counts, sigmas=sigmas, target=target,
                target_index=np.int64(j), step=np.int32(step_index))


def _collect_finish(args, options, state, n_snapshots) -> None:
    print(f"Episodes -- " + ", ".join(f"{k}: {v}" for k, v in state.kinds.items())
          + f"; snapshots (raw): {n_snapshots}")


def _collect_report(args, options, particles, weights, steps, stage) -> None:
    if stage != "final":
        return
    print(f"Dataset: particles {particles.shape} (agent-relative, raw units), weights "
          f"{weights.shape} (uniform)")
    for dim in range(particles.shape[-1]):
        column = particles[:, :, dim]
        print(f"  dim {dim}: [{column.min():.2f}, {column.max():.2f}]")
    print(f"  particle_scale (recorded for pretraining): {ARENA_SCALE}")


def _collect_metadata_extras(args, options, particles, weights, steps) -> dict:
    return dict(task=options["task"], episode_cap=episode_cap(args.variant),
                n_clusters=int(_env_config(args.variant).n_clusters),
                pursuit_frac=args.pursuit_frac, n_active_choices=list(options["n_active_choices"]),
                env_kwargs=dict(gym.spec(resolve(args.variant).env_id).kwargs),
                label_arrays=["agent", "centers", "alive", "counts", "sigmas", "target",
                              "target_index", "step"])


HUNT_COLLECTION = Collection(
    add_arguments=_collect_add_arguments,
    defaults={"seed": 7, "num_episodes": 4000},
    resolve_arguments=_collect_resolve_arguments,
    particle_scale=lambda args, options: ARENA_SCALE,
    prepare=_collect_prepare,
    make_env=_collect_make_env,
    begin_episode=_collect_begin_episode,
    finish=_collect_finish,
    report=_collect_report,
    metadata_extras=_collect_metadata_extras,
    snapshot_extras=_collect_snapshot_extras,
    progress_desc="Collecting hunt episodes",
)


# ---------------------------------------------------------------------------
# The `task` pretraining objective (batch 9.2): the encoder + a 3-layer head trained on the
# dataset's labels, as src/hunt_tasks/pretrain/{heads,pretrain}.py did (bodies moved). The
# encoder is the harness extractor built through the shared encoder table, so the checkpoint
# loads in rl/train.py without translation; the head reads the belief features only.
#   pick_target (least_mass, most_var): MSE from the head to the agent-relative offset of the
#                                       target cluster (the stage-1 probe objective).
#   collect_all (cluster_hunt):         permutation-matched slot loss over K (dx, dy, alive).
# The record's control, Chamfer reconstruction, is the generic objective with
# `--loss_type chamfer --ignore_weights` (decision 9c-7).
# ---------------------------------------------------------------------------

TASK_ENCODERS = ("st", "cgf", "deepset", "pointnet")

_PERM_CACHE: dict = {}


def perms(k: int, device) -> torch.Tensor:
    """All permutations of ``range(k)`` as a [k!, k] index tensor, cached per device
    (moved from src/mode_recovery_probe/probe2d/metrics2d.py)."""
    key = (k, str(device))
    if key not in _PERM_CACHE:
        _PERM_CACHE[key] = torch.tensor(list(itertools.permutations(range(k))),
                                        device=device, dtype=torch.long)
    return _PERM_CACHE[key]


def matched_slot_loss(off_pred, alive_logit, off_tgt, alive_tgt):
    """Permutation-invariant loss over K unordered cluster slots.

    Position error counts only for clusters that are actually there; the alive
    flag is scored for every slot. K = 5 gives 120 permutations, which is one
    vectorized reduction, so no assignment solver is needed.
    """
    B, K, _ = off_pred.shape
    P = perms(K, off_pred.device)                                  # [Pn, K]
    op = off_pred[:, P, :]                                         # [B, Pn, K, 2]
    ap = alive_logit[:, P]                                         # [B, Pn, K]
    pos = ((op - off_tgt[:, None]) ** 2).sum(-1) * alive_tgt[:, None]
    bce = F.binary_cross_entropy_with_logits(
        ap, alive_tgt[:, None].expand_as(ap), reduction="none")
    cost = (pos + 0.1 * bce).mean(-1)                              # [B, Pn]
    best = cost.argmin(1)
    return cost[torch.arange(B, device=cost.device), best].mean()


class TaskData:
    """The dataset on the device, split episode-disjoint: states arrive in rollout order and
    neighbouring states within an episode are near duplicates, so a random split leaks and the
    validation metric reads far higher than the encoder deserves (the record's rule: the last
    ``val_frac`` of rows are validation)."""

    LABELS = ("centers", "alive", "counts", "sigmas", "target", "target_index")

    def __init__(self, path: str, val_frac: float, device: torch.device):
        with np.load(path, allow_pickle=True) as z:
            missing = [k for k in ("particles", "weights", "agent", *self.LABELS) if k not in z.files]
            if missing:
                raise ValueError(f"{path} is not a hunt dataset: missing arrays {missing} "
                                 "(collect it with python -m set_transformer.rl.collect --domain hunt)")
            self.metadata = json.loads(str(z["metadata"])) if "metadata" in z.files else {}
            arrays = {k: np.asarray(z[k]) for k in ("particles", "weights", "agent", *self.LABELS)}
        n = len(arrays["particles"])
        n_val = int(val_frac * n)
        if n_val < 1 or n - n_val < 1:
            raise ValueError(f"{n} rows cannot be split with val_frac={val_frac}")
        self.particle_scale = float(self.metadata.get("particle_scale", ARENA_SCALE))
        self.n_train, self.n_val = n - n_val, n_val
        self.device = device
        self.train = {k: torch.as_tensor(v[:n - n_val]).to(device) for k, v in arrays.items()}
        self.val = {k: torch.as_tensor(v[n - n_val:]).to(device) for k, v in arrays.items()}
        self.num_particles = int(arrays["particles"].shape[1])
        self.dim_particles = int(arrays["particles"].shape[2])

    @staticmethod
    def obs(split: dict, index) -> dict:
        """What the RL extractor is handed: the agent position, the RAW agent-relative cloud
        (the extractor divides by arena_scale) and the (uniform) weights."""
        return {"obs": split["agent"][index], "particles": split["particles"][index],
                "weights": split["weights"][index]}


class TaskEncoderWithHead(nn.Module):
    """extractor -> belief features (the agent passthrough dropped) -> 3-layer head."""

    def __init__(self, extractor: nn.Module, obs_dim: int, out_dim: int, hidden: int = 256):
        super().__init__()
        self.extractor = extractor
        self.obs_dim = obs_dim
        d = extractor.features_dim - obs_dim
        self.head = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))

    def features(self, obs: dict) -> torch.Tensor:
        return self.extractor(obs)[:, self.obs_dim:]

    def forward(self, obs: dict) -> torch.Tensor:
        return self.head(self.features(obs))


def task_loss(y: torch.Tensor, batch: dict, task: str, n_clusters: int) -> torch.Tensor:
    if task == "collect_all":
        y = y.view(-1, n_clusters, 3)
        return matched_slot_loss(y[..., :2], y[..., 2], batch["centers"], batch["alive"])
    return F.mse_loss(y, batch["target"])


@torch.no_grad()
def task_metrics(y: torch.Tensor, batch: dict, task: str, n_clusters: int,
                 scale: float) -> dict:
    """The numbers that decide whether pretraining worked, in arena units (the record's
    `PretrainModel.evaluate`, moved): identification rate and location error on the pick-a-target
    tasks; centre error and alive accuracy on Cluster-Hunt."""
    if task == "collect_all":
        y = y.view(-1, n_clusters, 3)
        P = perms(n_clusters, y.device)
        op = y[:, P, :2]
        al = y[:, P, 2]
        pos = ((op - batch["centers"][:, None]) ** 2).sum(-1) * batch["alive"][:, None]
        bce = F.binary_cross_entropy_with_logits(
            al, batch["alive"][:, None].expand_as(al), reduction="none")
        best = (pos + 0.1 * bce).mean(-1).argmin(1)
        idx = torch.arange(len(y), device=y.device)
        e = scale * torch.linalg.norm(op[idx, best] - batch["centers"], dim=-1)
        m = batch["alive"] > 0
        return {"centre_mae": float(e[m].mean()),
                "alive_acc": float(((al[idx, best] > 0).float() == batch["alive"]).float().mean())}
    err = scale * torch.linalg.norm(y - batch["target"], dim=-1)
    d = torch.linalg.norm(batch["centers"] - y[:, None, :], dim=-1)
    d = d.masked_fill(batch["alive"] < 0.5, float("inf"))
    return {"loc_mae": float(err.mean()),
            "identify_acc": float((d.argmin(1) == batch["target_index"]).float().mean()),
            "within_1.0": float((err < 1.0).float().mean())}


def build_task_extractor(args, space: gym.spaces.Dict, scale: float):
    """The encoder under pretraining, as the RL arm's own SB3 extractor class, built from the
    flags THROUGH THE SHARED ENCODER TABLE (rl/encoders.py): the same construction
    `rl/train.py --encoder <name>` performs."""
    from set_transformer.rl import encoders as _encoders   # noqa: PLC0415 - see odd_even.build_extractor
    encoder = _encoders.get(args.encoder)
    kwargs = encoder.extractor_kwargs(args)
    kwargs["arena_scale"] = float(scale)
    kwargs[encoder.extractor_class.PRETRAINED_PATH_KWARG] = None
    return encoder.extractor_class(space, **kwargs)


def extractor_geometry(extractor) -> dict:
    for attr in ("_st_geometry", "_cgf_geometry", "_geometry"):
        if hasattr(extractor, attr):
            return dict(getattr(extractor, attr))
    return {}


def save_task_checkpoint(model: TaskEncoderWithHead, path: Path, args, epoch: int, val: dict,
                         geometry: dict, task: str) -> None:
    """The format each extractor's own loader reads. ST: encoder keys under `set_transformer.`;
    CGF: the extractor's whole state_dict (t, norm statistics and readout ARE the encoder);
    deepset / pointnet: the encoder under `encoder.` with `particle_scale` top-level."""
    extractor = model.extractor
    if args.encoder == "st":
        state = {f"set_transformer.{k}": v.detach().cpu() for k, v in extractor.encoder.state_dict().items()}
    elif args.encoder == "cgf":
        state = {k: v.detach().cpu() for k, v in extractor.state_dict().items()}
    else:
        state = {f"encoder.{k}": v.detach().cpu() for k, v in extractor.encoder.state_dict().items()}
    config = {**geometry, "objective": "task", "task": task, "variant": args.variant,
              "arena_scale": float(extractor.arena_scale), "encoder": args.encoder,
              "encoder_params": int(getattr(args, "encoder_params", 0)),
              "pretraining": "set_transformer.rl.pretrain --domain hunt --objective task"}
    if args.encoder in ("deepset", "pointnet"):
        config["weighted_particles"] = bool(extractor.weight_channel)
    torch.save({
        "model_state_dict": state,
        "head_state_dict": {k: v.detach().cpu() for k, v in model.head.state_dict().items()},
        "config": config,
        "particle_scale": float(extractor.arena_scale),
        "epoch": epoch,
        "val": val,
        "args": vars(args),
    }, path)


def _task_add_arguments(parser, domain=None) -> None:
    g = parser.add_argument_group("task objective: data")
    g.add_argument("--data_path", type=str, default=None,
                   help="The collected hunt dataset (.npz with the label arrays). Default: the "
                        "variant's dataset under the output root.")
    g.add_argument("--val_frac", type=float, default=0.1,
                   help="Last fraction of rows (rollout order, so episode-disjoint) held out.")
    t = parser.add_argument_group("task objective: training (the record's settings)")
    t.add_argument("--num_epochs", type=int, default=120)
    t.add_argument("--patience", type=int, default=15,
                   help="Stop after this many epochs without a validation improvement.")
    t.add_argument("--batch_size", type=int, default=256)
    t.add_argument("--learning_rate", type=float, default=1e-3)
    t.add_argument("--head_hidden", type=int, default=256)


def _task_locate(args) -> dict:
    if not args.data_path:
        return {"variant": None, "env_id": None}
    from set_transformer.rl.pretrain_objectives.reconstruction import dataset_metadata  # noqa: PLC0415
    meta = dataset_metadata(args.data_path)
    return {"variant": meta.get("variant"), "env_id": meta.get("env_id")}


def _task_resolve_arguments(parser, args, domain, encoder) -> None:
    if encoder.name not in TASK_ENCODERS:
        parser.error(f"the task objective is implemented for --encoder "
                     f"{' | '.join(TASK_ENCODERS)}, not {encoder.name!r}")
    if args.data_path is None:
        if args.variant is None:
            parser.error("--data_path or --variant is needed: the task objective reads the "
                         "collected dataset")
        from set_transformer.rl import run_records   # noqa: PLC0415
        args.data_path = str(run_records.dataset_path(domain.name, args.variant, root=args.output_root))
        print(f"--data_path not given; the variant's dataset under the root: {args.data_path}")
    if not os.path.isfile(args.data_path):
        parser.error(f"dataset {args.data_path} does not exist (collect it with "
                     "python -m set_transformer.rl.collect --domain hunt --variant <v>)")


def _task_prepare(parser, args, domain, encoder, device):
    data = TaskData(args.data_path, args.val_frac, torch.device(device))
    recorded = data.metadata.get("variant")
    if recorded is not None and args.variant is not None and recorded != args.variant:
        parser.error(f"--variant {args.variant} but the dataset was collected on {recorded!r}")
    if args.variant is None:
        args.variant = recorded
    args.num_particles = data.num_particles
    args.dim_particles = data.dim_particles
    args.arena_scale = data.particle_scale
    return data


def _task_run_name(args, now: datetime) -> str:
    return f"{now.strftime('%Y%m%d_%H%M%S')}_task_{args.encoder}_seed{args.seed}"


def _task_run(args, ctx: PretrainContext) -> PretrainResult:
    """The training loop of src/hunt_tasks/pretrain/pretrain.py::main, moved: Adam, halve the
    rate on a plateau, early stop, keep the best encoder WITH its matching head."""
    args.encoder = ctx.encoder.name
    data: TaskData = ctx.data
    device = torch.device(ctx.device)
    variant = resolve(args.variant)
    task = variant.task
    n_clusters = int(_env_config(args.variant).n_clusters)
    run_dir = Path(ctx.run_dir)
    checkpoint_dir = Path(ctx.checkpoint_dir) if ctx.checkpoint_dir else run_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (data.num_particles, data.dim_particles), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (data.num_particles,), np.float32),
    })
    extractor = build_task_extractor(args, space, data.particle_scale)
    geometry = extractor_geometry(extractor)
    out_dim = n_clusters * 3 if task == "collect_all" else 2
    model = TaskEncoderWithHead(extractor, obs_dim=2, out_dim=out_dim, hidden=args.head_hidden).to(device)
    n_encoder = sum(p.numel() for p in extractor.encoder_parameters())
    args.encoder_params = int(n_encoder)
    print(f"[{args.variant}/task/{args.encoder}] train={data.n_train:,} val={data.n_val:,} "
          f"encoder params={n_encoder:,} head out={out_dim} device={device}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    def val_loss():
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, data.n_val, 2048):
                s = slice(i, min(i + 2048, data.n_val))
                b = {k: v[s] for k, v in data.val.items()}
                tot += float(task_loss(model(TaskData.obs(data.val, s)), b, task, n_clusters)) * (s.stop - s.start)
                n += s.stop - s.start
        return tot / max(n, 1)

    best, best_epoch, best_state, best_head, bad, hist = float("inf"), None, None, None, 0, []
    t0 = time.time()
    for ep in range(args.num_epochs):
        model.train()
        perm = torch.randperm(data.n_train, device=device)
        run = 0.0
        for i in range(0, data.n_train, args.batch_size):
            j = perm[i:i + args.batch_size]
            b = {k: v[j] for k, v in data.train.items()}
            loss = task_loss(model(TaskData.obs(data.train, j)), b, task, n_clusters)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += float(loss.detach()) * len(j)
        v = val_loss()
        sched.step(v)
        hist.append({"epoch": ep, "train": run / data.n_train, "val": v})
        if v < best - 1e-6:
            best, best_epoch, bad = v, ep, 0
            # Snapshot the HEAD as well as the encoder: the metrics below are computed with
            # encoder + head, and a best-epoch encoder paired with a final-epoch head is a
            # combination that never existed during training (the record's ST dim_hidden=512
            # run reported a good best_val next to a near-chance identify_acc that way).
            best_state = {k: t.detach().clone() for k, t in extractor.state_dict().items()}
            best_head = {k: t.detach().clone() for k, t in model.head.state_dict().items()}
        else:
            bad += 1
        if ep % 10 == 0 or bad >= args.patience or ep == args.num_epochs - 1:
            print(f"  ep{ep:>3} train={run / data.n_train:.5f} val={v:.5f} best={best:.5f}", flush=True)
        (run_dir / "history.json").write_text(json.dumps(hist, indent=1))
        if bad >= args.patience:
            break
    save_task_checkpoint(model, checkpoint_dir / "checkpoint_last.pt", args, len(hist) - 1,
                         {"loss": hist[-1]["val"]}, geometry, task)

    extractor.load_state_dict(best_state)
    model.head.load_state_dict(best_head)
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, data.n_val, 4096):
            s = slice(i, min(i + 4096, data.n_val))
            b = {k: v[s] for k, v in data.val.items()}
            chunks.append(task_metrics(model(TaskData.obs(data.val, s)), b, task, n_clusters,
                                       data.particle_scale))
    metrics = {k: float(np.mean([c[k] for c in chunks])) for k in chunks[0]}
    save_task_checkpoint(model, checkpoint_dir / "checkpoint_best.pt", args, best_epoch,
                         {"loss": best, **metrics}, geometry, task)
    (run_dir / "metrics.json").write_text(json.dumps(
        dict(best_val=best, best_epoch=best_epoch, epochs=len(hist), minutes=(time.time() - t0) / 60,
             val_metrics=metrics, history=hist), indent=2))
    print(f"[{args.variant}/task/{args.encoder}] best_val={best:.5f} (epoch {best_epoch})  "
          + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
          + f"  ({(time.time() - t0) / 60:.1f} min) -> {checkpoint_dir}", flush=True)
    checkpoints = {"best": checkpoint_dir / "checkpoint_best.pt", "last": checkpoint_dir / "checkpoint_last.pt"}
    return PretrainResult(run_dir=run_dir, rl_checkpoint=checkpoints["best"], checkpoints=checkpoints,
                          summary={"best_val_loss": best, "best_epoch": best_epoch,
                                   "epochs": len(hist), **metrics})


TASK_OBJECTIVE = Objective(
    name="task",
    description="encoder + 3-layer head -> the dataset's labels: the offset to the target "
                "cluster (least_mass / most_var, MSE) or every centre + alive flag "
                "(cluster_hunt, permutation-matched); the record's pretraining objective",
    add_arguments=_task_add_arguments,
    run=_task_run,
    run_name=_task_run_name,
    locate=_task_locate,
    resolve_arguments=_task_resolve_arguments,
    prepare=_task_prepare,
    default_experiment_name=lambda encoder_name: f"{encoder_name}_task_pretrain",
)


# ---------------------------------------------------------------------------
# Encoder defaults
# ---------------------------------------------------------------------------

#: Target for ``t_bound * (decision-relevant length / arena half-width)`` (PITFALLS section 9;
#: the same constant Ant-Tag uses). The decision-relevant length here is the hit radius.
CGF_TILT_TARGET = 3.0


def cgf_t_bound(name: str, target: float = CGF_TILT_TARGET) -> float:
    """``target / (hit_radius / ARENA_SCALE)``: 3 / (0.6 / 10) = 50 for all three variants."""
    hit_radius = float(_env_config(name).hit_radius)
    if hit_radius <= 0:
        raise ValueError(f"{resolve(name).env_id} reports hit_radius={hit_radius}")
    return float(target) / (hit_radius / ARENA_SCALE)


def _cgf_t_bound_default(variant: str) -> float:
    bound = cgf_t_bound(variant)
    print(f"CGF t_bound from the sizing rule: {bound:.3g} "
          f"(= {CGF_TILT_TARGET} / (hit_radius / arena half-width) for variant {variant!r})")
    return bound


def _cgf_t_init_max_default(args):
    """Ant-Tag's rule: the ``spread`` init tops out at 0.8 * t_bound; otherwise unset."""
    if args.t_param == "clamp":
        return None
    if args.t_init_mode == "spread":
        return 0.8 * float(args.t_bound)
    return None


HUNT = Domain(
    name="hunt",
    particle_dim=2,
    default_variant="least_mass",
    variants=VARIANTS,
    resolve=resolve,
    episode_cap=episode_cap,
    run_subdir=run_subdir,
    add_variant_argument=add_variant_argument,
    print_variants=print_variants,
    make_env=_make_env,
    make_vec_env_from_fns=_make_vec_env_from_fns,
    make_vec_normalize=_make_vec_normalize,
    particle_filter=lambda args: resolve(args.variant).particle_filter,
    add_arguments=_add_arguments,
    resolve_arguments=_resolve_arguments,
    schedules=_schedules,
    run_config_extras=_run_config_extras,
    default_num_particles=lambda variant: int(_env_config(variant).n_particles),
    default_arena_scale=lambda variant: ARENA_SCALE,
    default_device="cpu",
    default_total_timesteps=3_000_000,
    encoder_defaults={
        # The 2-D probe recipe (polar ball, spread init, raw features), bound from the sizing
        # rule; no legacy hunt run on the harness to reproduce. --feature_mode stays the CLI's.
        "cgf": dict(t_param="polar", t_bound=None, t_init_mode="spread", feature_norm="none",
                    t_bound_default=_cgf_t_bound_default,
                    t_init_max_default=_cgf_t_init_max_default),
        # The small ST (16 inducing points, hidden 64, two post-PMA SABs; ~109k parameters).
        "st": dict(num_inds=16, dim_hidden=64, num_post_sab=2),
    },
    evaluation=Evaluation(default_n_episodes=300, reseed_per_episode=True, report=_eval_report),
    # The record's supervised objective; `rl/pretrain.py --domain hunt` runs it when --objective
    # is omitted. The generic reconstruction (Chamfer + --ignore_weights = the record's control)
    # needs no declaration.
    pretraining=Pretraining(objectives={"task": TASK_OBJECTIVE}, default_objective="task"),
    # Step 2: the record's behaviour policy, labels per snapshot, no rebalancing (batch 9.2).
    collection=HUNT_COLLECTION,
)
