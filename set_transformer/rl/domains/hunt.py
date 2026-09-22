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
from collections.abc import Callable
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pdomains  # noqa: F401 - registers the pdomains-* env ids
import torch
import torch.nn.functional as F
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from set_transformer.rl.curriculum import Schedule, ScheduleRouter, parse_curriculum
from set_transformer.rl.domains.base import (
    Collection,
    Domain,
    Evaluation,
    FreshLayouts,
    Objective,
    PretrainContext,
    PretrainResult,
    Pretraining,
)
from set_transformer.rl.particle_filters.hunt import EnvEmittedBeliefFilter
from set_transformer.rl.wrappers.particle_filter import PFDictWithWeightsObservationWrapper
from set_transformer.rl.pretrain_objectives.task_head import (   # 10.9: the shared task-head loop
    GeneratedTaskData,      # 2026-09-21: the two generated sources
    MixedTaskData,
    TaskData as _TaskData,
    TaskEncoderWithHead,   # noqa: F401 - re-exported (tests, entry points)
    add_task_arguments,
    parse_val_sources,      # noqa: F401 - re-exported (tests import it from here)
    build_task_extractor,   # noqa: F401 - re-exported
    check_variant_and_geometry,
    data_source_of,
    extractor_geometry,   # noqa: F401 - re-exported
    locate_from_dataset,
    resolve_k_choices,
    resolve_task_arguments,
    run_task_training,
    save_task_checkpoint as _save_task_checkpoint,
    task_run_name,
)

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
    #: 2026-09-21 (fresh layouts): ``(cfg, k_choices, n_rows, rng) -> dict`` of the collector's label
    #: arrays plus ``particles`` / ``weights``, drawn by THIS env class's reset rules with the
    #: constants read off ``cfg`` (never literals). None: ``--data_source fresh`` is refused for the
    #: variant. Checked against ``env.reset()`` per variant per k in tests/test_hunt_fresh_layouts.py.
    layout_generator: Callable | None = None


# ---------------------------------------------------------------------------
# Fresh layout generators (2026-09-21; change_mds/fresh_layout_pretraining_2026-09-21.md section 1.1)
#
# The task objective's training rows can be REGENERATED every epoch instead of read from a collected
# file. The reason is measured: a collected file freezes one layout per episode (only the agent
# moves and the cloud is redrawn), so most_var's 3,600 layouts were each seen ~100 times and the
# encoder memorised them; generating fresh ones took its identify accuracy 0.371 -> 0.885 with the
# same objective, architecture and PPO block. These two functions supersede the standalone
# src/scripts/{most_var_gap/pretrain_spread,least_mass_gap/pretrain_fresh_lm}.py generators, which
# hardcoded the constants.
#
# Each function mirrors ONE env class's reset() and draws n rows at once. Every constant comes from
# the live `cfg` (`_env_config(variant)`), so a change to a registration or to a dataclass default
# reaches the generator without an edit here. The layout a row carries is the collector's own
# (`_collect_snapshot_extras` below): agent (pos - 10) / 10, particles agent-relative in RAW arena
# units, weights 1/N, centres (c - pos) / 10 zero-padded to n_clusters slots, alive, counts, sigmas
# raw, target = centers[target_index].
# ---------------------------------------------------------------------------


def _sample_centres(cfg, k: int, n: int, rng) -> np.ndarray:
    """``[n, k, 2]`` centres uniform in ``[mean_lo, mean_hi]^2`` with pairwise separation
    ``>= min_sep`` (the envs' ``_sample_centers`` rejection, in blocks instead of one row at a
    time; at k < 2 there is no pair to separate)."""
    out = np.empty((0, k, 2))
    need = n
    while need > 0:
        cand = rng.uniform(cfg.mean_lo, cfg.mean_hi, size=(3 * need + 8, k, 2))
        if k < 2:
            keep = cand
        else:
            d = np.linalg.norm(cand[:, :, None] - cand[:, None], axis=-1) + 1e9 * np.eye(k)[None]
            keep = cand[d.min(axis=(1, 2)) >= cfg.min_sep]
        out = np.concatenate([out, keep])[:n]
        need = n - len(out)
    return out


def _sample_sigmas_with_margin(cfg, k: int, n: int, rng) -> np.ndarray:
    """``[n, k]`` widths whose largest exceeds the second largest by ``sigma_margin``
    (``MinMassHuntEnv._sample_sigmas``, the ``max_var`` rule, in blocks)."""
    out = np.empty((0, k))
    need = n
    while need > 0:
        cand = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(4 * need + 8, k))
        if k < 2:
            keep = cand
        else:
            top = np.sort(cand, 1)
            keep = cand[(top[:, -1] - top[:, -2]) >= cfg.sigma_margin]
        out = np.concatenate([out, keep])[:n]
        need = n - len(out)
    return out


def _draw_clouds(cfg, centres: np.ndarray, sigmas: np.ndarray, counts: np.ndarray, rng) -> np.ndarray:
    """``[n, n_particles, 2]``: ``counts[j]`` draws from ``N(centres[j], sigmas[j])`` per row,
    clipped to the arena and shuffled (both envs' ``_draw`` / ``_draw_particles``)."""
    n, _ = counts.shape
    particles = int(cfg.n_particles)
    # which cluster each particle belongs to: np.repeat(arange(k), counts) per row, vectorised
    index = (np.arange(particles)[None, :, None] >= np.cumsum(counts, 1)[:, None, :]).sum(-1)
    pts = (np.take_along_axis(centres, index[..., None], 1)
           + np.take_along_axis(sigmas, index, 1)[..., None] * rng.standard_normal((n, particles, 2)))
    pts = np.clip(pts, 0.0, 2.0 * ARENA_SCALE)
    return np.take_along_axis(pts, rng.random((n, particles)).argsort(1)[..., None], 1)


#: Every array a generated row carries, in the collector's spelling. ``step`` is not among them: a
#: fresh layout is not a step of an episode, and no task objective reads it.
GENERATED_ARRAYS = ("agent", "particles", "weights", "centers", "alive", "counts", "sigmas",
                    "target", "target_index")


def _pack(cfg, centres, sigmas, counts, pos, target_index, alive_k: int) -> dict:
    """The collector's row layout for one block of rows that share a cluster count."""
    n, k = counts.shape
    K, N = int(cfg.n_clusters), int(cfg.n_particles)
    centers = np.zeros((n, K, 2), np.float32)
    centers[:, :k] = (centres - pos[:, None]) / ARENA_SCALE
    alive = np.zeros((n, K), np.float32)
    alive[:, :alive_k] = 1.0
    counts_out = np.zeros((n, K), np.float32)
    counts_out[:, :k] = counts
    sigmas_out = np.zeros((n, K), np.float32)
    sigmas_out[:, :k] = sigmas
    return {"agent": ((pos - ARENA_SCALE) / ARENA_SCALE).astype(np.float32),
            "weights": np.full((n, N), 1.0 / N, np.float32),
            "centers": centers, "alive": alive, "counts": counts_out, "sigmas": sigmas_out,
            "target": centers[np.arange(n), target_index],
            "target_index": target_index.astype(np.int64)}


def _concatenate_and_shuffle(blocks: list[dict], rng) -> dict:
    out = {key: np.concatenate([b[key] for b in blocks]) for key in GENERATED_ARRAYS}
    order = rng.permutation(len(out["agent"]))
    return {key: value[order] for key, value in out.items()}


def generate_pick_target_layouts(cfg, k_choices, n: int, rng) -> dict:
    """``MinMassHuntEnv.reset``, vectorised: least_mass (``target_rule`` ``min_mass``) and most_var
    (``max_var``) in one function, branching where the env branches.

    ``min_mass``: widths uniform, counts by ``_sample_counts`` (a minimum in
    ``[n_min_lo, n_min_hi]``, every other cluster at least ``mass_margin`` above it, the surplus
    cut at random points, then a permutation), target = the lightest. ``max_var``: widths by the
    ``sigma_margin`` rejection, counts as equal as possible with the remainder to RANDOM clusters,
    target = the widest. ``k`` is clipped to ``[2, n_clusters]``, as the env's own setter clips it.
    """
    K, N = int(cfg.n_clusters), int(cfg.n_particles)
    ks = np.clip(np.asarray(rng.choice(np.asarray(k_choices), size=n)), 2, K)
    blocks = []
    for k in sorted({int(v) for v in ks}):
        m = int((ks == k).sum())
        centres = _sample_centres(cfg, k, m, rng)
        if cfg.target_rule == "min_mass":
            sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(m, k))
            n_min = rng.integers(cfg.n_min_lo, cfg.n_min_hi + 1, size=m)
            base = n_min + int(cfg.mass_margin)
            surplus = N - n_min - (k - 1) * base
            if (surplus < 0).any():
                raise ValueError(
                    f"the env's count rule cannot be satisfied at k={k}: a minimum of up to "
                    f"{cfg.n_min_hi} plus mass_margin {cfg.mass_margin} needs more than {N} "
                    "particles (MinMassHuntEnv._sample_counts would retry and raise)")
            if k == 2:
                counts = np.stack([n_min, base + surplus], 1)
            else:
                cuts = np.sort(rng.integers(0, surplus[:, None] + 1, size=(m, k - 2)), axis=1)
                parts = np.diff(np.concatenate(
                    [np.zeros((m, 1), int), cuts, surplus[:, None]], 1), axis=1)
                counts = np.concatenate([n_min[:, None], base[:, None] + parts], 1)
            counts = np.take_along_axis(counts, rng.random((m, k)).argsort(1), 1)   # the env permutes
            target_index = counts.argmin(1)
        else:
            sigmas = _sample_sigmas_with_margin(cfg, k, m, rng)
            share, remainder = divmod(N, k)
            counts = np.full((m, k), share, int)
            if remainder:
                # _equal_counts: the remainder goes to clusters chosen at random, so mass carries
                # no information about the target
                np.put_along_axis(counts, rng.random((m, k)).argsort(1)[:, :remainder], share + 1, 1)
            target_index = sigmas.argmax(1)
        pos = rng.uniform(0.0, 2.0 * ARENA_SCALE, size=(m, 2))
        block = _pack(cfg, centres, sigmas, counts, pos, target_index, alive_k=k)
        block["particles"] = (_draw_clouds(cfg, centres, sigmas, counts, rng)
                              - pos[:, None]).astype(np.float32)
        blocks.append(block)
    return _concatenate_and_shuffle(blocks, rng)


def generate_collect_all_layouts(cfg, k_choices, n: int, rng) -> dict:
    """``ClusterHuntEnv.reset``, vectorised: ALL ``n_clusters`` centres and widths are drawn (the
    dead ones too, as the collector labels them), ``alive`` is the FIRST ``k`` slots, and the
    particles are split as equally as possible over the live clusters with the remainder to the
    FIRST live ones (``_particle_counts``). The env has no target here, so the row's ``target`` /
    ``target_index`` follow the collector's convention -- the NEAREST LIVE centre -- and the
    ``task`` objective's labels exist while ``task_nearest`` recomputes its own as it does on a
    file. ``k`` is clipped to ``[1, n_clusters]``, as the env's own setter clips it.
    """
    K, N = int(cfg.n_clusters), int(cfg.n_particles)
    ks = np.clip(np.asarray(rng.choice(np.asarray(k_choices), size=n)), 1, K)
    blocks = []
    for k in sorted({int(v) for v in ks}):
        m = int((ks == k).sum())
        centres = _sample_centres(cfg, K, m, rng)
        sigmas = rng.uniform(cfg.sigma_lo, cfg.sigma_hi, size=(m, K))
        counts = np.zeros((m, K), int)
        share, remainder = divmod(N, k)
        counts[:, :k] = share
        counts[:, :remainder] += 1
        pos = rng.uniform(0.0, 2.0 * ARENA_SCALE, size=(m, 2))
        distance = np.linalg.norm(centres - pos[:, None], axis=-1)
        distance[:, k:] = np.inf                                   # a dead slot is not a candidate
        block = _pack(cfg, centres, sigmas, counts, pos, distance.argmin(1), alive_k=k)
        block["particles"] = (_draw_clouds(cfg, centres, sigmas, counts, rng)
                              - pos[:, None]).astype(np.float32)
        blocks.append(block)
    return _concatenate_and_shuffle(blocks, rng)


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
        layout_generator=generate_collect_all_layouts,
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
        layout_generator=generate_pick_target_layouts,
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
        layout_generator=generate_pick_target_layouts,
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
    # Change C (2026-09-19): "left at the default" is the trainer's marker, not a value comparison.
    # Before this an EXPLICIT --total_timesteps 3000000 on cluster_hunt (equal to the domain
    # default) was silently swapped for the variant's 1,500,000 (debug_plans/ch_fixes.md).
    if not getattr(args, "total_timesteps_given", True) \
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
    # 2026-09-21 (plan section 3): `env_kwargs` records only what the REGISTRATION overrides, so a
    # dataclass default is invisible in a run record (least_mass's timeout_penalty was read as 20
    # from the dataclass while the registration had not yet set 40). `env_config` is the
    # constructed config, every field of it. Both are kept: no recorded reader changes.
    return dict(episode_cap=episode_cap(args.variant), task=variant.task,
                env_kwargs=dict(gym.spec(variant.env_id).kwargs),
                env_config=_env_config(args.variant).to_dict())


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
                # 2026-09-21 (plan section 3): the CONSTRUCTED config beside the registration's
                # overrides, so a dataclass default is on the record too (see _run_config_extras).
                env_config=_env_config(args.variant).to_dict(),
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

# Every learned encoder of the shared table (checked as `encoder.learned`; batch 10.4 replaced a
# hand-kept tuple). Kept as a name for the messages.
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


_COLLECT_HINT = "collect it with python -m set_transformer.rl.collect --domain hunt --variant <v>"


class TaskData(_TaskData):
    """The hunt dataset (``rl/pretrain_objectives/task_head.py::TaskData`` with hunt's arrays):
    the agent position is the ``obs`` passthrough, the record's six label arrays are required."""

    LABELS = ("centers", "alive", "counts", "sigmas", "target", "target_index")

    def __init__(self, path: str, val_frac: float, device: torch.device,
                 val_sources: tuple[int, ...] | None = None):
        super().__init__(path, val_frac, device, labels=("agent", *self.LABELS), obs_key="agent",
                         scale_default=ARENA_SCALE, collect_hint=_COLLECT_HINT, kind="hunt dataset",
                         val_sources=val_sources)



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


def _task_checkpoint_config(args, task: str) -> dict:
    return {"objective": "task", "task": task, "variant": args.variant,
            "pretraining": "set_transformer.rl.pretrain --domain hunt --objective task"}


def save_task_checkpoint(model: TaskEncoderWithHead, path: Path, args, epoch: int, val: dict,
                         geometry: dict, task: str) -> None:
    """Hunt's spelling of the shared writer (``geometry`` is kept in the signature; the
    extractor's own record is what gets written)."""
    _save_task_checkpoint(model, path, args, epoch, val, _task_checkpoint_config(args, task))



def _task_add_arguments(parser, domain=None) -> None:
    # 2026-09-21: every hunt variant declares a layout_generator, so the fresh-layout group is
    # offered here (Ant-Tag's task objective, which has none, keeps the command line it had).
    add_task_arguments(parser, dataset_help="The collected hunt dataset (.npz with the label arrays).",
                       fresh_layouts=True)


_task_locate = locate_from_dataset


def _task_resolve_arguments(parser, args, domain, encoder) -> None:
    resolve_task_arguments(parser, args, domain, encoder, collect_hint=_COLLECT_HINT,
                           fresh_layouts=True)


def _generated_task_data(args, device):
    """The generated (``fresh``) or file-plus-generated (``mixed``) source for the task objective
    (2026-09-21). The env constants come from the live cfg, the cluster-count draw from
    ``--k_choices`` or the collector's own, the row budget from ``--fresh_rows_per_epoch``."""
    variant = resolve(args.variant)
    cfg = _env_config(args.variant)
    k_choices = resolve_k_choices(args, N_ACTIVE_CHOICES[variant.task])
    common = dict(variant=args.variant, env_id=variant.env_id, k_choices=k_choices,
                  val_rows=int(args.fresh_val_rows), seed=int(args.fresh_seed),
                  device=torch.device(device), label_arrays=GENERATED_ARRAYS,
                  val_select=args.val_select)
    if data_source_of(args) == "fresh":
        return GeneratedTaskData(variant.layout_generator, cfg, obs_key="agent",
                                 rows_per_epoch=int(args.fresh_rows_per_epoch),
                                 particle_scale=ARENA_SCALE, **common)
    file_data = TaskData(args.data_path, args.val_frac, torch.device(device),
                         val_sources=parse_val_sources(getattr(args, "val_sources", None)))
    # The mixed default keeps steps per epoch at twice the file's, which is what the 0.936 / 0.885
    # standalone runs had; both epoch-counted schedules are tuned at that scale.
    rows = args.fresh_rows_per_epoch
    if rows is None:
        rows = file_data.n_train
        print(f"--fresh_rows_per_epoch not given; the file's own {rows:,} training rows")
    return MixedTaskData(file_data, variant.layout_generator, cfg,
                         rows_per_epoch=int(rows), **common)


def _task_prepare(parser, args, domain, encoder, device):
    if data_source_of(args) != "file":
        data = _generated_task_data(args, device)
    else:
        data = TaskData(args.data_path, args.val_frac, torch.device(device),
                        val_sources=parse_val_sources(getattr(args, "val_sources", None)))
    check_variant_and_geometry(parser, args, data)
    return data


_task_run_name = task_run_name


def _task_run(args, ctx: PretrainContext) -> PretrainResult:
    """Hunt's task objective on the shared loop (``rl/pretrain_objectives/task_head.py``, batch
    10.9; the loop itself was here from 9.2 to 10.9): the matched-slot loss on Cluster-Hunt, the
    offset MSE on the pick-a-target tasks, the record's metrics in arena units."""
    variant = resolve(args.variant)
    task = variant.task
    n_clusters = int(_env_config(args.variant).n_clusters)
    out_dim = n_clusters * 3 if task == "collect_all" else 2
    return run_task_training(
        args, ctx, data=ctx.data, obs_dim=2, out_dim=out_dim,
        loss_fn=lambda y, batch: task_loss(y, batch, task, n_clusters),
        metrics_fn=lambda y, batch: task_metrics(y, batch, task, n_clusters, ctx.data.particle_scale),
        checkpoint_config=_task_checkpoint_config(args, task),
        label=f"{args.variant}/task/{ctx.encoder.name}")



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


# -- task_nearest: Cluster-Hunt's campaign task objective (batch 10.10, 2026-09-14; plan 10.10) --------
#
# Predict the offset to the NEAREST LIVE cluster only: K = 1, the least-mass shape (2 numbers, squared
# error, identify metric = nearest live centre). The label is computed here from the batch's `centers` /
# `alive` (agent-relative, scaled), so no collector or file change; continuous except at a tie for
# first place (broken by the lowest cluster index); decision-shaped (the greedy tour's next waypoint).
# Cost, written down: the encoder is asked for one cluster at a time and may not carry the layout for
# the later stops -- the comparison against reconstruction, which keeps the whole cloud, is the
# experiment. `task` (the record's matched-slot loss) stays as the record-reproducing condition;
# `task_sorted` (slots ordered by distance) was rejected: discontinuous at every rank tie.

def nearest_live_offset(centers: torch.Tensor, alive: torch.Tensor):
    """``(offset [B, 2], index [B], has_live [B])``: the closest live centre per row (ties -> the lowest
    index), and which rows have a live cluster at all (the belief recorded after the last collection
    has none; those rows are excluded from the loss and the metrics)."""
    d = torch.linalg.norm(centers, dim=-1).masked_fill(alive < 0.5, float("inf"))
    index = d.argmin(1)
    has_live = alive.sum(1) > 0
    offset = centers[torch.arange(len(centers), device=centers.device), index]
    return offset, index, has_live


def task_nearest_loss(y: torch.Tensor, batch: dict) -> torch.Tensor:
    offset, _, live = nearest_live_offset(batch["centers"], batch["alive"])
    if not bool(live.any()):
        return (y * 0.0).sum()
    return F.mse_loss(y[live], offset[live])


@torch.no_grad()
def task_nearest_metrics(y: torch.Tensor, batch: dict, scale: float) -> dict:
    """The pick-a-target metrics against the nearest live centre, over the rows that have one."""
    offset, index, live = nearest_live_offset(batch["centers"], batch["alive"])
    y, offset, index = y[live], offset[live], index[live]
    centers, alive = batch["centers"][live], batch["alive"][live]
    if len(y) == 0:
        return {"loc_mae": float("nan"), "identify_acc": float("nan"), "within_1.0": float("nan")}
    err = scale * torch.linalg.norm(y - offset, dim=-1)
    d = torch.linalg.norm(centers - y[:, None, :], dim=-1).masked_fill(alive < 0.5, float("inf"))
    return {"loc_mae": float(err.mean()),
            "identify_acc": float((d.argmin(1) == index).float().mean()),
            "within_1.0": float((err < 1.0).float().mean())}


def _task_nearest_resolve_arguments(parser, args, domain, encoder) -> None:
    _task_resolve_arguments(parser, args, domain, encoder)


def _task_nearest_prepare(parser, args, domain, encoder, device):
    data = _task_prepare(parser, args, domain, encoder, device)
    if resolve(args.variant).task != "collect_all":
        parser.error(f"task_nearest is Cluster-Hunt's objective (collect_all variants); on {args.variant!r} the "
                     "target is not the nearest cluster -- use --objective task")
    return data


def _task_nearest_run_name(args, now: datetime) -> str:
    return f"{now.strftime('%Y%m%d_%H%M%S')}_task_nearest_{args.encoder}_seed{args.seed}"


def _task_nearest_run(args, ctx: PretrainContext) -> PretrainResult:
    return run_task_training(
        args, ctx, data=ctx.data, obs_dim=2, out_dim=2,
        loss_fn=task_nearest_loss,
        metrics_fn=lambda y, batch: task_nearest_metrics(y, batch, ctx.data.particle_scale),
        checkpoint_config={"objective": "task_nearest", "task": "nearest", "variant": args.variant,
                           "pretraining": "set_transformer.rl.pretrain --domain hunt --objective task_nearest"},
        label=f"{args.variant}/task_nearest/{ctx.encoder.name}")


TASK_NEAREST_OBJECTIVE = Objective(
    name="task_nearest",
    description="encoder + 3-layer head -> the offset to the NEAREST LIVE cluster (2 numbers, MSE; the "
                "greedy tour's next waypoint), computed from the file's centres / alive flags; Cluster-Hunt "
                "only (the campaign's task condition there; `task` is the record's matched-slot loss)",
    add_arguments=_task_add_arguments,
    run=_task_nearest_run,
    run_name=_task_nearest_run_name,
    locate=_task_locate,
    resolve_arguments=_task_nearest_resolve_arguments,
    prepare=_task_nearest_prepare,
    default_experiment_name=lambda encoder_name: f"{encoder_name}_task_nearest_pretrain",
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


def _fresh_layouts(name: str) -> FreshLayouts | None:
    """``Domain.fresh_layouts`` (2026-09-22): the variant's generator bound to the LIVE env config,
    the collector's cluster-count draw and the collector's frame, for the generic reconstruction
    objective's ``--data_source fresh | mixed`` (``rl/pretrain_objectives/fresh_layouts.py``). None
    for a variant without a generator. The task objective keeps its own path
    (:func:`_generated_task_data`), unchanged."""
    variant = resolve(name)
    if variant.layout_generator is None:
        return None
    cfg = _env_config(name)
    return FreshLayouts(
        generator=lambda k_choices, n, rng: variant.layout_generator(cfg, k_choices, n, rng),
        k_choices=tuple(N_ACTIVE_CHOICES[variant.task]), particle_scale=ARENA_SCALE,
        particle_centre=0.0, env_id=variant.env_id, constants=cfg.to_dict())


HUNT = Domain(
    name="hunt",
    particle_dim=2,
    default_sinkhorn_blur=0.02,      # 2026-09-19: the recipes' recorded blur
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
    pretraining=Pretraining(objectives={"task": TASK_OBJECTIVE, "task_nearest": TASK_NEAREST_OBJECTIVE},
                            default_objective="task"),
    # Step 2: the record's behaviour policy, labels per snapshot, no rebalancing (batch 9.2).
    collection=HUNT_COLLECTION,
    # 2026-09-22: rows drawn from a variant's reset rules for the generic reconstruction objective.
    fresh_layouts=_fresh_layouts,
)
