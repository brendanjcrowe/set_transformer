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

The numbers this module produces are a NEW table under the record's protocol, not a
reproduction of ``domain_mds/{cluster_hunt,least_mass}.md``: the extractors are the harness
ones, the ST reads a (constant) weight channel, and the trainer is the shared PPO loop.
``src/hunt_tasks/`` stays untouched as the record (decision 9c-6).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pdomains  # noqa: F401 - registers the pdomains-* env ids
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from set_transformer.rl.curriculum import Schedule, ScheduleRouter, parse_curriculum
from set_transformer.rl.domains.base import Domain, Evaluation
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
)
