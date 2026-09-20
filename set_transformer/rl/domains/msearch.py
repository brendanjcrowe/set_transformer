"""The ``msearch`` domain: Multimodal Search on the shared RL harness (plan section 10, batch
10.3, 2026-09-14).

``pdomains-multimodal-search-v0`` (``pdomains/multimodal_search.py``, moved there from
``set_transformer/rl/envs/`` in batch 9.0b): a point agent on a 28 x 28 arena; a static target
hides in one of K ~ U[2, 10] random Gaussian modes; the agent sees perfectly within radius 1.5
and nothing outside; each step costs 1, finding pays 43, the cap is 42 steps. The mode centres
are translated so their centroid is exactly the origin every episode, so the belief MEAN carries
no information about the target and a Gaussian summary is blind by construction; the geometry
stays random so there is no fixed sweep to memorise. Built for this campaign
(``experiments/benchmark/ENV_VIABILITY.md``: "purpose-built, discriminating by construction").

**The filter is exact** (:class:`~set_transformer.rl.particle_filters.multimodal_search.MultimodalSearchParticleFilter`):
the posterior of a static target is the prior restricted to the region not yet observed, so
the particles are drawn from the modes at reset and the ones the agent sweeps are refuted and
redrawn from the unswept prior; a sighting collapses the cloud onto the target. The prior's
mode parameters ride in the observation (60 of its 67 entries) so the filter can build it.

Four things this module decides (decisions 10c-4, 5, 6, 13 of ``refactor_plans.md``, user
GO 2026-09-14):

* **The policy never sees the mode parameters** (10c-4). :class:`MsearchAgentObsWrapper`, ABOVE
  the filter wrapper, cuts the ``obs`` key to the 7 base entries (agent xy, previous xy, target
  offset when visible, visibility flag). The filter wrapper below it still reads the raw
  observation. Otherwise every arm reads the mixture directly and the encoder comparison is empty.
* **Particles are AGENT-RELATIVE** (10c-5, reversing the plan's first recommendation):
  :class:`AgentRelativeMultimodalSearchParticleFilter` hands out ``particles - agent_pos``.
  Brendan measured that nothing learned this env until the particles were in the agent's frame
  (``rl/benchmark/registry.py``); in that frame the belief mean is ``-agent_pos``, which the
  policy already has from the observation, so the Gaussian arm gains nothing about the target
  and the pinned-mean property is kept. Hunt uses the same frame. The subclass also HONOURS the
  wrapper's per-episode ``rng_seed`` -- the parent draws its prior from an unseeded generator
  (PITFALLS 13.7).
* **Training reward = task reward + information-gain proxy** (10c-13): Brendan's
  ``pf_belief_information_potential(scale=100)`` with gamma 1 and no terminal zeroing -- an
  episode's bonus is 100 x the nats of prior mass ruled out. Not policy-invariant at this scale,
  but sweeping the mode that holds the target ends the episode, so gathering information and
  finding cannot be separated; and "an unshaped learner sees almost no signal". The EVAL env
  (``training=False``) is unshaped, so every reported number is the true task reward.
  ``--shaping none`` turns it off.
* **PPO with hunt's block, 3M steps** (10c-6, option A). Brendan's PPO (Odd-Even's shape) gave
  success 0.12-0.48 at 600k steps against SAC's 0.54-0.89; the campaign stays PPO, SAC is the
  fallback wave. The block itself is the driver recipe's.

**Evaluation.** Found rate (``info["found"]``) = success, timeout rate, mean steps among the
found episodes, mean reward; 300 episodes, re-seeded per episode (the modes are drawn at reset).

**Collection and pretraining.** :data:`MSEARCH_COLLECTION` drives the unshaped belief env with
a mix of the viability probe's informed tour (nearest unvisited mode, read off the live env)
and an OU random walk (``--tour_frac``), and records per-snapshot labels (agent xy, target
offset, mode means / covariances padded to k_max, validity, number of modes, target mode, step)
in the scaled agent-relative frame; particles as the filter hands them (agent-relative, raw
units), uniform weights, no rebalancing. Pretraining: the generic ``reconstruction`` objective
(and alignment); a ``task`` objective is a later line.

**Encoder defaults.** CGF polar, ``t_bound`` = 3 / (visibility 1.5 / arena half 14) = 28
(PITFALLS section 9 rule), spread init to 0.8 x bound, no norm; ST 16 / 64 / 2 post-SABs;
100 particles (the benchmark's count); ``particle_scale`` = arena half-width 14.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pdomains  # noqa: F401 - registers the pdomains-* env ids
from pdomains.multimodal_search import BASE_OBS_DIM
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from set_transformer.rl.domains.base import Collection, Domain, Evaluation
from set_transformer.rl.particle_filters.base import BaseParticleFilter
from set_transformer.rl.particle_filters.multimodal_search import MultimodalSearchParticleFilter
from set_transformer.rl.wrappers.particle_filter import PFDictWithWeightsObservationWrapper
from set_transformer.rl.wrappers.shaping import (
    PotentialBasedShapingWrapper,
    pf_belief_information_potential,
)

ENV_ID = "pdomains-multimodal-search-v0"
#: The registered config's arena half-width: the ``particle_scale`` the extractors divide by.
#: Written as a number so the module loads against any pdomains checkout; a test pins it.
ARENA_HALF = 14.0
#: The benchmark's particle count (``rl/benchmark/registry.py``).
DEFAULT_NUM_PARTICLES = 100
#: Brendan's proxy-reward scale: 100 x nats of prior mass ruled out (decision 10c-13).
SHAPING_SCALE = 100.0
SHAPING_MODES = ("info_gain", "none")
#: Number of entries of the raw observation the POLICY sees (agent xy, previous xy, target
#: offset, visible flag); the 6-per-mode block behind them is the filter's, not the agent's.
AGENT_OBS_DIM = BASE_OBS_DIM


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One msearch configuration plus the filter and the training-reward default."""

    env_id: str
    particle_filter: type
    notes: str = ""
    #: ``info_gain`` (Brendan's proxy reward on the TRAINING env) or ``none``.
    default_shaping: str = "info_gain"
    default_total_timesteps: int = 3_000_000


VARIANTS: dict[str, Variant] = {
    "msearch": Variant(
        env_id=ENV_ID,
        particle_filter=None,   # filled below (the class is defined after the registry)
        notes=("The registered config: arena half 14, K in [2, 10], mode sigma [0.3, 1.0], "
               "separation 5, visibility 1.5, speed 3, cap 42, step -1 / find +43. Belief mean "
               "pinned at the origin; a Gaussian summary is blind by construction. Training "
               "reward adds the information-gain proxy (scale 100); eval is the true reward."),
    ),
}


def resolve(name: str) -> Variant:
    """Look up a variant, listing the valid names on a typo."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"Unknown variant {name!r}. Available: {sorted(VARIANTS)}") from None


def episode_cap(name: str) -> int:
    """The episode cap from the gym registration (42)."""
    spec = gym.spec(resolve(name).env_id)
    if spec.max_episode_steps is None:
        raise ValueError(f"{spec.id} registers no max_episode_steps; pass --max_steps")
    return int(spec.max_episode_steps)


def run_subdir(encoder: str, name: str) -> str:
    """The legacy-layout folder name for this (encoder, variant) pair."""
    resolve(name)
    return f"msearch_{encoder}_{name}"


def add_variant_argument(parser, default: str = "msearch") -> None:
    parser.add_argument(
        "--variant", type=str, default=default, choices=sorted(VARIANTS),
        help=f"Multimodal Search configuration (default: {default}).")
    parser.add_argument("--list_variants", action="store_true",
                        help="Print the variant registry and exit.")


def print_variants() -> None:
    for name, variant in VARIANTS.items():
        try:
            cap = episode_cap(name)
        except Exception:  # noqa: BLE001 - listing must not fail on one bad entry
            cap = "?"
        print(f"{name:8s} {variant.env_id:32s} cap={cap:<5} shaping={variant.default_shaping:10s} "
              f"{variant.particle_filter.__name__}")
        if variant.notes:
            print(f"{'':8s}   {variant.notes}")


def _env_config(name: str):
    """The live env's :class:`MultimodalSearchConfig` for a variant."""
    env = gym.make(resolve(name).env_id)
    try:
        return env.unwrapped.config
    finally:
        env.close()


# ---------------------------------------------------------------------------
# The filter in the agent's frame
# ---------------------------------------------------------------------------


class AgentRelativeMultimodalSearchParticleFilter(MultimodalSearchParticleFilter):
    """The exact msearch filter, handing out ``particles - agent_pos`` and seeded per episode.

    The parent keeps ABSOLUTE particles (its sweep test and its prior are in arena
    coordinates) and that is unchanged: :attr:`absolute_particles` is what it holds,
    :attr:`particles` is the same cloud translated by the agent's position read off the latest
    observation (entries 0:2). Decision 10c-5: the optimal policy is "head for the nearest
    unvisited mode", a relation between the agent and the set; in world coordinates every arm
    has to learn to subtract its own position first. The belief mean becomes ``-agent_pos``,
    which the policy has anyway, so the Gaussian arm learns nothing about the target from it.

    The constructor is the parent's, re-spelled (five assignments) because the parent creates an
    UNSEEDED generator before it draws the prior: here ``rng_seed`` -- the per-episode seed the
    shared PF wrapper derives from ``(worker seed, episode index)`` -- seeds that generator, so
    two runs with the same seed hold the same particles (PITFALLS 13.7).
    """

    def __init__(self, num_particles: int, initial_env_obs: np.ndarray, k_max: int = 10,
                 arena_half: float = ARENA_HALF, visibility_radius: float = 1.5,
                 rng_seed: int | None = None, **kwargs):
        self.k_max = int(k_max)
        self.arena_half = float(arena_half)
        self.visibility_radius = float(visibility_radius)
        self._particle_dim = 2
        self._rng = np.random.default_rng(rng_seed)
        self._agent_pos = np.asarray(initial_env_obs, dtype=np.float64).ravel()[0:2].copy()
        BaseParticleFilter.__init__(self, num_particles, initial_env_obs, **kwargs)

    def update(self, obs_from_env: np.ndarray, **kwargs) -> None:
        self._agent_pos = np.asarray(obs_from_env, dtype=np.float64).ravel()[0:2].copy()
        super().update(obs_from_env, **kwargs)

    @property
    def absolute_particles(self) -> np.ndarray:
        """The cloud in arena coordinates, as the parent holds it."""
        return self._particles

    @property
    def agent_pos(self) -> np.ndarray:
        return self._agent_pos

    @property
    def particles(self) -> np.ndarray:
        """The cloud relative to the agent: ``absolute - agent_pos``."""
        return self._particles - self._agent_pos


VARIANTS["msearch"] = Variant(
    env_id=VARIANTS["msearch"].env_id,
    particle_filter=AgentRelativeMultimodalSearchParticleFilter,
    notes=VARIANTS["msearch"].notes,
    default_shaping=VARIANTS["msearch"].default_shaping,
    default_total_timesteps=VARIANTS["msearch"].default_total_timesteps,
)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class MsearchAgentObsWrapper(gym.ObservationWrapper):
    """Above the filter wrapper: cut the Dict's ``obs`` key to its 7 base entries.

    The raw observation is ``[agent xy, previous xy, target offset xy, visible, <6 per mode up
    to k_max>]``. The filter wrapper below this one receives the whole vector (its prior IS the
    mode block); the policy receives the first :data:`AGENT_OBS_DIM` entries only (decision
    10c-4). Zeroing the block, as the benchmark did, gives the policy the same information;
    removing it keeps 60 constant zeros out of the observation normaliser.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        space = env.observation_space
        if not isinstance(space, gym.spaces.Dict) or "obs" not in space.spaces:
            raise TypeError(f"expected the PF Dict observation space with an 'obs' key, got {space}")
        base = space.spaces["obs"]
        if base.shape[0] < AGENT_OBS_DIM:
            raise ValueError(f"the base observation has {base.shape[0]} entries; msearch's has "
                             f"at least {AGENT_OBS_DIM}")
        spaces = dict(space.spaces)
        spaces["obs"] = gym.spaces.Box(-np.inf, np.inf, (AGENT_OBS_DIM,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

    def observation(self, observation):
        out = dict(observation)
        out["obs"] = np.asarray(observation["obs"][:AGENT_OBS_DIM], dtype=np.float32)
        return out


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


def make_msearch_belief_env(
    num_particles: int,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    variant: str = "msearch",
    env_id: str | None = None,
    particle_filter_class: type | None = None,
    shaping: str = "info_gain",
):
    """Return a callable that builds one msearch belief env.

    Stack, inside out: ``gym.make(env_id)`` (TimeLimit at the registered cap) ->
    :class:`PFDictWithWeightsObservationWrapper` with the agent-relative filter (built with the
    env's own k_max / arena half / visibility radius; driven positionally, no mapper, nothing
    masked -- the RAW observation reaches the filter) -> :class:`MsearchAgentObsWrapper` (the
    policy's 7 entries) -> :class:`PotentialBasedShapingWrapper` with the information-gain
    proxy (``shaping == "info_gain"`` only; the eval env is built with ``"none"``) -> Monitor.
    """
    if shaping not in SHAPING_MODES:
        raise ValueError(f"shaping must be one of {SHAPING_MODES}, got {shaping!r}")
    resolved = resolve(variant)
    env_id = env_id or resolved.env_id
    particle_filter_class = particle_filter_class or resolved.particle_filter

    def _init():
        # Registered HERE too, not only at module import: this closure is cloudpickled by
        # value into a SubprocVecEnv child, which never runs this module's top-level import.
        import pdomains  # noqa: F401,PLC0415

        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        cfg = env.unwrapped.config
        env = PFDictWithWeightsObservationWrapper(
            env=env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs={"k_max": int(cfg.k_max), "arena_half": float(cfg.arena_half),
                                    "visibility_radius": float(cfg.visibility_radius)},
            num_particles=num_particles,
            pf_interaction_mapper=None,        # the filter reads the raw observation itself
            obs_mask_indices=None,             # the strip wrapper above does the hiding
            particle_filter_seed=seed + rank,
        )
        env = MsearchAgentObsWrapper(env)
        if shaping == "info_gain":
            env = PotentialBasedShapingWrapper(
                env, pf_belief_information_potential(scale=SHAPING_SCALE),
                gamma=1.0, zero_at_termination=False)
        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        else:
            env = Monitor(env)
        return env

    return _init


def _make_vec_env_from_fns(env_fns, n_envs: int):
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def _make_vec_normalize(vec_env, training: bool, norm_reward: bool):
    """Normalise the base obs (7 entries) and the reward, never the PF weights."""
    return VecNormalize(vec_env, training=training, norm_obs=True, norm_reward=norm_reward,
                        norm_obs_keys=["obs"])


# ---------------------------------------------------------------------------
# The domain's flags and env hook
# ---------------------------------------------------------------------------


def _add_arguments(parser) -> None:
    parser.add_argument(
        "--shaping", type=str, default=None, choices=SHAPING_MODES,
        help="Training-reward shaping. 'info_gain': Brendan's information-gain proxy reward "
             f"(scale {SHAPING_SCALE:g} x nats of prior mass ruled out; gamma 1; not zeroed at "
             "termination) added on the TRAINING env only. 'none': the task reward alone. "
             "Default: the variant's (info_gain). Evaluation is always on the true reward.")


def _resolve_arguments(parser, args) -> dict:
    variant = resolve(args.variant)
    if args.shaping is None:
        args.shaping = variant.default_shaping
    if not getattr(args, "total_timesteps_given", True) \
            and variant.default_total_timesteps != args.total_timesteps:   # change C, 2026-09-19
        args.total_timesteps = variant.default_total_timesteps
    print(f"msearch variant {args.variant!r}: training shaping {args.shaping}"
          + (f" (scale {SHAPING_SCALE:g})" if args.shaping == "info_gain" else "")
          + "; evaluation on the true reward")
    return {"shaping": args.shaping}


def _config_record(name: str) -> dict:
    cfg = _env_config(name)
    return dict(arena_half=float(cfg.arena_half), k_min=int(cfg.k_min), k_max=int(cfg.k_max),
                mode_scale_lo=float(cfg.mode_scale_lo), mode_scale_hi=float(cfg.mode_scale_hi),
                min_separation=float(cfg.min_separation), visibility_radius=float(cfg.visibility_radius),
                speed=float(cfg.speed), max_steps=int(cfg.max_steps), step_penalty=float(cfg.step_penalty),
                find_bonus=float(cfg.find_bonus))


def _run_config_extras(args) -> dict:
    # `shaping` itself is recorded by the trainer from the parsed flags; only its scale is added.
    extras = dict(episode_cap=episode_cap(args.variant),
                  agent_obs_dim=AGENT_OBS_DIM, particle_frame="agent_relative",
                  env_config=_config_record(args.variant))
    if args.shaping == "info_gain":
        extras["shaping_scale"] = SHAPING_SCALE
    return extras


def _make_env(variant: str, *, num_particles: int, particle_filter_class: type, seed: int,
              rank: int, monitor_dir: str | None, training: bool, options: dict):
    """One worker's env. The eval env (``training=False``) is UNSHAPED: reported numbers are
    the true task reward."""
    shaping = options.get("shaping", resolve(variant).default_shaping) if training else "none"
    return make_msearch_belief_env(
        num_particles=num_particles, rank=rank, seed=seed, monitor_dir=monitor_dir,
        variant=variant, particle_filter_class=particle_filter_class, shaping=shaping)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _eval_report(episodes, references, args, variant, cap) -> dict:
    """Found (= success) / timeout from the env's own ``info["found"]``; mean steps among the
    found episodes; mean reward (the true task reward: the eval env is unshaped)."""
    n = len(episodes)
    rewards = np.array([episode.total_reward for episode in episodes], dtype=float)
    lengths = np.array([episode.length for episode in episodes], dtype=int)
    found = np.array([bool(episode.final_info.get("found", False)) for episode in episodes], dtype=bool)
    successes = int(found.sum())
    summary = dict(
        n_episodes=n, successes=successes,
        success_rate=float(successes / n) if n else float("nan"),
        outcome_counts={"found": successes, "timeout": int(n - successes)},
        outcome_found=float(successes / n) if n else float("nan"),
        outcome_timeout=float((n - successes) / n) if n else float("nan"),
        mean_reward=float(rewards.mean()), std_reward=float(rewards.std()),
        mean_length=float(lengths.mean()), std_length=float(lengths.std()))
    print(f"\n=== Eval over {n} episodes (deterministic={args.deterministic}, true reward) ===")
    print(f"Found         : {successes}/{n} ({100 * summary['success_rate']:.1f}%)")
    print(f"Outcomes      : found {successes}, timeout {n - successes}")
    print(f"Mean reward   : {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Mean length   : {lengths.mean():.1f} ± {lengths.std():.1f} (cap {cap})")
    if successes:
        summary["found_mean_length"] = float(lengths[found].mean())
        print(f"When found    : mean_len={lengths[found].mean():.1f}")
    summary["episodes"] = [dict(reward=float(r), length=int(l), success=bool(f))
                           for r, l, f in zip(rewards, lengths, found)]
    return summary


# ---------------------------------------------------------------------------
# Dataset collection (the shared loop is rl/collect.py; what is here is what only msearch
# knows: the behaviour mix and the labels). Particles are stored as the agent-relative filter
# hands them (raw arena units, scale ARENA_HALF, no centre), uniform weights, no rebalancing.
# ---------------------------------------------------------------------------


def _collect_add_arguments(parser) -> None:
    parser.add_argument("--tour_frac", type=float, default=0.6,
                        help="Fraction of episodes driven by the informed mode tour (nearest "
                             "unvisited mode, read off the live env, plus a little noise); the "
                             "rest are OU random walks.")
    parser.add_argument("--tour_noise", type=float, default=0.2,
                        help="Std of the Gaussian noise added to the tour's unit direction.")


def _collect_resolve_arguments(parser, args, domain) -> dict:
    variant = resolve(args.variant)
    if args.timesteps is None:
        args.timesteps = episode_cap(args.variant)
    if args.num_particles is None:
        args.num_particles = DEFAULT_NUM_PARTICLES
    cfg = _env_config(args.variant)
    print(f"Variant: {args.variant} | env: {variant.env_id} | cap={args.timesteps} | "
          f"k_max={cfg.k_max} | tour_frac={args.tour_frac} (unshaped env; the reward is not recorded)")
    return {"env_id": variant.env_id, "particle_filter_class": variant.particle_filter,
            "k_max": int(cfg.k_max), "arena_half": float(cfg.arena_half)}


def _collect_prepare(args, options):
    from types import SimpleNamespace   # noqa: PLC0415
    return SimpleNamespace(rng=np.random.default_rng(args.seed), kinds={"walk": 0, "tour": 0})


def _collect_make_env(args, options, state):
    return make_msearch_belief_env(num_particles=args.num_particles, rank=0, seed=args.seed,
                                   variant=args.variant,
                                   particle_filter_class=options["particle_filter_class"],
                                   shaping="none")()


def _collect_begin_episode(args, options, state, env, episode):
    """Draw order: reset seed, behaviour kind. 'tour' = the viability probe's informed policy
    (``experiments/benchmark/probe_env.py::_msearch_informed``: nearest unvisited mode, arrival
    tolerance 3.0, then a random walk once every mode is visited) with a little noise; 'walk' =
    hunt's OU random walk."""
    rng = state.rng
    unwrapped = env.unwrapped
    reset_seed = int(rng.integers(1 << 30))
    kind = "tour" if rng.random() < args.tour_frac else "walk"
    state.kinds[kind] += 1
    memo = {"prev": np.zeros(2), "visited": set()}

    def act(obs):
        pos = np.asarray(unwrapped.agent_pos, dtype=np.float64)
        if kind == "tour":
            modes = np.asarray(unwrapped.mode_means, dtype=np.float64)
            remaining = [i for i in range(len(modes)) if i not in memo["visited"]]
            if remaining:
                j = min(remaining, key=lambda i: np.linalg.norm(modes[i] - pos))
                d = modes[j] - pos
                if np.linalg.norm(d) < 3.0:
                    memo["visited"].add(j)
                n = np.linalg.norm(d)
                a = (d / n if n > 1e-6 else rng.standard_normal(2)) + args.tour_noise * rng.standard_normal(2)
                memo["prev"] = np.clip(a, -1.0, 1.0)
                return memo["prev"]
        memo["prev"] = np.clip(0.8 * memo["prev"] + 0.6 * rng.standard_normal(2), -1.0, 1.0)
        return memo["prev"]

    return {"seed": reset_seed}, act


def _collect_snapshot_extras(args, options, state, env, obs, step_index) -> dict:
    """The labels of one snapshot in the SCALED, AGENT-RELATIVE frame the particles use
    (divide by arena_half; offsets from the agent): the target, every mode's mean and
    covariance padded to k_max with a validity flag, the count, the target's mode."""
    u = env.unwrapped
    K = int(options["k_max"])
    scale = float(options["arena_half"])
    pos = np.asarray(u.agent_pos, dtype=np.float64)
    k = len(u.mode_means)
    means = np.zeros((K, 2), np.float32)
    covs = np.zeros((K, 2, 2), np.float32)
    valid = np.zeros(K, np.float32)
    means[:k] = (np.asarray(u.mode_means) - pos) / scale
    covs[:k] = np.asarray(u.mode_covs) / scale ** 2
    valid[:k] = 1.0
    return dict(agent=np.asarray(pos / scale, np.float32),
                target=np.asarray((np.asarray(u.target_pos) - pos) / scale, np.float32),
                mode_means=means, mode_covs=covs, mode_valid=valid,
                num_modes=np.int64(k), target_mode=np.int64(u.target_mode),
                step=np.int32(step_index))


def _collect_finish(args, options, state, n_snapshots) -> None:
    print("Episodes -- " + ", ".join(f"{k}: {v}" for k, v in state.kinds.items())
          + f"; snapshots (raw): {n_snapshots}")


def _collect_report(args, options, particles, weights, steps, stage) -> None:
    if stage != "final":
        return
    print(f"Dataset: particles {particles.shape} (agent-relative, raw units), weights "
          f"{weights.shape} (uniform)")
    for dim in range(particles.shape[-1]):
        column = particles[:, :, dim]
        print(f"  dim {dim}: [{column.min():.2f}, {column.max():.2f}]")
    print(f"  particle_scale (recorded for pretraining): {options['arena_half']}")


def _collect_metadata_extras(args, options, particles, weights, steps) -> dict:
    return dict(episode_cap=episode_cap(args.variant), tour_frac=args.tour_frac,
                tour_noise=args.tour_noise, particle_frame="agent_relative",
                env_config=_config_record(args.variant),
                label_arrays=["agent", "target", "mode_means", "mode_covs", "mode_valid",
                              "num_modes", "target_mode", "step"])


MSEARCH_COLLECTION = Collection(
    add_arguments=_collect_add_arguments,
    defaults={"seed": 7, "num_episodes": 4000},
    resolve_arguments=_collect_resolve_arguments,
    particle_scale=lambda args, options: float(options["arena_half"]),
    prepare=_collect_prepare,
    make_env=_collect_make_env,
    begin_episode=_collect_begin_episode,
    finish=_collect_finish,
    report=_collect_report,
    metadata_extras=_collect_metadata_extras,
    snapshot_extras=_collect_snapshot_extras,
    progress_desc="Collecting msearch episodes",
)


# ---------------------------------------------------------------------------
# Encoder defaults
# ---------------------------------------------------------------------------

#: Target for ``t_bound * (decision-relevant length / arena half-width)`` (PITFALLS section 9);
#: the decision-relevant length here is the visibility radius.
CGF_TILT_TARGET = 3.0


def cgf_t_bound(name: str, target: float = CGF_TILT_TARGET) -> float:
    """``target / (visibility_radius / arena_half)``: 3 / (1.5 / 14) = 28."""
    cfg = _env_config(name)
    return float(target) / (float(cfg.visibility_radius) / float(cfg.arena_half))


def _cgf_t_bound_default(variant: str) -> float:
    bound = cgf_t_bound(variant)
    print(f"CGF t_bound from the sizing rule: {bound:.3g} "
          f"(= {CGF_TILT_TARGET} / (visibility radius / arena half-width) for variant {variant!r})")
    return bound


def _cgf_t_init_max_default(args):
    """The Ant-Tag / hunt rule: the ``spread`` init tops out at 0.8 * t_bound; otherwise unset."""
    if args.t_param == "clamp":
        return None
    if args.t_init_mode == "spread":
        return 0.8 * float(args.t_bound)
    return None


MSEARCH = Domain(
    name="msearch",
    particle_dim=2,
    default_sinkhorn_blur=0.02,      # 2026-09-19: the recipes' recorded blur
    default_variant="msearch",
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
    schedules=lambda args: (),
    run_config_extras=_run_config_extras,
    default_num_particles=lambda variant: DEFAULT_NUM_PARTICLES,
    default_arena_scale=lambda variant: float(_env_config(variant).arena_half),
    default_device="cpu",
    default_total_timesteps=3_000_000,
    encoder_defaults={
        # The 2-D probe recipe (polar ball, spread init, raw features), bound from the sizing rule.
        "cgf": dict(t_param="polar", t_bound=None, t_init_mode="spread", feature_norm="none",
                    t_bound_default=_cgf_t_bound_default,
                    t_init_max_default=_cgf_t_init_max_default),
        # The small ST (16 inducing points, hidden 64, two post-PMA SABs; ~109k parameters).
        "st": dict(num_inds=16, dim_hidden=64, num_post_sab=2),
    },
    evaluation=Evaluation(default_n_episodes=300, reseed_per_episode=True, report=_eval_report),
    # The generic reconstruction objective (and alignment) is the pretraining here; no objective
    # of msearch's own yet (a `task` head is a later line).
    collection=MSEARCH_COLLECTION,
)
