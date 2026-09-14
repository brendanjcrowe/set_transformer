"""Ant-Tag domain for the RL harness: variant registry, wrappers, curriculum, PF glue, env factory.

Three moves built this module (harness centralisation, ``refactor_plans.md`` in the parent
repo, 2026-09-12): change 1a brought the wrappers and particle-filter glue over from
``experiments/ant_tag/4_train_rl_frozen.py``; change 1b the variant registry from
``experiments/ant_tag/variants.py``; change 1c the env factory and its helpers from
``experiments/ant_tag/4_train_rl_cgf.py``. Every block is byte-identical to its original, and
each old file re-exports the names under its historical flat module name, which the numbered
scripts, eval scripts, diagnostics and tests import.

Variant registry
  Every script needs the same facts about a variant: the gym env id, the particle filter
  that mirrors its target motion model, its default schedules, and where its runs live.
  They are stated once in ``VARIANTS`` and every script takes ``--variant``. Two things this
  makes structural rather than remembered: (1) the env/filter pairing is one line, not an
  assertion repeated per script (a filter that drifts from its env makes belief propagation
  silently wrong); (2) the episode cap is READ from the gym registration by ``episode_cap``,
  so ``--max_steps`` is an override, not a required flag (a wrong cap counts timeouts as
  tags). Precedence for schedules: CLI, then the variant's default, then the script's.
  Adding a variant: one entry here, and it appears in every script at once.

Wrappers, curriculum, PF glue
  PFRewardShapingWrapper        dense shaping from the PF belief (distance, weight entropy,
                                tag bonus, spread gain); coefficients settable mid-run
  CurriculumVisibilityWrapper   visibility radius and evasion scale settable mid-run
  ANT_TAG_SCHEDULES             the three annealed quantities (visibility radius, four reward
                                coefficients, evasion scale) as rl/curriculum.Schedule
                                descriptions: setter name, values per waypoint, script default
  CurriculumCallback            adapter with the historical three-schedule signature over
                                rl/curriculum.ScheduleCallback (goes with the scripts, change 5)
  _CurriculumRouter             rl/curriculum.ScheduleRouter over the three setters, so
                                SubprocVecEnv.env_method reaches the inner wrappers
  _set_*_recursive              older per-wrapper setters, no longer called (kept for the
                                forwarding file until change 5)
  get_ant_tag_pf_kwargs         AntTagParticleFilter constructor kwargs read off the live env
  ant_tag_pf_interaction_mapper bridge from one step's observation and env state to the
                                filter's predict / update kwargs (visibility, evasion, target
                                speed, ghost ping, den and counterweighted-den geometry)

Env factory
  make_ant_tag_cgf_env          returns a thunk building one worker's env: gym.make ->
                                CurriculumVisibilityWrapper -> PFDictWithWeightsObservationWrapper
                                (filter seeded per worker and episode) -> PFRewardShapingWrapper
                                (training only) -> Monitor -> _CurriculumRouter. Every arm and
                                the eval build the env through it, so the belief the encoders
                                see is the same everywhere.
  get_ant_tag_arena_scale       cage half-width off a live env (the encoders' particle scale;
                                same derivation as the registry's ``arena_scale``)
  get_env_visible_radius        the real visibility radius off a live env (eval env)
  _make_vec_normalize           VecNormalize over the base obs and reward only, never the
                                PF weights (``norm_obs_keys=["obs"]``)
  _make_vec_env_from_fns        SubprocVecEnv above one worker, DummyVecEnv at one

``import pdomains`` below registers the ``pdomains-ant-tag-*`` env ids the registry names;
it does not load MuJoCo (that happens on ``gym.make``), so importing this module stays cheap.
The factory thunk is pickled by reference to this module when SubprocVecEnv starts workers,
and the child's import of it registers the envs there too.
"""

import os
from dataclasses import dataclass, replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers the pdomains-ant-tag-* envs
from set_transformer.rl.curriculum import (
    Schedule,
    ScheduleCallback,
    ScheduleRouter,
    interpolate,
    parse_curriculum,
    parse_reward_schedule,
)
from set_transformer.rl.curriculum import parse_reward_schedule as _parse_reward_schedule
from set_transformer.rl.domains.base import (
    Collection,
    Domain,
    Evaluation,
    Objective,
    PretrainContext,
    PretrainResult,
    Pretraining,
)
from set_transformer.rl.particle_filters import ant_tag as _ant_tag_filters
from set_transformer.rl.particle_filters.ant_tag import (
    AntTagParticleFilter,
    CounterweightedDenAntTagParticleFilter,
    GhostAntTagParticleFilter,
    SmartAntTagParticleFilter,
    TwinDenAntTagParticleFilter,
)
from set_transformer.rl.particle_filters.base import BaseParticleFilter
from set_transformer.rl.pretrain_objectives.task_head import (   # 10.9: the shared task-head loop
    TaskData as _TaskData,
    add_task_arguments,
    check_variant_and_geometry,
    locate_from_dataset,
    resolve_task_arguments,
    run_task_training,
    task_run_name,
)
from set_transformer.rl.wrappers.particle_filter import PFDictWithWeightsObservationWrapper

# ---------------------------------------------------------------------------
# Variant registry (moved from experiments/ant_tag/variants.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One Ant-Tag env plus the filter that mirrors its target motion model."""

    env_id: str
    particle_filter: type
    #: What the belief looks like in this env, and anything a caller has to
    #: know that the code cannot enforce. Printed by --list_variants.
    notes: str = ""
    #: Curriculum string that suits this env's geometry, used when the caller
    #: passes no --curriculum. None means the script's own default.
    default_curriculum: str | None = None
    #: Evasion schedule that suits this env. None means the script's default.
    default_evasion_curriculum: str | None = None
    #: Reward-shaping schedule 'frac:distance:entropy:tag_bonus,...' that is
    #: THE recipe for this env, used when the caller passes no
    #: --reward_schedule. None means the script's own default, which has
    #: PF-entropy 0 throughout -- the pre-2026-07-31 no-search-incentive
    #: configuration (domain_mds/smart_ant_tag.md). Set it only where a
    #: recipe has been established on that variant.
    default_reward_schedule: str | None = None
    #: The heads of the ``task`` pretraining objective on this env (batch 10.9, 2026-09-14;
    #: :data:`TASK_HEADS` lists them): ``position`` = the true target position in the scaled
    #: frame (unimodal, fleeing target: the smart family); ``den_mass`` = the weighted particle
    #: shares in the heavy den / light den / outside, soft cross-entropy against the row's OWN
    #: cloud (the counterweighted-den family; ``den_occupied`` is stored for the report only).
    #: Empty = the objective refuses the variant (``ghost`` / ``dens``: not designed yet).
    task_heads: tuple[str, ...] = ()


VARIANTS: dict[str, Variant] = {
    "base": Variant(
        env_id="pdomains-ant-tag-v0",
        particle_filter=AntTagParticleFilter,
        task_heads=("position",),
        notes="Original dumb-target Ant-Tag. Unimodal belief.",
    ),
    "smart": Variant(
        env_id="pdomains-ant-tag-smart-v0",
        particle_filter=SmartAntTagParticleFilter,
        task_heads=("position",),
        notes="Unimodal; the target flees harder as the ant closes.",
        # The schedules every July 2026 `smart` run recorded in its
        # run_config.json (the CGF 79% / Gaussian 65% comparison). Stated here
        # because the script fallback differs on both counts and neither
        # difference raises: it breaks the ramp at 0.3/0.7 instead of 0.2/0.5,
        # and a missing evasion default resolves to a CONSTANT scale 1.0, i.e.
        # a full-strength evading target during the locomotion warm-start that
        # the warm-start exists to avoid.
        default_curriculum="0:100,0.2:100,0.5:3,1:3",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
        # entropy_flat: distance 1 -> 0.15, PF-entropy flat 2, tag 0 -> 50.
        # The 2026-07-31 winner (CGF 79%, Gaussian 65%); every run behind a
        # headline number on this variant used exactly this string.
        default_reward_schedule="0:1:2:0,0.2:1:2:0,0.5:0.15:2:50,1:0.15:2:50",
    ),
    "smart_hard": Variant(
        env_id="pdomains-ant-tag-smart-hard-v0",
        particle_filter=SmartAntTagParticleFilter,
        task_heads=("position",),
        notes=("SmartAntTag with the cdens_hard sensing/tagging geometry: "
               "tag_radius 0.6, visible_radius 1.0 (base env 1.5 / 3.0). "
               "Same 9x9 cage and 400-step cap as `smart`; unimodal belief."),
        default_curriculum="0:100,0.2:100,0.5:1.0,1:1.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "smart_mid": Variant(
        env_id="pdomains-ant-tag-smart-mid-v0",
        particle_filter=SmartAntTagParticleFilter,
        task_heads=("position",),
        notes=("SmartAntTag, tag_radius 1.0, visible_radius 2.0, target_step "
               "0.5. Geometry sweep around smart_hard (visible 2.0 covers ~16% "
               "of the 9x9 cage)."),
        default_curriculum="0:100,0.2:100,0.5:2.0,1:2.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "smart_hard_slow": Variant(
        env_id="pdomains-ant-tag-smart-hard-slow-v0",
        particle_filter=SmartAntTagParticleFilter,
        task_heads=("position",),
        notes=("smart_hard radii (tag 0.6, visible 1.0) with a slower target: "
               "target_step 0.3 instead of 0.5. The filter reads target_step "
               "off the env."),
        default_curriculum="0:100,0.2:100,0.5:1.0,1:1.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "smart_mid_slow": Variant(
        env_id="pdomains-ant-tag-smart-mid-slow-v0",
        particle_filter=SmartAntTagParticleFilter,
        task_heads=("position",),
        notes=("Both relaxations: tag 1.0, visible 2.0, target_step 0.3."),
        default_curriculum="0:100,0.2:100,0.5:2.0,1:2.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "smart_mid_slow_v15": Variant(
        env_id="pdomains-ant-tag-smart-mid-slow-v15-v0",
        particle_filter=SmartAntTagParticleFilter,
        task_heads=("position",),
        notes=("smart_mid_slow with visible_radius 1.5 instead of 2.0 (tag 1.0, "
               "target_step 0.3). Smaller flee zone AND smaller visible area."),
        # Reaches the real radius at 40%, not 50%: this is what all 57 ST runs
        # in the 9-arm comparison passed via the driver's VIS_CURRICULUM. The
        # registry said 0.5 until 2026-09-09, so a rerun that took the default
        # would not have matched those results.
        default_curriculum="0:100,0.2:100,0.4:1.5,1:1.5",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
        # The same entropy_flat recipe (RECIPE=entropy_flat in the driver);
        # 39 of the 57 ST runs here, including the 9-arm comparison, used it.
        # The fork ablation's dist0 variant is a paired arm, not the default.
        default_reward_schedule="0:1:2:0,0.2:1:2:0,0.5:0.15:2:50,1:0.15:2:50",
    ),
    "ghost": Variant(
        env_id="pdomains-ant-tag-ghost-v0",
        particle_filter=GhostAntTagParticleFilter,
        notes="An unreliable long-range ping spawns a second belief mode.",
    ),
    "dens": Variant(
        env_id="pdomains-ant-tag-dens-v0",
        particle_filter=TwinDenAntTagParticleFilter,
        notes="Two mirrored hideouts, one tight and one loose. 200-step cap.",
    ),
    "cdens": Variant(
        env_id="pdomains-ant-tag-cdens-v0",
        particle_filter=CounterweightedDenAntTagParticleFilter,
        task_heads=("den_mass",),
        notes=("Heavy-near / light-far dens; the pooled belief mean is pinned "
               "at 0 by construction, so the mirror bit lives only in the odd "
               "moments. Visibility radius 1.8."),
        default_curriculum="0:100,0.2:100,0.5:1.8,1:1.8",
    ),
    "cdens_hard": Variant(
        env_id="pdomains-ant-tag-cdens-hard-v0",
        particle_filter=CounterweightedDenAntTagParticleFilter,
        task_heads=("den_mass",),
        notes=("Counterweighted dens with harder sensing/tagging geometry: "
               "tag_radius 0.6, visible_radius 1.0, spook_radius 1.4."),
        default_curriculum="0:100,0.2:100,0.5:1.0,1:1.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "cdens_terminal": Variant(
        env_id="pdomains-ant-tag-cdens-terminal-v0",
        particle_filter=CounterweightedDenAntTagParticleFilter,
        task_heads=("den_mass",),
        notes=("Counterweighted dens where the two INACTIVE candidate "
               "positions are terminal hazards (-300), so probing the wrong "
               "arrangement ends the episode instead of wasting time."),
        default_curriculum="0:100,0.2:100,0.5:1.0,1:1.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "cdens_nospook": Variant(
        env_id="pdomains-ant-tag-cdens-nospook-v0",
        particle_filter=CounterweightedDenAntTagParticleFilter,
        task_heads=("den_mass",),
        notes="Counterweighted dens with the spook alarm disabled (ablation).",
    ),
}

#: Variants whose target evades (SmartAntTagEnv and its subclasses). Only these
#: respond to --evasion_curriculum and --target_speed_scale; set membership
#: keyed on the registry, rather than a branch in every script.
EVADING = {"smart", "smart_hard", "smart_mid", "smart_hard_slow",
           "smart_mid_slow", "smart_mid_slow_v15", "ghost", "dens", "cdens",
           "cdens_hard", "cdens_terminal", "cdens_nospook"}


def resolve(name: str) -> Variant:
    """Look up a variant, listing the valid names on a typo."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(
            f"Unknown variant {name!r}. Available: {sorted(VARIANTS)}"
        ) from None


def episode_cap(name: str) -> int:
    """The variant's episode cap, read from its gym registration.

    The single source of truth. An eval that hardcodes a different number
    counts timeouts as tags.
    """
    spec = gym.spec(resolve(name).env_id)
    if spec.max_episode_steps is None:
        raise ValueError(
            f"{spec.id} registers no max_episode_steps; pass --max_steps")
    return int(spec.max_episode_steps)


#: Target for ``t_bound * (tag_radius / arena half-width)``. The CGF only
#: resolves structure at length L once t * L is of order 2..4 (oddeven.md
#: 2026-09-04/05: at t * L = 0.16 every feature was a multiple of the
#: posterior mean); 3 is the middle of that range. Frame-invariant: shifting
#: every particle by a constant changes K by an additive term and K' by the
#: shift, and leaves the tilted weights untouched.
CGF_TILT_TARGET = 3.0


def arena_scale(name: str) -> float:
    """The CGF / ST particle normalisation: the cage half-width, read off the
    live env (`cage_max_x`; the arena is square). Same derivation as
    `4_train_rl_cgf.get_ant_tag_arena_scale`, exposed here so the registry can
    size the CGF bound without importing a training script."""
    env = gym.make(resolve(name).env_id, rendering=False)
    try:
        unwrapped = env.unwrapped
        if not abs(float(unwrapped.cage_max_x) - float(unwrapped.cage_max_y)) < 1e-9:
            raise ValueError(
                f"{resolve(name).env_id} is not a square arena "
                f"({unwrapped.cage_max_x} x {unwrapped.cage_max_y})")
        return float(unwrapped.cage_max_x)
    finally:
        env.close()


def cgf_t_bound(name: str, target: float = CGF_TILT_TARGET) -> float:
    """The CGF probe bound for this variant: ``target / (tag_radius / arena_scale)``.

    smart (tag 1.5 / 4.5) -> 9.0, smart_mid_slow_v15 (1.0 / 4.5) -> 13.5,
    cdens_terminal (0.6 / 7.0) -> 35.0. The legacy clamp of 2.0 puts every
    variant at t * L between 0.17 and 0.67, the mean regime. Used as the
    default ``--t_bound`` of the CGF arm's tanh / polar modes; recorded in
    run_config.json either way.
    """
    env = gym.make(resolve(name).env_id, rendering=False)
    try:
        tag_radius = float(env.unwrapped.tag_radius)
    finally:
        env.close()
    if tag_radius <= 0:
        raise ValueError(f"{resolve(name).env_id} reports tag_radius={tag_radius}")
    return float(target) / (tag_radius / arena_scale(name))


def run_subdir(encoder: str, name: str) -> str:
    """Where this (encoder, variant) pair's runs live under runs/.

    "base" keeps the historical bare name (runs/ant_tag_gaussian), so existing
    run directories stay where they are.
    """
    resolve(name)
    return f"ant_tag_{encoder}" if name == "base" else f"ant_tag_{encoder}_{name}"


def warn_if_not_evading(name: str, evasion_curriculum, target_speed_scale) -> None:
    """Warn when evasion knobs are set on an env that has no evasion.

    `evasion_scale` and `target_speed_scale` are read off the env by
    `getattr(..., default)`, so on the base dumb-target env they are silent
    no-ops. Someone who passes --evasion_curriculum there gets a run that
    looks configured and is not, and the run_config records the schedule as
    if it had applied.
    """
    if name in EVADING:
        return
    unused = []
    if evasion_curriculum is not None:
        unused.append("--evasion_curriculum")
    if target_speed_scale is not None:
        unused.append("--target_speed_scale")
    if unused:
        print(f"WARNING: variant {name!r} has no evading target; "
              f"{', '.join(unused)} will have no effect.")


def warn_if_override_contradicts(name: str, env_id=None, particle_filter=None) -> None:
    """Warn when an explicit override disagrees with the chosen variant.

    The whole point of the registry is that an env travels with the filter
    that mirrors its target motion model. Overriding one and not the other
    reintroduces exactly the silent divergence the registry removes, so it
    must never happen quietly.
    """
    variant = resolve(name)
    if env_id is not None and env_id != variant.env_id:
        print(f"WARNING: --env_id {env_id!r} overrides variant {name!r} "
              f"(env {variant.env_id!r}) while keeping its particle filter "
              f"{variant.particle_filter.__name__}. If that filter does not "
              "mirror the override env's target motion, the stored/propagated "
              "beliefs diverge from the env with no further error.")
    if (particle_filter is not None
            and particle_filter is not variant.particle_filter):
        print(f"WARNING: --particle_filter {particle_filter.__name__} "
              f"overrides variant {name!r}, whose env {variant.env_id!r} is "
              f"mirrored by {variant.particle_filter.__name__}.")


def resolve_schedule(name: str, cli_value, field: str, script_default):
    """Pick a schedule string: CLI wins, then the variant's, then the script's.

    Lets a variant carry the curriculum that suits its geometry (the
    counterweighted dens need a 1.0-1.8 visibility radius, not the base env's
    3) without every caller having to remember it, while an explicit flag
    still overrides.
    """
    if cli_value is not None:
        return cli_value
    variant_default = getattr(resolve(name), field)
    return variant_default if variant_default is not None else script_default


def add_variant_argument(parser, default: str = "base") -> None:
    """Attach the shared --variant / --list_variants flags to a parser."""
    parser.add_argument(
        "--variant", type=str, default=default, choices=sorted(VARIANTS),
        help=f"Ant-Tag env variant (default: {default}). Selects env id, the "
             "matching particle filter, the run subdirectory and the episode "
             "cap together, so they cannot disagree.",
    )
    parser.add_argument(
        "--list_variants", action="store_true",
        help="Print the variant registry and exit.",
    )


def print_variants() -> None:
    """Human-readable dump of the registry, for --list_variants."""
    for name, variant in VARIANTS.items():
        try:
            cap = episode_cap(name)
        except Exception:  # noqa: BLE001 - listing must not fail on one bad entry
            cap = "?"
        print(f"{name:16s} {variant.env_id:38s} cap={cap:<5} "
              f"{variant.particle_filter.__name__}"
              + (f"  task_heads={'+'.join(variant.task_heads)}" if variant.task_heads else ""))
        if variant.notes:
            print(f"{'':16s}   {variant.notes}")



# ---------------------------------------------------------------------------
# Wrappers, curriculum callback, PF glue (moved from experiments/ant_tag/4_train_rl_frozen.py)
# ---------------------------------------------------------------------------


#: The three quantities the Ant-Tag curriculum anneals (change 3, 2026-09-12): where each
#: goes (the setter on the wrapper below), how many values a waypoint carries, the status-line
#: labels, and the historical SCRIPT default (a variant's own defaults, `Variant.default_*`,
#: take precedence over these; the CLI over both). rl/curriculum.ScheduleCallback runs them;
#: the shared trainer (change 4) reads this tuple to build the callback and the CLI flags.
ANT_TAG_SCHEDULES = (
    Schedule("visibility", target="set_curriculum_radius", n_values=1,
             waypoints=((0.0, 100.0), (0.3, 100.0), (0.7, 3.0), (1.0, 3.0)),
             labels=(("vis_radius", ".2f"),)),
    Schedule("reward", target="set_reward_coeffs", n_values=4,
             waypoints=((0.0, 1.0, 0.0, 0.0), (0.3, 1.0, 0.0, 0.0),
                        (0.7, 0.0, 0.0, 50.0), (1.0, 0.0, 0.0, 50.0)),
             labels=(("dist_coeff", ".3f"), ("ent_coeff", ".3f"),
                     ("tag_bonus", ".1f"), ("gain_coeff", ".2f"))),
    Schedule("evasion", target="set_evasion_scale", n_values=1,
             waypoints=((0.0, 1.0), (1.0, 1.0)),
             labels=(("evasion_scale", ".2f"),)),
)


class PFRewardShapingWrapper(gym.Wrapper):
    """
    Dense reward shaping using the particle filter belief state.

    Four components:
      1. Distance: -dist_to_pf_mean (closer to belief mean = higher reward)
      2. Entropy: -sum(w_i * log(w_i)) of PF weights (lower entropy = higher reward)
      3. Tag bonus: large reward on successful tag (terminated=True)
      4. Spread gain (2026-09-08): + (spread_{t-1} - spread_t), where spread is
         the weighted std of the particle POSITIONS averaged over coordinates
         (the collector's belief-spread measure). Pays for shrinking the
         region the target could be in -- a sweep through a mass-carrying
         part of a diffuse belief removes particles and earns it; standing
         still (diffusion grows the spread) or re-sweeping an empty region
         earns nothing. Potential-based: over an episode it telescopes to the
         total shrinkage. Added because the weight entropy (2) is ~constant
         during a blind search (weights are uniform after every resample),
         so it carries no signal about WHERE to look
         (domain_mds/smart_mid_slow_v15_ant_tag.md, 2026-09-08). gain_coeff 0
         (the default and the meaning of a 3- or 4-field schedule) reproduces
         the historical reward exactly.

    Coefficients are mutable — the CurriculumCallback can phase distance out
    and tag bonus in over the course of training.
    """

    def __init__(self, env: gym.Env, distance_coeff: float = 1.0,
                 entropy_coeff: float = 0.0, tag_bonus_coeff: float = 0.0,
                 gain_coeff: float = 0.0):
        super().__init__(env)
        self.distance_coeff = distance_coeff
        self.entropy_coeff = entropy_coeff
        self.tag_bonus_coeff = tag_bonus_coeff
        self.gain_coeff = gain_coeff
        self._prev_spread: float | None = None

    def set_reward_coeffs(self, distance_coeff: float, entropy_coeff: float,
                          tag_bonus_coeff: float, gain_coeff: float | None = None):
        """Called by CurriculumCallback to phase reward components.

        gain_coeff is optional so 3-argument callers (every pre-2026-09-08
        schedule) keep working; None leaves the current value untouched."""
        self.distance_coeff = distance_coeff
        self.entropy_coeff = entropy_coeff
        self.tag_bonus_coeff = tag_bonus_coeff
        if gain_coeff is not None:
            self.gain_coeff = gain_coeff

    def _get_pf_wrapper(self):
        """Walk the wrapper stack to find a wrapper exposing a particle_filter."""
        e = self.env
        while e is not None:
            if getattr(e, "particle_filter", None) is not None:
                return e
            e = getattr(e, "env", None)
        raise RuntimeError("PFRewardShapingWrapper requires a PF wrapper (with .particle_filter) in the stack")

    @staticmethod
    def _pf_entropy(weights: np.ndarray) -> float:
        """Shannon entropy: -sum(w_i * log(w_i)), with 0*log(0) = 0."""
        w = weights[weights > 0]
        return -float(np.sum(w * np.log(w)))

    @staticmethod
    def _pf_spread(particles: np.ndarray, weights: np.ndarray) -> float:
        """Weighted std of the particle positions per coordinate, averaged
        (same measure as 2_collect_pf_dataset._weighted_spread)."""
        w = np.clip(np.nan_to_num(weights, nan=0.0), 0.0, None)
        w = w / max(float(w.sum()), 1e-12)
        mu = (w[:, None] * particles).sum(axis=0)
        var = (w[:, None] * (particles - mu) ** 2).sum(axis=0)
        return float(np.sqrt(np.maximum(var, 0.0)).mean())

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pf = self._get_pf_wrapper().particle_filter
        self._prev_spread = self._pf_spread(pf.particles, pf.weights)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pf_wrapper = self._get_pf_wrapper()
        pf = pf_wrapper.particle_filter

        # Support both flat (concatenated) obs and Dict obs ({"obs": ..., "particles": ...})
        base_obs = obs["obs"] if isinstance(obs, dict) else obs
        ant_pos = base_obs[:2]
        pf_mean = np.average(pf.particles, weights=pf.weights, axis=0)
        dist_to_mean = float(np.linalg.norm(ant_pos - pf_mean))
        entropy = self._pf_entropy(pf.weights)

        spread = self._pf_spread(pf.particles, pf.weights)
        if self._prev_spread is None:      # step() without reset(): no baseline
            self._prev_spread = spread
        spread_gain = self._prev_spread - spread
        self._prev_spread = spread

        distance_reward = self.distance_coeff * (-dist_to_mean)
        entropy_reward = self.entropy_coeff * (-entropy)
        gain_reward = self.gain_coeff * spread_gain
        # Historically every base AntTag termination was a successful tag.
        # New variants may also terminate on an explicit failure state, in
        # which case the environment publishes is_success=False.  The fallback
        # preserves behavior for every legacy environment.
        is_success = bool(info.get("is_success", terminated))
        tag_bonus = self.tag_bonus_coeff if is_success else 0.0

        shaped_reward = reward + distance_reward + entropy_reward + tag_bonus + gain_reward
        info["distance_reward"] = distance_reward
        info["entropy_reward"] = entropy_reward
        info["tag_bonus"] = tag_bonus
        info["gain_reward"] = gain_reward
        info["pf_entropy"] = entropy
        info["pf_spread"] = spread
        info["pf_spread_gain"] = spread_gain
        info["dist_to_pf_mean"] = dist_to_mean
        return obs, shaped_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Curriculum: controllable visibility wrapper + annealing callback
# ---------------------------------------------------------------------------

class CurriculumVisibilityWrapper(gym.Wrapper):
    """
    Controls target visibility via a mutable visibility_radius.

    Sits between the base env and the PF wrapper. Overrides obs[-2:] based
    on the current curriculum radius rather than the env's fixed 3.0.

    During warm-start (visibility_radius=large), the target is always
    revealed so the PF collapses to the true position and the agent
    effectively trains with full observability through the ST pipeline.
    """

    def __init__(self, env: gym.Env, initial_visibility_radius: float = 100.0):
        super().__init__(env)
        # Publish immediately, not just in set_curriculum_radius(): the PF
        # interaction mapper only sees the unwrapped env, and falls back to
        # the env's fixed `visible_radius` (3.0) when this attribute is
        # absent. Without this, any env whose curriculum radius is never
        # updated -- every eval / render / dataset-collection script, which
        # never call set_curriculum_radius -- would silently tell the PF's
        # negative-information update a different radius than the one this
        # wrapper actually used to decide reveal/no-reveal. That is harmless
        # only while initial_visibility_radius happens to equal 3.0 (true of
        # every caller today) and silently wrong the moment it doesn't.
        self.set_curriculum_radius(initial_visibility_radius)

    def set_curriculum_radius(self, radius: float):
        """Called by CurriculumCallback via env_method.

        Also publishes the live radius onto the base env so the PF
        interaction mapper (which only sees `unwrapped_env`) can read the
        radius that's actually deciding reveal/no-reveal this step, instead
        of the base env's fixed `visible_radius`.
        """
        self.visibility_radius = radius
        self.env.unwrapped.current_visibility_radius = radius

    def set_evasion_scale(self, scale: float):
        """Called by CurriculumCallback via env_method.

        No-op for envs without an evasion_scale knob (e.g. base AntTagEnv),
        so this wrapper stays usable for both the dumb and smart target envs.
        """
        base = self.env.unwrapped
        if hasattr(base, "evasion_scale"):
            base.evasion_scale = scale

    def _apply_curriculum_visibility(self, obs: np.ndarray) -> np.ndarray:
        ant_pos = obs[:2]
        true_target = self.env.unwrapped.get_target_pos()
        dist = float(np.linalg.norm(ant_pos - true_target))

        obs = obs.copy()
        if dist < self.visibility_radius:
            obs[-2:] = true_target
        else:
            obs[-2:] = np.zeros(2)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._apply_curriculum_visibility(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._apply_curriculum_visibility(obs), reward, terminated, truncated, info


class CurriculumCallback(ScheduleCallback):
    """
    SB3 callback that anneals visibility_radius and reward coefficients.

    Visibility schedule: list of (progress_fraction, radius) waypoints.
    Reward schedule: list of (progress_fraction, distance_coeff, entropy_coeff).
    Between waypoints, values are linearly interpolated.

    Default visibility schedule:
      - 0-30%: radius=100 (fully observed)
      - 30-70%: linear decrease 100 -> 3.0
      - 70-100%: radius=3.0 (real POMDP)

    Default reward schedule (frac, distance_coeff, entropy_coeff, tag_bonus):
      - 0-30%: distance=1.0, entropy=0.0, tag=0 (pure distance while fully visible)
      - 30-70%: distance 1.0->0.0, entropy 0->0, tag 0->50 (phase in tag bonus)
      - 70-100%: distance=0.0, entropy=0.0, tag=50 (sparse tag reward at real POMDP)

    Since change 3 (2026-09-12) this is an ADAPTER: the historical three-schedule signature
    and defaults (ANT_TAG_SCHEDULES) over rl/curriculum.ScheduleCallback, which does the
    interpolation, the pushing and the status line for any domain. The four Ant-Tag scripts
    and several tests construct it this way; the shared trainer (change 4) builds
    ScheduleCallback from ANT_TAG_SCHEDULES directly, after which this class goes (change 5).
    `schedule` / `reward_schedule` / `evasion_schedule` / `_interpolate_schedule` are kept
    because tests read them.
    """

    def __init__(
        self,
        total_timesteps: int,
        schedule: list[tuple[float, float]] | None = None,
        reward_schedule: list[tuple[float, ...]] | None = None,
        evasion_schedule: list[tuple[float, float]] | None = None,
        verbose: int = 0,
    ):
        by_name = {s.name: s for s in ANT_TAG_SCHEDULES}
        if schedule is None:
            schedule = list(by_name["visibility"].waypoints)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        if reward_schedule is None:
            reward_schedule = list(by_name["reward"].waypoints)
        self.reward_schedule = sorted(reward_schedule, key=lambda x: x[0])
        # Constant scale=1.0 (full smart-target strength throughout) is a
        # no-op that matches SmartAntTagEnv's own default, so omitting this
        # arg preserves today's behavior exactly. Only meaningful for the
        # smart target env; harmlessly ignored by the base AntTagEnv.
        if evasion_schedule is None:
            evasion_schedule = list(by_name["evasion"].waypoints)
        self.evasion_schedule = sorted(evasion_schedule, key=lambda x: x[0])
        super().__init__(total_timesteps, (
            replace(by_name["visibility"], waypoints=tuple(self.schedule)),
            replace(by_name["reward"], waypoints=tuple(self.reward_schedule)),
            replace(by_name["evasion"], waypoints=tuple(self.evasion_schedule)),
        ), verbose=verbose)

    def _interpolate_schedule(self, schedule, progress: float):
        """Linearly interpolate a schedule. Returns all values after the fraction."""
        return interpolate(schedule, progress)


#: The three per-wrapper setters below predate ScheduleRouter and are no longer called by
#: the package (the router now finds the setter by name); kept for the forwarding file
#: `experiments/ant_tag/4_train_rl_frozen.py`, which re-exports them, until change 5.
def _set_radius_recursive(env, radius: float):
    """Walk the wrapper stack and set visibility_radius on CurriculumVisibilityWrapper."""
    e = env
    while e is not None:
        if isinstance(e, CurriculumVisibilityWrapper):
            e.set_curriculum_radius(radius)
            return
        e = getattr(e, "env", None)


def _set_evasion_scale_recursive(env, scale: float):
    """Walk the wrapper stack and set evasion_scale on the base smart-target env."""
    e = env
    while e is not None:
        if isinstance(e, CurriculumVisibilityWrapper):
            e.set_evasion_scale(scale)
            return
        e = getattr(e, "env", None)


def _set_reward_coeffs_recursive(env, distance_coeff: float, entropy_coeff: float,
                                  tag_bonus_coeff: float, gain_coeff: float | None = None):
    """Walk the wrapper stack and set reward coefficients on PFRewardShapingWrapper."""
    e = env
    while e is not None:
        if isinstance(e, PFRewardShapingWrapper):
            e.set_reward_coeffs(distance_coeff, entropy_coeff, tag_bonus_coeff, gain_coeff)
            return
        e = getattr(e, "env", None)


class _CurriculumRouter(ScheduleRouter):
    """Thin outermost wrapper so SubprocVecEnv.env_method can reach inner wrappers.

    Since change 3 (2026-09-12) a rl/curriculum.ScheduleRouter over the three Ant-Tag
    setters (the targets of ANT_TAG_SCHEDULES): `set_curriculum_radius`,
    `set_reward_coeffs`, `set_evasion_scale` each reach the first wrapper below that has
    them. Kept under this name, constructed as before with the env only, for the factory,
    the eval / render scripts and the tests.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env, targets=tuple(s.target for s in ANT_TAG_SCHEDULES))


# ---------------------------------------------------------------------------
# PF interaction mapper for AntTag
# ---------------------------------------------------------------------------

def get_ant_tag_pf_kwargs(env) -> dict:
    """Build AntTagParticleFilter kwargs from the live AntTag environment."""
    unwrapped = env.unwrapped
    cage_max_x = float(unwrapped.cage_max_x)
    cage_max_y = float(unwrapped.cage_max_y)
    if not np.isclose(cage_max_x, cage_max_y):
        raise ValueError(
            "AntTagParticleFilter currently assumes a square arena, but "
            f"got cage_max_x={cage_max_x}, cage_max_y={cage_max_y}"
        )
    return {
        "arena_limits": (-cage_max_x, cage_max_x),
        "target_step": float(unwrapped.target_step),
        "visibility_radius": float(unwrapped.visible_radius),
        "min_initial_distance": float(unwrapped.min_distance),
        "tag_radius": float(unwrapped.tag_radius),
    }


def ant_tag_pf_interaction_mapper(
    base_env_obs: np.ndarray,
    base_env_info: dict,
    base_env_action: np.ndarray | None = None,
    unwrapped_env=None,
    previous_base_env_obs: np.ndarray | None = None,
) -> dict:
    """
    Bridge between AntTag observations and the AntTagParticleFilter
    predict / update interface.

    Visibility is determined by obs[-2:]: the CurriculumVisibilityWrapper
    puts the true target there when within the curriculum radius, or zeros
    when not visible. So we check for non-zero obs[-2:] rather than
    hardcoding a radius.
    """
    ant_pos = base_env_obs[:2].copy()
    ant_pos_for_prediction = (
        previous_base_env_obs[:2].copy()
        if previous_base_env_obs is not None
        else ant_pos
    )
    target_in_obs = base_env_obs[-2:].copy()

    # CurriculumVisibilityWrapper sets obs[-2:] to true target when visible,
    # zeros when not. A zero target at the origin is astronomically unlikely.
    visible = np.any(target_in_obs != 0.0)

    if visible:
        observed_target = target_in_obs
    else:
        observed_target = np.array([np.nan, np.nan])

    # getattr default 1.0 keeps this a no-op for envs without the knob
    # (base AntTagEnv, or SmartAntTagEnv with no evasion curriculum active).
    evasion_scale = float(getattr(unwrapped_env, "evasion_scale", 1.0))

    # Must track the env's live value: if the PF propagates particles at a
    # different target speed than the env actually moves the target, the
    # belief is systematically wrong. Default 0.0 matches SmartAntTagEnv's
    # own default (constant target_step, no cornered speed-up) and is a
    # no-op for the base AntTagEnv, which has no such knob.
    target_speed_scale = float(getattr(unwrapped_env, "target_speed_scale", 0.0))

    # getattr default falls back to the env's fixed radius for envs without
    # a curriculum wrapper (eval scripts, dataset collection), so the PF's
    # negative-info update always matches whatever radius actually decided
    # reveal/no-reveal this step, curriculum or not.
    current_visibility_radius = float(
        getattr(unwrapped_env, "current_visibility_radius", unwrapped_env.visible_radius)
    )

    # Ghost-ping passthrough (GhostAntTagEnv only). base_env_info carries the
    # ping; the env attributes carry the live sensor parameters (same pattern
    # as target_speed_scale). For envs without pings these are None/defaults
    # and the base PFs' **kwargs absorb them silently.
    # NOTE: info["ghost_ping_is_true"] is deliberately NOT forwarded — it is
    # ground truth for offline analysis only.
    ghost_ping = None
    if base_env_info:
        ghost_ping = base_env_info.get("ghost_ping", None)
    ping_beta = float(getattr(unwrapped_env, "ping_beta", 0.35))
    ping_sigma = float(getattr(unwrapped_env, "ping_sigma", 0.8))

    # Twin-den passthrough (TwinDenAntTagEnv only). The tight/loose
    # assignment and den geometry are motion-model parameters the PF is
    # entitled to know (same convention as evasion_scale). None-defaults for
    # envs without dens; other PFs' **kwargs absorb them silently.
    # NOTE: info["den_committed"] (which den the target actually chose) is
    # deliberately NOT forwarded — it is the hidden episode latent, ground
    # truth for offline analysis only.
    den_tight = getattr(unwrapped_env, "tight_den", None)
    den_positions = getattr(unwrapped_env, "den_positions", None)
    if den_positions is not None:
        den_positions = np.asarray(den_positions, dtype=np.float64).copy()
    den_radius_tight = getattr(unwrapped_env, "den_radius_tight", None)
    den_radius_loose = getattr(unwrapped_env, "den_radius_loose", None)

    # Counterweighted-den passthrough (CounterweightedDenAntTagEnv only). The
    # per-episode den geometry, the occupancy prior and the spook state are
    # motion- and observation-model parameters the PF is entitled to know
    # (same convention as evasion_scale). None-defaults for every other env;
    # other PFs' **kwargs absorb them silently.
    # NOTE: info["cden_occupied"] (WHICH den the target actually occupies) is
    # deliberately NOT forwarded — it is the hidden episode latent, ground
    # truth for offline analysis only. Same precedent as den_committed.
    cden_heavy_pos = getattr(unwrapped_env, "cden_heavy_pos", None)
    if cden_heavy_pos is not None:
        cden_heavy_pos = np.asarray(cden_heavy_pos, dtype=np.float64).copy()
    cden_light_pos = getattr(unwrapped_env, "cden_light_pos", None)
    if cden_light_pos is not None:
        cden_light_pos = np.asarray(cden_light_pos, dtype=np.float64).copy()
    cden_w_heavy = getattr(unwrapped_env, "cden_w_heavy", None)
    cden_r = getattr(unwrapped_env, "cden_r", None)
    cden_spook_enabled = bool(getattr(unwrapped_env, "cden_spook_enabled",
                                      False))
    cden_spooked = bool(getattr(unwrapped_env, "cden_spooked", False))
    cden_spook_pos = getattr(unwrapped_env, "cden_spook_pos", None)
    if cden_spook_pos is not None:
        cden_spook_pos = np.asarray(cden_spook_pos, dtype=np.float64).copy()
    cden_spook_radius = getattr(unwrapped_env, "cden_spook_radius", None)

    return {
        "predict_args": {
            "ant_current_pos_from_obs": ant_pos_for_prediction,
            "evasion_scale": evasion_scale,
            "target_speed_scale": target_speed_scale,
            "den_tight": den_tight,
            "den_positions": den_positions,
            "den_radius_tight": den_radius_tight,
            "den_radius_loose": den_radius_loose,
            "cden_heavy_pos": cden_heavy_pos,
            "cden_light_pos": cden_light_pos,
            "cden_w_heavy": cden_w_heavy,
            "cden_r": cden_r,
            "cden_spooked": cden_spooked,
        },
        "update_args": {
            "observed_target_pos": observed_target,
            "ant_current_pos_from_obs": ant_pos,
            "visibility_radius": current_visibility_radius,
            "ghost_ping": ghost_ping,
            "ping_beta": ping_beta,
            "ping_sigma": ping_sigma,
            "cden_heavy_pos": cden_heavy_pos,
            "cden_light_pos": cden_light_pos,
            "cden_r": cden_r,
            "cden_spook_enabled": cden_spook_enabled,
            "cden_spooked": cden_spooked,
            "cden_spook_pos": cden_spook_pos,
            "cden_spook_radius": cden_spook_radius,
        },
    }


# ---------------------------------------------------------------------------
# Env factory and helpers (moved from experiments/ant_tag/4_train_rl_cgf.py)
# ---------------------------------------------------------------------------


def get_ant_tag_arena_scale(env_id: str = "pdomains-ant-tag-v0") -> float:
    """Derive the CGF particle-normalization scale from the live AntTag arena.

    Mirrors get_ant_tag_pf_kwargs's arena_limits derivation, so the CGF
    extractor's particle normalization always matches the PF's actual
    arena_limits instead of relying on a hardcoded default that would go
    stale if the env's arena size ever changes.
    """
    env = gym.make(env_id, rendering=False)
    try:
        unwrapped = env.unwrapped
        cage_max_x = float(unwrapped.cage_max_x)
        cage_max_y = float(unwrapped.cage_max_y)
        if not np.isclose(cage_max_x, cage_max_y):
            raise ValueError(
                "WeightedCGFFeaturesExtractor currently assumes a square arena, "
                f"but got cage_max_x={cage_max_x}, cage_max_y={cage_max_y}"
            )
        return cage_max_x
    finally:
        env.close()


def get_env_visible_radius(env_id: str = "pdomains-ant-tag-v0") -> float:
    """Derive the evaluation visibility radius from the live env.

    Mirror of get_ant_tag_arena_scale. Replaces the hardcoded 3.0 that used
    to be baked into every eval-env construction: every legacy env reports
    visible_radius == 3.0, so this is behavior-identical for them, while
    arena-scaled variants (CounterweightedDenAntTagEnv, 1.8) are evaluated at
    THEIR real POMDP difficulty instead of a stale constant.
    """
    env = gym.make(env_id, rendering=False)
    try:
        return float(env.unwrapped.visible_radius)
    finally:
        env.close()


def _make_vec_normalize(vec_env, training: bool, norm_reward: bool):
    """Normalize only base obs so PF weights remain valid probabilities."""
    try:
        return VecNormalize(
            vec_env,
            training=training,
            norm_obs=True,
            norm_reward=norm_reward,
            norm_obs_keys=["obs"],
        )
    except TypeError:
        print(
            "Warning: this SB3 VecNormalize lacks norm_obs_keys; disabling "
            "obs normalization to avoid corrupting PF weights."
        )
        return VecNormalize(
            vec_env,
            training=training,
            norm_obs=False,
            norm_reward=norm_reward,
        )


def make_ant_tag_cgf_env(
    num_particles: int,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    distance_coeff: float = 1.0,
    entropy_coeff: float = 0.0,
    tag_bonus_coeff: float = 0.0,
    gain_coeff: float = 0.0,
    initial_visibility_radius: float = 100.0,
    obs_mask_indices: list[int] | None = None,
    apply_reward_shaping: bool = True,
    env_id: str = "pdomains-ant-tag-v0",
    particle_filter_class: type = AntTagParticleFilter,
    target_speed_scale: float | None = None,
):
    """Return a callable that creates a weighted-CGF AntTag env.

    apply_reward_shaping=False skips PFRewardShapingWrapper entirely, so
    Monitor sees the env's true sparse reward (-1/step, 0-and-terminate on
    tag). Use this for the eval env: CurriculumCallback only ever updates
    reward coefficients on the training env, so a shaped eval env would
    report reward numbers stuck at their initial (dense) coefficients for
    the entire run, making EvalCallback's "best_model" selection
    meaningless. Eval envs already fix visibility at the real POMDP radius
    regardless of training progress; this applies that same principle to
    the reward too.

    env_id / particle_filter_class default to the original dumb-target
    AntTag env + its matching PF; pass "pdomains-ant-tag-smart-v0" +
    SmartAntTagParticleFilter to train on the smart-target variant instead,
    so belief propagation matches that env's true motion model.

    target_speed_scale=None (default) leaves the env at its own default
    (0.0 for SmartAntTagEnv: the target flees more OFTEN when cornered but
    never faster). Pass a float to override; only SmartAntTagEnv has this
    knob, so passing it with the base AntTag env is a hard error rather
    than a silently ignored no-op. The PF is told the live value each step
    by ant_tag_pf_interaction_mapper, so belief propagation always matches
    whatever the env is actually doing.
    """

    def _init():
        env_make_kwargs = {"rendering": False}
        if target_speed_scale is not None:
            env_make_kwargs["target_speed_scale"] = target_speed_scale
        try:
            env = gym.make(env_id, **env_make_kwargs)
        except TypeError as exc:
            if target_speed_scale is None or "target_speed_scale" not in str(exc):
                raise
            raise ValueError(
                f"--target_speed_scale was given ({target_speed_scale}) but env_id="
                f"{env_id!r} does not support it. Only SmartAntTagEnv "
                "('pdomains-ant-tag-smart-v0') has a cornered-speed knob; the base "
                "AntTag target always moves at a constant target_step. Either drop "
                "the flag or train on the smart env."
            ) from exc
        if target_speed_scale is not None:
            # Fail loudly if it didn't land: a silently-dropped kwarg here
            # would train against a different target speed than requested,
            # and the mapper would faithfully feed that wrong value to the PF.
            actual = getattr(env.unwrapped, "target_speed_scale", None)
            if actual is None or not np.isclose(actual, target_speed_scale):
                raise ValueError(
                    f"target_speed_scale={target_speed_scale} did not take effect on "
                    f"env_id={env_id!r} (env reports {actual!r}). Only "
                    "SmartAntTagEnv ('pdomains-ant-tag-smart-v0') supports this knob."
                )
        env.reset(seed=seed + rank)
        particle_filter_kwargs = get_ant_tag_pf_kwargs(env)

        env = CurriculumVisibilityWrapper(
            env,
            initial_visibility_radius=initial_visibility_radius,
        )
        env = PFDictWithWeightsObservationWrapper(
            env=env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs=particle_filter_kwargs,
            num_particles=num_particles,
            pf_interaction_mapper=ant_tag_pf_interaction_mapper,
            obs_mask_indices=obs_mask_indices,
            particle_filter_seed=seed + rank,
        )
        if apply_reward_shaping:
            env = PFRewardShapingWrapper(
                env,
                distance_coeff=distance_coeff,
                entropy_coeff=entropy_coeff,
                tag_bonus_coeff=tag_bonus_coeff,
                gain_coeff=gain_coeff,
            )

        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        else:
            env = Monitor(env)
        env = _CurriculumRouter(env)
        return env

    return _init


def _make_vec_env_from_fns(env_fns, n_envs: int):
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


# ---------------------------------------------------------------------------
# The Domain description (change 4, 2026-09-12): what the shared trainer needs from Ant-Tag
# ---------------------------------------------------------------------------


#: The SCRIPT defaults for the two schedule strings, used when neither the CLI nor the
#: variant supplies one (precedence: CLI > variant > these). The strings every Ant-Tag
#: training script has carried since the registry was written.
DEFAULT_CURRICULUM = "0:100,0.3:100,0.7:3,1:3"
DEFAULT_REWARD_SCHEDULE = "0:1:0:0,0.3:1:0:0,0.7:0:0:50,1:0:0:50"


def _resolve_reward_shaping(parser, args) -> None:
    """Reconcile --distance_coeff/--entropy_coeff with --reward_schedule, in place.

    CurriculumCallback writes the schedule's interpolated coefficients into
    every env on every step, so a coefficient flag given alongside a schedule
    used to govern the first n_envs steps only, while run_config.json recorded
    it as live. Nobody noticed because the default schedule's first waypoint
    equals the flag defaults. Two outcomes now:

    * ``--reward_schedule none`` (or empty): the flags (defaults 1.0 / 0.0)
      become a constant schedule, so they really do hold for the whole run.
    * a schedule plus an explicit flag: ``parser.error``. The user asked for
      two things that cannot both happen.

    In both cases ``args.distance_coeff`` / ``args.entropy_coeff`` are set to
    the values in force at progress 0 and ``args.reward_schedule`` to a
    parseable string, so ``run_config.json`` records what actually ran.
    Shared by the Gaussian and ST arms.
    """
    explicit = [f"--{name}" for name in ("distance_coeff", "entropy_coeff")
                if getattr(args, name) is not None]
    schedule_str = (args.reward_schedule or "").strip()
    if schedule_str.lower() in ("", "none"):
        distance = 1.0 if args.distance_coeff is None else float(args.distance_coeff)
        entropy = 0.0 if args.entropy_coeff is None else float(args.entropy_coeff)
        args.reward_schedule = f"0:{distance}:{entropy}:0,1:{distance}:{entropy}:0"
    else:
        if explicit:
            parser.error(
                f"{' and '.join(explicit)} cannot be combined with "
                "--reward_schedule: the schedule sets the shaping coefficients "
                "on every step, so the flag would govern the first rollout "
                "only. Either drop the flag and put the values in the schedule "
                "(entries are frac:distance:entropy[:tag_bonus]) or pass "
                "--reward_schedule none to run on constant coefficients."
            )
        first = _parse_reward_schedule(schedule_str)[0]
        distance, entropy = float(first[1]), float(first[2])
    args.distance_coeff = distance
    args.entropy_coeff = entropy


def make_schedules(curriculum_schedule=None, reward_schedule=None, evasion_schedule=None):
    """The three Ant-Tag :class:`Schedule` values for one run, from parsed waypoint lists.

    ``None`` for any of them means ANT_TAG_SCHEDULES' default waypoints. Exactly what the
    :class:`CurriculumCallback` adapter builds from the same three arguments (waypoints sorted
    by fraction), stated once so the trainer and the adapter cannot drift.
    """
    by_name = {s.name: s for s in ANT_TAG_SCHEDULES}
    picked = []
    for name, given in (("visibility", curriculum_schedule), ("reward", reward_schedule),
                        ("evasion", evasion_schedule)):
        waypoints = list(by_name[name].waypoints) if given is None else list(given)
        picked.append(replace(by_name[name], waypoints=tuple(sorted(waypoints, key=lambda x: x[0]))))
    return tuple(picked)
# ---------------------------------------------------------------------------
# Dataset collection (moved from experiments/ant_tag/2_collect_pf_dataset.py, batch 7.4,
# 2026-09-13; that script is now an entry point of rl/collect.py and re-exports these names).
# The shared loop lives in rl/collect.py; what is here is what only Ant-Tag knows: the
# trajectory mix (fully observed / pursuit with the locomotion policy / random), the
# visibility radius drawn per trajectory through the RL curriculum knob, and the rebalancing
# by weighted spread in arena units. Bodies unchanged; the registry calls that were
# `variants.<fn>` in the script are this module's own functions.
# ---------------------------------------------------------------------------

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

    candidate = getattr(_ant_tag_filters, name, None)
    if not _is_concrete_filter(candidate):
        available = sorted(
            attr for attr in dir(_ant_tag_filters)
            if _is_concrete_filter(getattr(_ant_tag_filters, attr))
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
    locomotion_policy: "PPO",
    vecnorm_stats=None,
) -> np.ndarray:
    """Action from the locomotion policy, aimed at the PF's belief mean.

    The locomotion policy was trained fully observed and expects obs[-2:] to
    hold the target position, so the belief mean is substituted there. This is
    what makes the collected beliefs cover states an actual pursuer reaches,
    instead of only those a random walk stumbles into.
    """
    return _goal_action(base_obs, particle_filter.estimate_opponent_pos(), locomotion_policy, vecnorm_stats)


def _goal_action(base_obs: np.ndarray, goal_xy, locomotion_policy: "PPO", vecnorm_stats=None) -> np.ndarray:
    """Action from the locomotion policy aimed at ``goal_xy`` (the body of :func:`_pursuit_action`,
    moved in the waypoint batch of 2026-09-14: the belief mean and a sampled waypoint are two goals)."""
    policy_obs = base_obs.copy()
    policy_obs[-2:] = np.asarray(goal_xy, dtype=policy_obs.dtype)
    if vecnorm_stats is not None:
        policy_obs = _normalize_obs(policy_obs, vecnorm_stats)
    action, _ = locomotion_policy.predict(policy_obs, deterministic=False)
    return action


def sample_waypoint(rng: np.random.Generator, ant_xy, target_xy, *, half_width: float, min_dist: float,
                    target_clearance: float, tries: int = 50) -> np.ndarray:
    """A point of the cage ``[-half_width, half_width]^2`` at least ``min_dist`` from the ant and at least
    ``target_clearance`` from the TRUE target (so the walk stays blind and sweeps). Up to ``tries`` uniform
    draws; if none satisfies both, the draw farthest from the ant among those clear of the target, else
    the farthest from the ant. The target is a collection-time heuristic, never a label or an input."""
    ant = np.asarray(ant_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    best, best_d = None, -1.0
    for _ in range(int(tries)):
        cand = rng.uniform(-half_width, half_width, size=2)
        d_ant = float(np.linalg.norm(cand - ant))
        clear = float(np.linalg.norm(cand - target)) >= target_clearance
        if d_ant >= min_dist and clear:
            return cand
        score = d_ant + (half_width * 10.0 if clear else 0.0)
        if score > best_d:
            best, best_d = cand, score
    return best


class _WaypointWalk:
    """The ``waypoint`` episode's driver: the locomotion policy walks to a sampled point; on arrival
    (within ``reach``) or after ``patience`` steps without arriving (stuck on a wall) the next point is
    drawn. ``n_sampled`` counts the waypoints of the episode."""

    def __init__(self, args, state, env, half_width: float):
        self.args, self.state, self.env, self.half_width = args, state, env, half_width
        self.goal = None
        self.steps_on_goal = 0
        self.n_sampled = 0

    def _resample(self, ant_xy):
        u = self.env.unwrapped
        self.goal = sample_waypoint(self.state.rng, ant_xy, u.get_target_pos(), half_width=self.half_width,
                                    min_dist=self.args.waypoint_min_dist, target_clearance=self.args.waypoint_target_clearance)
        self.steps_on_goal = 0
        self.n_sampled += 1
        self.state.waypoints_sampled += 1

    def __call__(self, obs):
        ant_xy = np.asarray(obs["obs"][:2], dtype=np.float64)
        if (self.goal is None or np.linalg.norm(ant_xy - self.goal) <= self.args.waypoint_reach
                or self.steps_on_goal >= self.args.waypoint_patience):
            self._resample(ant_xy)
        self.steps_on_goal += 1
        return _goal_action(obs["obs"], self.goal, self.state.locomotion_policy, self.state.vecnorm_stats)


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


def _rebalance_index_by_spread(
    particles: np.ndarray,
    weights: np.ndarray,
    collapsed_frac: float = 0.30,
    intermediate_frac: float = 0.40,
    diffuse_frac: float = 0.30,
    collapsed_threshold: float = 0.5,
    diffuse_threshold: float = 4.0,
    seed: int = 42,
    upsample: bool = True,
) -> np.ndarray:
    """The rows :func:`_rebalance_by_spread` keeps, in output order (its body, moved here in
    batch 10.9 so the collector can apply the same selection to the per-row labels).

    Rebalance so intermediate-spread beliefs are well represented.

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
    return all_idx


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
    """Rebalance by belief spread (see :func:`_rebalance_index_by_spread`); the historical
    two-array form the entry point and its tests read."""
    all_idx = _rebalance_index_by_spread(
        particles, weights, collapsed_frac=collapsed_frac, intermediate_frac=intermediate_frac,
        diffuse_frac=diffuse_frac, collapsed_threshold=collapsed_threshold,
        diffuse_threshold=diffuse_threshold, seed=seed, upsample=upsample)
    particles, weights = particles[all_idx], weights[all_idx]
    spreads_out = _weighted_spread(particles, weights)
    print(f"Post-rebalance: {len(particles)} samples — "
          f"collapsed={int((spreads_out < collapsed_threshold).sum())}, "
          f"intermediate={int(((spreads_out >= collapsed_threshold) & (spreads_out < diffuse_threshold)).sum())}, "
          f"diffuse={int((spreads_out >= diffuse_threshold).sum())}")
    return particles, weights


# -- the Collection hooks (rl/domains/base.py::Collection) -----------------------------------

def _collect_add_arguments(parser) -> None:
    """Ant-Tag's own collection flags (help texts from the script); the shared ones
    (--num_trajectories, --timesteps, --num_particles, --seed, --max_snapshots, --no_rebalance,
    --output_file) are rl/collect.py's."""
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
    w = parser.add_argument_group(
        "waypoint episodes (2026-09-14): the locomotion policy walks to sampled points while BLIND at the "
        "env's real visibility radius, so the file holds the sweeping beliefs a searching policy lives in "
        "(v15 record 2026-09-08: 76-95 %% of a policy's steps are hollow perimeter frames, 19 %% of the "
        "collector's file). Needs --locomotion_policy_path; without it these episodes are random.")
    w.add_argument("--waypoint_fraction", type=float, default=0.0,
                   help="Fraction of episodes of this type (drawn after fully-observed and pursuit; the rest "
                        "are random). 0 = the historical three-type mix.")
    w.add_argument("--waypoint_min_dist", type=float, default=None,
                   help="A new waypoint is at least this far from the ant. ENV UNITS; default half the cage half-width.")
    w.add_argument("--waypoint_target_clearance", type=float, default=None,
                   help="... and at least this far from the TRUE target (collection-time heuristic only). "
                        "ENV UNITS; default the env's visible radius.")
    w.add_argument("--waypoint_reach", type=float, default=0.75,
                   help="The ant has arrived when within this distance; the next waypoint is drawn. ENV UNITS.")
    w.add_argument("--waypoint_patience", type=int, default=80,
                   help="Steps without arriving before the next waypoint is drawn anyway (a wall).")
    w.add_argument("--waypoint_radius", choices=("real", "range"), default="real",
                   help="Visibility radius of a waypoint episode: the env's real one (default) or drawn from "
                        "the --visibility_radius_min/max range like the other types.")
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


def _collect_resolve_arguments(parser, args, domain) -> dict:
    """The env id and filter (overrides included, with the registry's warning), and the
    VecNormalize stats found beside the locomotion policy."""
    variant = resolve(args.variant)
    resolved_env_id = args.env_id or variant.env_id
    resolved_pf = (resolve_particle_filter(args.particle_filter)
                   if args.particle_filter else variant.particle_filter)
    warn_if_override_contradicts(
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
    facts = _env_facts(resolved_env_id)
    if not 0.0 <= args.waypoint_fraction <= 1.0:
        parser.error(f"--waypoint_fraction {args.waypoint_fraction} is not in [0, 1]")
    if args.fully_observed_fraction + args.pursuit_fraction + args.waypoint_fraction > 1.0 + 1e-9:
        parser.error("--fully_observed_fraction + --pursuit_fraction + --waypoint_fraction exceeds 1")
    if args.waypoint_min_dist is None:
        args.waypoint_min_dist = 0.5 * float(get_ant_tag_arena_scale(resolved_env_id))
    if args.waypoint_target_clearance is None:
        args.waypoint_target_clearance = float(facts["visible_radius"])
    if args.waypoint_fraction > 0:
        print(f"Waypoint episodes: {args.waypoint_fraction:.0%}, min dist {args.waypoint_min_dist:g}, target "
              f"clearance {args.waypoint_target_clearance:g}, reach {args.waypoint_reach:g}, patience "
              f"{args.waypoint_patience}, radius {args.waypoint_radius}")
    return {"env_id": resolved_env_id, "particle_filter_class": resolved_pf, "env_facts": facts}


def _env_facts(env_id: str) -> dict:
    """The env constants the labels of :func:`_collect_snapshot_extras` are read against, written
    into the dataset's metadata (batch 10.9): the tag and visible radii, and on the
    counterweighted-den envs the den radius (``cden_r``, one constant for both dens)."""
    env = gym.make(env_id, rendering=False)
    try:
        u = env.unwrapped
        facts = {"tag_radius": float(u.tag_radius), "visible_radius": float(u.visible_radius)}
        if hasattr(u, "cden_r"):
            facts["den_radius"] = float(u.cden_r)
            # the four STATIC candidate centres, the env's (and the filter's) order [-h, +h, -f, +f]
            facts["den_candidates"] = np.asarray(u.cden_candidates, dtype=float).round(6).tolist()
        return facts
    finally:
        env.close()


def _collect_prepare(args, options):
    """The trajectory RNG (its own Generator, untouched by PPO.load's global re-seeding), the
    locomotion policy and its VecNormalize stats, the trajectory-type counts."""
    state = SimpleNamespace(rng=np.random.default_rng(args.seed), locomotion_policy=None,
                            vecnorm_stats=None,
                            counts={"fully_observed": 0, "pursuit": 0, "waypoint": 0, "random": 0},
                            waypoints_sampled=0)
    if args.locomotion_policy_path and os.path.exists(args.locomotion_policy_path):
        from stable_baselines3 import PPO   # imported here: the domain module is env-side code
        state.locomotion_policy = PPO.load(args.locomotion_policy_path)
        print(f"Loaded locomotion policy from {args.locomotion_policy_path}")
        if args.locomotion_vecnorm_path and os.path.exists(args.locomotion_vecnorm_path):
            import pickle
            with open(args.locomotion_vecnorm_path, "rb") as handle:
                state.vecnorm_stats = pickle.load(handle)
            print(f"Loaded VecNormalize stats from {args.locomotion_vecnorm_path}")
    elif args.pursuit_fraction > 0 or args.fully_observed_fraction > 0 or args.waypoint_fraction > 0:
        print("WARNING: no locomotion policy provided; pursuit, fully-observed and "
              "waypoint trajectories fall back to random actions.")
    return state


def _collect_make_env(args, options, state):
    # The same factory the RL scripts use: same PF, same interaction mapper,
    # same visibility wrapper. Reward shaping off — nothing here reads reward.
    env = make_ant_tag_cgf_env(
        num_particles=args.num_particles,
        rank=0,
        seed=args.seed,
        monitor_dir=None,
        initial_visibility_radius=args.visibility_radius_max,
        obs_mask_indices=None,   # the pursuit policy needs the real base obs
        apply_reward_shaping=False,
        env_id=options["env_id"],
        particle_filter_class=options["particle_filter_class"],
        target_speed_scale=args.target_speed_scale,
    )()
    env.set_evasion_scale(args.evasion_scale)
    return env


def _collect_begin_episode(args, options, state, env, episode):
    """Draw the trajectory type and the visibility radius, set the radius through the SAME
    curriculum knob RL uses (so a collected belief is always one the RL agent could hold),
    and return how to act for this trajectory."""
    rng = state.rng
    roll = rng.random()
    has_policy = state.locomotion_policy is not None
    if roll < args.fully_observed_fraction:
        traj_type = "fully_observed"
    elif has_policy and roll < args.fully_observed_fraction + args.pursuit_fraction:
        traj_type = "pursuit"
    elif has_policy and roll < args.fully_observed_fraction + args.pursuit_fraction + args.waypoint_fraction:
        traj_type = "waypoint"          # 2026-09-14: blind sweeps between sampled points
    else:
        traj_type = "random"
    state.counts[traj_type] += 1

    if traj_type == "fully_observed":
        radius = _ALWAYS_VISIBLE_RADIUS
    elif traj_type == "waypoint" and args.waypoint_radius == "real":
        radius = float(env.unwrapped.visible_radius)
    else:
        radius = float(rng.uniform(args.visibility_radius_min, args.visibility_radius_max))
    env.set_curriculum_radius(radius)

    if traj_type == "waypoint":
        # The walk needs the true target for its clearance rule; it reads it off the live env at each
        # draw (the env has been reset by the time act() is first called).
        return {}, _WaypointWalk(args, state, env, half_width=float(get_ant_tag_arena_scale(options["env_id"])))

    def act(obs):
        if traj_type in ("pursuit", "fully_observed") and has_policy:
            return _pursuit_action(
                obs["obs"], _find_particle_filter(env),
                state.locomotion_policy, state.vecnorm_stats,
            )
        return env.action_space.sample()

    return {}, act


#: The per-row label arrays every Ant-Tag dataset carries since batch 10.9 (RAW env units, like
#: the particles): the ant position, the true target position, the step index.
TASK_LABELS = ("ant", "target", "step")
#: The two more the counterweighted-den envs carry: the heavy and light den centres of the row's
#: episode (they flip sides at every reset; motion-model parameters the filter receives, not a
#: privileged label) and the index, in the metadata's static ``den_candidates`` order
#: ``[-h, +h, -f, +f]``, of the candidate disc that holds the target (REPORT ONLY).
DEN_LABELS = ("den_positions", "den_occupied")


def _collect_snapshot_extras(args, options, state, env, obs, step_index) -> dict:
    """The labels of one snapshot, read off the env's hidden state as the belief is recorded."""
    u = env.unwrapped
    out = dict(ant=np.asarray(u.data.qpos[:2], np.float32),
               target=np.asarray(u.get_target_pos(), np.float32),
               step=np.int32(step_index))
    if "den_radius" in options["env_facts"]:
        out["den_positions"] = np.stack([u.cden_heavy_pos, u.cden_light_pos]).astype(np.float32)
        occupied = u.cden_heavy_pos if u._occupied_is_heavy else u.cden_light_pos
        candidates = np.asarray(options["env_facts"]["den_candidates"])
        out["den_occupied"] = np.int64(np.argmin(np.linalg.norm(candidates - occupied, axis=1)))
    return out


def _collect_metadata_extras(args, options, particles, weights, steps) -> dict:
    facts = options["env_facts"]
    labels = TASK_LABELS + (DEN_LABELS if "den_radius" in facts else ())
    heads = VARIANTS[args.variant].task_heads if args.variant in VARIANTS else ()
    return dict(label_arrays=list(labels), task_heads=list(heads), **facts,
                episode_counts=options.get("episode_counts", {}), waypoints_sampled=options.get("waypoints_sampled", 0))


def _collect_finish(args, options, state, n_snapshots) -> None:
    print(f"Trajectories — " + ", ".join(f"{k}: {v}" for k, v in state.counts.items())
          + (f"; waypoints sampled: {state.waypoints_sampled}" if state.counts.get("waypoint") else ""))
    print(f"Total snapshots (raw): {n_snapshots}")
    options["episode_counts"] = dict(state.counts)          # for the metadata (the hook has no state)
    options["waypoints_sampled"] = int(state.waypoints_sampled)


def _collect_report(args, options, particles, weights, steps, stage) -> None:
    if stage != "final":
        return
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
    # alongside them so pretraining can train the encoder on exactly the
    # inputs it will be handed at RL time. Without this the encoder would be
    # pretrained on coordinates several times larger than the ones it later
    # sees, and the pretrained weights would be near-useless.
    print(f"  arena scale (recorded for pretraining): {_collect_particle_scale(args, options)}")


def _collect_particle_scale(args, options) -> float:
    return float(get_ant_tag_arena_scale(options["env_id"]))


def _collect_rebalance(args, options, particles, weights, steps):
    """The four-value form (batch 10.5 contract): the kept rows come back too, so the collector
    applies the same selection to the per-row labels of :func:`_collect_snapshot_extras`."""
    kept = _rebalance_index_by_spread(
        particles, weights,
        collapsed_threshold=args.collapsed_threshold,
        diffuse_threshold=args.diffuse_threshold,
        seed=args.seed,
        upsample=not args.rebalance_no_upsample,
    )
    particles, weights = particles[kept], weights[kept]
    spreads_out = _weighted_spread(particles, weights)
    print(f"Post-rebalance: {len(particles)} samples — "
          f"collapsed={int((spreads_out < args.collapsed_threshold).sum())}, "
          f"intermediate={int(((spreads_out >= args.collapsed_threshold) & (spreads_out < args.diffuse_threshold)).sum())}, "
          f"diffuse={int((spreads_out >= args.diffuse_threshold).sum())}")
    return particles, weights, None, kept      # the step index is not kept on this domain


ANT_TAG_COLLECTION = Collection(
    add_arguments=_collect_add_arguments,
    # The script's defaults for the shared flags.
    defaults={"seed": 42, "num_particles": 100, "timesteps": 200, "num_episodes": 200},
    resolve_arguments=_collect_resolve_arguments,
    particle_scale=_collect_particle_scale,
    prepare=_collect_prepare,
    make_env=_collect_make_env,
    begin_episode=_collect_begin_episode,
    finish=_collect_finish,
    report=_collect_report,
    rebalance=_collect_rebalance,
    snapshot_extras=_collect_snapshot_extras,
    metadata_extras=_collect_metadata_extras,
    progress_desc="Collecting trajectories",
)


# ---------------------------------------------------------------------------
# The ``task`` pretraining objective (batch 10.9, 2026-09-14; design in refactor_plans.md 10.9)
# ---------------------------------------------------------------------------

#: head name -> output size. ``position``: the true target position in the scaled frame
#: (label / arena half-width, the frame the encoder sees), squared error; the report multiplies
#: back to env units. ``den_mass``: logits over the FOUR static candidate discs ``[-h, +h, -f, +f]``
#: plus outside (five-way; decision 10c-15, user 2026-09-14: the step-0 belief is the filter's
#: four-candidate prior and a partition on the episode's two actual dens would file its phantom
#: mass under "outside"), cross-entropy against the weighted particle shares of the row's own
#: cloud (:func:`den_shares`); selection on the held-out loss, report = KL plus the argmax agreements.
TASK_HEADS = {"position": 2, "den_mass": 5}
_TASK_COLLECT_HINT = ("collect it with python -m set_transformer.rl.collect --domain ant_tag --variant <v>; "
                      "the labels exist since 2026-09-14 (batch 10.9), the recorded files under "
                      "experiments/ant_tag/data carry none")


def den_shares(particles: np.ndarray, weights: np.ndarray, dens: np.ndarray, radius: float) -> np.ndarray:
    """``[S, K + 1]`` weighted particle mass inside each of the K discs, then outside all of them,
    per row. ``particles [S, N, 2]`` and ``dens [K, 2]`` (static, the four candidates) or
    ``[S, K, 2]`` (per row) in the same (raw) frame; weights are normalised per row first. The
    candidate discs never overlap (2.4 and 6.75 out on the diagonal, radius 0.4), so the shares sum
    to one. The rim counts as inside with a small tolerance: the env and the filter project the
    target and the particles back ONTO the disc boundary at every step (a 0.5 step against a 0.4
    radius), so a strict ``<= radius`` test dropped a third of the mass to floating point
    (found 2026-09-14 on the waypoint-collector measurement)."""
    w = np.clip(np.asarray(weights, np.float64), 0.0, None)
    w = w / np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
    dens = np.asarray(dens, np.float64)
    if dens.ndim == 2:
        dens = np.broadcast_to(dens[None], (len(w),) + dens.shape)
    d = np.linalg.norm(np.asarray(particles, np.float64)[:, :, None, :]
                       - dens[:, None, :, :], axis=-1)                                 # [S, N, K]
    inside = (w[:, :, None] * (d <= float(radius) * (1.0 + 1e-6) + 1e-9)).sum(axis=1)  # [S, K]; rim inside
    outside = np.clip(1.0 - inside.sum(axis=1, keepdims=True), 0.0, 1.0)
    return np.concatenate([inside, outside], axis=1).astype(np.float32)


class AntTagTaskData(_TaskData):
    """The labelled Ant-Tag dataset: the ant position is the ``obs`` passthrough; the den
    arrays are required when the variant's heads include ``den_mass``."""

    def __init__(self, path: str, val_frac: float, device: torch.device, *, heads: tuple[str, ...]):
        labels = TASK_LABELS + (DEN_LABELS if "den_mass" in heads else ())
        super().__init__(path, val_frac, device, labels=labels, obs_key="ant", scale_default=float("nan"),
                         collect_hint=_TASK_COLLECT_HINT, kind="labelled Ant-Tag dataset")


def _head_slices(heads: tuple[str, ...]) -> list[tuple[str, slice]]:
    out, offset = [], 0
    for h in heads:
        out.append((h, slice(offset, offset + TASK_HEADS[h])))
        offset += TASK_HEADS[h]
    return out


def task_loss(y: torch.Tensor, batch: dict, heads: tuple[str, ...]) -> torch.Tensor:
    total = y.new_zeros(())
    for head, sl in _head_slices(heads):
        if head == "position":
            total = total + F.mse_loss(y[:, sl], batch["target_scaled"])
        else:
            total = total - (batch["den_shares"] * F.log_softmax(y[:, sl], dim=-1)).sum(-1).mean()
    return total


@torch.no_grad()
def task_metrics(y: torch.Tensor, batch: dict, heads: tuple[str, ...], scale: float, tag_radius: float) -> dict:
    """In env units: mean position error and the fraction of rows predicted within the tag
    radius; for the den head the held-out KL(shares || predicted), the argmax agreement with the
    shares (does the encoder read its own belief) and with the true candidate (does the belief
    track the truth; the truth is never the "outside" class)."""
    out = {}
    for head, sl in _head_slices(heads):
        z = y[:, sl]
        if head == "position":
            err = scale * torch.linalg.norm(z - batch["target_scaled"], dim=-1)
            out["position_mae"] = float(err.mean())
            out["within_tag_radius"] = float((err < tag_radius).float().mean())
        else:
            p = F.softmax(z, dim=-1)
            q = batch["den_shares"]
            kl = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(p.clamp_min(1e-12)))).sum(-1)
            out["den_kl"] = float(kl.mean())
            out["den_acc_belief"] = float((p.argmax(-1) == q.argmax(-1)).float().mean())
            out["den_acc_truth"] = float((p.argmax(-1) == batch["den_occupied"]).float().mean())
    return out


def _task_add_arguments(parser, domain=None) -> None:
    add_task_arguments(parser, dataset_help="The labelled Ant-Tag dataset (.npz from the harness collector: "
                                            "ant / target / step [/ den_positions / den_occupied] arrays).")


def _task_resolve_arguments(parser, args, domain, encoder) -> None:
    resolve_task_arguments(parser, args, domain, encoder, collect_hint=_TASK_COLLECT_HINT)


def _task_prepare(parser, args, domain, encoder, device):
    recorded = locate_from_dataset(args)["variant"]
    if recorded is not None and args.variant is not None and recorded != args.variant:
        parser.error(f"--variant {args.variant} but the dataset was collected on {recorded!r}")
    variant_name = args.variant or recorded
    if variant_name is None:
        parser.error("the dataset records no variant; pass --variant")
    heads = resolve(variant_name).task_heads
    if not heads:
        parser.error(f"variant {variant_name!r} declares no task heads (Variant.task_heads); the task "
                     f"objective is defined for: " + ", ".join(n for n, v in VARIANTS.items() if v.task_heads))
    try:
        data = AntTagTaskData(args.data_path, args.val_frac, torch.device(device), heads=heads)
    except ValueError as exc:            # missing label arrays, an unsplittable file: a command-line refusal
        parser.error(str(exc))
    if "particle_scale" not in data.metadata:
        parser.error(f"{args.data_path} records no particle_scale in its metadata ({_TASK_COLLECT_HINT})")
    check_variant_and_geometry(parser, args, data)
    scale = data.particle_scale
    facts = data.metadata
    if "tag_radius" not in facts:
        facts = {**facts, **_env_facts(resolve(args.variant).env_id)}
    for split in (data.train, data.val):
        split["target_scaled"] = split["target"] / scale
        if "den_mass" in heads:
            if "den_radius" not in facts or "den_candidates" not in facts:
                parser.error(f"{args.data_path} records no den_radius / den_candidates, so the den shares "
                             f"cannot be computed ({_TASK_COLLECT_HINT})")
            shares = den_shares(split["particles"].cpu().numpy(), split["weights"].cpu().numpy(),
                                np.asarray(facts["den_candidates"]), facts["den_radius"])
            split["den_shares"] = torch.as_tensor(shares, device=data.device)
    args.task_heads = list(heads)
    args.tag_radius = float(facts["tag_radius"])
    return data


def _task_run(args, ctx: PretrainContext) -> PretrainResult:
    heads = tuple(args.task_heads)
    data = ctx.data
    return run_task_training(
        args, ctx, data=data, obs_dim=2, out_dim=sum(TASK_HEADS[h] for h in heads),
        loss_fn=lambda y, batch: task_loss(y, batch, heads),
        metrics_fn=lambda y, batch: task_metrics(y, batch, heads, data.particle_scale, args.tag_radius),
        checkpoint_config={"objective": "task", "task": "+".join(heads), "task_heads": list(heads),
                           "variant": args.variant,
                           "pretraining": "set_transformer.rl.pretrain --domain ant_tag --objective task"},
        label=f"{args.variant}/task/{ctx.encoder.name}")


ANT_TAG_TASK_OBJECTIVE = Objective(
    name="task",
    description="encoder + 3-layer head -> the variant's task heads (Variant.task_heads): `position`, "
                "the true target position in the scaled frame (MSE; the smart family), or `den_mass`, "
                "the weighted particle shares in the four candidate dens [-h, +h, -f, +f] / outside (soft "
                "cross-entropy; the counterweighted-den family). Reads the labelled dataset the harness "
                "collector writes.",
    add_arguments=_task_add_arguments,
    run=_task_run,
    run_name=task_run_name,
    locate=locate_from_dataset,
    resolve_arguments=_task_resolve_arguments,
    prepare=_task_prepare,
    default_experiment_name=lambda encoder_name: f"{encoder_name}_task_pretrain",
)




def _add_arguments(parser) -> None:
    """The flags that belong to the Ant-Tag problem (help texts from 4_train_rl_cgf.py)."""
    parser.add_argument(
        "--distance_coeff", type=float, default=None,
        help="Constant PF-mean-distance shaping coefficient (default 1.0). "
             "Only honoured with --reward_schedule none: the schedule sets "
             "these coefficients on every step, so combining the two is an "
             "error rather than a silent override.")
    parser.add_argument(
        "--entropy_coeff", type=float, default=None,
        help="Constant PF belief-entropy shaping coefficient (default 0.0). "
             "NOT PPO's entropy bonus. Same rule as --distance_coeff.")
    parser.add_argument(
        "--curriculum", type=str, default=None,
        help="Visibility curriculum 'frac:radius,...'. Defaults to the "
             "variant's own curriculum, else the base schedule "
             f"'{DEFAULT_CURRICULUM}'.")
    parser.add_argument(
        "--reward_schedule", type=str, default=None,
        help="Defaults to the variant's own recipe (Variant.default_reward_schedule), else "
             f"'{DEFAULT_REWARD_SCHEDULE}' (PF-entropy 0 throughout). "
             "Shaping schedule 'frac:distance:entropy[:tag_bonus[:spread_gain]],...' "
             "(spread_gain pays gain*(spread_{t-1}-spread_t) of the belief's weighted std; "
             "0 when omitted), interpolated over training progress and applied on every step. "
             "Pass 'none' to run on the constant --distance_coeff / --entropy_coeff values "
             "instead; giving both is an error.")
    parser.add_argument(
        "--evasion_curriculum", type=str, default=None,
        help="Optional frac:scale schedule for SmartAntTagEnv.evasion_scale "
             "(0=dumb-target behavior, 1=full smart evasion). E.g. "
             "'0:0.2,0.5:1,1:1' ramps evasion strength up over the first "
             "half of training. Default: the variant's, else constant 1.0 (full strength "
             "throughout, i.e. no curriculum). No-op on envs without an "
             "evasion_scale knob (e.g. the base AntTag env).")
    parser.add_argument(
        "--mask_target_obs", action="store_true", default=True,
        help="Zero out obs[-2:] for the agent. Default on.")
    parser.add_argument("--no_mask_target_obs", dest="mask_target_obs", action="store_false")
    parser.add_argument(
        "--target_speed_scale", type=float, default=None,
        help="SmartAntTagEnv only. How much faster the target moves as it gets "
             "cornered: step = target_step * (1 + urgency * scale). Omit to use "
             "the env default (0.0 = constant speed; the target flees more often "
             "when cornered but never faster). 1.0 restores the old up-to-2x "
             "behavior. Applied to both the training and eval envs.")


def _resolve_arguments(parser, args) -> dict:
    """Fill the schedule strings (CLI > variant > script default), reconcile the shaping flags,
    warn about evasion knobs on a non-evading env, and return the env factory's options.

    Same order as the scripts' ``main``: the reward schedule is resolved BEFORE
    ``_resolve_reward_shaping`` (which reads an empty schedule as "use the constant flags").
    """
    args.reward_schedule = resolve_schedule(
        args.variant, args.reward_schedule, "default_reward_schedule", DEFAULT_REWARD_SCHEDULE)
    _resolve_reward_shaping(parser, args)
    args.curriculum = resolve_schedule(
        args.variant, args.curriculum, "default_curriculum", DEFAULT_CURRICULUM)
    warn_if_not_evading(args.variant, args.evasion_curriculum, args.target_speed_scale)
    args.evasion_curriculum = resolve_schedule(
        args.variant, args.evasion_curriculum, "default_evasion_curriculum", None)
    curriculum_schedule = parse_curriculum(args.curriculum)
    return dict(
        distance_coeff=args.distance_coeff,
        entropy_coeff=args.entropy_coeff,
        tag_bonus_coeff=0.0,
        # The FIRST entry of the string as written, not the smallest fraction: what every
        # script did (`curriculum_schedule[0][1]`).
        initial_visibility_radius=curriculum_schedule[0][1] if curriculum_schedule else 100.0,
        obs_mask_indices=[-2, -1] if args.mask_target_obs else None,
        target_speed_scale=args.target_speed_scale,
    )


def _schedules(args):
    return make_schedules(
        parse_curriculum(args.curriculum),
        parse_reward_schedule(args.reward_schedule),
        parse_curriculum(args.evasion_curriculum) if args.evasion_curriculum else None,
    )


def _make_env(variant: str, *, num_particles: int, particle_filter_class: type, seed: int,
              rank: int, monitor_dir: str | None, training: bool, options: dict):
    """One worker's env through `make_ant_tag_cgf_env`. The eval env (``training=False``)
    runs at the env's real visibility radius and without reward shaping, exactly as the
    scripts built it (``eval_env_kw`` in every ``train_ant_tag_*``)."""
    env_id = resolve(variant).env_id
    options = dict(options)
    if not training:
        options["initial_visibility_radius"] = get_env_visible_radius(env_id)
        options["apply_reward_shaping"] = False
    return make_ant_tag_cgf_env(
        num_particles=num_particles, rank=rank, seed=seed, monitor_dir=monitor_dir,
        env_id=env_id, particle_filter_class=particle_filter_class, **options)


def make_eval_env(num_particles: int, obs_mask_indices, seed: int,
                  env_id: str = "pdomains-ant-tag-v0",
                  particle_filter_class: type = AntTagParticleFilter):
    """The EVAL env with the signature ``eval_scripts/eval_true_reward_cgf.py`` gave it
    (the diagnostics build their env through it): the env's real visibility radius, no
    reward shaping, Monitor on the env's own reward. Same stack as
    ``_make_env(training=False)``, which the shared evaluation script uses."""
    return make_ant_tag_cgf_env(
        num_particles=num_particles, seed=seed, env_id=env_id,
        particle_filter_class=particle_filter_class,
        initial_visibility_radius=get_env_visible_radius(env_id),
        obs_mask_indices=obs_mask_indices, apply_reward_shaping=False)


def _eval_add_arguments(parser) -> None:
    parser.add_argument("--no_mask", action="store_true",
                        help="Do not zero obs[-2:] (the target position) for the agent. "
                             "Must match how the checkpoint was trained (masked by default).")


def _eval_options(args) -> dict:
    """The env-factory options of the eval env: only the target mask (the shaping
    coefficients are irrelevant with shaping off, and the visibility radius is the env's)."""
    return dict(obs_mask_indices=None if args.no_mask else [-2, -1])


def _cgf_t_init_max_default(args):
    """Ant-Tag's rule for a ``--t_init_max`` left unset (4_train_rl_cgf.resolve_cgf_encoder_args):
    clamp mode keeps None (the extractor's legacy 2.8 ceiling, recorded as null as every old
    run did); tanh / polar with the ``spread`` init take 0.8 * t_bound; otherwise None."""
    if args.t_param == "clamp":
        return None
    if args.t_init_mode == "spread":
        return 0.8 * float(args.t_bound)
    return None


def _cgf_t_bound_default(variant: str) -> float:
    """Ant-Tag's default ``--t_bound`` for tanh / polar: the registry's ``cgf_t_bound``."""
    bound = cgf_t_bound(variant)
    print(f"CGF t_bound from the registry: {bound:.3g} "
          f"(= {CGF_TILT_TARGET} / (tag_radius / arena half-width) "
          f"for variant {variant!r})")
    return bound


ANT_TAG = Domain(
    name="ant_tag",
    particle_dim=2,
    default_variant="base",
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
    run_config_extras=lambda args: {},
    default_num_particles=lambda variant: 100,
    default_arena_scale=lambda variant: get_ant_tag_arena_scale(resolve(variant).env_id),
    default_device="cuda:1",
    default_total_timesteps=3_000_000,
    encoder_defaults={
        # Every CGF default is the LEGACY value (clamp 2.0, K, no norm, linspace init), so a
        # bare re-run reproduces recorded Ant-Tag runs (change_mds/ant_tag_cgf_port_2026-09-10.md).
        "cgf": dict(t_param="clamp", t_bound=None, t_init_mode="linspace_all_dims",
                    feature_norm="none",
                    t_bound_default=_cgf_t_bound_default,
                    t_init_max_default=_cgf_t_init_max_default),
        # The Ant-Tag ST geometry: 32 inducing points, hidden 128, two post-PMA SABs.
        "st": dict(num_inds=32, dim_hidden=128, num_post_sab=2),
    },
    # The generic protocol: success = is_success from info, else ended before the cap.
    evaluation=Evaluation(add_arguments=_eval_add_arguments, options=_eval_options,
                          default_n_episodes=50),
    # Step 2 of the pipeline: the pursuit / random trajectory mix with the locomotion policy,
    # rebalanced by weighted spread (batch 7.4).
    collection=ANT_TAG_COLLECTION,
    # 10.9: the task objective next to the generic reconstruction, which stays the default (every
    # recorded Ant-Tag pretraining command runs reconstruction without naming it).
    pretraining=Pretraining(objectives={"task": ANT_TAG_TASK_OBJECTIVE}, default_objective=None),
)
