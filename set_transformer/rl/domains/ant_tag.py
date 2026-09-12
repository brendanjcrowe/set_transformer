"""Ant-Tag domain for the RL harness: variant registry, reward shaping, curriculum, PF glue.

Two moves built this module (harness centralisation, ``refactor_plans.md`` in the parent
repo): 2026-09-12 change 1a brought the wrappers and particle-filter glue over from
``experiments/ant_tag/4_train_rl_frozen.py``; change 1b brought the variant registry over
from ``experiments/ant_tag/variants.py``. Both blocks are byte-identical to the originals,
and both old files re-export these names under their historical flat module names, which
the numbered scripts, eval scripts, diagnostics and tests import.

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
  CurriculumCallback            SB3 callback: interpolates the visibility, reward and evasion
                                schedules over training progress and pushes the values into
                                every worker (DummyVecEnv directly, SubprocVecEnv via env_method)
  _CurriculumRouter             outermost wrapper exposing the three setters to env_method
  _set_*_recursive              walk a wrapper stack and set one thing
  get_ant_tag_pf_kwargs         AntTagParticleFilter constructor kwargs read off the live env
  ant_tag_pf_interaction_mapper bridge from one step's observation and env state to the
                                filter's predict / update kwargs (visibility, evasion, target
                                speed, ghost ping, den and counterweighted-den geometry)

``import pdomains`` below registers the ``pdomains-ant-tag-*`` env ids the registry names;
it does not load MuJoCo (that happens on ``gym.make``), so importing this module stays cheap.
"""

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import pdomains  # noqa: F401 - registers the pdomains-ant-tag-* envs
from set_transformer.rl.particle_filters.ant_tag import (
    AntTagParticleFilter,
    CounterweightedDenAntTagParticleFilter,
    GhostAntTagParticleFilter,
    SmartAntTagParticleFilter,
    TwinDenAntTagParticleFilter,
)


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


VARIANTS: dict[str, Variant] = {
    "base": Variant(
        env_id="pdomains-ant-tag-v0",
        particle_filter=AntTagParticleFilter,
        notes="Original dumb-target Ant-Tag. Unimodal belief.",
    ),
    "smart": Variant(
        env_id="pdomains-ant-tag-smart-v0",
        particle_filter=SmartAntTagParticleFilter,
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
        notes=("SmartAntTag with the cdens_hard sensing/tagging geometry: "
               "tag_radius 0.6, visible_radius 1.0 (base env 1.5 / 3.0). "
               "Same 9x9 cage and 400-step cap as `smart`; unimodal belief."),
        default_curriculum="0:100,0.2:100,0.5:1.0,1:1.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "smart_mid": Variant(
        env_id="pdomains-ant-tag-smart-mid-v0",
        particle_filter=SmartAntTagParticleFilter,
        notes=("SmartAntTag, tag_radius 1.0, visible_radius 2.0, target_step "
               "0.5. Geometry sweep around smart_hard (visible 2.0 covers ~16% "
               "of the 9x9 cage)."),
        default_curriculum="0:100,0.2:100,0.5:2.0,1:2.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "smart_hard_slow": Variant(
        env_id="pdomains-ant-tag-smart-hard-slow-v0",
        particle_filter=SmartAntTagParticleFilter,
        notes=("smart_hard radii (tag 0.6, visible 1.0) with a slower target: "
               "target_step 0.3 instead of 0.5. The filter reads target_step "
               "off the env."),
        default_curriculum="0:100,0.2:100,0.5:1.0,1:1.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "smart_mid_slow": Variant(
        env_id="pdomains-ant-tag-smart-mid-slow-v0",
        particle_filter=SmartAntTagParticleFilter,
        notes=("Both relaxations: tag 1.0, visible 2.0, target_step 0.3."),
        default_curriculum="0:100,0.2:100,0.5:2.0,1:2.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "smart_mid_slow_v15": Variant(
        env_id="pdomains-ant-tag-smart-mid-slow-v15-v0",
        particle_filter=SmartAntTagParticleFilter,
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
        notes=("Heavy-near / light-far dens; the pooled belief mean is pinned "
               "at 0 by construction, so the mirror bit lives only in the odd "
               "moments. Visibility radius 1.8."),
        default_curriculum="0:100,0.2:100,0.5:1.8,1:1.8",
    ),
    "cdens_hard": Variant(
        env_id="pdomains-ant-tag-cdens-hard-v0",
        particle_filter=CounterweightedDenAntTagParticleFilter,
        notes=("Counterweighted dens with harder sensing/tagging geometry: "
               "tag_radius 0.6, visible_radius 1.0, spook_radius 1.4."),
        default_curriculum="0:100,0.2:100,0.5:1.0,1:1.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "cdens_terminal": Variant(
        env_id="pdomains-ant-tag-cdens-terminal-v0",
        particle_filter=CounterweightedDenAntTagParticleFilter,
        notes=("Counterweighted dens where the two INACTIVE candidate "
               "positions are terminal hazards (-300), so probing the wrong "
               "arrangement ends the episode instead of wasting time."),
        default_curriculum="0:100,0.2:100,0.5:1.0,1:1.0",
        default_evasion_curriculum="0:0,0.2:0,0.5:1,1:1",
    ),
    "cdens_nospook": Variant(
        env_id="pdomains-ant-tag-cdens-nospook-v0",
        particle_filter=CounterweightedDenAntTagParticleFilter,
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
              f"{variant.particle_filter.__name__}")
        if variant.notes:
            print(f"{'':16s}   {variant.notes}")



# ---------------------------------------------------------------------------
# Wrappers, curriculum callback, PF glue (moved from experiments/ant_tag/4_train_rl_frozen.py)
# ---------------------------------------------------------------------------


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


class CurriculumCallback(BaseCallback):
    """
    SB3 callback that anneals visibility_radius and reward coefficients.

    Visibility schedule: list of (progress_fraction, radius) waypoints.
    Reward schedule: list of (progress_fraction, distance_coeff, entropy_coeff).
    Between waypoints, values are linearly interpolated.

    Default visibility schedule:
      - 0-30%: radius=100 (fully observed)
      - 30-70%: linear decrease 100 → 3.0
      - 70-100%: radius=3.0 (real POMDP)

    Default reward schedule (frac, distance_coeff, entropy_coeff, tag_bonus):
      - 0-30%: distance=1.0, entropy=0.0, tag=0 (pure distance while fully visible)
      - 30-70%: distance 1.0→0.0, entropy 0→0, tag 0→50 (phase in tag bonus)
      - 70-100%: distance=0.0, entropy=0.0, tag=50 (sparse tag reward at real POMDP)
    """

    def __init__(
        self,
        total_timesteps: int,
        schedule: list[tuple[float, float]] | None = None,
        reward_schedule: list[tuple[float, ...]] | None = None,
        evasion_schedule: list[tuple[float, float]] | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        if schedule is None:
            schedule = [(0.0, 100.0), (0.3, 100.0), (0.7, 3.0), (1.0, 3.0)]
        self.schedule = sorted(schedule, key=lambda x: x[0])

        if reward_schedule is None:
            reward_schedule = [
                (0.0, 1.0, 0.0, 0.0),
                (0.3, 1.0, 0.0, 0.0),
                (0.7, 0.0, 0.0, 50.0),
                (1.0, 0.0, 0.0, 50.0),
            ]
        self.reward_schedule = sorted(reward_schedule, key=lambda x: x[0])

        # Constant scale=1.0 (full smart-target strength throughout) is a
        # no-op that matches SmartAntTagEnv's own default, so omitting this
        # arg preserves today's behavior exactly. Only meaningful for the
        # smart target env; harmlessly ignored by the base AntTagEnv.
        if evasion_schedule is None:
            evasion_schedule = [(0.0, 1.0), (1.0, 1.0)]
        self.evasion_schedule = sorted(evasion_schedule, key=lambda x: x[0])

    def _interpolate_schedule(self, schedule, progress: float):
        """Linearly interpolate a schedule. Returns all values after the fraction."""
        if progress <= schedule[0][0]:
            return schedule[0][1:]
        if progress >= schedule[-1][0]:
            return schedule[-1][1:]
        for i in range(len(schedule) - 1):
            frac_lo = schedule[i][0]
            frac_hi = schedule[i + 1][0]
            if frac_lo <= progress <= frac_hi:
                t = (progress - frac_lo) / (frac_hi - frac_lo) if frac_hi > frac_lo else 1.0
                vals_lo = schedule[i][1:]
                vals_hi = schedule[i + 1][1:]
                return tuple(lo + t * (hi - lo) for lo, hi in zip(vals_lo, vals_hi))
        return schedule[-1][1:]

    def _on_training_start(self) -> None:
        # Apply the schedule at the CURRENT progress before the first env step.
        # For a fresh run progress is 0 and this repeats the constructor's
        # values; for a run resumed from a mid-run checkpoint (4_train_rl_st.py
        # --resume_from, 2026-09-08) the envs were just built with the
        # progress-0 radius and coefficients, and without this the first
        # n_envs steps of the fork would be rewarded under the wrong schedule.
        self._apply(self.num_timesteps / self.total_timesteps, announce=True)

    def _on_step(self) -> bool:
        self._apply(self.num_timesteps / self.total_timesteps)
        return True

    def _apply(self, progress: float, announce: bool = False) -> None:
        (radius,) = self._interpolate_schedule(self.schedule, progress)
        reward_vals = self._interpolate_schedule(self.reward_schedule, progress)
        dist_coeff, ent_coeff = reward_vals[0], reward_vals[1]
        tag_bonus_coeff = reward_vals[2] if len(reward_vals) > 2 else 0.0
        gain_coeff = reward_vals[3] if len(reward_vals) > 3 else 0.0
        (evasion_scale,) = self._interpolate_schedule(self.evasion_schedule, progress)

        # Update all training envs (works through VecNormalize → SubprocVecEnv)
        vec_env = self.training_env
        # Unwrap VecNormalize if present
        while hasattr(vec_env, "venv"):
            vec_env = vec_env.venv

        # For both SubprocVecEnv and DummyVecEnv, walk wrapper stacks
        if hasattr(vec_env, "envs"):
            # DummyVecEnv — direct access
            for env in vec_env.envs:
                _set_radius_recursive(env, radius)
                _set_reward_coeffs_recursive(env, dist_coeff, ent_coeff, tag_bonus_coeff, gain_coeff)
                _set_evasion_scale_recursive(env, evasion_scale)
        elif hasattr(vec_env, "env_method"):
            # SubprocVecEnv — call into subprocesses
            vec_env.env_method("set_curriculum_radius", radius)
            vec_env.env_method("set_reward_coeffs", dist_coeff, ent_coeff, tag_bonus_coeff, gain_coeff)
            vec_env.env_method("set_evasion_scale", evasion_scale)

        if self.verbose > 0 and (announce or self.num_timesteps % 10000 < (self.training_env.num_envs if self.training_env else 1)):
            print(
                f"[Curriculum] step={self.num_timesteps}, progress={progress:.2f}, "
                f"vis_radius={radius:.2f}, dist_coeff={dist_coeff:.3f}, "
                f"ent_coeff={ent_coeff:.3f}, tag_bonus={tag_bonus_coeff:.1f}, "
                f"gain_coeff={gain_coeff:.2f}, evasion_scale={evasion_scale:.2f}"
                + (" (applied at training start)" if announce else "")
            )


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


class _CurriculumRouter(gym.Wrapper):
    """Thin outermost wrapper so SubprocVecEnv.env_method can reach inner wrappers."""

    def set_curriculum_radius(self, radius: float):
        _set_radius_recursive(self.env, radius)

    def set_reward_coeffs(self, distance_coeff: float, entropy_coeff: float,
                          tag_bonus_coeff: float, gain_coeff: float | None = None):
        _set_reward_coeffs_recursive(self.env, distance_coeff, entropy_coeff,
                                     tag_bonus_coeff, gain_coeff)

    def set_evasion_scale(self, scale: float):
        _set_evasion_scale_recursive(self.env, scale)


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
