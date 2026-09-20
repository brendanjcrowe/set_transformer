"""Odd-Even domain for the RL harness: variant registry and belief env.

Two moves built this module (harness centralisation, ``refactor_plans.md`` in the parent
repo, 2026-09-12): change 1b brought the variant registry over from
``experiments/odd_even/variants.py``; change 1c the belief env (wrappers, filter glue, env
factory, vec-env helpers) from ``experiments/odd_even/odd_even_belief_env.py``. The blocks
are byte-identical to the originals (``variants`` below names the registry, so the
factory's qualified calls stay as written). Both old files re-export these names for the scripts, diagnostics, viz and tests
that load them through ``_sibling.load(...)``.

VARIANT REGISTRY
----------------
Why a registry: every script needs the same facts about a variant -- the gym env id, the
matching particle filter, the state range and where its runs live -- and a fact copied into
seven scripts is a fact that can disagree with itself. Two things it makes structural rather
than remembered:

1. **The env/filter pairing.** The filter's likelihood table MIRRORS the env's own
   observation model (the Gaussian evaluated on the candidate's parity set and renormalized
   over just that set). If the two drift apart, belief propagation diverges from the env
   silently -- the weights are simply wrong and nothing raises. ``n_dist_size`` travels with
   the pair for the same reason: the filter builds its table from it, so a value that
   disagrees with the env's registration produces a filter for a different POMDP.

2. **The episode cap.** Gymnasium already knows it, so ``episode_cap()`` READS
   ``max_episode_steps`` off the registration rather than defaulting. On this domain a wrong
   cap does not merely mis-measure: the transient (steps 1-21, where the belief is still
   sharpening) is 42% of a 50-step episode and 10% of a 200-step one, so a pooled mean taken
   at the wrong cap is not comparable with anything.

What is deliberately ABSENT, compared with the Ant-Tag registry: there is no visibility
curriculum, no evasion schedule and no EVADING set. The hidden state here is static and the
action is a prediction, so nothing the agent does changes the state or the observation
stream. There is no schedule to ramp.

Adding a variant: register the env id in pdomains, then add one entry here, and it appears
in every script at once.

BELIEF ENV
----------
The Odd-Even belief env, built once and shared by every arm and the eval.

Every encoder arm (4_train_rl_{cgf,st,gaussian}.py), the dataset collector
(2_collect_pf_dataset.py) and the evaluator all build the SAME env through
`make_odd_even_belief_env` here. That is the point of the module: a belief
distribution collected under one env and encoded under a slightly different
one is not a comparison of encoders, and the difference does not raise.

The env stack, innermost first:

    gym.make(env_id)                        raw POMDP, obs = the observation
    StepIndexObservationWrapper             obs := step_count / cap
    OddEvenPFDictWrapper                    obs := {obs, particles, weights}
    ParticleCentringWrapper                 particles -= (n + 1) / 2
    Monitor                                 episode returns for SB3

THE OBSERVATION SPLIT, which is the one thing here that is easy to get
catastrophically wrong in either direction.

`obs_dict["obs"]` is the NORMALIZED STEP INDEX, and nothing else. Three
candidates were considered (domain_mds/oddeven.md, Gap 9):

* The raw observation. Rejected. It reveals s*'s parity outright -- the
  observation model only ever emits integers of s*'s own parity -- and
  locates the state to about +/-3.2 at n=50. A memoryless policy could then
  bypass the belief encoder entirely, which is the same failure the Ant-Tag
  arms avoid by masking obs[-2:], and it would make every encoder arm score
  the same.
* An empty Box. Rejected: a poor SB3 citizen, and VecNormalize cannot
  normalize a zero-width observation.
* The normalized step index. Adopted. It leaks nothing about s* (the step
  count is identical in every episode) and it gives the value function the
  one genuinely useful non-belief signal there is: how long the belief has
  had to sharpen, which is exactly what predicts the reward this step.

But the FILTER still needs the real observation, or its belief never moves
past the prior. `PFDictWithWeightsObservationWrapper` hands the base env's
observation to `pf_interaction_mapper`, and by then that observation is
already the step index -- so the mapper cannot use it. It reads
`info["observations"]` instead, which the env populates at both reset and
step with the integers it actually emitted. `obs_mask_indices` would not
help: it zeroes entries, and a zeroed 1-D observation is a constant.

Both directions are pinned by tests in tests/test_odd_even_pipeline.py --
`test_base_obs_is_the_step_index_not_the_observation` (the agent is not
handed the answer) and `test_filter_still_receives_the_real_observation`
(the filter is not starved). Either failure is silent at runtime.

PARTICLE SCALING. Particles are the raw integer states 1..n. The extractors
divide by `arena_scale`, so the state range maps onto about [-1, 1] via

    (s - (n + 1) / 2) / ((n - 1) / 2)

The centring is done here, by `ParticleCentringWrapper`, because the
extractors only divide. `variants.state_scale` / `variants.state_centre` are
the single source of both numbers, and 2_collect_pf_dataset.py records the
scale in the dataset so pretraining and RL agree (PITFALLS.md section 4).

EXACT-POSTERIOR PRETRAINING (batch 7.3, 2026-09-13). The supervised pretraining that was
``experiments/odd_even/3_pretrain_st_belief.py`` lives here as three ``Objective`` records
(``belief_kl`` / ``mode_ce`` / ``state_ce``, declared on the Domain record's ``pretraining``),
together with the mode-readout probe pieces its end-of-run report uses; see the block's
own header below. Only this problem can compute the target (the env's exact posterior),
which is why the objective is declared here and not in ``rl/pretrain_objectives/``.

``import pdomains`` below registers the ``pdomains-odd-even-*`` env ids. The factory thunk is
pickled by reference to this module when SubprocVecEnv starts workers; the child's import of
it (and the ``import pdomains`` inside the thunk) registers the envs there too.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers the pdomains-odd-even-* envs
from set_transformer.latent_alignment import (LambdaRamp, OnlineAlignment, add_alignment_arguments,
                                              resolve_sinkhorn_blur)
from set_transformer.rl.particle_filters.odd_even import (
    OddEvenBootstrapParticleFilter,
    OddEvenExactSupportParticleFilter,
)
from set_transformer.rl.wrappers.particle_filter import (
    PFDictWithWeightsObservationWrapper,
)
from set_transformer.rl.domains.base import (
    Collection,
    Domain,
    Evaluation,
    Objective,
    PretrainContext,
    PretrainResult,
    Pretraining,
)


# ---------------------------------------------------------------------------
# Variant registry (moved from experiments/odd_even/variants.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One Odd-Even env plus the filter that mirrors its observation model."""

    env_id: str
    particle_filter: type
    #: Upper end of the state range [1, n_dist_size]. Must equal the env
    #: registration's own n_dist_size: the filter builds its likelihood table
    #: from this value, and a mismatch is a filter for a different POMDP.
    n_dist_size: int
    #: What the belief looks like here, and anything a caller has to know that
    #: the code cannot enforce. Printed by --list_variants.
    notes: str = ""


VARIANTS: dict[str, Variant] = {
    "oe10": Variant(
        env_id="pdomains-odd-even-10-v0",
        particle_filter=OddEvenExactSupportParticleFilter,
        n_dist_size=10,
        notes=("Cheap. std_dev 2.000. The belief locks on by about step 14, "
               "so the transient is 28% of the episode."),
    ),
    "oe50": Variant(
        env_id="pdomains-odd-even-50-v0",
        particle_filter=OddEvenExactSupportParticleFilter,
        n_dist_size=50,
        notes=("Main case. std_dev 3.236 spans about 3 same-parity "
               "neighbours each side, so telling s* from s*+/-2 needs many "
               "observations. Transient is 42% of the episode."),
    ),
    "oe50_short": Variant(
        env_id="pdomains-odd-even-50-short-v0",
        particle_filter=OddEvenExactSupportParticleFilter,
        n_dist_size=50,
        notes=("THE encoder-comparison variant. n=50 on a 30-step cap, so the "
               "episode stays inside the transient where the encodings differ. "
               "A mean+variance encoding of the belief is parity-BLIND while "
               "the belief is wide (rounding the belief mean recovers parity "
               "at 0.357 after 2 observations, 0.512 after 8, 0.845 after 30), "
               "so a longer cap averages the effect away."),
    ),

    "oe50_long": Variant(
        env_id="pdomains-odd-even-50-long-v0",
        particle_filter=OddEvenExactSupportParticleFilter,
        n_dist_size=50,
        notes=("Steady-state stress: the same env on a 200-step cap, so "
               "about 90% of the episode is one-hot belief. Use it to show "
               "that long-episode averaging hides encoder differences."),
    ),
}

#: The bootstrap filter is an arm, not a variant: it is the SAME env with a
#: lossier belief. Selecting it through --particle_filter (rather than adding
#: three more registry keys) keeps the env axis and the filter axis separate.
PARTICLE_FILTERS: dict[str, type] = {
    "exact_support": OddEvenExactSupportParticleFilter,
    "bootstrap": OddEvenBootstrapParticleFilter,
}


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

    The single source of truth (PITFALLS.md section 5). Never defaulted: the
    two n=50 variants differ ONLY in this number, so a hardcoded cap silently
    evaluates one as the other.
    """
    spec = gym.spec(resolve(name).env_id)
    if spec.max_episode_steps is None:
        raise ValueError(
            f"{spec.id} registers no max_episode_steps; pass --max_steps")
    return int(spec.max_episode_steps)


def resolve_particle_filter(name: str, filter_name: str | None = None) -> type:
    """The filter for this variant, or the named override.

    An override is legitimate here -- the bootstrap filter is a deliberate
    arm on the same env -- but it must be an Odd-Even filter, because every
    one of them mirrors this env's observation model.
    """
    if filter_name is None:
        return resolve(name).particle_filter
    try:
        return PARTICLE_FILTERS[filter_name]
    except KeyError:
        raise ValueError(
            f"Unknown particle filter {filter_name!r}. Available: "
            f"{sorted(PARTICLE_FILTERS)}"
        ) from None


def state_scale(name: str) -> float:
    """Half-width of the state range: (n - 1) / 2.

    The particle normalization for every encoder arm, and the
    `particle_scale` recorded in the pretraining dataset. Combined with the
    centre below it maps the states [1, n] onto about [-1, 1], which is the
    range the Ant-Tag arena normalization puts its particles in -- so the
    encoders see inputs of the scale they were designed around.

    PITFALLS.md section 4: pretraining and RL must use the SAME scale, which
    is why both read it from here.
    """
    n = resolve(name).n_dist_size
    return (n - 1) / 2.0


def state_centre(name: str) -> float:
    """Centre of the state range: (n + 1) / 2.

    Subtracted before scaling. Without it the mapped particles sit in
    [2/(n-1), 2n/(n-1)] -- roughly [0, 2] -- and the CGF's sign-symmetric
    t directions would all look at the same side of the cloud.
    """
    n = resolve(name).n_dist_size
    return (n + 1) / 2.0


def run_subdir(encoder: str, name: str) -> str:
    """Where this (encoder, variant) pair's runs live under runs/.

    Unlike the Ant-Tag registry there is no historical bare name to preserve
    -- no Odd-Even run has been made on this pipeline -- so every variant is
    named explicitly.
    """
    resolve(name)
    return f"odd_even_{encoder}_{name}"


def warn_if_override_contradicts(name: str, env_id=None,
                                 n_dist_size=None) -> None:
    """Warn when an explicit override disagrees with the chosen variant.

    The registry exists so an env travels with the filter and the state range
    that match it. Overriding one and not the others reintroduces exactly the
    silent divergence the registry removes.
    """
    variant = resolve(name)
    if env_id is not None and env_id != variant.env_id:
        print(f"WARNING: --env_id {env_id!r} overrides variant {name!r} "
              f"(env {variant.env_id!r}) while keeping its filter "
              f"{variant.particle_filter.__name__} and n_dist_size="
              f"{variant.n_dist_size}. If the override env's state range or "
              "observation model differs, the filter's likelihood table is "
              "wrong and belief propagation diverges with no error.")
    if n_dist_size is not None and int(n_dist_size) != variant.n_dist_size:
        print(f"WARNING: --n_dist_size {n_dist_size} contradicts variant "
              f"{name!r} (env {variant.env_id!r} registers "
              f"{variant.n_dist_size}). The filter and the env will disagree "
              "about the state range.")


def add_variant_argument(parser, default: str = "oe50") -> None:
    """Attach the shared --variant / --list_variants flags to a parser.

    The default is oe50, the main case: n=50 on a 50-step cap, where the
    transient is 42% of the episode and the encoder comparison has the most
    room to resolve.
    """
    parser.add_argument(
        "--variant", type=str, default=default, choices=sorted(VARIANTS),
        help=f"Odd-Even env variant (default: {default}). Selects env id, "
             "the matching particle filter, n_dist_size, the run "
             "subdirectory and the episode cap together, so they cannot "
             "disagree.",
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
        except Exception:  # noqa: BLE001 - listing must not fail on one entry
            cap = "?"
        print(f"{name:12s} {variant.env_id:32s} n={variant.n_dist_size:<4} "
              f"cap={cap:<5} {variant.particle_filter.__name__}")
        if variant.notes:
            print(f"{'':12s}   {variant.notes}")


# ---------------------------------------------------------------------------
# Belief env: filter glue, wrappers, factory, vec-env helpers
# (moved from experiments/odd_even/odd_even_belief_env.py)
# ---------------------------------------------------------------------------


#: The block below was written against the registry as a separate module, reached as
#: `variants.<fn>`. Those calls are kept verbatim: `make_odd_even_belief_env` has a parameter
#: named `episode_cap`, which would shadow the registry function of the same name if the
#: calls were unqualified. So `variants` names the registry here.
variants = SimpleNamespace(resolve=resolve, episode_cap=episode_cap,
                           state_scale=state_scale, state_centre=state_centre)


def odd_even_pf_interaction_mapper(base_env_obs, base_env_info,
                                   base_env_action=None, unwrapped_env=None,
                                   previous_base_env_obs=None):
    """Feed the filter the observations the env really emitted.

    `base_env_obs` reaching this function is the step index, not an
    observation, because StepIndexObservationWrapper sits below the PF
    wrapper (see the module docstring). The real integers travel in
    `info["observations"]`, written by both `reset()` and `step()`.

    Raises rather than falling back. A filter quietly fed the step index
    would score candidate states against a number in [0, 1] -- outside the
    state range, so the filter's own range check fires -- but a filter fed
    NOTHING would hold the prior for the whole episode and produce a
    plausible-looking uniform belief. That is the failure this raise exists
    to make loud.
    """
    if base_env_info is None or "observations" not in base_env_info:
        raise KeyError(
            "OddEvenPOMDP must report the observations it emitted in "
            "info['observations']; without it the particle filter has no "
            "evidence and would hold the uniform prior for the whole "
            "episode. The base observation is the step index by design.")
    observations = np.asarray(base_env_info["observations"]).ravel()
    return {"predict_args": {}, "update_args": {"obs_from_env": observations}}


class StepIndexObservationWrapper(gym.ObservationWrapper):
    """Replace the env's observation with the normalized step index.

    Applied BELOW PFDictWithWeightsObservationWrapper, so what reaches the
    agent as `obs_dict["obs"]` never contains an observation of s*. The
    filter is unaffected: it is fed through the interaction mapper from
    `info["observations"]` instead.

    Normalized by the episode cap so the value is in [0, 1] whatever the
    variant's horizon, which keeps the two n=50 variants (caps 50 and 200)
    on one input scale.
    """

    def __init__(self, env: gym.Env, episode_cap: int):
        super().__init__(env)
        if episode_cap <= 0:
            raise ValueError(f"episode_cap must be positive, got {episode_cap}")
        self.episode_cap = int(episode_cap)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step_count = 0

    def observation(self, observation):
        # The raw `observation` is DISCARDED here on purpose. It is still
        # available to the filter through info["observations"].
        return np.array([self._step_count / self.episode_cap],
                        dtype=np.float32)

    def reset(self, **kwargs):
        self._step_count = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        return (self.observation(obs), reward, terminated, truncated, info)


class OddEvenPFDictWrapper(PFDictWithWeightsObservationWrapper):
    """The shared weighted dict wrapper, taught where o0 lives on this env.

    The base class builds the filter with `initial_env_obs = base_env_obs`,
    the env's reset observation. On every other domain that IS the
    observation. Here it is the step index, because
    StepIndexObservationWrapper sits below -- so the base class would hand
    the filter 0.0, which is outside the state range [1, n].

    Only `reset()` needs the substitution. `step()` already routes evidence
    through `pf_interaction_mapper`, which reads info['observations'].

    Why this is a subclass and not a quiet fallback inside the filter:
    `b0 = P(s | o0)` is the standard POMDP convention and the env's own
    reset() applies it, so a filter that silently skipped o0 would sit
    exactly one update behind the env for the whole episode with nothing
    raised. At n=50 on a 50-step cap, o0 is the only evidence the first
    action can use, and step 1 alone is about 18% of the oracle's pooled
    reward -- it was 81% before the env folded o0 in.
    """

    def __init__(self, env, particle_filter_class, particle_filter_kwargs,
                 num_particles, **kwargs):
        # The base class's __init__ probes particle_dim by constructing a
        # throwaway filter with initial_env_obs = the reset observation. That
        # is the step index here, so the filter's range check fires. Wrap the
        # filter class so every construction reads its evidence from the
        # env's info instead of from the base observation -- the probe
        # included, where there is no info yet and the prior is the right
        # starting point.
        self._real_particle_filter_class = particle_filter_class
        self._pending_initial_obs = None
        super().__init__(
            env,
            self._make_filter,
            particle_filter_kwargs,
            num_particles,
            **kwargs,
        )
        # Restore the real class as the public attribute. Callers, tests and
        # run_config records read `particle_filter_class` expecting the class
        # the run actually used, not this adapter.
        self.particle_filter_class = particle_filter_class

    def _make_filter(self, num_particles, initial_env_obs=None, **kwargs):
        """Build the real filter, substituting the evidence it should get.

        `initial_env_obs` from the base class is the step index and is
        DISCARDED. `self._pending_initial_obs`, set by reset() from
        info['observations'], is used instead; None means the uniform prior,
        which is correct for the dimension probe (no episode has started).
        """
        return self._real_particle_filter_class(
            num_particles=num_particles,
            initial_env_obs=self._pending_initial_obs,
            **kwargs,
        )

    def reset(self, **kwargs):
        # Peek at the episode's first observation and stash it where
        # _make_filter will pick it up, then let the base class run
        # unchanged -- so the per-episode PF seeding, the weight sanitizing
        # and the dict assembly stay in one place.
        base_env_obs, info = self.env.reset(**kwargs)
        mapped = odd_even_pf_interaction_mapper(
            base_env_obs=None, base_env_info=info)
        self._pending_initial_obs = np.asarray(
            mapped["update_args"]["obs_from_env"], dtype=np.float32)

        base_env_obs_float = np.asarray(base_env_obs, dtype=np.float32)
        pf_init_kwargs = self.particle_filter_kwargs.copy()
        pf_init_kwargs["initial_env_obs"] = base_env_obs_float
        if self.particle_filter_seed is not None:
            pf_init_kwargs["rng_seed"] = self._derive_pf_seed(
                self._pf_reset_count)
            self._pf_reset_count += 1
        self.particle_filter = self._make_filter(
            num_particles=self.num_particles,
            **{k: v for k, v in pf_init_kwargs.items()
               if k != "initial_env_obs"},
        )
        self._last_base_env_obs_float = base_env_obs_float.copy()
        return self._get_dict_obs(base_env_obs_float), info


class ParticleCentringWrapper(gym.Wrapper):
    """Centre the particle coordinates on the middle of the state range.

    The feature extractors only DIVIDE by `arena_scale`, so the centring half
    of the (s - centre) / scale mapping has to happen before they see the
    particles. Doing it here, rather than inside a filter, keeps the filter's
    particles equal to the actual integer states -- which is what the exact
    belief comparison in the tests and the oracle both depend on.

    The stored dataset records RAW states and the same recipe, so an encoder
    pretrained on the dataset receives the same inputs at RL time.
    """

    def __init__(self, env: gym.Env, centre: float):
        super().__init__(env)
        self.centre = float(centre)

    def _shift(self, obs):
        obs = dict(obs)
        obs["particles"] = (np.asarray(obs["particles"], dtype=np.float32)
                            - np.float32(self.centre))
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._shift(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._shift(obs), reward, terminated, truncated, info


def make_odd_even_belief_env(
    num_particles: int,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    variant: str = "oe50",
    env_id: str | None = None,
    particle_filter_class: type | None = None,
    n_dist_size: int | None = None,
    episode_cap: int | None = None,
    particle_filter_kwargs: dict | None = None,
):
    """Return a callable that builds one Odd-Even belief env.

    Args:
        num_particles: Filter set size. Must be >= n_dist_size, or the filters
            raise: the state is static, so a state with no particle is refuted
            before the first observation and can never come back.
        rank: Worker index. Offsets both the env seed and the filter's
            per-episode seed stream, so parallel workers do not replay one
            another's episodes -- which on this env would mean replaying one
            hidden state.
        seed: Base seed for this worker set.
        variant: Registry key. Supplies env id, filter, n_dist_size and cap.
        env_id / particle_filter_class / n_dist_size / episode_cap: Overrides
            for the registry, for a test or a one-off. Each one is a chance
            for the env and the filter to disagree, so prefer `variant`.
    """
    resolved = variants.resolve(variant)
    env_id = env_id or resolved.env_id
    particle_filter_class = (particle_filter_class
                             or resolved.particle_filter)
    n_dist_size = int(n_dist_size if n_dist_size is not None
                      else resolved.n_dist_size)
    episode_cap = int(episode_cap if episode_cap is not None
                      else variants.episode_cap(variant))
    centre = variants.state_centre(variant)

    pf_kwargs = {
        # The filter's likelihood table MIRRORS the env's observation model,
        # and n_dist_size and sigma_divisor are what parameterize it. They are
        # passed explicitly (rather than left at the filter's own defaults, 10
        # and sqrt(10)) so the pair is stated in one place: the filter's
        # default n_dist_size of 10 against an n=50 env is a filter for a
        # different POMDP, and nothing raises.
        "n_dist_size": n_dist_size,
    }
    if particle_filter_kwargs:
        pf_kwargs.update(particle_filter_kwargs)

    def _init():
        # Registration must happen HERE, not only at module import. This
        # closure is cloudpickled BY VALUE into a SubprocVecEnv child, so the
        # child never runs this module's top-level `import pdomains` and
        # gym.make raises NameNotFound. Whether the child happens to inherit
        # the parent's imports depends on the multiprocessing start method,
        # so relying on it makes n_envs > 1 work or fail by accident.
        import pdomains  # noqa: F401,PLC0415 - registers the envs in-process

        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        # Assert the pairing rather than trust it. n_dist_size drives the
        # filter's whole likelihood table, so a registry entry that has
        # drifted from the registration must not reach the filter.
        env_n = int(env.unwrapped.n_dist_size)
        if env_n != n_dist_size:
            raise ValueError(
                f"{env_id} has n_dist_size={env_n} but the filter would be "
                f"built with {n_dist_size}. The filter's likelihood table "
                "would be for a different POMDP and belief propagation would "
                "diverge from the env silently.")
        env = StepIndexObservationWrapper(env, episode_cap=episode_cap)
        env = OddEvenPFDictWrapper(
            env=env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs=pf_kwargs,
            num_particles=num_particles,
            pf_interaction_mapper=odd_even_pf_interaction_mapper,
            # No masking: the base observation is already the step index, and
            # zeroing a 1-D observation would leave a useless constant.
            obs_mask_indices=None,
            # Deterministic per-episode PF seeding (PITFALLS.md section 2).
            # Derived from (worker seed, episode index) inside the wrapper.
            particle_filter_seed=seed + rank,
        )
        env = ParticleCentringWrapper(env, centre=centre)
        if monitor_dir:
            env = Monitor(env, os.path.join(monitor_dir, str(rank)))
        else:
            env = Monitor(env)
        return env

    return _init


def make_vec_env_from_fns(env_fns, n_envs: int):
    """SubprocVecEnv above one worker, DummyVecEnv at one."""
    if n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def make_vec_normalize(vec_env, training: bool, norm_reward: bool):
    """Normalize the base obs and the reward -- never the PF weights.

    `norm_obs_keys=["obs"]` is the whole point. The weights are a probability
    vector; normalizing them to zero mean and unit variance would leave
    negative "probabilities" summing to nothing, and every weighted encoder
    reads them directly.

    Reward normalization matters more here than on Ant-Tag: -(pred - s*)^2
    reaches -2401 at n=50, and PITFALLS.md records a value loss of order 10^3
    dominating a learned encoder's parameters when the extractor is shared
    between the policy and the value head. Training normalizes the reward;
    every eval reads the RAW reward.

    Reimplemented rather than imported from 4_train_rl_cgf.py: importing an
    Ant-Tag script would pull in MuJoCo and hit the Gap 12 flat-name
    collision (both directories hold a 4_train_rl_cgf.py).
    """
    try:
        return VecNormalize(
            vec_env,
            training=training,
            norm_obs=True,
            norm_reward=norm_reward,
            norm_obs_keys=["obs"],
        )
    except TypeError:
        print("Warning: this SB3 VecNormalize lacks norm_obs_keys; disabling "
              "obs normalization to avoid corrupting PF weights.")
        return VecNormalize(
            vec_env,
            training=training,
            norm_obs=False,
            norm_reward=norm_reward,
        )


# ---------------------------------------------------------------------------
# The Odd-Even ST collapse sentinel (moved from experiments/odd_even/st_feature_sentinel.py,
# change 4, 2026-09-12; that file re-exports these names). The measurement record that was
# the module's docstring is kept verbatim below as ST_SENTINEL_NOTES.
# ---------------------------------------------------------------------------


ST_SENTINEL_NOTES = """An Odd-Even-local encoder-collapse sentinel for the ST arm.

WHY NOT THE SHARED CALLBACK. `STFeatureLoggingCallback` in
set_transformer/rl/feature_extractors/st.py logs `st/feat_std_mean`: the
per-feature standard deviation across the encoder's LAST CACHED FORWARD
BATCH, averaged over features. Three properties make it unsuitable as the
collapse sentinel on this domain, and none of them is a bug in the Ant-Tag
context it was written for:

1. **It is a 1-sample std at n_envs=1.** The statistic is computed from
   `extractor.last_st_features`, which at rollout end holds whatever the last
   forward pass saw. With one worker that is a single observation, so
   `std(dim=0)` is undefined and the metric logs **NaN** -- measured. The
   sentinel is then silently absent in exactly the cheap configuration people
   use for a first run. This callback aggregates over the whole ROLLOUT
   instead, so the sample size is `n_steps * n_envs` regardless of n_envs.

2. **It is not scale-free, so PITFALLS.md's absolute ~0.01 threshold does not
   transfer.** The ST feature scale here depends on how long the encoder was
   pretrained, over more than three orders of magnitude, while the encoder
   stays perfectly informative. A threshold calibrated on Ant-Tag reads a
   healthy Odd-Even encoder as dead.

3. **Averaging over features can hide a live subspace.** A mean over 64
   features is dragged down by dead ones; the max is not. Both are logged
   here.

WHAT THE NUMBERS ACTUALLY LOOK LIKE ON THIS DOMAIN. Measured on 600 held-out
episodes, exact-support filter beliefs at step 8, oe50 (n=50), 8x8 features;
`R^2` is a 5-fold-CV Ridge probe predicting the true state from the feature
vector -- i.e. "is the state still linearly decodable from this encoding?":

    encoder                 feat_std_mean   feat_std_relative   probe R^2
    random init (control)      8.5e-03           3.7e-02          +0.688 [*]
    Sinkhorn pretrained,  5ep  8.5e-02           6.9e-02          +0.985
    Sinkhorn pretrained,  8ep  1.5e-02           1.6e-02          +0.941
    Sinkhorn pretrained, 30ep  9.0e-05           9.3e-04          -0.015
    any encoder, final Linear zeroed
                               0.0               0.0              -0.015

[*] UNSEEDED, and an outlier. Seeded across 8 torch seeds a random-init
    encoder probes at 0.984..0.992. Do not read +0.688 as the control value;
    see "A LINEAR PROBE CANNOT..." below, which is why no probe number in
    this table should be trusted to rank an encoder.

Read the third and fourth rows together, because they are the whole reason
this file exists:

* **A low absolute reading is genuinely ambiguous.** The 8-epoch encoder
  reads 1.5e-2 and is healthy (R^2 = 0.94); the 30-epoch encoder reads
  9.0e-5 and is *really* collapsed -- its feature vectors are identical
  across samples to within 2.5e-4, and the state is no longer decodable at
  all. So `feat_std_mean` alone cannot distinguish "small features" from
  "no features", and an abort rule on it would fire on the healthy run and
  the dead one in whichever order the scale happened to fall.
* **The RELATIVE spread does separate them**: every live encoder measured
  here sits at 1.6e-02 or above, and both dead ones at 9.3e-04 or below --
  more than an order of magnitude of clear air. That is why the relative
  statistic is the sentinel and the absolute one is kept only as context.

**A LINEAR PROBE CANNOT TELL A LIVE ENCODER FROM A DEAD ONE HERE.** This is
the sharpest limit in this file, and it is why the sentinel exists rather
than a probe. Every encoder in the table above -- spanning three orders of
magnitude of feature spread, from live to exactly constant -- probes within
about 0.10 of the others once the encoder init is seeded. Measured across 8
torch seeds, a random-init encoder probes at R^2 = 0.984..0.992 (mean 0.988);
the `+0.688` in the table above is ONE unlucky unseeded draw and should not
be read as the control value. A collapsed encoder still probes at R^2 ~ 0.99.

The reason: **a ~1e-4 signal riding on an O(1) offset is linearly recoverable
but not learnable by a policy head.** Ridge on z-scored features will happily
amplify a 1e-4 direction by 1e4; PPO's policy head, training on advantages
through a shared trunk, will not. So the probe answers "is the information
present at all?" and says yes for a near-constant encoding, while the
question that decides the run is "is it present at a magnitude a policy can
use?" -- which is what the relative spread measures.

Peak-to-peak spread relative to feature magnitude,
`mean((max - min) / |mean|)` over features, on held-out beliefs, separates
them where the probe does not. Two independent measurements, which agree on
every ordering and on which encoders are constant, and disagree on magnitude
by up to 20x -- so treat the REGIME, not the digit, as the finding:

    encoder                  feat_std_mean   p2p/|mean|      verdict
    8ep (this file's)          1.5e-02      7.0e-02 / 3.4e-02  varies
    30ep                       9.1e-05      4.3e-03 / 3.7e-04  CONSTANT
    a separate 8ep run         1.8e-04      --      / 6.3e-04  CONSTANT
    random init, seeds 0-3     8.2e-03..2.0e-02  2.3e-01..4.8e-01  varies

(Second column of each pair is the reviewer's independent run; the third row
is the reviewer's own 8-epoch checkpoint, which sits in the same regime as
the 30-epoch collapse. Two 8-epoch runs landing on opposite sides is itself
the point: at 8 epochs this pretraining is marginal, and which side a run
lands on is not predictable from the loss.)

**Consequence for how this domain is instrumented.** Use the relative-spread
sentinel to decide whether an encoder is ALIVE. Use the probe only for what
it actually measures -- whether the information is present at all. This is a
harder version of PITFALLS.md section 6's "the probe measures capacity, not
use": there the caveat was that a random ISAB projection probed at 0.997, so
a high score does not imply a good policy. Here the caveat is stronger --
on this domain the probe cannot distinguish live from dead at all, so a
chance-level score no longer even implies a frozen encoder cannot work,
because nothing scores at chance level. `oddeven.md`'s build-order step 4
("probe before you train") must be read with that limit in mind: run the
probe to rule out missing information, then check the spread before
concluding an encoder is usable.

**Long Sinkhorn pretraining collapsed the encoder on this domain.** Thirty
epochs on 2040 snapshots drove it to a constant. That is a reportable
property, not a fluke of one seed, and it is the opposite failure from
Ant-Tag's (where pretraining preserved the cloud and destroyed the
decision-relevant asymmetry). It is also a direct instance of PITFALLS.md
section 4's "watch for convergence-in-epoch-10" and "a flat loss is not
necessarily collapse" -- here the loss stayed flat and the encoder WAS
collapsing. Check a probe, not the epoch count.

NO ABORT IS IMPLEMENTED. This logs and warns; it never stops a run. The
threshold below is a reporting aid drawn from five measured encoders on one
variant, which is not enough evidence to kill a multi-hour run on.
"""


#: Relative-spread reading below which the encoding is reported as suspect.
#: Placed in the order-of-magnitude gap measured between live encoders
#: (>= 1.6e-02) and collapsed ones (<= 9.3e-04). Deliberately nearer the
#: dead end: a false "suspect" costs a log line, while a false "healthy"
#: costs the run.
COLLAPSE_RELATIVE_SPREAD = 5e-3


class OddEvenSTFeatureSentinel(BaseCallback):
    """Log scale-free ST feature-spread statistics over each whole rollout.

    Metrics, all under ``st/``:

    ``feat_std_relative``
        ``mean(std_across_rollout / (|mean_across_rollout| + eps))``. The
        sentinel. Scale-free, so it is comparable across pretraining lengths
        and across domains in a way ``feat_std_mean`` is not.
    ``feat_std_relative_max``
        The same ratio's maximum over features. A live subspace inside a
        mostly-dead encoding shows here and not in the mean.
    ``feat_std_max``
        Max per-feature absolute std. Kept for the same reason.
    ``feat_std_mean``
        The shared callback's statistic, recomputed over the rollout rather
        than one forward batch. Logged for continuity with Ant-Tag runs and
        with PITFALLS.md, NOT as the sentinel -- see the module docstring.
    ``feat_abs_mean``
        Mean ``|feature|``. Makes the denominator of the ratio visible, so a
        relative reading can be interpreted rather than guessed at.

    Features are collected by a forward hook on the extractor, so nothing
    depends on ``last_st_features`` and nothing in the shared package is
    touched.
    """

    def __init__(self, eps: float = 1e-8, verbose: int = 0):
        super().__init__(verbose)
        self.eps = float(eps)
        self._warned = False
        #: The statistics from the most recent rollout, under their logged
        #: names. Kept so a test or a post-run check can read exactly what
        #: was recorded; SB3's logger flushes name_to_value each dump.
        self.last_stats: dict[str, float] = {}

    def _on_step(self) -> bool:  # required abstract method
        return True

    def _rollout_features(self) -> torch.Tensor | None:
        """Encoder features of EXACTLY the observations in the rollout buffer.

        Until 2026-09-06 this was a forward hook on the extractor, cleared at
        rollout end. That captured every PPO minibatch forward (n_epochs x the
        rollout) and every EvalCallback forward as well, so at n_epochs=10
        about 91% of the "rollout" sample was re-forwards of the PREVIOUS
        rollout through an encoder mid-update, and the first logged point
        was a different population from all later ones (PITFALLS.md section
        8 item 3). Re-encoding the buffer costs one extra pass over
        n_steps x n_envs observations per rollout and is exact.
        """
        from stable_baselines3.common.utils import obs_as_tensor

        extractor = getattr(self.model.policy, "features_extractor", None)
        buffer = getattr(self.model, "rollout_buffer", None)
        if extractor is None or buffer is None or not isinstance(
                buffer.observations, dict):
            return None
        flat = {key: np.asarray(value).reshape(-1, *value.shape[2:])
                for key, value in buffer.observations.items()}
        n = next(iter(flat.values())).shape[0]
        # The extractor returns [base_obs, st_features]; the base observation
        # is a passthrough and would dilute the statistic with the step
        # index's own (large, deterministic) variation.
        obs_dim = int(np.prod(self.model.observation_space["obs"].shape))
        was_training = extractor.training
        extractor.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, n, 1024):
                batch = {key: value[start:start + 1024] for key, value in flat.items()}
                out = extractor(obs_as_tensor(batch, self.model.device))
                chunks.append(out[:, obs_dim:].detach().float().cpu())
        extractor.train(was_training)
        return torch.cat(chunks, dim=0)

    def _on_rollout_end(self) -> None:
        features = self._rollout_features()
        if features is None:
            return
        if features.shape[0] < 2:
            # Cannot form a std from one sample. Say so rather than logging
            # NaN, which is how the shared callback's reading disappears at
            # n_envs=1 without anyone noticing.
            if not self._warned:
                print("OddEvenSTFeatureSentinel: only "
                      f"{features.shape[0]} feature sample(s) in the "
                      "rollout; no spread statistic is computable.")
                self._warned = True
            return

        with torch.no_grad():
            std = features.std(dim=0)
            abs_mean = features.mean(dim=0).abs()
            relative = _relative(std, abs_mean, self.eps)
            std_mean = float(std.mean())
            relative_mean = float(relative.mean())
            relative_max = float(relative.max())

        self.last_stats = {
            "st/feat_std_relative": relative_mean,
            "st/feat_std_relative_max": relative_max,
            "st/feat_std_max": float(std.max()),
            # NOT "st/feat_std_mean": that key belongs to the shared
            # STFeatureLoggingCallback (last-batch definition, comparable with
            # Ant-Tag). SB3's logger is last-write-wins, and this callback runs
            # after the shared one, so reusing the name silently replaced it.
            "st/feat_std_mean_rollout": std_mean,
            "st/feat_abs_mean": float(abs_mean.mean()),
            "st/feat_samples": float(features.shape[0]),
        }
        for key, value in self.last_stats.items():
            self.logger.record(key, value)

        if relative_mean < COLLAPSE_RELATIVE_SPREAD:
            # A warning, never an abort: five measured encoders on one
            # variant is not enough evidence to kill a run on.
            print("OddEvenSTFeatureSentinel: WARNING relative feature "
                  f"spread {relative_mean:.2e} is below "
                  f"{COLLAPSE_RELATIVE_SPREAD:.0e}; the encoder may have "
                  "collapsed to a constant. Probe it (can a linear readout "
                  "of the encoding recover the true state?) before trusting "
                  f"this run. Absolute feat_std_mean is {std_mean:.2e}, "
                  "which on this domain is NOT itself evidence either way.")


def _relative(std, abs_mean, eps: float):
    """``std / |mean|``, guarded so the guard itself cannot set the scale.

    A fixed additive epsilon would quietly reintroduce an absolute scale:
    with ``eps = 1e-8`` and features of order 1e-4 the epsilon is no longer
    negligible against ``|mean|``, so the ratio drifts under a pure
    rescaling -- measured, 3.029e-02 falling to 3.026e-02 under a 1e-4x
    scaling. Since defeating any absolute scale is the whole point of this
    statistic, the floor is made proportional to the batch's own typical
    magnitude instead. A genuinely all-zero encoding (mean and std both
    exactly 0) still yields 0 rather than a division by zero.
    """
    typical = float(abs_mean.mean())
    floor = eps * typical if typical > 0 else eps
    return std / (abs_mean + floor)


def relative_feature_spread(features, eps: float = 1e-8) -> float:
    """The sentinel statistic, for offline use on a feature matrix.

    Exposed so a probe script or a test can compute exactly what the callback
    logs, on a ``[num_samples, num_features]`` array, without running PPO.
    """
    features = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError(
            "need a [num_samples, num_features] matrix with at least 2 "
            f"samples, got shape {tuple(features.shape)}")
    std = features.std(dim=0)
    abs_mean = features.mean(dim=0).abs()
    return float(_relative(std, abs_mean, eps).mean())


# ---------------------------------------------------------------------------
# Evaluation (change 5.1, 2026-09-12): how Odd-Even scores a policy. Moved here from
# experiments/odd_even/eval_scripts/eval_true_reward_odd_even.py, which now forwards to the
# shared script rl/eval_true_reward.py and re-exports these names for the tests.
# ---------------------------------------------------------------------------
#
# THERE IS NO "SUCCESS" ON THIS DOMAIN, so the Ant-Tag success-rate metric does not transfer.
# What is reported instead, and why each piece is load-bearing:
#
# * **Mean reward per step, SPLIT at the collapse step.** The belief tightens to ~2 live
#   candidates by about step 8 and keeps sharpening after that; the split separates the
#   informative transient from the settled tail. So the pooled mean is largely a measurement of
#   how much uninformed guessing the cap dilutes: at n=50 / cap 50 step 1 alone is about 18% of
#   the oracle's pooled mean, and before reset() folded in its own observation it was 81%.
#   Worse, the pooled mean is not comparable ACROSS CAPS -- the two n=50 variants share a
#   protocol and have IDENTICAL transient columns, and their whole pooled difference (-0.941
#   against -0.259) is dilution. **Steady state is the headline.**
# * **Exact-match rate and mean absolute error**, split the same way. Since 2026-09-03 the
#   env's reward IS the exact-match indicator, so the reward and the exact-match rate now
#   measure the same thing; MAE remains the separate signal, saying how wrong the policy is
#   when it misses.
# * **The Bayes oracle and the play-the-previous-observation baseline**, measured on the SAME
#   episodes. The oracle is info["optimal_prediction"], which under the current 0/1
#   exact-match reward is the posterior MODE. (Until 2026-09-03 the reward was -(pred - s*)^2
#   and that helper returned the posterior MEAN, which was Bayes-optimal for THAT rule. The env
#   now derives it from the reward in force -- see OddEvenPOMDP.get_optimal_prediction -- so
#   this code needs no change to follow it, but any number quoted from a squared-error run is
#   not comparable to a current one.) The gap between the oracle's steady state (-0.329) and
#   play-the-previous-observation (-9.750) is about 9.4 reward per step: that is the value of
#   accumulating evidence, and it is exactly what a belief encoding either delivers or loses.
#
# TWO SEEDING RULES, both from PITFALLS.md section 2 and both sharper here than on Ant-Tag:
#
# 1. **Re-seed AFTER PPO.load** (the shared script does it for every domain).
# 2. **Vary the seed per episode** (``Evaluation.reseed_per_episode``). On this env the hidden
#    state is drawn at reset, so THE SEED IS THE EPISODE: reset(seed=42) replays one episode
#    byte for byte. A constant seed makes 100 episodes one episode counted 100 times, and the
#    tell -- byte-identical results across "different" seeds -- looks like a robust policy.

#: Transient/steady boundary: the measured step at which the n=50 exact posterior has locked
#: on (max(belief) > 0.9). Steps 1..COLLAPSE_STEP are the transient; COLLAPSE_STEP+1..cap are
#: the steady state.
COLLAPSE_STEP = 21


def summarize_episode(rewards, collapse_step: int = COLLAPSE_STEP) -> dict:
    """Split one episode's per-step rewards into transient / steady / pooled.

    `rewards[0]` is step 1. The transient is steps 1..collapse_step and the steady state is
    everything after, so the two are disjoint and their lengths reconstruct the episode
    exactly.

    A steady segment can be EMPTY (an episode shorter than collapse_step), in which case its
    mean is NaN rather than 0.0 -- averaging a missing segment as zero would pull a very
    negative mean toward the oracle and read as improvement.
    """
    rewards = np.asarray(rewards, dtype=np.float64).ravel()
    transient = rewards[:collapse_step]
    steady = rewards[collapse_step:]
    return {
        "transient": float(transient.mean()) if transient.size else float("nan"),
        "steady": float(steady.mean()) if steady.size else float("nan"),
        "pooled": float(rewards.mean()) if rewards.size else float("nan"),
        "n_transient": int(transient.size),
        "n_steady": int(steady.size),
        "n_pooled": int(rewards.size),
    }


def split_metric(values, collapse_step: int = COLLAPSE_STEP) -> dict:
    """Per-episode arrays -> transient / steady / pooled means over all steps.

    MEANS pool across episodes at the STEP level, not by averaging per-episode means:
    episodes here can differ in length, and a per-episode average would weight a short
    episode's steps more heavily.

    SEMs are PER EPISODE: the standard error of the per-episode means over the episodes that
    have any step in the split. Steps within an episode are not independent samples -- once
    the belief locks on, the policy repeats the same right or wrong guess, and 82% of oracle
    episodes have all nine steady rewards identical -- so a step-pooled SEM understated the
    uncertainty ~2.5x (0.009 vs 0.022 at 150 episodes; PITFALLS.md section 8 item 1). Every
    number quoted before 2026-09-06 used the step-pooled SEM; the means were unaffected.
    """
    transient, steady, pooled = [], [], []
    for row in values:
        row = np.asarray(row, dtype=np.float64).ravel()
        transient.append(row[:collapse_step])
        steady.append(row[collapse_step:])
        pooled.append(row)

    def _mean(chunks):
        flat = np.concatenate(chunks) if chunks else np.array([])
        return float(flat.mean()) if flat.size else float("nan")

    def _sem(chunks):
        per_episode = np.array([c.mean() for c in chunks if c.size], dtype=np.float64)
        return (float(per_episode.std(ddof=1) / np.sqrt(per_episode.size))
                if per_episode.size > 1 else float("nan"))

    return {
        "transient": _mean(transient), "transient_sem": _sem(transient),
        "steady": _mean(steady), "steady_sem": _sem(steady),
        "pooled": _mean(pooled), "pooled_sem": _sem(pooled),
    }


def episode_metrics(rewards, predictions, true_states,
                    collapse_step: int = COLLAPSE_STEP) -> dict:
    """Reward, exact-match and absolute error, each split the same way."""
    exact, abs_err = [], []
    for preds, truth in zip(predictions, true_states):
        preds = np.asarray(preds, dtype=np.float64)
        exact.append((preds == float(truth)).astype(np.float64))
        abs_err.append(np.abs(preds - float(truth)))
    return {
        "reward": split_metric(rewards, collapse_step),
        "exact_match": split_metric(exact, collapse_step),
        "abs_error": split_metric(abs_err, collapse_step),
    }


def run_reference_policies(variant: str, n_episodes: int, seed: int,
                           collapse_step: int = COLLAPSE_STEP) -> dict:
    """The Bayes oracle (info['optimal_prediction']) and play-the-previous-obs.

    Run on the RAW env, not the belief env: neither policy uses a particle filter, and both
    read what the env already reports in `info`. Deciding happens BEFORE each step's
    observation arrives, because `step()` converts the action to a prediction first and only
    then draws observations -- an oracle that peeked at the current step's observation would
    score about -1.098 per step at n=50, which no policy can reach.

    Uses `seed + episode_index`, so these are the same episodes the policy is evaluated on
    when it is given the same --seed.
    """
    resolved = resolve(variant)
    cap = episode_cap(variant)
    env = gym.make(resolved.env_id)
    results = {}
    for name in ("oracle", "prev_obs"):
        rewards, predictions, truths = [], [], []
        for episode in range(n_episodes):
            _obs, info = env.reset(seed=seed + episode)
            previous = int(np.asarray(info["observations"]).ravel()[-1])
            episode_rewards, episode_predictions = [], []
            for _step in range(cap):
                if name == "oracle":
                    prediction = int(info["optimal_prediction"])
                else:
                    prediction = previous
                _obs, reward, terminated, truncated, info = env.step(prediction - 1)
                episode_rewards.append(float(reward))
                episode_predictions.append(float(prediction))
                previous = int(np.asarray(info["observations"]).ravel()[-1])
                if terminated or truncated:
                    break
            rewards.append(episode_rewards)
            predictions.append(episode_predictions)
            truths.append(int(info["true_state"]))
        results[name] = episode_metrics(rewards, predictions, truths, collapse_step)
    env.close()
    return results


def print_metrics_block(name: str, metrics: dict, collapse_step: int) -> None:
    reward = metrics["reward"]
    exact = metrics["exact_match"]
    error = metrics["abs_error"]
    print(f"\n{name}")
    print(f"  {'':14s} {'steady (HEADLINE)':>20s} {'transient':>14s} "
          f"{'pooled':>14s}")
    print(f"  {'reward/step':14s} {reward['steady']:20.3f} "
          f"{reward['transient']:14.3f} {reward['pooled']:14.3f}")
    print(f"  {'exact match':14s} {exact['steady']:20.3f} "
          f"{exact['transient']:14.3f} {exact['pooled']:14.3f}")
    print(f"  {'abs error':14s} {error['steady']:20.3f} "
          f"{error['transient']:14.3f} {error['pooled']:14.3f}")
    print(f"  (reward SEM: steady {reward['steady_sem']:.3f}, "
          f"transient {reward['transient_sem']:.3f}, "
          f"pooled {reward['pooled_sem']:.3f})")


def _eval_add_arguments(parser) -> None:
    parser.add_argument(
        "--collapse_step", type=int, default=COLLAPSE_STEP,
        help=f"Transient/steady boundary (default {COLLAPSE_STEP}, the measured n=50 "
             "collapse step). Steps 1..this are the transient.")
    parser.add_argument(
        "--baselines_only", "--oracle_only", action="store_true", dest="baselines_only",
        help="Report just the Bayes oracle and the prev-obs reference. No checkpoint needed "
             "-- use it to reproduce the reference table in domain_mds/oddeven.md. "
             "--oracle_only is an alias: the flag reports BOTH references, so neither name "
             "is quite right on its own and both are accepted.")


def _eval_references(args, variant) -> dict:
    """Print the header and both reference policies, on the episodes the policy will see."""
    cap = episode_cap(args.variant)
    print(f"n={variant.n_dist_size} | Episodes: {args.n_episodes}, seeds {args.seed}.."
          f"{args.seed + args.n_episodes - 1} (one per episode: on this env the hidden "
          "state is drawn at reset, so the seed IS the episode)")
    print(f"Transient = steps 1-{args.collapse_step}, steady = {args.collapse_step + 1}-{cap}")
    references = run_reference_policies(args.variant, args.n_episodes, args.seed,
                                        args.collapse_step)
    print_metrics_block("Bayes oracle (info['optimal_prediction'])",
                        references["oracle"], args.collapse_step)
    print_metrics_block("play the previous observation",
                        references["prev_obs"], args.collapse_step)
    return references


def _eval_report(episodes, references, args, variant, cap) -> dict:
    """The policy's block and its placement on the oracle-to-naive span; returns the policy
    metrics plus the placement for the JSON summary (the references travel separately)."""
    # The env's own reward is what is being measured, so the prediction is read back from
    # info rather than recomputed from the action -- the env owns the 0-indexed to 1-indexed
    # shift.
    rewards = [episode.rewards for episode in episodes]
    predictions = [[float(info["predicted_state"]) for info in episode.infos]
                   for episode in episodes]
    truths = [int(episode.final_info["true_state"]) for episode in episodes]
    metrics = episode_metrics(rewards, predictions, truths, args.collapse_step)
    print_metrics_block(f"policy ({os.path.basename(args.model_path)}, "
                        f"deterministic={args.deterministic})",
                        metrics, args.collapse_step)

    oracle_steady = references["oracle"]["reward"]["steady"]
    prev_steady = references["prev_obs"]["reward"]["steady"]
    policy_steady = metrics["reward"]["steady"]
    span = oracle_steady - prev_steady
    print(f"\nSteady-state placement: oracle {oracle_steady:.3f}, "
          f"policy {policy_steady:.3f}, prev-obs {prev_steady:.3f}")
    fraction = None
    if np.isfinite(span) and span != 0:
        # Where the policy sits on the oracle-to-naive span. This is the quantity the encoder
        # comparison is about: 0 means the belief bought nothing, 1 means the encoding
        # delivered the whole value of accumulating evidence.
        fraction = (policy_steady - prev_steady) / span
        print(f"  fraction of the oracle-to-naive span recovered: {fraction:.3f}")
    return dict(metrics, collapse_step=args.collapse_step,
                steady_state_placement=dict(oracle=oracle_steady, policy=policy_steady,
                                            prev_obs=prev_steady,
                                            fraction_of_span_recovered=fraction))


# ---------------------------------------------------------------------------
# The mode-readout probe's shared pieces (moved from
# experiments/odd_even/diagnostics/probe_cgf_mode_readout.py, batch 7.3, 2026-09-13; the
# diagnostic imports them back). Used by the exact-posterior pretraining's end-of-run report
# below, which is why they live in the package rather than in a diagnostic loaded by path.
# ---------------------------------------------------------------------------


#: The transient/steady split. Steps 1..21 are the transient -- the belief is
#: still sharpening and the encodings provably differ there; 22..cap is the
#: steady state. domain_mds/oddeven.md uses this same boundary for every
#: reward number, so a probe on a different one is not comparable to them.
#: (The same number as COLLAPSE_STEP below, kept under the probe's own name.)
TRANSIENT_MAX_STEP = 21


def collect_rollouts(variant: str, n_episodes: int, seed: int):
    """Roll the RL arms' own belief env, recording one snapshot per decision.

    Rolls `make_odd_even_belief_env` -- the same factory every arm and the
    eval build -- so the belief distribution probed is the one the policy
    sees, including the float32 cast of the weights and the wrapper's
    centring of the particles.

    THE TIMING, which is the one thing here that is easy to get silently
    wrong. `step()` fixes the prediction BEFORE drawing that step's
    observations, so the belief available when choosing the action for step t
    holds o_0 .. o_{t-1}. The reset observation is therefore the snapshot for
    step 1, and the obs returned by step t is the snapshot for step t+1. The
    obs after the final step is never acted on and is dropped. Recording the
    post-step belief against step t instead would hand every encoding one
    extra observation and inflate every number in the table.

    Labels come from the SAME info dict that produced the snapshot:
        target A  info['true_state']         -- s*, constant within an episode
        target B  info['optimal_prediction'] -- argmax_s P(s | o_0..o_{t-1})
    Both are read off the env's float64 posterior rather than recomputed from
    the float32 weights in the observation, so an argmax tie broken
    differently by the cast cannot mislabel a row.

    Returns particles/weights as the extractors will receive them, plus
    labels, episode groups and 1-based step indices.
    """
    resolved = variants.resolve(variant)
    num_states = resolved.n_dist_size
    cap = variants.episode_cap(variant)

    env = make_odd_even_belief_env(
        variant=variant, num_particles=num_states, rank=0, seed=seed)()

    particles, weights, base_obs = [], [], []
    true_states, modes, groups, steps = [], [], [], []

    for episode in range(n_episodes):
        # seed + episode, never a constant seed: the hidden state is drawn at
        # reset, so one seed IS one episode replayed N times (PITFALLS.md #2
        # and the Gap 4 seeding trap).
        obs, info = env.reset(seed=seed + episode)
        for step in range(1, cap + 1):
            particles.append(np.asarray(obs["particles"], dtype=np.float32))
            weights.append(np.asarray(obs["weights"], dtype=np.float32))
            base_obs.append(np.asarray(obs["obs"], dtype=np.float32))
            true_states.append(int(info["true_state"]))
            modes.append(int(info["optimal_prediction"]))
            groups.append(episode)
            steps.append(step)
            # The action is irrelevant to the belief trajectory: it is a
            # prediction and touches neither the hidden state nor the
            # observation model, so the posterior evolves identically under
            # any policy (domain_mds/oddeven.md, 2026-09-03 visualisation).
            obs, _reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
    env.close()

    return {
        "particles": np.stack(particles),
        "weights": np.stack(weights),
        "base_obs": np.stack(base_obs),
        "true_state": np.asarray(true_states, dtype=np.int64),
        "mode": np.asarray(modes, dtype=np.int64),
        "group": np.asarray(groups, dtype=np.int64),
        "step": np.asarray(steps, dtype=np.int64),
        "num_states": num_states,
        "cap": cap,
    }


def _run_extractor(extractor, data, batch_size=1024) -> np.ndarray:
    """Features from a real SB3 extractor, dropping the base-obs passthrough.

    Column 0 of every extractor's output is `obs_dict["obs"]` -- here the
    normalized step index, which is not part of the belief encoding and which
    a 50-way readout would otherwise use as a strong prior over the mode
    (later steps concentrate). The probe must measure the ENCODING, so the
    passthrough is dropped.
    """
    extractor.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data["particles"]), batch_size):
            sl = slice(i, i + batch_size)
            out.append(extractor({
                "obs": torch.from_numpy(data["base_obs"][sl]),
                "particles": torch.from_numpy(data["particles"][sl]),
                "weights": torch.from_numpy(data["weights"][sl]),
            })[:, 1:].numpy())
    return np.concatenate(out).astype(np.float64)


def geometry(features: np.ndarray, posterior_mean: np.ndarray) -> dict:
    """How many DIRECTIONS the encoding actually varies along, and along what.

    This is the mechanism behind the accuracy table and the reason a 64-wide
    encoding can score like a 1-wide one. Three numbers, all scale-free:

        eff_rank      exp(entropy of the normalized singular-value spectrum)
                      of the centred features. 1.0 means one direction
                      carries the variance however many columns there are.
        pc1_var_frac  variance fraction in the leading direction.
        pc1_corr_mean |corr| between that direction and the POSTERIOR MEAN.
                      If this is ~1, the encoding is a reparameterisation of
                      the mean and cannot carry more than the mean does.
        min_pair_corr smallest |corr| between any two features. Near 1 means
                      every column is a monotone restatement of one number;
                      GAUSS2 reads ~0.02 here because mean and variance are
                      genuinely independent, which is the useful contrast.

    Computed in float64 on the same features the classifiers get. Note the
    numerical rank can still be full while eff_rank is 1.0 -- the tail
    directions exist but at a magnitude the standardised/unstandardised
    contrast is exactly about.
    """
    # Absolute per-feature spread, reported ALONGSIDE the relative one because
    # the relative measure divides by the GLOBAL mean magnitude and so cannot
    # separate "small signal" from "large constant offset". ST_E2E is exactly
    # that case: relative spread 9.5e-4 but absolute per-column std 9.1e-5 on
    # features whose mean magnitude is 0.096.
    abs_std = float(features.std(axis=0).mean())

    centred = features - features.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)
    total = (singular ** 2).sum()
    if total <= 0:
        return {"eff_rank": 1.0, "pc1_var_frac": 1.0, "abs_std": abs_std,
                "pc1_corr_mean": float("nan"), "min_pair_corr": float("nan")}
    spectrum = singular ** 2 / total
    eff_rank = float(np.exp(-(spectrum * np.log(spectrum + 1e-300)).sum()))

    left, values, _ = np.linalg.svd(centred, full_matrices=False)
    pc1 = left[:, 0] * values[0]
    pc1_corr = abs(float(np.corrcoef(pc1, posterior_mean)[0, 1]))

    if features.shape[1] > 1:
        corr = np.corrcoef(features.T)
        off_diagonal = corr[~np.eye(features.shape[1], dtype=bool)]
        min_pair = float(np.abs(off_diagonal).min())
    else:
        min_pair = float("nan")

    return {"eff_rank": eff_rank, "pc1_var_frac": float(spectrum[0]),
            "abs_std": abs_std,
            "pc1_corr_mean": pc1_corr, "min_pair_corr": min_pair}


def _fit_predict(features, labels, groups, n_splits, classifier, standardise,
                 seed):
    """Grouped-CV out-of-fold predictions for one (classifier, scaling) pair.

    GroupKFold by episode: every step of one episode shares s*, so an
    ungrouped split puts the same label on both sides and the score measures
    memorisation. Scaling statistics are fitted on the TRAINING fold only --
    fitting them on everything leaks the test fold's distribution into the
    amplification the whole standardised/unstandardised contrast is about.
    """
    # 7.3: imported here, not at module top -- this module is on every RL run's import path
    # and scikit-learn is needed only by the probe.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier

    preds = np.empty_like(labels)
    for train, test in GroupKFold(n_splits=n_splits).split(
            features, labels, groups):
        x_train, x_test = features[train], features[test]
        if standardise:
            mu = x_train.mean(axis=0)
            sd = x_train.std(axis=0)
            sd = np.where(sd < 1e-12, 1.0, sd)
            x_train = (x_train - mu) / sd
            x_test = (x_test - mu) / sd
        if classifier == "logreg":
            # Multinomial (softmax over all 50 classes), which is
            # LogisticRegression's only behaviour from sklearn 1.7 -- the
            # `multi_class` argument that used to select it was removed in
            # 1.9, so passing it raises rather than being ignored.
            model = LogisticRegression(max_iter=3000, C=1.0)
        elif classifier == "mlp":
            model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=600,
                                  random_state=seed, early_stopping=False)
        else:
            raise ValueError(f"unknown classifier {classifier!r}")
        model.fit(x_train, labels[train])
        preds[test] = model.predict(x_test)
    return preds


def _split_accuracy(preds, labels, steps):
    """Top-1 accuracy in the transient, the steady state, and pooled."""
    correct = preds == labels
    transient = steps <= TRANSIENT_MAX_STEP
    steady = ~transient
    return {
        "transient": float(correct[transient].mean()) if transient.any()
        else float("nan"),
        "steady": float(correct[steady].mean()) if steady.any()
        else float("nan"),
        "pooled": float(correct.mean()),
    }


# ---------------------------------------------------------------------------
# Supervised pretraining on the exact posterior (moved from
# experiments/odd_even/3_pretrain_st_belief.py, batch 7.3, 2026-09-13; that script is now an
# entry point of rl/pretrain.py). Three objectives, declared on the Domain record below and
# offered by `rl/pretrain.py --domain odd_even --objective belief_kl|mode_ce|state_ce`:
#
#     belief_kl   soft cross-entropy against the exact posterior. DEFAULT.
#                 Asks for the whole belief, which subsumes both targets.
#     mode_ce     hard cross-entropy against the posterior argmax (target B).
#     state_ce    hard cross-entropy against the true state (target A). The
#                 Bayes-optimal predictor of s* IS the posterior mode, so this
#                 is a noisier version of mode_ce.
#
# WHY (domain_mds/oddeven.md, 2026-09-04/05): every encoder the pipeline had produced was a
# reparameterisation of the posterior MEAN (Sinkhorn decoders collapsed to a point mass at
# the weighted mean; the end-to-end ST collapsed under a sparse reward), so the training
# signal never asked the encoder for the belief. This asks for it directly: the encoder is the
# RL arm's own extractor (same geometry, weight channel and normalisation), a linear head maps
# its features to n logits, and the loss is the cross-entropy against the exact posterior from
# `info['belief']` (= KL(posterior || model) up to a constant). The checkpoint is saved in the
# format `rl/train.py --pretrained_path` loads (geometry `config` included).
#
# PROTOCOL. Data is rolled from the raw env with `data_seed + episode`, one snapshot per
# decision, BEFORE that step's observation (the same timing as the readout probe and the RL
# arms). Particles are the states centred on (n + 1) / 2, weights are the exact posterior --
# identical to what the exact-support filter hands the RL arm (oddeven.md Gap 5). Train and
# validation episodes come from disjoint seed ranges. The end-of-run report (`--skip_probe`
# turns it off) probes the FROZEN best checkpoint with the mode-readout protocol above on its
# default seed (9000), so the rows join that table.
#
# ENCODERS: `st` and `cgf`. For cgf the same objective, data and head fit t (unless
# --t_frozen), the running feature norm's statistics and a readout MLP sized with
# --match_params to the ST's parameter count -- the size- and supervision-matched pretraining
# test the RL comparison needs.
# ---------------------------------------------------------------------------


# Encoders the exact-posterior objectives can pretrain: EVERY learned encoder of the shared table
# (st, cgf, deepset, pointnet; batch 10.4, 2026-09-14 -- before it a hand-kept tuple ("st", "cgf")).
# The check is `encoder.learned` in `_belief_resolve_arguments`; the pooled arms read the exact
# posterior through their weight channel (weights x N as an input column) and their weighted /
# masked pooling, and their checkpoint is written by the extractor's own `checkpoint_state()`.


# -- data ----------------------------------------------------------------------------------

def collect_posterior_snapshots(variant: str, n_episodes: int, seed: int) -> dict:
    """One snapshot per decision: exact posterior, true state, mode, step."""
    resolved = variants.resolve(variant)
    cap = variants.episode_cap(variant)
    env = gym.make(resolved.env_id)
    beliefs, true_states, modes, steps, groups = [], [], [], [], []
    for episode in range(n_episodes):
        _obs, info = env.reset(seed=seed + episode)
        for step in range(1, cap + 1):
            beliefs.append(np.asarray(info["belief"], dtype=np.float32))
            true_states.append(int(info["true_state"]))
            modes.append(int(info["optimal_prediction"]))
            steps.append(step)
            groups.append(episode)
            _obs, _r, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break
    env.close()
    return {
        "weights": np.stack(beliefs),
        "true_state": np.asarray(true_states, dtype=np.int64),
        "mode": np.asarray(modes, dtype=np.int64),
        "step": np.asarray(steps, dtype=np.int64),
        "group": np.asarray(groups, dtype=np.int64),
        "n": int(resolved.n_dist_size),
        "cap": cap,
    }


def load_posterior_snapshots(path, variant: str, val_frac: float) -> tuple[dict, dict]:
    """The collected dataset (`python -m set_transformer.rl.collect --domain odd_even`, labelled
    since batch 10.5) as the two dicts `collect_posterior_snapshots` returns, split by EPISODE.

    On the exact-support filter the row's weights ARE the exact posterior over the states 1..n,
    so the row is its own target; the file's `true_state`, `optimal_prediction` and `episode`
    arrays are the labels. Two rules keep the rows equal to what the rolling path produces:
    (1) the collector records the belief after the LAST step too (step index == cap), which no
    policy ever acts on and the recorded runs never saw -- dropped here; (2) the collector's step
    index counts from 0 at the reset, the rolling path from 1 -- shifted here. The validation
    episodes are the LAST `val_frac` of the episodes (rounded up, at least one), disjoint from
    training: neighbouring rows of one episode are near duplicates, so a random row split leaks.
    """
    resolved = variants.resolve(variant)
    n = int(resolved.n_dist_size)
    cap = variants.episode_cap(variant)
    with np.load(path, allow_pickle=True) as z:
        missing = [k for k in ("particles", "weights", "steps", "true_state", "optimal_prediction",
                               "episode") if k not in z.files]
        if missing:
            raise ValueError(f"{path} is not a labelled Odd-Even dataset: missing arrays {missing} "
                             "(collect it with python -m set_transformer.rl.collect --domain odd_even "
                             "--variant <variant>; datasets written before 2026-09-14 carry no labels)")
        meta = json.loads(str(z["metadata"])) if "metadata" in z.files else {}
        particles = np.asarray(z["particles"], dtype=np.float32)
        weights = np.asarray(z["weights"], dtype=np.float32)
        steps = np.asarray(z["steps"]).astype(np.int64)
        true_state = np.asarray(z["true_state"]).astype(np.int64)
        mode = np.asarray(z["optimal_prediction"]).astype(np.int64)
        episode = np.asarray(z["episode"]).astype(np.int64)
    if meta.get("variant") not in (None, variant):
        raise ValueError(f"{path} was collected on variant {meta.get('variant')!r}, this run is "
                         f"{variant!r}")
    if particles.shape[1:] != (n, 1):
        raise ValueError(f"{path} holds {particles.shape[1:]} particles per row; the exact-posterior "
                         f"objectives need one particle per state, ({n}, 1) on {variant}")
    states = np.arange(1, n + 1, dtype=np.float32)
    if not np.array_equal(particles[:, :, 0], np.broadcast_to(states, particles[:, :, 0].shape)):
        raise ValueError(f"{path} does not hold the exact-support filter (particles must be the "
                         f"states 1..{n} in order; collect with the variant's default filter)")
    keep = steps < cap                       # rule (1)
    if not keep.any():
        raise ValueError(f"{path} has no belief a policy acts on (every step index >= cap {cap})")
    weights, steps = weights[keep], steps[keep] + 1          # rule (2)
    true_state, mode, episode = true_state[keep], mode[keep], episode[keep]
    episodes = np.unique(episode)
    if len(episodes) < 2:
        raise ValueError(f"{path} holds {len(episodes)} episode(s); an episode-disjoint split needs 2+")
    n_val = max(1, int(math.ceil(val_frac * len(episodes))))
    if n_val >= len(episodes):
        raise ValueError(f"val_frac={val_frac} leaves no training episode of {len(episodes)}")
    val_episodes = set(episodes[len(episodes) - n_val:].tolist())
    is_val = np.array([e in val_episodes for e in episode.tolist()])

    def part(mask):
        return {"weights": weights[mask], "true_state": true_state[mask], "mode": mode[mask],
                "step": steps[mask], "group": episode[mask], "n": n, "cap": cap}
    return part(~is_val), part(is_val)


class BeliefBatches:
    """Tensors on the device; particles are the centred states, shared."""

    def __init__(self, data: dict, centre: float, device: torch.device):
        n = data["n"]
        self.n_rows = len(data["weights"])
        self.weights = torch.from_numpy(data["weights"]).to(device)
        self.true_state = torch.from_numpy(data["true_state"] - 1).to(device)
        self.mode = torch.from_numpy(data["mode"] - 1).to(device)
        self.step = data["step"]
        self.group = data["group"]
        states = torch.arange(1, n + 1, dtype=torch.float32) - float(centre)
        self.particles = states.reshape(1, n, 1).to(device)      # [1, N, 1]
        self.device = device

    def obs(self, index: torch.Tensor) -> dict:
        b = len(index)
        return {
            "obs": torch.zeros(b, 1, device=self.device),
            "particles": self.particles.expand(b, -1, -1),
            "weights": self.weights[index],
        }


# -- model ---------------------------------------------------------------------------------

def _belief_space(n_states: int) -> gym.spaces.Dict:
    """The observation space the extractor is constructed against (n centred states, 1-D)."""
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (n_states, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (n_states,), np.float32),
    })


def build_extractor(args, space, scale: float, pretrained_path: str | None = None):
    """The encoder under test, as the RL arm's own SB3 extractor class, built from the flags
    THROUGH THE SHARED ENCODER TABLE (rl/encoders.py) -- the same construction
    `rl/train.py --encoder <name>` performs, so a checkpoint pretrained here loads there
    without translation. ``args.encoder`` names the arm; ``scale`` is the arena scale;
    ``pretrained_path`` (the ST's --init_from, or a finished checkpoint for the probe) is
    loaded by the extractor's own loader."""
    # Imported here: rl/encoders.py imports rl/domains/base.py, and this module is imported
    # by rl/domains/__init__.py's lookup, so a top-level import would be a cycle.
    from set_transformer.rl import encoders as _encoders
    encoder = _encoders.get(args.encoder)
    kwargs = encoder.extractor_kwargs(args)
    kwargs["arena_scale"] = float(scale)
    kwargs[encoder.extractor_class.PRETRAINED_PATH_KWARG] = pretrained_path
    return encoder.extractor_class(space, **kwargs)


def extractor_geometry(extractor) -> dict:
    """The geometry record the extractor's loader checks; since batch 10.4 the extractor says it
    itself (`checkpoint_config`), for every learned encoder."""
    return extractor.checkpoint_config()


class BeliefEncoderWithHead(nn.Module):
    def __init__(self, extractor: nn.Module, n_states: int):
        super().__init__()
        self.extractor = extractor
        st_dim = extractor.features_dim - 1                        # drop the obs passthrough
        self.head = nn.Linear(st_dim, n_states)

    def features(self, obs: dict) -> torch.Tensor:
        return self.extractor(obs)[:, 1:]

    def forward(self, obs: dict) -> torch.Tensor:
        return self.head(self.features(obs))


def belief_loss(logits: torch.Tensor, batches: BeliefBatches, index: torch.Tensor,
                objective: str) -> torch.Tensor:
    if objective == "belief_kl":
        target = batches.weights[index]
        return -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    if objective == "mode_ce":
        return F.cross_entropy(logits, batches.mode[index])
    if objective == "state_ce":
        return F.cross_entropy(logits, batches.true_state[index])
    raise ValueError(objective)


@torch.no_grad()
def evaluate_belief_head(model, batches: BeliefBatches, objective: str, batch_size: int,
                         transient_max_step: int) -> dict:
    model.eval()
    losses, preds, feats = [], [], []
    for start in range(0, batches.n_rows, batch_size):
        index = torch.arange(start, min(start + batch_size, batches.n_rows),
                             device=batches.device)
        obs = batches.obs(index)
        f = model.features(obs)
        logits = model.head(f)
        losses.append(belief_loss(logits, batches, index, objective).item() * len(index))
        preds.append(logits.argmax(dim=-1))
        feats.append(f)
    preds = torch.cat(preds)
    feats = torch.cat(feats).double()
    transient = torch.from_numpy(batches.step <= transient_max_step).to(batches.device)

    def _split(hit):
        return (float(hit[transient].float().mean()),
                float(hit[~transient].float().mean()))

    mode_tr, mode_st = _split(preds == batches.mode)
    state_tr, state_st = _split(preds == batches.true_state)
    centred = feats - feats.mean(dim=0)
    singular = torch.linalg.svdvals(centred)
    spectrum = singular ** 2 / (singular ** 2).sum().clamp_min(1e-300)
    eff_rank = float(torch.exp(-(spectrum * torch.log(spectrum + 1e-300)).sum()))
    return {
        "loss": sum(losses) / batches.n_rows,
        "head_mode_acc": {"transient": mode_tr, "steady": mode_st},
        "head_true_state_acc": {"transient": state_tr, "steady": state_st},
        "feature_abs_std": float(feats.std(dim=0).mean()),
        "feature_eff_rank": eff_rank,
    }


def save_belief_checkpoint(model: BeliefEncoderWithHead, path: Path, args, epoch: int,
                           val: dict, geometry: dict, alignment: dict | None = None) -> None:
    """The format the RL arm's extractor loads, assembled by
    :func:`~set_transformer.rl.pretrained_encoder.encoder_checkpoint` from the extractor's own
    `checkpoint_state()` / `checkpoint_config()` (batch 10.4; before it an if-chain on the encoder
    name lived here: ST keys under ``set_transformer.``, CGF the whole state). ``geometry`` is the
    same record the extractor reports and is kept in the signature for the entry point; the
    extractor's is written. Top-level ``particle_scale`` (the Trainer's convention) is new since
    10.4; the loaders compare it with their ``arena_scale``. ``alignment`` (2026-09-19): the
    online latent metric-alignment record of the run, written into the config when the term
    was on; absent otherwise, so the recorded checkpoints' shape is unchanged."""
    from set_transformer.rl.pretrained_encoder import encoder_checkpoint   # noqa: PLC0415 - cycle
    torch.save(encoder_checkpoint(
        model.extractor,
        config={"objective": args.objective, "variant": args.variant,
                "arena_scale": float(model.extractor.arena_scale),
                "encoder": args.encoder,
                "encoder_params": int(getattr(args, "encoder_params", 0)),
                # The producer's historical name, kept so the recorded checkpoints' readers
                # keep working; the unified record block is batch 7.5's.
                "pretraining": "3_pretrain_st_belief.py",
                **({"alignment": alignment} if alignment is not None else {})},
        head_state_dict={k: v.detach().cpu() for k, v in model.head.state_dict().items()},
        epoch=epoch, val=val, args=vars(args)), path)


# -- the Objective hooks (rl/domains/base.py::Objective) -------------------------------------

def _belief_add_arguments(parser: argparse.ArgumentParser, domain=None) -> None:
    """The objective's flags, spelled as the package command spells them (decision 2 of plan
    section 7: --num_epochs / --learning_rate; the entry point maps --epochs / --lr)."""
    g = parser.add_argument_group("exact-posterior objective: data")
    g.add_argument("--data_path", type=str, default=None,
                   help="A labelled collected dataset (python -m set_transformer.rl.collect --domain "
                        "odd_even; batch 10.5). Default: the variant's dataset under the run root "
                        "when it exists, else the episodes are rolled here (--n_train_episodes).")
    g.add_argument("--data_source", choices=("auto", "dataset", "rolled"), default="auto",
                   help="auto (default): --data_path, else the variant's dataset under the root if "
                        "it exists, else rolled. dataset: the file is required. rolled: always roll "
                        "(the recorded runs' path).")
    g.add_argument("--val_frac", type=float, default=0.1,
                   help="Dataset path only: the LAST val_frac of the episodes are validation "
                        "(the recorded runs rolled 400 of 4,400).")
    g.add_argument("--n_train_episodes", type=int, default=4000)
    g.add_argument("--n_val_episodes", type=int, default=400)
    g.add_argument("--data_seed", type=int, default=100000,
                   help="Train episodes use data_seed + e; validation "
                        "episodes data_seed + 10_000_000 + e. Both are far "
                        "from the probe's 9000 + e.")
    t = parser.add_argument_group("exact-posterior objective: training")
    t.add_argument("--num_epochs", type=int, default=40)
    t.add_argument("--batch_size", type=int, default=512)
    t.add_argument("--learning_rate", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--init_from", default=None,
                   help="ST only: warm-start the encoder from a reconstruction (Sinkhorn) "
                        "checkpoint instead of random init. Geometry must match the flags.")
    t.add_argument("--freeze_encoder", action="store_true",
                   help="Train only the linear head; the encoder (random or --init_from) "
                        "is held fixed. The linear-readout anchor for --init_from.")
    p = parser.add_argument_group("exact-posterior objective: end-of-run probe")
    p.add_argument("--probe_episodes", type=int, default=300)
    p.add_argument("--probe_seed", type=int, default=9000,
                   help="probe_cgf_mode_readout.py's default, so rows join its table")
    p.add_argument("--probe_splits", type=int, default=5)
    p.add_argument("--skip_probe", action="store_true")
    # 2026-09-19: the latent metric-alignment term with online Sinkhorn targets (the exact
    # posterior on the shared support, weighted: an unweighted target is identically zero here);
    # off at the default lambda 0. Spelled by the shared helper, as the task objective's.
    al = parser.add_argument_group("exact-posterior objective: latent metric alignment (optional; "
                                   "online Sinkhorn targets, read only with --align_lambda > 0)")
    add_alignment_arguments(al, sinkhorn=True)     # --sinkhorn_blur: None -> the domain's default


def _belief_resolve_arguments(parser, args, domain, encoder) -> None:
    if not encoder.learned:
        parser.error(f"exact-posterior pretraining needs a learned encoder (st, cgf, deepset, "
                     f"pointnet); {encoder.name!r} has no parameters to train")
    if args.variant is None:
        parser.error("exact-posterior pretraining rolls the env itself: pass --variant "
                     "<registry key> (--list_variants shows them)")
    if args.init_from and encoder.name != "st":
        raise SystemExit("--init_from is implemented for --encoder st only")
    resolve_sinkhorn_blur(args, domain)          # 2026-09-19: the domain's default when omitted
    # 10.5 (decision 6): where the beliefs come from, decided once, printed, recorded.
    if args.data_source == "rolled":
        if args.data_path:
            parser.error("--data_source rolled and --data_path contradict each other")
        args.data_path = None
    else:
        if args.data_path is None:
            from set_transformer.rl import run_records
            candidate = run_records.dataset_path(domain.name, args.variant,
                                                 root=getattr(args, "output_root", None))
            if candidate.exists():
                args.data_path = str(candidate)
            elif args.data_source == "dataset":
                parser.error(f"--data_source dataset: no collected dataset at {candidate} "
                             f"(python -m set_transformer.rl.collect --domain {domain.name} "
                             f"--variant {args.variant}) and no --data_path")
        elif not os.path.isfile(args.data_path):
            parser.error(f"dataset {args.data_path} does not exist")
        if not (0.0 < args.val_frac < 1.0):
            parser.error("--val_frac must be in (0, 1)")
    args.data_source = "dataset" if args.data_path else "rolled"
    print(f"Beliefs: {args.data_source}" + (f" ({args.data_path}; validation = the last "
                                            f"{args.val_frac:g} of the episodes)" if args.data_path
                                            else f" (rolled: {args.n_train_episodes} + "
                                                 f"{args.n_val_episodes} episodes, data_seed "
                                                 f"{args.data_seed})"))


def _belief_locate(args) -> dict:
    """Where a given dataset says it belongs (10.5): the collector records `variant` and `env_id`;
    without --data_path there is nothing to read (the objective rolls the env)."""
    path = getattr(args, "data_path", None)
    if not path or not os.path.isfile(path):
        return {}
    with np.load(path, allow_pickle=True) as z:
        meta = json.loads(str(z["metadata"])) if "metadata" in z.files else {}
    return {"variant": meta.get("variant"), "env_id": meta.get("env_id")}


def _belief_prepare(parser, args, domain, encoder, device):
    """The geometry the encoder's own resolution needs: n states, 1-D, the state scale. The
    episodes themselves are rolled in `run` (they draw nothing from torch's RNG, so the
    checkpoints are unchanged by the order; a --dry_run then costs nothing)."""
    resolved = variants.resolve(args.variant)
    args.num_particles = int(resolved.n_dist_size)
    args.dim_particles = 1
    args.arena_scale = float(variants.state_scale(args.variant))
    return None


def _belief_run_name(args, now: datetime, target: str) -> str:
    """`<stamp>_<objective>_seed<n>`, with the encoder's name inserted for every arm but the
    ST (the script's naming; the run tag is appended by the command)."""
    stamp = now.strftime("%Y%m%d_%H%M%S")
    if args.encoder != "st":
        return f"{stamp}_{args.encoder}_{target}_seed{args.seed}"
    return f"{stamp}_{target}_seed{args.seed}"


def _belief_run(args, ctx: PretrainContext, target: str) -> PretrainResult:
    """The training loop of `3_pretrain_st_belief.py::main`, moved: roll the episodes, fit the
    encoder + linear head, keep the best checkpoint by validation loss."""
    # The command's parse guarantees these; recorded in args.json as the run's truth.
    args.objective = target
    args.encoder = ctx.encoder.name
    device = torch.device(ctx.device)
    resolved = variants.resolve(args.variant)
    centre, scale = variants.state_centre(args.variant), variants.state_scale(args.variant)
    n_states = resolved.n_dist_size
    transient_max_step = TRANSIENT_MAX_STEP

    run_dir = Path(ctx.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # 7.5: checkpoints/ under the root layout; the run folder itself under the entry point's
    # historical layout (ctx.checkpoint_dir is None there).
    checkpoint_dir = Path(ctx.checkpoint_dir) if ctx.checkpoint_dir else run_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"run dir: {run_dir}")
    print(f"variant {args.variant} | env {resolved.env_id} | n={n_states} | "
          f"cap {variants.episode_cap(args.variant)} | normalisation (s - {centre}) / {scale} | "
          f"device {device}")

    # ---- data ----------------------------------------------------------------
    t0 = time.time()
    if getattr(args, "data_path", None):
        # 10.5 (decision 6): the collected, labelled dataset; episode-disjoint split.
        train, val = load_posterior_snapshots(args.data_path, args.variant, args.val_frac)
        print(f"loaded {len(train['weights'])} train rows ({len(np.unique(train['group']))} episodes) / "
              f"{len(val['weights'])} val rows ({len(np.unique(val['group']))} episodes) from "
              f"{args.data_path} in {time.time() - t0:.0f}s; train distinct s*: "
              f"{len(set(train['true_state'].tolist()))}")
    else:
        train = collect_posterior_snapshots(args.variant, args.n_train_episodes, args.data_seed)
        val = collect_posterior_snapshots(args.variant, args.n_val_episodes, args.data_seed + 10_000_000)
        print(f"collected {len(train['weights'])} train rows / {len(val['weights'])} val rows "
              f"in {time.time() - t0:.0f}s; train distinct s*: "
              f"{len(set(train['true_state'].tolist()))}")
    train_b = BeliefBatches(train, centre, device)
    val_b = BeliefBatches(val, centre, device)

    # ---- model ---------------------------------------------------------------
    space = _belief_space(n_states)
    if args.init_from and args.encoder != "st":
        raise SystemExit("--init_from is implemented for --encoder st only")
    extractor = build_extractor(args, space, scale, args.init_from)
    if args.init_from:
        # Verify the warm start landed (PITFALLS.md section 1 habit): compare every
        # encoder tensor against the checkpoint.
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        ref = {k[len("set_transformer."):]: v for k, v in ck["model_state_dict"].items()
               if k.startswith("set_transformer.")}
        cur = extractor.encoder.state_dict()
        delta = max(float((ref[k] - cur[k].cpu()).abs().max()) for k in ref)
        print(f"Verified: encoder warm-started from {args.init_from} "
              f"({len(ref)} tensors, max|delta| = {delta})")
        if delta != 0.0:
            raise SystemExit("warm start did not land exactly")
    if args.freeze_encoder:
        for p in extractor.parameters():
            p.requires_grad_(False)
        extractor.eval()
    geometry = extractor_geometry(extractor)
    model = BeliefEncoderWithHead(extractor, n_states).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_head = model.head.weight.numel() + n_states
    n_encoder_buffers = sum(b.numel() for b in extractor.buffers())
    print(f"encoder+head parameters: {n_params:,}  (head {n_head:,}; encoder "
          f"{n_params - n_head:,} trainable + {n_encoder_buffers:,} buffer values)")
    if args.encoder != "cgf":                  # the CGF's count comes from its readout sizing
        args.encoder_params = int(n_params - n_head)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in trainable):,}"
          + ("  (encoder FROZEN, head only)" if args.freeze_encoder else ""))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(train_b.n_rows / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs * steps_per_epoch)

    # Reference: the entropy of the exact posterior is the floor of the
    # belief_kl loss (CE = H(posterior) + KL). Print it so the loss is readable.
    if target == "belief_kl":
        w = val_b.weights.clamp_min(1e-30)
        entropy = float(-(val_b.weights * w.log()).sum(dim=-1).mean())
        print(f"val posterior entropy (loss floor for belief_kl): {entropy:.4f}")

    # 2026-09-19 (online alignment): the latent metric-alignment term on the encoder's features,
    # its targets the debiased Sinkhorn divergences between the batch's own beliefs (the shared
    # centred support divided by the state scale, as the extractor does; the exact posteriors as
    # the weights). Off at the default lambda 0: the loop below is then the recorded one, untouched.
    align_lambda = float(getattr(args, "align_lambda", 0.0) or 0.0)
    align = ramp = align_record = None
    val_align_pairs = val_align_targets = None
    if align_lambda > 0:
        blur_source = resolve_sinkhorn_blur(args, getattr(ctx, "domain", None))   # a caller that skipped resolve
        align = OnlineAlignment(blur=float(args.sinkhorn_blur), scaling=float(args.sinkhorn_scaling),
                                metric=args.align_metric, pairs=args.align_pairs,
                                seed=int(getattr(args, "seed", 0) or 0))
        ramp = LambdaRamp(align_lambda, int(args.align_warmup_epochs), int(args.align_ramp_epochs))
        val_align_pairs = align.fixed_pairs(val_b.n_rows, args.align_val_pairs)
        val_align_targets = align.targets(val_b.particles.expand(val_b.n_rows, -1, -1) / scale,
                                          val_b.weights, val_align_pairs)
        align_record = {"lambda": align_lambda, "warmup_epochs": int(args.align_warmup_epochs),
                        "ramp_epochs": int(args.align_ramp_epochs), **align.record(),
                        "val_pairs": int(len(val_align_pairs[0]))}
        print(f"latent alignment ON: lambda={align_lambda} ({args.align_metric}), warmup "
              f"{args.align_warmup_epochs} / ramp {args.align_ramp_epochs} epochs; online targets (blur "
              f"{args.sinkhorn_blur} [{blur_source}] = {args.sinkhorn_blur * scale:.3f} states, scaling {args.sinkhorn_scaling}, "
              f"{args.align_pairs} pairs per batch); val_align_r over {len(val_align_pairs[0])} fixed pairs "
              f"of the {val_b.n_rows} held-out rows; the validation loss stays the objective's alone")
        if args.align_warmup_epochs + args.align_ramp_epochs >= args.num_epochs:
            print("WARNING: warmup + ramp >= num_epochs: lambda never reaches its target, so no "
                  "checkpoint from this run is fully aligned")

    def _val_align_r() -> float:
        model.eval()
        feats = []
        with torch.no_grad():
            for start in range(0, val_b.n_rows, 2048):
                index = torch.arange(start, min(start + 2048, val_b.n_rows), device=device)
                feats.append(model.features(val_b.obs(index)))
        return align.correlation(torch.cat(feats), val_align_pairs, val_align_targets)

    # ---- train ---------------------------------------------------------------
    history, best_val, best_epoch = [], float("inf"), None
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        if args.freeze_encoder:
            model.extractor.eval()
        perm = torch.randperm(train_b.n_rows, device=device)
        running, t_epoch = 0.0, time.time()
        lam_epoch = ramp(epoch - 1) if ramp is not None else 0.0     # the ramp counts epochs from 0
        run_align = run_r = 0.0
        n_r = 0
        for start in range(0, train_b.n_rows, args.batch_size):
            index = perm[start:start + args.batch_size]
            if align is None:
                logits = model(train_b.obs(index))
                loss = belief_loss(logits, train_b, index, target)
            else:
                # the same forward, split so the features feed the alignment term too
                obs = train_b.obs(index)
                feats = model.features(obs)
                loss = belief_loss(model.head(feats), train_b, index, target)
                a_loss, r, _ = align.term(feats, obs["particles"] / scale, obs["weights"])
                loss = loss + lam_epoch * a_loss
                run_align += float(a_loss.detach()) * len(index)
                if r is not None:
                    run_r += float(r.detach()) * len(index)
                    n_r += len(index)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(index)
        metrics = evaluate_belief_head(model, val_b, target, 2048, transient_max_step)
        metrics.update(epoch=epoch, train_loss=running / train_b.n_rows,
                       lr=scheduler.get_last_lr()[0], seconds=time.time() - t_epoch)
        if align is not None:
            # train_loss above includes lambda * align (the Trainer's epoch loss does the same)
            metrics.update(align=run_align / train_b.n_rows, align_r=(run_r / n_r if n_r else float("nan")),
                           align_lambda=lam_epoch, val_align_r=_val_align_r())
        history.append(metrics)
        flag = ""
        if metrics["loss"] < best_val:
            best_val, best_epoch = metrics["loss"], epoch
            save_belief_checkpoint(model, checkpoint_dir / "checkpoint_best.pt", args, epoch, metrics, geometry,
                                   alignment=None if align_record is None else
                                   {**align_record, "lambda_at_best_epoch": lam_epoch,
                                    "val_align_r_at_best_epoch": metrics["val_align_r"]})
            flag = "  *best*"
        print(f"epoch {epoch:3d} | train {metrics['train_loss']:.4f} | val {metrics['loss']:.4f} | "
              f"head mode acc tr/st {metrics['head_mode_acc']['transient']:.3f}/"
              f"{metrics['head_mode_acc']['steady']:.3f} | "
              f"s* acc {metrics['head_true_state_acc']['transient']:.3f}/"
              f"{metrics['head_true_state_acc']['steady']:.3f} | "
              f"feat std {metrics['feature_abs_std']:.2e} | eff.rank {metrics['feature_eff_rank']:.1f} | "
              f"{metrics['seconds']:.0f}s{flag}", flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
    if align_record is not None and best_epoch is not None:
        # Model selection is by the objective's validation loss, blind to alignment (the Trainer's
        # rule); say so when the best epoch predates the end of the ramp.
        align_record["lambda_at_best_epoch"] = float(ramp(best_epoch - 1))
        align_record["val_align_r_at_best_epoch"] = float(history[best_epoch - 1]["val_align_r"])
        if align_record["lambda_at_best_epoch"] < align_lambda:
            print(f"WARNING: best-by-val-loss epoch {best_epoch} has align_lambda="
                  f"{align_record['lambda_at_best_epoch']:.3f} < target {align_lambda}; "
                  "checkpoint_best.pt is NOT fully aligned")
    save_belief_checkpoint(model, checkpoint_dir / "checkpoint_last.pt", args, args.num_epochs, history[-1], geometry,
                           alignment=align_record)
    print(f"best val loss {best_val:.4f}; checkpoints in {checkpoint_dir}")

    checkpoints = {"best": checkpoint_dir / "checkpoint_best.pt", "last": checkpoint_dir / "checkpoint_last.pt"}
    return PretrainResult(run_dir=run_dir, rl_checkpoint=checkpoints["best"], checkpoints=checkpoints,
                          summary={"best_val_loss": best_val, "best_epoch": best_epoch})


def _belief_report(args, ctx: PretrainContext, result: PretrainResult) -> None:
    """The end-of-run probe of `3_pretrain_st_belief.py`, moved: the FROZEN best-checkpoint
    latent under the mode-readout protocol (geometry, the trained head's accuracy, and grouped-CV
    readouts against EXACT and GAUSS2), written to probe_results.json. `--skip_probe` skips it."""
    if args.skip_probe:
        return
    run_dir = Path(result.run_dir)
    n_states = int(args.num_particles)
    scale = float(args.arena_scale)
    space = _belief_space(n_states)

    # ---- probe the FROZEN latent the way the readout probe does ----------------
    print("\nprobing the frozen best-checkpoint latent with the mode-readout protocol ...")
    best_path = Path(result.checkpoints["best"])
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    probe_extractor = build_extractor(args, space, scale, pretrained_path=str(best_path))
    probe_extractor.eval()
    head = nn.Linear(probe_extractor.features_dim - 1, n_states)
    head.load_state_dict(best["head_state_dict"])
    enc_label = f"{args.encoder.upper()}_BELIEF"

    data = collect_rollouts(args.variant, args.probe_episodes, args.probe_seed)
    st_feats = _run_extractor(probe_extractor, data)                 # [R, 64] float64
    with torch.no_grad():
        head_pred = head(torch.from_numpy(st_feats).float()).argmax(dim=-1).numpy() + 1

    weights = data["weights"].astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    x = data["particles"].astype(np.float64)[:, :, 0] / scale
    mean = (weights * x).sum(axis=1)
    var = (weights * (x - mean[:, None]) ** 2).sum(axis=1)
    feature_sets = {"EXACT": weights, "GAUSS2": np.c_[mean, var], enc_label: st_feats}

    results = {"geometry": {}, "head": {}, "targets": {}}
    print(f"\n{'encoding':<12} {'width':>6} {'abs.std':>10} {'eff.rank':>9} {'|r(PC1,mean)|':>14}")
    for name, feats in feature_sets.items():
        g = geometry(feats, mean)
        results["geometry"][name] = g
        print(f"{name:<12} {feats.shape[1]:>6} {g['abs_std']:>10.2e} {g['eff_rank']:>9.2f} "
              f"{g['pc1_corr_mean']:>14.5f}")

    for target_label, truth in (("B_posterior_mode", data["mode"]),
                                ("A_true_state", data["true_state"])):
        acc = _split_accuracy(head_pred, truth, data["step"])
        results["head"][target_label] = acc
        print(f"\ntrained linear head on {enc_label} -> {target_label}: "
              f"tr {acc['transient']:.3f} / st {acc['steady']:.3f}")

    for target_label, truth in (("B_posterior_mode", data["mode"]),
                                ("A_true_state", data["true_state"])):
        results["targets"][target_label] = {}
        print(f"\n=== target {target_label}  (50-way, chance 0.020; GroupKFold {args.probe_splits}) ===")
        print(f"{'encoding':<12} {'logreg raw':>16} {'logreg z':>16} {'mlp raw':>16} {'mlp z':>16}")
        for name, feats in feature_sets.items():
            row, cells = {}, []
            for classifier in ("logreg", "mlp"):
                for standardise in (False, True):
                    preds = _fit_predict(feats, truth, data["group"], args.probe_splits,
                                         classifier, standardise, args.seed)
                    acc = _split_accuracy(preds, truth, data["step"])
                    row[f"{classifier}_{'z' if standardise else 'raw'}"] = acc
                    cells.append(f"{acc['transient']:.3f} / {acc['steady']:.3f}")
            results["targets"][target_label][name] = row
            print(f"{name:<12} " + " ".join(f"{c:>16}" for c in cells), flush=True)

    (run_dir / "probe_results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {run_dir / 'probe_results.json'}")


def _exact_posterior_objective(target: str, description: str) -> Objective:
    """One of the three exact-posterior objectives; they share every hook and differ in the
    loss target (`belief_loss`) and in the run folder's name."""
    return Objective(
        name=target,
        description=description,
        add_arguments=_belief_add_arguments,
        run=lambda args, ctx: _belief_run(args, ctx, target),
        run_name=lambda args, now: _belief_run_name(args, now, target),
        locate=_belief_locate,
        resolve_arguments=_belief_resolve_arguments,
        prepare=_belief_prepare,
        report=_belief_report,
        default_experiment_name=lambda encoder_name: f"{encoder_name}_belief_pretrain",
    )


#: Declared on the Domain record below; `rl/pretrain.py --domain odd_even` offers them next
#: to the generic `reconstruction`, and runs `belief_kl` when --objective is omitted.
EXACT_POSTERIOR_OBJECTIVES = {
    "belief_kl": _exact_posterior_objective(
        "belief_kl",
        "encoder + linear head -> the env's exact posterior; soft cross-entropy "
        "(= KL up to a constant); asks for the whole belief (Odd-Even's default)"),
    "mode_ce": _exact_posterior_objective(
        "mode_ce",
        "encoder + linear head -> the posterior argmax (target B); hard cross-entropy"),
    "state_ce": _exact_posterior_objective(
        "state_ce",
        "encoder + linear head -> the true state (target A); hard cross-entropy -- the Bayes "
        "predictor of s* is the mode, so a noisier mode_ce"),
}


# ---------------------------------------------------------------------------
# Dataset collection (moved from experiments/odd_even/2_collect_pf_dataset.py, batch 7.4,
# 2026-09-13; that script is now an entry point of rl/collect.py and re-exports these names).
# The shared loop lives in rl/collect.py; what is here is what only Odd-Even knows.
#
# WHAT THIS DOMAIN DOES *NOT* HAVE, and why. The Ant-Tag collection carries a locomotion
# policy, a pursuit-versus-random action mix, a visibility radius and spread thresholds in
# arena units. None of it applies: on this env the action is a PREDICTION, so it changes
# neither the hidden state nor the observation stream. The belief trajectory is a function of
# the observations alone, and random actions therefore give the correct, unbiased belief
# distribution. There is no exploration policy to design and step 1 of the pipeline drops out.
#
# WHAT REPLACES THE SPREAD REBALANCING. The belief here locks on by about step 21 of 50, so
# uniform sampling over an episode leaves roughly 60% of snapshots one-hot -- the regime where
# every encoder is equivalent, and so the regime an encoder comparison learns nothing from.
# Rebalancing is by STEP INDEX (or, with --rebalance_by ess, by effective sample size), never
# by "spread in arena units": on a 1-D integer state the spread of a near-one-hot belief and
# of a two-mode belief can coincide, while the step index is exactly the axis along which the
# belief sharpens. Bodies unchanged.
# ---------------------------------------------------------------------------

#: Steps below this still carry a genuinely BROAD belief. Measured at n=50:
#: the share of snapshots with effective sample size above 3 (of 50) is 0.90,
#: 0.95, 0.95 at steps 0, 1, 2 and collapses to 0.20, 0.05, 0.00 at steps
#: 3, 4, 5. The belief therefore sharpens several times faster than the
#: max(belief) > 0.9 criterion suggests, so the 'early' bucket has to be
#: narrow to hold anything an encoder can distinguish.
EARLY_STEP = 3


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
    """Rebalance the snapshot mix across the belief's own sharpening axis (see
    `_rebalance_index`, which holds the body since batch 10.5 and returns the kept rows)."""
    idx = _rebalance_index(weights, steps, by=by, early_step=early_step, collapse_step=collapse_step,
                           diffuse_ess=diffuse_ess, collapsed_ess=collapsed_ess, early_frac=early_frac,
                           mid_frac=mid_frac, late_frac=late_frac, seed=seed)
    return particles[idx], weights[idx], steps[idx]


def _rebalance_index(
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
) -> np.ndarray:
    """The rows `_rebalance` keeps, in output order (batch 10.5, 2026-09-14: returned as an index so
    the collector can apply the same selection to the per-snapshot labels).

    Rebalance the snapshot mix across the belief's own sharpening axis.

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

    n_total = len(weights)
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
    return all_idx


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


# -- the Collection hooks (rl/domains/base.py::Collection) -----------------------------------

def _collect_add_arguments(parser) -> None:
    """Odd-Even's own collection flags (help texts from the script); the shared ones
    (--num_episodes, --timesteps, --num_particles, --seed, --max_snapshots, --no_rebalance,
    --output) are rl/collect.py's."""
    parser.add_argument(
        "--particle_filter", type=str, default=None,
        choices=sorted(PARTICLE_FILTERS),
        help="Override the variant's filter. The bootstrap filter is a "
             "deliberate arm (a lossier belief on the same env), not a "
             "different domain.")
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


def _collect_resolve_arguments(parser, args, domain) -> dict:
    """The variant's filter (or the override), and the script's defaults: steps per episode =
    the registered cap, set size = the state count (an EXACT belief, one particle per state)."""
    resolved = variants.resolve(args.variant)
    particle_filter_class = resolve_particle_filter(args.variant, args.particle_filter)
    if args.timesteps is None:
        args.timesteps = variants.episode_cap(args.variant)
    if args.num_particles is None:
        args.num_particles = resolved.n_dist_size
    print(f"Variant: {args.variant} | env: {resolved.env_id} | "
          f"filter: {particle_filter_class.__name__} | "
          f"n={resolved.n_dist_size} | cap={variants.episode_cap(args.variant)}")
    return {"env_id": resolved.env_id, "particle_filter_class": particle_filter_class}


def _collect_prepare(args, options) -> dict:
    """The per-run state: the episode being rolled (for the `episode` label; batch 10.5)."""
    return {"episode": None}


def _collect_make_env(args, options, state):
    return make_odd_even_belief_env(
        num_particles=args.num_particles,
        rank=0,
        seed=args.seed,
        variant=args.variant,
        particle_filter_class=options["particle_filter_class"],
    )()


def _collect_begin_episode(args, options, state, env, episode):
    # seed + episode: on this env the hidden state is drawn at reset, so
    # THE SEED IS THE EPISODE. A constant reset seed would collect one
    # episode num_episodes times (PITFALLS.md section 2). Actions are drawn
    # uniformly: the action is a prediction and does not move the state or the
    # observation stream, so the belief distribution collected under random
    # actions is the same one any policy would induce.
    state["episode"] = int(episode)
    return {"seed": args.seed + episode}, (lambda obs: env.action_space.sample())


def _collect_snapshot_extras(args, options, state, env, obs, step_index) -> dict:
    """Per-snapshot LABELS read off the live env (batch 10.5, decision 6): the hidden state, the
    posterior's argmax and the episode index -- what `collect_rollouts` reads off `info` -- so the
    exact-posterior objectives can train on this file (on the exact-support filter the row's
    weights ARE the posterior). `optimal_prediction` comes from the env's float64 posterior, not
    from the float32 weights in the row, so a tie broken differently by the cast cannot mislabel it."""
    base = env.unwrapped
    return {"true_state": np.int64(base.true_state),
            "optimal_prediction": np.int64(base.get_optimal_prediction()),
            "episode": np.int64(state["episode"])}


def _collect_report(args, options, particles, weights, steps, stage) -> None:
    if stage == "raw":
        print(f"\nRaw snapshots: {len(particles)}")
        _report_distribution(particles, weights, steps, args.collapse_step)
        return
    if not args.no_rebalance:
        print("\nPost-rebalance:")
        _report_distribution(particles, weights, steps, args.collapse_step)
    print(f"\n  particle_scale (recorded for pretraining): "
          f"{variants.state_scale(args.variant)}")
    print(f"  particle_centre: {variants.state_centre(args.variant)}")


def _collect_rebalance(args, options, particles, weights, steps):
    """Returns the kept rows AND their index (a 4-tuple), so the collector applies the same
    selection to the per-snapshot labels (batch 10.5)."""
    print("\nRebalancing...")
    idx = _rebalance_index(
        weights, steps,
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
    return particles[idx], weights[idx], steps[idx], idx


def _collect_metadata_extras(args, options, particles, weights, steps) -> dict:
    """What this domain records beyond the shared facts (the script's `_build_metadata`)."""
    resolved = variants.resolve(args.variant)
    return {
        "n_dist_size": resolved.n_dist_size,
        "episode_cap": variants.episode_cap(args.variant),
        # The centre the RL env subtracts before dividing by particle_scale. Recorded so the
        # two halves of the mapping cannot drift apart (PITFALLS.md section 4).
        "particle_centre": variants.state_centre(args.variant),
        "step_index_min": int(steps.min()) if len(steps) else None,
        "step_index_max": int(steps.max()) if len(steps) else None,
    }


def _collect_extra_arrays(args, options, particles, weights, steps) -> dict:
    return {
        # Top-level so get_dataset() reads it without parsing the metadata.
        # dataset.py applies (x - centre) / scale, matching the RL wrapper.
        "particle_centre": np.float32(variants.state_centre(args.variant)),
        # Per-row step index, so the transient/steady mix can be checked or
        # re-split downstream (only min/max used to be recorded).
        "steps": steps.astype(np.int32),
    }


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
    from set_transformer.rl import collect as _collect   # imported here: rl/collect imports the domains

    matches = [name for name, v in VARIANTS.items()
               if v.n_dist_size == int(ns)]
    if not matches:
        raise ValueError(
            f"No registered variant with n_dist_size={ns}; have "
            + ", ".join(f"{n}(n={v.n_dist_size})"
                        for n, v in VARIANTS.items()))
    # The shortest cap among the matches: a test wants the cheapest env whose
    # state range is the one it asked for.
    variant = min(matches, key=variants.episode_cap)

    parser = _collect.build_parser(ODD_EVEN, selectors=False)
    args = parser.parse_args(["--variant", variant, "--num_episodes", str(num_episodes),
                              "--timesteps", str(timesteps), "--num_particles", str(num_particles),
                              "--seed", str(seed)])
    options = ODD_EVEN_COLLECTION.resolve_arguments(parser, args, ODD_EVEN)
    particles, weights, steps = _collect.collect_arrays(ODD_EVEN, args, options, progress=False)
    particles, weights, steps = _rebalance(particles, weights, steps,
                                           by="step", seed=seed)
    metadata = _collect.build_metadata(ODD_EVEN, args, options, particles, weights, steps,
                                       record_args=False)
    return particles, weights, metadata


ODD_EVEN_COLLECTION = Collection(
    add_arguments=_collect_add_arguments,
    # The script's defaults for the shared flags; --timesteps and --num_particles are resolved
    # from the variant in _collect_resolve_arguments.
    defaults={"seed": 0, "num_episodes": 200},
    resolve_arguments=_collect_resolve_arguments,
    particle_scale=lambda args, options: variants.state_scale(args.variant),
    particle_centre=lambda args, options: variants.state_centre(args.variant),
    prepare=_collect_prepare,
    make_env=_collect_make_env,
    begin_episode=_collect_begin_episode,
    snapshot_extras=_collect_snapshot_extras,
    report=_collect_report,
    rebalance=_collect_rebalance,
    metadata_extras=_collect_metadata_extras,
    extra_arrays=_collect_extra_arrays,
    progress_desc="Collecting episodes",
)


# ---------------------------------------------------------------------------
# The Domain description (change 4, 2026-09-12): what the shared trainer needs from Odd-Even
# ---------------------------------------------------------------------------


def _add_arguments(parser) -> None:
    """The one flag that belongs to the Odd-Even problem (help text from 4_train_rl_cgf.py)."""
    parser.add_argument(
        "--particle_filter", type=str, default=None,
        choices=sorted(PARTICLE_FILTERS),
        help="Override the variant's filter. The bootstrap filter is a "
             "deliberate arm (a lossier belief on the same env).")


def _make_env(variant: str, *, num_particles: int, particle_filter_class: type, seed: int,
              rank: int, monitor_dir: str | None, training: bool, options: dict):
    """One worker's env through `make_odd_even_belief_env`. The eval env is the training env
    at another rank without a monitor directory, as `4_train_rl_cgf.build_envs` built it;
    `training` and `options` are unused here (no shaping, no curriculum)."""
    return make_odd_even_belief_env(
        num_particles=num_particles, rank=rank, seed=seed, monitor_dir=monitor_dir,
        variant=variant, particle_filter_class=particle_filter_class)


def _cgf_t_init_max_default(args):
    """Odd-Even's rule for a ``--t_init_max`` left unset (4_train_rl_cgf.resolve_t_init_max):
    40 in tanh mode, t_clamp in clamp mode -- whatever the init mode."""
    return 40.0 if args.t_param != "clamp" else float(args.t_clamp)


ODD_EVEN = Domain(
    name="odd_even",
    particle_dim=1,
    default_sinkhorn_blur=0.02,      # 2026-09-19: the recipes' recorded blur
    default_variant="oe50",
    variants=VARIANTS,
    resolve=resolve,
    episode_cap=episode_cap,
    run_subdir=run_subdir,
    add_variant_argument=add_variant_argument,
    print_variants=print_variants,
    make_env=_make_env,
    make_vec_env_from_fns=make_vec_env_from_fns,
    make_vec_normalize=make_vec_normalize,
    particle_filter=lambda args: resolve_particle_filter(args.variant, args.particle_filter),
    add_arguments=_add_arguments,
    resolve_arguments=lambda parser, args: {},
    schedules=lambda args: (),
    run_config_extras=lambda args: dict(n_dist_size=resolve(args.variant).n_dist_size,
                                        episode_cap=episode_cap(args.variant)),
    default_num_particles=lambda variant: resolve(variant).n_dist_size,
    default_arena_scale=state_scale,
    default_device="cpu",
    default_total_timesteps=1_000_000,
    encoder_defaults={
        # The tanh-50 / running-norm / spread_1d recipe of 2026-09-05/06 (module docstring of
        # experiments/odd_even/4_train_rl_cgf.py); t_bound is a plain default here, not a
        # registry lookup, so no `t_bound_default` hook.
        "cgf": dict(t_param="tanh", t_bound=50.0, t_init_mode="spread_1d",
                    feature_norm="running",
                    t_init_max_default=_cgf_t_init_max_default),
        # The small ClusterHunt-sized encoder adopted 2026-09-05 (~109k parameters).
        "st": dict(num_inds=16, dim_hidden=64, num_post_sab=2),
    },
    # The ST arm gets the collapse sentinel on top of the shared feature logging, in that
    # order (the sentinel records under its own keys so it cannot overwrite the shared ones).
    encoder_callbacks=lambda encoder_name: (
        [OddEvenSTFeatureSentinel()] if encoder_name == "st" else []),
    # No success on this domain: transient / steady split against the Bayes oracle and the
    # previous-observation baseline, on the same episodes, re-seeded per episode.
    evaluation=Evaluation(add_arguments=_eval_add_arguments, default_n_episodes=400,
                          reseed_per_episode=True, references=_eval_references,
                          references_only=lambda args: bool(args.baselines_only),
                          report=_eval_report),
    # Supervised pretraining on the exact posterior (3_pretrain_st_belief.py's three
    # objectives, batch 7.3); the generic reconstruction objective needs no declaration.
    # `belief_kl` is what `rl/pretrain.py --domain odd_even` runs when --objective is omitted.
    pretraining=Pretraining(objectives=EXACT_POSTERIOR_OBJECTIVES, default_objective="belief_kl"),
    # Step 2 of the pipeline: random actions (the action is a prediction), rebalanced by step
    # index or effective sample size (batch 7.4).
    collection=ODD_EVEN_COLLECTION,
)
