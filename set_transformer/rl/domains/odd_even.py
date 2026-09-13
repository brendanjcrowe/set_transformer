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

``import pdomains`` below registers the ``pdomains-odd-even-*`` env ids. The factory thunk is
pickled by reference to this module when SubprocVecEnv starts workers; the child's import of
it (and the ``import pdomains`` inside the thunk) registers the envs there too.
"""

import os
from dataclasses import dataclass
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

import pdomains  # noqa: F401 - registers the pdomains-odd-even-* envs
from set_transformer.rl.particle_filters.odd_even import (
    OddEvenBootstrapParticleFilter,
    OddEvenExactSupportParticleFilter,
)
from set_transformer.rl.wrappers.particle_filter import (
    PFDictWithWeightsObservationWrapper,
)
from set_transformer.rl.domains.base import Domain, Evaluation


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
)
