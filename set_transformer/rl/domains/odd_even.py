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
