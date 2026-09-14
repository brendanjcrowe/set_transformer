"""The ``car_flag`` domain: Car-Flag on the shared RL harness (plan section 10, batch 10.2,
2026-09-14).

``pdomains-car-flag-v0`` (``pdomains/car_flag.py::CarEnv``) is a one-dimensional car on
``[-1.1, 1.1]``. Heaven is at one end and hell at the other; which end is drawn 50/50 at
reset and never moves. The observation is ``[position, velocity, direction]``; ``direction``
is 0 everywhere except inside the priest zone ``[0.3, 0.7]``, where it equals the heaven
side (+1 right, -1 left). Reaching either end terminates the episode; the cap is 160 steps.

**This domain is the campaign's CONTROL, not a benchmark**
(``experiments/benchmark/ENV_VIABILITY.md``). The hidden state is one bit, so the filter
(:class:`~set_transformer.rl.particle_filters.car_flag.CarFlagParticleFilter`, particles in
``{+1, -1}``) has exactly THREE reachable beliefs: 50/50, all +1, all -1. The mean of the
particle set is a sufficient statistic, so no encoder can beat the Gaussian summary here in
principle; the flat result across arms is the correct answer. For the same reason there is
nothing to pretrain on -- three point clouds -- and this domain declares no collector and no
objective of its own (``collection=None``; the driver recipe has no pretraining conditions).

**Reward (decision 10c-1).** Under the stock reward (-1 per step, 0 / +1 at heaven, -5 at
hell) the priest detour costs about 27 steps and the information is worth about 3, so the
optimal policy ignores the belief and drives to one end blind. The campaign variant ``shaped``
puts Brendan's :class:`~set_transformer.rl.benchmark.envs.CarFlagRewardWrapper` below the
filter (step -0.01, heaven +1, hell -1; measured: information is worth +0.83 over gambling).
``stock`` keeps the registered reward, for the record. Observations, dynamics, termination and
the belief are identical in both.

**What the agent sees.** The full 3-vector as the ``obs`` key (the priest's reading IS an
observation; the belief adds its memory after the car leaves the zone), the filter's particles
(scale 1, centre 0) and their weights. The filter is driven positionally -- the shared wrapper
calls ``update(obs)`` with the plain observation when no mapper is given -- exactly as the
filter was written to be driven.

**Evaluation.** :func:`_eval_report` counts heaven / hell / timeout from the outcome the
:class:`CarFlagOutcomeWrapper` writes into the final ``info``; success = heaven. 300 episodes,
re-seeded per episode because the heaven side is drawn at reset.

**Encoder defaults (decision 10c-2).** CGF ``tanh`` with ``t_bound`` 3: the particles are
+-1 at scale 1 and the log-MGF of a +-1 variable is flat past ``|t|`` of about 3; ``spread_1d``
init up to 0.8 x bound (the Ant-Tag / hunt rule). ST 16 / 64 / 2 post-SABs. 100 particles
(the benchmark's setting; the filter has no natural size). Horizon 1M steps (decision 10c-3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pdomains  # noqa: F401 - registers the pdomains-* env ids
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from set_transformer.rl.benchmark.envs import (
    CAR_FLAG_HEAVEN_REWARD,
    CAR_FLAG_HELL_REWARD,
    CAR_FLAG_STEP_PENALTY,
    CarFlagRewardWrapper,
)
from set_transformer.rl.domains.base import Domain, Evaluation
from set_transformer.rl.particle_filters.car_flag import CarFlagParticleFilter
from set_transformer.rl.wrappers.particle_filter import PFDictWithWeightsObservationWrapper

ENV_ID = "pdomains-car-flag-v0"
#: Particles are +-1: raw units ARE the encoder's units.
PARTICLE_SCALE = 1.0
#: The benchmark's particle count (``rl/benchmark/registry.py``); the filter has no natural size.
DEFAULT_NUM_PARTICLES = 100
#: CGF probe bound for +-1 particles (decision 10c-2).
CGF_T_BOUND = 3.0
#: The three episode outcomes the report counts.
OUTCOMES = ("heaven", "hell", "timeout")


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """The Car-Flag env plus its filter and which reward the run trains on."""

    env_id: str
    particle_filter: type
    #: ``"shaped"``: :class:`CarFlagRewardWrapper` below the filter; ``"stock"``: the
    #: registered reward. Nothing else differs.
    reward: str
    #: Printed by --list_variants.
    notes: str = ""


VARIANTS: dict[str, Variant] = {
    "shaped": Variant(
        env_id=ENV_ID,
        particle_filter=CarFlagParticleFilter,
        reward="shaped",
        notes=(f"THE campaign variant (control). Reward step {CAR_FLAG_STEP_PENALTY}, heaven "
               f"{CAR_FLAG_HEAVEN_REWARD:+g}, hell {CAR_FLAG_HELL_REWARD:+g}: information is "
               "worth its cost. Belief = one bit, three reachable states; no encoder can win."),
    ),
    "stock": Variant(
        env_id=ENV_ID,
        particle_filter=CarFlagParticleFilter,
        reward="stock",
        notes=("The registered reward (-1 per step, 0 / +1 at heaven, -5 at hell): the priest "
               "detour costs more than the information is worth, so the optimal policy ignores "
               "the belief. For the record only."),
    ),
}


def resolve(name: str) -> Variant:
    """Look up a variant, listing the valid names on a typo."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"Unknown variant {name!r}. Available: {sorted(VARIANTS)}") from None


def episode_cap(name: str) -> int:
    """The episode cap from the gym registration (160)."""
    spec = gym.spec(resolve(name).env_id)
    if spec.max_episode_steps is None:
        raise ValueError(f"{spec.id} registers no max_episode_steps; pass --max_steps")
    return int(spec.max_episode_steps)


def run_subdir(encoder: str, name: str) -> str:
    """The legacy-layout folder name for this (encoder, variant) pair."""
    resolve(name)
    return f"car_flag_{encoder}_{name}"


def add_variant_argument(parser, default: str = "shaped") -> None:
    parser.add_argument(
        "--variant", type=str, default=default, choices=sorted(VARIANTS),
        help=f"Car-Flag reward variant (default: {default}). 'shaped' is the campaign's; "
             "'stock' the registered reward.")
    parser.add_argument("--list_variants", action="store_true",
                        help="Print the variant registry and exit.")


def print_variants() -> None:
    for name, variant in VARIANTS.items():
        try:
            cap = episode_cap(name)
        except Exception:  # noqa: BLE001 - listing must not fail on one bad entry
            cap = "?"
        print(f"{name:8s} {variant.env_id:24s} cap={cap:<5} reward={variant.reward:7s} "
              f"{variant.particle_filter.__name__}")
        if variant.notes:
            print(f"{'':8s}   {variant.notes}")


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class CarFlagOutcomeWrapper(gym.Wrapper):
    """Write the episode's outcome into ``info`` on BOTH reward variants.

    The env terminates exactly when ``|position| >= 1``, at one of the two flags, and pays
    the outcome through the reward only. The evaluation script sees rewards and infos, so the
    outcome has to be in the info: ``outcome`` in ``heaven`` / ``hell`` (terminated) or
    ``timeout`` (truncated at the cap), ``reached_heaven`` and ``is_success`` (= heaven).
    Reading the heaven side here is not a leak: the env itself reads it to pay the terminal
    reward, and nothing reaches the observation. :class:`CarFlagRewardWrapper` writes the
    same ``reached_heaven`` on the shaped variant; the two agree by construction.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            position = float(np.asarray(obs).reshape(-1)[0])
            heaven = float(self.env.unwrapped.heaven_position)
            reached = bool(np.sign(position) == np.sign(heaven))
            info["reached_heaven"] = reached
            info["outcome"] = "heaven" if reached else "hell"
            info["is_success"] = reached
        elif truncated:
            info["outcome"] = "timeout"
            info["is_success"] = False
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------


def make_car_flag_belief_env(
    num_particles: int,
    rank: int = 0,
    seed: int = 0,
    monitor_dir: str | None = None,
    variant: str = "shaped",
    env_id: str | None = None,
    particle_filter_class: type | None = None,
):
    """Return a callable that builds one Car-Flag belief env.

    Stack, inside out: ``gym.make(env_id)`` (TimeLimit at the registered cap 160) ->
    :class:`CarFlagRewardWrapper` (``shaped`` only) -> :class:`CarFlagOutcomeWrapper` ->
    :class:`PFDictWithWeightsObservationWrapper` with the Car-Flag filter driven positionally
    (no mapper: the filter reads the priest's direction off the plain observation) -> Monitor.
    No curriculum, no shaping flags, no masking: the whole observation is legitimate.
    """
    resolved = resolve(variant)
    env_id = env_id or resolved.env_id
    particle_filter_class = particle_filter_class or resolved.particle_filter
    reward = resolved.reward

    def _init():
        # Registered HERE too, not only at module import: this closure is cloudpickled by
        # value into a SubprocVecEnv child, which never runs this module's top-level import.
        import pdomains  # noqa: F401,PLC0415

        env = gym.make(env_id)
        env.reset(seed=seed + rank)
        if reward == "shaped":
            env = CarFlagRewardWrapper(env)
        env = CarFlagOutcomeWrapper(env)
        env = PFDictWithWeightsObservationWrapper(
            env=env,
            particle_filter_class=particle_filter_class,
            particle_filter_kwargs={},
            num_particles=num_particles,
            pf_interaction_mapper=None,        # direction is in the observation: update(obs)
            obs_mask_indices=None,             # nothing to hide
            particle_filter_seed=seed + rank,
        )
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
    """Normalise the base obs (position, velocity, direction) and the reward, never the PF
    weights."""
    return VecNormalize(vec_env, training=training, norm_obs=True, norm_reward=norm_reward,
                        norm_obs_keys=["obs"])


# ---------------------------------------------------------------------------
# The domain's flags and env hook
# ---------------------------------------------------------------------------


def _add_arguments(parser) -> None:
    """Car-Flag has no flags of its own: no curriculum, no shaping, no masking."""


def _resolve_arguments(parser, args) -> dict:
    variant = resolve(args.variant)
    print(f"Car-Flag variant {args.variant!r}: reward {variant.reward} "
          + ("(step %g, heaven %+g, hell %+g)" % (CAR_FLAG_STEP_PENALTY, CAR_FLAG_HEAVEN_REWARD,
                                                    CAR_FLAG_HELL_REWARD)
             if variant.reward == "shaped" else "(registered: -1 / step, 0 or +1 heaven, -5 hell)"))
    return {}


def _run_config_extras(args) -> dict:
    variant = resolve(args.variant)
    extras = dict(episode_cap=episode_cap(args.variant), reward=variant.reward)
    if variant.reward == "shaped":
        extras["reward_constants"] = dict(step=CAR_FLAG_STEP_PENALTY, heaven=CAR_FLAG_HEAVEN_REWARD,
                                          hell=CAR_FLAG_HELL_REWARD)
    return extras


def _make_env(variant: str, *, num_particles: int, particle_filter_class: type, seed: int,
              rank: int, monitor_dir: str | None, training: bool, options: dict):
    """One worker's env. Training and eval envs are the same env: there is no curriculum and
    no shaping to switch off."""
    return make_car_flag_belief_env(
        num_particles=num_particles, rank=rank, seed=seed, monitor_dir=monitor_dir,
        variant=variant, particle_filter_class=particle_filter_class)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _eval_report(episodes, references, args, variant, cap) -> dict:
    """Heaven / hell / timeout from the outcome wrapper's final ``info``; success = heaven.

    An episode the rollout cut at ``--max_steps`` before the env's own end has no outcome key
    and counts as a timeout. Also printed: the mean length of the heaven episodes (a policy
    that visits the priest takes longer than one that gambles) and the mean reward.
    """
    n = len(episodes)
    rewards = np.array([episode.total_reward for episode in episodes], dtype=float)
    lengths = np.array([episode.length for episode in episodes], dtype=int)
    outcomes = [episode.final_info.get("outcome", "timeout") for episode in episodes]
    counts = {k: int(sum(o == k for o in outcomes)) for k in OUTCOMES}
    successes = counts["heaven"]
    summary = dict(
        reward=variant.reward, n_episodes=n, successes=successes,
        success_rate=float(successes / n) if n else float("nan"),
        outcome_counts=counts,
        **{f"outcome_{k}": (v / n if n else float("nan")) for k, v in counts.items()},
        mean_reward=float(rewards.mean()), std_reward=float(rewards.std()),
        mean_length=float(lengths.mean()), std_length=float(lengths.std()))
    print(f"\n=== Eval over {n} episodes (deterministic={args.deterministic}, "
          f"reward={variant.reward}) ===")
    print(f"Success (heaven): {successes}/{n} ({100 * summary['success_rate']:.1f}%)")
    print(f"Outcomes      : heaven {counts['heaven']}, hell {counts['hell']}, "
          f"timeout {counts['timeout']}")
    print(f"Mean reward   : {rewards.mean():.3f} ± {rewards.std():.3f}")
    print(f"Mean length   : {lengths.mean():.1f} ± {lengths.std():.1f} (cap {cap})")
    heaven = np.array([o == "heaven" for o in outcomes], dtype=bool)
    if successes:
        summary["heaven_mean_length"] = float(lengths[heaven].mean())
        print(f"When heaven   : mean_len={lengths[heaven].mean():.1f}")
    summary["episodes"] = [dict(reward=float(r), length=int(l), success=bool(s), outcome=o)
                           for r, l, s, o in zip(rewards, lengths, heaven, outcomes)]
    return summary


# ---------------------------------------------------------------------------
# Encoder defaults
# ---------------------------------------------------------------------------


def _cgf_t_init_max_default(args):
    """The Ant-Tag / hunt rule: the spread init tops out at 0.8 x t_bound; unset in clamp mode."""
    if args.t_param == "clamp":
        return None
    return 0.8 * float(args.t_bound)


CAR_FLAG = Domain(
    name="car_flag",
    particle_dim=1,
    default_variant="shaped",
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
    default_arena_scale=lambda variant: PARTICLE_SCALE,
    default_device="cpu",
    default_total_timesteps=1_000_000,
    encoder_defaults={
        # tanh with bound 3 for +-1 particles (decision 10c-2); spread_1d init to 0.8 x bound;
        # raw features. t_bound is a plain default, so no `t_bound_default` hook.
        "cgf": dict(t_param="tanh", t_bound=CGF_T_BOUND, t_init_mode="spread_1d",
                    feature_norm="none", t_init_max_default=_cgf_t_init_max_default),
        # The small ST (16 inducing points, hidden 64, two post-PMA SABs; ~109k parameters).
        "st": dict(num_inds=16, dim_hidden=64, num_post_sab=2),
    },
    evaluation=Evaluation(default_n_episodes=300, reseed_per_episode=True, report=_eval_report),
    # No collector and no objective of its own: three reachable beliefs, nothing to pretrain on
    # (ENV_VIABILITY.md; decision 10c-8). The generic reconstruction objective is still offered by
    # the pretraining door, as for every domain, but no recipe uses it here.
    collection=None,
)
