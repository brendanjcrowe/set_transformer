"""Odd-Even domain for the RL harness: variant registry.

Moved here 2026-09-12 from ``experiments/odd_even/variants.py`` (change 1b of the harness
centralisation, ``refactor_plans.md`` in the parent repo); the block below is byte-identical
to the original, and that file now re-exports these names for the scripts, diagnostics,
viz and tests that load it through ``_sibling.load("variants")``. The env wrappers and
factory (``odd_even_belief_env.py``) follow in change 1c.

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

``import pdomains`` below registers the ``pdomains-odd-even-*`` env ids the registry names.
"""

from dataclasses import dataclass

import gymnasium as gym

import pdomains  # noqa: F401 - registers the pdomains-odd-even-* envs
from set_transformer.rl.particle_filters.odd_even import (
    OddEvenBootstrapParticleFilter,
    OddEvenExactSupportParticleFilter,
)


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


