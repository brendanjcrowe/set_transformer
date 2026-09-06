"""Registry of Ant-Tag env variants: which particle filter goes with which env.

Every script in this directory needs the same three facts about a variant --
the gym env id, the matching particle filter, and where its runs live. Those
facts used to be copy-pasted into a one-line wrapper file per (variant x
script) pair: 19 files, each nine statements long, and 49 of them possible.
Adding a variant meant writing seven more.

They are stated once here instead, and every script takes `--variant`.

Two things this makes structural rather than remembered:

1. **The env/filter pairing.** A filter mirrors its env's target motion model;
   if they drift apart, belief propagation silently diverges from the env with
   no error anywhere. The pairing is now a single line, not an assertion
   repeated in nineteen files where one can be missed.

2. **The episode cap.** Gymnasium already knows it -- `max_episode_steps` is in
   the registration -- so `episode_cap()` reads it rather than asking the user
   for `--max_steps`. Passing the wrong value used to make every timeout count
   as a successful tag (a `dens` eval left at the 400 default reads ~100%
   success). That failure mode is now unreachable by default.

Adding a variant: one entry here, and it appears in every script at once.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym

import pdomains  # noqa: F401 - registers the pdomains-ant-tag-* envs
from set_transformer.rl.particle_filters.ant_tag import (
    AntTagParticleFilter,
    CounterweightedDenAntTagParticleFilter,
    GhostAntTagParticleFilter,
    SmartAntTagParticleFilter,
    TwinDenAntTagParticleFilter,
)


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
EVADING = {"smart", "smart_hard", "ghost", "dens", "cdens", "cdens_hard",
           "cdens_terminal", "cdens_nospook"}


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


if __name__ == "__main__":
    print_variants()
