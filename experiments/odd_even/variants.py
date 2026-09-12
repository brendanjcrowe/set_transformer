"""Registry of Odd-Even POMDP variants -- FORWARDING FILE (since 2026-09-12).

The registry -- the `Variant` description, the `VARIANTS` and `PARTICLE_FILTERS` tables and
the helpers (`resolve`, `episode_cap`, `resolve_particle_filter`, `state_scale`,
`state_centre`, `run_subdir`, `warn_if_override_contradicts`, `add_variant_argument`,
`print_variants`) -- lives in ``set_transformer.rl.domains.odd_even`` (change 1b of the
harness centralisation; that module's docstring explains why the registry exists). This
file re-exports it for the scripts, diagnostics, viz and tests that load it by path through
``_sibling.load("variants")``.

Run directly it prints the registry, as before:  python3 variants.py
"""

import sys
from pathlib import Path

# Package root on sys.path so `set_transformer` resolves to the package and not to the
# submodule directory of the same name (PITFALLS.md section 7; needed when this file is
# run directly). parents[2] is correct at experiments/<domain>/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from set_transformer.rl.domains.odd_even import (  # noqa: E402,F401
    PARTICLE_FILTERS,
    VARIANTS,
    Variant,
    add_variant_argument,
    episode_cap,
    print_variants,
    resolve,
    resolve_particle_filter,
    run_subdir,
    state_centre,
    state_scale,
    warn_if_override_contradicts,
)

__all__ = [
    "PARTICLE_FILTERS",
    "VARIANTS",
    "Variant",
    "add_variant_argument",
    "episode_cap",
    "print_variants",
    "resolve",
    "resolve_particle_filter",
    "run_subdir",
    "state_centre",
    "state_scale",
    "warn_if_override_contradicts",
]


if __name__ == "__main__":
    print_variants()
