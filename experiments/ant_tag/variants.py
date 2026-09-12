"""Registry of Ant-Tag env variants -- FORWARDING FILE (since 2026-09-12).

The registry -- the `Variant` description, the `VARIANTS` table, `EVADING`, and the helpers
(`resolve`, `episode_cap`, `arena_scale`, `cgf_t_bound`, `run_subdir`, `resolve_schedule`,
`warn_if_not_evading`, `warn_if_override_contradicts`, `add_variant_argument`,
`print_variants`) -- lives in ``set_transformer.rl.domains.ant_tag`` (change 1b of the
harness centralisation; that module's docstring explains why the registry exists). This
file re-exports it under the historical flat name that every script in this directory,
eval_scripts/, diagnostics/ and the tests use (``import variants``).

Run directly it prints the registry, as before:  python3 variants.py
"""

import sys
from pathlib import Path

# Package root on sys.path so `set_transformer` resolves to the package and not to the
# submodule directory of the same name (needed when this file is run directly).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from set_transformer.rl.domains.ant_tag import (  # noqa: E402,F401
    CGF_TILT_TARGET,
    EVADING,
    VARIANTS,
    Variant,
    add_variant_argument,
    arena_scale,
    cgf_t_bound,
    episode_cap,
    print_variants,
    resolve,
    resolve_schedule,
    run_subdir,
    warn_if_not_evading,
    warn_if_override_contradicts,
)

__all__ = [
    "CGF_TILT_TARGET",
    "EVADING",
    "VARIANTS",
    "Variant",
    "add_variant_argument",
    "arena_scale",
    "cgf_t_bound",
    "episode_cap",
    "print_variants",
    "resolve",
    "resolve_schedule",
    "run_subdir",
    "warn_if_not_evading",
    "warn_if_override_contradicts",
]


if __name__ == "__main__":
    print_variants()
