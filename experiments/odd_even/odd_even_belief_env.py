"""The Odd-Even belief env -- FORWARDING FILE (since 2026-09-12).

The env stack and its factory (``odd_even_pf_interaction_mapper``,
``StepIndexObservationWrapper``, ``OddEvenPFDictWrapper``, ``ParticleCentringWrapper``,
``make_odd_even_belief_env``, ``make_vec_env_from_fns``, ``make_vec_normalize``) live in
``set_transformer.rl.domains.odd_even`` (change 1c of the harness centralisation; that
module's docstring carries the reasoning, in particular THE OBSERVATION SPLIT: the agent's
``obs`` is the normalized step index, the filter reads ``info["observations"]``). This file
re-exports the same objects for the scripts, diagnostics, viz and tests that load it by path
through ``_sibling.load("odd_even_belief_env")``.
"""

import sys
from pathlib import Path

# Package root on sys.path so `set_transformer` resolves to the package and not to the
# submodule directory of the same name (PITFALLS.md section 7).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from set_transformer.rl.domains.odd_even import (  # noqa: E402,F401
    OddEvenPFDictWrapper,
    ParticleCentringWrapper,
    StepIndexObservationWrapper,
    make_odd_even_belief_env,
    make_vec_env_from_fns,
    make_vec_normalize,
    odd_even_pf_interaction_mapper,
)

__all__ = [
    "OddEvenPFDictWrapper",
    "ParticleCentringWrapper",
    "StepIndexObservationWrapper",
    "make_odd_even_belief_env",
    "make_vec_env_from_fns",
    "make_vec_normalize",
    "odd_even_pf_interaction_mapper",
]
