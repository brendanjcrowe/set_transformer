"""Multimodal Search -- forwarding import (batch 9.1, 2026-09-13).

The env moved VERBATIM to ``pomdp-domains`` (``pdomains/multimodal_search.py``, batch 9.0b) and
is registered there as ``pdomains-multimodal-search-v0`` (cap 42). Every name this module
exported is re-exported here unchanged, so the particle filter
(``rl/particle_filters/multimodal_search.py``), Brendan's benchmark registry
(``rl/benchmark/envs.py``), ``tests/test_multimodal_search.py`` and
``experiments/benchmark/probe_env.py`` keep their import path. New code should import from
``pdomains.multimodal_search`` or build the env with ``gym.make("pdomains-multimodal-search-v0")``.
"""

from pdomains.multimodal_search import (  # noqa: F401
    BASE_OBS_DIM,
    MODE_OBS_DIM,
    MultimodalSearchConfig,
    MultimodalSearchEnv,
    make_env,
    pack_observation,
    segment_distance,
    unpack_modes,
)

__all__ = [
    "BASE_OBS_DIM",
    "MODE_OBS_DIM",
    "MultimodalSearchConfig",
    "MultimodalSearchEnv",
    "make_env",
    "pack_observation",
    "segment_distance",
    "unpack_modes",
]
