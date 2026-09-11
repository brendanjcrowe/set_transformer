"""Registry-driven benchmark harness for PF-belief encoders on POMDPs.

- :mod:`registry` — ``EnvSpec`` / ``MethodSpec`` dataclasses and the ``ENV_REGISTRY`` /
  ``METHOD_REGISTRY`` tables that make the unified trainer extensible.
- :mod:`envs` — per-env base-env factories and potential functions.
- :mod:`results` — cross-machine ingest of run records + bootstrap statistics.

Entry points: ``experiments/benchmark/{train,aggregate,plot}.py``.
"""

from set_transformer.rl.benchmark.registry import (
    ENV_REGISTRY,
    METHOD_REGISTRY,
    EnvSpec,
    MethodSpec,
    build_extractor_kwargs,
    get_env_spec,
    get_method_spec,
)
from set_transformer.rl.benchmark.results import (
    RunRecord,
    align_curves,
    bootstrap_mean_ci,
    curve_band,
    discover_runs,
    encoder_cost_rows,
    group_runs,
    load_run,
    summarize,
)

__all__ = [
    "EnvSpec",
    "MethodSpec",
    "ENV_REGISTRY",
    "METHOD_REGISTRY",
    "get_env_spec",
    "get_method_spec",
    "build_extractor_kwargs",
    "RunRecord",
    "load_run",
    "discover_runs",
    "group_runs",
    "summarize",
    "encoder_cost_rows",
    "align_curves",
    "curve_band",
    "bootstrap_mean_ci",
]
