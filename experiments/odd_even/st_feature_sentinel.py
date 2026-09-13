"""Odd-Even ST collapse sentinel -- FORWARDING FILE (since 2026-09-12, change 4).

The callback, its threshold and the offline statistic live in
``set_transformer.rl.domains.odd_even`` (``OddEvenSTFeatureSentinel``,
``COLLAPSE_RELATIVE_SPREAD``, ``relative_feature_spread``), together with the measurement
record that used to be this module's docstring (``ST_SENTINEL_NOTES`` there). This file
re-exports them for ``4_train_rl_st.py``, the diagnostics and the tests that load it through
``_sibling.load("st_feature_sentinel")``.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from set_transformer.rl.domains.odd_even import (  # noqa: E402,F401
    COLLAPSE_RELATIVE_SPREAD,
    ST_SENTINEL_NOTES,
    OddEvenSTFeatureSentinel,
    _relative,
    relative_feature_spread,
)

__all__ = [
    "COLLAPSE_RELATIVE_SPREAD",
    "ST_SENTINEL_NOTES",
    "OddEvenSTFeatureSentinel",
    "_relative",
    "relative_feature_spread",
]
