"""Env-specific particle-filter interaction mappers.

A ``pf_interaction_mapper`` bridges a base env's observation to the ``predict`` / ``update``
kwargs of that env's :class:`~set_transformer.rl.particle_filters.base.BaseParticleFilter`,
for :class:`~set_transformer.rl.wrappers.particle_filter.PFDictObservationWrapper`.

Lifted here (out of the digit-prefixed ``experiments/ant_tag/4_train_rl_*.py`` scripts) so
the unified benchmark trainer can import them from the package instead of via the old
``importlib.import_module("4_train_rl_frozen")`` hack.
"""

from __future__ import annotations

import numpy as np


def ant_tag_pf_interaction_mapper(
    base_env_obs: np.ndarray,
    base_env_info: dict,
    base_env_action: np.ndarray | None = None,
    unwrapped_env=None,
) -> dict:
    """Bridge AntTag observations to the AntTagParticleFilter predict/update interface.

    Visibility is read from ``obs[-2:]``: the base AntTag env (and the curriculum wrapper)
    write the true target there when it is within the visible radius, or zeros when not.
    We therefore treat a non-zero ``obs[-2:]`` as "target observed" rather than hardcoding
    a radius. A zero target exactly at the origin is astronomically unlikely.
    """
    ant_pos = base_env_obs[:2].copy()
    target_in_obs = base_env_obs[-2:].copy()

    visible = np.any(target_in_obs != 0.0)
    observed_target = target_in_obs if visible else np.array([np.nan, np.nan])

    return {
        "predict_args": {"ant_current_pos_from_obs": ant_pos},
        "update_args": {
            "observed_target_pos": observed_target,
            "ant_current_pos_from_obs": ant_pos,
        },
    }
