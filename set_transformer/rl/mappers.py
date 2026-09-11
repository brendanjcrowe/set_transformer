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


def get_ant_tag_pf_kwargs(env) -> dict:
    """Build :class:`AntTagParticleFilter` kwargs from the live AntTag environment.

    Reading these off the env instead of hardcoding them keeps the belief's motion model,
    arena and reset prior in sync with whichever AntTag variant is being run — the base
    9x9 env and e.g. the arena-scaled den variants use different values.
    """
    unwrapped = env.unwrapped
    cage_max_x = float(unwrapped.cage_max_x)
    cage_max_y = float(unwrapped.cage_max_y)
    if not np.isclose(cage_max_x, cage_max_y):
        raise ValueError(
            "AntTagParticleFilter currently assumes a square arena, but "
            f"got cage_max_x={cage_max_x}, cage_max_y={cage_max_y}"
        )
    return {
        "arena_limits": (-cage_max_x, cage_max_x),
        "target_step": float(unwrapped.target_step),
        "visibility_radius": float(unwrapped.visible_radius),
        "min_initial_distance": float(unwrapped.min_distance),
    }


def ant_tag_pf_interaction_mapper(
    base_env_obs: np.ndarray,
    base_env_info: dict,
    base_env_action: np.ndarray | None = None,
    unwrapped_env=None,
    previous_base_env_obs: np.ndarray | None = None,
) -> dict:
    """Bridge AntTag observations to the AntTagParticleFilter predict/update interface.

    Visibility is read from ``obs[-2:]``: the base AntTag env (and the curriculum wrapper)
    write the true target there when it is within the visible radius, or zeros when not.
    We therefore treat a non-zero ``obs[-2:]`` as "target observed" rather than hardcoding
    a radius. A zero target exactly at the origin is astronomically unlikely.

    ``predict`` gets the ant position from *before* the step: the target moved in response
    to where the ant was when it chose its move, so propagating the belief against the
    post-step position would evaluate the motion model one step out of phase. ``update``
    still uses the current position, since that is what decided this step's visibility.
    The wrapper supplies ``previous_base_env_obs`` opportunistically (see
    ``_call_pf_interaction_mapper``); on the first step it is ``None`` and we fall back to
    the current observation.
    """
    ant_pos = base_env_obs[:2].copy()
    ant_pos_for_prediction = (
        previous_base_env_obs[:2].copy()
        if previous_base_env_obs is not None
        else ant_pos
    )
    target_in_obs = base_env_obs[-2:].copy()

    visible = np.any(target_in_obs != 0.0)
    observed_target = target_in_obs if visible else np.array([np.nan, np.nan])

    return {
        "predict_args": {"ant_current_pos_from_obs": ant_pos_for_prediction},
        "update_args": {
            "observed_target_pos": observed_target,
            "ant_current_pos_from_obs": ant_pos,
        },
    }
