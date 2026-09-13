"""
Shared Ant-Tag RL pieces -- FORWARDING FILE (since 2026-09-12).

The code that lived here (reward shaping, visibility / evasion curriculum, the curriculum
callback, PF glue) is now ``set_transformer.rl.domains.ant_tag`` (change 1a of the harness
centralisation; ``refactor_plans.md`` in the parent repo). This file re-exports the same
objects under its historical name because 4_train_rl_cgf.py (and through it the ST /
Gaussian / pool arms), 4_train_rl_finetune.py, sample_and_render.py and several tests
import it by flat name
(``importlib.import_module("4_train_rl_frozen")``; the digit makes ``import`` impossible).

History: until 2026-09-11 this was the May-2026 "pretrained Set Transformer as a fixed
feature processor inside the env" arm; that runnable half was parked in to_be_deleted/ and
deleted on 2026-09-12 (change 5.3a; git history before that commit has it). The live frozen
arm is
    python3 4_train_rl_st.py --variant <variant> --pretrained_st_model_path <ckpt> --st_frozen

Side effects kept on purpose, because importers have relied on them: the package-root
sys.path insert, matplotlib's Agg backend, and ``import pdomains`` (env registration).
"""

import sys
from pathlib import Path

# Same bootstrap as the rest of this directory: put the package root on
# sys.path so `set_transformer` resolves to the package, not the submodule
# directory of the same name.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import matplotlib

matplotlib.use("Agg")

import pdomains  # noqa: F401 — registers pdomains-ant-tag-v0
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter  # noqa: F401
from set_transformer.rl.domains.ant_tag import (  # noqa: F401
    CurriculumCallback,
    CurriculumVisibilityWrapper,
    PFRewardShapingWrapper,
    _CurriculumRouter,
    _set_evasion_scale_recursive,
    _set_radius_recursive,
    _set_reward_coeffs_recursive,
    ant_tag_pf_interaction_mapper,
    get_ant_tag_pf_kwargs,
)

__all__ = [
    "AntTagParticleFilter",
    "CurriculumCallback",
    "CurriculumVisibilityWrapper",
    "PFRewardShapingWrapper",
    "_CurriculumRouter",
    "_set_evasion_scale_recursive",
    "_set_radius_recursive",
    "_set_reward_coeffs_recursive",
    "ant_tag_pf_interaction_mapper",
    "get_ant_tag_pf_kwargs",
]


if __name__ == "__main__":
    raise SystemExit(
        "4_train_rl_frozen.py is a forwarding file for set_transformer.rl.domains.ant_tag, "
        "not a runnable arm (since 2026-09-11). For a frozen pretrained Set Transformer use:\n"
        "  python3 4_train_rl_st.py --variant <variant> --pretrained_st_model_path <ckpt> --st_frozen"
    )
