"""
Evaluate a weighted-CGF checkpoint trained on Ghost Ant-Tag, on the REAL
sparse tag reward. Identical to eval_true_reward_cgf.py's eval loop, just
pointed at pdomains-ant-tag-ghost-v0 + GhostAntTagParticleFilter so the PF
matches the ghost-ping sensor model and the smart target's motion model.

Usage:
    python3 eval_scripts/eval_true_reward_cgf_ghost.py \
        --model_path runs/ant_tag_cgf_ghost/<run>/models/best_model/best_model.zip \
        --vecnormalize_path runs/ant_tag_cgf_ghost/<run>/models/vecnormalize.pkl \
        --n_episodes 100
"""
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401 - registers pdomains-ant-tag-ghost-v0
from set_transformer.rl.particle_filters.ant_tag import GhostAntTagParticleFilter

_eval_cgf = importlib.import_module("eval_true_reward_cgf")

if __name__ == "__main__":
    _eval_cgf.main(
        env_id="pdomains-ant-tag-ghost-v0",
        particle_filter_class=GhostAntTagParticleFilter,
    )
