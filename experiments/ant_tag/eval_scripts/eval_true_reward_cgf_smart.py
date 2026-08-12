"""
Evaluate a weighted-CGF checkpoint trained on Smart Ant-Tag, on the REAL
sparse tag reward. Identical to eval_true_reward_cgf.py's eval loop, just
pointed at pdomains-ant-tag-smart-v0 + SmartAntTagParticleFilter so the PF
matches the smart target's true motion model.

Usage:
    python experiments/ant_tag/eval_scripts/eval_true_reward_cgf_smart.py \
        --model_path runs/ant_tag_cgf_smart/<run>/models/best_model/best_model.zip \
        --vecnormalize_path runs/ant_tag_cgf_smart/<run>/models/vecnormalize.pkl \
        --n_episodes 50
"""
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401 - registers pdomains-ant-tag-smart-v0
from set_transformer.rl.particle_filters.ant_tag import SmartAntTagParticleFilter

_eval_cgf = importlib.import_module("eval_true_reward_cgf")

if __name__ == "__main__":
    _eval_cgf.main(
        env_id="pdomains-ant-tag-smart-v0",
        particle_filter_class=SmartAntTagParticleFilter,
    )
