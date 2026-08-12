"""
Evaluate a weighted-Gaussian checkpoint trained on Twin-Den Ant-Tag, on the
REAL sparse tag reward. Identical to eval_true_reward_gaussian.py's eval loop,
just pointed at pdomains-ant-tag-dens-v0 + TwinDenAntTagParticleFilter so the
PF matches the leashed twin-den motion model.

IMPORTANT: pass --max_steps 200. pdomains-ant-tag-dens-v0 truncates at 200,
not 400; without the flag the eval script's default 400 success convention
would count every timeout as a tag (success would read ~100%).

Usage:
    python3 eval_scripts/eval_true_reward_gaussian_dens.py \
        --model_path runs/ant_tag_gaussian_dens/<run>/models/best_model/best_model.zip \
        --vecnormalize_path runs/ant_tag_gaussian_dens/<run>/models/vecnormalize.pkl \
        --n_episodes 100 --max_steps 200
"""
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401 - registers pdomains-ant-tag-dens-v0
from set_transformer.rl.particle_filters.ant_tag import TwinDenAntTagParticleFilter

_eval_gaussian = importlib.import_module("eval_true_reward_gaussian")

if __name__ == "__main__":
    _eval_gaussian.main(
        env_id="pdomains-ant-tag-dens-v0",
        particle_filter_class=TwinDenAntTagParticleFilter,
    )
