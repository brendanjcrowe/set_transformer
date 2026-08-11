"""RL training with the weighted Gaussian (mean + covariance) belief encoder
on Ghost Ant-Tag (pdomains-ant-tag-ghost-v0: SmartAntTag + unreliable
long-range pings that create ghost belief modes). Same pattern as
4_train_rl_gaussian_smart.py: reuse 4_train_rl_gaussian.py's full
pipeline/CLI, swapping only env_id + PF class."""

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401 - registers pdomains-ant-tag-ghost-v0
from set_transformer.rl.particle_filters.ant_tag import GhostAntTagParticleFilter

_train_rl_gaussian = importlib.import_module("4_train_rl_gaussian")

if __name__ == "__main__":
    _train_rl_gaussian.main(
        env_id="pdomains-ant-tag-ghost-v0",
        particle_filter_class=GhostAntTagParticleFilter,
        run_subdir="ant_tag_gaussian_ghost",
    )
