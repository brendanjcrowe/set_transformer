"""RL training with the weighted Gaussian (mean+covariance) belief encoder on
Twin-Den Ant-Tag (pdomains-ant-tag-dens-v0: two mirrored tight/loose hideouts,
moment-matched belief construction). Same pattern as
4_train_rl_gaussian_ghost.py: reuse 4_train_rl_gaussian.py's full pipeline/CLI,
swapping only env_id + PF class.

This is the "blind" arm of the experiment: at balanced den weights the pooled
mean and covariance are provably invariant to the tight/loose swap, so these
5 features cannot carry the ordering bit."""

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401 - registers pdomains-ant-tag-dens-v0
from set_transformer.rl.particle_filters.ant_tag import TwinDenAntTagParticleFilter

_train_rl_gaussian = importlib.import_module("4_train_rl_gaussian")

if __name__ == "__main__":
    _train_rl_gaussian.main(
        env_id="pdomains-ant-tag-dens-v0",
        particle_filter_class=TwinDenAntTagParticleFilter,
        run_subdir="ant_tag_gaussian_dens",
    )
