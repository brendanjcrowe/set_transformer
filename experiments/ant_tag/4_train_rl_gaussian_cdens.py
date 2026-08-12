"""RL training with the weighted Gaussian (mean+covariance) belief encoder on
Counterweighted-Den Ant-Tag (pdomains-ant-tag-cdens-v0). Same pattern as
4_train_rl_gaussian_dens.py: reuse 4_train_rl_gaussian.py's full pipeline/CLI,
swapping only env_id + PF class.

This is the "blind" arm of the experiment: the occupancy weights satisfy
w*h = (1-w)*f as an IDENTITY, so the pooled mean is pinned at 0 and the
covariance is invariant under the point reflection that IS the mirror-bit
swap -- these 5 features provably cannot carry the bit."""

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401 - registers pdomains-ant-tag-cdens-v0
from set_transformer.rl.particle_filters.ant_tag import (
    CounterweightedDenAntTagParticleFilter,
)

_train_rl_gaussian = importlib.import_module("4_train_rl_gaussian")

if __name__ == "__main__":
    _train_rl_gaussian.main(
        env_id="pdomains-ant-tag-cdens-v0",
        particle_filter_class=CounterweightedDenAntTagParticleFilter,
        run_subdir="ant_tag_gaussian_cdens",
    )
