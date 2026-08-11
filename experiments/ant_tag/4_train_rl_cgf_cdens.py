"""RL training with the weighted CGF belief encoder on Counterweighted-Den
Ant-Tag (pdomains-ant-tag-cdens-v0: heavy-near / light-far dens whose
occupancy weights pin the pooled belief mean at 0 by construction, so the
mirror bit lives only in the odd moments). Same pattern as
4_train_rl_cgf_dens.py: reuse 4_train_rl_cgf.py's full pipeline/CLI, swapping
only env_id + PF class.

Note the 300-step cap and the 1.8 visibility radius: pass
--curriculum "0:100,0.2:100,0.5:1.8,1:1.8"."""

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

_train_rl_cgf = importlib.import_module("4_train_rl_cgf")

if __name__ == "__main__":
    _train_rl_cgf.main(
        env_id="pdomains-ant-tag-cdens-v0",
        particle_filter_class=CounterweightedDenAntTagParticleFilter,
        run_subdir="ant_tag_cgf_cdens",
    )
