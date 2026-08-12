"""
RL training with a weighted CGF belief encoder for the Smart Ant-Tag variant.

This is 4_train_rl_cgf.py's exact pipeline (PF, curriculum, reward shaping,
WeightedCGFFeaturesExtractor, training loop, CLI) pointed at
`pdomains-ant-tag-smart-v0` (pdomains.ant_tag.SmartAntTagEnv) instead of
`pdomains-ant-tag-v0`. The only two things that actually change:

  1. env_id: the smart env flees harder/faster as the ant closes in and
     slides along the wall instead of freezing against it.
  2. particle_filter_class: SmartAntTagParticleFilter mirrors that same
     motion model during belief propagation, instead of assuming the base
     env's flat 25/25/25/25 + freeze-at-wall dynamics.

Everything else is reused unchanged from 4_train_rl_cgf.py's main() via
import. See that file for the full CLI (--evasion_curriculum, etc.).
"""

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401 - registers pdomains-ant-tag-smart-v0
from set_transformer.rl.particle_filters.ant_tag import SmartAntTagParticleFilter

# Sibling module name starts with a digit, so importlib is required.
_train_rl_cgf = importlib.import_module("4_train_rl_cgf")

if __name__ == "__main__":
    _train_rl_cgf.main(
        env_id="pdomains-ant-tag-smart-v0",
        particle_filter_class=SmartAntTagParticleFilter,
        run_subdir="ant_tag_cgf_smart",
    )
