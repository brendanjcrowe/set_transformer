"""
RL training with the kmoments belief encoder for Ant-Tag: WeightedKMomentsFeaturesExtractor.

Entry point only. The trainer, argparse and run-dir conventions live in
4_train_rl_pool.py (main(encoder="kmoments")); this file exists so the arm has the
numbered-script name every driver, eval script and record expects, and so the
symbols below stay importable by name like the other arms' (see
tests/test_ant_tag_shared_pieces_regression.py).

    python3 4_train_rl_kmoments.py --variant smart --seed 0 [--run_tag <tag>]
    python3 eval_scripts/eval_true_reward_kmoments.py --variant smart --model_path <zip> ...
"""

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_pool = importlib.import_module("4_train_rl_pool")
WeightedKMomentsFeaturesExtractor = _pool.WeightedKMomentsFeaturesExtractor
PFDictWithWeightsObservationWrapper = _pool.PFDictWithWeightsObservationWrapper
make_ant_tag_belief_env = _pool.make_ant_tag_belief_env
train_ant_tag_pool = _pool.train_ant_tag_pool
ENCODER = "kmoments"


def main():
    _pool.main(encoder=ENCODER)


if __name__ == "__main__":
    main()
