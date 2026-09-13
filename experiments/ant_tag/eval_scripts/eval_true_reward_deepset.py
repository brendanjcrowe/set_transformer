"""
Evaluate a deepset checkpoint (4_train_rl_pool.py) on the REAL sparse tag reward.

Entry point of the shared evaluation script ``set_transformer.rl.eval_true_reward`` with the
domain fixed to Ant-Tag (change 5.1 of the harness centralisation, 2026-09-12). Every flag
this script took still works; the output lines are unchanged (the wave drivers grep
"Success rate" and "Median length"). Equivalently:

    python3 -m set_transformer.rl.eval_true_reward --domain ant_tag --variant smart ...

The eval env is the domain's (``rl/domains/ant_tag.py``): the env's real visibility radius,
no reward shaping, so Monitor sees the env's own -1/step, 0-on-tag reward. Importing
``4_train_rl_pool`` below is what lets SB3 unpickle ``WeightedDeepSetFeaturesExtractor`` out of a saved
``policy_kwargs`` when the zip names that module.

Usage:
    python3 eval_scripts/eval_true_reward_deepset.py --variant smart \\
        --model_path runs/ant_tag_deepset_smart/<run>/models/deepset_agent.zip \\
        --vecnormalize_path runs/ant_tag_deepset_smart/<run>/models/vecnormalize.pkl \\
        --n_episodes 100
"""
import importlib
import sys
from pathlib import Path

# parents[3] is the package root from experiments/ant_tag/eval_scripts/; the numbered
# training scripts live one level up and are imported by flat name (PITFALLS.md section 7).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _ANT_TAG_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pdomains  # noqa: F401,E402 - registers pdomains-ant-tag-*

# Registers the extractor class under the module name recorded in the checkpoint.
importlib.import_module("4_train_rl_pool")  # noqa: F401

from set_transformer.rl.eval_true_reward import main as _shared_main  # noqa: E402


def main(argv=None):
    """--variant selects env id, particle filter AND the episode cap together."""
    return _shared_main(argv, domain="ant_tag", prog="eval_true_reward_deepset.py")


if __name__ == "__main__":
    main()
