"""
Evaluate a deepset checkpoint (4_train_rl_deepset.py) on the REAL sparse tag reward.

WeightedDeepSetFeaturesExtractor consumes the same Dict observation as WeightedCGFFeaturesExtractor
({"obs", "particles", "weights"}), so eval_true_reward_cgf.py's environment and loop are
reused verbatim. The extractor class lives in the set_transformer package
(rl/feature_extractors/pooled.py), so SB3 resolves it from the saved policy_kwargs
without this module; importing 4_train_rl_pool here keeps the arms symmetric.

    python3 eval_scripts/eval_true_reward_deepset.py --variant smart \
        --model_path runs/ant_tag_deepset_smart/<run>/models/deepset_agent.zip \
        --vecnormalize_path runs/ant_tag_deepset_smart/<run>/models/vecnormalize.pkl --n_episodes 100
"""

import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
_EVAL_DIR = Path(__file__).resolve().parent
for _path in (_REPO_ROOT, _ANT_TAG_DIR, _EVAL_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pdomains  # noqa: F401,E402

_train_rl_pool = importlib.import_module("4_train_rl_pool")  # noqa: F401
_eval_cgf = importlib.import_module("eval_true_reward_cgf")
make_eval_env = _eval_cgf.make_eval_env
main = _eval_cgf.main

if __name__ == "__main__":
    main()
