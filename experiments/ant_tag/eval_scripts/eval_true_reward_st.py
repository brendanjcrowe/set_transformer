"""
Evaluate a Set Transformer checkpoint on the REAL sparse tag reward.

SetTransformerFeaturesExtractor consumes exactly the same Dict observation
as WeightedCGFFeaturesExtractor ({"obs", "particles", "weights"}), so the
eval environment is identical to eval_true_reward_cgf.py's and is reused
verbatim rather than copied. The one thing this module adds is importing
4_train_rl_st, so that SB3 can resolve the extractor class recorded in the
saved policy_kwargs when it unpickles the checkpoint.

Usage:
    python3 experiments/ant_tag/eval_scripts/eval_true_reward_st.py \
        --model_path runs/ant_tag_st/<run>/models/best_model/best_model.zip \
        --vecnormalize_path runs/ant_tag_st/<run>/models/vecnormalize.pkl \
        --n_episodes 50
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

# Registers SetTransformerFeaturesExtractor under the module name recorded
# in the checkpoint. Without this import PPO.load raises on unpickling.
_train_rl_st = importlib.import_module("4_train_rl_st")  # noqa: F401
_eval_cgf = importlib.import_module("eval_true_reward_cgf")

# Same dict-obs eval env and same eval loop as the CGF arm.
make_eval_env = _eval_cgf.make_eval_env
main = _eval_cgf.main


if __name__ == "__main__":
    main()
