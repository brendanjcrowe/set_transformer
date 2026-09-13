"""
Evaluate a checkpoint on the REAL sparse tag reward -- forwarder (change 5.1, 2026-09-12).

Historically the eval for 4_train_rl_finetune.py's checkpoints, with the base env, the
unweighted PFDictObservationWrapper and a hardcoded cap of 400. It now forwards to the shared
evaluation script ``set_transformer.rl.eval_true_reward`` with the domain fixed to Ant-Tag:
``--variant`` (default ``base``) selects env, filter and cap, and the env carries PF weights
like every current arm's. A checkpoint trained by 4_train_rl_finetune.py on the unweighted
observation does NOT load through this env (its observation space has no "weights" key); that
script is the older, pre-run_config.json finetune path and is not comparable with the arms
anyway (CLAUDE.md). Use eval_true_reward_<encoder>.py for the arms.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _ANT_TAG_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pdomains  # noqa: F401,E402 - registers pdomains-ant-tag-*

from set_transformer.rl.eval_true_reward import main as _shared_main  # noqa: E402


def main(argv=None):
    return _shared_main(argv, domain="ant_tag", prog="eval_true_reward.py")


if __name__ == "__main__":
    main()
