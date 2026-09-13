"""
Evaluate an Odd-Even checkpoint on the env's own 0/1 exact-match reward, split transient /
steady against the Bayes oracle -- entry point of the shared evaluation script
``set_transformer.rl.eval_true_reward`` with the domain fixed to Odd-Even (change 5.1 of the
harness centralisation, 2026-09-12). Every flag this script took still works. Equivalently:

    python3 -m set_transformer.rl.eval_true_reward --domain odd_even --variant oe50_short ...

One eval serves all arms: 4_train_rl_{cgf,st,gaussian}.py read the same {"obs",
"particles", "weights"} dict observation, so a checkpoint from any of them runs through this
env unchanged.

What is reported and why (no "success" on this domain; steady state is the headline; the
oracle and the previous-observation baseline on the SAME episodes; the seed IS the episode,
so the env is re-seeded before every reset) is documented above the metric code, which lives
in ``set_transformer/rl/domains/odd_even.py`` and is re-exported below for the tests.

Usage:
    python3 eval_scripts/eval_true_reward_odd_even.py --variant oe50 \\
        --model_path runs/odd_even_cgf_oe50/<...>/models/cgf_agent.zip \\
        --vecnormalize_path runs/odd_even_cgf_oe50/<...>/models/vecnormalize.pkl \\
        --n_episodes 400

    # Baselines only, no checkpoint needed -- this reproduces the reference
    # table in domain_mds/oddeven.md. --oracle_only is an accepted alias.
    python3 eval_scripts/eval_true_reward_odd_even.py --variant oe50 \\
        --baselines_only --n_episodes 400
"""

import sys
from pathlib import Path

# Depth matters (PITFALLS.md section 7). parents[3] is the package root from
# experiments/odd_even/eval_scripts/; parents[2] would be correct only at the top level.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401,E402

from set_transformer.rl.domains.odd_even import (  # noqa: E402,F401
    COLLAPSE_STEP,
    episode_metrics as _episode_metrics,
    print_metrics_block as _print_block,
    run_reference_policies,
    split_metric as _split_metric,
    summarize_episode,
)
from set_transformer.rl.eval_true_reward import (  # noqa: E402,F401
    checkpoint_num_particles as _checkpoint_num_particles,
    main as _shared_main,
)
from set_transformer.rl.run_records import read_run_status as _read_run_status  # noqa: E402,F401


def main(argv=None) -> None:
    _shared_main(argv, domain="odd_even", prog="eval_true_reward_odd_even.py")


if __name__ == "__main__":
    main()
