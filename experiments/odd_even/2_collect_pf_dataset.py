"""
Collect a particle-filter dataset from the Odd-Even POMDP (pipeline step 2).

Since batch 7.4 of the harness centralisation (2026-09-13) this file is an ENTRY POINT of the
shared collector, ``set_transformer/rl/collect.py``, with the domain fixed to ``odd_even``::

    python3 -m set_transformer.rl.collect --domain odd_even --variant oe50 --num_episodes 200

is the same run. Every flag this script took still works (``--output`` is the shared command's
``--output_file``). The Odd-Even pieces -- random actions (the action is a prediction and
changes neither the hidden state nor the observation stream), rebalancing by step index or
effective sample size, the particle centre and per-row step index stored in the file -- live in
``set_transformer/rl/domains/odd_even.py`` (block "Dataset collection") and are re-exported
below for the tests that read them off this module.

The .npz contract is the one the reconstruction pretraining reads, so step 3 needs no
Odd-Even version at all:

    particles      [num_samples, num_particles, dim]  float32, RAW states
    weights        [num_samples, num_particles]        float32, PF weights
    particle_scale scalar: the half-width the RL extractors divide by
    particle_centre scalar: the centre the RL env subtracts first
    steps          [num_samples] int32, the step index of every row
    metadata       JSON string: env id, filter, CLI args, git provenance

WHERE IT GOES (decision 1 of plan section 7): --output as given, else
<root>/odd_even/<variant>/data/<variant>_pf_dataset[_<run_tag>].npz under the shared run root
(--output_root > $RL_BMDP_RUNS > the parent repo's runs/). Before 2026-09-13 the default was
data/<variant>_pf_dataset.npz beside this script, where the recorded datasets still are.

Usage:
    python3 2_collect_pf_dataset.py --variant oe50 \\
        --num_episodes 200 --timesteps 50 --num_particles 50

    python3 2_collect_pf_dataset.py --list_variants
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from set_transformer.rl.collect import main as _shared_main  # noqa: E402
from set_transformer.rl.domains import odd_even as variants  # noqa: E402,F401
from set_transformer.rl.domains.odd_even import (  # noqa: E402,F401 - the moved pieces, re-exported
    COLLAPSE_STEP,
    EARLY_STEP,
    _effective_sample_size,
    _rebalance,
    _report_distribution,
    collect_dataset_for_test,
    make_odd_even_belief_env,
)
from set_transformer.rl.run_records import git_provenance as _git_provenance  # noqa: E402,F401


def main(argv=None):
    return _shared_main(argv, domain="odd_even", prog="2_collect_pf_dataset.py")


if __name__ == "__main__":
    main()
