"""
Collect a particle-filter dataset from an Ant-Tag POMDP environment (pipeline step 2).

Since batch 7.4 of the harness centralisation (2026-09-13) this file is an ENTRY POINT of the
shared collector, ``set_transformer/rl/collect.py``, with the domain fixed to ``ant_tag``::

    python3 -m set_transformer.rl.collect --domain ant_tag --variant cdens_terminal ...

is the same run. Every flag this script took still works (``--num_trajectories`` and
``--output_file`` are the shared command's ``--num_episodes`` / ``--output_file``). The
Ant-Tag pieces -- the trajectory mix (fully observed / pursuit with the locomotion policy /
random), the visibility radius drawn per trajectory through the RL curriculum knob, the
rebalancing by weighted spread in arena units -- live in ``set_transformer/rl/domains/ant_tag.py``
(block "Dataset collection") and are re-exported below for the diagnostics and tests that read
them off this module.

GENERIC OVER ENV + FILTER. Nothing here is tied to one env variant. Pass
--variant to select the env and its matching filter together. The rollout reuses the RL
scripts' env factory, so the belief distribution written to disk is produced by exactly the
same particle filter, interaction mapper and visibility wrapper that will run at RL time.

WEIGHTS ARE STORED. Output is an .npz with

    particles      [num_samples, num_particles, dim]  float32, RAW env coords
    weights        [num_samples, num_particles]       float32, PF weights
    particle_scale scalar: the arena half-width the RL extractors divide by
    metadata       JSON string: env id, filter, CLI args, git provenance

WHERE IT GOES (decision 1 of plan section 7): --output_file as given, else
<root>/ant_tag/<variant>/data/<variant>_pf_dataset[_<run_tag>].npz under the shared run root
(--output_root > $RL_BMDP_RUNS > the parent repo's runs/). Before 2026-09-13 the default was
data/<variant>_pf_dataset.npz beside this script, where the recorded datasets still are.

Usage:
    python3 2_collect_pf_dataset.py --variant cdens_terminal \\
        --num_trajectories 300 --timesteps 300 --num_particles 100 \\
        --locomotion_policy_path models/ant_locomotion_policy.zip

    python3 2_collect_pf_dataset.py --list_variants
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from set_transformer.rl.collect import main as _shared_main  # noqa: E402
from set_transformer.rl.domains.ant_tag import (  # noqa: E402,F401 - the moved pieces, re-exported
    _ALWAYS_VISIBLE_RADIUS,
    _find_particle_filter,
    _normalize_obs,
    _pursuit_action,
    _rebalance_by_spread,
    _weighted_spread,
    get_ant_tag_arena_scale,
    resolve_particle_filter,
)
from set_transformer.rl.domains.ant_tag import make_ant_tag_cgf_env as make_ant_tag_belief_env  # noqa: E402,F401
from set_transformer.rl.run_records import git_provenance as _git_provenance  # noqa: E402,F401


def main(argv=None):
    return _shared_main(argv, domain="ant_tag", prog="2_collect_pf_dataset.py")


if __name__ == "__main__":
    main()
