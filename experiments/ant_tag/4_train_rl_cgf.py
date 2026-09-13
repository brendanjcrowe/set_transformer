"""
RL training with a weighted CGF belief encoder for Ant-Tag.

The AntTag particle filter, curriculum and reward-shaping pipeline with a
trainable CGF feature extractor as the belief encoder:

    CGF_j = log(sum_i w_i * exp(<t_j, x_i>))

where x_i is the target-position particle and w_i is its particle-filter weight.

Since 2026-09-12 (change 4.5 of the harness centralisation) this file is an
ENTRY POINT: the flags, their resolution (CLI > pretraining checkpoint >
registry / legacy default, the sizing rule, the frame check), the run record
and the PPO loop are the shared ones in ``set_transformer.rl.train`` (domain
``ant_tag``, encoder ``cgf``). Every flag this script ever took still works
and every default is still the LEGACY value (clamp 2.0, K, no norm, linspace
init), so a bare re-run of a recorded command trains what it recorded. The
run directory is still the cwd-relative ``runs/ant_tag_cgf[_<variant>]/...``
(``legacy_layout=True``) until change 5. Equivalently:

    python3 -m set_transformer.rl.train --domain ant_tag --encoder cgf --variant smart ...

The names below are re-exported under this module's name: SB3 pickles a
policy's features-extractor CLASS into the saved zip by module path, so
loading one of the 1,257 recorded CGF checkpoints runs
``getattr(import_module("4_train_rl_cgf"), "WeightedCGFFeaturesExtractor")``;
and 2_collect_pf_dataset.py, the eval scripts, the diagnostics and the tests
read the env factory, the wrappers, the PF glue, the schedule parsers and the
run-record helpers off this module by name.

Usage:
    python3 4_train_rl_cgf.py --variant smart --seed 0 --run_tag cgf_v1
    python3 4_train_rl_cgf.py --variant smart --t_param polar --t_init_mode spread \\
        --feature_mode K_grad --feature_norm none
    python3 4_train_rl_cgf.py --list_variants
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401,E402 - registers pdomains-ant-tag-*

# The belief encoder and its callbacks, as the SAME objects the package defines.
from set_transformer.rl.feature_extractors.cgf import (  # noqa: E402,F401
    EncoderDriftLoggingCallback,
    RolloutFeatureNormCallback,
    TNormLoggingCallback,
    WeightedCGFFeaturesExtractor,
    cgf_raw_dim,
    matched_readout_hidden,
    non_readout_param_count,
    readout_param_count,
)
from set_transformer.rl.pretrained_encoder import reload_pretrained_cgf  # noqa: E402,F401
from set_transformer.rl.wrappers.particle_filter import (  # noqa: E402,F401
    PFDictWithWeightsObservationWrapper,
)
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter  # noqa: E402,F401

# The Ant-Tag domain: wrappers, curriculum adapter, PF glue, env factory, flag resolution.
from set_transformer.rl.domains.ant_tag import (  # noqa: E402,F401
    CurriculumCallback,
    CurriculumVisibilityWrapper,
    PFRewardShapingWrapper,
    _CurriculumRouter,
    _make_vec_env_from_fns,
    _make_vec_normalize,
    _resolve_reward_shaping,
    ant_tag_pf_interaction_mapper,
    get_ant_tag_arena_scale,
    get_ant_tag_pf_kwargs,
    get_env_visible_radius,
    make_ant_tag_cgf_env,
)

# Schedule parsers and run bookkeeping, under the names the other Ant-Tag files and the
# tests read (and monkeypatch) off this module.
from set_transformer.rl.curriculum import (  # noqa: E402,F401
    parse_curriculum as _parse_curriculum,
    parse_reward_schedule as _parse_reward_schedule,
)
from set_transformer.rl.run_records import (  # noqa: E402,F401
    TeeStream as _TeeStream,
    default_run_dir as _shared_default_run_dir,
    git_provenance as _git_provenance,
    tee_stdout_stderr as _tee_stdout_stderr,
    write_run_config as _write_run_config,
)
from set_transformer.rl.train import main as _shared_main  # noqa: E402

#: CGF geometry a pretrained checkpoint carries in its ``config``; the shared resolver's
#: table, under this module's historical name.
from set_transformer.rl.encoders import CGF_GEOMETRY_FLAGS  # noqa: E402,F401

#: Ant-Tag particles are the target's (x, y): 2-D by construction of every registered
#: filter (``Domain.particle_dim``).
ANT_TAG_PARTICLE_DIM = 2


def _default_run_dir(seed: int, run_subdir: str = "ant_tag_cgf",
                      run_tag: str | None = None) -> str:
    """run_records.default_run_dir with this script's historical default subfolder."""
    return _shared_default_run_dir(seed, run_subdir, run_tag)


def main(argv=None):
    """The shared command line with this arm's domain and encoder fixed. --variant
    selects env id, particle filter and run subdir together, so the three cannot disagree."""
    return _shared_main(argv, domain="ant_tag", encoder="cgf", legacy_layout=True,
                        prog="4_train_rl_cgf.py")


if __name__ == "__main__":
    main()
