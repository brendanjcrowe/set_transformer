"""
RL training with a weighted Gaussian (mean + covariance) belief encoder for Ant-Tag.

The AntTag particle filter, curriculum and reward-shaping pipeline of the CGF
arm, with the belief encoder replaced by the lowest-order moments of the
weighted particle distribution:

    mean_d      = sum_i w_i * x_i[d]
    cov_{d,d'}  = sum_i w_i * (x_i[d] - mean_d) * (x_i[d'] - mean_d')

for particle dimension d, d' in {0, ..., D-1} (D=2 for AntTag: target x, y).
The policy receives [mean, var_x, var_y, cov_xy] (5 features for D=2), analogous
to mean_var_encoding_odd_even_beliefmdp.py's [mean, var] for the 1D OddEven belief.

Unlike the CGF and Set Transformer encoders, this extractor has no learnable
parameters -- it is a fixed, closed-form summary of the particle set.

Since 2026-09-12 (change 4.5 of the harness centralisation) this file is an
ENTRY POINT: the flags, their resolution, the run record and the PPO loop are
the shared ones in ``set_transformer.rl.train`` (domain ``ant_tag``, encoder
``gaussian``). Every flag this script ever took still works; the run
directory is still the cwd-relative ``runs/ant_tag_gaussian[_<variant>]/...``
(``legacy_layout=True``) until change 5. Equivalently:

    python3 -m set_transformer.rl.train --domain ant_tag --encoder gaussian --variant smart ...

The names below are re-exported under this module's name: SB3 pickles a
policy's features-extractor CLASS into the saved zip by module path, so
loading one of the 817 recorded Gaussian checkpoints runs
``getattr(import_module("4_train_rl_gaussian"), "WeightedGaussianFeaturesExtractor")``;
eval_scripts/eval_true_reward_gaussian.py and the tests read the wrappers and
the PF glue off this module.

Usage:
    python3 4_train_rl_gaussian.py --variant smart --seed 0 --run_tag gaussian_v1
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pdomains  # noqa: F401,E402 - registers pdomains-ant-tag-*

from set_transformer.rl.feature_extractors.gaussian import (  # noqa: E402,F401
    WeightedGaussianFeaturesExtractor,
)
from set_transformer.rl.wrappers.particle_filter import (  # noqa: E402,F401
    PFDictWithWeightsObservationWrapper,
)
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter  # noqa: E402,F401

# The Ant-Tag domain pieces this arm shared with the CGF script, same objects.
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
    make_ant_tag_cgf_env as make_ant_tag_belief_env,
)
from set_transformer.rl.curriculum import (  # noqa: E402,F401
    parse_curriculum as _parse_curriculum,
    parse_reward_schedule as _parse_reward_schedule,
)
from set_transformer.rl.run_records import (  # noqa: E402,F401
    default_run_dir as _shared_default_run_dir,
    git_provenance as _git_provenance,
    tee_stdout_stderr as _tee_stdout_stderr,
    write_run_config as _write_run_config,
)
from set_transformer.rl.train import main as _shared_main  # noqa: E402


def _default_run_dir(seed: int, run_subdir: str = "ant_tag_gaussian",
                      run_tag: str | None = None) -> str:
    """run_records.default_run_dir with this script's historical default subfolder."""
    return _shared_default_run_dir(seed, run_subdir, run_tag)


def main(argv=None):
    """The shared command line with this arm's domain and encoder fixed. --variant
    selects env id, particle filter and run subdir together, so the three cannot disagree."""
    return _shared_main(argv, domain="ant_tag", encoder="gaussian", legacy_layout=True,
                        prog="4_train_rl_gaussian.py")


if __name__ == "__main__":
    main()
