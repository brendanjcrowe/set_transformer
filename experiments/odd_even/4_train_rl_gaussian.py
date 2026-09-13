"""
RL training with a weighted Gaussian belief encoder on the Odd-Even POMDP.

Third arm of the encoder comparison, alongside 4_train_rl_cgf.py and
4_train_rl_st.py.

    mean = sum_i w_i * x_i        var = sum_i w_i * (x_i - mean)^2

On a 1-D particle that is TWO features, and no learnable parameters at all --
which is what makes it the right control. The Bayes-optimal action under
squared-error loss is the posterior MEAN, so this encoder hands the policy the
optimal statistic directly. It is therefore not a weak baseline here but close
to a ceiling on this particular reward, and the interesting question is
whether the CGF and ST arms match it. (It is blind to everything else about
the posterior: two-mode and single-mode beliefs with the same mean and
variance are indistinguishable to it, which is what a reward other than
squared error would expose.)

Recording that up front matters, because a result where the parameter-free arm
wins is easy to misread as "the encoders failed" when it is partly a property
of the reward.

Since 2026-09-12 (change 4.3 of the harness centralisation) this file is an
ENTRY POINT ONLY: the flags, their resolution, the run record and the PPO loop
are the shared ones in ``set_transformer.rl.train`` (domain ``odd_even``,
encoder ``gaussian``). Every flag this script ever took still works; the run directory is ``<output root>/odd_even/<variant>/rl/gaussian/<timestamp>_seed<seed>[_<run_tag>]/`` (change 5.2; the root is ``--output_root`` > ``$RL_BMDP_RUNS`` > the parent repo's ``runs/``, never the current directory). Equivalently:

    python3 -m set_transformer.rl.train --domain odd_even --encoder gaussian --variant oe50 ...

Usage:
    python3 4_train_rl_gaussian.py --variant oe50 \
        --total_timesteps 500000 --run_tag gaussian_v1
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-exported under this module's name: SB3 pickles a policy's features-extractor CLASS
# into the saved zip by module path, and a zip saved by an older version of this script
# may name it here.
from set_transformer.rl.feature_extractors.gaussian import (  # noqa: E402,F401
    WeightedGaussianFeaturesExtractor,
)
from set_transformer.rl.train import main as _shared_main  # noqa: E402


def main(argv=None):
    """The shared command line with this arm's domain and encoder fixed."""
    return _shared_main(argv, domain="odd_even", encoder="gaussian",
                        prog="4_train_rl_gaussian.py")


if __name__ == "__main__":
    main()
