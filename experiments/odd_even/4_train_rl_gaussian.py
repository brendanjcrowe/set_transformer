"""
RL training with a weighted Gaussian belief encoder on the Odd-Even POMDP.

Third arm of the encoder comparison, alongside 4_train_rl_cgf.py and
4_train_rl_st.py. It imports the CGF module's training loop and env factory
and swaps ONLY the features extractor.

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

Usage:
    python3 4_train_rl_gaussian.py --variant oe50 \
        --total_timesteps 500000 --run_tag gaussian_v1
"""

import argparse
import importlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# This directory too, so `import _sibling` works when a test loads this file
# by path rather than running it as a script (where it is sys.path[0]).
# ONLY this directory -- never a sibling experiment directory (Gap 12).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# `variants` is loaded BY PATH, not as a flat name: experiments/ant_tag/ has a
# variants.py too, and sys.modules is process-wide, so a plain
# `import variants` returns whichever one was imported first anywhere in the
# process. See _sibling.py -- this was measured happening in the test suite.
import _sibling  # noqa: E402
variants = _sibling.load("variants")

# Loaded by explicit path: `4_train_rl_cgf` names two different files, one
# here and one in experiments/ant_tag/. See _sibling.py.
_train_rl_cgf = _sibling.load("4_train_rl_cgf")
train_odd_even = _train_rl_cgf.train_odd_even
add_common_arguments = _train_rl_cgf.add_common_arguments
resolve_common = _train_rl_cgf.resolve_common
net_arch_from_args = _train_rl_cgf.net_arch_from_args
make_odd_even_belief_env = _train_rl_cgf.make_odd_even_belief_env

from set_transformer.rl.feature_extractors.gaussian import (  # noqa: E402
    WeightedGaussianFeaturesExtractor,
)


def main() -> None:
    encoder = "gaussian"
    parser = argparse.ArgumentParser(
        description="RL with weighted Gaussian (mean + variance) "
                    "particle-belief features on the Odd-Even POMDP.")
    add_common_arguments(parser, encoder)
    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    (_run_dir, log_dir, model_save_path, run_subdir,
     particle_filter_class) = resolve_common(args, encoder)

    policy_kwargs = {
        "features_extractor_class": WeightedGaussianFeaturesExtractor,
        # arena_scale only, matching the other two arms' particle
        # normalization: this extractor has no other knobs.
        "features_extractor_kwargs": dict(arena_scale=args.arena_scale),
    }
    net_arch = net_arch_from_args(args)
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    train_odd_even(
        policy_kwargs=policy_kwargs,
        encoder=encoder,
        variant=args.variant,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        ppo_n_steps=args.ppo_n_steps,
        n_epochs=args.n_epochs,
        num_particles=args.num_particles,
        particle_filter_class=particle_filter_class,
        device=args.device,
        seed=args.seed,
        run_subdir=run_subdir,
        log_dir=log_dir,
        model_save_path=model_save_path,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        n_eval_episodes=args.n_eval_episodes,
        use_vec_normalize=not args.no_vec_normalize,
        lr_anneal=args.lr_anneal,
        target_kl=args.target_kl,
        progress_bar=args.progress_bar,
        # No t_values and no cached ST features, so both logging callbacks
        # would be no-ops. Omitted rather than passed for show.
    )


if __name__ == "__main__":
    main()
