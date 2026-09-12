"""
RL training with a Set Transformer belief encoder on the Odd-Even POMDP.

Second arm of the encoder comparison, alongside 4_train_rl_cgf.py (weighted
empirical CGF) and 4_train_rl_gaussian.py (weighted mean + variance). It
imports the CGF module's training loop, env factory and run bookkeeping and
swaps ONLY the features extractor -- exactly the Ant-Tag arrangement -- so
the arms differ in the encoder alone.

WHAT THIS DOMAIN ASKS OF THE ENCODER, which is not what Ant-Tag asks. The
particle support here is the constant set {1, ..., n} at every step of every
episode; the entire belief lives in the WEIGHTS. So the ST is being asked to
compress an n-vector of masses attached to fixed positions, not a 2-D cloud
whose shape carries the information. Expect the ranking against CGF to differ
from Ant-Tag. --st_weight_channel (default on) is therefore not an option
here in the way it is there: with it off, the encoder reads the same constant
set forever and can carry no belief information at all.

PRETRAINED WEIGHTS ARE OPTIONAL AND OFF BY DEFAULT, because the CGF baseline
learns its t_values from scratch under PPO and the matched ST run is likewise
trained end to end. --pretrained_st_model_path (from 3_train_st.py on a
2_collect_pf_dataset.py .npz) and --st_frozen exist for the pretrained
variant, mirroring the CGF arm's --t_frozen.

PITFALLS.md SECTION 1 APPLIES IN FULL, and it is the most expensive bug in
this repo. SB3's ActorCriticPolicy._build ends with

    self.apply(partial(self.init_weights, gain=...))

which walks the WHOLE policy -- features extractor included -- and
re-initializes every Linear. An encoder loaded in the extractor's __init__ is
overwritten a moment later, and the "loaded encoder ... FROZEN" line prints
BEFORE the overwrite, so the run looks correct. That cost two 6M-step runs,
both reported 0%. This script therefore reloads the encoder AFTER PPO(...)
returns, re-freezes it, and ASSERTS max|delta| == 0 against the checkpoint.
The assertion is not decoration: it is the only thing that distinguishes a
working pretrained arm from a frozen-noise arm in the logs.

Usage:
    python3 4_train_rl_st.py --variant oe50 --total_timesteps 500000 \
        --num_encodings 8 --dim_encoder 8 --run_tag st_v1

    python3 4_train_rl_st.py --variant oe50 \
        --pretrained_st_model_path experiments/.../checkpoint_best.pt \
        --st_frozen
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

import torch
from stable_baselines3.common.callbacks import BaseCallback

# `variants` is loaded BY PATH, not as a flat name: experiments/ant_tag/ has a
# variants.py too, and sys.modules is process-wide, so a plain
# `import variants` returns whichever one was imported first anywhere in the
# process. See _sibling.py -- this was measured happening in the test suite.
import _sibling  # noqa: E402
variants = _sibling.load("variants")
_pretrained = _sibling.load("pretrained_encoder")

# `4_train_rl_cgf` names TWO different files -- ant_tag has one as well -- and
# a flat-name import resolves off sys.path and the process-wide sys.modules
# cache. So it is loaded by explicit path, like every other sibling here.
_train_rl_cgf = _sibling.load("4_train_rl_cgf")
train_odd_even = _train_rl_cgf.train_odd_even
add_common_arguments = _train_rl_cgf.add_common_arguments
resolve_common = _train_rl_cgf.resolve_common
net_arch_from_args = _train_rl_cgf.net_arch_from_args
make_odd_even_belief_env = _train_rl_cgf.make_odd_even_belief_env

# From the package, like every other shared piece.
from set_transformer.rl.feature_extractors.st import (  # noqa: E402
    STFeatureLoggingCallback,
    SetTransformerFeaturesExtractor,
)
from set_transformer.rl.encoder_finetune import (  # noqa: E402 - shared with experiments/ant_tag
    EncoderLRLoggingCallback,
    UnfreezeEncoderCallback,
    _ScaledLRParamGroup,
    _group_collapsing,
    scale_encoder_learning_rate,
)

# The Odd-Even collapse sentinel. The shared STFeatureLoggingCallback logs a
# 1-sample std at n_envs=1 (so it reads NaN), and its absolute statistic does
# not separate a healthy encoder from a collapsed one on this domain -- a
# healthy 8-epoch encoder reads 1.5e-2 while a genuinely dead 30-epoch one
# reads 9.0e-5. Both are logged; only the relative one is the sentinel. See
# st_feature_sentinel.py for the measurements.
OddEvenSTFeatureSentinel = _sibling.load(
    "st_feature_sentinel").OddEvenSTFeatureSentinel


# Since change 2 of the harness centralisation (2026-09-12) the reload-after-PPO step is
# the shared set_transformer.rl.pretrained_encoder.reload_pretrained: reload every extractor
# on the policy, re-freeze, ASSERT max|delta| == 0 against the checkpoint, blank the
# checkpoint path in policy_kwargs (PITFALLS.md sections 1 and 7). Kept under this module's
# historical name for the tests and the ST-side callers.
reload_pretrained_encoder = _pretrained.reload_pretrained


# The finetune-collapse fixes live in set_transformer/rl/encoder_finetune.py
# (shared with the Ant-Tag ST arm); imported above.
SMALL_GEOMETRY = {"num_inds": 16, "dim_hidden": 64, "num_post_sab": 2}


def resolve_st_geometry(args, parser) -> None:
    """Fill --num_inds / --dim_hidden: CLI > checkpoint config > small default.

    A checkpoint from 3_pretrain_st_belief.py carries its geometry in
    checkpoint["config"]. Taking it from there means the pretrained arm can
    never be launched with the wrong shape, and an explicit flag that
    disagrees with the checkpoint is refused rather than silently building a
    mismatched encoder for the loader to reject later.
    """
    from_ckpt = {}
    if args.pretrained_st_model_path:
        checkpoint = torch.load(args.pretrained_st_model_path,
                                map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        from_ckpt = {k: config[k] for k in SMALL_GEOMETRY if k in config}
    for key, small in SMALL_GEOMETRY.items():
        given = getattr(args, key)
        if given is None:
            setattr(args, key, from_ckpt.get(key, small))
        elif key in from_ckpt and from_ckpt[key] != given:
            parser.error(
                f"--{key} {given} disagrees with the checkpoint's {key}="
                f"{from_ckpt[key]} ({args.pretrained_st_model_path}). Drop the "
                "flag to take the checkpoint's geometry.")
    src = ("checkpoint" if from_ckpt else "default")
    print(f"ST geometry: num_inds={args.num_inds} dim_hidden={args.dim_hidden} "
          f"({src})")


def assert_encoder_matches_checkpoint(model, path: str) -> None:
    """max|delta| == 0 between the live encoder and the checkpoint.

    The verification snippet from PITFALLS.md section 1, run once per
    training start. Costs milliseconds and is the only positive evidence
    that the reload landed. The comparison itself lives in
    pretrained_encoder.py, shared with the CGF arm.
    """
    extractor = model.policy.features_extractor
    _pretrained.verify_matches_checkpoint(extractor.reference_state(path),
                                          extractor.encoder_state_dict(), path,
                                          label="ST encoder")


def main() -> None:
    encoder = "st"
    parser = argparse.ArgumentParser(
        description="RL with Set Transformer particle-belief features on the "
                    "Odd-Even POMDP.")
    add_common_arguments(parser, encoder)
    parser.add_argument("--num_encodings", type=int, default=8)
    parser.add_argument("--dim_encoder", type=int, default=8)
    parser.add_argument(
        "--num_inds", type=int, default=None,
        help="Inducing points. Default: the checkpoint's own geometry when "
             "--pretrained_st_model_path is given, else 16 (the ClusterHunt-"
             "sized encoder adopted 2026-09-05; ~109k params). An explicit "
             "value that disagrees with the checkpoint is an error, not an "
             "override.")
    parser.add_argument(
        "--dim_hidden", type=int, default=None,
        help="Hidden width. Same resolution rule as --num_inds; small default 64.")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument(
        "--num_post_sab", type=int, default=None,
        help="SAB blocks after the PMA. Same resolution rule as --num_inds; "
             "default 2. 0 is the ClusterHunt-style PMA -> Linear head.")
    parser.add_argument("--no_layer_norm", action="store_true")
    parser.add_argument(
        "--st_weight_channel", action="store_true", default=True,
        help="Append the normalized weight (scaled by N) as an extra input "
             "channel. Default ON, and effectively mandatory here: the "
             "particle support is constant, so an unweighted encoder reads "
             "the same set at every step.")
    parser.add_argument(
        "--no_st_weight_channel", dest="st_weight_channel",
        action="store_false",
        help="The unweighted ablation. The arm is then no longer "
             "information-matched to CGF.")
    parser.add_argument(
        "--pretrained_st_model_path", type=str, default=None,
        help="Encoder checkpoint from 3_train_st.py. Its geometry must match "
             "this run's --num_encodings / --dim_encoder / --num_inds / "
             "--dim_hidden / --num_heads / layer norm and weight-channel "
             "setting, and its dataset's particle_scale must match "
             "--arena_scale (PITFALLS.md section 4).")
    parser.add_argument(
        "--st_frozen", action="store_true",
        help="Freeze the encoder; only the policy/value MLP learns. Use "
             "together with --pretrained_st_model_path.")
    parser.add_argument(
        "--st_encoder_lr_scale", type=float, default=1.0,
        help="Finetune fix 1: multiply the encoder's learning rate by this "
             "(heads keep --learning_rate; --lr_anneal applies to both). 1.0 = "
             "off, i.e. the shared-LR finetune that collapsed. Requires "
             "--pretrained_st_model_path and not --st_frozen.")
    parser.add_argument(
        "--st_unfreeze_at", type=int, default=None,
        help="Finetune fix 2: keep the pretrained encoder frozen for this many "
             "environment steps, then release it for the rest of training. "
             "Requires --pretrained_st_model_path; incompatible with --st_frozen "
             "(which freezes for the whole run).")
    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    if (args.st_encoder_lr_scale != 1.0 or args.st_unfreeze_at is not None) \
            and not args.pretrained_st_model_path:
        parser.error("--st_encoder_lr_scale / --st_unfreeze_at are finetune "
                     "fixes for a PRETRAINED encoder; pass --pretrained_st_model_path.")
    if args.st_frozen and (args.st_encoder_lr_scale != 1.0 or args.st_unfreeze_at is not None):
        parser.error("--st_frozen freezes the encoder for the whole run; the "
                     "finetune fixes have nothing to act on. Drop one.")
    if args.st_encoder_lr_scale <= 0:
        parser.error("--st_encoder_lr_scale must be positive")

    if args.st_frozen and not args.pretrained_st_model_path:
        parser.error(
            "--st_frozen without --pretrained_st_model_path would freeze a "
            "RANDOM encoder, which is a capacity control and not the "
            "pretrained arm. Drop --st_frozen to train the encoder under "
            "PPO, or pass a checkpoint.")

    resolve_st_geometry(args, parser)

    (_run_dir, log_dir, model_save_path, run_subdir,
     particle_filter_class) = resolve_common(args, encoder)

    policy_kwargs = {
        "features_extractor_class": SetTransformerFeaturesExtractor,
        "features_extractor_kwargs": dict(
            num_encodings=args.num_encodings,
            dim_encoder=args.dim_encoder,
            num_inds=args.num_inds,
            dim_hidden=args.dim_hidden,
            num_heads=args.num_heads,
            num_post_sab=args.num_post_sab,
            ln=not args.no_layer_norm,
            arena_scale=args.arena_scale,
            weight_channel=args.st_weight_channel,
            pretrained_st_model_path=args.pretrained_st_model_path,
            st_frozen=args.st_frozen,
        ),
    }
    net_arch = net_arch_from_args(args)
    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    post_construct = None
    if args.pretrained_st_model_path:
        # Fix 2 starts frozen and releases later; the reload must therefore
        # freeze whenever either flag asks for it.
        freeze_at_start = args.st_frozen or args.st_unfreeze_at is not None

        def post_construct(model):
            reload_pretrained_encoder(
                model, args.pretrained_st_model_path, freeze_at_start)
            if args.st_encoder_lr_scale != 1.0:
                scale_encoder_learning_rate(model, args.st_encoder_lr_scale)
            if args.st_unfreeze_at is not None:
                print(f"ST encoder frozen until step {args.st_unfreeze_at:,}, "
                      "then finetuned")

    finetune_callbacks = []
    if args.st_encoder_lr_scale != 1.0:
        finetune_callbacks.append(EncoderLRLoggingCallback())
    if args.st_unfreeze_at is not None:
        finetune_callbacks.append(UnfreezeEncoderCallback(args.st_unfreeze_at))

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
        post_construct=post_construct,
        # TWO callbacks, on purpose. The shared one keeps st/feat_std_mean
        # (last-batch definition) and the feature-norm quantiles comparable
        # with the Ant-Tag runs. The Odd-Even one adds the SCALE-FREE sentinel
        # (st/feat_std_relative, st/feat_std_mean_rollout) computed by
        # re-encoding exactly the rollout buffer's n_steps * n_envs
        # observations, because the shared reading is a 1-sample std at
        # n_envs=1 and its absolute scale does not separate live from
        # collapsed on this domain. It records under its own keys so it
        # cannot overwrite the shared callback's. PITFALLS.md section 6's
        # ~0.01 abort threshold does NOT transfer here; neither callback
        # aborts.
        extra_callbacks=[STFeatureLoggingCallback(),
                         OddEvenSTFeatureSentinel()] + finetune_callbacks,
    )


if __name__ == "__main__":
    main()
