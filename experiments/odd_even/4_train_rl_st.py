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

# `variants` is loaded BY PATH, not as a flat name: experiments/ant_tag/ has a
# variants.py too, and sys.modules is process-wide, so a plain
# `import variants` returns whichever one was imported first anywhere in the
# process. See _sibling.py -- this was measured happening in the test suite.
import _sibling  # noqa: E402
variants = _sibling.load("variants")

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

# The Odd-Even collapse sentinel. The shared STFeatureLoggingCallback logs a
# 1-sample std at n_envs=1 (so it reads NaN), and its absolute statistic does
# not separate a healthy encoder from a collapsed one on this domain -- a
# healthy 8-epoch encoder reads 1.5e-2 while a genuinely dead 30-epoch one
# reads 9.0e-5. Both are logged; only the relative one is the sentinel. See
# st_feature_sentinel.py for the measurements.
OddEvenSTFeatureSentinel = _sibling.load(
    "st_feature_sentinel").OddEvenSTFeatureSentinel


def reload_pretrained_encoder(model, path: str, frozen: bool,
                              verify: bool = True) -> None:
    """Reload the pretrained encoder AFTER PPO construction, and verify it.

    See this module's docstring and PITFALLS.md section 1. Three steps, all
    required:

    1. Reload. The extractor's own __init__ already loaded these weights and
       SB3's _build then overwrote them with orthogonal-init noise. Nothing
       re-initializes the policy after this point.
    2. Re-freeze. requires_grad does survive apply(), but re-setting it keeps
       the freeze and the load in one place.
    3. Verify max|delta| == 0 against the checkpoint. Without this the two
       states -- correctly loaded, and frozen noise -- are indistinguishable
       in the logs, which is exactly how two 6M-step runs were lost.
    """
    extractor = model.policy.features_extractor
    extractor._load_pretrained_encoder(path, extractor.dim_input)
    if frozen:
        extractor.encoder.eval()
        for param in extractor.encoder.parameters():
            param.requires_grad_(False)
    print("SetTransformerFeaturesExtractor: encoder RE-loaded after PPO "
          "construction (SB3 init_weights would otherwise overwrite it)"
          + (" and re-frozen" if frozen else ""))

    # SAC-style policies build critics that may hold their own extractor.
    # PPO does not, but the loop costs nothing and the omission would be
    # silent.
    for attr in ("actor", "critic", "critic_target"):
        module = getattr(model.policy, attr, None)
        other = getattr(module, "features_extractor", None)
        if other is not None and other is not extractor:
            other._load_pretrained_encoder(path, other.dim_input)
            if frozen:
                other.encoder.eval()
                for param in other.encoder.parameters():
                    param.requires_grad_(False)

    if verify:
        assert_encoder_matches_checkpoint(model, path)

    # The pretrained path has done its job. Leaving it in policy_kwargs would
    # bake an ABSOLUTE path to the pretraining checkpoint into every saved
    # policy, and SB3 re-runs the extractor constructor on load -- so
    # evaluating the trained agent would die with FileNotFoundError once the
    # pretraining directory moved, even though the trained weights are in the
    # zip (PITFALLS.md section 7).
    model.policy_kwargs["features_extractor_kwargs"][
        "pretrained_st_model_path"] = None


def assert_encoder_matches_checkpoint(model, path: str) -> None:
    """max|delta| == 0 between the live encoder and the checkpoint.

    The verification snippet from PITFALLS.md section 1, run once per
    training start. Costs milliseconds and is the only positive evidence
    that the reload landed.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = (checkpoint["model_state_dict"]
             if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
             else checkpoint)
    prefix = "set_transformer."
    reference = {key[len(prefix):]: value for key, value in state.items()
                 if key.startswith(prefix)}
    if not reference:
        reference = {key: value for key, value in state.items()
                     if not key.startswith("decoder.")}
    live = {key[len("features_extractor.encoder."):]: value
            for key, value in model.policy.state_dict().items()
            if key.startswith("features_extractor.encoder.")}
    shared = [key for key in reference if key in live]
    if not shared:
        raise AssertionError(
            f"{path} shares no encoder parameter names with the live policy; "
            "the reload cannot have done anything.")
    worst = max(float((reference[key] - live[key].cpu()).abs().max())
                for key in shared)
    assert worst == 0.0, (
        f"pretrained encoder does not match {path} after PPO construction "
        f"(max|delta| = {worst}). SB3's init_weights overwrote it; the "
        "reload must happen AFTER PPO(...) returns.")
    print(f"Verified: encoder matches {path} exactly "
          f"({len(shared)} tensors, max|delta| = 0.0)")


def main() -> None:
    encoder = "st"
    parser = argparse.ArgumentParser(
        description="RL with Set Transformer particle-belief features on the "
                    "Odd-Even POMDP.")
    add_common_arguments(parser, encoder)
    parser.add_argument("--num_encodings", type=int, default=8)
    parser.add_argument("--dim_encoder", type=int, default=8)
    parser.add_argument("--num_inds", type=int, default=32)
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
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
    args = parser.parse_args()
    if args.list_variants:
        variants.print_variants()
        return

    if args.st_frozen and not args.pretrained_st_model_path:
        parser.error(
            "--st_frozen without --pretrained_st_model_path would freeze a "
            "RANDOM encoder, which is a capacity control and not the "
            "pretrained arm. Drop --st_frozen to train the encoder under "
            "PPO, or pass a checkpoint.")

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
        def post_construct(model):
            reload_pretrained_encoder(
                model, args.pretrained_st_model_path, args.st_frozen)

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
        # and the feature-norm quantiles comparable with the Ant-Tag runs.
        # The Odd-Even one adds the SCALE-FREE sentinel (st/feat_std_relative)
        # and computes its statistics over the whole rollout, because the
        # shared reading is a 1-sample std at n_envs=1 and its absolute scale
        # does not separate live from collapsed on this domain. PITFALLS.md
        # section 6's ~0.01 abort threshold does NOT transfer here; neither
        # callback aborts.
        extra_callbacks=[STFeatureLoggingCallback(),
                         OddEvenSTFeatureSentinel()],
    )


if __name__ == "__main__":
    main()
