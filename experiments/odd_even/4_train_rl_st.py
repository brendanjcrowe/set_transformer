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


class _ScaledLRParamGroup(dict):
    """An optimizer param group whose ``lr`` is always ``lr_scale`` x the value written.

    SB3's ``utils.update_learning_rate`` does ``param_group["lr"] = lr`` on every
    group at the start of each ``train()``, which would wipe out a plain second
    group's smaller rate. torch keeps the very dict object it was given in
    ``optimizer.param_groups``, so a dict subclass intercepting ``__setitem__``
    for ``"lr"`` makes the scale stick through every schedule update, with no
    change to the shared training loop and nothing extra for ``model.save`` to
    pickle (``state_dict()`` copies groups into plain dicts).
    """

    def __setitem__(self, key, value):
        if key == "lr":
            value = value * dict.get(self, "lr_scale", 1.0)
        super().__setitem__(key, value)


class EncoderLRLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        groups = self.model.policy.optimizer.param_groups
        self.logger.record("train/encoder_learning_rate", float(groups[0]["lr"]))


def _group_collapsing(optimizer_class):
    """Optimizer subclass whose saved ``state_dict`` looks like a FRESH
    single-group optimizer over all parameters.

    ``PPO.load`` rebuilds a plain policy (one param group) and then calls
    ``optimizer.load_state_dict`` with ``exact_match=True``; a two-group state
    dict is refused ("different number of parameter groups") and every saved
    agent -- final, best_model, checkpoints -- becomes unloadable. Evaluation
    never needs the Adam moments, so the state is dropped and the groups
    merged. Resuming TRAINING from such a save restarts the moments; that is
    the documented cost of fix 1.
    """
    class GroupCollapsing(optimizer_class):
        def state_dict(self):
            sd = super().state_dict()
            groups = sd["param_groups"]
            n = sum(len(g["params"]) for g in groups)
            plain = {k: v for k, v in groups[-1].items() if k not in ("params", "lr_scale")}
            return {"state": {}, "param_groups": [{**plain, "params": list(range(n))}]}
    GroupCollapsing.__name__ = f"{optimizer_class.__name__}GroupCollapsing"
    return GroupCollapsing


def scale_encoder_learning_rate(model, scale: float) -> None:
    """Finetune-collapse fix 1: give the encoder its own, smaller learning rate.

    Rebuilds the policy optimizer with two param groups: the encoder in a
    :class:`_ScaledLRParamGroup` carrying ``lr_scale``, everything else plain.
    ``--lr_anneal`` still applies to both (the scaled group tracks the schedule
    at ``scale`` x the rate).
    """
    policy = model.policy
    encoder_params = list(policy.features_extractor.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    other_params = [p for p in policy.parameters() if id(p) not in encoder_ids]
    base_lr = model.lr_schedule(1.0)
    encoder_group = _ScaledLRParamGroup(params=encoder_params, lr_scale=scale)
    encoder_group["lr"] = base_lr          # -> base_lr * scale via __setitem__
    policy.optimizer = _group_collapsing(policy.optimizer_class)(
        [encoder_group, {"params": other_params, "lr": base_lr}],
        lr=base_lr, **policy.optimizer_kwargs)
    assert policy.optimizer.param_groups[0] is encoder_group
    print(f"ST encoder learning rate scaled by {scale} "
          f"({len(encoder_params)} encoder tensors at {encoder_group['lr']:.2e}, "
          f"{len(other_params)} head tensors at {base_lr:.2e}; anneal applies to both)")


class UnfreezeEncoderCallback(BaseCallback):
    """Finetune-collapse fix 2: keep the encoder frozen for the first
    ``unfreeze_at`` environment steps, then release it.

    The heads first learn to read the pretrained code while it cannot move;
    only then does the encoder see gradients, which by that point are
    informative rather than the noise a random head emits. The encoder's
    parameters were in the optimizer all along (SB3 builds it over
    ``policy.parameters()`` regardless of requires_grad), so flipping the flag
    is sufficient. Logs ``st/encoder_trainable`` so the switch is visible in
    TensorBoard, and prints once.
    """

    def __init__(self, unfreeze_at: int):
        super().__init__()
        self.unfreeze_at = int(unfreeze_at)
        self.done = False

    def _encoders(self):
        found = [self.model.policy.features_extractor]
        for attr in ("actor", "critic", "critic_target"):
            module = getattr(self.model.policy, attr, None)
            other = getattr(module, "features_extractor", None)
            if other is not None and other is not found[0]:
                found.append(other)
        return found

    def _on_step(self) -> bool:
        if not self.done and self.num_timesteps >= self.unfreeze_at:
            n = 0
            for extractor in self._encoders():
                # The extractor's own flag wraps forward() in torch.no_grad()
                # when set (st.py); requires_grad alone would leave the
                # encoder trainable in name only.
                extractor.st_frozen = False
                extractor.encoder.train()
                for param in extractor.encoder.parameters():
                    param.requires_grad_(True)
                    n += 1
            self.done = True
            print(f"UnfreezeEncoderCallback: encoder UNFROZEN at step "
                  f"{self.num_timesteps:,} ({n} tensors now trainable)", flush=True)
        self.logger.record("st/encoder_trainable", float(self.done))
        return True


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
    _pretrained.verify_matches_checkpoint(reference, live, path, label="ST encoder")


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
