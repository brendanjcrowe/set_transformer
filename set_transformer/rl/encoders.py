"""The encoder table: one entry per belief encoder the RL harness can train.

Change 4 of the harness centralisation (``refactor_plans.md`` in the parent repo,
2026-09-12). Each :class:`Encoder` names the SB3 features-extractor class, says whether it is
learned (and so can be pretrained, frozen or finetuned), and carries four functions the
shared command line (``rl/train.py``) calls in order:

* ``add_arguments(parser, domain)`` -- the encoder's own flag group, with this domain's
  defaults (``Domain.encoder_defaults[name]``) applied;
* ``resolve_arguments(parser, args, domain)`` -- fill flags left at their defaults from a
  pretraining checkpoint and the registry, refuse contradictions with ``parser.error``, size
  readouts, record ``args.encoder_params``;
* ``extractor_kwargs(args)`` -- the ``features_extractor_kwargs`` SB3 receives;
* ``callbacks(features_extractor_kwargs, encoder_options)`` -- the encoder's logging
  callbacks (CGF: tilt norms, drift, the running-norm refresh; ST: feature statistics).

The start mode -- pretrained checkpoint, frozen, encoder learning-rate scale, unfreeze step --
is ONE flag group for every learned encoder (:func:`add_start_mode_arguments`): the generic
spellings ``--pretrained_path`` / ``--frozen`` / ``--encoder_lr_scale`` / ``--unfreeze_at``
plus each encoder's historical spellings as aliases (``--pretrained_st_model_path``,
``--st_frozen``, ``--pretrained_cgf_model_path``, ``--cgf_frozen``, ``--pretrained_model_path``,
...), which the bash drivers and the recorded commands use. The value is stored under the
HISTORICAL name (``pretrained_st_model_path``, ``st_frozen``, ...), so ``run_config.json`` keeps
the keys every recorded run has, and the stored name is the extractor's own constructor
argument. :func:`resolve_start_mode` applies the refusals every arm had.

The CGF flag group and its resolution existed twice before this table -- once per domain,
with different defaults (Ant-Tag: the legacy clamp-2.0 / K / no-norm recipe so recorded runs
reproduce; Odd-Even: tanh-50 / running norm / spread_1d) and three genuinely different rules:
the default tilt bound (Ant-Tag: the registry's ``cgf_t_bound``; Odd-Even: 50), the default
initial tilt size (Ant-Tag: 0.8 x bound in spread mode, else unset; Odd-Even: 40 in tanh, the
clamp in clamp mode) and the particle dimension. Those three come from the ``Domain``; the
rest is one body. Two behaviours are unified on the stricter side: a checkpoint fitted at
another ``arena_scale`` is refused on both domains (Odd-Even used to take the checkpoint's
scale), and ``t_bound`` is recorded as null in clamp mode on both (Odd-Even recorded its 50).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass, asdict

import numpy as np
import torch

from set_transformer.rl.domains.base import Domain
from set_transformer.rl.feature_extractors.cgf import (
    EncoderDriftLoggingCallback,
    RolloutFeatureNormCallback,
    TNormLoggingCallback,
    WeightedCGFFeaturesExtractor,
    cgf_raw_dim,
    matched_readout_hidden,
    non_readout_param_count,
    readout_param_count,
)
from set_transformer.rl.feature_extractors.gaussian import WeightedGaussianFeaturesExtractor
from set_transformer.rl.feature_extractors.pooled import (
    PointNetFeaturesExtractor,
    WeightedDeepSetFeaturesExtractor,
    WeightedKMomentsFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.st import (
    STFeatureLoggingCallback,
    SetTransformerFeaturesExtractor,
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartMode:
    """How the encoder starts and what PPO may do to it. ``kind`` is ``e2e`` (random init,
    trained under PPO), ``frozen`` (pretrained, fixed for the whole run) or ``finetune``
    (pretrained, trained under PPO -- at a scaled rate and/or after an unfreeze step)."""

    pretrained_path: str | None = None
    frozen: bool = False
    encoder_lr_scale: float = 1.0
    unfreeze_at: int | None = None

    @property
    def kind(self) -> str:
        if not self.pretrained_path:
            return "e2e"
        return "frozen" if self.frozen else "finetune"

    def as_record(self) -> dict:
        return dict(kind=self.kind, pretrained_path=self.pretrained_path, frozen=self.frozen,
                    encoder_lr_scale=self.encoder_lr_scale, unfreeze_at=self.unfreeze_at)


@dataclass(frozen=True)
class Encoder:
    """One belief encoder, as the trainer's command line sees it (module docstring)."""

    name: str
    extractor_class: type
    learned: bool
    add_arguments: Callable[[argparse.ArgumentParser, Domain], None]
    resolve_arguments: Callable[[argparse.ArgumentParser, argparse.Namespace, Domain], None]
    extractor_kwargs: Callable[[argparse.Namespace], dict]
    callbacks: Callable[[dict, dict], list]
    #: ``encoder_options(args) -> dict``: settings that steer the callbacks but are not
    #: extractor constructor arguments (CGF: ``running_norm_update``).
    encoder_options: Callable[[argparse.Namespace], dict] = lambda args: {}
    #: Where the four start-mode values are stored in ``args`` / the run record (learned only).
    pretrained_dest: str | None = None
    frozen_dest: str | None = None
    lr_scale_dest: str | None = None
    unfreeze_dest: str | None = None
    #: Historical spellings of the four start-mode flags, in that order.
    start_mode_aliases: tuple[tuple[str, ...], ...] = ((), (), (), ())

    def start_mode(self, args: argparse.Namespace) -> StartMode:
        if not self.learned:
            return StartMode()
        return StartMode(
            pretrained_path=getattr(args, self.pretrained_dest) or None,
            frozen=bool(getattr(args, self.frozen_dest)),
            encoder_lr_scale=float(getattr(args, self.lr_scale_dest)),
            unfreeze_at=getattr(args, self.unfreeze_dest),
        )


# ---------------------------------------------------------------------------
# Start mode: one flag group for every learned encoder
# ---------------------------------------------------------------------------


GENERIC_START_MODE_FLAGS = ("--pretrained_path", "--frozen", "--encoder_lr_scale", "--unfreeze_at")


def add_start_mode_arguments(parser: argparse.ArgumentParser, encoder: Encoder) -> None:
    """The pretrained / frozen / lr-scale / unfreeze flags, generic spellings first, the
    encoder's historical spellings as aliases, stored under the historical names."""
    if not encoder.learned:
        return
    a_path, a_frozen, a_scale, a_unfreeze = encoder.start_mode_aliases
    group = parser.add_argument_group(
        "start mode", "Omit --pretrained_path to train the encoder from a random init under "
        "PPO (the arm that matches CGF). Every flag also answers to its historical spelling.")
    group.add_argument(
        "--pretrained_path", *a_path, dest=encoder.pretrained_dest, type=str, default=None,
        help="Pretraining checkpoint for the encoder. Loaded into the extractor, RE-loaded "
             "after PPO construction (SB3's init_weights would otherwise overwrite it) and "
             "verified max|delta| == 0 (PITFALLS.md section 1). Geometry flags left at their "
             "defaults are taken from it; its arena_scale must equal the variant's.")
    group.add_argument(
        "--frozen", *a_frozen, dest=encoder.frozen_dest, action="store_true",
        help="Freeze the pretrained encoder; only the policy/value MLP learns. Requires "
             "--pretrained_path.")
    group.add_argument(
        "--encoder_lr_scale", *a_scale, dest=encoder.lr_scale_dest, type=float, default=1.0,
        help="Finetune-collapse fix 1: the pretrained encoder learns at this x the head "
             "learning rate (--lr_anneal applies to both). 1.0 = off, the shared-rate "
             "finetune. Requires --pretrained_path and not --frozen. PPO only.")
    group.add_argument(
        "--unfreeze_at", *a_unfreeze, dest=encoder.unfreeze_dest, type=int, default=None,
        help="Finetune-collapse fix 2: keep the pretrained encoder frozen for this many "
             "environment steps, then release it for the rest of training. Requires "
             "--pretrained_path; incompatible with --frozen.")


def check_pretrained_path(parser: argparse.ArgumentParser, args: argparse.Namespace,
                          encoder: Encoder) -> None:
    """Refuse a missing checkpoint file before anything tries to load it."""
    if not encoder.learned:
        return
    path = getattr(args, encoder.pretrained_dest)
    if path and not os.path.isfile(path):
        parser.error(f"--pretrained_path {path} does not exist")


def resolve_start_mode(parser: argparse.ArgumentParser, args: argparse.Namespace,
                       encoder: Encoder, algorithm: str = "PPO",
                       resume_from: str | None = None) -> StartMode:
    """The start mode for this run, after the refusals every arm applied."""
    if not encoder.learned:
        return StartMode()
    check_pretrained_path(parser, args, encoder)
    start = encoder.start_mode(args)
    if start.frozen and not start.pretrained_path:
        parser.error("--frozen without --pretrained_path would freeze a RANDOM encoder. Pass "
                     "a checkpoint, or drop --frozen to train the encoder under PPO.")
    if start.encoder_lr_scale <= 0:
        parser.error("--encoder_lr_scale must be positive")
    if start.encoder_lr_scale != 1.0 and not (start.pretrained_path or resume_from):
        parser.error("--encoder_lr_scale is a finetune fix for a PRETRAINED encoder; pass "
                     "--pretrained_path (an end-to-end encoder has no pretrained geometry to "
                     "protect).")
    if start.encoder_lr_scale != 1.0 and start.frozen:
        parser.error("--frozen freezes the encoder for the whole run; --encoder_lr_scale has "
                     "nothing to act on. Drop one.")
    if start.encoder_lr_scale != 1.0 and algorithm.upper() != "PPO":
        parser.error("--encoder_lr_scale is implemented for PPO only")
    if start.unfreeze_at is not None and not start.pretrained_path:
        parser.error("--unfreeze_at is a finetune fix for a PRETRAINED encoder; pass "
                     "--pretrained_path.")
    if start.unfreeze_at is not None and start.frozen:
        parser.error("--frozen freezes the encoder for the whole run; --unfreeze_at has "
                     "nothing to act on. Drop one.")
    return start


def checkpoint_config(path: str | None) -> dict | None:
    """The ``config`` a pretraining checkpoint carries, as a JSON-able dict, or None."""
    if not path:
        return None
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    config = loaded.get("config") if isinstance(loaded, dict) else None
    if config is None:
        return None
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    elif not isinstance(config, dict):
        config = dict(vars(config)) if hasattr(config, "__dict__") else {"repr": repr(config)}
    return {str(k): v for k, v in config.items()}


def _checkpoint_config_for_flags(path: str) -> dict:
    """``config`` of a checkpoint, dict or dataclass, as a plain dict (missing -> {})."""
    return checkpoint_config(path) or {}


# ---------------------------------------------------------------------------
# CGF (one flag group and one resolver for both domains; module docstring)
# ---------------------------------------------------------------------------


#: CGF geometry a pretrained checkpoint carries in its ``config``. With a checkpoint, a flag
#: left at its default takes the checkpoint's value and an explicit value that disagrees is
#: an error; ``arena_scale`` is resolved from the variant first and the checkpoint must agree.
CGF_GEOMETRY_FLAGS = ("num_cgf_features", "feature_mode", "t_param", "t_bound",
                      "t_clamp", "feature_norm", "readout_hidden", "readout_depth",
                      "readout_dim", "arena_scale", "t_init_mode", "t_init_max",
                      "x_embed_dim", "x_embed_hidden", "x_embed_depth")


def add_readout_and_pretrained_arguments(parser) -> None:
    """The readout MLP and the pretrained-encoder flags (2026-09-05), for
    ``3_pretrain_st_belief.py --encoder cgf``: the pretraining side spells a
    checkpoint's geometry with the same flags the RL side reads, so a checkpoint
    built there loads here without translation. The shared RL command line has
    its own copy of these flags in :func:`_cgf_add_arguments`; this trio is what
    the pretraining script borrows (moved from
    ``experiments/odd_even/4_train_rl_cgf.py`` in change 5.3a, verbatim).
    """
    parser.add_argument(
        "--readout_hidden", type=int, default=0,
        help="Width of the readout MLP between the (normalised) CGF block and "
             "the policy. 0 (default) = no readout, the extractor as before. "
             "This is where a parameter-matched CGF arm keeps its budget; see "
             "--match_params.")
    parser.add_argument(
        "--readout_depth", type=int, default=0,
        help="Hidden layers in the readout MLP. 0 = none. --match_params "
             "with depth 0 uses 2 (ClusterHunt's).")
    parser.add_argument(
        "--readout_dim", type=int, default=None,
        help="Readout output width. Default 64, the ST arm's feature width, "
             "whatever --num_cgf_features is, so the policy heads are "
             "identical across arms.")
    parser.add_argument(
        "--match_params", type=int, default=None,
        help="Pick --readout_hidden so the encoder's parameter total (learned "
             "t + norm affine + readout) lands closest to this count, e.g. "
             "109448 for the small 2-SAB ST encoder. The chosen width and "
             "the exact total are printed and recorded in run_config.json.")
    parser.add_argument(
        "--pretrained_cgf_model_path", type=str, default=None,
        help="Checkpoint from 3_pretrain_st_belief.py --encoder cgf. Loaded "
             "into the extractor, RE-loaded after PPO construction and "
             "verified max|delta| == 0 (PITFALLS.md section 1). Geometry "
             "flags left at their defaults are taken from it.")
    parser.add_argument(
        "--x_embed_dim", type=int, default=0,
        help="Learned per-particle embedding phi: R^D -> R^d before the CGF, "
             "with t in R^d (idea 3, 2026-09-05). 0 = off, the plain CGF. NOTE "
             "this makes the arm a learned Deep-Set-family encoder, not a "
             "parameter-free statistic; report it as such.")
    parser.add_argument("--x_embed_hidden", type=int, default=64)
    parser.add_argument("--x_embed_depth", type=int, default=1)
    parser.add_argument(
        "--cgf_frozen", action="store_true",
        help="Freeze the WHOLE pretrained encoder: t, the running-norm "
             "statistics and the readout. Only PPO's heads learn -- the arm "
             "that matches the ST's --st_frozen. Requires a checkpoint.")


def resolve_t_init_max(args, tanh_default: float = 40.0) -> None:
    """Fill a None --t_init_max: ``tanh_default`` in tanh mode, t_clamp in
    clamp mode. Called AFTER resolve_cgf_geometry so a checkpoint's value
    wins, and BEFORE resolve_common so run_config.json records the number
    that ran. The extractor refuses t_init_max > t_clamp in clamp mode; this
    is what keeps a bare ``--t_param clamp`` run legal."""
    if args.t_init_max is None:
        args.t_init_max = (float(tanh_default) if args.t_param != "clamp"
                           else float(args.t_clamp))


def resolve_cgf_geometry(args, parser, particle_dim: int = 1) -> None:
    """CLI > checkpoint config > default for the geometry flags, then size
    the readout if --match_params asks for it. Prints the resulting encoder
    parameter total so every run's size is on record. ``particle_dim`` is the
    problem's (Odd-Even: 1, a scalar state)."""
    from_ckpt = {}
    if args.pretrained_cgf_model_path:
        import torch
        checkpoint = torch.load(args.pretrained_cgf_model_path,
                                map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        from_ckpt = {k: config[k] for k in CGF_GEOMETRY_FLAGS if k in config}
        for key, ckpt_value in from_ckpt.items():
            given = getattr(args, key)
            if given == parser.get_default(key):
                setattr(args, key, ckpt_value)
            elif given != ckpt_value:
                parser.error(
                    f"--{key} {given!r} disagrees with the checkpoint's {key}="
                    f"{ckpt_value!r} ({args.pretrained_cgf_model_path}). Drop the "
                    "flag to take the checkpoint's geometry.")
        if args.match_params is not None:
            parser.error("--match_params sizes a NEW readout; with a pretrained "
                         "checkpoint the readout shape comes from the checkpoint.")
    t_dim = args.x_embed_dim if args.x_embed_dim > 0 else particle_dim
    raw_dim = cgf_raw_dim(args.num_cgf_features, t_dim, args.feature_mode)
    fixed = non_readout_param_count(args.num_cgf_features, particle_dim,
                                    args.feature_mode, args.t_frozen, args.feature_norm,
                                    args.x_embed_dim, args.x_embed_hidden, args.x_embed_depth)
    if args.match_params is not None:
        if args.readout_depth <= 0:
            args.readout_depth = 2
        out_dim = (args.readout_dim if args.readout_dim is not None
                   else WeightedCGFFeaturesExtractor.DEFAULT_READOUT_DIM)
        args.readout_hidden, total = matched_readout_hidden(
            args.match_params, raw_dim, args.readout_depth, out_dim, fixed)
        print(f"CGF readout sized to match {args.match_params:,} params: "
              f"hidden={args.readout_hidden} depth={args.readout_depth} "
              f"-> encoder total {total:,} ({100 * (total - args.match_params) / args.match_params:+.2f}%)")
    else:
        out_dim = args.readout_dim if args.readout_dim is not None else (
            WeightedCGFFeaturesExtractor.DEFAULT_READOUT_DIM if args.readout_depth > 0 else raw_dim)
        total = fixed + readout_param_count(raw_dim, args.readout_hidden,
                                            args.readout_depth, out_dim)
        print(f"CGF encoder parameters: {total:,} (raw block {raw_dim}, "
              f"readout hidden={args.readout_hidden} depth={args.readout_depth})")
    args.encoder_params = int(total)
    if from_ckpt:
        print(f"CGF geometry taken from checkpoint: "
              + ", ".join(f"{k}={getattr(args, k)!r}" for k in from_ckpt))



def _cgf_add_arguments(parser: argparse.ArgumentParser, domain: Domain) -> None:
    d = domain.encoder_defaults.get("cgf", {})
    planar = domain.particle_dim >= 2
    # `spread` lays 8 planar directions and rejects 1-D particles; `spread_1d` is its 1-D
    # analogue (log-spaced magnitudes in both signs). Each domain is offered the one that
    # fits its particles, as the two scripts did.
    init_modes = ("linspace_all_dims", "linspace_first_dim", "random") + (
        ("spread",) if planar else ("spread_1d",))
    # `polar` learns a direction and a magnitude separately; in 1-D the direction is a sign
    # and the extractor refuses it.
    t_params = ("clamp", "tanh", "polar") if planar else ("clamp", "tanh")
    g = parser.add_argument_group("cgf encoder")
    g.add_argument("--num_cgf_features", type=int, default=64)
    g.add_argument(
        "--t_init_mode", type=str, default=d.get("t_init_mode", "linspace_all_dims"),
        choices=init_modes,
        help="How the probes t start. spread / spread_1d: log-spaced magnitudes from 0.25 up "
             "to --t_init_max, which starts where the signal already is; the 0.1-scale "
             "linspace inits have to grow ||t|| ~10x into place first.")
    g.add_argument(
        "--t_frozen", action="store_true",
        help="Register t_values as a BUFFER instead of a Parameter, so PPO cannot learn the "
             "projection directions. With a spread init this isolates representational "
             "CAPACITY from the optimization dynamics of t_j growth.")
    g.add_argument("--t_init_scale", type=float, default=0.1)
    g.add_argument("--t_clamp", type=float, default=2.0,
                   help="clamp mode only: the hard bound on each t component.")
    g.add_argument(
        "--exp_arg_clamp", type=float, default=20.0,
        help="DEPRECATED, no longer applied: the CGF is computed with logsumexp, which needs "
             "no clamp on the exponent. Accepted and recorded in run_config.json for "
             "compatibility only.")
    g.add_argument(
        "--t_param", type=str, default=d.get("t_param", "clamp"), choices=t_params,
        help="How the learned t is bounded. 'clamp': hard torch.clamp at +-t_clamp, zero "
             "gradient beyond it (the legacy Ant-Tag mode). 'tanh': t = t_bound * tanh(raw_t), "
             "a smooth box (Odd-Even's default). 'polar': t = t_bound * sigmoid(a) * v/|v|, a "
             "smooth BALL with direction and magnitude learned separately -- the recommended "
             "2-D mode.")
    g.add_argument(
        "--t_bound", type=float, default=d.get("t_bound"),
        help="tanh / polar only: the largest tilt. Pick it by t_bound * (decision-relevant "
             "length in normalized units) ~ 2..4. Default: this domain's rule (Ant-Tag: the "
             "registry's cgf_t_bound = 3 / (tag_radius / arena half-width); Odd-Even: 50).")
    g.add_argument(
        "--t_init_max", type=float, default=None,
        help="spread / spread_1d init: the largest probe norm of the log-spaced grid. Default: "
             "this domain's rule (Ant-Tag: unset in clamp mode -- the extractor's legacy 2.8 "
             "ceiling -- and 0.8 * t_bound otherwise; Odd-Even: 40 in tanh mode, t_clamp in "
             "clamp mode). An explicit value above t_clamp in clamp mode is refused.")
    g.add_argument(
        "--feature_mode", type=str, default="K", choices=["K", "K_grad", "both"],
        help="K (legacy): log-MGF at each probe, T features. K_grad: the tilted mean K'(t), "
             "T x D features, bounded by the particle range whatever t is. both: concatenated.")
    g.add_argument(
        "--feature_norm", type=str, default=d.get("feature_norm", "none"),
        choices=["none", "running", "layernorm"],
        help="Standardise the CGF block before the policy MLP. 'none': raw. 'running': "
             "per-feature z-score, statistics fixed per PPO cycle (see --running_norm_update). "
             "'layernorm': across features per sample (RunningFeatureNorm docstring).")
    g.add_argument(
        "--running_norm_update", type=str, default="rollout", choices=["rollout", "minibatch"],
        help="--feature_norm running only. 'rollout': statistics held fixed for each collect + "
             "update cycle and refreshed from the rollout buffer between cycles "
             "(RolloutFeatureNormCallback). 'minibatch': the pre-fix lerp on every minibatch, "
             "A/B control only (PITFALLS.md section 8 item 5).")
    g.add_argument(
        "--readout_hidden", type=int, default=0,
        help="Width of an MLP readout between the (normalised) CGF block and the policy. "
             "0 = none. Where a parameter-matched CGF arm keeps its budget; see --match_params.")
    g.add_argument("--readout_depth", type=int, default=0,
                   help="Hidden layers of the readout. 0 = none. --match_params with depth 0 uses 2.")
    g.add_argument(
        "--readout_dim", type=int, default=None,
        help="Readout output width; default 64 (the ST arm's 8 x 8) whatever the probe count, "
             "so the policy heads match across arms.")
    g.add_argument(
        "--match_params", type=int, default=None,
        help="Pick --readout_hidden so the encoder total (learned t + norm affine + readout + "
             "embedding) lands closest to this count, e.g. the ST arm's. Printed and recorded "
             "as encoder_params.")
    g.add_argument(
        "--x_embed_dim", type=int, default=0,
        help="Learned per-particle embedding phi: R^D -> R^d before the CGF (t then lives in "
             "R^d). 0 = off. Makes the arm a learned Deep-Set-family encoder, not a "
             "parameter-free statistic.")
    g.add_argument("--x_embed_hidden", type=int, default=64)
    g.add_argument("--x_embed_depth", type=int, default=1)


def _cgf_resolve_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace,
                           domain: Domain) -> None:
    """CLI > checkpoint > domain rule for the CGF flags, in place; then size the readout.

    Order, as the Ant-Tag script had it: ``arena_scale`` is already resolved (the shared
    command line does that from the domain before any encoder resolution); with a checkpoint,
    geometry flags at their defaults take the checkpoint's values, explicit disagreements are
    errors, and the checkpoint's ``arena_scale`` must equal the variant's; then ``t_bound``
    (tanh / polar: the domain's rule when None) and ``t_init_max`` (the domain's rule when
    None); then ``--match_params`` sizes the readout and ``encoder_params`` is recorded.
    """
    d = domain.encoder_defaults.get("cgf", {})
    path = args.pretrained_cgf_model_path
    from_ckpt = {}
    if path:
        config = _checkpoint_config_for_flags(path)
        from_ckpt = {k: config[k] for k in CGF_GEOMETRY_FLAGS if k in config}
        for key, ckpt_value in from_ckpt.items():
            given = getattr(args, key)
            if key == "arena_scale":
                # Resolved from the variant already; the checkpoint has to agree.
                if not np.isclose(float(given), float(ckpt_value), rtol=1e-6, atol=1e-9):
                    parser.error(
                        f"checkpoint {path} was fitted at arena_scale={ckpt_value!r}, this "
                        f"variant's is {given!r}; the t values are meaningless in another "
                        "frame (PITFALLS.md section 4).")
                continue
            if given == parser.get_default(key):
                setattr(args, key, ckpt_value)
            elif given != ckpt_value:
                parser.error(
                    f"--{key} {given!r} disagrees with the checkpoint's {key}={ckpt_value!r} "
                    f"({path}). Drop the flag to take the checkpoint's geometry.")
        if args.match_params is not None:
            parser.error("--match_params sizes a NEW readout; with a pretrained checkpoint the "
                         "readout shape comes from the checkpoint.")

    if args.t_param == "clamp":
        # Not in force in clamp mode, recorded as null.
        args.t_bound = None
        if args.t_init_max is not None and args.t_init_max > args.t_clamp:
            parser.error(
                f"--t_init_max {args.t_init_max} exceeds --t_clamp {args.t_clamp}: every probe "
                "beyond the clamp would be flattened to +-t_clamp with zero gradient. Use "
                "--t_param polar (or tanh) with a --t_bound, or lower --t_init_max.")
    elif args.t_bound is None:
        hook = d.get("t_bound_default")
        if hook is None:
            parser.error(f"--t_param {args.t_param} needs --t_bound")
        args.t_bound = float(hook(args.variant))
    if args.t_init_max is None:
        hook = d.get("t_init_max_default")
        if hook is not None:
            args.t_init_max = hook(args)
    if (args.t_param != "clamp" and args.t_init_max is not None
            and args.t_init_max >= args.t_bound):
        parser.error(f"--t_init_max {args.t_init_max} must be below --t_bound {args.t_bound}")

    particle_dim = domain.particle_dim
    t_dim = args.x_embed_dim if args.x_embed_dim > 0 else particle_dim
    raw_dim = cgf_raw_dim(args.num_cgf_features, t_dim, args.feature_mode)
    fixed = non_readout_param_count(
        args.num_cgf_features, particle_dim, args.feature_mode, args.t_frozen,
        args.feature_norm, args.x_embed_dim, args.x_embed_hidden, args.x_embed_depth)
    if args.t_param == "polar" and not args.t_frozen:
        fixed += args.num_cgf_features       # the extra magnitude scalar per probe
    if args.match_params is not None:
        if args.readout_depth <= 0:
            args.readout_depth = 2
        out_dim = (args.readout_dim if args.readout_dim is not None
                   else WeightedCGFFeaturesExtractor.DEFAULT_READOUT_DIM)
        args.readout_hidden, total = matched_readout_hidden(
            args.match_params, raw_dim, args.readout_depth, out_dim, fixed)
        print(f"CGF readout sized to match {args.match_params:,} params: "
              f"hidden={args.readout_hidden} depth={args.readout_depth} "
              f"-> encoder total {total:,} "
              f"({100 * (total - args.match_params) / args.match_params:+.2f}%)")
    else:
        out_dim = args.readout_dim if args.readout_dim is not None else (
            WeightedCGFFeaturesExtractor.DEFAULT_READOUT_DIM
            if args.readout_depth > 0 else raw_dim)
        total = fixed + readout_param_count(raw_dim, args.readout_hidden,
                                            args.readout_depth, out_dim)
    args.encoder_params = int(total)
    print(f"CGF encoder: t_param={args.t_param} t_bound={args.t_bound} "
          f"t_init_max={args.t_init_max} feature_mode={args.feature_mode} "
          f"feature_norm={args.feature_norm} raw block {raw_dim} -> "
          f"{args.encoder_params:,} encoder parameters")
    if from_ckpt:
        print("CGF geometry taken from checkpoint: "
              + ", ".join(f"{k}={getattr(args, k)!r}" for k in from_ckpt))


def _cgf_extractor_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        num_cgf_features=args.num_cgf_features,
        arena_scale=args.arena_scale,
        t_init_mode=args.t_init_mode,
        t_init_scale=args.t_init_scale,
        t_clamp=args.t_clamp,
        exp_arg_clamp=args.exp_arg_clamp,
        t_frozen=args.t_frozen,
        t_param=args.t_param,
        t_bound=args.t_bound if args.t_param != "clamp" else None,
        t_init_max=args.t_init_max,
        feature_mode=args.feature_mode,
        feature_norm=args.feature_norm,
        readout_hidden=args.readout_hidden,
        readout_depth=args.readout_depth,
        readout_dim=args.readout_dim,
        pretrained_cgf_model_path=args.pretrained_cgf_model_path,
        cgf_frozen=args.cgf_frozen,
        x_embed_dim=args.x_embed_dim,
        x_embed_hidden=args.x_embed_hidden,
        x_embed_depth=args.x_embed_depth,
    )


def _cgf_callbacks(kwargs: dict, options: dict) -> list:
    """cgf/t_norm_q*: does PPO grow ||t||; cgf/drift_*: relative movement of every extractor
    tensor group since training start (exactly 0 for a frozen encoder). The running norm's
    per-cycle refresh only when that norm is in use and unfrozen (PITFALLS.md section 8
    item 5)."""
    callbacks = [TNormLoggingCallback(), EncoderDriftLoggingCallback()]
    if (kwargs.get("feature_norm") == "running"
            and options.get("running_norm_update", "rollout") == "rollout"
            and not kwargs.get("cgf_frozen", False)):
        callbacks.append(RolloutFeatureNormCallback())
    return callbacks


# ---------------------------------------------------------------------------
# Set Transformer
# ---------------------------------------------------------------------------


#: Geometry a pretraining checkpoint's ``config`` may carry; a flag left unset takes the
#: checkpoint's value, then the domain's default; an explicit disagreement is an error.
ST_GEOMETRY_FLAGS = ("num_inds", "dim_hidden", "num_post_sab")


def _st_add_arguments(parser: argparse.ArgumentParser, domain: Domain) -> None:
    d = domain.encoder_defaults.get("st", {})
    g = parser.add_argument_group("st encoder")
    g.add_argument(
        "--num_encodings", type=int, default=8,
        help="PMA seed vectors. num_encodings x dim_encoder is the belief feature width; the "
             "default 8 x 8 = 64 matches the CGF arm's --num_cgf_features 64.")
    g.add_argument("--dim_encoder", type=int, default=8)
    g.add_argument(
        "--num_inds", type=int, default=None,
        help="ISAB inducing points. Default: the checkpoint's own geometry when "
             f"--pretrained_path is given, else {d.get('num_inds', 32)} on this domain. An "
             "explicit value that disagrees with the checkpoint is an error, not an override.")
    g.add_argument(
        "--dim_hidden", type=int, default=None,
        help=f"Hidden width. Same resolution rule as --num_inds; default {d.get('dim_hidden', 128)}.")
    g.add_argument("--num_heads", type=int, default=4)
    g.add_argument(
        "--num_post_sab", type=int, default=None,
        help="SAB blocks after the PMA. Same resolution rule as --num_inds; default "
             f"{d.get('num_post_sab', 2)}. 0 is the ClusterHunt-style PMA -> Linear head.")
    g.add_argument("--ln", action="store_true", default=True,
                   help="LayerNorm inside the attention blocks. Default on.")
    g.add_argument("--no_ln", "--no_layer_norm", dest="ln", action="store_false")
    g.add_argument(
        "--output_norm", action="store_true", default=False,
        help="Layer-normalise the encoder's flattened code (no learned gain / bias) as its last "
             "operation, so the belief features always have mean 0 and length sqrt(width). Default "
             "off (every checkpoint before 2026-09-19); a pretrained checkpoint must have been built "
             "with the same setting (debug_plans/ch_fixes.md).")
    g.add_argument(
        "--st_weight_channel", "--weight_channel", dest="weight_channel",
        action="store_true", default=True,
        help="Append the normalized PF weight (scaled by N) as an extra per-particle input "
             "channel, so the ST reads the same weighted belief as the CGF and Gaussian arms. "
             "Default on.")
    g.add_argument(
        "--no_st_weight_channel", "--no_weight_channel", dest="weight_channel",
        action="store_false",
        help="Legacy unweighted ST input (coordinates only). The arm is then NOT "
             "information-matched to the CGF baseline.")


def _st_resolve_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace,
                          domain: Domain) -> None:
    """CLI > checkpoint config > domain default for --num_inds / --dim_hidden / --num_post_sab.

    A checkpoint from 3_pretrain_st_belief.py (a dict ``config``) or 3_train_st.py (a
    TrainingConfig) carries its geometry. Taking it from there means the pretrained arm can
    never be launched with the wrong shape, and an explicit flag that disagrees is refused
    rather than silently building a mismatched encoder for the loader to reject later.
    """
    d = domain.encoder_defaults.get("st", {})
    defaults = {"num_inds": d.get("num_inds", 32), "dim_hidden": d.get("dim_hidden", 128),
                "num_post_sab": d.get("num_post_sab", 2)}
    from_ckpt = {}
    path = args.pretrained_st_model_path
    if path:
        config = _checkpoint_config_for_flags(path)
        from_ckpt = {k: config[k] for k in ST_GEOMETRY_FLAGS if config.get(k) is not None}
    for key, fallback in defaults.items():
        given = getattr(args, key)
        if given is None:
            setattr(args, key, from_ckpt.get(key, fallback))
        elif key in from_ckpt and from_ckpt[key] != given:
            parser.error(
                f"--{key} {given} disagrees with the checkpoint's {key}={from_ckpt[key]} "
                f"({path}). Drop the flag to take the checkpoint's geometry.")
    src = "checkpoint" if from_ckpt else "default"
    print(f"ST geometry: num_inds={args.num_inds} dim_hidden={args.dim_hidden} "
          f"num_post_sab={args.num_post_sab} ({src})")


def _st_extractor_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        num_encodings=args.num_encodings,
        dim_encoder=args.dim_encoder,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        num_post_sab=args.num_post_sab,
        ln=args.ln,
        arena_scale=args.arena_scale,
        weight_channel=args.weight_channel,
        pretrained_st_model_path=args.pretrained_st_model_path,
        st_frozen=args.st_frozen,
        output_norm=bool(getattr(args, "output_norm", False)),
    )


# ---------------------------------------------------------------------------
# Gaussian, pooled (DeepSet / PointNet), k-moments
# ---------------------------------------------------------------------------


def _no_arguments(parser: argparse.ArgumentParser, domain: Domain) -> None:
    return None


def _no_resolution(parser: argparse.ArgumentParser, args: argparse.Namespace,
                   domain: Domain) -> None:
    return None


def _pooled_add_arguments(cls: type, name: str):
    def add(parser: argparse.ArgumentParser, domain: Domain) -> None:
        g = parser.add_argument_group(f"{name} encoder")
        g.add_argument("--num_encodings", type=int, default=8)
        g.add_argument("--dim_encoder", type=int, default=8,
                       help="Code = num_encodings x dim_encoder features (the ST arm's default "
                            "8 x 8 = 64).")
        g.add_argument("--dim_hidden", type=int, default=128)
        g.add_argument("--weight_channel", dest="weight_channel", action="store_true", default=True,
                       help="Feed the PF mass as an extra input channel (default on).")
        g.add_argument("--no_weight_channel", dest="weight_channel", action="store_false",
                       help="Coordinates only; required to load an unweighted DeepSetAE / "
                            "PointNetAE checkpoint.")
        g.add_argument("--pooling", type=str, default=None, choices=cls.POOLINGS,
                       help=f"Pool operator; default {cls.POOLINGS[0]!r}.")
        g.add_argument("--output_norm", action="store_true", default=False,
                       help="Layer-normalise the flattened code (no affine) as the encoder's last "
                            "operation; default off; must match the checkpoint (debug_plans/ch_fixes.md).")
    return add


def _pooled_extractor_kwargs(args: argparse.Namespace) -> dict:
    return dict(num_encodings=args.num_encodings, dim_encoder=args.dim_encoder,
                dim_hidden=args.dim_hidden, arena_scale=args.arena_scale,
                weight_channel=args.weight_channel, pooling=args.pooling,
                pretrained_model_path=args.pretrained_model_path, frozen=args.frozen,
                output_norm=bool(getattr(args, "output_norm", False)))


def _kmoments_add_arguments(parser: argparse.ArgumentParser, domain: Domain) -> None:
    g = parser.add_argument_group("kmoments encoder")
    g.add_argument("--k", dest="k_moments", type=int, default=4,
                   help="Moment orders 1..k per coordinate (k=2: mean + variance).")


def _pooled_encoder(name: str, cls: type) -> Encoder:
    return Encoder(
        name=name, extractor_class=cls, learned=True,
        add_arguments=_pooled_add_arguments(cls, name),
        resolve_arguments=_no_resolution,
        extractor_kwargs=_pooled_extractor_kwargs,
        # A7 (2026-09-19): the same feature-length logger / guard as the ST arm (tags <name>/feat_*).
        callbacks=lambda kwargs, options: [STFeatureLoggingCallback()],
        pretrained_dest="pretrained_model_path", frozen_dest="frozen",
        lr_scale_dest="encoder_lr_scale", unfreeze_dest="unfreeze_at",
        start_mode_aliases=(("--pretrained_model_path",), (), (), ()),
    )


ENCODERS: dict[str, Encoder] = {
    "cgf": Encoder(
        name="cgf", extractor_class=WeightedCGFFeaturesExtractor, learned=True,
        add_arguments=_cgf_add_arguments,
        resolve_arguments=_cgf_resolve_arguments,
        extractor_kwargs=_cgf_extractor_kwargs,
        callbacks=_cgf_callbacks,
        encoder_options=lambda args: dict(running_norm_update=args.running_norm_update),
        pretrained_dest="pretrained_cgf_model_path", frozen_dest="cgf_frozen",
        lr_scale_dest="encoder_lr_scale", unfreeze_dest="unfreeze_at",
        start_mode_aliases=(("--pretrained_cgf_model_path",), ("--cgf_frozen",), (), ()),
    ),
    "st": Encoder(
        name="st", extractor_class=SetTransformerFeaturesExtractor, learned=True,
        add_arguments=_st_add_arguments,
        resolve_arguments=_st_resolve_arguments,
        extractor_kwargs=_st_extractor_kwargs,
        callbacks=lambda kwargs, options: [STFeatureLoggingCallback()],
        pretrained_dest="pretrained_st_model_path", frozen_dest="st_frozen",
        lr_scale_dest="st_encoder_lr_scale", unfreeze_dest="st_unfreeze_at",
        start_mode_aliases=(("--pretrained_st_model_path",), ("--st_frozen",),
                            ("--st_encoder_lr_scale",), ("--st_unfreeze_at",)),
    ),
    "gaussian": Encoder(
        name="gaussian", extractor_class=WeightedGaussianFeaturesExtractor, learned=False,
        add_arguments=_no_arguments, resolve_arguments=_no_resolution,
        extractor_kwargs=lambda args: dict(arena_scale=args.arena_scale),
        callbacks=lambda kwargs, options: [],
    ),
    "deepset": _pooled_encoder("deepset", WeightedDeepSetFeaturesExtractor),
    "pointnet": _pooled_encoder("pointnet", PointNetFeaturesExtractor),
    "kmoments": Encoder(
        name="kmoments", extractor_class=WeightedKMomentsFeaturesExtractor, learned=False,
        add_arguments=_kmoments_add_arguments, resolve_arguments=_no_resolution,
        extractor_kwargs=lambda args: dict(k=args.k_moments, arena_scale=args.arena_scale),
        callbacks=lambda kwargs, options: [],
    ),
}


def get(name: str | Encoder) -> Encoder:
    if isinstance(name, Encoder):
        return name
    try:
        return ENCODERS[name]
    except KeyError:
        raise ValueError(f"Unknown encoder {name!r}. Available: {sorted(ENCODERS)}") from None


def print_encoders() -> None:
    """Human-readable dump of the table, for --list_encoders."""
    for name, encoder in ENCODERS.items():
        kind = "learned (pretrain / freeze / finetune)" if encoder.learned else "analytic (no parameters)"
        print(f"{name:10s} {encoder.extractor_class.__name__:36s} {kind}")
