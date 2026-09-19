"""The ``reconstruction`` pretraining objective: encoder -> ``PFDecoder`` rebuilds the particle
cloud, Sinkhorn (or Chamfer) loss, optional latent-metric alignment against a precomputed EMD
matrix. GENERIC: it needs only a collected ``.npz`` from step 2 of the pipeline, so every domain
gets it without declaring anything (``rl/domains/base.py::Objective``).

Batch 7.2 of the harness centralisation (``refactor_plans.md`` section 7, 2026-09-13). The
bodies below are the body of ``experiments/ant_tag/3_train_st.py::main`` MOVED here in three
pieces -- ``resolve_arguments`` (the checks that need no data), ``prepare`` (dataset loading and
every validation against it) and ``run`` (the ``Trainer`` and the CGF export) -- with the
encoder-geometry flags taken from the shared ``Encoder`` table (``rl/encoders.py``) instead of
the script's own copies, and the placement (``--base_dir`` / ``--output_root`` / ``--domain``
/ ``--variant``), the seed and the device handled by ``rl/pretrain.py``. Two lines changed
shape and are marked ``# 7.2:`` where they occur: the layer-norm flag is read as ``args.ln``
(the table's spelling; ``--no_layer_norm`` still parses) and the ST arm's ``--no_st_weight_channel``
counts as ``--ignore_weights`` (both mean "no mass channel on the encoder input").

WEIGHTED SETS. A ``.npz`` from the collectors carries the PF weights alongside the particles,
and they are used by default. The encoder then reads D coordinates plus a mass channel, the
decoder still reconstructs D coordinates, and the reconstruction loss compares the weighted
target measure against the uniform reconstruction -- mass in the measure, never in the ground
metric. ``--ignore_weights`` gives the unweighted ablation. A legacy ``.npy`` dataset has no
weights and trains unweighted either way.

``--encoder cgf`` pretrains the CGF arm's block (``WeightedCGFFeaturesExtractor`` through a
``PFDecoder``, same loss / alignment / frame) and exports ``checkpoints/checkpoint_best_cgf_arm.pt``
for ``rl/train.py --encoder cgf --pretrained_path``; the ST checkpoint is ``checkpoint_best.pt``
itself. ``--encoder deepset|pointnet`` (batch 10.11, 2026-09-14) pretrains the pooled arm's own
extractor the same way (``models/pooled_arm_ae.py``: weight channel, weighted / masked pool) and
exports ``checkpoints/checkpoint_best_<encoder>_arm.pt``; alignment (``--align_lambda``) runs on
every one of the four, the Trainer reading ``encode()`` off each arm autoencoder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from set_transformer.data.dataset import get_data_loader
from set_transformer.emd_matrix import dataset_sha256, read_sidecar
from set_transformer.models.arm_export import export_arm_checkpoint
from set_transformer.models.cgf_arm_ae import CGFArmAutoencoder, make_arm_extractor
from set_transformer.models.pooled_arm_ae import PooledArmAutoencoder, make_pooled_arm_extractor
from set_transformer.rl.domains.base import Objective, PretrainContext, PretrainResult
from set_transformer.training.config import ExperimentConfig, TrainingConfig
from set_transformer.training.trainer import Trainer

#: Encoders this objective can pretrain by reconstruction: every learned encoder of the table
#: (``rl/encoders.py``; pinned equal by a test). The ST trains as the Trainer's own ``pf_st``
#: autoencoder; the other three train as ARM autoencoders around their RL extractor.
ENCODERS = ("st", "cgf", "deepset", "pointnet")
#: The pooled arms (``models/pooled_arm_ae.py``).
POOLED_ENCODERS = ("deepset", "pointnet")


def dataset_metadata(data_path: str) -> dict:
    """The collector's metadata JSON stored in an .npz dataset ({} for a legacy .npy or a
    dataset without one). Lazy: reads one small member of the archive."""
    if not str(data_path).endswith(".npz"):
        return {}
    with np.load(data_path, allow_pickle=True) as archive:
        if "metadata" not in archive.files:
            return {}
        try:
            return json.loads(str(archive["metadata"]))
        except (TypeError, ValueError):
            return {}


def add_arguments(parser: argparse.ArgumentParser, domain=None) -> None:
    """The objective's flags: data, training, loss, alignment, scheduler, logging, loading.
    Spelled as ``3_train_st.py`` spelled them (decision 2 of plan section 7)."""
    g = parser.add_argument_group("reconstruction objective: data")
    g.add_argument(
        "--data_path", type=str, default=None,
        help="Path to the .npz (or legacy .npy) dataset from the collector. Default (7.5): "
             "the variant's collected dataset under the run root, "
             "<root>/<domain>/<variant>/data/<variant>_pf_dataset.npz -- needs --variant.")
    g.add_argument(
        "--particle_scale", type=float, default=None,
        help="Divide coordinates by this before training. Default: the scale "
             "recorded in the dataset (the arena half-width the RL feature "
             "extractor normalizes by), so the encoder is pretrained on the "
             "same input range it will be given at RL time.")
    g.add_argument(
        "--no_particle_scaling", action="store_true",
        help="Train on raw coordinates, ignoring the recorded scale. Only "
             "correct if the RL side also runs unscaled.")
    g.add_argument(
        "--ignore_weights", action="store_true",
        help="Train on the unweighted particle cloud even when the dataset "
             "carries PF weights. The ablation, not the default.")
    # Model geometry that the DATASET decides. dim_particles is the COORDINATE dimension;
    # in weighted mode the encoder input is dim_particles + 1 internally.
    g.add_argument(
        "--num_particles", type=int, default=None,
        help="Set size. Default: read from the dataset. An explicit value "
             "must match the dataset -- a wrong one is otherwise silent, "
             "because Sinkhorn happily compares sets of different sizes and "
             "the ISAB encoder accepts any set size.")
    g.add_argument(
        "--dim_particles", type=int, default=None,
        help="Coordinate dimension. Default: read from the dataset.")
    g.add_argument(
        "--max_samples", type=int, default=None,
        help="Train on the first K rows of the dataset only (file order; the "
             "collectors shuffle after rebalancing). With --emd_matrix_path the "
             "matrix's own row count is used and this must agree with it. Give "
             "the PLAIN arm the same value as the aligned arm's matrix so the two "
             "train on identical rows.")
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--train_split", type=float, default=0.8)

    t = parser.add_argument_group("reconstruction objective: training")
    t.add_argument("--num_epochs", type=int, default=100)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--learning_rate", type=float, default=1e-3)
    t.add_argument("--weight_decay", type=float, default=0.0)
    t.add_argument("--clip_grad_norm", type=float, default=1.0)
    t.add_argument("--scheduler_type", type=str, default="cosine",
                   choices=["cosine", "step", "none"])
    t.add_argument("--warmup_epochs", type=int, default=10)
    t.add_argument("--min_lr", type=float, default=1e-6)
    t.add_argument("--log_freq", type=int, default=100)
    t.add_argument("--eval_freq", type=int, default=1000)
    t.add_argument("--save_freq", type=int, default=5000)
    t.add_argument("--keep_last_n_checkpoints", type=int, default=5)

    lo = parser.add_argument_group("reconstruction objective: loss")
    lo.add_argument(
        "--loss_type", type=str, default="sinkhorn", choices=["chamfer", "sinkhorn"],
        help="Training objective. sinkhorn is the only one that accepts "
             "weighted sets; chamfer is unweighted only. (emd is the eval "
             "metric and has no gradient; hausdorff is broken upstream.)")
    lo.add_argument(
        "--sinkhorn_blur", type=float, default=0.05,
        help="Sinkhorn regularization in COORDINATE UNITS; must be well below "
             "the smallest belief structure to resolve.")
    lo.add_argument("--sinkhorn_scaling", type=float, default=0.5)

    # The decoder's geometry (and, for the ST arm, the encoder's -- the same PFSetTransformer
    # flags). The ST encoder group already carries these; the CGF arm's group does not, and
    # its PFDecoder still needs --num_encodings / --dim_hidden (and TrainingConfig records the
    # rest), so add whichever are missing, spelled and defaulted as the shared ST group has them.
    have = set(parser._option_string_actions)
    d = parser.add_argument_group("reconstruction objective: decoder geometry (PFDecoder)")
    if "--num_encodings" not in have:
        d.add_argument("--num_encodings", type=int, default=8,
                       help="Decoder attention slots (PMA seeds) for the CGF arm's PFDecoder.")
    if "--dim_encoder" not in have:
        d.add_argument("--dim_encoder", type=int, default=8,
                       help="Recorded in the checkpoint config; the CGF arm derives its code "
                            "width from the block, so this does not shape the model.")
    if "--dim_hidden" not in have:
        d.add_argument("--dim_hidden", type=int, default=128, help="Decoder hidden width.")
    if "--num_inds" not in have:
        d.add_argument("--num_inds", type=int, default=32, help="Recorded only (ISAB inducing points).")
    if "--num_heads" not in have:
        d.add_argument("--num_heads", type=int, default=4)
    if "--num_post_sab" not in have:
        d.add_argument("--num_post_sab", type=int, default=2, help="Recorded only for the CGF arm.")
    if "--no_layer_norm" not in have:
        d.add_argument("--no_layer_norm", "--no_ln", dest="ln", action="store_false", default=True)
    # Change B (2026-09-19, debug_plans/ch_fixes.md): a learned attention temperature in the
    # PFDecoder, so it sharpens its softmax without asking the encoder for large codes. Default
    # off (no parameter added); recorded in TrainingConfig. --output_norm (change A) comes from
    # the encoder table for the ST and pooled arms.
    d.add_argument("--decoder_temperature", action="store_true", default=False,
                   help="Learned scalar temperature on the PFDecoder's attention scores (init 1). "
                        "Default off.")

    a = parser.add_argument_group("reconstruction objective: latent metric alignment "
                                  "(optional; needs 2b_precompute_emd.py first)")
    a.add_argument(
        "--align_lambda", type=float, default=0.0,
        help="Weight of the latent metric-alignment term, 1 - pearson_r between "
             "the batch's latent pairwise distances and the precomputed pairwise "
             "EMD distances. 0 (default) = off. The collaborator's MoG operating "
             "point was 0.2 with warmup 15 / ramp 15 over 60 epochs; re-tune per "
             "domain (our encoders converge or collapse far sooner).")
    a.add_argument(
        "--emd_matrix_path", type=str, default=None,
        help="Matrix from 2b_precompute_emd.py over THIS dataset. Its .json "
             "sidecar must agree with this run on blur, scaling, weightedness, "
             "frame and dataset hash, or the run is refused.")
    a.add_argument("--align_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    a.add_argument("--align_warmup_epochs", type=int, default=0, help="Epochs with lambda held at 0.")
    a.add_argument("--align_ramp_epochs", type=int, default=0,
                   help="Epochs over which lambda ramps linearly to its target.")
    a.add_argument("--align_val_max_samples", type=int, default=2000,
                   help="Val rows used for the held-out val/align_r metric.")


def locate(args: argparse.Namespace) -> dict:
    """Where the dataset says it belongs: the collector records ``variant`` and ``env_id``."""
    metadata = dataset_metadata(args.data_path) if getattr(args, "data_path", None) else {}
    return {"variant": metadata.get("variant"), "env_id": metadata.get("env_id")}


def resolve_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace,
                      domain, encoder) -> None:
    """The checks that need no data (the script ran them right after parsing)."""
    if args.data_path is None:
        # 7.5 (decision 1): the dataset the collector puts under the run root for this variant.
        if not getattr(args, "variant", None):
            parser.error("--data_path not given: pass it, or --variant <registry key> to take "
                         "<root>/<domain>/<variant>/data/<variant>_pf_dataset.npz")
        from set_transformer.rl import run_records
        candidate = run_records.dataset_path(domain.name, args.variant,
                                             root=getattr(args, "output_root", None))
        if not candidate.exists():
            parser.error(f"no collected dataset at {candidate}: collect one (python3 -m "
                         f"set_transformer.rl.collect --domain {domain.name} --variant "
                         f"{args.variant}) or pass --data_path")
        args.data_path = str(candidate)
        print(f"Dataset: {candidate} (the variant's collected dataset under the run root)")
    if encoder.name not in ENCODERS:
        parser.error(f"--objective reconstruction pretrains {ENCODERS}; "
                     f"encoder {encoder.name!r} has no reconstruction path"
                     + ("" if encoder.learned else " (an analytic encoder has no parameters to pretrain)")
                     + ".")
    if args.emd_matrix_path is None and args.align_lambda > 0:
        parser.error("--align_lambda > 0 needs --emd_matrix_path "
                     "(run 2b_precompute_emd.py on this dataset first)")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        parser.error("--max_samples must be positive")
    if args.align_lambda < 0:
        parser.error("--align_lambda must be >= 0")
    post_sab = getattr(args, "num_post_sab", None)
    if post_sab is not None and int(post_sab) < 0:
        parser.error(f"--num_post_sab {post_sab} must be >= 0")


@dataclass
class ReconstructionData:
    """What :func:`prepare` loaded and validated, for :func:`run`."""

    train_loader: object
    val_loader: object
    train_size: int
    val_size: int
    base_dataset: object
    weighted: bool
    aligning: bool
    sidecar: dict | None
    max_samples: int | None
    particle_scale: float | None
    particle_centre: float | None


def prepare(parser: argparse.ArgumentParser, args: argparse.Namespace, domain, encoder,
            device: str) -> ReconstructionData:
    """Load the dataset and validate everything against it; leave the dataset-derived
    ``num_particles`` / ``dim_particles`` / ``arena_scale`` on ``args``."""
    # CUDA multiprocessing. In a fresh process (the scripts) this sets `spawn` as the script
    # always did; called in-process after another component fixed the context (the test
    # suite's SubprocVecEnv tests), torch refuses to change it, and the loaders run with
    # --num_workers 0 anyway, so the existing context is kept.
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    # Does the dataset carry PF weights? np.load on an .npz is lazy, so this
    # reads the archive index only, not the arrays.
    dataset_has_weights = False
    if args.data_path.endswith(".npz"):
        with np.load(args.data_path) as archive:
            dataset_has_weights = "weights" in archive.files
    # 7.2: the ST arm's --no_st_weight_channel means the same as --ignore_weights (no mass
    # channel on the encoder input); 10.11: so does the pooled arms' --no_weight_channel (same dest).
    # channel on the encoder input); either spelling turns the weights off.
    ignore_weights = bool(args.ignore_weights) or not getattr(args, "weight_channel", True)
    weighted = dataset_has_weights and not ignore_weights
    if weighted and args.loss_type == "chamfer":
        parser.error(
            "--loss_type chamfer cannot train on weighted particle sets: "
            "nearest-neighbour distances carry no mass. Use sinkhorn, or pass "
            "--ignore_weights to train on the unweighted cloud."
        )
    if ignore_weights and not dataset_has_weights:
        print("NOTE: --ignore_weights given but the dataset has no weights.")

    particle_scale = 1.0 if args.no_particle_scaling else args.particle_scale
    # The centre is the other half of the RL-side (x - centre) / scale
    # mapping (Odd-Even centres its state range on 0; Ant-Tag's centre is
    # 0 already). None = the value recorded in the dataset, 0.0 if none.
    particle_centre = 0.0 if args.no_particle_scaling else None

    # Alignment: the matrix must have been built from this exact dataset,
    # in this frame, with this metric. Everything the sidecar records is
    # checked before any data is loaded; frame and row count are checked
    # again against the loaded dataset below.
    aligning = args.align_lambda > 0
    max_samples = args.max_samples
    sidecar = None
    if aligning:
        if not args.emd_matrix_path:
            parser.error("--align_lambda > 0 needs --emd_matrix_path "
                         "(run 2b_precompute_emd.py on this dataset first)")
        try:
            sidecar = read_sidecar(Path(args.emd_matrix_path))
        except FileNotFoundError as exc:
            parser.error(str(exc))
        problems = []
        if abs(float(sidecar["blur"]) - args.sinkhorn_blur) > 1e-12:
            problems.append(f"blur: matrix {sidecar['blur']}, this run {args.sinkhorn_blur}")
        if abs(float(sidecar.get("scaling", 0.5)) - args.sinkhorn_scaling) > 1e-12:
            problems.append(f"scaling: matrix {sidecar.get('scaling', 0.5)}, "
                            f"this run {args.sinkhorn_scaling}")
        if bool(sidecar.get("weighted", False)) != weighted:
            problems.append(f"weighted: matrix {sidecar.get('weighted', False)}, "
                            f"this run {weighted} (--ignore_weights must match)")
        if sidecar.get("data_sha256"):
            actual = dataset_sha256(Path(args.data_path))
            if actual != sidecar["data_sha256"]:
                problems.append("data_sha256: the matrix was built from a different "
                                f"dataset file ({sidecar.get('data_path')})")
        if problems:
            parser.error("--emd_matrix_path does not match this run:\n  "
                         + "\n  ".join(problems)
                         + "\nRecompute it with 2b_precompute_emd.py using the same flags.")
        max_samples = int(sidecar["n_samples"])
        if args.max_samples is not None and int(args.max_samples) != max_samples:
            parser.error(f"--max_samples {args.max_samples} contradicts the matrix, "
                         f"which covers the first {max_samples} rows. Omit the flag.")
        if args.batch_size < 16:
            print(f"WARNING: --batch_size {args.batch_size} gives only "
                  f"{args.batch_size * (args.batch_size - 1) // 2} pairs per batch for "
                  "the alignment correlation; 16 (120 pairs) is the practical floor.")

    # Load data
    train_loader, val_loader, train_size, val_size = get_data_loader(
        batch_size=args.batch_size,
        data_path=args.data_path,
        device=device,
        train_split=args.train_split,
        num_workers=args.num_workers,
        load_weights=weighted,
        particle_scale=particle_scale,
        particle_centre=particle_centre,
        seed=args.seed,
        indexed=aligning,
        max_samples=max_samples,
    )
    print(f"Dataset: {train_size} train / {val_size} val samples "
          f"({'weighted' if weighted else 'unweighted'} particle sets)")

    # The dataset is ground truth for the set geometry. Adopt it when the user
    # said nothing, and refuse a contradiction rather than train on it.
    # random_split wraps the POMDPDataset in a Subset.
    base_dataset = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    for flag, actual in (("num_particles", base_dataset.num_particles),
                         ("dim_particles", base_dataset.particle_dim)):
        given = getattr(args, flag)
        if given is None:
            setattr(args, flag, actual)
        elif given != actual:
            parser.error(
                f"--{flag} {given} contradicts the dataset ({actual}). Omit "
                "the flag to take it from the data."
            )
    if sidecar is not None:
        frame_problems = []
        for key, actual in (("particle_scale", base_dataset.particle_scale),
                            ("particle_centre", base_dataset.particle_centre)):
            recorded = sidecar.get(key)
            if recorded is not None and abs(float(recorded) - float(actual)) > 1e-9:
                frame_problems.append(f"{key}: matrix {recorded}, this run {actual}")
        if len(base_dataset) != int(sidecar["n_samples"]):
            frame_problems.append(f"n_samples: matrix {sidecar['n_samples']}, "
                                  f"dataset {len(base_dataset)}")
        if frame_problems:
            parser.error("--emd_matrix_path was built in a different coordinate frame "
                         "or over different rows than this run loads:\n  "
                         + "\n  ".join(frame_problems))

    # The frame the encoder is trained in IS the RL extractor's arena_scale (the block
    # divides by it; the loader already divided by the dataset's particle_scale).
    args.arena_scale = float(base_dataset.particle_scale)
    if encoder.name == "cgf" and args.dim_particles != domain.particle_dim:
        # The CGF flag resolution sizes the probes with the domain's particle dimension.
        parser.error(f"the dataset holds {args.dim_particles}-D particles but domain "
                     f"{domain.name!r} has {domain.particle_dim}-D beliefs; pass the domain "
                     "the dataset was collected on (--domain).")
    return ReconstructionData(
        train_loader=train_loader, val_loader=val_loader, train_size=train_size,
        val_size=val_size, base_dataset=base_dataset, weighted=weighted, aligning=aligning,
        sidecar=sidecar, max_samples=max_samples, particle_scale=particle_scale,
        particle_centre=particle_centre)


def _pooled_kwargs(encoder, args: argparse.Namespace, *, arena_scale: float, weighted: bool,
                   pretrained_path: str | None = None) -> dict:
    """The pooled extractor's constructor kwargs from the resolved flags, through the table
    (``Encoder.extractor_kwargs``), with the dataset's frame and weightedness put in (10.11)."""
    kwargs = encoder.extractor_kwargs(args)
    kwargs["arena_scale"] = float(arena_scale)
    kwargs["weight_channel"] = bool(weighted)
    kwargs[encoder.extractor_class.PRETRAINED_PATH_KWARG] = pretrained_path
    kwargs["frozen"] = False
    return kwargs


def _export_arm(args: argparse.Namespace, experiment_config: ExperimentConfig, extractor, *,
                encoder_name: str, suffix: str, aligning: bool, weighted: bool,
                particle_centre: float, build_fresh) -> tuple[dict, str]:
    """Export ``checkpoint_{best,latest}.pt`` of an ARM autoencoder as
    ``checkpoint_<tag>_<suffix>.pt`` (the RL-loadable files) and prove the primary one
    round-trips: ``build_fresh(path)`` builds the RL extractor with the export as its pretrained
    path, and every tensor must equal the file's. The body is the CGF export loop of the
    2026-09-11 script, shared with the pooled arms since 10.11. Returns ``(exported, primary)``."""
    objective = f"reconstruction_{args.loss_type}" + ("_aligned" if aligning else "")
    exported = {}
    for tag in ("best", "latest"):
        src = experiment_config.checkpoint_dir / f"checkpoint_{tag}.pt"
        if not src.exists():
            continue
        dst = experiment_config.checkpoint_dir / f"checkpoint_{tag}_{suffix}.pt"
        export_arm_checkpoint(
            src, dst, extractor, encoder_name=encoder_name, particle_centre=particle_centre,
            objective=objective, data_path=args.data_path,
            extra_config={"weighted_pretraining": bool(weighted),
                          "sinkhorn_blur": args.sinkhorn_blur,
                          "sinkhorn_scaling": args.sinkhorn_scaling})
        exported[tag] = dst
    primary = "best" if "best" in exported else "latest"     # 7.5: see rl_checkpoint in run()
    check = build_fresh(str(exported[primary]))
    ref = torch.load(exported[primary], map_location="cpu", weights_only=False)["model_state_dict"]
    live = check.checkpoint_state()
    worst = max(float((ref[k].float() - live[k].float()).abs().max()) for k in ref)
    if worst > 0.0:
        raise RuntimeError(f"exported checkpoint does not round-trip (max |delta| {worst})")
    return exported, primary


def run_name(args: argparse.Namespace, now: datetime) -> str:
    """``<loss>_<YYYY-MM-DD_HH-MM-SS>``, the Trainer run folder the script always used."""
    return f"{args.loss_type}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"


def run(args: argparse.Namespace, ctx: PretrainContext) -> PretrainResult:
    data: ReconstructionData = ctx.data
    weighted, aligning, sidecar = data.weighted, data.aligning, data.sidecar
    base_dataset = data.base_dataset
    encoder_name = ctx.encoder.name

    # Build configs
    training_config = TrainingConfig(
        num_particles=args.num_particles,
        dim_particles=args.dim_particles,
        num_encodings=args.num_encodings,
        dim_encoder=args.dim_encoder,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        use_layer_norm=bool(args.ln),          # 7.2: the table's spelling of --no_layer_norm
        num_post_sab=int(args.num_post_sab),   # 7.2: configurable (was the constructor's fixed 2)
        output_norm=bool(getattr(args, "output_norm", False)),          # change A (2026-09-19)
        decoder_temperature=bool(getattr(args, "decoder_temperature", False)),   # change B
        weighted_particles=weighted,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        loss_type=args.loss_type,
        sinkhorn_blur=args.sinkhorn_blur,
        sinkhorn_scaling=args.sinkhorn_scaling,
        device=ctx.device,
        num_workers=args.num_workers,
        seed=args.seed,
        align_lambda=args.align_lambda,
        align_metric=args.align_metric,
        align_warmup_epochs=args.align_warmup_epochs,
        align_ramp_epochs=args.align_ramp_epochs,
        emd_matrix_path=args.emd_matrix_path if aligning else None,
        align_val_max_samples=args.align_val_max_samples,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
    )
    experiment_config = ExperimentConfig(
        experiment_name=ctx.experiment_name,
        run_name=ctx.run_name,
        base_dir=Path(ctx.base_dir),
    )
    print(f"Experiment: {experiment_config.run_dir}")

    encoder_input_dim = args.dim_particles + (1 if weighted else 0)
    applied_scale = base_dataset.particle_scale
    print(f"Config: loss={args.loss_type}, particles={args.num_particles}x{args.dim_particles}, "
          f"encodings={args.num_encodings}x{args.dim_encoder}, "
          f"encoder_input_dim={encoder_input_dim}, weighted={weighted}")
    # Blur is a length in the TRAINING coordinate system. With scaling applied
    # it is measured in normalized units, so print what that is in env units --
    # the number to compare against the belief structure you want resolved.
    applied_centre = base_dataset.particle_centre
    print(f"Coordinates mapped as (x - {applied_centre}) / {applied_scale} "
          f"[particle_centre / particle_scale]; seed={args.seed}; "
          f"sinkhorn_blur={args.sinkhorn_blur} "
          f"(= {args.sinkhorn_blur * applied_scale:.4f} env units)")
    if aligning:
        print(f"Latent alignment: lambda={args.align_lambda} ({args.align_metric}), "
              f"warmup {args.align_warmup_epochs} / ramp {args.align_ramp_epochs} epochs "
              f"of {args.num_epochs}; matrix {args.emd_matrix_path} "
              f"({'weighted' if sidecar.get('weighted') else 'uniform'}, "
              f"{sidecar['n_samples']} rows, offdiag std {sidecar.get('offdiag_std', float('nan')):.4g})")
        if args.align_warmup_epochs + args.align_ramp_epochs >= args.num_epochs:
            print("WARNING: warmup + ramp >= num_epochs: lambda never reaches its "
                  "target, so no checkpoint from this run is fully aligned.")
    else:
        print("Latent alignment: off")

    # Train
    model = None
    cgf_extractor = None
    cgf_kwargs = None
    pooled_extractor = None
    if encoder_name == "cgf":
        # The block divides by arena_scale; the loader already divided by the
        # dataset's particle_scale. They are the same number here (checked in
        # the autoencoder) so the exported t are in the frame PPO uses.
        cgf_kwargs = dict(
            num_cgf_features=args.num_cgf_features, t_init_mode=args.t_init_mode,
            t_init_scale=args.t_init_scale, t_clamp=args.t_clamp, t_frozen=args.t_frozen,
            exp_arg_clamp=args.exp_arg_clamp,   # 7.2: the table offers the flag; pass it on
            t_param=args.t_param, t_bound=args.t_bound, t_init_max=args.t_init_max,
            feature_mode=args.feature_mode, feature_norm=args.feature_norm,
            readout_hidden=args.readout_hidden, readout_depth=args.readout_depth,
            readout_dim=args.readout_dim, x_embed_dim=args.x_embed_dim,
            x_embed_hidden=args.x_embed_hidden, x_embed_depth=args.x_embed_depth)
        cgf_extractor = make_arm_extractor(args.num_particles, args.dim_particles,
                                           arena_scale=applied_scale, **cgf_kwargs)
        model = CGFArmAutoencoder(
            cgf_extractor, num_particles=args.num_particles,
            dim_particles=args.dim_particles, particle_scale=applied_scale,
            num_encodings=args.num_encodings, dim_hidden=args.dim_hidden,
            weighted=weighted, decoder_temperature=training_config.decoder_temperature)
        training_config.model_type = "cgf_arm_ae"
        print(f"CGF arm encoder: {cgf_extractor._cgf_geometry}")
        print(f"  block width {cgf_extractor.readout_dim} -> decoder code "
              f"{model.num_encodings} x {model.dim_encoder}; "
              f"{cgf_extractor.encoder_parameter_count()} encoder params, "
              f"{sum(p.numel() for p in model.decoder.parameters())} decoder params")
    elif encoder_name in POOLED_ENCODERS:
        # 10.11: the pooled arm's own RL extractor, built THROUGH THE ENCODER TABLE from the
        # resolved flags (the construction rl/train.py performs), with the dataset's frame and
        # weightedness: weight_channel follows `weighted` (--no_weight_channel == --ignore_weights).
        pooled_extractor = make_pooled_arm_extractor(
            ctx.encoder.extractor_class, args.num_particles, args.dim_particles,
            **_pooled_kwargs(ctx.encoder, args, arena_scale=applied_scale, weighted=weighted))
        model = PooledArmAutoencoder(
            pooled_extractor, num_particles=args.num_particles,
            dim_particles=args.dim_particles, particle_scale=applied_scale,
            dim_hidden=args.dim_hidden, weighted=weighted,
            decoder_temperature=training_config.decoder_temperature)
        training_config.model_type = "pooled_arm_ae"
        print(f"{encoder_name} arm encoder: {pooled_extractor.checkpoint_config()}")
        print(f"  code {model.num_encodings} x {model.dim_encoder}; "
              f"{pooled_extractor.encoder_parameter_count()} encoder params, "
              f"{sum(p.numel() for p in model.decoder.parameters())} decoder params")

    trainer = Trainer(
        training_config=training_config,
        experiment_config=experiment_config,
        train_loader=data.train_loader,
        val_loader=data.val_loader,
        model=model,
    )
    trainer.train()

    checkpoints = {}
    for tag in ("best", "latest"):
        path = experiment_config.checkpoint_dir / f"checkpoint_{tag}.pt"
        if path.exists():
            checkpoints[tag] = path
    # 7.5: a run too short to have evaluated has no checkpoint_best.pt; the RL-loadable file
    # is then the latest one (the drivers fell back the same way).
    rl_checkpoint = checkpoints.get("best") or checkpoints.get("latest")

    if encoder_name == "cgf":
        # The RL loader wants the extractor's own keys and its geometry, not the
        # whole autoencoder. Export best and latest, then prove the export loads
        # into a fresh extractor with the CLI's geometry (strict keys + the
        # loader's field-by-field geometry check).
        exported, primary = _export_arm(
            args, experiment_config, cgf_extractor, encoder_name="cgf", suffix="cgf_arm",
            aligning=aligning, weighted=weighted, particle_centre=applied_centre,
            build_fresh=lambda path: make_arm_extractor(
                args.num_particles, args.dim_particles, arena_scale=applied_scale,
                **cgf_kwargs, pretrained_cgf_model_path=path))
        print(f"Exported CGF arm encoder: {exported[primary]}"
              + (f" (and {exported['latest']})" if primary == "best" and "latest" in exported else ""))
        print("  loads strict into WeightedCGFFeaturesExtractor with this geometry. Use:\n"
              f"    python3 4_train_rl_cgf.py --variant <variant> "
              f"--pretrained_cgf_model_path {exported[primary]} [--cgf_frozen | "
              "--st_encoder_lr_scale 0.1]\n"
              "  (flags left at default take the checkpoint's geometry; arena_scale must "
              f"equal {applied_scale}).")
        checkpoints.update({f"{tag}_export": path for tag, path in exported.items()})
        rl_checkpoint = exported[primary]
    elif encoder_name in POOLED_ENCODERS:
        # 10.11: the same export and round trip for the pooled arm, the fresh extractor built
        # through the table with the export as its pretrained path (what rl/train.py does).
        exported, primary = _export_arm(
            args, experiment_config, pooled_extractor, encoder_name=encoder_name,
            suffix=f"{encoder_name}_arm", aligning=aligning, weighted=weighted,
            particle_centre=applied_centre,
            build_fresh=lambda path: make_pooled_arm_extractor(
                ctx.encoder.extractor_class, args.num_particles, args.dim_particles,
                **_pooled_kwargs(ctx.encoder, args, arena_scale=applied_scale, weighted=weighted,
                                 pretrained_path=path)))
        print(f"Exported {encoder_name} arm encoder: {exported[primary]}"
              + (f" (and {exported['latest']})" if primary == "best" and "latest" in exported else ""))
        print(f"  loads strict into {ctx.encoder.extractor_class.__name__} with this geometry. Use:\n"
              f"    python3 -m set_transformer.rl.train --domain <domain> --encoder {encoder_name} "
              f"--variant <variant> --pretrained_path {exported[primary]} [--frozen | "
              "--encoder_lr_scale 0.1]\n"
              f"  (pass the same --num_encodings / --dim_encoder / --dim_hidden / --pooling"
              f"{'' if weighted else ' and --no_weight_channel'}; arena_scale must equal {applied_scale}).")
        checkpoints.update({f"{tag}_export": path for tag, path in exported.items()})
        rl_checkpoint = exported[primary]

    summary = {}
    best = checkpoints.get("best")
    if best is not None:
        payload = torch.load(best, map_location="cpu", weights_only=False)
        summary = {"best_val_loss": payload.get("best_val_loss"),
                   "best_epoch": payload.get("best_epoch")}
    return PretrainResult(run_dir=experiment_config.run_dir, rl_checkpoint=rl_checkpoint,
                          checkpoints=checkpoints, summary=summary)


OBJECTIVE = Objective(
    name="reconstruction",
    description="encoder -> PFDecoder rebuilds the particle cloud; Sinkhorn loss on the "
                "weighted measure; optional latent-metric alignment (any collected dataset)",
    add_arguments=add_arguments,
    run=run,
    run_name=run_name,
    locate=locate,
    resolve_arguments=resolve_arguments,
    prepare=prepare,
    default_experiment_name=lambda encoder_name: f"{encoder_name}_reconstruction",
)
