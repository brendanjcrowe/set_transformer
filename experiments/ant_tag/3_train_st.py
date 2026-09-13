"""
Train a Set Transformer on a particle-filter dataset.

Step 3 of the pipeline:
  1) Train locomotion policy   (1_train_locomotion.py)
  2) Collect PF dataset        (2_collect_pf_dataset.py)
  3) Train Set Transformer     (this script)
  4) Train RL with the encoder (4_train_rl_st.py --pretrained_st_model_path)

--encoder cgf pretrains the CGF arm's block instead (WeightedCGFFeaturesExtractor
through a PFDecoder, same loss / alignment / frame) and exports
checkpoints/checkpoint_best_cgf_arm.pt for 4_train_rl_cgf.py --pretrained_cgf_model_path.

Nothing here is env-specific: the script consumes whatever
2_collect_pf_dataset.py wrote, for any env + particle filter pair. The
architecture flags must match what the RL feature extractor will build.

WEIGHTED SETS. A .npz from 2_collect_pf_dataset.py carries the PF weights
alongside the particles, and they are used by default. The encoder then reads
D coordinates plus a mass channel, the decoder still reconstructs D
coordinates, and the reconstruction loss compares the weighted target measure
against the uniform reconstruction — mass in the measure, never in the ground
metric. Pass --ignore_weights for the unweighted ablation. A legacy .npy
dataset has no weights and trains unweighted either way.

To pretrain an encoder for 4_train_rl_st.py's defaults, match its geometry:
--num_encodings 8 --dim_encoder 8, weights on (so the RL side keeps its
default --st_weight_channel).

Usage:
    python3 3_train_st.py \
        --data_path data/cdens_terminal_pf_dataset.npz \
        --num_encodings 8 --dim_encoder 8 \
        --sinkhorn_blur 0.01 --num_epochs 100
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Same bootstrap as every other script in this directory: put the package root
# on sys.path so `set_transformer` resolves to the package rather than to the
# submodule directory of the same name.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.multiprocessing as mp

from set_transformer.data.dataset import get_data_loader
from set_transformer.emd_matrix import dataset_sha256, read_sidecar
from set_transformer.training.config import ExperimentConfig, TrainingConfig
from set_transformer.training.trainer import Trainer
from set_transformer.models.cgf_arm_ae import (  # noqa: E402
    CGFArmAutoencoder,
    export_arm_checkpoint,
    make_arm_extractor,
)


def _dataset_metadata(data_path: str) -> dict:
    """The collector's metadata JSON stored in an .npz dataset ({} for a legacy .npy or a
    dataset without one). Lazy: reads one small member of the archive."""
    if not data_path.endswith(".npz"):
        return {}
    with np.load(data_path, allow_pickle=True) as archive:
        if "metadata" not in archive.files:
            return {}
        try:
            return json.loads(str(archive["metadata"]))
        except (TypeError, ValueError):
            return {}


def _resolve_base_dir(parser, args) -> Path:
    """--base_dir as given, else <output root>/<domain>/<variant>/pretrain/ with the domain
    and variant from the flags or the dataset's metadata (PITFALLS.md: the collector records
    `variant` and `env_id`, so the output lands beside the RL runs of the same env)."""
    if args.base_dir:
        return Path(args.base_dir)
    from set_transformer.rl import domains as _domains
    from set_transformer.rl import run_records
    metadata = _dataset_metadata(args.data_path)
    variant = args.variant or metadata.get("variant")
    if not variant:
        parser.error("--base_dir not given and the dataset records no variant: pass "
                     "--variant <registry key> (and --domain) or --base_dir <folder>")
    if args.domain:
        domain = _domains.get(args.domain)
        if variant not in domain.variants:
            parser.error(f"--variant {variant!r} is not a {args.domain} variant "
                         f"({domain.variant_names()})")
    else:
        try:
            domain = _domains.domain_of_variant(
                variant, None if args.variant else metadata.get("env_id"))
        except ValueError as exc:
            parser.error(f"{exc}; pass --domain")
    experiment_dir = run_records.pretrain_dir(domain.name, variant, args.experiment_name,
                                              root=args.output_root)
    print(f"Output root layout: {experiment_dir}/ (domain {domain.name}, variant {variant})")
    return experiment_dir.parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Set Transformer on ant-tag PF dataset (pipeline step 3)"
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the .npz (or legacy .npy) dataset from "
             "2_collect_pf_dataset.py",
    )
    parser.add_argument(
        "--particle_scale", type=float, default=None,
        help="Divide coordinates by this before training. Default: the scale "
             "recorded in the dataset (the arena half-width the RL feature "
             "extractor normalizes by), so the encoder is pretrained on the "
             "same input range it will be given at RL time.",
    )
    parser.add_argument(
        "--no_particle_scaling", action="store_true",
        help="Train on raw coordinates, ignoring the recorded scale. Only "
             "correct if the RL side also runs unscaled.",
    )
    parser.add_argument(
        "--ignore_weights", action="store_true",
        help="Train on the unweighted particle cloud even when the dataset "
             "carries PF weights. The ablation, not the default.",
    )

    # Training
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)

    # Loss
    parser.add_argument(
        "--loss_type", type=str, default="sinkhorn",
        choices=["chamfer", "sinkhorn"],
        help="Training objective. sinkhorn is the only one that accepts "
             "weighted sets; chamfer is unweighted only. (emd is the eval "
             "metric and has no gradient; hausdorff is broken upstream.)",
    )
    parser.add_argument(
        "--sinkhorn_blur", type=float, default=0.05,
        help="Sinkhorn regularization in COORDINATE UNITS; must be well below "
             "the smallest belief structure to resolve.")
    parser.add_argument("--sinkhorn_scaling", type=float, default=0.5)

    # Latent metric alignment (optional; needs 2b_precompute_emd.py first).
    parser.add_argument(
        "--align_lambda", type=float, default=0.0,
        help="Weight of the latent metric-alignment term, 1 - pearson_r between "
             "the batch's latent pairwise distances and the precomputed pairwise "
             "EMD distances. 0 (default) = off. The collaborator's MoG operating "
             "point was 0.2 with warmup 15 / ramp 15 over 60 epochs; re-tune per "
             "domain (our encoders converge or collapse far sooner).")
    parser.add_argument(
        "--emd_matrix_path", type=str, default=None,
        help="Matrix from 2b_precompute_emd.py over THIS dataset. Its .json "
             "sidecar must agree with this run on blur, scaling, weightedness, "
             "frame and dataset hash, or the run is refused.")
    parser.add_argument("--align_metric", type=str, default="cosine",
                        choices=["cosine", "euclidean"])
    parser.add_argument("--align_warmup_epochs", type=int, default=0,
                        help="Epochs with lambda held at 0.")
    parser.add_argument("--align_ramp_epochs", type=int, default=0,
                        help="Epochs over which lambda ramps linearly to its target.")
    parser.add_argument("--align_val_max_samples", type=int, default=2000,
                        help="Val rows used for the held-out val/align_r metric.")
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Train on the first K rows of the dataset only (file order; the "
             "collectors shuffle after rebalancing). With --emd_matrix_path the "
             "matrix's own row count is used and this must agree with it. Give "
             "the PLAIN arm the same value as the aligned arm's matrix so the two "
             "train on identical rows.")

    # Model architecture. dim_particles is the COORDINATE dimension; in
    # weighted mode the encoder input is dim_particles + 1 internally.
    parser.add_argument(
        "--num_particles", type=int, default=None,
        help="Set size. Default: read from the dataset. An explicit value "
             "must match the dataset — a wrong one is otherwise silent, "
             "because Sinkhorn happily compares sets of different sizes and "
             "the ISAB encoder accepts any set size.",
    )
    parser.add_argument(
        "--dim_particles", type=int, default=None,
        help="Coordinate dimension. Default: read from the dataset.",
    )
    parser.add_argument("--num_encodings", type=int, default=8)
    parser.add_argument("--dim_encoder", type=int, default=2)
    parser.add_argument("--num_inds", type=int, default=32)
    parser.add_argument("--dim_hidden", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--no_layer_norm", action="store_true")

    # LR scheduler
    parser.add_argument(
        "--scheduler_type", type=str, default="cosine",
        choices=["cosine", "step", "none"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # Logging / checkpointing
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=5)

    # Experiment management. Output goes to <base_dir>/<experiment_name>/<loss>_<timestamp>/.
    # Since change 5.2 of the harness centralisation --base_dir defaults to the shared root
    # layout, <output root>/<domain>/<variant>/pretrain/ (run_records.pretrain_dir), with the
    # domain and variant read from the dataset's metadata (the collector records both) or
    # given explicitly. An explicit --base_dir still wins, so the old cwd-relative
    # `experiments/` is one flag away.
    parser.add_argument("--experiment_name", type=str, default="ant_tag_st")
    parser.add_argument(
        "--base_dir", type=str, default=None,
        help="Default: <output root>/<domain>/<variant>/pretrain/ -- see --output_root.")
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Owner of the dataset's variant (ant_tag / odd_even). Default: looked up from the "
             "variant recorded in the dataset's metadata. Only used to place the output.")
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Registry key of the env the dataset was collected on. Default: the dataset's "
             "metadata. Only used to place the output.")
    parser.add_argument(
        "--output_root", type=str, default=None,
        help="Root of the shared run layout when --base_dir is not given: $RL_BMDP_RUNS, else "
             "<parent repo>/runs when this checkout is a submodule, else <checkout>/runs.")

    # Data loading
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_split", type=float, default=0.8)
    # Encoder family. "st" is the historical path (PFSetTransformer). "cgf"
    # pretrains the ARM's WeightedCGFFeaturesExtractor block through a PFDecoder
    # under the same loss / alignment / checkpointing, and exports the file
    # 4_train_rl_cgf.py --pretrained_cgf_model_path loads. Flags below are
    # spelled exactly as in 4_train_rl_cgf.py; the RL side takes a flag left at
    # its default from the checkpoint and refuses one that disagrees.
    parser.add_argument(
        "--encoder", type=str, default="st", choices=["st", "cgf"],
        help="Which arm's encoder to pretrain by reconstruction. 'cgf' ignores "
             "--num_inds/--dim_hidden/--num_heads/--no_layer_norm (ST geometry); "
             "--num_encodings then sets the decoder's attention slots and "
             "--dim_encoder is derived from the CGF block width.")
    cgf = parser.add_argument_group("CGF encoder (--encoder cgf)")
    cgf.add_argument("--num_cgf_features", type=int, default=64)
    cgf.add_argument("--t_init_mode", type=str, default="spread",
                     help="spread (2-D), spread_1d, linspace_all_dims, "
                          "linspace_first_dim, random")
    cgf.add_argument("--t_init_scale", type=float, default=0.1)
    cgf.add_argument("--t_clamp", type=float, default=2.0)
    cgf.add_argument("--t_frozen", action="store_true",
                     help="Fix t; only the readout / embedding / norm learn.")
    cgf.add_argument("--t_param", type=str, default="clamp", choices=["clamp", "tanh", "polar"],
                     help="Probe parameterisation. polar (a ball of radius t_bound) is "
                          "the recommended 2-D mode; PITFALLS.md section 9.")
    cgf.add_argument("--t_bound", type=float, default=None,
                     help="Required with tanh / polar. Size it by t_bound x (decision "
                          "length in normalised units) ~ 2..4: smart 9.0, "
                          "smart_mid_slow_v15 13.5, cdens_terminal 35 "
                          "(variants.cgf_t_bound). There is no env here to read it from.")
    cgf.add_argument("--t_init_max", type=float, default=None,
                     help="Ceiling of the spread init. None: 2.8 under clamp (the "
                          "recorded behaviour), 0.8 * t_bound under tanh / polar.")
    cgf.add_argument("--feature_mode", type=str, default="K", choices=["K", "K_grad", "both"])
    cgf.add_argument("--feature_norm", type=str, default="none",
                     choices=["none", "running", "layernorm"],
                     help="'running' updates its statistics every training batch here "
                          "(no PPO cycle); the RL side loads them as fixed buffers.")
    cgf.add_argument("--readout_hidden", type=int, default=0)
    cgf.add_argument("--readout_depth", type=int, default=0)
    cgf.add_argument("--readout_dim", type=int, default=None)
    cgf.add_argument("--x_embed_dim", type=int, default=0)
    cgf.add_argument("--x_embed_hidden", type=int, default=64)
    cgf.add_argument("--x_embed_depth", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device for training (cpu, cuda, cuda:1). Default: cuda when "
             "available, else cpu -- what the script always did. Pass cpu for a "
             "bit-for-bit comparison between two checkouts.")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seeds model init, the shuffle order and the train/val split. "
             "Without it every invocation gets a different val set, so "
             "best_val_loss is not comparable across runs.",
    )

    args = parser.parse_args()
    base_dir = _resolve_base_dir(parser, args)
    if args.encoder == "cgf":
        if args.t_param in ("tanh", "polar") and (args.t_bound is None or args.t_bound <= 0):
            parser.error(f"--t_param {args.t_param} needs --t_bound > 0 (see its help "
                         "for the sizing rule); this script has no env to read it from.")
        if args.t_param == "clamp":
            args.t_bound = None
        elif args.t_init_max is None:
            args.t_init_max = 0.8 * args.t_bound   # 4_train_rl_cgf.resolve_cgf_encoder_args
        if args.emd_matrix_path is None and args.align_lambda > 0:
            parser.error("--align_lambda > 0 needs --emd_matrix_path")

    # Seed before anything that draws: the split (via get_data_loader's
    # generator), the model init and the batch shuffle. Recorded in the
    # checkpoint through TrainingConfig.seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # CUDA multiprocessing
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Does the dataset carry PF weights? np.load on an .npz is lazy, so this
    # reads the archive index only, not the arrays.
    dataset_has_weights = False
    if args.data_path.endswith(".npz"):
        with np.load(args.data_path) as archive:
            dataset_has_weights = "weights" in archive.files
    weighted = dataset_has_weights and not args.ignore_weights

    if weighted and args.loss_type == "chamfer":
        parser.error(
            "--loss_type chamfer cannot train on weighted particle sets: "
            "nearest-neighbour distances carry no mass. Use sinkhorn, or pass "
            "--ignore_weights to train on the unweighted cloud."
        )
    if args.ignore_weights and not dataset_has_weights:
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
    if max_samples is not None and int(max_samples) <= 0:
        parser.error("--max_samples must be positive")
    if args.align_lambda < 0:
        parser.error("--align_lambda must be >= 0")
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

    # Build configs
    training_config = TrainingConfig(
        num_particles=args.num_particles,
        dim_particles=args.dim_particles,
        num_encodings=args.num_encodings,
        dim_encoder=args.dim_encoder,
        num_inds=args.num_inds,
        dim_hidden=args.dim_hidden,
        num_heads=args.num_heads,
        use_layer_norm=not args.no_layer_norm,
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
        device=device,
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

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.loss_type}_{timestamp}"

    experiment_config = ExperimentConfig(
        experiment_name=args.experiment_name,
        run_name=run_name,
        base_dir=base_dir,
    )

    print(f"Experiment: {experiment_config.run_dir}")
    encoder_input_dim = args.dim_particles + (1 if weighted else 0)
    applied_scale = base_dataset.particle_scale
    print(f"Config: loss={args.loss_type}, particles={args.num_particles}x{args.dim_particles}, "
          f"encodings={args.num_encodings}x{args.dim_encoder}, "
          f"encoder_input_dim={encoder_input_dim}, weighted={weighted}")
    # Blur is a length in the TRAINING coordinate system. With scaling applied
    # it is measured in normalized units, so print what that is in env units —
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
    if args.encoder == "cgf":
        # The block divides by arena_scale; the loader already divided by the
        # dataset's particle_scale. They are the same number here (checked in
        # the autoencoder) so the exported t are in the frame PPO uses.
        cgf_kwargs = dict(
            num_cgf_features=args.num_cgf_features, t_init_mode=args.t_init_mode,
            t_init_scale=args.t_init_scale, t_clamp=args.t_clamp, t_frozen=args.t_frozen,
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
            weighted=weighted)
        training_config.model_type = "cgf_arm_ae"
        print(f"CGF arm encoder: {cgf_extractor._cgf_geometry}")
        print(f"  block width {cgf_extractor.readout_dim} -> decoder code "
              f"{model.num_encodings} x {model.dim_encoder}; "
              f"{cgf_extractor.encoder_parameter_count()} encoder params, "
              f"{sum(p.numel() for p in model.decoder.parameters())} decoder params")
    trainer = Trainer(
        training_config=training_config,
        experiment_config=experiment_config,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
    )
    trainer.train()

    if args.encoder == "cgf":
        # The RL loader wants the extractor's own keys and its geometry, not the
        # whole autoencoder. Export best and latest, then prove the export loads
        # into a fresh extractor with the CLI's geometry (strict keys + the
        # loader's field-by-field geometry check).
        objective = f"reconstruction_{args.loss_type}" + ("_aligned" if aligning else "")
        exported = {}
        for tag in ("best", "latest"):
            src = experiment_config.checkpoint_dir / f"checkpoint_{tag}.pt"
            if not src.exists():
                continue
            dst = experiment_config.checkpoint_dir / f"checkpoint_{tag}_cgf_arm.pt"
            export_arm_checkpoint(
                src, dst, cgf_extractor, particle_centre=applied_centre,
                objective=objective, data_path=args.data_path,
                extra_config={"weighted_pretraining": bool(weighted),
                              "sinkhorn_blur": args.sinkhorn_blur,
                              "sinkhorn_scaling": args.sinkhorn_scaling})
            exported[tag] = dst
        check = make_arm_extractor(args.num_particles, args.dim_particles,
                                   arena_scale=applied_scale, **cgf_kwargs,
                                   pretrained_cgf_model_path=str(exported["best"]))
        ref = torch.load(exported["best"], map_location="cpu", weights_only=False)["model_state_dict"]
        live = check.state_dict()
        worst = max(float((ref[k].float() - live[k].float()).abs().max()) for k in ref)
        if worst > 0.0:
            raise RuntimeError(f"exported checkpoint does not round-trip (max |delta| {worst})")
        print(f"Exported CGF arm encoder: {exported['best']}"
              + (f" (and {exported['latest']})" if "latest" in exported else ""))
        print("  loads strict into WeightedCGFFeaturesExtractor with this geometry. Use:\n"
              f"    python3 4_train_rl_cgf.py --variant <variant> "
              f"--pretrained_cgf_model_path {exported['best']} [--cgf_frozen | "
              "--st_encoder_lr_scale 0.1]\n"
              "  (flags left at default take the checkpoint's geometry; arena_scale must "
              f"equal {applied_scale}).")


if __name__ == "__main__":
    main()
