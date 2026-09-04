"""
Train a Set Transformer on a particle-filter dataset.

Step 3 of the pipeline:
  1) Train locomotion policy   (1_train_locomotion.py)
  2) Collect PF dataset        (2_collect_pf_dataset.py)
  3) Train Set Transformer     (this script)
  4) Train RL with the encoder (4_train_rl_st.py --pretrained_st_model_path)

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
from set_transformer.training.config import ExperimentConfig, TrainingConfig
from set_transformer.training.trainer import Trainer


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
        choices=["emd", "chamfer", "sinkhorn", "hausdorff"],
    )
    parser.add_argument(
        "--sinkhorn_blur", type=float, default=0.05,
        help="Sinkhorn regularization in COORDINATE UNITS; must be well below "
             "the smallest belief structure to resolve.")
    parser.add_argument("--sinkhorn_scaling", type=float, default=0.5)

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

    # Experiment management
    parser.add_argument("--experiment_name", type=str, default="ant_tag_st")
    parser.add_argument("--base_dir", type=str, default="experiments")

    # Data loading
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_split", type=float, default=0.8)

    args = parser.parse_args()

    # CUDA multiprocessing
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Load data
    train_loader, val_loader, train_size, val_size = get_data_loader(
        batch_size=args.batch_size,
        data_path=args.data_path,
        device=device,
        train_split=args.train_split,
        num_workers=args.num_workers,
        load_weights=weighted,
        particle_scale=particle_scale,
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
        base_dir=Path(args.base_dir),
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
    print(f"Coordinates divided by particle_scale={applied_scale}; "
          f"sinkhorn_blur={args.sinkhorn_blur} "
          f"(= {args.sinkhorn_blur * applied_scale:.4f} env units)")

    # Train
    trainer = Trainer(
        training_config=training_config,
        experiment_config=experiment_config,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    trainer.train()


if __name__ == "__main__":
    main()
