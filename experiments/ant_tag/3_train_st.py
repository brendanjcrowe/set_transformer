"""
Train a Set Transformer on the ant-tag particle filter dataset.

This is Step 3 of the ant-tag RL pipeline:
  1) Train locomotion policy          (train_ant_locomotion.py)
  2) Collect PF dataset               (collect_ant_tag_pf_dataset.py)
  3) Train Set Transformer            (this script)
  4) Train RL with pretrained ST      (training/train_ant_tag_pretrained.py)

Usage:
    python training/train_ant_tag_st.py \
        --data_path data/ant_tag_pf_dataset.npy \
        --num_epochs 100
"""

import argparse
from datetime import datetime
from pathlib import Path

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
        help="Path to .npy dataset from collect_ant_tag_pf_dataset.py",
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
    parser.add_argument("--sinkhorn_blur", type=float, default=0.5)
    parser.add_argument("--sinkhorn_scaling", type=float, default=0.5)

    # Model architecture (defaults match AntTagPretrainedProcessor)
    parser.add_argument("--num_particles", type=int, default=100)
    parser.add_argument("--dim_particles", type=int, default=2)
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

    # Load data
    train_loader, val_loader, train_size, val_size = get_data_loader(
        batch_size=args.batch_size,
        data_path=args.data_path,
        device=device,
        train_split=args.train_split,
        num_workers=args.num_workers,
    )
    print(f"Dataset: {train_size} train / {val_size} val samples")

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
    print(f"Config: loss={args.loss_type}, particles={args.num_particles}x{args.dim_particles}, "
          f"encodings={args.num_encodings}x{args.dim_encoder}")

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
