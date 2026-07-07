"""Train DeepSet VQ-VAE (mean-pool encoder + EMA codebook + PFDecoder)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from set_transformer.training.experiment import run_training_experiments

from _common import RUNS_DIR, build_config, build_loaders, disable_wandb_if_unset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--codebook_size", type=int, default=64)
    parser.add_argument("--commitment_weight", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out_subdir", type=str, default="ds_vqvae")
    args = parser.parse_args()
    disable_wandb_if_unset()

    overrides = {
        "codebook_size": args.codebook_size,
        "commitment_weight": args.commitment_weight,
        "ema_decay": args.ema_decay,
    }
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    cfg = build_config("ds_vqvae", **overrides)

    train_loader, val_loader = build_loaders(
        cfg.batch_size, num_workers=args.num_workers
    )

    run_training_experiments(
        experiment_name=args.out_subdir,
        train_loader=train_loader,
        val_loader=val_loader,
        configs=[cfg],
        base_dir=str(RUNS_DIR),
    )


if __name__ == "__main__":
    main()
