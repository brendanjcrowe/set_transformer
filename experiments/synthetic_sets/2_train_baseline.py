"""Train vanilla PFSetTransformer on the synthetic point-set mix."""

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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--out_subdir",
        type=str,
        default="baseline",
        help="Subdir under experiments/synthetic_sets/runs/",
    )
    args = parser.parse_args()
    disable_wandb_if_unset()

    overrides = {}
    if args.num_epochs is not None:
        overrides["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    cfg = build_config("pf_st", **overrides)

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
