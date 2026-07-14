"""Shared config + data loading for the synthetic-sets experiments.

All three `2_train_*.py` scripts go through `build_config` and `build_loaders`
so the only differences between runs are the variant-specific hparams.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from set_transformer.data.dataset import POMDPDataset
from set_transformer.training.config import TrainingConfig

DATA_DIR = Path("experiments/synthetic_sets/data")
RUNS_DIR = Path("experiments/synthetic_sets/runs")

# Shared base config; all 3 variants train with identical knobs except the
# variant-specific ones (model_type / kl_weight / codebook_size / ...).
BASE_CONFIG = TrainingConfig(
    num_particles=100,
    dim_particles=2,
    num_encodings=8,
    dim_encoder=16,
    num_inds=32,
    dim_hidden=128,
    num_heads=4,
    use_layer_norm=True,
    batch_size=64,
    learning_rate=1e-3,
    num_epochs=50,
    weight_decay=0.0,
    clip_grad_norm=1.0,
    scheduler_type="cosine",
    warmup_epochs=0,
    min_lr=1e-6,
    loss_type="chamfer",
    log_freq=20,
    eval_freq=100,
    save_freq=200,
    keep_last_n_checkpoints=2,
)


def build_loaders(batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Train + val DataLoaders backed by the deterministic synthetic split."""
    train_pts = np.load(DATA_DIR / "train.points.npy")
    eval_pts = np.load(DATA_DIR / "eval.points.npy")
    train_ds = POMDPDataset(train_pts)
    eval_ds = POMDPDataset(eval_pts)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def build_config(model_type: str, **overrides) -> TrainingConfig:
    cfg = replace(BASE_CONFIG, model_type=model_type, **overrides)
    return cfg


def disable_wandb_if_unset() -> None:
    """Default to offline-style wandb so scripts run without login.

    Override by exporting WANDB_MODE=online before invoking the script.
    """
    os.environ.setdefault("WANDB_MODE", "disabled")
