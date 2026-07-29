"""Train all four methods x N seeds on the MoG data.

For each (method, seed) writes ``runs/<method>/seed<seed>/`` :
    history.npz     per-epoch epoch / train_loss / val_emd (the learning-curve data)
    checkpoint_best.pt   best-by-val-EMD model state + rebuild info

Runs are independent; a single process sweeps them serially on the GPU (each is small).
Use ``--methods`` / ``--seeds`` to run a subset (e.g. to parallelise across processes).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from set_transformer.data.dataset import POMDPDataset

from _common import (
    ARCH,
    BATCH_SIZE,
    DATA_DIR,
    METHOD_ORDER,
    METHOD_REGISTRY,
    NUM_EPOCHS,
    RUNS_DIR,
    train_one,
)


def build_loaders(batch_size: int, data_dir: Path = DATA_DIR):
    train_pts = np.load(data_dir / "train.points.npy")
    eval_pts = np.load(data_dir / "eval.points.npy")
    train_loader = DataLoader(POMDPDataset(train_pts), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(POMDPDataset(eval_pts), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--loss", choices=["chamfer", "sinkhorn"], default="chamfer",
                   help="differentiable training objective; eval is always exact EMD")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", type=Path, default=RUNS_DIR)
    p.add_argument("--data_dir", type=Path, default=DATA_DIR)
    args = p.parse_args()

    train_loader, val_loader = build_loaders(BATCH_SIZE, args.data_dir)
    total = len(args.methods) * len(args.seeds)
    done = 0
    for method in args.methods:
        spec = METHOD_REGISTRY[method]
        for seed in args.seeds:
            t0 = time.time()
            best_state, history, best_emd = train_one(
                method, train_loader, val_loader, seed, args.device, args.num_epochs,
                loss_type=args.loss,
            )
            run_dir = args.out_dir / method / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            np.savez(run_dir / "history.npz", **history)
            torch.save(
                {
                    "method": method,
                    "model_type": spec.model_type,
                    "arch": ARCH,
                    "loss_type": args.loss,
                    "seed": seed,
                    "state_dict": best_state,
                    "best_val_emd": best_emd,
                },
                run_dir / "checkpoint_best.pt",
            )
            done += 1
            print(f"[{done}/{total}] {spec.label} seed{seed}: "
                  f"best val EMD {best_emd:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
