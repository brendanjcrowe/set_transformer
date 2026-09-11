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

from set_transformer.data.dataset import IndexedDataset, POMDPDataset

from _common import (
    ARCH,
    BATCH_SIZE,
    DATA_DIR,
    METHOD_ORDER,
    METHOD_REGISTRY,
    NUM_EPOCHS,
    RUNS_DIR,
    AlignConfig,
    emd_pairs_from_matrix,
    load_emd_matrix,
    train_one,
)

# The alignment loss correlates over the batch's N(N-1)/2 pairs, so a batch too small
# gives a noisy (or degenerate) correlation. 16 samples = 120 pairs is the floor.
MIN_ALIGN_BATCH = 16


def build_loaders(batch_size: int, data_dir: Path = DATA_DIR, indexed: bool = False):
    """Train/val loaders. ``indexed`` makes the train loader yield ``(batch, indices)``,
    which the alignment loss needs to look up rows of the precomputed EMD matrix."""
    train_pts = np.load(data_dir / "train.points.npy")
    eval_pts = np.load(data_dir / "eval.points.npy")
    train_ds = POMDPDataset(train_pts)
    if indexed:
        train_ds = IndexedDataset(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(POMDPDataset(eval_pts), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, train_pts, eval_pts


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
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    # --- latent metric-alignment (off unless --align_lambda > 0) ---
    p.add_argument("--align_lambda", type=float, default=0.0,
                   help="weight on the Pearson latent-alignment term; 0 disables it")
    p.add_argument("--align_metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--align_warmup", type=int, default=15,
                   help="epochs with lambda=0 before the ramp starts")
    p.add_argument("--align_ramp", type=int, default=15,
                   help="epochs to ramp lambda linearly up to --align_lambda")
    p.add_argument("--emd_matrix", type=Path, default=None,
                   help="precomputed train EMD matrix (default <data_dir>/emd_train.npy)")
    p.add_argument("--eval_emd_matrix", type=Path, default=None,
                   help="precomputed eval EMD matrix (default <data_dir>/emd_eval.npy); "
                        "enables the held-out val_r metric")
    p.add_argument("--emd_mmap", action="store_true",
                   help="memory-map the train EMD matrix instead of loading it into RAM")
    args = p.parse_args()

    align = None
    if args.align_lambda > 0:
        if args.batch_size < MIN_ALIGN_BATCH:
            raise SystemExit(
                f"--batch_size {args.batch_size} is too small for the alignment loss "
                f"(needs >= {MIN_ALIGN_BATCH} for a meaningful number of pairs)")
        align = AlignConfig(args.align_lambda, args.align_metric,
                            args.align_warmup, args.align_ramp)

    train_loader, val_loader, train_pts, eval_pts = build_loaders(
        args.batch_size, args.data_dir, indexed=align is not None)

    emd_matrix = val_points = val_emd_pairs = None
    if align is not None:
        matrix_path = args.emd_matrix or (args.data_dir / "emd_train.npy")
        emd_matrix = load_emd_matrix(matrix_path, len(train_pts), mmap=args.emd_mmap)
        print(f"alignment: {align}  emd matrix {matrix_path} {emd_matrix.shape}", flush=True)
        eval_path = args.eval_emd_matrix or (args.data_dir / "emd_eval.npy")
        if eval_path.exists():
            val_points = torch.from_numpy(eval_pts).float()
            val_emd_pairs = emd_pairs_from_matrix(
                load_emd_matrix(eval_path, len(eval_pts))).to(args.device)
        else:
            print(f"  (no {eval_path}; val_r will be nan)", flush=True)

    total = len(args.methods) * len(args.seeds)
    done = 0
    for method in args.methods:
        spec = METHOD_REGISTRY[method]
        for seed in args.seeds:
            t0 = time.time()
            best_state, history, best_emd = train_one(
                method, train_loader, val_loader, seed, args.device, args.num_epochs,
                loss_type=args.loss, align=align, emd_matrix=emd_matrix,
                val_points=val_points, val_emd_pairs=val_emd_pairs,
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
                    "align": None if align is None else vars(align),
                },
                run_dir / "checkpoint_best.pt",
            )
            done += 1
            extra = ""
            if align is not None and len(history["val_r"]):
                extra = f", final val r {history['val_r'][-1]:.3f}"
            print(f"[{done}/{total}] {spec.label} seed{seed}: "
                  f"best val EMD {best_emd:.4f}{extra} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
