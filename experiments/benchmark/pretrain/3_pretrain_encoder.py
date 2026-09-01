"""Pretrain a set autoencoder on an env's PF beliefs, with or without latent alignment.

Produces the checkpoints the benchmark's 12 pretrained methods consume. Two arms:

    plain  reconstruction only (Sinkhorn / Chamfer)
    align  reconstruction + the Pearson latent<->EMD alignment term, so clouds that are
           close in EMD get codes that are close in latent space

On the MoG sets the aligned arm took the Set Transformer's held-out latent<->EMD
correlation from 0.674 to 0.993 and more than doubled its kNN overlap at no reconstruction
cost; whether that transfers to a *policy's* performance is what the RL sweep tests.

The encoder arch mirrors the capacity-matched RL configuration exactly (see
``registry.POOLING_ARCH`` / the ``train.py`` defaults) — a checkpoint trained at a
different width simply will not load into its extractor.

    python experiments/benchmark/pretrain/3_pretrain_encoder.py --env odd_even --encoder st
    python experiments/benchmark/pretrain/3_pretrain_encoder.py --env odd_even --encoder st --align
    # every encoder x arm for one env:
    python experiments/benchmark/pretrain/3_pretrain_encoder.py --env odd_even --all

Writes ``<out_dir>/<env>/<encoder>_<arm>/{checkpoint_best.pt,history.npz,meta.json}`` —
the layout ``run_sweep.sh`` looks up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from set_transformer.data.dataset import IndexedDataset, POMDPDataset
from set_transformer.emd_matrix import DEFAULT_BLUR, load_matrix
from set_transformer.latent_alignment import flatten_upper_triangle
from set_transformer.models import DeepSetAE, PFSetTransformer, PointNetAE
from set_transformer.training.autoencoder import (
    MIN_ALIGN_BATCH,
    AlignConfig,
    train_autoencoder,
)

DEFAULT_DATA = Path("experiments/benchmark/pretrain/data")
DEFAULT_OUT = Path("experiments/benchmark/pretrained")

# Encoder families, keyed as in the method registry. ``arch`` must match what the RL
# feature extractor builds, or the state_dict will not load.
ENCODERS = {
    "st": (PFSetTransformer, dict(num_inds=32, dim_hidden=64, num_heads=4, ln=True)),
    "ds": (DeepSetAE, dict(dim_hidden=128)),
    "pn": (PointNetAE, dict(dim_hidden=128)),
}


def build_encoder(kind: str, num_particles: int, particle_dim: int,
                  num_encodings: int, dim_encoder: int) -> torch.nn.Module:
    cls, arch = ENCODERS[kind]
    return cls(num_particles=num_particles, dim_particles=particle_dim,
               num_encodings=num_encodings, dim_encoder=dim_encoder, **arch)


def encoder_module(model: torch.nn.Module) -> torch.nn.Module:
    """The encoder half — the part the RL policy actually runs, and so the part the
    capacity-parity claim is about. ``PFSetTransformer`` names it ``set_transformer``;
    the pooling autoencoders name it ``encoder``."""
    return getattr(model, "encoder", None) or model.set_transformer


def load_split(data_dir: Path, split: str) -> np.ndarray:
    path = data_dir / f"{split}.points.npy"
    if not path.exists():
        raise SystemExit(f"{path} not found — run 1_collect_pf_dataset.py first")
    return np.load(path)


def run_one(kind: str, aligned: bool, args, train_pts, eval_pts, emd_train, val_points,
            val_pairs) -> dict:
    arm = "align" if aligned else "plain"
    tag = f"{kind}_{arm}"
    print(f"\n=== {args.env} / {tag} ===", flush=True)

    train_ds = POMDPDataset(train_pts)
    if aligned:
        train_ds = IndexedDataset(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(POMDPDataset(eval_pts), batch_size=args.batch_size,
                            shuffle=False)

    model = build_encoder(kind, train_pts.shape[1], train_pts.shape[2],
                          args.num_encodings, args.dim_encoder).to(args.device)
    n_params = sum(p.numel() for p in encoder_module(model).parameters())
    print(f"encoder params: {n_params:,}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    align = AlignConfig(args.align_lambda, args.align_metric, args.align_warmup,
                        args.align_ramp) if aligned else None

    t0 = time.time()
    state, history, best_emd = train_autoencoder(
        model, train_loader, val_loader, args.device, args.num_epochs,
        lr=args.lr, loss_type=args.loss, sinkhorn_blur=args.blur,
        align=align, emd_matrix=emd_train if aligned else None,
        val_points=val_points, val_emd_pairs=val_pairs,
        progress_every=args.progress_every,
    )
    wall = time.time() - t0

    run_dir = args.out_dir / args.env / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "history.npz", **history)
    torch.save({
        "env": args.env, "encoder": kind, "aligned": aligned, "seed": args.seed,
        "arch": {"num_encodings": args.num_encodings, "dim_encoder": args.dim_encoder,
                 **ENCODERS[kind][1]},
        "num_particles": int(train_pts.shape[1]), "particle_dim": int(train_pts.shape[2]),
        "loss_type": args.loss, "state_dict": state, "best_val_emd": best_emd,
        "align": None if align is None else vars(align),
    }, run_dir / "checkpoint_best.pt")

    meta = {
        "env": args.env, "encoder": kind, "arm": arm, "seed": args.seed,
        "encoder_params": int(n_params), "num_epochs": args.num_epochs,
        "loss_type": args.loss, "blur": args.blur, "best_val_emd": float(best_emd),
        "wall_clock_sec": round(wall, 1),
        "align": None if align is None else vars(align),
        "best_epoch": int(history["best_epoch"]),
        "final_val_r": float(history["val_r"][-1]) if "val_r" in history else None,
        # val_r AT the selected checkpoint -- the number that describes what actually
        # ships, as opposed to where training happened to end.
        "selected_val_r": (float(history["val_r"][int(history["best_epoch"])])
                           if "val_r" in history else None),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    extra = (f", selected val r {meta['selected_val_r']:.3f} (epoch {meta['best_epoch']})"
             if meta["selected_val_r"] is not None else "")
    print(f"{tag}: best val EMD {best_emd:.4f}{extra} ({wall:.0f}s) -> {run_dir}",
          flush=True)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--encoder", choices=list(ENCODERS), default=None)
    ap.add_argument("--align", action="store_true", help="train the aligned arm")
    ap.add_argument("--all", action="store_true",
                    help="every encoder x arm for this env (6 checkpoints)")
    ap.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    # arch — must match the RL extractor
    ap.add_argument("--num_encodings", type=int, default=8)
    ap.add_argument("--dim_encoder", type=int, default=2)
    # optimization
    ap.add_argument("--num_epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["chamfer", "sinkhorn"], default="sinkhorn")
    ap.add_argument("--blur", type=float, default=DEFAULT_BLUR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress_every", type=int, default=10)
    # alignment
    ap.add_argument("--align_lambda", type=float, default=0.2)
    ap.add_argument("--align_metric", choices=["cosine", "euclidean"], default="cosine")
    ap.add_argument("--align_warmup", type=int, default=15)
    ap.add_argument("--align_ramp", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.all and args.encoder is None:
        raise SystemExit("give --encoder {st,ds,pn} or --all")
    if args.batch_size < MIN_ALIGN_BATCH and (args.align or args.all):
        raise SystemExit(f"--batch_size {args.batch_size} is too small for the alignment "
                         f"loss (needs >= {MIN_ALIGN_BATCH} for a meaningful correlation)")

    env_data = args.data_dir / args.env
    train_pts, eval_pts = load_split(env_data, "train"), load_split(env_data, "eval")
    print(f"{args.env}: train {train_pts.shape}, eval {eval_pts.shape}", flush=True)

    jobs = ([(k, a) for k in ENCODERS for a in (False, True)] if args.all
            else [(args.encoder, args.align)])

    emd_train = val_points = val_pairs = None
    if any(aligned for _, aligned in jobs):
        emd_path = env_data / "emd_train.npy"
        if not emd_path.exists():
            raise SystemExit(f"{emd_path} not found — run 2_precompute_emd.py first")
        emd_train = load_matrix(emd_path, len(train_pts))
        eval_emd = env_data / "emd_eval.npy"
        if eval_emd.exists():
            val_points = torch.from_numpy(eval_pts).float()
            val_pairs = flatten_upper_triangle(
                torch.from_numpy(np.array(load_matrix(eval_emd, len(eval_pts)),
                                          dtype=np.float32))).to(args.device)
        else:
            print(f"  (no {eval_emd}; held-out val_r will be nan)", flush=True)

    summary = [run_one(kind, aligned, args, train_pts, eval_pts, emd_train,
                       val_points, val_pairs) for kind, aligned in jobs]

    print("\n=== summary ===", flush=True)
    for m in summary:
        r = "" if m["selected_val_r"] is None else f"  val_r {m['selected_val_r']:.3f}"
        print(f"  {m['encoder']}_{m['arm']:5s} val_emd {m['best_val_emd']:.4f}{r}",
              flush=True)


if __name__ == "__main__":
    main()
