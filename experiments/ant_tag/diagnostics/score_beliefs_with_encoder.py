"""Score saved beliefs with pretraining checkpoints -- did re-pretraining close the belief shift?

Companion to probe_belief_shift.py (which rolls policies out). This one takes
the beliefs that probe already saved (``belief_shift/beliefs_*/<run>.npz``)
and any number of autoencoder checkpoints, and reports each checkpoint's
Sinkhorn reconstruction loss per spread class, on:

  * the collector's dataset (random rows; the encoder's "home" distribution),
  * the saved policy beliefs, EXCLUDING the rows a mixed pretraining set used
    (read from that set's metadata, ``policy_beliefs[*].indices``, widened by
    ``--exclude_margin`` steps on each side because consecutive beliefs of one
    episode are near-duplicates). So the number is a held-out score.

Success criterion for the 2026-09-08 mixed pretraining: the loss on HELD-OUT
policy beliefs matches the loss on the policy beliefs the encoder trained on
(no residual shift). Not "ratio to the collector rows near 1": hollow
perimeter frames are intrinsically harder to reconstruct through the 64-d
bottleneck, so the new plain encoder sits at 3.0x the collector loss with
in-sample 0.0081 vs held-out 0.0083 -- shift closed -- against 6.5x for the
collector-only encoder.

    python3 diagnostics/score_beliefs_with_encoder.py \
        --dataset data/smart_mid_slow_v15_pf_dataset.npz \
        --beliefs "belief_shift/beliefs_3M/*.npz" "belief_shift/beliefs_final/*.npz" \
        --exclude_from data/smart_mid_slow_v15_mixed10k10k_pf_dataset.npz \
        --ckpt old_plain=<ckpt> old_aligned=<ckpt> new_plain=<ckpt> new_aligned=<ckpt>
"""
from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (str(_REPO_ROOT), str(_ANT_TAG_DIR), str(_ANT_TAG_DIR / "diagnostics")):
    if p not in sys.path:
        sys.path.insert(0, p)

pbs = importlib.import_module("probe_belief_shift")
CLASSES = ("collapsed", "intermediate", "diffuse")


def spread_of(P, W):
    w = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    mu = (w[..., None] * P).sum(1, keepdims=True)
    return np.sqrt((w[..., None] * (P - mu) ** 2).sum(1)).mean(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="collector .npz (reference distribution)")
    ap.add_argument("--n_dataset", type=int, default=2000)
    ap.add_argument("--beliefs", nargs="+", required=True, help="glob(s) of probe_belief_shift belief files")
    ap.add_argument("--exclude_from", default=None, help="mixed pretraining .npz whose metadata lists the policy rows it used")
    ap.add_argument("--exclude_margin", type=int, default=5, help="also exclude this many neighbouring steps on each side")
    ap.add_argument("--stride", type=int, default=7, help="thin the held-out stream (temporal correlation)")
    ap.add_argument("--max_per_file", type=int, default=600)
    ap.add_argument("--ckpt", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(8)

    z = np.load(args.dataset, allow_pickle=True)
    di = rng.choice(len(z["particles"]), min(args.n_dataset, len(z["particles"])), replace=False)
    D_P, D_W = z["particles"][di].astype(np.float32), z["weights"][di].astype(np.float32)

    used = {}
    if args.exclude_from:
        md = json.loads(str(np.load(args.exclude_from, allow_pickle=True)["metadata"]))
        for rec in md["policy_beliefs"]:
            used[os.path.normpath(rec["file"])] = np.asarray(rec["indices"], int)

    files = sorted(f for pat in args.beliefs for f in glob.glob(pat))
    H_P, H_W, n_excl = [], [], 0
    for f in files:
        b = np.load(f); n = len(b["particles"])
        keep = np.ones(n, bool)
        key = os.path.normpath(os.path.relpath(f, _ANT_TAG_DIR))
        if key in used:
            for i in used[key]:
                keep[max(0, i - args.exclude_margin): i + args.exclude_margin + 1] = False
        idx = np.nonzero(keep)[0][:: args.stride]
        n_excl += int((~keep).sum())
        if len(idx) > args.max_per_file:
            idx = np.sort(rng.choice(idx, args.max_per_file, replace=False))
        H_P.append(b["particles"][idx].astype(np.float32)); H_W.append(b["weights"][idx].astype(np.float32))
    H_P, H_W = np.concatenate(H_P), np.concatenate(H_W)
    print(f"reference: {len(D_P)} dataset rows; held-out policy beliefs: {len(H_P)} from {len(files)} files "
          f"({n_excl} rows excluded as used-or-adjacent in {args.exclude_from or 'nothing'}, stride {args.stride})")

    sets = {"dataset": (D_P, D_W), "policy_heldout": (H_P, H_W)}
    cls = {k: pbs.classify(P, W)[0] for k, (P, W) in sets.items()}   # classify -> (class ids, spread)
    for k, (P, W) in sets.items():
        c = cls[k]; print(f"  {k:15s} mix: " + " / ".join(f"{CLASSES[i]} {100 * (c == i).mean():.0f}%" for i in range(3))
              + f", frames(>2.7) {100 * (spread_of(P, W) > 2.7).mean():.0f}%")

    results = {}
    print(f"\n{'checkpoint':14s} {'set':15s} | {'loss':>8s} | {'x dataset':>9s} | " + " | ".join(f"{c:>12s}" for c in CLASSES))
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        model, frame = pbs.load_autoencoder(path)
        res = {}
        for k, (P, W) in sets.items():
            losses = pbs.per_sample_loss(model, frame, P, W)
            c = cls[k]
            res[k] = {"loss_mean": float(losses.mean()),
                      **{f"{CLASSES[i]}_loss": (float(losses[c == i].mean()) if (c == i).any() else None) for i in range(3)}}
        ref = res["dataset"]["loss_mean"]
        for k in sets:
            r = res[k]
            print(f"{name:14s} {k:15s} | {r['loss_mean']:8.4f} | {r['loss_mean'] / ref:9.2f} | "
                  + " | ".join(f"{(r[f'{c}_loss'] if r[f'{c}_loss'] is not None else float('nan')):12.4f}" for c in CLASSES))
        results[name] = {"ckpt": path, "val_loss_recorded": frame.get("val_loss"), **res}
    if args.out_json:
        json.dump({"args": vars(args), "results": results}, open(args.out_json, "w"), indent=1)
        print("wrote", args.out_json)


if __name__ == "__main__":
    main()
