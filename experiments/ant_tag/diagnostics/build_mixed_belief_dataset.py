"""Build a pretraining set that contains the beliefs the RL policy actually sees.

Motivation (domain_mds/smart_mid_slow_v15_ant_tag.md, belief-shift probe,
2026-09-08): the collector's dataset is 41 / 34 / 25 % collapsed /
intermediate / diffuse with one-or-two-wall diffuse clouds, while trained
policies at radius 1.5 spend 78-95 % of their steps in hollow perimeter-frame
beliefs the pretrained encoders reconstruct 4-9x worse than their validation
loss. This script mixes:

  * ``--n_original`` rows sampled uniformly (seeded, without replacement) from
    the collector's .npz -- keeps the collapsed / intermediate coverage and
    makes the new set a superset in kind of the old one; and
  * ``--n_policy`` rows from the belief files probe_belief_shift.py saved
    (``belief_shift/beliefs_*/<run>.npz``: particles, weights, spread per
    step of 30 deterministic episodes at the real radius). Each file gets an
    equal share, taken at EVENLY SPACED indices, so every policy x stage
    contributes the same number of beliefs and consecutive rows of one
    episode (~300 steps) are ~30 steps apart -- weak temporal correlation,
    which matters because 3_train_st.py splits train/val at random
    (PITFALLS.md section 4 on duplicates leaking into the val split).

Output has the collector's exact format (particles [S,N,2] float32 in RAW env
coordinates, weights [S,N], particle_scale, metadata JSON) so 2b_precompute_emd.py
and 3_train_st.py consume it unchanged. Rows are shuffled (seeded).

    python3 diagnostics/build_mixed_belief_dataset.py \
        --original data/smart_mid_slow_v15_pf_dataset.npz \
        --policy_glob "belief_shift/beliefs_3M/*.npz" "belief_shift/beliefs_final/*.npz" \
        --n_original 10000 --n_policy 10000 \
        --out data/smart_mid_slow_v15_mixed10k10k_pf_dataset.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ANT_TAG_DIR = Path(__file__).resolve().parents[1]


def weighted_spread(P: np.ndarray, W: np.ndarray) -> np.ndarray:
    """The collector's / probe's measure: per-coordinate weighted std, averaged over coordinates."""
    w = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    mu = (w[..., None] * P).sum(1, keepdims=True)
    var = (w[..., None] * (P - mu) ** 2).sum(1)
    return np.sqrt(var).mean(1)


def class_mix(spread: np.ndarray) -> dict:
    return {
        "collapsed_lt0.5": float((spread < 0.5).mean()),
        "intermediate": float(((spread >= 0.5) & (spread < 2.5)).mean()),
        "diffuse_ge2.5": float((spread >= 2.5).mean()),
        "frame_gt2.7": float((spread > 2.7).mean()),
        "spread_median": float(np.median(spread)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--original", required=True, help="collector .npz")
    ap.add_argument("--policy_glob", nargs="+", required=True, help="glob(s) for probe_belief_shift belief files")
    ap.add_argument("--n_original", type=int, default=10000)
    ap.add_argument("--n_policy", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry_run", action="store_true", help="report the mix, write nothing")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    z = np.load(args.original, allow_pickle=True)
    P0, W0 = z["particles"], z["weights"]
    scale = float(z["particle_scale"]) if "particle_scale" in z.files else 4.5
    orig_md = json.loads(str(z["metadata"])) if "metadata" in z.files else {}
    if args.n_original > len(P0):
        sys.exit(f"--n_original {args.n_original} > {len(P0)} rows in {args.original}")
    idx0 = np.sort(rng.choice(len(P0), args.n_original, replace=False))

    files = sorted(f for pat in args.policy_glob for f in glob.glob(pat))
    if not files:
        sys.exit("no policy belief files matched")
    per_file = np.full(len(files), args.n_policy // len(files))
    per_file[: args.n_policy - per_file.sum()] += 1        # spread the remainder
    P1, W1, prov = [], [], []
    for f, k in zip(files, per_file):
        b = np.load(f)
        n = len(b["particles"])
        if k > n:
            sys.exit(f"{f} has {n} beliefs, need {k}")
        # evenly spaced with a random phase, so files of different length all
        # yield k rows ~n/k steps apart
        start = rng.integers(0, max(1, n // k))
        sel = np.linspace(start, n - 1, k).round().astype(int)
        sel = np.unique(sel)
        P1.append(b["particles"][sel].astype(np.float32)); W1.append(b["weights"][sel].astype(np.float32))
        prov.append({"file": os.path.relpath(f, _ANT_TAG_DIR), "n_available": int(n), "n_taken": int(len(sel)),
                     "mean_step_gap": float(np.diff(sel).mean()) if len(sel) > 1 else None,
                     # exact rows used, so a held-out evaluation can exclude them
                     "indices": sel.tolist()})
    P1, W1 = np.concatenate(P1), np.concatenate(W1)

    P = np.concatenate([P0[idx0].astype(np.float32), P1]); W = np.concatenate([W0[idx0].astype(np.float32), W1])
    src = np.concatenate([np.zeros(len(idx0), np.int8), np.ones(len(P1), np.int8)])
    perm = rng.permutation(len(P)); P, W, src = P[perm], W[perm], src[perm]
    assert P.shape[1:] == P0.shape[1:] and W.shape[1] == P.shape[1]
    assert np.all(np.isfinite(P)) and np.all(W >= 0)

    sp = weighted_spread(P, W)
    report = {
        "n_rows": int(len(P)), "n_original": int(len(idx0)), "n_policy": int(len(P1)),
        "mix_all": class_mix(sp), "mix_original_half": class_mix(sp[src == 0]),
        "mix_policy_half": class_mix(sp[src == 1]),
        "mix_collector_file_full": class_mix(weighted_spread(P0, W0)),
    }
    print(json.dumps(report, indent=1))
    if args.dry_run:
        return

    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=_ANT_TAG_DIR, text=True).strip()
        except Exception:
            return None
    metadata = {
        "built_by": "diagnostics/build_mixed_belief_dataset.py", "built_at": datetime.now().isoformat(timespec="seconds"),
        "variant": orig_md.get("variant"), "env_id": orig_md.get("env_id"),
        "particle_filter_class": orig_md.get("particle_filter_class"),
        "num_particles": int(P.shape[1]), "dim_particles": int(P.shape[2]), "particle_scale": scale,
        "original": {"path": os.path.relpath(args.original, _ANT_TAG_DIR), "n_rows_in_file": int(len(P0)),
                     "n_taken": int(len(idx0)), "metadata": orig_md},
        "policy_beliefs": prov, "seed": args.seed, "report": report,
        "git": {"set_transformer_head": _git("rev-parse", "HEAD")},
        "args": vars(args),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, particles=P, weights=W, particle_scale=np.float32(scale),
             source_is_policy=src, metadata=json.dumps(metadata))
    print(f"wrote {args.out}: particles {P.shape}, weights {W.shape}")


if __name__ == "__main__":
    main()
