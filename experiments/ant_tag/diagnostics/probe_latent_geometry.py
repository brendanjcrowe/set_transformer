"""Does latent-metric alignment compress the beliefs the policy actually needs to tell apart?

Hypothesis under test (smart_mid_slow_v15, 2026-09-08): alignment matches latent
distances to Sinkhorn distances between whole beliefs. Two near-uniform
"perimeter frame" beliefs that differ only in WHICH wall the hole is on are
close in Sinkhorn distance, so the aligned encoder should place them closer
together than the plain encoder does, and a linear readout of hole/wall
structure should be worse from the aligned latent. Collapsed beliefs, which
the metric separates well, should show the opposite or no difference.

Two tests on beliefs saved by probe_belief_shift.py (--save_beliefs_dir):

  1. Latent resolution per spread class: mean pairwise latent distance within
     a class divided by the mean pairwise distance over all beliefs, and the
     participation ratio (effective dimensionality) of the class's latents.
  2. Linear readout (ridge, 5-fold CV R^2) from the 64-d frozen latent to
     belief-derived targets -- weighted mean (2), weighted spread (1), mass on
     each wall band (4: x<-3.5, x>3.5, y<-3.5, y>3.5) -- and, when the files
     carry ant_xy / target_xy, to the TRUE target position relative to the ant
     (2) and to the true target's absolute position (2). Reported per class.

    python3 diagnostics/probe_latent_geometry.py \
        --plain_ckpt <plain checkpoint_best.pt> --align_ckpt <aligned checkpoint_best.pt> \
        --beliefs belief_shift/beliefs_3M/*.npz --n_per_class 1500
"""
from __future__ import annotations

import argparse
import glob
import importlib
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

CLASSES = ("collapsed", "intermediate", "frame")   # frame = spread > 2.7 (more than uniform)


@torch.no_grad()
def latents(model, frame, P, W, batch=512) -> np.ndarray:
    out = []
    for i in range(0, len(P), batch):
        x = torch.from_numpy((P[i:i + batch] - frame["centre"]) / frame["scale"]).float()
        w = torch.from_numpy(W[i:i + batch]).float()
        z = model.set_transformer(pbs.model_input(x, w, frame["weighted"]))
        out.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(out)


def classes(spread: np.ndarray) -> np.ndarray:
    return np.where(spread < 0.5, 0, np.where(spread <= 2.7, 1, 2))


def wall_mass(P, W, band=3.5):
    w = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    return np.stack([(w * (P[..., 0] < -band)).sum(1), (w * (P[..., 0] > band)).sum(1),
                     (w * (P[..., 1] < -band)).sum(1), (w * (P[..., 1] > band)).sum(1)], axis=1)


def mean_pairwise(Z, rng, n=1200):
    idx = rng.choice(len(Z), min(n, len(Z)), replace=False)
    z = Z[idx]
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    return float(d[np.triu_indices(len(z), 1)].mean())


def participation_ratio(Z):
    zc = Z - Z.mean(0)
    ev = np.linalg.eigvalsh(np.cov(zc.T))
    ev = np.clip(ev, 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))


def ridge_cv_r2(Z, Y, rng, alpha=1.0, folds=5) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    Y = Y.reshape(len(Y), -1)
    preds = np.zeros_like(Y, dtype=float)
    for tr, te in KFold(folds, shuffle=True, random_state=int(rng.integers(1e6))).split(Z):
        sc = StandardScaler().fit(Z[tr])
        m = Ridge(alpha=alpha).fit(sc.transform(Z[tr]), Y[tr])
        preds[te] = m.predict(sc.transform(Z[te])).reshape(len(te), -1)
    return float(r2_score(Y, preds, multioutput="uniform_average"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plain_ckpt", required=True)
    ap.add_argument("--align_ckpt", required=True)
    ap.add_argument("--beliefs", nargs="+", required=True, help="npz files from probe_belief_shift --save_beliefs_dir")
    ap.add_argument("--n_per_class", type=int, default=1500)
    ap.add_argument("--stride", type=int, default=3, help="keep every k-th step (beliefs are temporally correlated)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(8)

    files = [f for pat in args.beliefs for f in sorted(glob.glob(pat))]
    P, W, S, A, T = [], [], [], [], []
    for f in files:
        z = np.load(f)
        sl = slice(0, None, args.stride)
        P.append(z["particles"][sl]); W.append(z["weights"][sl]); S.append(z["spread"][sl])
        if "ant_xy" in z.files:
            A.append(z["ant_xy"][sl]); T.append(z["target_xy"][sl])
    P, W, S = np.concatenate(P), np.concatenate(W), np.concatenate(S)
    have_state = len(A) == len(files) and len(files) > 0
    if have_state:
        A, T = np.concatenate(A), np.concatenate(T)
    cls = classes(S)
    # stratified subsample so every class has the same n
    keep = np.concatenate([rng.choice(np.nonzero(cls == k)[0], min(args.n_per_class, int((cls == k).sum())), replace=False)
                           for k in range(3)])
    P, W, S, cls = P[keep], W[keep], S[keep], cls[keep]
    if have_state:
        A, T = A[keep], T[keep]
    print(f"{len(files)} files -> {len(P)} beliefs after stride {args.stride} + stratification: "
          + ", ".join(f"{c} {int((cls == k).sum())}" for k, c in enumerate(CLASSES))
          + (f"; true ant/target positions available" if have_state else "; no env state in files (belief-derived targets only)"))

    encoders = {}
    for name, ck in (("plain", args.plain_ckpt), ("aligned", args.align_ckpt)):
        model, frame = pbs.load_autoencoder(ck)
        encoders[name] = latents(model, frame, P, W)

    # ---- Test 1: latent resolution per class ---------------------------------
    print("\nTEST 1 - latent geometry per belief class")
    print("  relative resolution = mean pairwise latent distance within the class / over all beliefs; "
          "PR = participation ratio (effective dimensionality) of the class's latents (max 64)")
    print(f"  {'class':13s} | {'plain: rel.res  PR':22s} | {'aligned: rel.res  PR':22s} | aligned/plain rel.res")
    for k, c in enumerate(CLASSES):
        m = cls == k
        row = []
        for name in ("plain", "aligned"):
            Z = encoders[name]
            row.append((mean_pairwise(Z[m], rng) / mean_pairwise(Z, rng), participation_ratio(Z[m])))
        print(f"  {c:13s} | {row[0][0]:7.3f}   {row[0][1]:6.1f}      | {row[1][0]:7.3f}   {row[1][1]:6.1f}      | x{row[1][0] / row[0][0]:.2f}")

    # ---- Test 2: linear readouts ----------------------------------------------
    w = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    targets = {
        "belief mean xy (2)": (w[..., None] * P).sum(1),
        "belief spread (1)": S[:, None],
        "wall mass 4-vector": wall_mass(P, W),
    }
    if have_state:
        targets["TRUE target rel. to ant (2)"] = T - A
        targets["TRUE target abs xy (2)"] = T
    print("\nTEST 2 - ridge readout from the 64-d frozen latent, 5-fold CV R^2 (1.0 = perfectly linearly decodable)")
    print(f"  {'target':28s} {'class':13s} | {'plain':>7s} | {'aligned':>7s} | diff (aligned - plain)")
    for tname, Y in targets.items():
        for k, c in list(enumerate(CLASSES)) + [(None, "all")]:
            m = np.ones(len(P), bool) if k is None else cls == k
            if m.sum() < 100:
                continue
            r = {name: ridge_cv_r2(encoders[name][m], Y[m], rng) for name in ("plain", "aligned")}
            print(f"  {tname:28s} {c:13s} | {r['plain']:7.3f} | {r['aligned']:7.3f} | {r['aligned'] - r['plain']:+.3f}")
        print()


if __name__ == "__main__":
    main()
