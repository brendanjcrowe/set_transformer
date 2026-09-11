"""Does the particle filter actually track the target? Scored on saved rollouts.

Uses the belief files probe_belief_shift.py saved with env state
(``belief_shift/beliefs_final/*.npz``, ``beliefs_state/*.npz``: particles,
weights, ant_xy, target_xy per step of 30 deterministic episodes of a trained
policy at the real radius). For every step it asks where the TRUE target sits
relative to the belief:

  * mass_within_r  -- PF probability mass within r of the true target (r = tag
                      radius 1.0 and visibility radius 1.5). A uniform belief
                      over the 9x9 cage puts pi*r^2/81 = 3.9 % / 8.7 % there.
  * hole           -- no particle at all within r of the true target: the filter
                      has (wrongly) ruled the true position out.
  * mean_dist      -- distance from the weighted belief mean to the true target.
  * nn_dist        -- distance from the true target to the nearest particle.
  * ESS            -- effective sample size 1/sum(w^2), i.e. how non-uniform the
                      weights are (100 = uniform).

Everything is split by whether the target was visible at that step (ant-target
distance < visible radius) and, when not, by how many consecutive steps it has
been out of sight (inferred from the visibility sequence within a file).

    python3 diagnostics/check_pf_tracking.py --beliefs "belief_shift/beliefs_final/*.npz" \
        --visible_radius 1.5 --tag_radius 1.0 --half_width 4.5
"""
from __future__ import annotations

import argparse
import glob

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--beliefs", nargs="+", required=True)
    ap.add_argument("--visible_radius", type=float, default=1.5)
    ap.add_argument("--tag_radius", type=float, default=1.0)
    ap.add_argument("--half_width", type=float, default=4.5)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    files = sorted(f for pat in args.beliefs for f in glob.glob(pat))
    P, W, A, T, blind = [], [], [], [], []
    for f in files:
        z = np.load(f)
        if "target_xy" not in z.files:
            continue
        p, w, a, t = z["particles"], z["weights"], z["ant_xy"], z["target_xy"]
        vis = np.linalg.norm(a - t, axis=1) < args.visible_radius
        # consecutive blind steps so far (resets on a sighting; episode boundaries
        # are not stored, so a new episode's first blind steps continue the count
        # of the previous episode's tail -- a small contamination, noted)
        b = np.zeros(len(vis), int); run = 0
        for i, v in enumerate(vis):
            run = 0 if v else run + 1
            b[i] = run
        sl = slice(0, None, args.stride)
        P.append(p[sl]); W.append(w[sl]); A.append(a[sl]); T.append(t[sl]); blind.append(b[sl])
    if not P:
        raise SystemExit("no belief files with target_xy matched")
    P, W, A, T, blind = map(np.concatenate, (P, W, A, T, blind))
    w = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    n = len(P)
    print(f"{len(files)} files, {n} steps with true state")

    d_pt = np.linalg.norm(P - T[:, None, :], axis=2)                    # particle-to-true-target
    mass_tag = (w * (d_pt < args.tag_radius)).sum(1)
    mass_vis = (w * (d_pt < args.visible_radius)).sum(1)
    nn = d_pt.min(1)
    mean = (w[..., None] * P).sum(1)
    mean_d = np.linalg.norm(mean - T, axis=1)
    ess = 1.0 / np.maximum((w ** 2).sum(1), 1e-12)
    vis = np.linalg.norm(A - T, axis=1) < args.visible_radius
    area = (2 * args.half_width) ** 2
    u_tag, u_vis = np.pi * args.tag_radius ** 2 / area, np.pi * args.visible_radius ** 2 / area

    def row(name, m):
        if m.sum() == 0:
            return
        print(f"  {name:28s} n={m.sum():6d} | mass<{args.tag_radius:g} {100 * mass_tag[m].mean():5.1f}% (unif {100 * u_tag:.1f}) "
              f"| mass<{args.visible_radius:g} {100 * mass_vis[m].mean():5.1f}% (unif {100 * u_vis:.1f}) "
              f"| hole<{args.tag_radius:g} {100 * (nn[m] > args.tag_radius).mean():5.1f}% | hole<{args.visible_radius:g} {100 * (nn[m] > args.visible_radius).mean():5.1f}% "
              f"| mean-dist med {np.median(mean_d[m]):4.2f} | nn-dist med {np.median(nn[m]):4.2f} | ESS med {np.median(ess[m]):5.1f}")

    print("\nWEIGHTS: ESS median %.1f, 10th pct %.1f; steps with all-uniform weights %.1f%%; steps with ESS<50 %.1f%%"
          % (np.median(ess), np.percentile(ess, 10), 100 * np.mean(np.abs(w - 1 / P.shape[1]).max(1) < 1e-9), 100 * np.mean(ess < 50)))
    print("\nTRACKING (rows: visibility state; unif = what a uniform belief over the cage would score)")
    row("all steps", np.ones(n, bool))
    row("target VISIBLE", vis)
    row("not visible, all", ~vis)
    for lo, hi in ((1, 5), (6, 20), (21, 50), (51, 100), (101, 200), (201, 10 ** 9)):
        row(f"blind for {lo}-{hi if hi < 10**9 else '400'} steps", (~vis) & (blind >= lo) & (blind <= hi))
    print("\nVISIBLE steps: fraction with belief mean within 0.5 of the target: %.1f%%; within 1.0: %.1f%%"
          % (100 * (mean_d[vis] < 0.5).mean(), 100 * (mean_d[vis] < 1.0).mean()))
    print("NOT-visible steps: fraction where the true target is inside the ant's visibility disc per the belief? (mass within %.1f of the ANT): %.2f%% (should be ~0: the filter down-weights that disc)"
          % (args.visible_radius, 100 * (w * (np.linalg.norm(P - A[:, None, :], axis=2) < args.visible_radius)).sum(1)[~vis].mean()))


if __name__ == "__main__":
    main()
