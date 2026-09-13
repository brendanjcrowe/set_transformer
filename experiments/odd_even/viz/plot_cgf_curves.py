"""Plot the weighted CGF K(t) = log sum_i w_i exp(t x_i) of a few real Odd-Even
posteriors, over a range of t, in the RL arm's own coordinates.

Why this picture: the mode-readout probe (domain_mds/oddeven.md, 2026-09-04)
found the 64 CGF features have effective rank 1 and every pair correlates at
|r| >= 0.999, with the leading direction tracking the posterior mean at
r = 1.00000. This script shows the mechanism directly. With particles
normalised as (s - 25.5) / 24.5 the whole posterior lives inside an interval
about 0.1 wide, so for |t| <= 10 the exponent t * (x_i - mean) stays below
~0.5 and K(t) is dominated by its linear term t * mean. Every feature is
(almost) a fixed multiple of one scalar. The shape of the belief -- the part
that says WHICH neighbour carries the mass -- sits in the residual
K(t) - t * mean, which is two to three orders of magnitude smaller.

Rows of the figure:
  1. the three posteriors themselves
  2. K(t) for each, on its own panel, with the same y range
  3. all three K(t) overlaid; the residual K(t) - t*mean overlaid, with a
     Gaussian of matched mean/variance dashed (what a mean+variance encoding
     amounts to); and K'(t), the tilted mean, overlaid

Usage:
    python3 viz/plot_cgf_curves.py --true_state 25 --steps 0 4 20 --t_max 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ODD_EVEN_DIR = _HERE.parent
_REPO_ROOT = _HERE.parents[3]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "set_transformer"), str(_ODD_EVEN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

import pdomains  # noqa: F401
from set_transformer.rl.domains import odd_even as variants  # noqa: E402 - the registry

COLOURS = ("#3b6fb6", "#d1495b", "#2a9d8f")


def collect_beliefs(variant: str, true_state: int, steps: list[int], seed: int):
    """Exact posteriors at the requested step indices of one pinned episode.

    Step k is the belief the policy acts on at step k+1: b_0 is the posterior
    after reset's own observation, b_k after k more observations.
    """
    resolved = variants.resolve(variant)
    env = gym.make(resolved.env_id, true_state=true_state)
    _, info = env.reset(seed=seed)
    points = np.asarray(info["belief_points"], dtype=float)
    beliefs = {0: np.asarray(info["belief"], dtype=float)}
    for k in range(1, max(steps) + 1):
        _, _, term, trunc, info = env.step(0)   # actions do not touch the belief
        beliefs[k] = np.asarray(info["belief"], dtype=float)
        if term or trunc:
            break
    env.close()
    return points, [beliefs[k] for k in steps]


def cgf(t: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """K(t) = log sum_i w_i exp(t x_i), via logsumexp. t: [T], x/w: [N]."""
    mask = w > 0
    x, w = x[mask], w[mask]
    a = t[:, None] * x[None, :] + np.log(w)[None, :]
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True))).ravel()


def cgf_grad(t: np.ndarray, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """K'(t) = sum_i softmax_i(t x_i + log w_i) x_i, the tilted mean."""
    mask = w > 0
    x, w = x[mask], w[mask]
    a = t[:, None] * x[None, :] + np.log(w)[None, :]
    a = a - a.max(axis=1, keepdims=True)
    p = np.exp(a)
    p /= p.sum(axis=1, keepdims=True)
    return p @ x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="oe50_short")
    ap.add_argument("--true_state", type=int, default=25)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 4, 20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t_max", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    points, beliefs = collect_beliefs(args.variant, args.true_state, args.steps, args.seed)
    centre, scale = variants.state_centre(args.variant), variants.state_scale(args.variant)
    x = (points - centre) / scale                       # the RL arm's coordinates
    t = np.linspace(-args.t_max, args.t_max, 801)

    curves, grads, stats = [], [], []
    for w in beliefs:
        mu = float(w @ x)
        var = float(w @ (x - mu) ** 2)
        curves.append(cgf(t, x, w))
        grads.append(cgf_grad(t, x, w))
        stats.append((mu, var))

    n = len(beliefs)
    fig, axes = plt.subplots(3, n, figsize=(5.2 * n, 12.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.28, left=0.06, right=0.98,
                        top=0.93, bottom=0.05)

    # row 1: the posteriors, zoomed to where the mass is
    lo = int(min(points[w > 1e-3].min() for w in beliefs)) - 2
    hi = int(max(points[w > 1e-3].max() for w in beliefs)) + 2
    for j, (w, k, c) in enumerate(zip(beliefs, args.steps, COLOURS)):
        ax = axes[0, j]
        ax.bar(points, w, color=c, width=0.8)
        ax.set_xlim(lo - 0.6, hi + 0.6)
        ax.set_xticks(range(max(1, lo), min(50, hi) + 1, 2))
        ax.set_ylim(0, 1.0)
        mu, var = stats[j]
        ax.set_title(f"belief after {k} observation{'s' if k != 1 else ''}"
                     f"  (true state {args.true_state})\n"
                     f"mean {mu * scale + centre:.2f}, std {np.sqrt(var) * scale:.2f} states"
                     f"   |   normalised: mean {mu:.3f}, std {np.sqrt(var):.4f}",
                     fontsize=10)
        ax.set_xlabel("state")
        ax.set_ylabel("probability mass")

    # row 2: K(t) per belief, shared y range so the panels are comparable
    ymin = min(c.min() for c in curves)
    ymax = max(c.max() for c in curves)
    pad = 0.05 * (ymax - ymin)
    for j, (K, k, c) in enumerate(zip(curves, args.steps, COLOURS)):
        ax = axes[1, j]
        mu, var = stats[j]
        ax.plot(t, K, color=c, lw=2.2, label="K(t)")
        ax.plot(t, t * mu, color="k", lw=1.0, ls=":", label="t · mean")
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.axhline(0, color="#999", lw=0.6)
        ax.axvline(0, color="#999", lw=0.6)
        ax.set_xlabel("t")
        ax.set_ylabel("K(t) = log Σ w exp(t x)")
        ax.set_title(f"K(t), belief after {k} obs", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # row 3, col 0: overlay of K(t)
    ax = axes[2, 0]
    for K, k, c in zip(curves, args.steps, COLOURS):
        ax.plot(t, K, color=c, lw=2.0, label=f"after {k} obs")
    ax.set_xlabel("t")
    ax.set_ylabel("K(t)")
    ax.set_title("all three K(t): nearly straight lines with slope = mean", fontsize=10)
    ax.legend(fontsize=8)

    # row 3, col 1: the residual, i.e. K(t) with the linear term removed
    ax = axes[2, 1]
    for K, (mu, var), k, c in zip(curves, stats, args.steps, COLOURS):
        ax.plot(t, K - t * mu, color=c, lw=2.0, label=f"after {k} obs")
        ax.plot(t, 0.5 * var * t ** 2, color=c, lw=1.2, ls="--")
    ax.set_xlabel("t")
    ax.set_ylabel("K(t) − t · mean")
    ax.set_title("residual after removing t·mean\n"
                 "dashed = ½ σ² t², the Gaussian (mean+variance) approximation",
                 fontsize=10)
    ax.legend(fontsize=8)

    # row 3, col 2: K'(t), the tilted mean, back in state units
    if n >= 3:
        ax = axes[2, 2]
        for g, k, c in zip(grads, args.steps, COLOURS):
            ax.plot(t, g * scale + centre, color=c, lw=2.0, label=f"after {k} obs")
        ax.axhline(args.true_state, color="k", lw=0.8, ls=":")
        ax.set_xlabel("t")
        ax.set_ylabel("K'(t) in state units")
        ax.set_title("K'(t): tilted mean, slides from the lowest\n"
                     "to the highest state with mass as t goes − to +", fontsize=10)
        ax.legend(fontsize=8)
        for j in range(3, n):
            axes[2, j].axis("off")

    fig.suptitle(f"Weighted CGF of real Odd-Even posteriors, {args.variant}, "
                 f"particles normalised as (s − {centre}) / {scale}, "
                 f"t ∈ [−{args.t_max:g}, {args.t_max:g}]", fontsize=12)

    out = Path(args.out) if args.out else _HERE / "out" / (
        f"cgf_curves_{args.variant}_true{args.true_state}_steps"
        f"{'_'.join(map(str, args.steps))}_t{args.t_max:g}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"figure -> {out}")

    # the numbers behind the picture
    print(f"\nnormalised coordinates: neighbouring same-parity states are "
          f"{2 / scale:.4f} apart; t_max * that = {args.t_max * 2 / scale:.3f}")
    print(f"{'after k obs':>12} {'mean':>8} {'std':>8} {'K(t_max)':>10} "
          f"{'t_max*mean':>11} {'residual':>10} {'½σ²t²':>9} {'resid-gauss':>12}")
    for K, (mu, var), k in zip(curves, stats, args.steps):
        Kmax = K[-1]
        resid = Kmax - args.t_max * mu
        gauss = 0.5 * var * args.t_max ** 2
        print(f"{k:>12} {mu:>8.4f} {np.sqrt(var):>8.4f} {Kmax:>10.4f} "
              f"{args.t_max * mu:>11.4f} {resid:>10.5f} {gauss:>9.5f} "
              f"{resid - gauss:>12.6f}")


if __name__ == "__main__":
    main()
