"""How fast does the exact posterior converge on Odd-Even? Many episodes.

The single-episode animation shows the mechanism; this shows the rate, over
many episodes, so a fast or slow seed cannot be mistaken for the typical
case.

The point of the figure: parity is settled at step 0 in EVERY episode
(the observation model puts zero mass on the opposite parity, so one draw
halves the space), while locating the state WITHIN that parity is slow and
has a long tail -- states near the boundary of [1,n] converge differently
from interior ones because the same-parity neighbourhood is one-sided.

Usage:
    python3 viz/convergence_profile.py --n 50 --steps 30 --episodes 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[3]), str(_HERE.parents[3] / "set_transformer"),
           str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from animate_episode import belief_stats, rollout


def run(n: int, steps: int, episodes: int, seed0: int, obs_per_step: int):
    truth, eff, ent_par, mean_err = [], [], [], []
    true_states = []
    for k in range(episodes):
        roll = rollout(n_dist_size=n, steps=steps, episode_seed=seed0 + k,
                       policy="posterior_mean", obs_per_step=obs_per_step)
        st = belief_stats(roll["points"], roll["beliefs"], roll["true_state"])
        truth.append(st["truth_mass"])
        eff.append(st["eff_support"])
        ent_par.append(st["parity_mass"])
        mean_err.append(np.abs(st["mean"] - roll["true_state"]))
        true_states.append(roll["true_state"])
    L = min(len(t) for t in truth)
    to = lambda xs: np.stack([x[:L] for x in xs])
    return {"truth": to(truth), "eff": to(eff), "parity": to(ent_par),
            "mean_err": to(mean_err), "true_states": np.array(true_states),
            "n": n, "L": L}


def plot(res, out_path: Path, dpi: int = 130):
    L = res["L"]
    ts = np.arange(L)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    def band(ax, M, label, color):
        med = np.median(M, axis=0)
        q1, q3 = np.percentile(M, [25, 75], axis=0)
        p10, p90 = np.percentile(M, [10, 90], axis=0)
        ax.fill_between(ts, p10, p90, color=color, alpha=0.15, lw=0)
        ax.fill_between(ts, q1, q3, color=color, alpha=0.30, lw=0)
        ax.plot(ts, med, color=color, lw=1.9, label=f"{label} (median)")

    ax = axes[0, 0]
    band(ax, res["truth"], "P(true state)", "#2a9d8f")
    ax.axhline(1.0 / res["n"], color="#999", ls=":", lw=1.0)
    ax.text(L * 0.98, 1.0 / res["n"], " prior 1/n", fontsize=7, ha="right",
            va="bottom", color="#666")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("mass on the true state", fontsize=10)
    ax.set_xlabel("step"); ax.set_ylabel("P(true)")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[0, 1]
    band(ax, res["eff"], "$2^H$", "#3b6fb6")
    ax.axhline(res["n"], color="#999", ls=":", lw=1.0)
    ax.axhline(res["n"] / 2, color="#d1495b", ls=":", lw=1.2)
    ax.text(L * 0.98, res["n"] / 2, " n/2 = one parity", fontsize=7,
            ha="right", va="bottom", color="#d1495b")
    ax.set_yscale("log"); ax.set_ylim(0.85, res["n"] * 1.6)
    ax.set_title("effective support $2^H$ (states still live)", fontsize=10)
    ax.set_xlabel("step"); ax.set_ylabel("$2^H$")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1, 0]
    band(ax, res["parity"], "correct-parity mass", "#8a4fbd")
    ax.set_ylim(-0.03, 1.06)
    ax.set_title("parity is settled before step 1 in every episode", fontsize=10)
    ax.set_xlabel("step"); ax.set_ylabel("mass on true parity")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1, 1]
    band(ax, res["mean_err"], "|posterior mean - truth|", "#e07a1f")
    # 1/sqrt(t) reference, matched at t=1: the i.i.d. accumulation rate
    t = np.arange(1, L)
    ref = np.median(res["mean_err"][:, 1]) / np.sqrt(t)
    ax.plot(t, ref, color="#333", ls="--", lw=1.2,
            label=r"$\propto 1/\sqrt{t}$ reference")
    ax.set_yscale("log")
    ax.set_title("error of the posterior mean vs the i.i.d. rate", fontsize=10)
    ax.set_xlabel("step"); ax.set_ylabel("|error| (states)")
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Odd-Even exact-posterior convergence: n={res['n']}, "
        f"{res['truth'].shape[0]} episodes, {L-1} steps  "
        f"(median, IQR, 10-90%)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--obs_per_step", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    res = run(a.n, a.steps, a.episodes, a.seed0, a.obs_per_step)
    out = Path(a.out) if a.out else _HERE / "out" / f"odd_even_convergence_n{a.n}.png"
    plot(res, out)
    print(f"figure -> {out}")

    T = res["truth"]; E = res["eff"]; P = res["parity"]
    print(f"\nn={a.n}  episodes={res['truth'].shape[0]}  steps={res['L']-1}")
    print(f"{'step':>5} {'P(true) med':>12} {'P(true) p10':>12} "
          f"{'2^H med':>9} {'parity med':>11} {'P(true)>0.9':>12}")
    for i in [0, 1, 2, 4, 8, 15, 22, res["L"] - 1]:
        if i >= res["L"]:
            continue
        print(f"{i:>5} {np.median(T[:, i]):>12.4f} "
              f"{np.percentile(T[:, i], 10):>12.4f} {np.median(E[:, i]):>9.2f} "
              f"{np.median(P[:, i]):>11.4f} {(T[:, i] > 0.9).mean():>12.3f}")
    print(f"\nparity mass at step 0: min over episodes = {P[:, 0].min():.6f} "
          f"(1.0 means one observation already excluded the other parity)")
    return res


if __name__ == "__main__":
    main()
