"""Animate one Odd-Even POMDP episode: how the exact posterior evolves.

What this shows and why it is the interesting picture on this domain:

The env keeps an EXACT posterior over the n candidate states (it is a
discrete-state / discrete-observation POMDP, so Bayes is a closed form, no
approximation). That posterior travels in `info['belief']` -- never in the
agent's observation, which is only the raw integer draw. So this animation
shows what a *perfect* belief tracker knows at each step. Any encoder arm
(CGF / Gaussian / Set Transformer) sees a lossy summary of exactly this
object, which is why the shape of the convergence here bounds what any of
them can do.

Two features to watch:

1. Parity locks immediately. The observation model puts zero mass on
   integers of the opposite parity to `true_state`, so a SINGLE observation
   annihilates half the state space. Frame 1 already shows a comb: 25 spikes
   on the true parity, exact zeros between them.
2. Within a parity, convergence is slow. Same-parity neighbours (s and s+2)
   are separated only by the Gaussian's width -- at n=50 the default
   std_dev = sqrt(50)/sqrt(10) + 1 = 3.236, so neighbouring candidates have
   nearly equal likelihood and evidence accumulates as ~1/sqrt(t).

The policy is irrelevant to the belief trajectory: actions are predictions
and do not touch the hidden state or the observation model, so the posterior
evolves identically under any policy. `--policy` only changes the prediction
marker drawn on the frame.

Usage:
    python3 viz/animate_episode.py --variant oe50_short --episode_seed 3
    python3 viz/animate_episode.py --n 50 --steps 60 --policy posterior_mean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ANT_TAG_SIBLING = _HERE.parent               # experiments/odd_even
_REPO_ROOT = _HERE.parents[3]                 # repo root (see CLAUDE.md convention)
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "set_transformer"), str(_ANT_TAG_SIBLING)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")          # headless: must precede pyplot
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec

from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig


# --------------------------------------------------------------------------
# rollout
# --------------------------------------------------------------------------

def rollout(n_dist_size: int, steps: int, episode_seed: int, policy: str,
            obs_per_step: int = 1, std_dev=None):
    """Run one episode, recording the exact posterior at every step.

    Returns a dict of arrays. `beliefs` has shape [steps+1, n]: row 0 is the
    posterior after reset() (which folds in its OWN observation, so it is
    P(s|o_0) and NOT the uniform prior), row t>=1 after step t.
    """
    cfg = OddEvenPOMDPConfig(
        n_dist_size=n_dist_size,
        obs_per_step=obs_per_step,
        std_dev=std_dev,
        seed=episode_seed,
    )
    env = OddEvenPOMDP(cfg)
    obs, info = env.reset(seed=episode_seed)

    points = np.asarray(info["belief_points"], dtype=int)
    beliefs = [np.asarray(info["belief"], dtype=float)]
    observations = [np.asarray(info.get("observations", obs), dtype=int)]
    preds, rewards = [np.nan], [np.nan]

    rng = np.random.default_rng(episode_seed + 99991)

    for _ in range(steps):
        belief = beliefs[-1]
        if policy == "posterior_mean":
            # round the posterior mean to the nearest candidate state
            ev = float(np.sum(points * belief))
            pred = int(points[np.argmin(np.abs(points - ev))])
        elif policy == "posterior_mode":
            pred = int(points[np.argmax(belief)])
        elif policy == "random":
            pred = int(rng.integers(1, n_dist_size + 1))
        else:
            raise ValueError(f"unknown policy {policy!r}")

        # env actions are 0-indexed predictions
        obs, reward, terminated, truncated, info = env.step(pred - 1)
        beliefs.append(np.asarray(info["belief"], dtype=float))
        observations.append(np.asarray(info["observations"], dtype=int))
        preds.append(pred)
        rewards.append(float(reward))
        if terminated or truncated:
            break

    return {
        "points": points,
        "beliefs": np.asarray(beliefs),
        "observations": observations,
        "preds": np.asarray(preds, dtype=float),
        "rewards": np.asarray(rewards, dtype=float),
        "true_state": int(env.true_state),
        "std_dev": float(env.std_dev),
        "n": n_dist_size,
    }


# --------------------------------------------------------------------------
# derived belief statistics
# --------------------------------------------------------------------------

def belief_stats(points: np.ndarray, beliefs: np.ndarray, true_state: int):
    """Per-step scalar summaries of belief quality."""
    p = beliefs
    mean = p @ points
    var = p @ (points.astype(float) ** 2) - mean ** 2
    # mass on the correct parity: the half of the space the first
    # observation should already have selected
    same_parity = (points % 2) == (true_state % 2)
    parity_mass = p[:, same_parity].sum(axis=1)
    # P(true state) under the belief -- the sharpness that actually matters
    truth_mass = p[:, points == true_state].ravel()
    # entropy in bits, and the effective support size 2**H
    # 0 * log 0 := 0. Compute log only where p > 0 so no invalid value is
    # ever produced (np.where alone would still evaluate log2 everywhere).
    plogp = np.zeros_like(p)
    nz = p > 0
    plogp[nz] = p[nz] * np.log2(p[nz])
    entropy = -plogp.sum(axis=1)
    return {
        "mean": mean,
        "std": np.sqrt(np.maximum(var, 0.0)),
        "parity_mass": parity_mass,
        "truth_mass": truth_mass,
        "entropy": entropy,
        "eff_support": 2.0 ** entropy,
    }


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------

def animate(roll, stats, out_path: Path, fps: int = 2, dpi: int = 120):
    points, beliefs = roll["points"], roll["beliefs"]
    n_frames = len(beliefs)
    true_state = roll["true_state"]
    n = roll["n"]

    fig = plt.figure(figsize=(13, 9.0))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.7, 1.5, 1.0, 1.0],
                  hspace=0.62, wspace=0.22,
                  left=0.07, right=0.97, top=0.905, bottom=0.06)

    ax_b = fig.add_subplot(gs[0, :])     # posterior, full state range
    ax_z = fig.add_subplot(gs[1, :])     # posterior, zoomed on the truth
    ax_e = fig.add_subplot(gs[2, 0])     # effective support (log)
    ax_t = fig.add_subplot(gs[2, 1])     # P(true state)
    ax_m = fig.add_subplot(gs[3, :])     # posterior mean +/- std vs truth

    # ---- top: posterior bars -------------------------------------------
    same_parity = (points % 2) == (true_state % 2)
    colors = np.where(same_parity, "#3b6fb6", "#c9ccd1")
    bars = ax_b.bar(points, beliefs[0], color=colors, width=0.82)
    ax_b.axvline(true_state, color="#d1495b", lw=2.0, ls="--", zorder=5,
                 label=f"true state = {true_state}")
    pred_line = ax_b.axvline(np.nan, color="#2a9d8f", lw=2.0, zorder=6,
                             label="prediction")
    ax_b.set_xlim(0.4, n + 0.6)
    ax_b.set_ylim(0, 1.0)
    ax_b.set_xlabel("candidate state")
    ax_b.set_ylabel("P(state | observations)")
    ax_b.legend(loc="upper right", fontsize=9, framealpha=0.9)
    title = ax_b.set_title("")

    # ---- second row: the same posterior, zoomed --------------------------
    # At n=50 the full axis cannot resolve the odd/even comb, which is the
    # structural feature of this domain: the observation model puts EXACTLY
    # zero mass on the opposite parity, so every other slot is empty for the
    # whole episode. This panel is where that is visible.
    zlo, zhi = max(1, true_state - 12), min(n, true_state + 12)
    zwin = (points >= zlo) & (points <= zhi)
    zpts = points[zwin]
    zbars = ax_z.bar(zpts, beliefs[0][zwin], width=0.82,
                     color=np.where(same_parity[zwin], "#3b6fb6", "#c9ccd1"),
                     edgecolor="#8a9099", linewidth=0.4)
    ax_z.axvline(true_state, color="#d1495b", lw=2.0, ls="--", zorder=5)
    zpred_line = ax_z.axvline(np.nan, color="#2a9d8f", lw=2.0, zorder=6)
    ax_z.set_xlim(zlo - 0.6, zhi + 0.6)
    ax_z.set_xticks(list(range(zlo, zhi + 1)))
    ax_z.tick_params(axis="x", labelsize=7)
    ax_z.set_xlabel("candidate state (zoomed)")
    ax_z.set_ylabel("P(state | obs)")
    ax_z.set_title(
        f"zoom [{zlo},{zhi}] -- grey slots are the opposite parity: "
        f"exactly zero mass, killed by the first observation", fontsize=9)

    # ---- middle-left: effective support, log scale ---------------------
    ts = np.arange(n_frames)
    ax_e.plot(ts, stats["eff_support"], color="#3b6fb6", lw=1.6)
    ax_e.axhline(n, color="#999", ls=":", lw=1.0)
    ax_e.axhline(n / 2, color="#d1495b", ls=":", lw=1.0)
    ax_e.text(n_frames * 0.99, n / 2, " n/2 (one parity)", fontsize=7,
              color="#d1495b", va="bottom", ha="right")
    ax_e.set_yscale("log")
    ax_e.set_ylim(0.8, n * 1.5)
    ax_e.set_ylabel("eff. support $2^H$")
    ax_e.set_xlabel("step")
    ax_e.set_title("how many states are still live", fontsize=9)
    dot_e, = ax_e.plot([], [], "o", color="#1b3a5c", ms=6, zorder=5)

    # ---- middle-right: P(true state) -----------------------------------
    ax_t.plot(ts, stats["truth_mass"], color="#2a9d8f", lw=1.6)
    ax_t.set_ylim(-0.03, 1.03)
    ax_t.set_ylabel("P(true state)")
    ax_t.set_xlabel("step")
    ax_t.set_title("mass on the truth", fontsize=9)
    dot_t, = ax_t.plot([], [], "o", color="#1a6b60", ms=6, zorder=5)

    # ---- bottom: mean +/- std vs truth ---------------------------------
    ax_m.fill_between(ts, stats["mean"] - stats["std"], stats["mean"] + stats["std"],
                      color="#3b6fb6", alpha=0.20, label=r"mean $\pm$ std")
    ax_m.plot(ts, stats["mean"], color="#3b6fb6", lw=1.6, label="posterior mean")
    ax_m.axhline(true_state, color="#d1495b", ls="--", lw=1.5, label="true state")
    ax_m.set_xlabel("step")
    ax_m.set_ylabel("state")
    # Auto-range to the trajectory, not to [1,n]: on a 50-state axis the
    # whole mean+/-std ribbon is a few pixels tall and the convergence it is
    # meant to show is invisible.
    _lo = min(float((stats["mean"] - stats["std"]).min()), float(true_state))
    _hi = max(float((stats["mean"] + stats["std"]).max()), float(true_state))
    _pad = max(1.5, 0.15 * (_hi - _lo))
    ax_m.set_ylim(_lo - _pad, _hi + _pad)
    ax_m.legend(loc="upper right", fontsize=8, ncol=3, framealpha=0.9)
    ax_m.set_title("the mean is close early; the SPREAD is what takes time",
                   fontsize=9)
    dot_m, = ax_m.plot([], [], "o", color="#1b3a5c", ms=6, zorder=5)

    def frame(i):
        for bar, h in zip(bars, beliefs[i]):
            bar.set_height(h)
        for bar, h in zip(zbars, beliefs[i][zwin]):
            bar.set_height(h)
        # rescale so the comb stays legible as it sharpens
        ylim = max(0.08, float(beliefs[i].max()) * 1.25)
        ax_b.set_ylim(0, ylim)
        ax_z.set_ylim(0, ylim)

        pred = roll["preds"][i]
        xd = [pred, pred] if np.isfinite(pred) else [np.nan, np.nan]
        pred_line.set_xdata(xd)
        zpred_line.set_xdata(xd)

        obs = roll["observations"][i]
        obs_txt = ", ".join(str(int(o)) for o in np.atleast_1d(obs)[:6])
        rew = roll["rewards"][i]
        rew_txt = "--" if not np.isfinite(rew) else f"{rew:.0f}"
        pred_txt = "--" if not np.isfinite(pred) else str(int(pred))
        stage = "reset (after folding in $o_0$)" if i == 0 else f"step {i}"
        title.set_text(
            f"Odd-Even POMDP  n={n}, std_dev={roll['std_dev']:.2f}   |   {stage}\n"
            f"observation(s): {obs_txt}    prediction: {pred_txt}    reward: {rew_txt}    "
            f"P(true)={stats['truth_mass'][i]:.3f}    "
            f"eff. support={stats['eff_support'][i]:.1f} of {n}    "
            f"parity mass={stats['parity_mass'][i]:.3f}"
        )
        dot_e.set_data([i], [stats["eff_support"][i]])
        dot_t.set_data([i], [stats["truth_mass"][i]])
        dot_m.set_data([i], [stats["mean"][i]])
        return (*bars, *zbars, pred_line, zpred_line, title, dot_e, dot_t, dot_m)

    anim = animation.FuncAnimation(fig, frame, frames=n_frames,
                                   interval=1000 / fps, blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        anim.save(str(out_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(str(out_path), writer=animation.FFMpegWriter(fps=fps, bitrate=2400),
                  dpi=dpi)
    plt.close(fig)
    return out_path


def save_summary_png(roll, stats, out_path: Path, dpi: int = 130):
    """Static small-multiples of the posterior at chosen steps."""
    beliefs, points = roll["beliefs"], roll["points"]
    n_frames = len(beliefs)
    picks = [p for p in (0, 1, 2, 4, 8, 15, 22, n_frames - 1) if p < n_frames]
    picks = sorted(set(picks))
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.1 * nrow),
                             squeeze=False)
    ts = roll["true_state"]
    same = (points % 2) == (ts % 2)
    # Zoom around the truth: at n=50 the full axis hides the odd/even comb,
    # which is the structural feature of this domain. +/-12 keeps ~12
    # same-parity candidates and the empty opposite-parity slots between
    # them both visible.
    lo, hi = max(1, ts - 12), min(roll["n"], ts + 12)
    win = (points >= lo) & (points <= hi)
    for ax, i in zip(axes.ravel(), picks):
        ax.bar(points[win], beliefs[i][win], width=0.85,
               color=np.where(same[win], "#3b6fb6", "#c9ccd1"))
        ax.axvline(ts, color="#d1495b", ls="--", lw=1.2)
        ax.set_xlim(lo - 0.6, hi + 0.6)
        ax.set_title(f"step {i}   P(true)={stats['truth_mass'][i]:.2f}   "
                     f"$2^H$={stats['eff_support'][i]:.1f}", fontsize=8)
        ax.set_ylim(0, max(0.08, float(beliefs[i].max()) * 1.2))
        ax.tick_params(labelsize=7)
        ax.set_xticks([t for t in range(lo, hi + 1) if t % 4 == 0])
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle(f"Odd-Even posterior, n={roll['n']}, true state {ts}, "
                 f"std_dev={roll['std_dev']:.2f}   "
                 f"(zoomed to [{lo},{hi}]; grey slots = opposite parity, "
                 f"exactly zero at every step)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default=None,
                    help="registered variant name, e.g. oe50_short. Supplies "
                         "n and the episode cap; --n/--steps override it.")
    ap.add_argument("--n", type=int, default=None, help="n_dist_size")
    ap.add_argument("--steps", type=int, default=None, help="steps to roll")
    ap.add_argument("--obs_per_step", type=int, default=1)
    ap.add_argument("--std_dev", type=float, default=None,
                    help="override the sqrt(n)-derived default")
    ap.add_argument("--episode_seed", type=int, default=0)
    ap.add_argument("--policy", default="posterior_mean",
                    choices=("posterior_mean", "posterior_mode", "random"))
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--also_gif", action="store_true")
    ap.add_argument("--no_summary_png", action="store_true")
    args = ap.parse_args(argv)

    n, steps = args.n, args.steps
    if args.variant is not None:
        # The Odd-Even registry lives in the package (set_transformer/rl/domains/odd_even.py).
        from set_transformer.rl.domains import odd_even as variants
        v = variants.resolve(args.variant)
        n = n if n is not None else v.n_dist_size
        steps = steps if steps is not None else variants.episode_cap(args.variant)
    n = 50 if n is None else n
    steps = 30 if steps is None else steps

    roll = rollout(n_dist_size=n, steps=steps, episode_seed=args.episode_seed,
                   policy=args.policy, obs_per_step=args.obs_per_step,
                   std_dev=args.std_dev)
    stats = belief_stats(roll["points"], roll["beliefs"], roll["true_state"])

    tag = f"n{n}_seed{args.episode_seed}_{args.policy}"
    out = Path(args.out) if args.out else _HERE / "out" / f"odd_even_belief_{tag}.mp4"
    animate(roll, stats, out, fps=args.fps)
    print(f"video   -> {out}")
    if args.also_gif:
        g = animate(roll, stats, out.with_suffix(".gif"), fps=args.fps)
        print(f"gif     -> {g}")
    if not args.no_summary_png:
        p = save_summary_png(roll, stats, out.with_name(out.stem + "_grid.png"))
        print(f"grid    -> {p}")

    # numbers behind the picture
    print(f"\ntrue state {roll['true_state']}  std_dev {roll['std_dev']:.3f}  "
          f"n={n}  steps={steps}  obs_per_step={args.obs_per_step}")
    print(f"{'step':>5} {'obs':>6} {'P(true)':>9} {'eff.supp':>9} "
          f"{'parity':>7} {'mean':>7} {'std':>6}")
    for i in range(len(roll["beliefs"])):
        o = np.atleast_1d(roll["observations"][i])
        print(f"{i:>5} {int(o[0]):>6} {stats['truth_mass'][i]:>9.4f} "
              f"{stats['eff_support'][i]:>9.2f} {stats['parity_mass'][i]:>7.3f} "
              f"{stats['mean'][i]:>7.2f} {stats['std'][i]:>6.2f}")
    return roll, stats


if __name__ == "__main__":
    main()
