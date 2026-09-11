"""Learning curves per arm from a directory of Ant-Tag PPO runs.

Reads, for every run under --runs_dir whose name matches --pattern:

  logs/evaluations.npz      SB3 EvalCallback: timesteps, per-episode returns and
                            lengths (an episode shorter than the cap = a tag)
  logs/PPO_*/events.*       TensorBoard scalars: rollout/ep_rew_mean (shaped
                            training return), st/feat_std_mean (encoder
                            collapse sentinel), ...
  run_config.json           total_timesteps + curriculum strings for the
                            phase markers

Runs are grouped into arms by directory name: ``<stamp>_seed<s>_<arm>`` -> arm.
Each panel draws one thin line per seed and the arm mean on a common step
grid. Also prints a sample-efficiency table: steps to the first eval at or
above --thresholds tag rate, best eval, and the mean of the last --last evals.

    python3 eval_scripts/plot_learning_curves.py \
        --runs_dir runs/ant_tag_st_smart_mid_slow_v15 --out curves.png
    python3 eval_scripts/plot_learning_curves.py --runs_dir runs/ant_tag_st_smart_hard \
        --pattern terminal_recipe_noent --exclude INCOMPLETE
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANT_TAG_DIR = Path(__file__).resolve().parents[1]
for p in (str(_REPO_ROOT), str(_ANT_TAG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

_RUN_RE = re.compile(r"^\d{8}_\d{6}_seed(?P<seed>\d+)_(?P<arm>.+)$")


def _tb_scalars(logs_dir: str, tags: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    out = {}
    for d in sorted(glob.glob(os.path.join(logs_dir, "*_[0-9]*"))):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        have = set(ea.Tags().get("scalars", []))
        for tag in tags:
            if tag in have:
                ev = ea.Scalars(tag)
                out[tag] = (np.array([e.step for e in ev], dtype=float),
                            np.array([e.value for e in ev], dtype=float))
    return out


def load_run(run_dir: str, cap: int) -> dict:
    ev = np.load(os.path.join(run_dir, "logs", "evaluations.npz"))
    lengths = ev["ep_lengths"]
    rec = dict(
        eval_steps=ev["timesteps"].astype(float),
        eval_tag_rate=(lengths < cap).mean(axis=1),
        eval_return=ev["results"].mean(axis=1),
        eval_len=lengths.mean(axis=1),
    )
    rec.update(_tb_scalars(os.path.join(run_dir, "logs"),
                           ["rollout/ep_rew_mean", "st/feat_std_mean", "train/approx_kl"]))
    cfg_path = os.path.join(run_dir, "run_config.json")
    rec["config"] = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    return rec


def _on_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.full_like(grid, np.nan, dtype=float)
    m = grid <= x.max()
    out[m] = np.interp(grid[m], x, y)
    return out


def _phase_marks(cfg: dict) -> list[tuple[float, str]]:
    total = float(cfg.get("total_timesteps", 0) or 0)
    marks = []
    if not total:
        return marks
    by_frac: dict[float, list[str]] = defaultdict(list)
    for key, label in (("curriculum", "vis"), ("evasion_curriculum", "evade"),
                       ("reward_schedule", "reward")):
        s = cfg.get(key) or ""
        for f in sorted({float(e.split(":")[0]) for e in s.split(",") if e.strip()}):
            if 0 < f < 1:
                by_frac[f].append(label)
    for f, labels in sorted(by_frac.items()):   # one line per waypoint, labels merged
        marks.append((f * total, f"{'/'.join(labels)} {f:.0%}"))
    return marks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--pattern", default="", help="substring the run dir name must contain")
    ap.add_argument("--exclude", default="INCOMPLETE", help="substring that excludes a run dir")
    ap.add_argument("--cap", type=int, default=None,
                    help="episode cap (default: from --variant via variants.episode_cap, else 400)")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--thresholds", default="0.1,0.25,0.5")
    ap.add_argument("--last", type=int, default=10, help="evals averaged for the 'final' column")
    ap.add_argument("--out", default=None, help="PNG path (default: <runs_dir>/learning_curves.png)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    cap = args.cap
    if cap is None and args.variant:
        import variants
        cap = variants.episode_cap(args.variant)
    cap = cap or 400

    arms: dict[str, dict[str, dict]] = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        name = os.path.basename(d)
        if not os.path.isdir(d) or args.pattern not in name or (args.exclude and args.exclude in name):
            continue
        m = _RUN_RE.match(name)
        if not m or not os.path.exists(os.path.join(d, "logs", "evaluations.npz")):
            continue
        arms[m["arm"]][m["seed"]] = load_run(d, cap)
    if not arms:
        sys.exit(f"no runs with evaluations under {args.runs_dir} matching {args.pattern!r}")

    any_cfg = next(iter(next(iter(arms.values())).values()))["config"]
    total = float(any_cfg.get("total_timesteps", 0) or 0)
    max_step = max(r["eval_steps"].max() for a in arms.values() for r in a.values())
    grid = np.linspace(0, max_step, 400)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("eval_tag_rate", "eval tag rate (true sparse reward, fraction of episodes ended before cap)", None),
        ("rollout/ep_rew_mean", "training shaped return per episode (rollout/ep_rew_mean)", None),
        ("eval_return", "eval mean episode return (eval env reward)", None),
        ("st/feat_std_mean", "encoder feature spread st/feat_std_mean (collapse sentinel)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    arm_names = sorted(arms)
    for ax, (key, title, _) in zip(axes.flat, panels):
        for ci, arm in enumerate(arm_names):
            curves = []
            for seed, rec in sorted(arms[arm].items()):
                if key in rec:
                    x, y = (rec["eval_steps"], rec[key]) if key.startswith("eval_") else rec[key]
                else:
                    continue
                if len(x) == 0:
                    continue
                ax.plot(x / 1e6, y, color=colors[ci % len(colors)], alpha=0.25, lw=0.8)
                curves.append(_on_grid(x, y, grid))
            if curves:
                mean = np.nanmean(np.vstack(curves), axis=0)
                ax.plot(grid / 1e6, mean, color=colors[ci % len(colors)], lw=2.2,
                        label=f"{arm} (n={len(curves)})")
        for xm, label in _phase_marks(any_cfg):
            ax.axvline(xm / 1e6, color="gray", ls=":", lw=0.8)
            ax.text(xm / 1e6, ax.get_ylim()[1], label, rotation=90, va="top", ha="right",
                    fontsize=7, color="gray")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(f"PPO steps (millions){'' if not total else f' of {total/1e6:.0f}M'}")
    axes[0][0].set_ylim(-0.02, 1.02)
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle(args.title or f"{os.path.basename(args.runs_dir.rstrip('/'))}: thin = seeds, thick = arm mean",
                 fontsize=12)
    fig.tight_layout()
    out = args.out or os.path.join(args.runs_dir, "learning_curves.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    # ---- sample-efficiency table ------------------------------------------
    thresholds = [float(t) for t in args.thresholds.split(",") if t]
    hdr = (["arm", "seeds", "evals"] + [f"steps to >= {t:.0%}" for t in thresholds]
           + ["best eval", f"mean of last {args.last}", "final feat_std"])
    rows = []
    for arm in arm_names:
        recs = list(arms[arm].values())
        n_ev = min(len(r["eval_steps"]) for r in recs)
        cells = [arm, str(len(recs)), str(n_ev)]
        for t in thresholds:
            hits = []
            for r in recs:
                idx = np.nonzero(r["eval_tag_rate"] >= t)[0]
                hits.append(r["eval_steps"][idx[0]] / 1e6 if idx.size else None)
            got = [h for h in hits if h is not None]
            cells.append("-" if not got else
                         f"{np.median(got):.2f}M ({len(got)}/{len(hits)} seeds)")
        cells.append(f"{np.mean([r['eval_tag_rate'].max() for r in recs]):.2f}")
        cells.append(f"{np.mean([r['eval_tag_rate'][-args.last:].mean() for r in recs]):.2f}")
        fs = [r["st/feat_std_mean"][1][-1] for r in recs if "st/feat_std_mean" in r]
        cells.append(f"{np.mean(fs):.2f}" if fs else "-")
        rows.append(cells)
    widths = [max(len(str(x)) for x in col) for col in zip(hdr, *rows)]
    for line in (hdr, *rows):
        print("  ".join(str(c).ljust(w) for c, w in zip(line, widths)))
    print(f"\n(steps to threshold = median over the seeds that reached it, at the FIRST eval "
          f"at or above the threshold; eval = 30 episodes every 40k by default; "
          f"cap {cap}; grid to {max_step/1e6:.2f}M)")


if __name__ == "__main__":
    main()
