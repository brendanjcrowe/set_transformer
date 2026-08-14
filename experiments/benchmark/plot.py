"""Paper figures for the PF-belief encoder benchmark.

Reads the same run records as ``aggregate.py`` (several roots merge; ``label=path``
records provenance) and writes, into ``--fig_dir``:

    curves_<env>.png     learning curve per env: eval TRUE return vs timesteps,
                         one line per method, mean over seeds + 95% bootstrap CI band
    curves_all.png       all envs side by side
    summary_<metric>.png grouped bars (x = env, one bar per method) with CI whiskers

Method colors are fixed in ``METHOD_COLORS`` so a method is the same color in every
figure of the paper.

Examples
--------
    python experiments/benchmark/plot.py results
    python experiments/benchmark/plot.py results --metrics final_return success_rate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `set_transformer` is a namespace package and the editable install may point at a
# different checkout; ensure THIS repo's package root is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")  # figures are written to disk; never needs a display
import matplotlib.pyplot as plt
import numpy as np

from set_transformer.rl.benchmark.results import (
    SUMMARY_METRICS,
    align_curves,
    curve_band,
    discover_runs,
    get_success_fns,
    group_runs,
    summarize,
)

#: Fixed method -> color so figures are consistent across the paper. Learned encoders
#: (ours) get the warm colors; statistical baselines the cool ones.
METHOD_COLORS = {
    "st_frozen": "tab:red",
    "st_finetune": "tab:orange",
    "st_scratch": "tab:brown",
    "cgf": "tab:purple",
    # Learned fixed-pool baselines (mean / max) — green family, between stats and ST.
    "deepset": "tab:green",
    "pointnet": "tab:olive",
    "gaussian": "tab:blue",
    "kmoments": "tab:cyan",
}
#: Preferred left-to-right / legend order; unknown methods are appended alphabetically.
#: analytic stats -> CGF -> learned fixed-pool (deepset/pointnet) -> Set Transformers.
METHOD_ORDER = [
    "gaussian", "kmoments", "cgf", "deepset", "pointnet",
    "st_scratch", "st_finetune", "st_frozen",
]


def order_methods(methods) -> list[str]:
    known = [m for m in METHOD_ORDER if m in methods]
    return known + sorted(m for m in methods if m not in METHOD_ORDER)


#: Extractor hyperparameters worth surfacing in the legend (meta key -> display name).
#: e.g. k-moments' order and CGF's number of evaluation points, so a reader can see the
#: baseline's capacity at a glance rather than digging through configs.
_LEGEND_HYPERPARAMS = {"k": "k", "num_t": "num_t"}


def build_method_labels(runs) -> dict[str, str]:
    """method -> display label, annotated with salient hyperparameters from meta.json.

    Reads ``extractor_kwargs`` recorded per run, so e.g. ``kmoments`` renders as
    ``kmoments (k=4)`` and ``cgf`` as ``cgf (num_t=16)``.
    """
    by_method: dict[str, list] = {}
    for r in runs:
        by_method.setdefault(r.method, []).append(r)
    labels = {}
    for method, method_runs in by_method.items():
        meta = next((r.meta for r in method_runs if r.meta), {})
        ek = meta.get("extractor_kwargs") or {}
        tags = [f"{disp}={ek[key]}" for key, disp in _LEGEND_HYPERPARAMS.items() if key in ek]
        labels[method] = f"{method} ({', '.join(tags)})" if tags else method
    return labels


def color_for(method: str) -> str:
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    # Deterministic fallback so an unregistered method still gets a stable color.
    return f"C{sum(map(ord, method)) % 10}"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {path}")


def plot_curves_axis(ax, env: str, cells, methods, n_points: int, n_bootstrap: int,
                     method_labels: dict[str, str], show_title: bool = True) -> bool:
    """Draw one env's learning curves onto ``ax``. Returns False if nothing plottable."""
    drew = False
    for method in methods:
        cell = cells.get((env, method))
        if not cell:
            continue
        grid, values = align_curves(cell, n_points=n_points)
        if grid is None:
            print(f"[plot] {env}/{method}: no overlapping curve range, skipped")
            continue
        mean, lo, hi = curve_band(values, n_bootstrap=n_bootstrap)
        color = color_for(method)
        label = f"{method_labels.get(method, method)}, n={values.shape[0]}"
        ax.plot(grid, mean, label=label, color=color, lw=1.8)
        if values.shape[0] > 1:
            ax.fill_between(grid, lo, hi, color=color, alpha=0.18, linewidth=0)
        drew = True
    if show_title:  # per-axis title only for the multi-env grid; single-env uses suptitle
        ax.set_title(env, fontsize=12)
    ax.set_xlabel("environment steps", fontsize=10)
    ax.grid(linestyle=":", alpha=0.4)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    return drew


def plot_curves(runs, fig_dir: Path, n_points: int, n_bootstrap: int,
                method_labels: dict[str, str]) -> None:
    cells = group_runs(runs)
    envs = sorted({e for e, _ in cells})
    methods = order_methods({m for _, m in cells})

    for env in envs:
        fig, ax = plt.subplots(figsize=(6, 4))
        if not plot_curves_axis(ax, env, cells, methods, n_points, n_bootstrap,
                                method_labels, show_title=False):
            plt.close(fig)
            continue
        ax.set_ylabel("eval return (true, unshaped)", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        fig.suptitle(f"{env} — learning curves (mean ± 95% CI over seeds)", fontsize=12)
        _save(fig, fig_dir / f"curves_{env}.png")

    if len(envs) > 1:
        fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 4))
        axes = np.atleast_1d(axes)
        for ax, env in zip(axes, envs):
            plot_curves_axis(ax, env, cells, methods, n_points, n_bootstrap, method_labels)
        axes[0].set_ylabel("eval return (true, unshaped)", fontsize=10)
        axes[-1].legend(fontsize=8, frameon=False)
        fig.suptitle("Learning curves (mean ± 95% CI over seeds)", fontsize=13)
        _save(fig, fig_dir / "curves_all.png")


def plot_summary_bars(summary_rows, fig_dir: Path, metrics: list[str],
                      method_labels: dict[str, str]) -> None:
    """Grouped bars: x = env, one bar per method, CI whiskers."""
    for metric in metrics:
        rows = [r for r in summary_rows if r["metric"] == metric]
        if not rows:
            print(f"[plot] no data for metric '{metric}', skipped")
            continue
        envs = sorted({r["env"] for r in rows})
        methods = order_methods({r["method"] for r in rows})
        cells = {(r["env"], r["method"]): r for r in rows}

        fig, ax = plt.subplots(figsize=(max(5.0, 0.7 * len(envs) * len(methods)), 4))
        x = np.arange(len(envs))
        width = 0.8 / max(len(methods), 1)
        data_hi, data_lo = [], []  # track extent so the legend can clear the bars
        for i, method in enumerate(methods):
            means, lo, hi = [], [], []
            for env in envs:
                c = cells.get((env, method))
                means.append(c["mean"] if c else np.nan)
                # Error bars are offsets from the mean, and CIs can be asymmetric.
                lo.append(c["mean"] - c["ci_lo"] if c else 0.0)
                hi.append(c["ci_hi"] - c["mean"] if c else 0.0)
                if c:
                    data_hi.append(c["ci_hi"])
                    data_lo.append(c["ci_lo"])
            offset = (i - (len(methods) - 1) / 2) * width
            ax.bar(x + offset, means, width=width, yerr=[lo, hi], capsize=2,
                   label=method_labels.get(method, method), color=color_for(method))
        direction = "higher is better" if SUMMARY_METRICS.get(metric, True) else "lower is better"
        ax.set_xticks(x)
        ax.set_xticklabels(envs, fontsize=10)
        ax.set_ylabel(f"{metric} ({direction})", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        # Pin the legend to the top-right and add headroom so it never overlaps the bars.
        top, bot = max(data_hi, default=1.0), min(data_lo + [0.0])
        ax.set_ylim(bot, top + 0.28 * (top - bot))
        ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
        fig.suptitle(f"{metric} by method (95% bootstrap CI over seeds)", fontsize=12)
        _save(fig, fig_dir / f"summary_{metric}.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="results roots; 'label=path' records provenance")
    p.add_argument("--fig_dir", default="results/figures")
    p.add_argument("--metrics", nargs="+", default=["final_return", "success_rate"],
                   help=f"summary-bar metrics; available: {list(SUMMARY_METRICS)}")
    p.add_argument("--n_points", type=int, default=100, help="curve resampling resolution")
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = p.parse_args()

    runs = discover_runs(args.roots)
    if not runs:
        raise SystemExit(f"No runs found under {args.roots}")
    print(f"[plot] {len(runs)} runs across {len(group_runs(runs))} (env, method) cells")

    method_labels = build_method_labels(runs)
    fig_dir = Path(args.fig_dir)
    plot_curves(runs, fig_dir, args.n_points, args.n_bootstrap, method_labels)
    summary_rows = summarize(runs, get_success_fns(), n_bootstrap=args.n_bootstrap, seed=args.seed)
    plot_summary_bars(summary_rows, fig_dir, args.metrics, method_labels)


if __name__ == "__main__":
    main()
