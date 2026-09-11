"""Aggregate benchmark runs from one or more results roots into tables.

Roots may come from different machines — pass each one (optionally ``label=path`` to
record provenance) and they merge on the shared ``<env>/<method>/seed<N>`` layout.

Outputs (into ``--out_dir``):
    runs.csv            one row per (env, method, seed): all scalar metrics + provenance
    curves.csv          long-form learning curves (env, method, seed, timestep, return)
    summary.csv         env x method x metric, mean with 95% bootstrap CI over seeds
    summary.md          the same, as a markdown method x env table per metric
    encoder_cost.csv    extractor param counts / feature dims / wall-clock

Examples
--------
    python experiments/benchmark/aggregate.py results
    python experiments/benchmark/aggregate.py workstation=results cluster=~/from_cluster \
        --out_dir results/aggregated
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# `set_transformer` is a namespace package and the editable install may point at a
# different checkout; ensure THIS repo's package root is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from set_transformer.rl.benchmark.results import (
    SUMMARY_METRICS,
    discover_runs,
    duplicate_seeds,
    encoder_cost_rows,
    get_success_fns,
    group_runs,
    summarize,
)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        print(f"[aggregate] no rows for {path.name}, skipped")
        return
    fieldnames = fieldnames or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[aggregate] wrote {path} ({len(rows)} rows)")


def runs_table(runs, success_fns, thresholds: dict[str, float]) -> list[dict]:
    rows = []
    for r in sorted(runs, key=lambda r: (r.env, r.method, r.seed)):
        row = {
            "env": r.env, "method": r.method, "seed": r.seed,
            "source": r.source, "path": str(r.path),
            "algo": r.meta.get("algo"), "shaped": r.meta.get("shaped"),
            "total_timesteps": r.meta.get("total_timesteps"),
            "git_commit": (r.meta.get("git_commit") or "")[:8],
            "n_evals": int(r.timesteps.size),
        }
        row.update(r.metrics(success_fns.get(r.env)))
        if r.env in thresholds:
            row["steps_to_threshold"] = r.steps_to_threshold(thresholds[r.env])
        rows.append(row)
    return rows


def curves_table(runs) -> list[dict]:
    rows = []
    for r in sorted(runs, key=lambda r: (r.env, r.method, r.seed)):
        if not r.has_curve:
            continue
        lengths = r.ep_lengths.mean(axis=1)
        for t, ret, ln in zip(r.timesteps, r.return_curve, lengths):
            rows.append({
                "env": r.env, "method": r.method, "seed": r.seed,
                "timestep": int(t), "return": float(ret), "ep_length": float(ln),
            })
    return rows


def summary_markdown(summary_rows: list[dict]) -> str:
    """One markdown method x env table per metric, cells as ``mean [lo, hi]`` (n)."""
    lines: list[str] = ["# Benchmark summary", ""]
    by_metric: dict[str, list[dict]] = {}
    for row in summary_rows:
        by_metric.setdefault(row["metric"], []).append(row)

    for metric, rows in by_metric.items():
        arrow = "higher is better" if SUMMARY_METRICS.get(metric, True) else "lower is better"
        envs = sorted({r["env"] for r in rows})
        methods = sorted({r["method"] for r in rows})
        cells = {(r["env"], r["method"]): r for r in rows}
        lines += [f"## {metric} ({arrow})", "",
                  "| method | " + " | ".join(envs) + " |",
                  "|---|" + "---|" * len(envs)]
        for method in methods:
            parts = []
            for env in envs:
                c = cells.get((env, method))
                parts.append(
                    f"{c['mean']:.3g} [{c['ci_lo']:.3g}, {c['ci_hi']:.3g}] (n={c['n_seeds']})"
                    if c else "—"
                )
            lines.append(f"| {method} | " + " | ".join(parts) + " |")
        lines.append("")
    lines += ["", "Cells are mean over seeds with a 95% bootstrap CI.", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("roots", nargs="+", help="results roots; 'label=path' records provenance")
    p.add_argument("--out_dir", default="results/aggregated")
    p.add_argument("--threshold", action="append", default=[], metavar="ENV=VALUE",
                   help="return threshold for steps-to-threshold, repeatable")
    p.add_argument("--n_bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    args = p.parse_args()

    thresholds = {}
    for item in args.threshold:
        env, _, val = item.partition("=")
        thresholds[env] = float(val)

    runs = discover_runs(args.roots)
    if not runs:
        raise SystemExit(f"No runs found under {args.roots}")

    cells = group_runs(runs)
    print(f"[aggregate] {len(runs)} runs across {len(cells)} (env, method) cells")
    for (env, method), cell in sorted(cells.items()):
        seeds = ",".join(str(r.seed) for r in cell)
        print(f"    {env:12s} {method:12s} seeds=[{seeds}] n={len(cell)}")

    dups = duplicate_seeds(runs)
    if dups:
        print("[aggregate] WARNING: same seed found under multiple sources "
              "(both are kept and will be averaged):")
        for (env, method, seed), sources in dups.items():
            print(f"    {env}/{method}/seed{seed}: {sources}")

    success_fns = get_success_fns()
    if not success_fns:
        print("[aggregate] note: registry unavailable (no [rl] deps?); "
              "success_rate falls back to meta.json")

    out = Path(args.out_dir)
    write_csv(out / "runs.csv", runs_table(runs, success_fns, thresholds))
    write_csv(out / "curves.csv", curves_table(runs))

    summary_rows = summarize(runs, success_fns, n_bootstrap=args.n_bootstrap, seed=args.seed)
    write_csv(out / "summary.csv", summary_rows,
              fieldnames=["env", "method", "metric", "mean", "ci_lo", "ci_hi", "n_seeds", "seeds"])
    md = out / "summary.md"
    md.write_text(summary_markdown(summary_rows))
    print(f"[aggregate] wrote {md}")
    write_csv(out / "encoder_cost.csv", encoder_cost_rows(runs))


if __name__ == "__main__":
    main()
