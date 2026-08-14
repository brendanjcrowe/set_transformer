"""Cross-machine ingest + statistics for benchmark run records.

A *run* is one ``(env, method, seed)`` directory written by
``experiments/benchmark/train.py``:

    <root>/<env>/<method>/seed<seed>/
        meta.json         # config + final metrics + encoder cost (optional)
        evaluations.npz   # SB3 EvalCallback: timesteps, results, ep_lengths

Results copied from several machines are merged by passing several roots — the layout
is self-describing, so directories union naturally. Runs missing ``meta.json`` are still
ingested by parsing ``<env>/<method>/seed<seed>`` out of the path, so a bare
``evaluations.npz`` scp'd from another box drops straight in.

Everything downstream (:mod:`experiments/benchmark/aggregate.py`,
``plot.py``) is built on :func:`discover_runs`, so figures and tables are reproducible
offline without re-running any RL.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

SEED_RE = re.compile(r"^seed[_-]?(\d+)$", re.IGNORECASE)

#: Metrics summarised across seeds, and whether larger is better.
SUMMARY_METRICS: dict[str, bool] = {
    "final_return": True,
    "best_return": True,
    "auc_return": True,
    "final_ep_length": False,
    "success_rate": True,
}


@dataclass
class RunRecord:
    """One ``(env, method, seed)`` run, with its full evaluation curve."""

    env: str
    method: str
    seed: int
    path: Path
    source: str = ""  # which results root it came from (i.e. which machine)
    meta: dict = field(default_factory=dict)
    timesteps: np.ndarray = field(default_factory=lambda: np.empty(0))
    returns: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # (n_evals, n_episodes)
    ep_lengths: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def has_curve(self) -> bool:
        return self.timesteps.size > 0 and self.returns.size > 0

    @property
    def return_curve(self) -> np.ndarray:
        """Mean true return per evaluation point."""
        return self.returns.mean(axis=1)

    def success_curve(self, success_fn) -> Optional[np.ndarray]:
        """Fraction of eval episodes counted as successful, per evaluation point."""
        if success_fn is None or not self.has_curve:
            return None
        return np.array([
            np.mean([bool(success_fn(float(r), int(l))) for r, l in zip(row_r, row_l)])
            for row_r, row_l in zip(self.returns, self.ep_lengths)
        ])

    def metrics(self, success_fn=None) -> dict[str, float]:
        """Scalar summary of this run. Sample efficiency comes from the whole curve."""
        out: dict[str, float] = {}
        if self.has_curve:
            curve = self.return_curve
            out["final_return"] = float(curve[-1])
            out["best_return"] = float(curve.max())
            out["auc_return"] = float(_normalized_auc(self.timesteps, curve))
            out["final_ep_length"] = float(self.ep_lengths[-1].mean())
        # Prefer a recomputed success rate (needs the env's success_fn); fall back to
        # whatever the trainer recorded, so runs from machines without the env still count.
        sc = self.success_curve(success_fn)
        if sc is not None:
            out["success_rate"] = float(sc[-1])
        elif "success_rate" in self.meta:
            out["success_rate"] = float(self.meta["success_rate"])
        return out

    def steps_to_threshold(self, threshold: float) -> Optional[int]:
        """First eval timestep whose mean return reaches ``threshold`` (None if never)."""
        if not self.has_curve:
            return None
        hit = np.nonzero(self.return_curve >= threshold)[0]
        return int(self.timesteps[hit[0]]) if hit.size else None


def _normalized_auc(timesteps: np.ndarray, curve: np.ndarray) -> float:
    """Area under the learning curve, divided by the training budget.

    Units match the return, so it reads as "average return over training" and is
    comparable across runs of different length. Rewards learning *fast*, not just far.
    """
    if timesteps.size < 2:
        return float(curve[-1]) if curve.size else float("nan")
    span = float(timesteps[-1] - timesteps[0])
    if span <= 0:
        return float(curve.mean())
    return float(np.trapz(curve, timesteps) / span)


# --- Ingest --------------------------------------------------------------------

def _parse_run_dir(run_dir: Path) -> Optional[tuple[str, str, int]]:
    """Recover (env, method, seed) from a ``<env>/<method>/seed<N>`` path tail."""
    m = SEED_RE.match(run_dir.name)
    if m is None or len(run_dir.parts) < 3:
        return None
    return run_dir.parent.parent.name, run_dir.parent.name, int(m.group(1))


def load_run(run_dir: Path, source: str = "") -> Optional[RunRecord]:
    """Load one run directory. ``meta.json`` wins over path-derived identity."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    parsed = _parse_run_dir(run_dir)
    env = meta.get("env") or (parsed[0] if parsed else None)
    method = meta.get("method") or (parsed[1] if parsed else None)
    seed = meta.get("seed", parsed[2] if parsed else None)
    if env is None or method is None or seed is None:
        return None

    rec = RunRecord(env=env, method=method, seed=int(seed), path=run_dir,
                    source=source or run_dir.parts[0], meta=meta)

    npz_path = run_dir / "evaluations.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        rec.timesteps = np.asarray(data["timesteps"], dtype=np.int64)
        rec.returns = np.atleast_2d(np.asarray(data["results"], dtype=np.float64))
        lengths = data["ep_lengths"] if "ep_lengths" in data else np.zeros_like(rec.returns)
        rec.ep_lengths = np.atleast_2d(np.asarray(lengths, dtype=np.float64))
    return rec


def discover_runs(roots: Iterable[Path | str]) -> list[RunRecord]:
    """Find every run under one or more results roots (e.g. one per machine).

    Roots may be given as ``path`` or ``label=path``; the label is recorded as the run's
    ``source`` so provenance survives merging.
    """
    runs: list[RunRecord] = []
    seen: set[tuple[str, str, int, str]] = set()
    for entry in roots:
        label, _, raw = str(entry).partition("=")
        root = Path(raw or label)
        source = label if raw else root.name
        if not root.exists():
            raise FileNotFoundError(f"results root does not exist: {root}")
        # A run dir is any dir holding meta.json or evaluations.npz.
        candidates = {p.parent for p in root.rglob("meta.json")}
        candidates |= {p.parent for p in root.rglob("evaluations.npz")}
        for run_dir in sorted(candidates):
            rec = load_run(run_dir, source=source)
            if rec is None:
                continue
            key = (rec.env, rec.method, rec.seed, source)
            if key in seen:  # same run seen twice under one root
                continue
            seen.add(key)
            runs.append(rec)
    return runs


def group_runs(runs: Iterable[RunRecord]) -> dict[tuple[str, str], list[RunRecord]]:
    """Group runs into ``(env, method) -> [run, ...]`` cells, seed-ordered."""
    cells: dict[tuple[str, str], list[RunRecord]] = {}
    for r in runs:
        cells.setdefault((r.env, r.method), []).append(r)
    for v in cells.values():
        v.sort(key=lambda r: r.seed)
    return cells


def duplicate_seeds(runs: Iterable[RunRecord]) -> dict[tuple[str, str, int], list[str]]:
    """Seeds appearing under more than one source — usually a copy mistake, so surface it."""
    seen: dict[tuple[str, str, int], list[str]] = {}
    for r in runs:
        seen.setdefault((r.env, r.method, r.seed), []).append(r.source)
    return {k: v for k, v in seen.items() if len(v) > 1}


# --- Statistics ----------------------------------------------------------------

def bootstrap_mean_ci(x, n_bootstrap: int = 2000, alpha: float = 0.05, rng=None):
    """Mean with a bootstrap CI over seeds. Returns ``(mean, ci_lo, ci_hi)``.

    With few seeds a bootstrap CI is narrow-biased, but it is assumption-free and
    consistent with the synthetic-sets figures; n is always reported alongside.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    if x.size == 1:
        return float(x[0]), float(x[0]), float(x[0])
    if rng is None:
        rng = np.random.default_rng(0)
    means = x[rng.integers(0, x.size, (n_bootstrap, x.size))].mean(axis=1)
    return (float(x.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def align_curves(runs: list[RunRecord], n_points: int = 100, metric: str = "return"):
    """Resample each seed's curve onto a shared timestep grid.

    Seeds (and machines) evaluate at different timesteps — different ``--eval_freq`` or
    ``--n_envs`` — so curves are interpolated onto a common grid spanning the timesteps
    all seeds actually cover. Returns ``(grid, values)`` with ``values`` shaped
    ``(n_seeds, n_points)``, or ``(None, None)`` if no run has a curve.
    """
    usable = [r for r in runs if r.has_curve and r.timesteps.size >= 2]
    if not usable:
        return None, None
    lo = max(int(r.timesteps[0]) for r in usable)
    hi = min(int(r.timesteps[-1]) for r in usable)
    if hi <= lo:  # no overlap (e.g. one run stopped very early)
        return None, None
    grid = np.linspace(lo, hi, n_points)
    rows = []
    for r in usable:
        curve = r.return_curve if metric == "return" else r.ep_lengths.mean(axis=1)
        rows.append(np.interp(grid, r.timesteps, curve))
    return grid, np.vstack(rows)


def curve_band(values: np.ndarray, n_bootstrap: int = 2000, alpha: float = 0.05, rng=None):
    """Per-timestep mean and bootstrap CI band across seeds."""
    if rng is None:
        rng = np.random.default_rng(0)
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        return mean, mean.copy(), mean.copy()
    idx = rng.integers(0, values.shape[0], (n_bootstrap, values.shape[0]))
    boot = values[idx].mean(axis=1)  # (n_bootstrap, n_points)
    return mean, np.quantile(boot, alpha / 2, axis=0), np.quantile(boot, 1 - alpha / 2, axis=0)


def summarize(runs: Iterable[RunRecord], success_fns: Optional[dict] = None,
              n_bootstrap: int = 2000, seed: int = 0) -> list[dict]:
    """Aggregate runs into tidy ``(env, method, metric, mean, ci_lo, ci_hi, n_seeds)`` rows."""
    success_fns = success_fns or {}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for (env, method), cell in sorted(group_runs(runs).items()):
        per_run = [r.metrics(success_fns.get(env)) for r in cell]
        for metric in SUMMARY_METRICS:
            vals = [m[metric] for m in per_run if metric in m and np.isfinite(m[metric])]
            if not vals:
                continue
            mean, lo, hi = bootstrap_mean_ci(vals, n_bootstrap=n_bootstrap, rng=rng)
            rows.append({
                "env": env, "method": method, "metric": metric,
                "mean": mean, "ci_lo": lo, "ci_hi": hi,
                "n_seeds": len(vals),
                "seeds": ",".join(str(r.seed) for r in cell),
            })
    return rows


def encoder_cost_rows(runs: Iterable[RunRecord]) -> list[dict]:
    """One row per (env, method): feature-extractor size, from ``meta.json``.

    Supports the "cheap statistic vs. learned encoder" cost comparison in the paper.
    """
    rows = []
    for (env, method), cell in sorted(group_runs(runs).items()):
        metas = [r.meta for r in cell if r.meta]
        if not metas:
            continue
        m = metas[0]
        rows.append({
            "env": env,
            "method": method,
            "params_total": m.get("extractor_params_total"),
            "params_trainable": m.get("extractor_params_trainable"),
            "particle_stat_dim": m.get("particle_stat_dim"),
            "features_dim": m.get("features_dim"),
            "wall_clock_sec_mean": float(np.mean([
                r.meta["wall_clock_sec"] for r in cell if "wall_clock_sec" in r.meta
            ])) if any("wall_clock_sec" in r.meta for r in cell) else None,
            "total_timesteps": m.get("total_timesteps"),
        })
    return rows


def get_success_fns() -> dict:
    """Per-env ``success_fn`` from the registry, if the ``[rl]`` deps are importable.

    Aggregation must work on a plotting-only machine with no MuJoCo, so this degrades
    to ``{}`` and metrics fall back to the ``success_rate`` recorded in ``meta.json``.
    """
    try:
        from set_transformer.rl.benchmark.registry import ENV_REGISTRY
    except Exception:
        return {}
    return {name: spec.success_fn for name, spec in ENV_REGISTRY.items()}
