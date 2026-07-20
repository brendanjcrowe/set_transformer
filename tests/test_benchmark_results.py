"""Tests for the cross-machine results harness (ingest + statistics).

Imports ``...benchmark.results`` directly rather than the package ``__init__`` so these
run without gym / SB3 / MuJoCo — aggregation must work on a plotting-only machine.
"""

import json

import numpy as np
import pytest

from set_transformer.rl.benchmark.results import (
    RunRecord,
    align_curves,
    bootstrap_mean_ci,
    curve_band,
    discover_runs,
    duplicate_seeds,
    encoder_cost_rows,
    group_runs,
    load_run,
    summarize,
)


def write_run(root, env, method, seed, timesteps, returns, ep_lengths=None, meta=None):
    """Create a fake run dir mimicking what train.py writes."""
    run_dir = root / env / method / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    returns = np.asarray(returns, dtype=float)
    if ep_lengths is None:
        ep_lengths = np.full_like(returns, 100.0)
    np.savez(
        run_dir / "evaluations.npz",
        timesteps=np.asarray(timesteps),
        results=returns,
        ep_lengths=np.asarray(ep_lengths, dtype=float),
    )
    if meta is not None:
        (run_dir / "meta.json").write_text(json.dumps(meta))
    return run_dir


# --- Ingest --------------------------------------------------------------------

def test_load_run_reads_curve_and_meta(tmp_path):
    meta = {"env": "ant_tag", "method": "cgf", "seed": 2, "algo": "PPO",
            "extractor_params_total": 1234, "wall_clock_sec": 60.0}
    run_dir = write_run(tmp_path, "ant_tag", "cgf", 2, [100, 200], [[1.0, 3.0], [5.0, 7.0]], meta=meta)
    rec = load_run(run_dir)
    assert (rec.env, rec.method, rec.seed) == ("ant_tag", "cgf", 2)
    assert rec.has_curve
    np.testing.assert_allclose(rec.return_curve, [2.0, 6.0])
    assert rec.meta["algo"] == "PPO"


def test_load_run_without_meta_infers_identity_from_path(tmp_path):
    """A bare evaluations.npz scp'd from another machine must still ingest."""
    run_dir = write_run(tmp_path, "odd_even", "gaussian", 7, [10, 20], [[1.0], [2.0]])
    rec = load_run(run_dir)
    assert (rec.env, rec.method, rec.seed) == ("odd_even", "gaussian", 7)
    assert rec.meta == {}


def test_discover_merges_multiple_roots_and_records_source(tmp_path):
    """Roots from different machines union on the shared layout."""
    a, b = tmp_path / "machineA", tmp_path / "machineB"
    write_run(a, "ant_tag", "cgf", 0, [100], [[1.0]])
    write_run(a, "ant_tag", "gaussian", 0, [100], [[2.0]])
    write_run(b, "ant_tag", "cgf", 1, [100], [[3.0]])  # same cell, different machine

    runs = discover_runs([f"A={a}", f"B={b}"])
    assert len(runs) == 3
    cells = group_runs(runs)
    assert sorted(r.seed for r in cells[("ant_tag", "cgf")]) == [0, 1]
    assert {r.source for r in runs} == {"A", "B"}


def test_discover_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_runs([tmp_path / "nope"])


def test_duplicate_seeds_detected(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, "ant_tag", "cgf", 0, [100], [[1.0]])
    write_run(b, "ant_tag", "cgf", 0, [100], [[9.0]])  # same seed copied twice
    dups = duplicate_seeds(discover_runs([a, b]))
    assert ("ant_tag", "cgf", 0) in dups


# --- Per-run metrics -----------------------------------------------------------

def test_run_metrics_final_best_and_auc():
    rec = RunRecord(env="e", method="m", seed=0, path=".",
                    timesteps=np.array([0, 100]),
                    returns=np.array([[0.0, 0.0], [10.0, 10.0]]),
                    ep_lengths=np.array([[50.0, 50.0], [30.0, 30.0]]))
    m = rec.metrics()
    assert m["final_return"] == 10.0
    assert m["best_return"] == 10.0
    assert m["auc_return"] == pytest.approx(5.0)  # trapezoid of 0->10 over the budget
    assert m["final_ep_length"] == 30.0


def test_success_rate_recomputed_from_curve_beats_meta():
    """success_fn is applied per eval episode; meta is only the fallback."""
    rec = RunRecord(env="ant_tag", method="m", seed=0, path=".",
                    meta={"success_rate": 0.0},
                    timesteps=np.array([0, 100]),
                    returns=np.zeros((2, 4)),
                    ep_lengths=np.array([[400, 400, 400, 400], [399, 399, 400, 400]]))
    success_fn = lambda ret, length: length < 400
    assert rec.metrics(success_fn)["success_rate"] == pytest.approx(0.5)
    assert rec.metrics(None)["success_rate"] == 0.0  # falls back to meta


def test_steps_to_threshold():
    rec = RunRecord(env="e", method="m", seed=0, path=".",
                    timesteps=np.array([10, 20, 30]),
                    returns=np.array([[0.0], [5.0], [9.0]]),
                    ep_lengths=np.ones((3, 1)))
    assert rec.steps_to_threshold(5.0) == 20
    assert rec.steps_to_threshold(100.0) is None


def test_metrics_empty_when_no_curve():
    assert RunRecord(env="e", method="m", seed=0, path=".").metrics() == {}


# --- Statistics ----------------------------------------------------------------

def test_bootstrap_mean_ci_brackets_mean_and_handles_edges():
    x = np.random.default_rng(0).normal(5.0, 1.0, 40)
    mean, lo, hi = bootstrap_mean_ci(x)
    assert mean == pytest.approx(x.mean())
    assert lo < mean < hi
    assert bootstrap_mean_ci([3.0]) == (3.0, 3.0, 3.0)  # single seed: degenerate CI
    assert np.isnan(bootstrap_mean_ci([])[0])


def test_align_curves_interpolates_onto_shared_grid():
    """Seeds evaluated at different timesteps (different eval_freq / n_envs) still align."""
    runs = [
        RunRecord(env="e", method="m", seed=0, path=".",
                  timesteps=np.array([0, 50, 100]),
                  returns=np.array([[0.0], [5.0], [10.0]]), ep_lengths=np.ones((3, 1))),
        RunRecord(env="e", method="m", seed=1, path=".",
                  timesteps=np.array([0, 100]),
                  returns=np.array([[0.0], [20.0]]), ep_lengths=np.ones((2, 1))),
    ]
    grid, values = align_curves(runs, n_points=11)
    assert values.shape == (2, 11)
    assert grid[0] == 0 and grid[-1] == 100
    np.testing.assert_allclose(values[0], np.linspace(0, 10, 11))
    np.testing.assert_allclose(values[1], np.linspace(0, 20, 11))


def test_align_curves_returns_none_without_overlap():
    runs = [
        RunRecord(env="e", method="m", seed=0, path=".",
                  timesteps=np.array([0, 10]),
                  returns=np.array([[0.0], [1.0]]), ep_lengths=np.ones((2, 1))),
        RunRecord(env="e", method="m", seed=1, path=".",
                  timesteps=np.array([100, 200]),
                  returns=np.array([[0.0], [1.0]]), ep_lengths=np.ones((2, 1))),
    ]
    assert align_curves(runs)[0] is None
    assert align_curves([])[0] is None


def test_curve_band_shapes_and_single_seed():
    values = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]])
    mean, lo, hi = curve_band(values)
    np.testing.assert_allclose(mean, [1.0, 2.0, 3.0])
    assert lo.shape == hi.shape == mean.shape
    assert np.all(lo <= mean + 1e-9) and np.all(hi >= mean - 1e-9)
    m1, l1, h1 = curve_band(values[:1])  # one seed => no band
    np.testing.assert_allclose(l1, m1)
    np.testing.assert_allclose(h1, m1)


def test_summarize_produces_tidy_rows(tmp_path):
    for seed, val in enumerate([1.0, 2.0, 3.0]):
        write_run(tmp_path, "ant_tag", "cgf", seed, [0, 100], [[0.0], [val]])
    rows = summarize(discover_runs([tmp_path]))
    final = next(r for r in rows if r["metric"] == "final_return")
    assert final["env"] == "ant_tag" and final["method"] == "cgf"
    assert final["mean"] == pytest.approx(2.0)
    assert final["n_seeds"] == 3
    assert final["ci_lo"] <= final["mean"] <= final["ci_hi"]


def test_encoder_cost_rows_from_meta(tmp_path):
    meta = {"env": "ant_tag", "method": "gaussian", "seed": 0,
            "extractor_params_total": 100, "extractor_params_trainable": 100,
            "particle_stat_dim": 5, "features_dim": 128,
            "wall_clock_sec": 10.0, "total_timesteps": 1000}
    write_run(tmp_path, "ant_tag", "gaussian", 0, [100], [[1.0]], meta=meta)
    rows = encoder_cost_rows(discover_runs([tmp_path]))
    assert rows[0]["params_total"] == 100
    assert rows[0]["wall_clock_sec_mean"] == pytest.approx(10.0)
