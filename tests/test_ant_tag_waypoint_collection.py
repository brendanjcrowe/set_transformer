"""The Ant-Tag collector's ``waypoint`` episode type (2026-09-14): the locomotion policy walks to sampled
points while blind at the env's real visibility radius, so the file holds the sweeping beliefs a searching
policy lives in (v15 record 2026-09-08). Pins: the sampler's bounds and clearances; the walk re-draws on
arrival and on patience; a tiny waypoint-only collection moves the ant far more than the random type and
records its counts; the default mix (fraction 0) is the historical three-type collector."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains")
pytest.importorskip("mujoco")

from set_transformer.rl import collect, run_records  # noqa: E402
from set_transformer.rl.domains import ant_tag  # noqa: E402

LOCOMOTION = _ST_ROOT / "experiments" / "ant_tag" / "models" / "ant_locomotion_policy.zip"
ANT_TAG = ant_tag.ANT_TAG


def test_sample_waypoint_respects_bounds_and_clearances():
    rng = np.random.default_rng(0)
    ant, target = np.array([0.0, 0.0]), np.array([3.0, 3.0])
    for _ in range(200):
        w = ant_tag.sample_waypoint(rng, ant, target, half_width=4.5, min_dist=2.25, target_clearance=1.5)
        assert np.all(np.abs(w) <= 4.5)
        assert np.linalg.norm(w - ant) >= 2.25 and np.linalg.norm(w - target) >= 1.5
    # impossible constraints fall back to the farthest draw that is clear of the target (or the farthest)
    w = ant_tag.sample_waypoint(rng, ant, target, half_width=1.0, min_dist=100.0, target_clearance=0.0, tries=10)
    assert np.all(np.abs(w) <= 1.0)
    w = ant_tag.sample_waypoint(rng, ant, ant, half_width=1.0, min_dist=0.0, target_clearance=100.0, tries=10)
    assert np.all(np.abs(w) <= 1.0)


def test_den_biased_waypoints_land_next_to_a_candidate_den():
    rng = np.random.default_rng(3)
    d = np.array([1.0, 1.0]) / np.sqrt(2)
    cand = np.stack([-2.4 * d, 2.4 * d, -6.75 * d, 6.75 * d])
    ant = np.array([0.0, 0.0])
    for _ in range(50):
        w = ant_tag.sample_waypoint(rng, ant, [9.0, 9.0], half_width=7.0, min_dist=3.5, target_clearance=1.0,
                                    candidates=cand, candidate_prob=1.0, candidate_offset=0.8)
        dist = np.linalg.norm(cand - w, axis=1)
        assert dist.min() == pytest.approx(0.8, abs=1e-6)                 # next to a den (min_dist does not apply)
        assert np.all(np.abs(w) <= 7.0)
    # phantom zones: a den behind a hazard on the straight path is skipped; the heavy den (2.4 out) is reached
    active = cand[1:3]                                                       # +2.4 heavy, -6.75 light
    phantoms = np.stack([-cand[1], -cand[2]])                                # -2.4, +6.75
    for _ in range(30):
        w = ant_tag.sample_waypoint(rng, ant, [9.0, 9.0], half_width=7.0, min_dist=3.5, target_clearance=1.0,
                                    candidates=active, candidate_prob=1.0, candidate_offset=0.8, hazards=phantoms, hazard_radius=1.9)
        assert np.linalg.norm(w - cand[1]) == pytest.approx(0.8, abs=1e-6), w   # only the heavy den has a safe approach
        assert ant_tag.segment_clear(ant, w, phantoms, 1.9)
    assert not ant_tag.segment_clear(ant, cand[2] + 0.8, phantoms, 1.9) and ant_tag.segment_clear([0, 0], [1, 0], [[0, 5]], 1.0)
    # uniform draws avoid unsafe paths too
    for _ in range(50):
        w = ant_tag.sample_waypoint(rng, ant, [9.0, 9.0], half_width=7.0, min_dist=3.5, target_clearance=1.0,
                                    hazards=phantoms, hazard_radius=1.9)
        assert ant_tag.segment_clear(ant, w, phantoms, 1.9)
    # probability 0 or no candidates: the uniform rule, clearance included
    for kw in (dict(candidates=cand, candidate_prob=0.0), dict(candidates=None, candidate_prob=1.0)):
        w = ant_tag.sample_waypoint(rng, ant, [3.0, 3.0], half_width=4.5, min_dist=2.25, target_clearance=1.5, **kw)
        assert np.linalg.norm(w - ant) >= 2.25 and np.linalg.norm(w - [3.0, 3.0]) >= 1.5
    # the walk hands the env's candidates through
    args = SimpleNamespace(waypoint_min_dist=3.5, waypoint_target_clearance=1.0, waypoint_reach=0.75, waypoint_patience=80,
                           waypoint_candidate_prob=1.0, waypoint_candidate_offset=0.8)
    state = SimpleNamespace(rng=np.random.default_rng(1), locomotion_policy=object(), vecnorm_stats=None, waypoints_sampled=0)
    # the walk hands the episode's two ACTIVE dens through, never the phantoms
    env = SimpleNamespace(unwrapped=SimpleNamespace(get_target_pos=lambda: np.array([4.0, 4.0]),
                                                    cden_heavy_pos=cand[1], cden_light_pos=cand[2], cden_candidates=cand,
                                                    cden_phantom_positions=np.stack([-cand[1], -cand[2]]), phantom_terminal_radius=1.4))
    assert np.allclose(ant_tag._active_dens(env.unwrapped), cand[1:3])
    assert ant_tag._active_dens(SimpleNamespace()) is None and ant_tag._phantom_hazards(SimpleNamespace()) == (None, 0.0)
    walk = ant_tag._WaypointWalk(args, state, env, half_width=7.0)
    for _ in range(20):
        walk._resample(ant)
        d = np.linalg.norm(cand - walk.goal, axis=1)
        assert d.min() == pytest.approx(0.8, abs=1e-6) and d.argmin() == 1      # the heavy den; the light one is behind a phantom


def test_waypoint_walk_redraws_on_arrival_and_on_patience(monkeypatch):
    calls = []
    monkeypatch.setattr(ant_tag, "_goal_action", lambda obs, goal, policy, vn: calls.append(np.array(goal)) or np.zeros(8))
    args = SimpleNamespace(waypoint_min_dist=2.0, waypoint_target_clearance=0.5, waypoint_reach=0.75, waypoint_patience=3)
    args.waypoint_candidate_prob, args.waypoint_candidate_offset = 0.0, 0.8
    state = SimpleNamespace(rng=np.random.default_rng(1), locomotion_policy=object(), vecnorm_stats=None, waypoints_sampled=0)
    env = SimpleNamespace(unwrapped=SimpleNamespace(get_target_pos=lambda: np.array([4.0, 4.0])))
    walk = ant_tag._WaypointWalk(args, state, env, half_width=4.5)
    obs = {"obs": np.zeros(31)}
    walk(obs)                                   # first call draws
    g0 = walk.goal.copy()
    assert walk.n_sampled == 1 and np.linalg.norm(g0) >= 2.0
    obs["obs"][:2] = g0 + 0.1                   # arrived -> redraw
    walk(obs)
    assert walk.n_sampled == 2 and not np.allclose(walk.goal, g0)
    g1 = walk.goal.copy()
    obs["obs"][:2] = 0.0                        # not arriving: the redraw at the arrival call counted as step 1 on g1;
    for _ in range(2):                          # two more steps keep it (steps 2, 3) ...
        walk(obs)
    assert walk.n_sampled == 2 and np.allclose(walk.goal, g1)
    walk(obs)                                   # ... and the step after patience (3) redraws
    assert walk.n_sampled == 3 and state.waypoints_sampled == 3
    assert len(calls) == 5 and np.allclose(calls[1], g1) and np.allclose(calls[3], g1)   # each step drove to the live goal


def _collect(tmp_path, monkeypatch, name, *flags, episodes=3, steps=60):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    out = tmp_path / f"{name}.npz"
    collect.main(["--domain", "ant_tag", "--variant", "smart", "--num_episodes", str(episodes), "--timesteps", str(steps),
                  "--num_particles", "30", "--seed", "0", "--no_rebalance", "--locomotion_policy_path", str(LOCOMOTION),
                  *flags, "--output_file", str(out)])
    with np.load(out, allow_pickle=True) as z:
        return {k: z[k] for k in ("ant", "target", "step")}, json.loads(str(z["metadata"]))


@pytest.mark.skipif(not LOCOMOTION.exists(), reason="locomotion policy not on this machine")
def test_waypoint_episodes_walk_toward_their_goals_and_are_counted(tmp_path, monkeypatch):
    # record every (ant, goal) the walk hands the locomotion policy
    trace = []
    real = ant_tag._goal_action

    def recording(base_obs, goal_xy, policy, vn):
        trace.append((np.array(base_obs[:2], dtype=float), np.array(goal_xy, dtype=float)))
        return real(base_obs, goal_xy, policy, vn)
    monkeypatch.setattr(ant_tag, "_goal_action", recording)
    way, meta_w = _collect(tmp_path, monkeypatch, "way", "--waypoint_fraction", "1.0", "--fully_observed_fraction", "0",
                           "--pursuit_fraction", "0", episodes=2, steps=120)
    assert meta_w["episode_counts"] == {"fully_observed": 0, "pursuit": 0, "waypoint": 2, "random": 0}
    assert meta_w["waypoints_sampled"] >= 2
    # the defaults were resolved in env units: half the cage half-width, the env's visible radius
    assert meta_w["args"]["waypoint_min_dist"] == pytest.approx(2.25) and meta_w["args"]["waypoint_target_clearance"] == 3.0
    # every goal lies in the cage and started at least min_dist from the ant
    segments, cur = [], []
    for ant, goal in trace:
        if cur and not np.allclose(goal, cur[-1][1]):
            segments.append(cur); cur = []
        cur.append((ant, goal))
    segments.append(cur)
    assert all(np.all(np.abs(seg[0][1]) <= 4.5 + 1e-6) and np.linalg.norm(seg[0][0] - seg[0][1]) >= 2.25 - 1e-6 for seg in segments)
    # the ant closes on the goal: over every segment of >= 20 steps the distance to the goal falls
    long = [seg for seg in segments if len(seg) >= 20]
    assert long, "no waypoint segment long enough to judge"
    closed = [np.linalg.norm(seg[0][0] - seg[0][1]) - np.linalg.norm(seg[-1][0] - seg[-1][1]) for seg in long]
    assert np.mean(closed) > 0.5 and sum(c > 0 for c in closed) >= 0.7 * len(closed), closed
    # the file's contract is unchanged (labels ride along, particles raw)
    assert way["target"].shape == (len(way["ant"]), 2)
    # the random type is unchanged and counted
    _, meta_r = _collect(tmp_path, monkeypatch, "rnd", "--fully_observed_fraction", "0", "--pursuit_fraction", "0")
    assert meta_r["episode_counts"] == {"fully_observed": 0, "pursuit": 0, "waypoint": 0, "random": 3}
    assert meta_r["waypoints_sampled"] == 0


@pytest.mark.skipif(not LOCOMOTION.exists(), reason="locomotion policy not on this machine")
def test_den_biased_waypoints_make_the_ant_check_dens_on_cdens(tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    out = tmp_path / "cd.npz"
    collect.main(["--domain", "ant_tag", "--variant", "cdens_terminal", "--num_episodes", "3", "--timesteps", "150",
                  "--num_particles", "30", "--seed", "1", "--no_rebalance", "--locomotion_policy_path", str(LOCOMOTION),
                  "--waypoint_fraction", "1.0", "--fully_observed_fraction", "0", "--pursuit_fraction", "0",
                  "--waypoint_candidate_prob", "1.0", "--output_file", str(out)])
    with np.load(out, allow_pickle=True) as z:
        ant, step = z["ant"], z["step"]
        meta = json.loads(str(z["metadata"]))
    cand = np.asarray(meta["den_candidates"])
    assert meta["episode_counts"]["waypoint"] == 3 and meta["args"]["waypoint_candidate_prob"] == 1.0
    starts = np.flatnonzero(step == 0)
    ends = list(starts[1:]) + [len(step)]
    closest = [np.linalg.norm(ant[s:e][:, None, :] - cand[None], axis=-1).min() for s, e in zip(starts, ends)]
    # in 150 steps the ant reaches the spook radius (1.4) of some den in most episodes (it spawns ~2.4 from the heavy dens)
    assert sum(c < 1.4 for c in closest) >= 2, closest


def test_default_mix_is_the_historical_collector_and_bad_fractions_are_refused(tmp_path, capsys):
    parser = collect.build_parser(ANT_TAG, selectors=False)
    a = parser.parse_args(["--variant", "smart"])
    assert a.waypoint_fraction == 0.0 and a.waypoint_radius == "real" and a.waypoint_reach == 0.75 and a.waypoint_patience == 80
    assert a.waypoint_candidate_prob == 0.0 and a.waypoint_candidate_offset == 0.8
    with pytest.raises(SystemExit):
        ANT_TAG.collection.resolve_arguments(parser, parser.parse_args(["--variant", "smart", "--waypoint_fraction", "0.5",
                                                                        "--pursuit_fraction", "0.6"]), ANT_TAG)
    assert "exceeds 1" in capsys.readouterr().err
