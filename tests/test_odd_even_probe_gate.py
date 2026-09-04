"""The step-4 probe gate must actually discriminate, and must score both axes.

`experiments/odd_even/diagnostics/probe_odd_even_belief_separability.py` is
what decides whether a multi-million-step run is worth starting (PITFALLS.md
section 6). A gate that cannot fail is worthless, so these tests pin that it
separates a live encoding from a dead one, and that it needs BOTH axes.

Why both axes: on this domain a probe alone cannot detect encoder collapse.
Encodings spanning three orders of magnitude of feature spread -- including
provably near-constant ones -- all probe s* at R^2 within 0.10 of each other,
because a ~1e-4 signal on an O(1) offset is linearly recoverable by ridge and
not learnable by a policy head.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("pdomains")
pytest.importorskip("sklearn")

_DIAG = _ST_ROOT / "experiments" / "odd_even" / "diagnostics"


def _gate():
    """Load the gate by explicit path (Gap 12: flat names collide)."""
    import importlib.util
    path = _DIAG / "probe_odd_even_belief_separability.py"
    spec = importlib.util.spec_from_file_location("_oe_probe_gate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_oe_probe_gate"] = module
    saved = list(sys.path)
    try:
        sys.path.insert(0, str(_DIAG.parent))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


@pytest.fixture(scope="module")
def beliefs():
    g = _gate()
    return g, g.collect_beliefs("oe50", 40, 12, 4000)


def test_collect_beliefs_covers_the_transient(beliefs):
    """Snapshots must span the pre-collapse steps, not only the one-hot tail.

    The belief locks on by about step 21, so a gate that sampled only late
    steps would compare encodings in the regime where all of them are
    equivalent.
    """
    _g, (states, weights, labels, groups, steps) = beliefs
    assert states.shape[0] == weights.shape[0] == len(labels) == len(steps)
    assert steps.min() == 0 and steps.max() == 11
    assert len(set(groups)) == 40, "episode groups must be distinct"
    # Grouping matters: consecutive steps share s*, so an ungrouped CV split
    # would leak the label between folds.
    for ep in set(groups):
        assert len(set(labels[groups == ep])) == 1


def test_relative_spread_is_scale_free(beliefs):
    """Scaling every feature must not change the spread statistic.

    This is the property feat_std_mean lacks, and the reason PITFALLS'
    absolute ~0.01 abort threshold does not transfer between domains.
    """
    g, _ = beliefs
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(200, 16)) + 5.0
    base = g.relative_spread(feats)
    for factor in (1e-3, 1e3):
        assert g.relative_spread(feats * factor) == pytest.approx(base, rel=1e-6)


def test_gate_flags_a_constant_encoding_as_collapsed(beliefs):
    """A constant encoding must be flagged, however well it probes.

    This is the discriminating test. A near-constant encoding carrying a tiny
    label-correlated offset still probes near-perfectly, so only the spread
    axis can reject it.
    """
    g, (states, weights, labels, groups, _steps) = beliefs
    live = weights.copy()
    # A near-constant encoding: an O(1) offset plus a 1e-6 label-linear signal.
    dead = np.ones((len(labels), 8)) * 1.7 + 1e-6 * labels[:, None]

    live_spread = g.relative_spread(live)
    dead_spread = g.relative_spread(dead)
    assert live_spread >= g.COLLAPSE_RELATIVE_SPREAD, live_spread
    assert dead_spread < g.COLLAPSE_RELATIVE_SPREAD, dead_spread

    # The probe alone CANNOT tell them apart -- that is the point.
    dead_r2, _ = g.probe(dead, labels, groups, 4)
    assert dead_r2 > 0.9, (
        f"a constant encoding probed {dead_r2:.3f}; if this ever drops, the "
        "two-axis design is no longer necessary and this test should be "
        "revisited")


def test_exact_posterior_is_informative_and_alive(beliefs):
    """The posterior itself is the ceiling and must pass both axes."""
    g, (states, weights, labels, groups, _steps) = beliefs
    r2, _sd = g.probe(weights, labels, groups, 4)
    assert r2 > 0.8, r2
    assert g.relative_spread(weights) >= g.COLLAPSE_RELATIVE_SPREAD


def test_reward_is_read_from_the_env_not_assumed():
    """The env's reward decides which belief readout is optimal. Read it.

    This test replaces test_squared_error_reward_makes_the_mean_sufficient,
    which computed -(pick - s*)**2 inline instead of calling env.get_reward().
    Its docstring promised to fail if the env's reward ever changed -- but
    because it scored with its own hardcoded copy of the old rule, it kept
    passing when the reward became 0/1 exact match on 2026-09-03. A gate that
    restates the thing it is guarding cannot detect the thing changing.
    PITFALLS.md #6: the probe measures what you asked it to, not what you meant.

    So: score with env.get_reward, and assert the readout that the CURRENT
    reward makes Bayes-optimal actually wins.
    """
    from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig
    from set_transformer.rl.particle_filters.odd_even import (
        OddEvenExactSupportParticleFilter as PF)
    ns, cap = 50, 30
    totals = {"mean": 0.0, "argmax": 0.0}
    for ep in range(60):
        env = OddEvenPOMDP(OddEvenPOMDPConfig(n_dist_size=ns, seed=9000 + ep))
        obs0, _ = env.reset(seed=9000 + ep)
        pf = PF(num_particles=ns, initial_env_obs=obs0, n_dist_size=ns,
                rng_seed=ep)
        for _t in range(cap):
            b = pf.weights / pf.weights.sum()
            s_ = pf.particles.ravel()
            picks = {"mean": int(np.clip(round(float((b * s_).sum())), 1, ns)),
                     "argmax": int(s_[int(np.argmax(b))])}
            for key, pick in picks.items():
                totals[key] += float(env.get_reward(pick))   # THE ENV'S rule
            emitted = env._draw_observations(1)
            pf.predict(np.array(0))
            pf.update(emitted)
    # Under 0/1 exact match the Bayes action is the posterior MODE.
    assert totals["argmax"] > totals["mean"], (
        "under 0/1 exact-match reward the posterior mode must beat the mean; "
        f"got argmax={totals['argmax']:.1f} mean={totals['mean']:.1f}. "
        "If the env's reward changed again, re-derive this AND "
        "OddEvenPOMDP.get_optimal_prediction, which must stay the argmax of "
        "E[R|a] under whatever reward is in force.")


def test_env_oracle_matches_the_env_reward():
    """get_optimal_prediction() must be the argmax of E[R|a], by brute force.

    The oracle and the reward are two halves of one definition and they drifted
    apart once already: get_optimal_prediction returned the rounded posterior
    mean, which was right under squared error and wrong under 0/1 (the rounded
    mean differs from the mode in ~34% of beliefs). Anything built on the
    oracle -- probe labels, the reference ceiling in the encoder comparison --
    silently inherits that error. So derive the optimum from get_reward itself
    rather than trusting either formula.
    """
    from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig
    ns = 50
    checked = 0
    for ep in range(12):
        env = OddEvenPOMDP(OddEvenPOMDPConfig(n_dist_size=ns, seed=400 + ep))
        env.reset(seed=400 + ep)
        for _t in range(12):
            b = env.belief
            pts = env.belief_points
            # E[R | a] under the env's own reward, for every legal action
            exp = np.array([
                sum(b[j] * _reward_if_truth(env, int(a), int(pts[j]))
                    for j in range(len(pts)))
                for a in pts])
            best = int(pts[int(np.argmax(exp))])
            got = env.get_optimal_prediction()
            assert abs(exp[got - 1] - exp[best - 1]) < 1e-12, (
                f"oracle picked {got} (E[R]={exp[got-1]:.4f}) but {best} "
                f"scores {exp[best-1]:.4f}")
            checked += 1
            env.step(got - 1)
    assert checked >= 100, checked


def _reward_if_truth(env, action: int, truth: int) -> float:
    """env.get_reward(action) evaluated as if `truth` were the true state."""
    saved = env.true_state
    try:
        env.true_state = truth
        return float(env.get_reward(action))
    finally:
        env.true_state = saved


