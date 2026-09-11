"""Tests for the parity-aware Odd-Even particle filter.

The properties that make this env a belief benchmark at all: parity is inferred exactly,
the belief is a comb over one parity (so a Gaussian summary can sit on an impossible
state), and the collapse is physical rather than weight-only.
"""

import numpy as np
import pytest

from set_transformer.rl.particle_filters import ParityAwareOddEvenParticleFilter

N, STD = 10, 2.0
GRID = np.arange(1, N + 1)


def sample_obs(true_state, k, rng):
    valid = GRID[GRID % 2 == true_state % 2]
    p = np.exp(-0.5 * ((valid - true_state) / STD) ** 2)
    return rng.choice(valid, p=p / p.sum(), size=k)


def make_pf(true_state=3, k=1, seed=0, num_particles=100):
    rng = np.random.default_rng(seed)
    pf = ParityAwareOddEvenParticleFilter(
        num_particles, sample_obs(true_state, k, rng), n_dist_size=N, std_dev=STD)
    return pf, rng


def test_prior_covers_both_parities_uniformly():
    pf, _ = make_pf()
    belief = pf.belief_over_states()
    assert np.allclose(belief, 1.0 / N), belief
    assert pf.particles.shape == (100, 1)
    assert pf.particle_dim == 1


@pytest.mark.parametrize("true_state", [1, 3, 4, 8, 10])
def test_belief_lands_on_the_observed_parity_only(true_state):
    """The observation model only ever emits same-parity integers, so one sample is
    enough to rule out half the state space exactly."""
    pf, rng = make_pf(true_state)
    pf.update(sample_obs(true_state, 1, rng))
    belief = pf.belief_over_states()
    wrong_parity = GRID % 2 != true_state % 2
    assert belief[wrong_parity].sum() == pytest.approx(0.0, abs=1e-12)
    assert belief[~wrong_parity].sum() == pytest.approx(1.0)


def test_belief_is_a_multimodal_comb_not_a_blob():
    """The discriminating structure: mass on several same-parity states with holes
    between them. A mean+covariance summary of such a belief can point at a state the
    parity forbids, which is exactly what the analytic baselines cannot represent."""
    pf, rng = make_pf(true_state=5)
    pf.update(sample_obs(5, 1, rng))
    belief = pf.belief_over_states()
    support = GRID[belief > 0.05]
    assert len(support) >= 2, belief
    assert set(support.tolist()) <= {1, 3, 5, 7, 9}
    # Gaps between occupied states: a comb, not a contiguous blob.
    assert np.all(np.diff(support) >= 2)


def test_prediction_is_static_and_preserves_the_comb():
    """`true_state` is fixed within an episode, so predict must not diffuse the belief."""
    pf, rng = make_pf()
    pf.update(sample_obs(3, 1, rng))
    before = pf.belief_over_states().copy()
    particles_before = pf.particles.copy()
    pf.predict(None)
    assert np.array_equal(pf.particles, particles_before)
    assert np.allclose(pf.belief_over_states(), before)


def test_belief_concentrates_on_the_truth_with_evidence():
    pf, rng = make_pf(true_state=7)
    for _ in range(40):
        pf.predict(None)
        pf.update(sample_obs(7, 1, rng))
    assert pf.belief_over_states().argmax() + 1 == 7


def test_many_samples_collapse_the_belief_in_one_step():
    """Documents why the env is run at 1 observation sample/step: with the env's default
    of 100 the posterior is a point mass immediately and no encoder can differentiate."""
    pf, rng = make_pf(true_state=3, k=100)
    pf.update(sample_obs(3, 100, rng))
    belief = pf.belief_over_states()
    assert belief.max() > 0.99
    assert (belief > 0.01).sum() == 1


def test_collapse_is_physical_not_weight_only():
    """Feature extractors read the particle set unweighted, so a belief that collapses
    only in the weight vector would still look uniform to them."""
    pf, rng = make_pf(true_state=9)
    for _ in range(40):
        pf.update(sample_obs(9, 1, rng))
    assert set(np.unique(np.rint(pf.particles)).astype(int)) <= {1, 3, 5, 7, 9}
    # Particle positions themselves carry the belief.
    assert len(np.unique(np.rint(pf.particles))) <= 3


def test_survives_an_impossible_observation():
    """A filter that raised mid-episode would take the training run down with it."""
    pf, rng = make_pf(true_state=3)
    for _ in range(30):
        pf.update(sample_obs(3, 1, rng))   # collapse onto odd
    pf.update(np.array([4.0]))             # even: impossible under every live particle
    belief = pf.belief_over_states()
    assert np.isfinite(belief).all()
    assert belief.sum() == pytest.approx(1.0)
    assert belief[GRID % 2 == 1].sum() == pytest.approx(0.0, abs=1e-12)


def test_weights_stay_finite_under_many_samples_per_step():
    """Log-space scoring: raw likelihoods of 100 samples underflow float64 long before
    the belief is actually degenerate."""
    pf, rng = make_pf(true_state=5, k=200)
    for _ in range(5):
        pf.update(sample_obs(5, 200, rng))
    assert np.isfinite(pf.weights).all()
    assert pf.weights.sum() == pytest.approx(1.0)


def test_matches_the_environments_own_exact_belief_update():
    """This filter reproduces the env's `update_belief`, so the two must agree."""
    pytest.importorskip("pdomains")
    from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig

    env = OddEvenPOMDP(OddEvenPOMDPConfig(n_dist_size=N, std_dev=STD, seed=1,
                                          n_particles=1))
    pf = ParityAwareOddEvenParticleFilter(
        20_000, np.array([env.true_state]), n_dist_size=N, std_dev=STD)
    rng = np.random.default_rng(3)
    for _ in range(3):
        obs = int(sample_obs(env.true_state, 1, rng)[0])
        env.update_belief(obs)
        pf.update(np.array([float(obs)]))
    # Monte-Carlo error at 20k particles is well under 0.02.
    assert np.allclose(pf.belief_over_states(), env.belief, atol=0.02)


def test_resampling_does_not_alias_with_the_particle_layout():
    """Regression: the deterministic prior lays states out periodically, and systematic
    resampling's fixed-stride pointers alias against that period, quantizing the belief
    (0.3/0.4/0.2/0.1 instead of 0.321/0.393/0.227/0.053). The init shuffles for this
    reason -- systematic resampling assumes index and weight are uncorrelated."""
    pf = ParityAwareOddEvenParticleFilter(20_000, np.array([6.0]), n_dist_size=N,
                                          std_dev=STD)
    log_lik = pf._log_likelihood(np.array([4.0]))
    exact = np.exp(log_lik - log_lik.max())
    exact /= exact.sum()
    pf.update(np.array([4.0]))
    assert np.allclose(pf.belief_over_states(), exact, atol=0.01), (
        pf.belief_over_states(), exact)
