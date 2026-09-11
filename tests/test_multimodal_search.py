"""Tests for the Multimodal Search environment and its particle filter.

The env exists to be *discriminating by construction*, so the tests pin the construction
itself: the belief mean must be a constant (or a Gaussian summary is not provably blind),
the geometry must stay random (or an agent could sweep a fixed path instead of using the
belief), and sweeping a mode must move particle POSITIONS (or the elimination is
invisible to extractors, which read the set unweighted).
"""

import numpy as np
import pytest

from set_transformer.rl.envs.multimodal_search import (
    MultimodalSearchConfig,
    MultimodalSearchEnv,
    unpack_modes,
)
from set_transformer.rl.particle_filters import MultimodalSearchParticleFilter


def make_env(**kw):
    return MultimodalSearchEnv(MultimodalSearchConfig(seed=0, **kw))


def make_pf(env, obs, n=200):
    return MultimodalSearchParticleFilter(
        n, obs, k_max=env.config.k_max, arena_half=env.config.arena_half,
        visibility_radius=env.config.visibility_radius)


# --- the construction ---------------------------------------------------------

def test_belief_mean_is_pinned_at_the_origin():
    """THE defining property: the belief mean is the same constant every episode, so it
    carries zero information about the target and a Gaussian summary is provably blind."""
    env = make_env()
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        means, _ = unpack_modes(obs, env.config.k_max)
        assert np.abs(means.mean(axis=0)).max() < 1e-5, means.mean(axis=0)


def test_mode_geometry_stays_random():
    """Pinning the centroid on a fixed ring would also zero the mean, but then an agent
    could memorise the ring and sweep it. Distances from the origin must vary widely."""
    env = make_env()
    radii = []
    for ep in range(100):
        obs, _ = env.reset(seed=ep)
        means, _ = unpack_modes(obs, env.config.k_max)
        radii.extend(np.linalg.norm(means, axis=1))
    radii = np.asarray(radii)
    assert radii.std() > 1.5, radii.std()
    assert radii.max() - radii.min() > 5.0


def test_mode_count_varies_within_configured_range():
    env = make_env(k_min=2, k_max=10)
    counts = {make_env(k_min=2, k_max=10).reset(seed=s)[1]["num_modes"] for s in range(60)}
    assert min(counts) >= 2 and max(counts) <= 10
    assert len(counts) >= 4, counts


def test_modes_respect_minimum_separation():
    env = make_env()
    for ep in range(40):
        obs, _ = env.reset(seed=ep)
        means, _ = unpack_modes(obs, env.config.k_max)
        if len(means) < 2:
            continue
        d = np.linalg.norm(means[:, None] - means[None, :], axis=-1) + np.eye(len(means)) * 1e9
        assert d.min() >= env.config.min_separation - 1e-6


def test_target_is_uniform_over_modes_and_drawn_from_that_mode():
    """Equal chance per mode, then sampled from that mode's own Gaussian."""
    env = make_env(k_min=4, k_max=4)
    hits = np.zeros(4)
    for ep in range(400):
        _, info = env.reset(seed=ep)
        hits[info["target_mode"]] += 1
        mu = info["mode_means"][info["target_mode"]]
        cov = info["mode_covs"][info["target_mode"]]
        # Within a generous Mahalanobis radius of its own mode.
        d = info["target_pos"] - mu
        assert float(d @ np.linalg.inv(cov) @ d) < 60.0
    assert hits.min() > 400 / 4 * 0.6, hits      # roughly uniform


def test_footprint_guard_rejects_configurations_that_stop_discriminating():
    """The env only discriminates while modes cover a small fraction of the arena; past
    that, visiting every mode costs as much as sweeping everything."""
    with pytest.raises(ValueError, match="discriminating"):
        MultimodalSearchConfig(arena_half=14.0, k_max=10, mode_scale_hi=3.0)


# --- dynamics and reward ------------------------------------------------------

def test_finding_the_target_terminates_with_a_positive_return():
    env = make_env()
    obs, info = env.reset(seed=1)
    env.agent_pos = info["target_pos"].copy()          # place the agent on the target
    obs, reward, terminated, truncated, info = env.step(np.zeros(2))
    assert terminated and info["found"]
    assert reward > 0


def test_unfound_episode_truncates_at_the_horizon_with_a_negative_return():
    env = make_env(max_steps=30, find_bonus=31.0)
    env.reset(seed=2)
    env.target_pos = np.array([1e3, 1e3])              # unreachable
    total = 0.0
    for _ in range(30):
        _, r, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
        total += r
    assert truncated and not terminated
    assert total < 0


def test_action_is_a_direction_not_a_diagonal_speed_bonus():
    env = make_env()
    env.reset(seed=3)
    start = env.agent_pos.copy()
    env.step(np.array([1.0, 1.0]))
    assert np.linalg.norm(env.agent_pos - start) <= env.config.speed + 1e-6


# --- particle filter ----------------------------------------------------------

def test_prior_reproduces_the_modes_and_its_mean_is_the_origin():
    env = make_env()
    obs, info = env.reset(seed=4)
    pf = make_pf(env, obs)
    assert pf.particles.shape == (200, 2)
    assert pf.particle_dim == 2
    assert np.abs(pf.particles.mean(axis=0)).max() < 2.0     # sampling noise about zero
    # every mode is represented
    for mu in info["mode_means"]:
        assert (np.linalg.norm(pf.particles - mu, axis=1) < 3.0).any()


def test_predict_is_a_noop_because_the_target_is_static():
    env = make_env()
    obs, _ = env.reset(seed=5)
    pf = make_pf(env, obs)
    before = pf.particles.copy()
    pf.predict(np.array([1.0, 0.0]))
    assert np.array_equal(pf.particles, before)


def test_sweeping_a_tight_mode_moves_particle_POSITIONS_not_just_weights():
    """Load-bearing: extractors read the particle set unweighted, and with K modes a
    sweep kills only 1/K of the mass -- so a filter that merely re-weighted would leave
    the belief the policy sees frozen at the prior for the whole episode.

    Uses modes narrow enough to fit inside the visibility radius, so "visited" and
    "eliminated" coincide and the claim is unambiguous.
    """
    env = make_env(mode_scale_lo=0.2, mode_scale_hi=0.35)
    obs, info = env.reset(seed=6)
    pf = make_pf(env, obs)
    empty = next(k for k in range(info["num_modes"]) if k != info["target_mode"])
    centre = info["mode_means"][empty]
    other = next(k for k in range(info["num_modes"]) if k not in (empty,))

    near = lambda c: (np.linalg.norm(pf.particles - c, axis=1) < 1.5).sum()
    before, other_before = near(centre), near(info["mode_means"][other])
    assert before > 0

    swept_obs = obs.copy()
    swept_obs[0:2] = centre        # agent standing on the empty mode, target not visible
    swept_obs[2:4] = swept_obs[0:2]; swept_obs[4:7] = 0.0
    for _ in range(4):
        pf.update(swept_obs)

    assert near(centre) == 0, "particles survived inside a fully observed mode"
    assert near(info["mode_means"][other]) > other_before, "mass did not redistribute"
    assert np.allclose(pf.weights, pf.weights[0])   # mass carried by positions, not weights
    assert len(pf.particles) == 200


def test_a_mode_wider_than_the_visibility_radius_needs_several_looks():
    """One visit clears only what was actually observed. Tail mass beyond the disc
    survives -- correctly, since the target could be there -- which is what makes mode
    SHAPE matter to the optimal policy and not merely mode position."""
    env = make_env(mode_scale_lo=0.9, mode_scale_hi=1.0)
    obs, info = env.reset(seed=6)
    pf = make_pf(env, obs)
    empty = next(k for k in range(info["num_modes"]) if k != info["target_mode"])
    centre = info["mode_means"][empty]
    r = env.config.visibility_radius

    swept_obs = obs.copy()
    swept_obs[0:2] = centre
    swept_obs[2:4] = swept_obs[0:2]; swept_obs[4:7] = 0.0
    for _ in range(4):
        pf.update(swept_obs)

    assert (np.linalg.norm(pf.particles - centre, axis=1) <= r).sum() == 0
    assert (np.linalg.norm(pf.particles - centre, axis=1) < 2.5).sum() > 0


def test_sighting_collapses_the_belief_onto_the_target():
    env = make_env()
    obs, info = env.reset(seed=7)
    pf = make_pf(env, obs)
    seen = obs.copy()
    seen[0:2] = info["target_pos"] - np.array([0.5, 0.0])
    seen[2:4] = seen[0:2]
    seen[4:6] = np.array([0.5, 0.0])
    seen[6] = 1.0
    pf.update(seen)
    assert np.allclose(pf.particles, info["target_pos"], atol=1e-5)


def test_belief_survives_being_completely_refuted():
    """An exhausted belief must not emit NaNs into the policy for the rest of the run."""
    env = make_env(arena_half=3.0, k_min=1, k_max=1, min_separation=0.0,
                   visibility_radius=10.0)
    obs, _ = env.reset(seed=8)
    pf = make_pf(env, obs)
    swept = obs.copy()
    swept[2:5] = 0.0
    for _ in range(3):
        swept[0:2] = pf.particles.mean(axis=0)
        pf.update(swept)
    assert np.isfinite(pf.particles).all()
    assert pf.weights.sum() == pytest.approx(1.0)


def test_mode_parameters_are_masked_from_the_agent():
    """The agent must learn the mode structure through its belief encoder, not read it."""
    from set_transformer.rl.benchmark.registry import get_env_spec

    spec = get_env_spec("msearch")
    env = spec.make_base_env(seed=0)
    obs, _ = env.reset(seed=0)
    means, _ = unpack_modes(obs, 10)
    assert len(means) >= 2                       # the raw observation carries the prior
    masked = np.asarray(obs).copy()
    masked[spec.obs_mask_indices] = 0.0
    assert len(unpack_modes(masked, 10)[0]) == 0  # the agent's view carries none of it
    env.close()


# --- exact posterior semantics -------------------------------------------------

def test_belief_is_untouched_until_visibility_crosses_particles():
    """The target is static, so nothing but observation can move the belief. Stepping
    through empty space must leave the particle set bit-identical."""
    env = make_env()
    obs, info = env.reset(seed=3)
    pf = make_pf(env, obs)
    before = pf.particles.copy()

    # A point far from every mode, so the visibility disc contains no particles.
    means = info["mode_means"]
    empty_spot = None
    for cand in ([0.0, 0.0], [13.5, 13.5], [-13.5, 13.5], [13.5, -13.5]):
        if np.linalg.norm(means - np.array(cand), axis=1).min() > 5.0:
            empty_spot = np.array(cand)
            break
    assert empty_spot is not None

    quiet = obs.copy()
    quiet[0:2] = empty_spot
    quiet[2:4] = quiet[0:2]; quiet[4:7] = 0.0
    for _ in range(25):
        pf.update(quiet)
    assert np.array_equal(pf.particles, before)


def test_no_particle_survives_inside_an_already_observed_region():
    """The posterior is the prior restricted to the unobserved region, so a refuted
    location must never be repopulated later in the episode."""
    env = make_env()
    obs, info = env.reset(seed=3)
    pf = make_pf(env, obs)
    K = info["num_modes"]
    order = [k for k in range(K) if k != info["target_mode"]][:3]
    for k in order:
        for ang in np.linspace(0, 2 * np.pi, 10):
            o = obs.copy()
            o[0:2] = info["mode_means"][k] + 0.9 * np.array([np.cos(ang), np.sin(ang)])
            o[2:4] = o[0:2]; o[4:7] = 0.0
            pf.update(o)
    assert not pf._is_swept(pf.particles).any()


def test_refuted_particles_are_redrawn_rather_than_duplicated():
    """Copying survivors would impoverish the cloud into a few atoms, and roughening the
    copies would diffuse a belief that must not diffuse -- the target never moves."""
    env = make_env()
    obs, info = env.reset(seed=3)
    pf = make_pf(env, obs)
    empty = next(k for k in range(info["num_modes"]) if k != info["target_mode"])
    for ang in np.linspace(0, 2 * np.pi, 12):
        o = obs.copy()
        o[0:2] = info["mode_means"][empty] + 0.9 * np.array([np.cos(ang), np.sin(ang)])
        o[2:4] = o[0:2]; o[4:7] = 0.0
        pf.update(o)
    assert len(np.unique(pf.particles, axis=0)) == len(pf.particles)


def test_observed_discs_are_pruned():
    """Consecutive discs overlap almost entirely at this speed; keeping them all would
    grow the rejection test without shrinking the accepted region."""
    env = make_env()
    obs, info = env.reset(seed=3)
    pf = make_pf(env, obs)
    centre = info["mode_means"][0]
    for _ in range(40):
        o = obs.copy()
        o[0:2] = centre
        o[2:4] = o[0:2]; o[4:7] = 0.0
        pf.update(o)
    assert len(pf._swept_centres) <= 3


# --- information gain (the proxy reward) --------------------------------------

def test_information_gain_matches_the_analytic_value():
    """Gain is -log(surviving prior mass), and with equal mass per mode, eliminating m of
    K modes must give exactly -log((K-m)/K) nats."""
    env = make_env()
    obs, info = env.reset(seed=3)
    pf = make_pf(env, obs, n=400)
    K = info["num_modes"]
    assert pf.information_gain == pytest.approx(0.0, abs=1e-9)

    eliminated = 0
    for k in [x for x in range(K) if x != info["target_mode"]][:3]:
        for ang in np.linspace(0, 2 * np.pi, 12):
            o = obs.copy()
            o[0:2] = info["mode_means"][k] + 0.9 * np.array([np.cos(ang), np.sin(ang)])
            o[2:4] = o[0:2]; o[4:7] = 0.0
            pf.update(o)
        eliminated += 1
        expected = -np.log((K - eliminated) / K)
        assert pf.information_gain == pytest.approx(expected, abs=0.06), (
            eliminated, pf.information_gain, expected)


def test_information_gain_is_monotone_and_pays_nothing_for_empty_space():
    env = make_env()
    obs, info = env.reset(seed=3)
    pf = make_pf(env, obs)
    empty = None
    for cand in ([0.0, 0.0], [13.5, 13.5], [-13.5, 13.5]):
        if np.linalg.norm(info["mode_means"] - np.array(cand), axis=1).min() > 5.0:
            empty = np.array(cand)
            break
    o = obs.copy()
    o[0:2] = empty
    o[2:4] = o[0:2]; o[4:7] = 0.0
    for _ in range(30):
        pf.update(o)
    assert pf.information_gain == pytest.approx(0.0, abs=1e-9)

    prev = 0.0
    for k in range(info["num_modes"]):
        o = obs.copy()
        o[0:2] = info["mode_means"][k]
        o[2:4] = o[0:2]; o[4:7] = 0.0
        pf.update(o)
        assert pf.information_gain >= prev - 1e-9      # never decreases
        prev = pf.information_gain


def test_proxy_reward_totals_the_information_gathered():
    """With gamma=1 and no terminal zeroing the wrapper stops being policy-invariant
    shaping and becomes an outright proxy reward, whose episode total is exactly the
    change in potential. That is the trade the sparse task requires."""
    import gymnasium as gym
    from set_transformer.rl.wrappers.shaping import PotentialBasedShapingWrapper

    class _Stub(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(-1, 1, (1,))
            self.phi = 0.0
        def reset(self, **kw):
            self.phi = 0.0
            return np.zeros(1, np.float32), {}
        def step(self, a):
            self.phi += 1.0
            return np.zeros(1, np.float32), 0.0, self.phi >= 3.0, False, {}

    phi = lambda env: env.unwrapped.phi
    proxy = PotentialBasedShapingWrapper(_Stub(), phi, gamma=1.0,
                                         zero_at_termination=False)
    proxy.reset()
    total = sum(proxy.step(0)[1] for _ in range(3))
    assert total == pytest.approx(3.0)          # kept: Phi_T - Phi_0

    invariant = PotentialBasedShapingWrapper(_Stub(), phi, gamma=1.0,
                                             zero_at_termination=True)
    invariant.reset()
    total = sum(invariant.step(0)[1] for _ in range(3))
    assert total == pytest.approx(0.0)          # charged back: telescopes to zero


def test_msearch_trains_shaped_and_evaluates_unshaped():
    """The proxy reward must never reach the reported numbers."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "benchmark"))
    import importlib.util
    spec_ = importlib.util.spec_from_file_location(
        "_bt", Path(__file__).resolve().parents[1] / "experiments" / "benchmark" / "train.py")
    train = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(train)
    from set_transformer.rl.benchmark.registry import get_env_spec

    env_spec = get_env_spec("msearch")
    assert env_spec.potential_fn is not None
    shaped = train.make_env_thunk(env_spec, 0, for_eval=False)()
    unshaped = train.make_env_thunk(env_spec, 0, for_eval=True)()
    shaped.reset(seed=0); unshaped.reset(seed=0)
    _, _, _, _, i_shaped = shaped.step(shaped.action_space.sample())
    _, _, _, _, i_eval = unshaped.step(unshaped.action_space.sample())
    assert "shaping_reward" in i_shaped
    assert "shaping_reward" not in i_eval
    shaped.close(); unshaped.close()


def test_particles_reach_the_encoder_agent_relative():
    """The optimal policy is a relation between the agent and the belief ('head for the
    nearest unvisited mode'). In world coordinates every method must learn to subtract its
    own position, which arrives through a separate MLP; re-centring makes the geometry
    immediate and does so identically for all methods."""
    import sys
    from pathlib import Path
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "experiments" / "benchmark"))
    spec_ = importlib.util.spec_from_file_location(
        "_bt_rel", root / "experiments" / "benchmark" / "train.py")
    train = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(train)
    from set_transformer.rl.benchmark.registry import get_env_spec

    env_spec = get_env_spec("msearch")
    assert env_spec.particle_origin_fn is not None
    env = train.make_env_thunk(env_spec, 0, for_eval=True)()
    obs, _ = env.reset(seed=0)

    inner = env
    while inner is not None and not hasattr(inner, "particle_filter"):
        inner = getattr(inner, "env", None)
    world = inner.particle_filter.particles
    agent = inner.unwrapped.agent_pos
    assert np.allclose(obs["particles"], world - agent, atol=1e-4)
    # The filter itself must stay in world coordinates, or its sweep logic breaks.
    assert not np.allclose(world, obs["particles"], atol=1e-4)
    env.close()


def test_world_frame_envs_are_untouched_by_the_relative_option():
    """particle_origin_fn is opt-in per env; the others must see world coordinates."""
    from set_transformer.rl.benchmark.registry import get_env_spec
    for name in ("car_flag", "odd_even", "ant_tag"):
        assert get_env_spec(name).particle_origin_fn is None
