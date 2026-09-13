"""Contract tests for the Odd-Even POMDP belief-encoding pipeline.

The env under test is `pdomains.odd_even_pomdp.OddEvenPOMDP`. It is NOT
`src/odd_even_HG/odd_even_beliefmdp.py`, which is a different environment with
a different observation, reward and reset. See domain_mds/oddeven.md.

Two kinds of test live here, and the marker says which:

* **Plain tests** pin facts that hold today. They are the regression anchors
  for the reference baselines and for the structural properties the whole
  experiment rests on (parity is free, the posterior goes rank-1, an env and
  its filter must agree).

* **`xfail(strict=True)` tests** pin behaviour that is BROKEN or MISSING
  today, one per gap in domain_mds/oddeven.md. `strict` matters: when a gap is
  fixed the test reports XPASS, which pytest treats as a failure, so the suite
  tells you to delete the marker instead of quietly going green. Never relax a
  marker to non-strict to silence it.

Gaps 1-12 are fixed, so no markers remain and every test here is plain. The
file stays the gap TRACKER: a new gap arrives as a new xfail(strict=True)
test, and gets its marker removed when the gap closes. Tests for the built
pipeline itself live in tests/test_odd_even_pipeline.py.

Cross-domain pitfalls these tests guard, from domain_mds/PITFALLS.md:
  section 2  a constant reset seed turns N samples into one, repeated
  section 3  weighted sets: mass in the measure, never in the metric
  section 5  the episode cap must come from the gym registration
  section 7  sys.path depth, and digit-leading sibling module imports
"""

import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_ROOT = Path(__file__).resolve().parents[1]
_ODD_EVEN_DIR = _ST_ROOT / "experiments" / "odd_even"
_ANT_TAG_DIR = _ST_ROOT / "experiments" / "ant_tag"
# Only the package roots go on sys.path. The experiment directories must NOT:
# see _load_experiment_module below.
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@contextmanager
def _sys_path(*directories):
    """Temporarily prepend directories, then restore sys.path exactly.

    Leaking an experiment directory into the global sys.path is how a flat
    module name silently resolves to the wrong file (Gap 12).
    """
    saved = list(sys.path)
    saved_modules = set(sys.modules)
    try:
        for directory in directories:
            sys.path.insert(0, str(directory))
        yield
    finally:
        sys.path[:] = saved
        for name in set(sys.modules) - saved_modules:
            sys.modules.pop(name, None)


def _load_experiment_module(directory, name):
    """Import experiments/<directory>/<name>.py BY PATH, under a unique key.

    PITFALLS.md section 7 covers importing a digit-leading sibling with
    importlib.import_module, which resolves off sys.path[0]. That is not
    enough here: experiments/ant_tag/ and experiments/odd_even/ hold
    same-named numbered scripts, so a flat-name import picks whichever
    directory happens to be first on sys.path. These tests found exactly that
    -- three Odd-Even assertions were silently exercising the Ant-Tag files.
    Loading by explicit path removes the ambiguity.
    """
    path = Path(directory) / f"{name}.py"
    if not path.exists():
        raise ModuleNotFoundError(f"{path} does not exist")
    unique = f"_oe_test_{Path(directory).name}_{name}"
    spec = importlib.util.spec_from_file_location(unique, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    with _sys_path(directory):
        spec.loader.exec_module(module)
    return module

pytest.importorskip("pdomains")

from pdomains.odd_even_pomdp import OddEvenPOMDP, OddEvenPOMDPConfig  # noqa: E402

#: n_dist_size used by most tests. Large enough that the same-parity grid
#: (spacing 2) needs several observations to resolve, small enough to be fast.
NS = 50

#: The collapse step measured in domain_mds/oddeven.md: the exact posterior
#: reaches argmax == truth with max(belief) > 0.9 by here, at NS = 50.
COLLAPSE_STEP = 21


def make_env(ns=NS, seed=0, obs_per_step=1, **kwargs):
    """An OddEvenPOMDP emitting `obs_per_step` observations per step.

    `n_particles` is the env's current name for the emitted-observation count
    (Gap 3). It is passed through here so these tests keep working after the
    field is renamed to `obs_per_step` with `n_particles` as an alias.
    """
    field = ("obs_per_step"
             if "obs_per_step" in OddEvenPOMDPConfig.__dataclass_fields__
             else "n_particles")
    config = OddEvenPOMDPConfig(n_dist_size=ns, seed=seed,
                                **{field: obs_per_step}, **kwargs)
    return OddEvenPOMDP(config)


def draw_observation(env):
    """One observation from the env's own generative model."""
    return int(env.rng.choice(env.valid_numbers, p=env.observation_probs))


def exact_posterior(env, n_steps, seed=0):
    """Run the env's own Bayes filter for n_steps and return the posterior."""
    env.reset(seed=seed)
    for _ in range(n_steps):
        env.update_belief(draw_observation(env))
    return env.belief.copy()


def ess(weights):
    """Effective sample size of a normalized weight vector."""
    w = np.asarray(weights, dtype=np.float64)
    return float(1.0 / np.sum((w / w.sum()) ** 2))


# ---------------------------------------------------------------------------
# Facts the experiment design rests on. These pass today. Keep them passing.
# ---------------------------------------------------------------------------


def test_observation_space_contract():
    """The declared space must accept the observation the env actually emits.

    The Odd-Even note in PITFALLS.md is about a float64 belief in a float32
    space on the OTHER Odd-Even env. This env emits float32 into a float32
    space, so it is consistent -- asserted here so it stays that way if the
    observation ever becomes the belief.
    """
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs), (
        f"obs dtype={obs.dtype} shape={obs.shape} rejected by "
        f"{env.observation_space}")


def test_observation_always_matches_true_parity():
    """Parity is FREE: one observation reveals it exactly.

    Observations are restricted to the integers sharing the true state's
    parity, so a single draw fixes the parity with certainty. This is why
    parity is the wrong probe target on this env, and why the raw observation
    must be kept out of the agent's obs -- see Gap 9 in oddeven.md.

    domain_mds/PITFALLS.md names parity as the decision-relevant bit, from an
    audit of src/odd_even_HG/. This test is the evidence that it is not one
    HERE.
    """
    for seed in range(25):
        env = make_env(seed=seed)
        env.reset(seed=seed)
        for _ in range(10):
            assert draw_observation(env) % 2 == env.true_state % 2


def test_update_belief_raises_on_impossible_observation():
    """A zero-posterior observation must raise, never reset to uniform.

    Silently recovering would hide a real bug -- an env/filter mismatch of
    exactly the kind PITFALLS.md warns about -- behind a plausible belief.
    """
    env = make_env(ns=10, seed=0)
    env.reset(seed=0)
    wrong_parity = 2 if env.true_state % 2 == 1 else 1
    env.update_belief(draw_observation(env))    # collapse onto one parity
    with pytest.raises(ValueError, match="Impossible observation"):
        env.update_belief(wrong_parity)


def test_env_likelihood_normalizes_over_candidate_parity_set():
    """P(obs | candidate) must sum to 1 over the candidate's own parity set.

    This is the likelihood any particle filter for this env has to call. If a
    filter rolls its own Gaussian instead, the two drift and belief
    propagation diverges from the env with no error raised.
    """
    env = make_env(ns=10, seed=0)
    for candidate in range(1, 11):
        support = (env.odd_numbers if candidate % 2 == 1 else env.even_numbers)
        total = sum(env._compute_observation_probability(int(o), candidate)
                    for o in support)
        assert total == pytest.approx(1.0), f"candidate {candidate}"
        # Zero mass on the opposite parity, for every candidate.
        other = (env.even_numbers if candidate % 2 == 1 else env.odd_numbers)
        for o in other:
            assert env._compute_observation_probability(int(o), candidate) == 0.0


def test_exact_posterior_collapses_to_rank_one():
    """The belief goes one-hot fast, so long-episode means hide encoders.

    Measured in oddeven.md: ESS at steps 1, 5, 10, 20, 50 is about
    5.2, 2.8, 2.3, 1.4, 1.0 at ns=50. An encoder comparison averaged over a
    long episode is therefore averaging mostly over the regime where every
    encoder is equivalent. Metrics must be split at COLLAPSE_STEP.
    """
    early, late = [], []
    for seed in range(30):
        env = make_env(seed=seed)
        early.append(ess(exact_posterior(env, 1, seed=seed)))
        env = make_env(seed=seed)
        late.append(ess(exact_posterior(env, 50, seed=seed)))
    assert np.median(early) > 3.0, f"prior should be diffuse, got {np.median(early)}"
    assert np.median(late) < 1.5, f"belief should be rank-1, got {np.median(late)}"


def test_belief_locks_on_by_the_documented_collapse_step():
    """argmax(posterior) == truth for most episodes by COLLAPSE_STEP."""
    hits = 0
    trials = 60
    for seed in range(trials):
        env = make_env(seed=seed)
        env.reset(seed=seed)
        for _ in range(COLLAPSE_STEP):
            env.update_belief(draw_observation(env))
        hits += int(env.get_max_likelihood_prediction() == env.true_state)
    assert hits / trials > 0.6, f"only {hits}/{trials} locked on"


def test_oracle_beats_playing_the_raw_observation():
    """The reference gap any encoder comparison must resolve.

    oddeven.md records -1.098 per step for the greedy-argmax oracle against
    -9.797 for playing the raw observation, at ns=50 over a 50-step cap. That
    8.7-per-step gap is what belief accumulation buys. Loose bounds here, so
    the test measures the gap and not RNG noise.
    """
    cap = 50
    oracle, naive = [], []
    for seed in range(60):
        env = make_env(seed=seed)
        env.reset(seed=seed)
        r_o = r_n = 0.0
        for _ in range(cap):
            obs = draw_observation(env)
            env.update_belief(obs)
            r_o += -float(env.get_max_likelihood_prediction() - env.true_state) ** 2
            r_n += -float(obs - env.true_state) ** 2
        oracle.append(r_o / cap)
        naive.append(r_n / cap)
    assert np.mean(oracle) > -3.0, f"oracle {np.mean(oracle):.3f}"
    assert np.mean(naive) < -6.0, f"naive {np.mean(naive):.3f}"
    assert np.mean(oracle) - np.mean(naive) > 4.0


def test_repeated_reset_with_same_seed_is_deterministic():
    """Correct gymnasium behaviour, and the trap it sets on this env.

    The hidden state is drawn AT RESET, so on this env the seed IS the
    episode. An eval loop that passes a constant --seed to every reset
    measures one episode N times and reports N samples. That is
    PITFALLS.md section 2's failure mode reached by a different route, so
    section 2's fix (re-seed after PPO.load) does not cover it.

    Pinned as a passing test because the env is right and the CALLER is what
    has to change -- see test_per_episode_seeding_gives_distinct_episodes.
    """
    env = make_env()
    episodes = []
    for _ in range(3):
        obs, info = env.reset(seed=42)
        episodes.append((float(obs[0]), info["true_state"]))
    assert len(set(episodes)) == 1, (
        "reset(seed=42) must be reproducible; got varying episodes "
        f"{episodes}")


def test_per_episode_seeding_gives_distinct_episodes():
    """The pattern every eval loop on this env must use: seed + episode_index."""
    env = make_env()
    states = [env.reset(seed=1000 + i)[1]["true_state"] for i in range(30)]
    assert len(set(states)) > 5, (
        f"seed + episode_index gave only {len(set(states))} distinct states "
        "out of 30 episodes; the seed is not reaching the episode")


def test_sb3_vec_env_seeding_is_not_frozen():
    """SB3 applies a seed on the first reset only, so vec-env eval is safe.

    Guard, not a bug: if this ever starts returning one repeated state, every
    vec-env evaluation on this domain silently becomes a single sample.
    """
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv

    class _Cap(gym.Wrapper):
        def __init__(self, env, cap=3):
            super().__init__(env)
            self.cap, self.t = cap, 0

        def reset(self, **kwargs):
            self.t = 0
            return self.env.reset(**kwargs)

        def step(self, action):
            obs, r, term, trunc, info = self.env.step(action)
            self.t += 1
            return obs, r, term, self.t >= self.cap or trunc, info

    venv = DummyVecEnv([lambda: _Cap(make_env())])
    venv.seed(7)
    states = []
    for _ in range(6):
        venv.reset()
        done = False
        while not done:
            _, _, dones, infos = venv.step(np.array([0]))
            done = bool(dones[0])
        states.append(infos[0].get("true_state"))
    venv.close()
    assert len(set(states)) > 1, f"vec-env replayed one episode: {states}"


# ---------------------------------------------------------------------------
# Gap 1 -- gym registration
# ---------------------------------------------------------------------------


def test_gym_registration_exists():
    """An Odd-Even env must be reachable by id, like every other pdomains env.

    Without a registration, gym.make(env_id) -- the call every generic
    pipeline script makes -- cannot build the env at all.
    """
    import gymnasium as gym
    import pdomains  # noqa: F401
    ids = [k for k in gym.registry if "odd" in k.lower() or "even" in k.lower()]
    assert ids, "pdomains registers no Odd-Even env id"


def test_episode_cap_comes_from_registry():
    """PITFALLS.md section 5: the cap must come from the env, not a flag.

    An eval default larger than the real cap counted every timeout as a
    success on Ant-Tag and read ~100%. The fix there was to read
    max_episode_steps off the registration, which requires one to exist.
    """
    import gymnasium as gym
    import pdomains  # noqa: F401
    ids = [k for k in gym.registry if "odd" in k.lower() or "even" in k.lower()]
    assert ids, "no Odd-Even env id to read a cap from"
    for env_id in ids:
        cap = gym.spec(env_id).max_episode_steps
        assert cap is not None and cap > 0, f"{env_id} registers no cap"


#: The registered ids. Named here rather than discovered, so a registration
#: that silently disappears fails the test instead of shrinking the loop.
REGISTERED_IDS = (
    "pdomains-odd-even-10-v0",
    "pdomains-odd-even-50-v0",
    "pdomains-odd-even-50-long-v0",
)

EXPECTED_CAPS = {
    "pdomains-odd-even-10-v0": 50,
    "pdomains-odd-even-50-v0": 50,
    "pdomains-odd-even-50-long-v0": 200,
}


def test_registered_ids_pass_sb3_env_checker():
    """Every registered id must build by id and satisfy SB3's checker.

    gym.make(env_id) is the call every generic pipeline script makes, and SB3
    refuses an env whose emitted observation is outside its declared space.
    Checked on each id because they differ in n_dist_size, and the space's
    bounds are derived from it.
    """
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    import pdomains  # noqa: F401
    from stable_baselines3.common.env_checker import check_env

    for env_id in REGISTERED_IDS:
        env = gym.make(env_id)
        try:
            # .unwrapped: gym.make adds TimeLimit and the passive checker, and
            # check_env only reports on what it is handed.
            check_env(env.unwrapped)
        finally:
            env.close()


def test_registered_caps_match_the_variant_intent():
    """The caps are the difference between the three ids, so pin them.

    PITFALLS.md section 5: the cap must come from the registration. A cap
    that drifts silently changes what a reported mean reward means -- the
    oracle's own per-step reward moves from -1.098 to -0.582 between a 50-
    and a 100-step cap purely from transient dilution.
    """
    import gymnasium as gym
    import pdomains  # noqa: F401
    for env_id, cap in EXPECTED_CAPS.items():
        assert gym.spec(env_id).max_episode_steps == cap, env_id


def test_registered_ids_emit_one_observation_per_step():
    """obs_per_step must be 1 on every registered id.

    This is the setting Gap 3 is about, and it is the one that decides
    whether there is a belief problem at all. Read off the built env rather
    than the registration kwargs, so an entry-point shim that drops kwargs
    is caught too.
    """
    import gymnasium as gym
    import pdomains  # noqa: F401
    for env_id in REGISTERED_IDS:
        env = gym.make(env_id)
        try:
            assert env.unwrapped.obs_per_step == 1, env_id
            assert env.observation_space.shape == (1,), env_id
        finally:
            env.close()


# ---------------------------------------------------------------------------
# Gap 2 -- step() never updates the belief
# ---------------------------------------------------------------------------


def test_step_updates_env_belief():
    """step() must advance the env's own posterior.

    update_belief() is correct but unreachable from step(), so env.belief
    stays exactly uniform for the whole episode. Anything that reads it to
    build an oracle or a probe label gets a constant.
    """
    env = make_env(ns=10, seed=0)
    env.reset(seed=0)
    before = env.belief.copy()
    for _ in range(5):
        env.step(0)
    assert not np.allclose(env.belief, before), (
        "env.belief is unchanged after 5 steps; it is still the uniform prior")


def test_oracle_prediction_is_not_constant():
    """The env's own oracle helpers must track the belief.

    get_max_likelihood_prediction() and get_optimal_prediction() read
    env.belief. With the belief frozen uniform they return the same index
    every episode -- state 1, the argmax of a tie -- which silently makes any
    oracle built on them a constant policy rather than a reference.

    Measured ACROSS episodes, not within one. Within a single episode a
    correct oracle may legitimately be constant: at ns=10 the belief can lock
    onto the truth from the very first observation and stay there (seeds 2 and
    3 do exactly that, both on true_state 9). Across episodes it cannot be
    constant, because it has to follow a state that is redrawn at every reset
    -- so this is the stronger reading of the same requirement, and it also
    checks the oracle is RIGHT and not merely moving.
    """
    predictions = set()
    correct = 0
    episodes = 8
    for seed in range(episodes):
        env = make_env(ns=10, seed=seed)
        env.reset(seed=seed)
        for _ in range(30):
            env.step(0)
            predictions.add(int(env.get_max_likelihood_prediction()))
        correct += int(env.get_max_likelihood_prediction() == env.true_state)
    assert len(predictions) > 1, (
        f"oracle returned the constant {predictions} over {episodes} episodes")
    assert correct >= episodes - 1, (
        f"oracle ended on the true state in only {correct}/{episodes} "
        "episodes; it is not tracking the posterior")


def test_reset_folds_in_its_own_observation():
    """b0 = P(s | o0). reset() must USE the observation it returns.

    reset() used to return an observation and leave the belief uniform, so a
    T-step episode drew T + 1 observations and used T. The wasted one is the
    most valuable in the episode: it is the only evidence available to the
    first action, and at ns=50 over a 50-step cap step 1 alone is 81% of the
    optimal policy's pooled mean reward -- measured, the optimal policy went
    from -4.956 to -0.941 per step once it was folded in.
    """
    for seed in range(10):
        env = make_env(ns=NS, seed=seed)
        obs, info = env.reset(seed=seed)
        assert not np.allclose(info["belief"], 1.0 / NS), (
            "reset left the uniform prior; its observation was discarded")
        # Exactly one Bayes update of the uniform prior on the returned
        # observation, computed independently of update_belief().
        observation = int(round(float(obs[0])))
        likelihood = np.array([
            env._compute_observation_probability(observation, state)
            for state in range(1, NS + 1)])
        expected = likelihood / likelihood.sum()
        np.testing.assert_allclose(info["belief"], expected, atol=1e-12)
        # And the belief is over the observation's own parity only.
        wrong_parity = np.arange(1, NS + 1) % 2 != observation % 2
        assert info["belief"][wrong_parity].max() == 0.0


def test_step_info_exposes_the_posterior():
    """The posterior must travel in info, and never in the observation.

    Probe labels and the greedy-argmax oracle need the exact belief; the
    agent must not have it, or the belief encoder under test is bypassed.
    """
    env = make_env(ns=10, seed=0)
    obs, info = env.reset(seed=0)
    assert not np.allclose(info["belief"], 1.0 / 10), (
        "reset must expose b0 = P(s | o0), not the uniform prior")
    obs, _reward, _term, _trunc, info = env.step(0)
    for key in ("belief", "belief_points", "optimal_prediction",
                "max_likelihood_prediction", "observations"):
        assert key in info, f"missing {key} in {sorted(info)}"
    assert np.isclose(np.sum(info["belief"]), 1.0)
    assert not np.allclose(info["belief"], 1.0 / 10), (
        "the posterior in info is still the prior")
    assert "optimal_prediction" in info
    # The observation carries the raw draw only. It is float32 and of length
    # obs_per_step, so it cannot be smuggling an n-vector belief.
    assert obs.shape == (1,) and obs.dtype == np.float32


# ---------------------------------------------------------------------------
# Gap 3 -- the default observation makes the problem degenerate
# ---------------------------------------------------------------------------


def test_obs_per_step_defaults_to_one():
    """One observation per step, and a name that says so.

    The emitted-observation count, the observation-space shape and the belief
    encoder's particle count are three different quantities that all read
    n_particles today. The default of 100 is what makes the belief problem
    degenerate (see the next test).
    """
    fields = OddEvenPOMDPConfig.__dataclass_fields__
    assert "obs_per_step" in fields, (
        "OddEvenPOMDPConfig has no obs_per_step; the emitted-observation "
        "count is still called n_particles and shared with other meanings")
    assert OddEvenPOMDPConfig().obs_per_step == 1


def test_single_step_does_not_reveal_the_state():
    """One step must not pin the state, or there is no belief to encode.

    At the current default the env emits 100 independent observations per
    step. At ns=50 that is a standard error of the mean of about 0.28 against
    a grid spacing of 1, so a memoryless policy solves the task at step 1 and
    every encoder scores the same.
    """
    errors = []
    for seed in range(20):
        config = OddEvenPOMDPConfig(n_dist_size=NS, seed=seed)   # defaults only
        env = OddEvenPOMDP(config)
        obs, info = env.reset(seed=seed)
        errors.append(abs(float(np.mean(obs)) - info["true_state"]))
    assert np.mean(errors) > 1.0, (
        f"one step already locates the state to +/-{np.mean(errors):.2f}; "
        "the belief problem is degenerate at the default obs_per_step")


def test_n_particles_alias_resolves_to_obs_per_step():
    """The deprecated name must keep working, and must warn."""
    with pytest.warns(DeprecationWarning):
        config = OddEvenPOMDPConfig(n_dist_size=10, n_particles=8)
    assert config.obs_per_step == 8
    env = OddEvenPOMDP(config)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,), "the observation space must follow obs_per_step"


def test_n_particles_alias_disagreement_raises():
    """Two contradicting values must raise, not silently pick one.

    The choice decides whether the task has a belief problem at all -- 1 vs
    100 observations per step is the difference between ~20 steps of belief
    accumulation and the state being pinned at step 1 -- so resolving it
    quietly would change the experiment without saying so.
    """
    with pytest.raises(ValueError, match="contradicts"):
        OddEvenPOMDPConfig(n_dist_size=10, obs_per_step=1, n_particles=100)
    # Agreeing values are fine.
    assert OddEvenPOMDPConfig(obs_per_step=4, n_particles=4).obs_per_step == 4


# ---------------------------------------------------------------------------
# Gap 4 -- crashing bugs and dead code
# ---------------------------------------------------------------------------


def test_true_particles_false_branch_runs():
    """The true_particles=False resampling branch must not crash.

    It calls rng.choice(..., p=1-weights) where 1-weights is not a
    distribution, assigns float draws into an integer array, and builds a
    Gaussian whose scale is 0 whenever every draw agrees.
    """
    env = make_env(ns=10, seed=0, obs_per_step=8, true_particles=False)
    env.reset(seed=0)
    for _ in range(5):
        env.step(0)


def test_get_particle_set_exists():
    """The module's own run_example() calls a method that is not there."""
    env = make_env(ns=10, seed=0)
    env.reset(seed=0)
    particles = env.get_particle_set(64)
    assert np.asarray(particles).shape[0] == 64


def test_no_dead_particle_helpers():
    """_init_particle_set() has no caller and returns unused uniform floats."""
    assert not hasattr(OddEvenPOMDP, "_init_particle_set")


# ---------------------------------------------------------------------------
# Gap 5 -- the particle filter models the wrong system
# ---------------------------------------------------------------------------


def _filter_module():
    return importlib.import_module(
        "set_transformer.rl.particle_filters.odd_even")


def _build_filter(name, ns=NS, num_particles=100, seed=0,
                  initial_env_obs=None, **kwargs):
    """Build a filter by class name, preferring a correct new one.

    `initial_env_obs` defaults to None -- the bare uniform prior -- because a
    filter consumes it as evidence (b0 = P(s | o0), matching the env's
    reset()). A FABRICATED initial observation is not a harmless placeholder
    here: it refutes every state of the opposite parity, so the filter then
    raises, correctly, as soon as the env's real same-parity observations
    arrive. Tests that need a real b0 pass the env's own reset observation.
    """
    module = _filter_module()
    cls = getattr(module, name, None)
    if cls is None:
        pytest.fail(f"{name} does not exist in {module.__name__}")
    return cls(num_particles=num_particles,
               initial_env_obs=initial_env_obs,
               n_dist_size=ns, rng_seed=seed, **kwargs)


def test_filter_particles_are_integer_states():
    """Particles must be integer states, because parity defines the support.

    The current filter initializes with np.random.uniform(1, n) and adds
    Gaussian process noise, giving particles like 5.77 and 6.24. A
    non-integer particle has no parity, so the filter cannot represent the one
    structural fact this domain is built on.
    """
    module = _filter_module()
    name = next((n for n in ("OddEvenExactSupportParticleFilter",
                              "OddEvenBootstrapParticleFilter",
                              "OddEvenParticleFilter")
                 if hasattr(module, n)), None)
    pf = _build_filter(name)
    particles = np.asarray(pf.particles, dtype=np.float64)
    assert np.allclose(particles, np.round(particles)), (
        f"{name} holds non-integer particles: {particles.ravel()[:6]}")
    assert particles.min() >= 1.0 and particles.max() <= float(NS)


def test_filter_likelihood_matches_env():
    """The filter must score particles with the ENV's likelihood.

    PITFALLS.md: an env and its filter must stay in sync or belief
    propagation diverges with no error raised. The current update() reduces
    the observation to np.mean(obs_from_env) and applies an unrelated
    Gaussian with a fixed obs_noise_std of 1.0, so it was never in sync.

    Checked behaviourally: after one observation, every particle whose parity
    differs from the observation must carry zero weight, because the env's
    likelihood is exactly zero there.
    """
    module = _filter_module()
    name = next((n for n in ("OddEvenExactSupportParticleFilter",
                              "OddEvenBootstrapParticleFilter",
                              "OddEvenParticleFilter")
                 if hasattr(module, n)), None)
    pf = _build_filter(name)
    observation = 7
    pf.predict(np.array(0))
    pf.update(np.array([float(observation)], dtype=np.float32))
    states = np.asarray(pf.particles, dtype=np.float64).ravel()
    weights = np.asarray(pf.weights, dtype=np.float64)
    wrong_parity = (np.round(states).astype(int) % 2) != (observation % 2)
    if wrong_parity.any():
        assert weights[wrong_parity].max() == 0.0, (
            "particles of the wrong parity kept non-zero weight; the filter "
            "is not using the env's observation likelihood")


def test_filter_rng_seed_is_honoured():
    """Deterministic per-episode PF seeding must be possible.

    PFDictWithWeightsObservationWrapper derives a per-episode rng_seed and
    passes it to the filter. This filter swallows it in **kwargs and uses
    global np.random, so two filters built with the same seed differ. Adding
    that seeding to the Ant-Tag arm changed results between two runs whose
    run_config.json was byte-identical -- which is why run_config.json now
    records git provenance.
    """
    module = _filter_module()
    name = next((n for n in ("OddEvenExactSupportParticleFilter",
                              "OddEvenBootstrapParticleFilter",
                              "OddEvenParticleFilter")
                 if hasattr(module, n)), None)
    first = _build_filter(name, seed=1)
    second = _build_filter(name, seed=1)
    third = _build_filter(name, seed=2)
    assert np.array_equal(first.particles, second.particles), (
        "same rng_seed gave different particles")
    if name != "OddEvenExactSupportParticleFilter":
        # A fixed-support filter is seed-independent at init by design.
        assert not np.array_equal(first.particles, third.particles), (
            "different rng_seed gave identical particles")


def test_static_state_filter_has_no_process_noise():
    """predict() must be a no-op: the hidden state never changes.

    Process noise on a static state destroys accumulated information every
    step and stops the belief from sharpening past the noise floor.
    """
    module = _filter_module()
    name = next((n for n in ("OddEvenExactSupportParticleFilter",
                              "OddEvenBootstrapParticleFilter",
                              "OddEvenParticleFilter")
                 if hasattr(module, n)), None)
    pf = _build_filter(name)
    before = np.array(pf.particles, dtype=np.float64, copy=True)
    for _ in range(10):
        pf.predict(np.array(0))
    assert np.array_equal(before, np.asarray(pf.particles, dtype=np.float64)), (
        "predict() moved the particles; the state is static")


def test_bootstrap_filter_covers_every_state():
    """The true state must always have at least one particle.

    With a static state there is no process noise to re-diversify, so a state
    with no particle can never be recovered. Uniform sampling misses the true
    state in 36.4% of episodes at N=50 and 13.3% at N=100 (ns=50); 100 uniform
    draws over 50 states cover only about 43 of them. Initialization must
    stratify over every state instead.
    """
    module = _filter_module()
    name = next((n for n in ("OddEvenExactSupportParticleFilter",
                              "OddEvenBootstrapParticleFilter")
                 if hasattr(module, n)), None)
    if name is None:
        pytest.fail("neither exact-support nor bootstrap Odd-Even filter exists")
    for seed in range(10):
        pf = _build_filter(name, num_particles=NS, seed=seed)
        covered = set(np.round(
            np.asarray(pf.particles, dtype=np.float64).ravel()).astype(int))
        missing = set(range(1, NS + 1)) - covered
        assert not missing, f"seed {seed}: states with no particle: {sorted(missing)[:8]}"


def test_exact_support_filter_matches_the_env_posterior():
    """The exact-support filter's weights must BE the env's posterior.

    With particles on every state and weights updated by the true likelihood,
    the filter is exact. That arm isolates encoder loss from filter loss,
    which is what makes "does the encoder matter?" answerable here.
    """
    module = _filter_module()
    cls = getattr(module, "OddEvenExactSupportParticleFilter", None)
    if cls is None:
        pytest.fail("OddEvenExactSupportParticleFilter does not exist")
    env = make_env(ns=NS, seed=11)
    initial_obs, initial_info = env.reset(seed=11)
    # The env folds its reset observation into b0, so the filter must too --
    # this is the test that keeps the two in sync, and it is checked at t=0
    # before any step, where a one-update offset would otherwise hide until
    # the beliefs happened to agree.
    pf = cls(num_particles=NS, initial_env_obs=initial_obs,
             n_dist_size=NS, rng_seed=0)
    order0 = np.argsort(np.asarray(pf.particles, dtype=np.float64).ravel())
    np.testing.assert_allclose(
        np.asarray(pf.weights, dtype=np.float64)[order0],
        initial_info["belief"], atol=1e-9)
    for _ in range(15):
        observation = draw_observation(env)
        env.update_belief(observation)
        pf.predict(np.array(0))
        pf.update(np.array([float(observation)], dtype=np.float32))
    order = np.argsort(np.asarray(pf.particles, dtype=np.float64).ravel())
    filter_belief = np.asarray(pf.weights, dtype=np.float64)[order]
    np.testing.assert_allclose(filter_belief, env.belief, atol=1e-9)


def test_filter_weights_are_a_valid_measure():
    """PITFALLS.md section 3: every set must carry positive, finite mass.

    POMDPDataset rejects NaN, inf and zero-total weights at load. A filter
    that emits them fails thousands of steps into an epoch instead, or --
    worse -- feeds a 2831-instead-of-0.157 loss. On this env the belief goes
    near one-hot, so the weights are the extreme case: ESS reaches about 1.0
    of 50 and the smallest weights underflow.
    """
    module = _filter_module()
    name = next((n for n in ("OddEvenExactSupportParticleFilter",
                              "OddEvenBootstrapParticleFilter")
                 if hasattr(module, n)), None)
    if name is None:
        pytest.fail("no correct Odd-Even filter exists yet")
    env = make_env(ns=NS, seed=5)
    env.reset(seed=5)
    pf = _build_filter(name, num_particles=NS, seed=5)
    for _ in range(60):
        observation = draw_observation(env)
        pf.predict(np.array(0))
        pf.update(np.array([float(observation)], dtype=np.float32))
        weights = np.asarray(pf.weights, dtype=np.float64)
        assert np.isfinite(weights).all(), "non-finite weight"
        assert (weights >= 0).all(), "negative weight"
        assert weights.sum() > 0, "zero total mass"
    assert ess(np.asarray(pf.weights, dtype=np.float64)) < 3.0, (
        "the belief should be near one-hot after 60 observations")


def test_bootstrap_filter_resamples_and_stays_a_valid_measure():
    """The bootstrap arm must resample, and survive doing it.

    Resampling is the whole difference between this arm and the exact-support
    one, and on a static state it is irreversible: each resample kills the
    states it does not duplicate, and no process noise brings them back. Pin
    that it happens, that the support really does shrink, and that the
    weights stay a valid measure throughout -- an all-zero weight vector
    after a resample would be an unrecoverable dead filter, not a belief.
    """
    module = _filter_module()
    cls = getattr(module, "OddEvenBootstrapParticleFilter", None)
    if cls is None:
        pytest.fail("OddEvenBootstrapParticleFilter does not exist")
    env = make_env(ns=NS, seed=7)
    initial_obs, _info = env.reset(seed=7)
    pf = cls(num_particles=2 * NS, initial_env_obs=initial_obs,
             n_dist_size=NS, rng_seed=7)
    covered_before = len(set(np.asarray(pf.particles).ravel().astype(int)))
    assert covered_before == NS, (
        "init must cover every state -- folding in the reset observation "
        "must not resample, or the stratified coverage is gone before the "
        "filter has propagated anything")
    for _ in range(40):
        observation = draw_observation(env)
        pf.predict(np.array(0))
        pf.update(np.array([float(observation)], dtype=np.float32))
        weights = np.asarray(pf.weights, dtype=np.float64)
        assert np.isfinite(weights).all() and (weights >= 0).all()
        assert weights.sum() > 0
        assert np.isclose(weights.sum(), 1.0)
    assert pf.resample_count > 0, "the filter never resampled"
    covered_after = len(set(np.asarray(pf.particles).ravel().astype(int)))
    assert covered_after < covered_before, (
        "resampling did not narrow the support")
    assert pf.map_state() == env.true_state or covered_after > 1


def test_bootstrap_filter_surplus_support_follows_rng_seed():
    """Coverage is deterministic; the surplus above it is seeded.

    With num_particles == n_dist_size the init is one particle per state and
    so is seed-independent by construction. Above that the extra particles
    come from the prior, and they must come from the FILTER'S OWN generator:
    the old filter drew from the process-global np.random, so no run using it
    was reproducible.
    """
    module = _filter_module()
    cls = getattr(module, "OddEvenBootstrapParticleFilter", None)
    if cls is None:
        pytest.fail("OddEvenBootstrapParticleFilter does not exist")

    def build(seed, initial_env_obs=None):
        return cls(num_particles=2 * NS, initial_env_obs=initial_env_obs,
                   n_dist_size=NS, rng_seed=seed)

    same = (build(1).particles, build(1).particles)
    assert np.array_equal(*same), "same rng_seed gave different particles"
    assert not np.array_equal(build(1).particles, build(2).particles), (
        "different rng_seed gave identical particles")
    # With no initial observation the PRIOR is exactly uniform whatever the
    # surplus draw was: a duplicated state must not carry double the mass.
    np.testing.assert_allclose(build(3).state_belief(), 1.0 / NS)
    # With one, the starting belief is b0 = P(s | o0) -- the same pooled
    # posterior a manual Bayes update of the uniform prior gives.
    env = make_env(ns=NS, seed=17)
    expected = np.array(
        [env._compute_observation_probability(7, s) for s in range(1, NS + 1)])
    expected = expected / expected.sum()
    pf = build(3, np.array([7.0], dtype=np.float32))
    np.testing.assert_allclose(pf.state_belief(), expected, atol=1e-12)


def test_filter_below_full_coverage_raises():
    """Fewer particles than states must raise, not silently drop states.

    A state with no particle is refuted before the first observation and can
    never be recovered, so a filter that quietly accepts num_particles < n is
    wrong in a way no metric would show.
    """
    module = _filter_module()
    for name in ("OddEvenExactSupportParticleFilter",
                 "OddEvenBootstrapParticleFilter"):
        cls = getattr(module, name)
        with pytest.raises(ValueError, match="n_dist_size"):
            cls(num_particles=NS - 1, initial_env_obs=None,
                n_dist_size=NS, rng_seed=0)


def test_filter_handles_multiple_observations_per_step():
    """update() takes an array of obs_per_step values, not just a scalar.

    Each observation is independent evidence given the state, so k of them
    must fold in as k Bayes updates. Averaging them -- what the old filter
    did -- discards k - 1 observations.
    """
    module = _filter_module()
    cls = getattr(module, "OddEvenExactSupportParticleFilter")
    env = make_env(ns=NS, seed=13, obs_per_step=4)
    env.reset(seed=13)
    batched = cls(num_particles=NS, initial_env_obs=None,
                  n_dist_size=NS, rng_seed=0)
    one_at_a_time = cls(num_particles=NS, initial_env_obs=None,
                        n_dist_size=NS, rng_seed=0)
    for _ in range(5):
        observations = np.array(
            [float(draw_observation(env)) for _ in range(4)],
            dtype=np.float32)
        batched.update(observations)
        for observation in observations:
            one_at_a_time.update(np.array([observation], dtype=np.float32))
        env.update_belief(int(observations[0]))
    np.testing.assert_allclose(batched.weights, one_at_a_time.weights,
                               atol=1e-12)
    assert ess(np.asarray(batched.weights)) < 5.0, (
        "20 observations should have collapsed the belief from 50")


# ---------------------------------------------------------------------------
# Gap 6 -- variant registry
# ---------------------------------------------------------------------------


def test_odd_even_variants_registry_is_consistent():
    """One registry, so env id / filter / cap cannot disagree.

    The Ant-Tag registry replaced 19 wrapper files and made the episode cap
    unreachable-by-default-wrong. Odd-Even needs the same, keyed on
    n_dist_size and obs_per_step rather than visibility curricula.
    """
    variants = _load_experiment_module(_ODD_EVEN_DIR, "variants")
    assert hasattr(variants, "VARIANTS") and variants.VARIANTS
    for name in variants.VARIANTS:
        variant = variants.resolve(name)
        assert variants.episode_cap(name) > 0
        assert variants.run_subdir("cgf", name)
        assert isinstance(variant.env_id, str)
        assert variant.particle_filter is not None


# ---------------------------------------------------------------------------
# Gap 7 -- the generic weighted dict wrapper is buried in an Ant-Tag script
# ---------------------------------------------------------------------------


def test_weighted_dict_wrapper_is_importable_without_ant_tag():
    """The {obs, particles, weights} wrapper has no Ant-Tag logic in it.

    It should sit next to its unweighted sibling in
    set_transformer/rl/wrappers/particle_filter.py so Odd-Even can use it
    without importing MuJoCo and the Ant-Tag envs.
    """
    module = importlib.import_module(
        "set_transformer.rl.wrappers.particle_filter")
    assert hasattr(module, "PFDictWithWeightsObservationWrapper")

    # Gap 7 is only really closed if the Ant-Tag re-export is the SAME object.
    # SB3 pickles the extractor class by module path, and 2074 of the 2788
    # checkpoints under experiments/ant_tag/runs/ resolve theirs through
    # getattr(import_module("4_train_rl_cgf"), name). A shim copy would load
    # but would no longer track fixes to the real class.
    pytest.importorskip("mujoco", reason="Ant-Tag needs MuJoCo")
    ant_tag = _load_experiment_module(_ANT_TAG_DIR, "4_train_rl_cgf")
    assert (ant_tag.PFDictWithWeightsObservationWrapper
            is module.PFDictWithWeightsObservationWrapper)


def test_ant_tag_still_exposes_the_wrapper_name():
    """Guard for the Gap 7 hoist: the Ant-Tag name must keep working.

    SB3 pickles the extractor class into a saved policy and re-imports it on
    load, so moving a symbol without re-exporting it breaks every existing
    checkpoint.
    """
    pytest.importorskip("mujoco", reason="Ant-Tag needs MuJoCo")
    cgf = _load_experiment_module(_ANT_TAG_DIR, "4_train_rl_cgf")
    assert hasattr(cgf, "PFDictWithWeightsObservationWrapper")
    assert hasattr(cgf, "WeightedCGFFeaturesExtractor")


# ---------------------------------------------------------------------------
# Gap 9 -- the RL arms
# ---------------------------------------------------------------------------


def _dict_space(num_particles=NS, particle_dim=1, obs_dim=1):
    import gymnasium as gym
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf,
                                     (num_particles, particle_dim), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (num_particles,), np.float32),
    })


def test_cgf_extractor_accepts_one_dimensional_particles():
    """Odd-Even particles are 1-D. The CGF extractor must handle that.

    t_init_mode="spread" raises unless particle_dim == 2, and
    "linspace_first_dim" writes noise into dimensions 1: which do not exist
    here. This test pins that the default init path works at particle_dim=1
    and produces finite, input-dependent features.
    """
    pytest.importorskip("mujoco", reason="4_train_rl_cgf imports pdomains/MuJoCo")
    import torch
    cgf = _load_experiment_module(_ANT_TAG_DIR, "4_train_rl_cgf")

    extractor = cgf.WeightedCGFFeaturesExtractor(
        _dict_space(), num_cgf_features=64, arena_scale=float(NS))
    torch.manual_seed(0)
    particles = torch.arange(1, NS + 1, dtype=torch.float32).reshape(1, NS, 1)
    peaked = torch.zeros(1, NS)
    peaked[0, 10] = 1.0
    other = torch.zeros(1, NS)
    other[0, 40] = 1.0
    a = extractor({"obs": torch.zeros(1, 1), "particles": particles,
                    "weights": peaked})
    b = extractor({"obs": torch.zeros(1, 1), "particles": particles,
                    "weights": other})
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b), (
        "the CGF features do not depend on which state carries the mass")


def test_cgf_spread_init_supports_one_dimensional_particles():
    """A 1-D spread init is the right default on this domain.

    On Ant-Tag the spread init (8 directions x log-spaced norms) is what put
    the CGF features where the signal already is, so no ~10x growth of ||t||
    was needed. The 1-D analogue is log-spaced magnitudes in BOTH signs.

    It is a SEPARATE mode, "spread_1d", not a widening of "spread". "spread"
    is intrinsically planar: 8 directions from angles 2*pi*k/8, and its
    rho_hi=2.8 is the largest norm the ELEMENTWISE t_clamp=2.0 permits in 2-D
    (the diagonal, 2*sqrt(2)). In 1-D there are two directions and that same
    clamp bounds ||t|| at 2.0, so neither number carries over. Overloading one
    flag name across two unrelated geometries would also change, silently, a
    mode that hundreds of saved 2-D checkpoints were built with -- which is
    why tests/test_ant_tag_shared_pieces_regression.py pins "spread" REJECTING
    particle_dim=1, and why that rejection is asserted again here.

    The extractor is imported from the PACKAGE. It moved there in Task A, so
    loading the Ant-Tag script (and with it MuJoCo) is no longer needed.
    """
    from set_transformer.rl.feature_extractors.cgf import (
        WeightedCGFFeaturesExtractor,
    )
    extractor = WeightedCGFFeaturesExtractor(
        _dict_space(), num_cgf_features=64, arena_scale=float(NS),
        t_init_mode="spread_1d")
    assert extractor.t_values.shape == (64, 1)
    # Both signs, so the CGF sees mass either side of the centred origin.
    values = extractor.t_values.detach().numpy().ravel()
    assert (values > 0).any() and (values < 0).any()

    # The two modes must stay distinct: "spread" still rejects 1-D, and
    # "spread_1d" rejects 2-D.
    with pytest.raises(ValueError, match="2D particles"):
        WeightedCGFFeaturesExtractor(
            _dict_space(), num_cgf_features=64, t_init_mode="spread")
    with pytest.raises(ValueError, match="1D particles"):
        WeightedCGFFeaturesExtractor(
            _dict_space(particle_dim=2), num_cgf_features=64,
            t_init_mode="spread_1d")


def test_base_obs_does_not_leak_the_state():
    """obs_dict["obs"] must carry no information about the hidden state.

    The raw observation reveals the parity outright and locates the state to
    about +/-3.2, so putting it in obs would let a memoryless policy bypass
    the encoder -- the same reason the Ant-Tag arms mask obs[-2:]. The
    recommended content is the normalized step index.
    """
    module = _load_experiment_module(_ODD_EVEN_DIR, "4_train_rl_cgf")
    env = module.make_odd_even_belief_env(num_particles=NS, rank=0, seed=0)()
    obs, info = env.reset(seed=0)
    true_state = info["true_state"]
    base = np.asarray(obs["obs"], dtype=np.float64).ravel()
    # A leak-free base obs cannot be correlated with the hidden state across
    # episodes, and at reset it cannot even mention it.
    assert not np.any(np.isclose(base, float(true_state))), (
        f"base obs {base} contains the hidden state {true_state}")
    env.close()


# ---------------------------------------------------------------------------
# Gaps 8 and 10 -- dataset contract and evaluation protocol
# ---------------------------------------------------------------------------


def test_collected_dataset_contract():
    """Step 2's .npz contract, so 3_train_st.py needs no change.

    particles [S, N, D] float32, weights [S, N] float32, a positive
    particle_scale, and a metadata JSON with git provenance. PITFALLS.md
    section 4: the coordinate scale must match between pretraining and RL, so
    it has to be recorded at collection.
    """
    collect = _load_experiment_module(_ODD_EVEN_DIR,
                                      "2_collect_pf_dataset")
    particles, weights, meta = collect.collect_dataset_for_test(
        ns=NS, num_episodes=4, timesteps=10, num_particles=NS, seed=0)
    assert particles.ndim == 3 and weights.ndim == 2
    assert particles.shape[:2] == weights.shape
    assert particles.dtype == np.float32 and weights.dtype == np.float32
    assert float(meta["particle_scale"]) > 0
    assert "git" in meta


def test_dataset_covers_the_belief_transient():
    """The pretraining set must not be almost all one-hot beliefs.

    The belief locks on by about step 21 of 50, so uniform sampling over an
    episode gives roughly 60% collapsed beliefs -- the regime where every
    encoder is equivalent. Rebalance by step index, not by spread in arena
    units.
    """
    collect = _load_experiment_module(_ODD_EVEN_DIR,
                                      "2_collect_pf_dataset")
    _particles, weights, _meta = collect.collect_dataset_for_test(
        ns=NS, num_episodes=20, timesteps=50, num_particles=NS, seed=0)
    diffuse = np.mean([ess(w) > 3.0 for w in weights])
    assert diffuse > 0.25, (
        f"only {diffuse:.0%} of snapshots are pre-collapse beliefs; the "
        "encoder would be pretrained almost entirely on one-hot sets")


def test_eval_reseeds_after_load():
    """PITFALLS.md section 2: re-seed AFTER PPO.load, and per episode.

    PPO.load restores the TRAINING seed and set_random_seed re-seeds the env
    with it, overriding --seed. On this env the seed is the episode, so the
    consequence is sharper than on Ant-Tag: every eval replays one episode.
    The shared eval script (which eval_true_reward_odd_even.py forwards to
    since change 5.1) must re-seed after the load, and the Odd-Even domain
    must ask for the per-episode re-seeding.
    """
    import set_transformer.rl.eval_true_reward as shared
    from set_transformer.rl.domains.odd_even import ODD_EVEN

    module = _load_experiment_module(
        _ODD_EVEN_DIR / "eval_scripts", "eval_true_reward_odd_even")
    assert module.main is not None and module._shared_main is shared.main
    source = Path(shared.__file__).read_text()
    load_at = source.index("PPO.load(")
    assert "env.seed(args.seed)" in source[load_at:], "no re-seed after PPO.load"
    assert ODD_EVEN.evaluation.reseed_per_episode is True, (
        "the Odd-Even eval does not vary the seed per episode")
    rollout_src = source[source.index("def rollout("):source.index("def report_success_rate(")]
    assert "env.seed(seed + index)" in rollout_src


def test_metrics_split_transient_steady():
    """Report transient and steady state separately, never only pooled.

    About 96% of a 500-step rollout is one-hot belief, and the oracle's own
    per-step reward moves from -1.098 to -0.582 between a 50- and a 100-step
    cap purely from transient dilution. A pooled mean therefore hides every
    encoder difference and is not comparable across caps.
    """
    module = _load_experiment_module(
        _ODD_EVEN_DIR / "eval_scripts", "eval_true_reward_odd_even")
    metrics = module.summarize_episode(
        rewards=np.arange(50, dtype=np.float64),
        collapse_step=COLLAPSE_STEP)
    for key in ("transient", "steady", "pooled"):
        assert key in metrics, f"missing {key} in {sorted(metrics)}"
