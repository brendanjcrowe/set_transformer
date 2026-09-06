"""Tests for the built Odd-Even pipeline: env factory, arms, eval, CGF init.

Companion to test_odd_even_pomdp_contract.py, which stays the GAP TRACKER (a
gap arrives there as an xfail and loses its marker when it closes). This file
tests the pipeline that closing those gaps produced, under
set_transformer/experiments/odd_even/.

What is covered, and why each item is here rather than trusted:

* **The observation split, in BOTH directions.** obs_dict["obs"] must carry
  no information about the hidden state, AND the particle filter must still
  receive the real observation. Either failure is silent at runtime: a leak
  lets a memoryless policy bypass the encoder and makes every arm score the
  same; a starved filter holds the uniform prior and produces a
  plausible-looking belief. This is the single most damaging mistake available
  in this pipeline, so both halves are asserted.
* **PITFALLS.md section 1**, for the ST arm. SB3's ActorCriticPolicy._build
  re-initializes every Linear in the whole policy, features extractor
  included, a moment after the extractor's __init__ loaded a pretrained
  encoder. It cost two 6M-step runs, and the logs looked correct throughout.
  Modelled on tests/test_st_pretrained_load.py.
* **A short end-to-end PPO run for each of the three arms**, so an arm cannot
  be broken by an import, a space mismatch or a policy_kwargs typo without a
  test failing.
* **The 1-D CGF spread init.** A separate mode from the 2-D "spread"; the
  Ant-Tag regression gate pins that "spread" keeps REJECTING 1-D particles.

Module loading follows test_odd_even_pomdp_contract.py: BY EXPLICIT PATH,
under a directory-qualified key, with sys.path restored afterwards.
experiments/odd_even/ and experiments/ant_tag/ both hold 4_train_rl_cgf.py,
2_collect_pf_dataset.py and variants.py, so a flat-name import picks whichever
directory is first on sys.path -- which has already silently pointed three
tests at the wrong file.
"""

import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ST_ROOT = Path(__file__).resolve().parents[1]
_ODD_EVEN_DIR = _ST_ROOT / "experiments" / "odd_even"
# Only the package roots go on sys.path. The experiment directories must not.
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("pdomains")
pytest.importorskip("stable_baselines3")

import torch  # noqa: E402

#: The main variant: n=50 on a 50-step cap, where the transient is 42% of the
#: episode. Its state range half-width is 24.5 and its centre 25.5.
VARIANT = "oe50"
NS = 50


@contextmanager
def _sys_path(*directories):
    """Temporarily prepend directories, then restore sys.path exactly."""
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


def _odd_even_sibling():
    """The pipeline's OWN sibling loader, experiments/odd_even/_sibling.py.

    Tests load the pipeline modules through the same loader the scripts use,
    rather than through a private spec_from_file_location key. Two reasons:

    * The module identities then match what a run really produces. A private
      key gave the classes a module name no fresh process can import, and
      SubprocVecEnv cloudpickles the env factory BY REFERENCE -- so the test
      failed in a way the scripts did not, and would have masked the reverse.
    * There is one definition of how the Gap 12 collision is avoided.
    """
    key = "_oe_pipe_sibling_loader"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    path = _ODD_EVEN_DIR / "_sibling.py"
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def _load(name, directory=_ODD_EVEN_DIR):
    """Import experiments/<directory>/<name>.py under an unambiguous key.

    Odd-Even modules go through their own _sibling loader (see above).
    Anything else -- e.g. eval_scripts/ -- is loaded by explicit path, with
    the directory on sys.path only while the module executes so a
    digit-leading sibling import inside it can resolve.
    """
    directory = Path(directory)
    if directory == _ODD_EVEN_DIR:
        # The odd_even directory must be importable for `import _sibling`
        # inside those modules; the loader itself never adds a SIBLING
        # experiment directory.
        with _sys_path(_ODD_EVEN_DIR):
            return _odd_even_sibling().load(name)
    path = directory / f"{name}.py"
    if not path.exists():
        raise ModuleNotFoundError(f"{path} does not exist")
    key = f"_oe_pipe_{directory.name}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    with _sys_path(directory, _ODD_EVEN_DIR):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def belief_env_module():
    return _load("odd_even_belief_env")


@pytest.fixture(scope="module")
def variants_module():
    return _load("variants")


# ---------------------------------------------------------------------------
# The observation split -- both directions
# ---------------------------------------------------------------------------


def test_base_obs_is_the_step_index_not_the_observation(belief_env_module):
    """Direction 1: the agent is NOT handed the answer.

    The raw observation reveals s*'s parity outright (the model only emits
    integers of s*'s own parity) and locates the state to about +/-3.2 at
    n=50. Putting it in obs_dict["obs"] would let a memoryless policy bypass
    the belief encoder entirely and make every arm score the same.

    Three assertions, because "does not equal the state" alone is too weak:
    the value must be the step index specifically, it must not be ANY emitted
    observation, and it must follow the same trajectory in every episode
    regardless of the hidden state -- which is what "leaks nothing" means.
    """
    env = belief_env_module.make_odd_even_belief_env(
        num_particles=NS, variant=VARIANT, seed=0)()
    cap = 50

    trajectories = []
    for episode in range(4):
        obs, info = env.reset(seed=100 + episode)
        emitted = {float(o) for o in np.asarray(info["observations"]).ravel()}
        base = np.asarray(obs["obs"], dtype=np.float64).ravel()
        assert base.shape == (1,)
        assert base[0] == pytest.approx(0.0), (
            f"step index at reset should be 0, got {base[0]}")
        assert not (set(base.tolist()) & emitted), (
            f"base obs {base} contains an emitted observation {emitted}")

        row = [base[0]]
        for step in range(6):
            obs, _r, _term, _trunc, info = env.step(env.action_space.sample())
            base = np.asarray(obs["obs"], dtype=np.float64).ravel()
            emitted = {float(o)
                       for o in np.asarray(info["observations"]).ravel()}
            assert base[0] == pytest.approx((step + 1) / cap), (
                "base obs is not the normalized step index")
            assert not (set(base.tolist()) & emitted), (
                f"base obs {base} leaked observation {emitted}")
            assert base[0] != pytest.approx(float(info["true_state"])), (
                "base obs equals the hidden state")
            row.append(base[0])
        trajectories.append(row)
    env.close()

    # Identical across episodes with different hidden states: the base
    # observation is a function of the step count alone.
    for row in trajectories[1:]:
        assert row == trajectories[0], (
            "the base observation differs between episodes, so it carries "
            "episode-specific (i.e. state-dependent) information")


def test_filter_still_receives_the_real_observation(belief_env_module):
    """Direction 2: the filter is NOT starved.

    The wrapper hands the base observation to pf_interaction_mapper, and by
    then that observation is the step index -- so the mapper reads
    info["observations"] instead. If that route broke, the filter would keep
    the uniform prior for the whole episode and hand every encoder a
    perfectly plausible flat belief with nothing raised.

    The check is against the env's OWN exact posterior, which is the strongest
    available: the exact-support filter's weights must equal env.belief. A
    prior-holding filter fails immediately (its weights stay at 1/n), and so
    does a filter fed the wrong observations.
    """
    env = belief_env_module.make_odd_even_belief_env(
        num_particles=NS, variant=VARIANT, seed=0)()
    obs, info = env.reset(seed=7)

    # b0 = P(s | o0). A filter that ignored the reset observation would sit
    # exactly one update behind for the whole episode.
    weights = np.asarray(obs["weights"], dtype=np.float64)
    assert np.abs(weights - info["belief"]).max() < 1e-6, (
        "the filter's belief does not match the env's posterior at b0; it "
        "was not given the reset observation")
    assert np.abs(weights - 1.0 / NS).max() > 1e-3, (
        "the filter is still holding the uniform prior, so it received no "
        "evidence at all")

    for _step in range(12):
        obs, _r, _term, _trunc, info = env.step(env.action_space.sample())
        weights = np.asarray(obs["weights"], dtype=np.float64)
        assert np.abs(weights - info["belief"]).max() < 1e-5, (
            "the filter and the env disagree about the posterior")

    # After 13 observations the exact posterior has sharpened hard. A filter
    # that quietly held the prior would still read ESS == NS here.
    ess = 1.0 / np.sum(weights ** 2)
    assert ess < 5.0, (
        f"effective sample size {ess:.2f} of {NS} after 13 observations; the "
        "belief is not accumulating evidence")
    env.close()


def test_particles_are_centred_on_the_state_range(belief_env_module,
                                                   variants_module):
    """The particles must land in about [-1, 1] after the extractor divides.

    PITFALLS.md section 4: pretraining and RL must use the same coordinate
    scale, and here both halves of (s - centre) / scale have to agree. The
    centring happens in the env and the division in the extractor, so a
    mismatch shows up as particles the encoder never sees at the right size.
    """
    env = belief_env_module.make_odd_even_belief_env(
        num_particles=NS, variant=VARIANT, seed=0)()
    obs, _info = env.reset(seed=1)
    particles = np.asarray(obs["particles"], dtype=np.float64).ravel()
    scale = variants_module.state_scale(VARIANT)
    centre = variants_module.state_centre(VARIANT)
    assert centre == pytest.approx((NS + 1) / 2)
    assert scale == pytest.approx((NS - 1) / 2)
    # Raw states recovered by undoing the centring.
    raw = np.sort(particles + centre)
    np.testing.assert_allclose(raw, np.arange(1, NS + 1), atol=1e-4)
    normalized = particles / scale
    assert normalized.min() == pytest.approx(-1.0, abs=1e-4)
    assert normalized.max() == pytest.approx(1.0, abs=1e-4)
    env.close()


def test_weights_are_a_probability_vector(belief_env_module):
    """Every arm reads the weights directly, so they must stay a measure.

    This is why VecNormalize uses norm_obs_keys=["obs"]: normalizing the
    weights to zero mean and unit variance would leave negative "probabilities"
    summing to nothing.
    """
    env = belief_env_module.make_odd_even_belief_env(
        num_particles=NS, variant=VARIANT, seed=0)()
    obs, _info = env.reset(seed=2)
    for _step in range(5):
        weights = np.asarray(obs["weights"], dtype=np.float64)
        assert (weights >= 0).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-5)
        obs, _r, _t, _tr, _i = env.step(env.action_space.sample())
    env.close()


def test_vec_normalize_leaves_the_weights_alone(belief_env_module):
    """The whole reason norm_obs_keys is passed. Asserted, not assumed."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([belief_env_module.make_odd_even_belief_env(
        num_particles=NS, variant=VARIANT, seed=0)])
    venv = belief_env_module.make_vec_normalize(
        venv, training=True, norm_reward=True)
    obs = venv.reset()
    for _step in range(20):
        obs, _r, _d, _i = venv.step(np.array([0]))
        weights = np.asarray(obs["weights"][0], dtype=np.float64)
        assert (weights >= 0).all(), "VecNormalize corrupted the PF weights"
        assert weights.sum() == pytest.approx(1.0, abs=1e-4)
    venv.close()


def test_per_episode_pf_seed_makes_runs_reproducible(belief_env_module):
    """PITFALLS.md section 2: the PF seed is derived per episode.

    Two envs built with the same worker seed must produce identical belief
    trajectories, and two different worker seeds must not collide. Without
    this an "identical" rerun is not reproducible, which already made two
    Ant-Tag runs uncomparable.
    """
    def trajectory(seed, rank=0):
        env = belief_env_module.make_odd_even_belief_env(
            num_particles=NS + 7, variant=VARIANT, seed=seed, rank=rank)()
        obs, _info = env.reset(seed=11)
        rows = [np.asarray(obs["particles"]).ravel().copy()]
        for _step in range(3):
            obs, _r, _t, _tr, _i = env.step(0)
            rows.append(np.asarray(obs["particles"]).ravel().copy())
        env.close()
        return np.stack(rows)

    np.testing.assert_array_equal(trajectory(5), trajectory(5))


# ---------------------------------------------------------------------------
# The 1-D CGF spread init
# ---------------------------------------------------------------------------


def _dict_space(num_particles=NS, particle_dim=1, obs_dim=1):
    import gymnasium as gym
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf,
                                     (num_particles, particle_dim), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (num_particles,), np.float32),
    })


def test_cgf_spread_1d_features_are_finite_and_input_dependent():
    """spread_1d must produce usable features, not just the right shape.

    A t that is finite but degenerate (all magnitudes equal, or one sign
    only) passes a shape assertion and carries almost nothing. So this checks
    the forward pass: two beliefs whose mass sits on different states must
    give different features, and nothing may be NaN or inf under the log.
    """
    from set_transformer.rl.feature_extractors.cgf import (
        WeightedCGFFeaturesExtractor,
    )
    extractor = WeightedCGFFeaturesExtractor(
        _dict_space(), num_cgf_features=64, arena_scale=(NS - 1) / 2,
        t_init_mode="spread_1d")

    t = extractor.t_values.detach()
    assert t.shape == (64, 1)
    assert torch.isfinite(t).all()
    # Log-spaced magnitudes in both signs: 32 distinct magnitudes, mirrored.
    magnitudes = torch.unique(t.abs().round(decimals=6))
    assert magnitudes.numel() == 32, (
        f"expected 32 distinct magnitudes, got {magnitudes.numel()}")
    assert float(t.max()) == pytest.approx(2.0), (
        "the largest magnitude should be 2.0, the bound the ELEMENTWISE "
        "t_clamp permits in 1-D")
    assert float(t.min()) == pytest.approx(-2.0)

    # Centred particles, as the env produces them.
    centre = (NS + 1) / 2
    particles = (torch.arange(1, NS + 1, dtype=torch.float32) - centre
                 ).reshape(1, NS, 1)
    features = []
    for state_index in (2, 24, 47):
        weights = torch.zeros(1, NS)
        weights[0, state_index] = 1.0
        out = extractor({"obs": torch.zeros(1, 1), "particles": particles,
                         "weights": weights})
        assert torch.isfinite(out).all()
        features.append(out)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            assert not torch.allclose(features[i], features[j]), (
                "the CGF features do not depend on which state holds the mass")

    # And a diffuse belief must differ from a peaked one at the same mean.
    diffuse = torch.full((1, NS), 1.0 / NS)
    peaked = torch.zeros(1, NS)
    peaked[0, NS // 2] = 1.0
    out_diffuse = extractor({"obs": torch.zeros(1, 1), "particles": particles,
                             "weights": diffuse})
    out_peaked = extractor({"obs": torch.zeros(1, 1), "particles": particles,
                            "weights": peaked})
    assert not torch.allclose(out_diffuse, out_peaked), (
        "the CGF cannot tell a uniform belief from a peaked one at the same "
        "mean, which is the one thing a CGF is supposed to do")


def test_cgf_spread_2d_is_unchanged_by_the_1d_addition():
    """The 2-D "spread" geometry must be byte-identical after the change.

    tests/test_ant_tag_shared_pieces_regression.py holds the authoritative
    goldens. This is the cheap local restatement: 8 planar directions, 8
    log-spaced norms, rho_hi 2.8.
    """
    from set_transformer.rl.feature_extractors.cgf import (
        WeightedCGFFeaturesExtractor,
    )
    extractor = WeightedCGFFeaturesExtractor(
        _dict_space(particle_dim=2), num_cgf_features=64, arena_scale=4.5,
        t_init_mode="spread")
    t = extractor.t_values.detach()
    assert t.shape == (64, 2)
    norms = torch.linalg.norm(t, dim=1)
    assert float(norms.max()) == pytest.approx(2.8, abs=1e-5)
    assert float(norms.min()) == pytest.approx(0.25, abs=1e-5)
    assert torch.unique(norms.round(decimals=5)).numel() == 8


# ---------------------------------------------------------------------------
# PITFALLS.md section 1 -- the ST arm's pretrained encoder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def st_pieces(tmp_path_factory):
    """A tiny pretrained checkpoint plus the vec env the ST arm builds."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    from set_transformer.models.pf_set_transformer import PFSetTransformer

    st_module = _load("4_train_rl_st")

    # A checkpoint in 3_train_st.py's format: encoder input is the coordinate
    # dim plus the weight channel, so 1 + 1 = 2 here.
    torch.manual_seed(0)
    pf_st = PFSetTransformer(
        num_particles=NS, dim_particles=2, num_encodings=8, dim_encoder=8,
        num_inds=32, dim_hidden=128, num_heads=4, ln=True,
        dim_output_particles=1)
    for param in pf_st.parameters():   # make every weight distinctive
        with torch.no_grad():
            param.add_(torch.randn_like(param) * 0.1)
    path = tmp_path_factory.mktemp("oe_ck") / "checkpoint_best.pt"
    torch.save({"model_state_dict": pf_st.state_dict()}, path)

    venv = DummyVecEnv([st_module.make_odd_even_belief_env(
        num_particles=NS, rank=0, seed=0, variant=VARIANT)])
    reference = {key[len("set_transformer."):]: value.clone()
                 for key, value in pf_st.state_dict().items()
                 if key.startswith("set_transformer.")}
    return st_module, venv, str(path), reference


def _encoder_delta(model, reference):
    prefix = "features_extractor.encoder."
    live = {key[len(prefix):]: value
            for key, value in model.policy.state_dict().items()
            if key.startswith(prefix)}
    assert live, "policy has no features_extractor.encoder.* parameters"
    return max(float((reference[key] - live[key].cpu()).abs().max())
               for key in reference if key in live)


def _build_st(st_module, venv, path, frozen):
    from stable_baselines3 import PPO

    model = PPO(
        "MultiInputPolicy", venv,
        policy_kwargs=dict(
            features_extractor_class=st_module.SetTransformerFeaturesExtractor,
            features_extractor_kwargs=dict(
                num_encodings=8, dim_encoder=8, num_inds=32, dim_hidden=128,
                num_heads=4, ln=True, arena_scale=(NS - 1) / 2,
                weight_channel=True, pretrained_st_model_path=path,
                st_frozen=frozen)),
        n_steps=64, batch_size=32, device="cpu", verbose=0, seed=0)
    # Exactly what the arm does after construction.
    st_module.reload_pretrained_encoder(model, path, frozen)
    return model


@pytest.mark.parametrize("frozen", [True, False])
def test_encoder_matches_checkpoint_after_ppo_construction(st_pieces, frozen):
    """PITFALLS.md section 1. Cost when missed: two 6M-step runs.

    SB3's ActorCriticPolicy._build ends with
    self.apply(partial(self.init_weights, ...)), which walks the WHOLE policy
    including the features extractor and re-initializes every Linear. The
    extractor's __init__ loaded the encoder a moment earlier, so without a
    reload after PPO(...) the run trains on orthogonal-init noise while the
    log says "loaded encoder ... FROZEN".
    """
    st_module, venv, path, reference = st_pieces
    model = _build_st(st_module, venv, path, frozen)
    assert _encoder_delta(model, reference) == 0.0, (
        "PPO construction overwrote the pretrained encoder; the reload after "
        "_build is missing or ineffective")


def test_reload_blanks_the_pretrained_path(st_pieces):
    """PITFALLS.md section 7: no absolute paths in policy_kwargs.

    SB3 pickles policy_kwargs into the saved zip and re-runs the extractor
    constructor on load, so a saved policy would die with FileNotFoundError
    once the pretraining directory moved -- even though the trained weights
    are in the zip.
    """
    st_module, venv, path, _reference = st_pieces
    model = _build_st(st_module, venv, path, frozen=False)
    assert model.policy_kwargs["features_extractor_kwargs"][
        "pretrained_st_model_path"] is None


def test_frozen_encoder_does_not_move_during_learn(st_pieces):
    st_module, venv, path, reference = st_pieces
    model = _build_st(st_module, venv, path, frozen=True)
    assert not any(
        param.requires_grad
        for param in model.policy.features_extractor.encoder.parameters())
    model.learn(total_timesteps=128)
    assert _encoder_delta(model, reference) == 0.0, "frozen encoder changed"


def test_unfrozen_encoder_does_move_during_learn(st_pieces):
    """The counterpart: without --st_frozen the encoder must really train,
    or the finetune arm would silently be the frozen arm."""
    st_module, venv, path, reference = st_pieces
    model = _build_st(st_module, venv, path, frozen=False)
    assert all(
        param.requires_grad
        for param in model.policy.features_extractor.encoder.parameters())
    model.learn(total_timesteps=128)
    assert _encoder_delta(model, reference) > 0.0, "encoder did not train"


def test_encoder_mismatch_verification_fails_loudly(st_pieces):
    """The assertion must actually fire when the encoder does NOT match.

    A verification that cannot fail is worse than none: it reads as positive
    evidence. So this deliberately perturbs the encoder and requires the
    check to raise.
    """
    st_module, venv, path, _reference = st_pieces
    model = _build_st(st_module, venv, path, frozen=False)
    with torch.no_grad():
        next(iter(model.policy.features_extractor.encoder.parameters())).add_(1.0)
    with pytest.raises(AssertionError, match="does not match"):
        st_module.assert_encoder_matches_checkpoint(model, path)


# ---------------------------------------------------------------------------
# The ST collapse sentinel
# ---------------------------------------------------------------------------


def _st_features_on_real_beliefs(belief_env_module, pretrained_path=None,
                                  kill_encoder=False, num_episodes=120,
                                  step=8):
    """Encode real exact-support beliefs and return the feature matrix.

    Real beliefs, not random weight vectors: the sentinel is a claim about
    what this encoder does on this domain's posteriors, whose effective
    sample size is about 2 of 50 by step 8. A synthetic uniform-random weight
    vector is a different and much easier input distribution.
    """
    import gymnasium as gym

    from set_transformer.rl.feature_extractors.st import (
        SetTransformerFeaturesExtractor,
    )

    env = belief_env_module.make_odd_even_belief_env(
        num_particles=NS, variant=VARIANT, seed=0)()
    particles, weights = [], []
    for episode in range(num_episodes):
        obs, _info = env.reset(seed=9000 + episode)
        for _t in range(step):
            obs, _r, _term, _trunc, _info = env.step(0)
        particles.append(obs["particles"])
        weights.append(obs["weights"])
    env.close()

    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (NS, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (NS,), np.float32),
    })
    torch.manual_seed(0)
    extractor = SetTransformerFeaturesExtractor(
        space, num_encodings=8, dim_encoder=8, arena_scale=(NS - 1) / 2,
        weight_channel=True, pretrained_st_model_path=pretrained_path)
    if kill_encoder:
        # Zero the encoder's LAST Linear (weight and bias), which makes the
        # encoding exactly constant regardless of the input. This is the
        # deliberately-dead control: without it the test would only show
        # that the sentinel reports SOME number, not that it discriminates.
        linears = [module for module in extractor.encoder.modules()
                   if isinstance(module, torch.nn.Linear)]
        with torch.no_grad():
            linears[-1].weight.zero_()
            linears[-1].bias.zero_()

    batch = {
        "obs": torch.zeros(len(weights), 1),
        "particles": torch.tensor(np.array(particles), dtype=torch.float32),
        "weights": torch.tensor(np.array(weights), dtype=torch.float32),
    }
    with torch.no_grad():
        output = extractor(batch)
    # Drop the base-obs passthrough, exactly as the callback's hook does.
    return output[:, 1:].numpy()


def test_sentinel_separates_a_live_encoder_from_a_dead_one(belief_env_module):
    """The sentinel must discriminate, not merely produce a number.

    Both halves matter. A sentinel that never fires is decorative; one that
    fires on a healthy encoder would have us abort a good run -- which is the
    live risk here, because a healthy Odd-Even encoder's ABSOLUTE feature
    spread can sit below PITFALLS.md section 6's ~0.01 Ant-Tag threshold
    while the state stays fully decodable.

    The dead control zeroes the encoder's final Linear, so the encoding is
    exactly constant and the relative spread is exactly 0.
    """
    sentinel = _load("st_feature_sentinel")

    live = _st_features_on_real_beliefs(belief_env_module)
    dead = _st_features_on_real_beliefs(belief_env_module, kill_encoder=True)

    live_spread = sentinel.relative_feature_spread(live)
    dead_spread = sentinel.relative_feature_spread(dead)

    assert dead_spread < sentinel.COLLAPSE_RELATIVE_SPREAD, (
        f"the deliberately constant encoder was not flagged "
        f"({dead_spread:.2e})")
    assert live_spread > sentinel.COLLAPSE_RELATIVE_SPREAD, (
        f"a randomly initialized encoder on real beliefs was flagged as "
        f"collapsed ({live_spread:.2e}); the sentinel would abort a healthy "
        "run")
    # An order of magnitude of clear air between the two, which is what makes
    # the threshold a judgement rather than a coin toss.
    assert live_spread > 10 * max(dead_spread, 1e-12)

    # And the dead encoding really is constant, so the control is the control.
    assert np.allclose(dead, dead[0:1], atol=1e-6)


def test_sentinel_needs_no_absolute_scale_assumption(belief_env_module):
    """Scaling every feature by a constant must not change the sentinel.

    This is the property `feat_std_mean` lacks and the reason the relative
    statistic exists: on this domain the ST feature scale moved over three
    orders of magnitude with pretraining length while the encoding stayed
    informative, so any absolute threshold reads a healthy encoder as dead
    at one scale and a dead one as healthy at another.
    """
    sentinel = _load("st_feature_sentinel")
    features = _st_features_on_real_beliefs(belief_env_module,
                                             num_episodes=60)
    base = sentinel.relative_feature_spread(features)
    for factor in (1e-4, 1e3):
        scaled = sentinel.relative_feature_spread(features * factor)
        assert scaled == pytest.approx(base, rel=1e-3), (
            f"the sentinel moved from {base:.4e} to {scaled:.4e} under a "
            f"pure {factor:g}x rescaling, so it is not scale-free")


def test_sentinel_refuses_a_one_sample_batch():
    """A std over one sample is undefined and must not be logged as a number.

    The shared callback computes its statistic from the last cached forward
    batch, so at n_envs=1 it logs NaN -- the sentinel silently vanishes in
    the cheapest configuration. This raises instead, and the callback
    aggregates over the whole rollout so the situation does not arise.
    """
    sentinel = _load("st_feature_sentinel")
    with pytest.raises(ValueError, match="at least 2"):
        sentinel.relative_feature_spread(np.zeros((1, 64)))


def test_sentinel_callback_logs_over_the_whole_rollout(tmp_path):
    """The callback must report many samples at n_envs=1, and no NaN.

    n_envs=1 is the configuration that breaks the shared callback, so it is
    the one worth pinning: with n_steps=128 the sentinel must see far more
    than one feature sample and every logged statistic must be finite.
    """
    from set_transformer.rl.feature_extractors.st import (
        SetTransformerFeaturesExtractor,
    )

    cgf = _load("4_train_rl_cgf")
    sentinel_module = _load("st_feature_sentinel")
    callback = sentinel_module.OddEvenSTFeatureSentinel()

    cgf.train_odd_even(
        policy_kwargs={
            "features_extractor_class": SetTransformerFeaturesExtractor,
            "features_extractor_kwargs": dict(
                num_encodings=2, dim_encoder=4, num_inds=4, dim_hidden=16,
                num_heads=2, ln=True, arena_scale=(NS - 1) / 2,
                weight_channel=True),
        },
        encoder="st", variant=VARIANT, total_timesteps=256, n_envs=1,
        ppo_n_steps=128, batch_size=32, n_epochs=2, num_particles=NS,
        device="cpu", seed=0, run_subdir="test_sentinel",
        log_dir=str(tmp_path / "logs") + "/",
        model_save_path=str(tmp_path / "models" / "st_agent.zip"),
        eval_freq=10**9, save_freq=10**9, n_eval_episodes=1,
        extra_callbacks=[callback],
    )

    # Read what the callback recorded, not the logger: SB3 clears
    # name_to_value on each dump, so by training end it is empty.
    recorded = callback.last_stats
    # EXACTLY the rollout: n_steps * n_envs = 128. With n_epochs=2 the old
    # forward hook also swallowed 2 x 128 training-minibatch forwards (and
    # the eval episode), reporting ~384 "rollout" samples of which two thirds
    # were re-forwards of the previous rollout mid-update.
    assert recorded["st/feat_samples"] == 128, recorded["st/feat_samples"]
    for key in ("st/feat_std_relative", "st/feat_std_relative_max",
                "st/feat_std_mean_rollout", "st/feat_std_max", "st/feat_abs_mean"):
        assert np.isfinite(recorded[key]), f"{key} is not finite"
    assert "st/feat_std_mean" not in recorded, (
        "the sentinel must not reuse the shared callback's key; SB3's logger "
        "is last-write-wins and this callback runs second")


# ---------------------------------------------------------------------------
# End-to-end: a short PPO run for each arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm,encoder", [
    ("4_train_rl_cgf", "cgf"),
    ("4_train_rl_st", "st"),
    ("4_train_rl_gaussian", "gaussian"),
])
def test_arm_trains_end_to_end(tmp_path, arm, encoder):
    """Each arm must build its env, its policy and complete learn().

    Tiny by design (one worker, a few hundred steps, cpu): this catches an
    import error, a features-extractor / observation-space mismatch or a
    policy_kwargs typo, none of which a unit test on the extractor alone
    would see. It is not a check on learning.
    """
    module = _load(arm)
    cgf = _load("4_train_rl_cgf")

    if encoder == "cgf":
        from set_transformer.rl.feature_extractors.cgf import (
            WeightedCGFFeaturesExtractor,
        )
        policy_kwargs = {
            "features_extractor_class": WeightedCGFFeaturesExtractor,
            "features_extractor_kwargs": dict(
                num_cgf_features=8, arena_scale=(NS - 1) / 2,
                t_init_mode="spread_1d"),
        }
    elif encoder == "st":
        from set_transformer.rl.feature_extractors.st import (
            SetTransformerFeaturesExtractor,
        )
        policy_kwargs = {
            "features_extractor_class": SetTransformerFeaturesExtractor,
            "features_extractor_kwargs": dict(
                num_encodings=2, dim_encoder=4, num_inds=4, dim_hidden=16,
                num_heads=2, ln=True, arena_scale=(NS - 1) / 2,
                weight_channel=True),
        }
    else:
        from set_transformer.rl.feature_extractors.gaussian import (
            WeightedGaussianFeaturesExtractor,
        )
        policy_kwargs = {
            "features_extractor_class": WeightedGaussianFeaturesExtractor,
            "features_extractor_kwargs": dict(arena_scale=(NS - 1) / 2),
        }

    model_path = tmp_path / "models" / f"{encoder}_agent.zip"
    model = cgf.train_odd_even(
        policy_kwargs=policy_kwargs,
        encoder=encoder,
        variant=VARIANT,
        total_timesteps=256,
        n_envs=1,
        ppo_n_steps=128,
        batch_size=32,
        n_epochs=1,
        num_particles=NS,
        device="cpu",
        seed=0,
        run_subdir=f"test_{encoder}",
        log_dir=str(tmp_path / "logs") + "/",
        model_save_path=str(model_path),
        eval_freq=10**9,
        save_freq=10**9,
        n_eval_episodes=1,
    )
    assert model_path.exists(), f"{arm} did not save a model"
    assert (tmp_path / "models" / "vecnormalize.pkl").exists(), (
        "VecNormalize statistics were not saved; an eval could not reproduce "
        "the policy's input distribution")
    # The features_dim must be the base obs (the 1-D step index) plus the
    # encoder's own width, or the arms are not reading the same observation.
    obs_dim = 1
    widths = {"cgf": 8, "st": 2 * 4, "gaussian": 2}
    assert model.policy.features_extractor.features_dim == obs_dim + widths[encoder]


def test_subproc_vec_env_can_pickle_the_env_factory(belief_env_module):
    """n_envs > 1 uses SubprocVecEnv, which CLOUDPICKLES the env factory.

    This has a failure mode invisible at n_envs=1, which is the default in
    every quick smoke run: cloudpickle serializes a class defined in an
    importable module BY REFERENCE, so the child process re-imports the
    module holding StepIndexObservationWrapper. Loading the siblings under a
    bare spec_from_file_location key made that import fail in the child with
    `PicklingError: ... import of module ... failed` -- and only once someone
    ran the real default n_envs=4.

    n_envs=4 is the training default, so this is the configuration that
    matters. The test is a construct-and-step, not a train.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    env_fns = [belief_env_module.make_odd_even_belief_env(
        num_particles=NS, rank=rank, seed=0, variant=VARIANT)
        for rank in range(2)]
    venv = SubprocVecEnv(env_fns)
    try:
        obs = venv.reset()
        assert set(obs) == {"obs", "particles", "weights"}
        obs, rewards, _dones, _infos = venv.step(np.array([0, 1]))
        assert obs["particles"].shape == (2, NS, 1)
        # The filter must be live in the child, not holding the prior.
        for row in obs["weights"]:
            assert np.abs(np.asarray(row, dtype=np.float64) - 1.0 / NS).max() > 1e-3
    finally:
        venv.close()


def test_run_subdir_is_honoured(tmp_path):
    """train_odd_even must write where it is told.

    The Ant-Tag train_* functions accept a run_subdir and then always call
    their own _default_run_dir(seed), so a programmatic call silently lands
    in the base variant's directory. That wart is deliberately not
    reproduced, so it is worth a test.
    """
    cgf = _load("4_train_rl_cgf")
    run_dir = cgf._default_run_dir(3, "odd_even_cgf_oe50", run_tag="tag one")
    assert run_dir.startswith("runs/odd_even_cgf_oe50/")
    assert run_dir.endswith("_seed3_tag_one"), run_dir


# ---------------------------------------------------------------------------
# Registry and eval-script contracts
# ---------------------------------------------------------------------------


def test_episode_cap_comes_from_the_registration(variants_module):
    """PITFALLS.md section 5. The two n=50 variants differ ONLY in the cap."""
    import gymnasium as gym

    assert variants_module.episode_cap("oe50") == 50
    assert variants_module.episode_cap("oe50_long") == 200
    for name, variant in variants_module.VARIANTS.items():
        spec = gym.spec(variant.env_id)
        assert variants_module.episode_cap(name) == spec.max_episode_steps
        # The registry's n_dist_size must match the registration's, or the
        # filter's likelihood table is for a different POMDP.
        assert spec.kwargs["n_dist_size"] == variant.n_dist_size
        # obs_per_step MUST be 1: at 100 draws per step the mean of a single
        # step's observations pins the state and there is no belief to build.
        assert spec.kwargs["obs_per_step"] == 1


def test_run_subdir_names_are_distinct(variants_module):
    names = {variants_module.run_subdir(encoder, variant)
             for encoder in ("cgf", "st", "gaussian")
             for variant in variants_module.VARIANTS}
    assert len(names) == 3 * len(variants_module.VARIANTS)


def test_registry_rejects_unknown_names(variants_module):
    with pytest.raises(ValueError, match="Unknown variant"):
        variants_module.resolve("oe999")
    with pytest.raises(ValueError, match="Unknown particle filter"):
        variants_module.resolve_particle_filter("oe50", "not_a_filter")


def test_summarize_episode_splits_are_disjoint_and_complete():
    """The split must partition the episode, and an empty steady segment
    must be NaN rather than 0.0 -- averaging a missing segment as zero would
    pull a very negative mean toward the oracle and read as improvement."""
    module = _load("eval_true_reward_odd_even", _ODD_EVEN_DIR / "eval_scripts")
    rewards = -np.arange(1, 51, dtype=np.float64)
    metrics = module.summarize_episode(rewards, collapse_step=21)
    assert metrics["n_transient"] == 21
    assert metrics["n_steady"] == 29
    assert metrics["n_pooled"] == 50
    assert metrics["transient"] == pytest.approx(rewards[:21].mean())
    assert metrics["steady"] == pytest.approx(rewards[21:].mean())
    assert metrics["pooled"] == pytest.approx(rewards.mean())

    short = module.summarize_episode(np.zeros(5), collapse_step=21)
    assert short["n_steady"] == 0
    assert np.isnan(short["steady"])


def test_oracle_beats_the_naive_baseline_by_the_recorded_margin():
    """The reference policies, on the real env, at the recorded values.

    Under the 0/1 exact-match reward (2026-09-03) domain_mds/oddeven.md
    records, on oe50_short, the Bayes oracle (posterior MODE) at 0.881
    reward/step in the steady state and 0.638 in the transient, and
    play-the-previous-observation at about 0.256 throughout. Reward per step
    IS the exact-match rate now. The oracle-minus-naive margin is the value
    of accumulating evidence and so the effect an encoder comparison has to
    resolve; if it ever collapses, the domain has stopped testing what it was
    chosen to test.

    Loose tolerances on purpose -- this pins the SIZE of the effect, not a
    Monte-Carlo digit. 40 episodes keeps it fast. (The previous version of
    this test pinned squared-error-era values and kept passing on the inline
    arithmetic; see PITFALLS.md on hollow tests.)
    """
    module = _load("eval_true_reward_odd_even", _ODD_EVEN_DIR / "eval_scripts")
    references = module.run_reference_policies(
        VARIANT, n_episodes=40, seed=0, collapse_step=21)
    oracle = references["oracle"]["reward"]
    naive = references["prev_obs"]["reward"]
    assert 0.75 < oracle["steady"] <= 1.0, oracle["steady"]
    assert 0.45 < oracle["transient"] < 0.85, oracle["transient"]
    assert naive["steady"] < 0.45, naive["steady"]
    assert oracle["steady"] - naive["steady"] > 0.4, (
        "the oracle's advantage over the naive baseline has collapsed")
    # Reward and exact match are the same measurement under 0/1.
    assert references["oracle"]["exact_match"]["steady"] == pytest.approx(
        oracle["steady"])
    # High but not perfect: telling s* from s*+/-2 needs many observations,
    # and the posterior itself concentrates past 0.9 in only ~63% of
    # episodes by step 30.
    assert 0.5 < references["oracle"]["exact_match"]["steady"] < 1.0


def test_eval_reads_num_particles_off_the_checkpoint():
    """PITFALLS.md section 5: never default the particle count.

    The helper returns None rather than raising on a bad path, so the caller's
    value stands and SB3 complains on its own.
    """
    module = _load("eval_true_reward_odd_even", _ODD_EVEN_DIR / "eval_scripts")
    assert module._checkpoint_num_particles("/nonexistent/model.zip") is None


def test_collected_dataset_loads_in_the_rl_frame(belief_env_module, tmp_path):
    """3_train_st.py must see the SAME inputs the RL extractor sees.

    The collector stores raw states and records particle_centre and
    particle_scale; get_dataset applies (x - centre) / scale. Until
    2026-09-03 only the scale was applied, so an encoder was pretrained on
    [0.04, 2.04] and then handed [-1, 1] under PPO -- every pretrained-ST
    Odd-Even number was measured on an out-of-distribution encoder.
    test_particles_are_centred_on_the_state_range covers the RL half; this
    covers the dataset half and pins the two to each other.
    """
    from set_transformer.data.dataset import get_dataset

    collector = _load("2_collect_pf_dataset", _ODD_EVEN_DIR)
    particles, weights, metadata = collector.collect_dataset_for_test(
        ns=NS, num_episodes=2, timesteps=10, num_particles=NS, seed=0)
    path = tmp_path / "dataset.npz"
    np.savez(path, particles=particles, weights=weights,
             particle_scale=np.float32(metadata["particle_scale"]),
             particle_centre=np.float32(metadata["particle_centre"]),
             metadata=json.dumps(metadata, default=str))

    dataset = get_dataset(str(path))
    coords = dataset.data.numpy().ravel()
    assert coords.min() == pytest.approx(-1.0, abs=1e-4)
    assert coords.max() == pytest.approx(1.0, abs=1e-4)

    env = belief_env_module.make_odd_even_belief_env(
        num_particles=NS, variant=VARIANT, seed=0)()
    obs, _info = env.reset(seed=1)
    env.close()
    rl_frame = np.sort(np.asarray(obs["particles"], dtype=np.float64).ravel()
                       / metadata["particle_scale"])
    # Exact-support particles are the same 50 states in every snapshot, so
    # any dataset row must equal any RL observation once both are sorted.
    np.testing.assert_allclose(
        np.sort(dataset.data[0].numpy().ravel()), rl_frame, atol=1e-4)


def test_rebalance_never_duplicates_rows():
    """Upsampling with replacement put identical snapshots on both sides of
    3_train_st.py's train/val split (5.84x duplication on oe50_short).
    Rebalancing now downsamples to the tightest bucket instead."""
    collector = _load("2_collect_pf_dataset", _ODD_EVEN_DIR)
    # 5 early (<3), 50 mid (3..20), 100 late (>=21): far from 40/35/25.
    steps = np.concatenate([np.zeros(5), np.full(50, 10), np.full(100, 25)]).astype(int)
    n = len(steps)
    row_id = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
    particles = np.repeat(row_id, NS, axis=1)          # row id in every particle
    weights = np.full((n, NS), 1.0 / NS, dtype=np.float32)

    out_p, out_w, out_s = collector._rebalance(
        particles, weights, steps, by="step", seed=0)
    ids = out_p[:, 0, 0]
    assert len(np.unique(ids)) == len(ids), "rebalance duplicated rows"
    # min(5/0.40, 50/0.35, 100/0.25) = 12 rows -> 5 / 4 / 3.
    assert len(ids) == 12
    assert int((out_s < 3).sum()) == 5
    assert int(((out_s >= 3) & (out_s < 21)).sum()) == 4
    assert int((out_s >= 21).sum()) == 3
    assert out_w.shape == (12, NS)


# ---------------------------------------------------------------------------
# Audit 2026-09-06 fixes (PITFALLS.md section 8, items 1-2)
# ---------------------------------------------------------------------------


def test_eval_sem_is_per_episode_not_per_step():
    """Item 1. Nine identical steady rewards in one episode are ONE sample.

    Two episodes, one all-hit and one all-miss in steady state: the per-
    episode SEM is std([1, 0], ddof=1) / sqrt(2) = 0.5. The old step-pooled
    SEM treated the 18 steady steps as independent and read 0.121.
    """
    module = _load("eval_true_reward_odd_even", _ODD_EVEN_DIR / "eval_scripts")
    collapse = module.COLLAPSE_STEP
    hit = np.concatenate([np.zeros(collapse), np.ones(9)])
    miss = np.zeros(collapse + 9)
    out = module._split_metric([hit, miss], collapse)
    assert out["steady"] == pytest.approx(0.5)
    assert out["steady_sem"] == pytest.approx(0.5)
    assert out["steady_sem"] > 0.12 * 3, "step-pooled SEM is back"
    # Transient: both episodes are all-zero -> identical means -> SEM 0.
    assert out["transient_sem"] == pytest.approx(0.0)
    # An episode with no steady steps is not a sample of the steady split.
    short = np.zeros(collapse)
    out2 = module._split_metric([hit, miss, short], collapse)
    assert out2["steady_sem"] == pytest.approx(0.5)


def test_st_checkpoint_with_a_different_arena_scale_is_refused(st_pieces, tmp_path):
    """Item 2. No parameter shape changes with the particle scale, so a
    checkpoint pretrained in one frame loaded silently into an encoder fed
    another (the CGF extractor already refused this). Both checkpoint
    formats must be caught: 3_pretrain_st_belief.py records arena_scale in
    config; 3_train_st.py (Trainer) records particle_scale top-level."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    st_module, venv, path, _reference = st_pieces
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    space = venv.observation_space
    kwargs = dict(num_encodings=8, dim_encoder=8, num_inds=32, dim_hidden=128,
                  num_heads=4, ln=True, weight_channel=True)

    supervised = tmp_path / "supervised.pt"
    torch.save({"model_state_dict": state, "config": {"arena_scale": 24.5}}, supervised)
    trainer = tmp_path / "trainer.pt"
    torch.save({"model_state_dict": state, "particle_scale": 24.5}, trainer)

    for ck in (supervised, trainer):
        # Matching scale loads.
        st_module.SetTransformerFeaturesExtractor(
            space, arena_scale=24.5, pretrained_st_model_path=str(ck), **kwargs)
        # A float that is equal within tolerance loads too.
        st_module.SetTransformerFeaturesExtractor(
            space, arena_scale=24.5 * (1 + 1e-9), pretrained_st_model_path=str(ck), **kwargs)
        with pytest.raises(RuntimeError, match="scale"):
            st_module.SetTransformerFeaturesExtractor(
                space, arena_scale=4.5, pretrained_st_model_path=str(ck), **kwargs)

    # A checkpoint that records neither (pre-2026-09-05) still loads.
    bare = tmp_path / "bare.pt"
    torch.save({"model_state_dict": state}, bare)
    st_module.SetTransformerFeaturesExtractor(
        space, arena_scale=4.5, pretrained_st_model_path=str(bare), **kwargs)
