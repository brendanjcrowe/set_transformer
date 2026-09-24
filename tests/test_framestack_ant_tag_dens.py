"""Ant-Tag ``--policy_obs dens``: the episode's two active dens as a ``static`` key (2026-09-24).

Pins ``change_mds/framestack_cdens_static_2026-09-24.md``:

* ``policy_obs="base"`` (the default) is the recorded env: no extra wrapper in the chain, the
  same observation space and the same arrays step for step as a factory call without the
  option;
* ``policy_obs="dens"`` adds ``static`` = ``[heavy / s, w, light / s, 1 - w]`` (heavy first,
  ``s`` the arena scale), constant over an episode, re-read at every reset (the mirror bit),
  and leaves ``obs``, ``particles`` and ``weights`` array for array as under ``base``;
* ``dens`` is refused on a variant without dens, at the train and collect doors;
* the curriculum setters still reach the visibility wrapper through the router, through
  ``apply_to_env`` and through ``env_method``, with the new wrapper (and the history wrapper)
  in the chain;
* the history wrapper stacks ``obs`` and never ``static``; the framestack extractor
  concatenates ``static`` when the space has it;
* the observation-key refusal at the eval and collect doors names ``--policy_obs dens``;
* a short real framestack run on cdens_terminal trains with ``static``, evaluates with
  ``--policy_obs dens --no_mask``, and is refused without the flag.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains", reason="envs are registered by pdomains")
pytest.importorskip("mujoco", reason="Ant-Tag needs MuJoCo")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from set_transformer.rl import collect as collect_mod  # noqa: E402
from set_transformer.rl import eval_true_reward as eval_mod  # noqa: E402
from set_transformer.rl import run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.curriculum import apply_to_env  # noqa: E402
from set_transformer.rl.domains import ant_tag as at  # noqa: E402
from set_transformer.rl.feature_extractors.framestack import (  # noqa: E402
    FrameStackFeaturesExtractor,
)
from set_transformer.rl.wrappers.obs_history import (  # noqa: E402
    ObsHistoryDictWrapper,
    checkpoint_obs_keys,
    obs_keys_mismatch,
)

VARIANT = "cdens_terminal"
N = 16                         # particles: small, the filter's cost is per particle
W_HEAVY = 6.75 / (2.4 + 6.75)  # cden_f / (cden_h + cden_f), the registered geometry


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    """PITFALLS.md 13.36: the file sets the variable itself."""
    monkeypatch.setenv("WANDB_MODE", "offline")


def _build(seed=0, variant=VARIANT, **options):
    v = at.resolve(variant)
    return at.make_ant_tag_cgf_env(num_particles=N, seed=seed, env_id=v.env_id,
                                   particle_filter_class=v.particle_filter,
                                   obs_mask_indices=[-2, -1], **options)()


def _chain(env):
    names = []
    while env is not None:
        names.append(type(env).__name__)
        env = getattr(env, "env", None)
    return names


def _actions(env, n, seed=123):
    rng = np.random.default_rng(seed)
    low, high = env.action_space.low, env.action_space.high
    return [rng.uniform(low, high).astype(np.float32) for _ in range(n)]


def _roll(env, actions, seed=5):
    frames = [env.reset(seed=seed)[0]]
    for a in actions:
        obs, _r, term, trunc, _i = env.step(a)
        frames.append(obs)
        if term or trunc:
            break
    return frames


# --------------------------------------------------------------------------
# 1. base is the recorded env
# --------------------------------------------------------------------------

def test_base_is_byte_identical_to_the_factory_without_the_option():
    plain, base = _build(), _build(policy_obs="base")
    try:
        assert plain.observation_space == base.observation_space
        assert "static" not in base.observation_space.spaces
        assert _chain(plain) == _chain(base)
        assert "DenMapObservationWrapper" not in _chain(base)
        acts = _actions(plain, 8)
        for a, b in zip(_roll(plain, acts), _roll(base, acts), strict=True):
            assert set(a) == set(b) == {"obs", "particles", "weights"}
            for k in a:
                assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k
    finally:
        plain.close()
        base.close()


def test_an_unknown_policy_obs_or_a_missing_arena_scale_is_refused():
    with pytest.raises(ValueError, match="unknown policy_obs"):
        at.make_ant_tag_cgf_env(num_particles=N, policy_obs="map")
    with pytest.raises(ValueError, match="needs arena_scale"):
        at.make_ant_tag_cgf_env(num_particles=N, policy_obs="dens")


# --------------------------------------------------------------------------
# 2. dens: the values, constant per episode, re-read at reset; the rest unchanged
# --------------------------------------------------------------------------

def test_dens_emits_the_active_dens_heavy_first_and_leaves_the_rest_alone():
    scale = at.get_ant_tag_arena_scale(at.resolve(VARIANT).env_id)
    assert scale == 7.0
    base, dens = _build(policy_obs="base"), _build(policy_obs="dens", arena_scale=scale)
    try:
        space = dens.observation_space["static"]
        assert (space.shape, space.dtype) == ((6,), np.float32)
        assert np.all(space.low == -1.0) and np.all(space.high == 1.0)
        assert _chain(dens)[:3] == ["_CurriculumRouter", "DenMapObservationWrapper", "Monitor"]
        assert _chain(dens)[2:] == _chain(base)[1:]
        acts = _actions(base, 8)
        frames_b, frames_d = _roll(base, acts), _roll(dens, acts)
        u = dens.unwrapped
        expected = np.array([*(u.cden_heavy_pos / scale), W_HEAVY,
                             *(u.cden_light_pos / scale), 1.0 - W_HEAVY], dtype=np.float32)
        # light is the heavy den's mirror, f / h farther out
        np.testing.assert_allclose(u.cden_light_pos, -u.cden_heavy_pos * 6.75 / 2.4, rtol=1e-12)
        assert np.isclose(u.cden_w_heavy, W_HEAVY)
        for fb, fd in zip(frames_b, frames_d, strict=True):
            assert set(fd) == set(fb) | {"static"}
            assert np.array_equal(fd["static"], expected)             # constant over the episode
            assert space.contains(fd["static"])
            for k in fb:
                assert fb[k].dtype == fd[k].dtype and np.array_equal(fb[k], fd[k]), k
    finally:
        base.close()
        dens.close()


def test_the_static_key_follows_the_mirror_bit_across_resets():
    env = _build(policy_obs="dens", arena_scale=7.0)
    try:
        signs = set()
        for seed in range(40):
            obs, _ = env.reset(seed=seed)
            u = env.unwrapped
            np.testing.assert_allclose(obs["static"][:2], u.cden_heavy_pos / 7.0, rtol=1e-6)
            np.testing.assert_allclose(obs["static"][3:5], u.cden_light_pos / 7.0, rtol=1e-6)
            assert np.sign(obs["static"][0]) == -np.sign(obs["static"][3])
            signs.add(float(np.sign(obs["static"][0])))
            if signs == {-1.0, 1.0}:
                break
        assert signs == {-1.0, 1.0}, "never saw both arrangements in 40 resets"
    finally:
        env.close()


def test_the_wrapper_refuses_an_env_without_dens_and_a_scale_that_leaves_the_box():
    smart = _build(variant="smart")
    try:
        with pytest.raises(ValueError, match="no dens"):
            at.DenMapObservationWrapper(smart, arena_scale=7.0)
    finally:
        smart.close()
    env = _build(policy_obs="dens", arena_scale=1.0)       # 4.77 / 1.0 leaves [-1, 1]
    try:
        with pytest.raises(ValueError, match="leave the Box"):
            env.reset(seed=0)
    finally:
        env.close()
    assert [v for v in at.VARIANTS if at.variant_has_dens(v)] == [
        "cdens", "cdens_hard", "cdens_terminal", "cdens_nospook"]


# --------------------------------------------------------------------------
# 3. the command line: refusal, record
# --------------------------------------------------------------------------

@pytest.fixture
def _quiet_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    return tmp_path


def _dry(tmp_path, variant, *extra, encoder="framestack"):
    root = tmp_path / "root"
    train_mod.main(["--variant", variant, "--output_root", str(root), "--dry_run",
                    "--run_tag", "t", *extra], domain="ant_tag", encoder=encoder)
    [config] = list(root.rglob("run_config.json"))
    return json.loads(config.read_text())


def test_dens_is_refused_on_smart_at_the_train_and_collect_doors(_quiet_runs, capsys):
    with pytest.raises(SystemExit):
        _dry(_quiet_runs, "smart", "--policy_obs", "dens")
    assert "--policy_obs dens needs a counterweighted-den variant" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        collect_mod.main(["--domain", "ant_tag", "--variant", "smart", "--policy_obs", "dens",
                          "--num_episodes", "1", "--output_root", str(_quiet_runs / "root")])
    assert "--policy_obs dens needs a counterweighted-den variant" in capsys.readouterr().err
    with pytest.raises(ValueError, match="counterweighted-den"):
        at._eval_options(type("A", (), {"no_mask": False, "policy_obs": "dens",
                                        "variant": "smart"})())


def test_every_ant_tag_record_carries_policy_obs_and_it_defaults_to_base(_quiet_runs):
    config = _dry(_quiet_runs, "smart", encoder="gaussian")
    assert config["policy_obs"] == "base"


def test_dens_options_travel_only_under_dens(_quiet_runs):
    parser = train_mod.build_parser(at.ANT_TAG, train_mod._encoders.get("gaussian"))
    args = parser.parse_args(["--variant", VARIANT])
    args.arena_scale = 7.0
    base = at._resolve_arguments(parser, args)
    assert "policy_obs" not in base and "arena_scale" not in base
    args = parser.parse_args(["--variant", VARIANT, "--policy_obs", "dens"])
    args.arena_scale = 7.0
    dens = at._resolve_arguments(parser, args)
    assert dens.pop("policy_obs") == "dens" and dens.pop("arena_scale") == 7.0
    assert dens == base
    eval_args = type("A", (), {"no_mask": True, "policy_obs": "base", "variant": VARIANT})()
    assert at._eval_options(eval_args) == {"obs_mask_indices": None}
    eval_args.policy_obs = "dens"
    assert at._eval_options(eval_args) == {"obs_mask_indices": None, "policy_obs": "dens",
                                           "arena_scale": 7.0}


# --------------------------------------------------------------------------
# 4. the curriculum still reaches the visibility wrapper
# --------------------------------------------------------------------------

def _visibility(env):
    while not isinstance(env, at.CurriculumVisibilityWrapper):
        env = env.env
    return env


@pytest.mark.parametrize("n_stack", [1, 3])
def test_the_curriculum_setters_reach_the_visibility_wrapper(n_stack):
    thunk = at.make_ant_tag_cgf_env(num_particles=N, seed=0, env_id=at.resolve(VARIANT).env_id,
                                    particle_filter_class=at.resolve(VARIANT).particle_filter,
                                    policy_obs="dens", arena_scale=7.0)
    make = (lambda: ObsHistoryDictWrapper(thunk(), n_stack)) if n_stack > 1 else thunk
    env = make()
    try:
        assert apply_to_env(env, "set_curriculum_radius", (2.5,))
        assert _visibility(env).visibility_radius == 2.5
        env.set_curriculum_radius(3.5)           # the router's method (forwarded on top)
        assert _visibility(env).visibility_radius == 3.5
        assert apply_to_env(env, "set_evasion_scale", (0.5,))
    finally:
        env.close()
    vec = DummyVecEnv([make])
    try:
        vec.env_method("set_curriculum_radius", 4.5)
        assert _visibility(vec.envs[0]).visibility_radius == 4.5
    finally:
        vec.close()


# --------------------------------------------------------------------------
# 5. the history wrapper and the extractor
# --------------------------------------------------------------------------

def test_the_history_wrapper_stacks_obs_and_not_static():
    env = ObsHistoryDictWrapper(_build(policy_obs="dens", arena_scale=7.0), 3)
    try:
        assert env.observation_space["obs"].shape == (93,)
        assert env.observation_space["static"] == gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        obs, _ = env.reset(seed=0)
        assert obs["obs"].shape == (93,) and obs["static"].shape == (6,)
        first = obs["static"].copy()
        for a in _actions(env, 3):
            obs, *_ = env.step(a)
            assert obs["static"].shape == (6,) and np.array_equal(obs["static"], first)
    finally:
        env.close()


def _space(width, static=None):
    spaces = {"obs": gym.spaces.Box(-np.inf, np.inf, (width,), np.float32),
              "particles": gym.spaces.Box(-np.inf, np.inf, (N, 2), np.float32),
              "weights": gym.spaces.Box(0.0, 1.0, (N,), np.float32)}
    if static:
        spaces["static"] = gym.spaces.Box(-1.0, 1.0, (static,), np.float32)
    return gym.spaces.Dict(spaces)


def test_the_extractor_appends_static_when_present_and_is_unchanged_without():
    plain = FrameStackFeaturesExtractor(_space(93), n_stack=3)
    assert plain.features_dim == 93 and plain._geometry["static_dim"] == 0
    with_static = FrameStackFeaturesExtractor(_space(93, static=6), n_stack=3)
    assert with_static.features_dim == 99 and with_static._geometry["static_dim"] == 6
    assert with_static._geometry["frame_dim"] == 31
    batch = {"obs": torch.randn(4, 93), "particles": torch.zeros(4, N, 2),
             "weights": torch.zeros(4, N), "static": torch.rand(4, 6)}
    assert torch.equal(plain(batch), batch["obs"])
    out = with_static(batch)
    assert out.shape == (4, 99)
    assert torch.equal(out[:, :93], batch["obs"]) and torch.equal(out[:, 93:], batch["static"])
    assert out.data_ptr() != batch["obs"].data_ptr()


# --------------------------------------------------------------------------
# 6. the observation-key refusal
# --------------------------------------------------------------------------

def test_the_key_check_is_silent_on_agreement_and_names_both_directions():
    keys = ("obs", "particles", "weights")
    assert obs_keys_mismatch(None, keys) == (None, (), ())
    assert obs_keys_mismatch(keys, reversed(keys)) == (None, (), ())
    message, missing, extra = obs_keys_mismatch(keys + ("static",), keys)
    assert missing == ("static",) and extra == () and "does not emit" in message
    hint = at._eval_obs_keys_hint(type("A", (), {"policy_obs": "base"})(), missing, extra)
    assert "Pass --policy_obs dens" in hint
    message, missing, extra = obs_keys_mismatch(keys, keys + ("static",))
    assert missing == () and extra == ("static",) and "not trained on" in message
    assert "Drop --policy_obs dens" in at._eval_obs_keys_hint(None, missing, extra)
    assert at._eval_obs_keys_hint(None, ("other",), ()) == ""


# --------------------------------------------------------------------------
# 7. End to end: train with static, eval and collect with and without --policy_obs dens
# --------------------------------------------------------------------------

def test_train_then_eval_and_collect_with_static_and_the_refusals_without(_quiet_runs, capsys):
    tmp_path = _quiet_runs
    model_path = tmp_path / "models" / "framestack_agent.zip"
    model = train_mod.main(
        ["--variant", VARIANT, "--n_stack", "3", "--policy_obs", "dens", "--no_mask_target_obs",
         "--total_timesteps", "128", "--n_envs", "1", "--ppo_n_steps", "64", "--batch_size", "64",
         "--n_epochs", "1", "--num_particles", str(N), "--device", "cpu",
         "--eval_freq", "1000000000", "--save_freq", "1000000000", "--n_eval_episodes", "1",
         "--output_root", str(tmp_path / "root"),
         "--log_dir", str(tmp_path / "logs") + "/", "--model_save_path", str(model_path)],
        domain="ant_tag", encoder="framestack")
    capsys.readouterr()
    assert model.observation_space["obs"].shape == (93,)
    assert model.observation_space["static"].shape == (6,)
    assert model.policy.features_extractor.features_dim == 99
    assert checkpoint_obs_keys(str(model_path)) == ("obs", "particles", "static", "weights")
    [config] = list((tmp_path / "root").rglob("run_config.json"))
    config = json.loads(config.read_text())
    assert (config["policy_obs"], config["arena_scale"], config["n_stack"],
            config["mask_target_obs"]) == ("dens", 7.0, 3, False)
    vecnorm = str(model_path.parent / "vecnormalize.pkl")

    common = ["--variant", VARIANT, "--model_path", str(model_path), "--vecnormalize_path", vecnorm,
              "--n_episodes", "1", "--max_steps", "5", "--seed", "0", "--no_summary", "--no_mask"]
    episodes = eval_mod.main(common + ["--policy_obs", "dens"], domain="ant_tag")
    assert len(episodes) == 1
    assert "Frame stacking: the checkpoint was trained on 3" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        eval_mod.main(common, domain="ant_tag")
    err = capsys.readouterr().err
    assert "['static'], which this env does not emit" in err and "Pass --policy_obs dens" in err

    collect_common = ["--domain", "ant_tag", "--variant", VARIANT, "--num_episodes", "1",
                      "--timesteps", "3", "--num_particles", str(N), "--behaviour", "policy",
                      "--policy_path", str(model_path),
                      "--no_rebalance", "--output_root", str(tmp_path / "root")]
    out = collect_mod.main(collect_common + ["--policy_obs", "dens", "--run_tag", "dens"])
    with np.load(out, allow_pickle=True) as z:
        assert len(z["particles"]) == 4
        assert json.loads(str(z["metadata"]))["args"]["policy_obs"] == "dens"
    with pytest.raises(ValueError, match="Pass --policy_obs dens"):
        collect_mod.main(collect_common + ["--run_tag", "base"])
