"""The ``msearch`` domain on the shared harness (plan section 10, batch 10.3, 2026-09-14).

Pins: the registry (one variant, cap 42 from the registration, arena half 14 pinned to the live
config, the agent-relative filter); the policy NEVER sees a mode entry while the filter still
builds its prior from them; the frame (``particles == absolute - agent_pos`` element for element,
and that is what the Dict carries); the filter's three facts through the stack (equal share per
mode at reset, no surviving particle inside the swept swath unless the target was found, a
sighting collapses the cloud onto the target); the per-episode seed is HONOURED (same seed, same
cloud); shaping present on the training env with the true reward in ``info`` and absent on the
eval env; the found / timeout report on hand-built episodes; ``--dry_run`` from a foreign cwd with
the polar-28 defaults; the collector's labels; collect -> reconstruction pretraining -> round trip
into the RL extractor -> a frozen dry run; a 128-step smoke train.
"""
from __future__ import annotations

import argparse
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
pdomains = pytest.importorskip("pdomains", reason="envs are registered by pdomains")
pytest.importorskip("pdomains.multimodal_search", reason="needs the pomdp-domains hunt-envs branch")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from pdomains.multimodal_search import BASE_OBS_DIM, MODE_OBS_DIM, segment_distance  # noqa: E402

from set_transformer.rl import collect, domains, encoders, pretrain, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.domains import msearch  # noqa: E402
from set_transformer.rl.domains.base import Domain  # noqa: E402
from set_transformer.rl.eval_true_reward import Episode  # noqa: E402
from set_transformer.rl.wrappers.particle_filter import PFDictWithWeightsObservationWrapper  # noqa: E402
from set_transformer.rl.wrappers.shaping import PotentialBasedShapingWrapper  # noqa: E402

MSEARCH = msearch.MSEARCH


def _env(*, training: bool, seed: int = 0, num_particles: int = 100, shaping: str | None = None):
    options = {} if shaping is None else {"shaping": shaping}
    return MSEARCH.make_env("msearch", num_particles=num_particles,
                            particle_filter_class=MSEARCH.resolve("msearch").particle_filter,
                            seed=seed, rank=0, monitor_dir=None, training=training, options=options)()


def _find(env, cls):
    e = env
    while not isinstance(e, cls):
        e = e.env
    return e


def _toward(u, goal):
    d = np.asarray(goal, dtype=np.float64) - np.asarray(u.agent_pos, dtype=np.float64)
    n = np.linalg.norm(d)
    return (d / n if n > 1e-9 else np.zeros(2)).astype(np.float32)


# --------------------------------------------------------------------------
# Registry and record
# --------------------------------------------------------------------------

def test_registry_record_and_the_arena_half_pinned_to_the_live_config():
    assert sorted(MSEARCH.variants) == ["msearch"] and MSEARCH.default_variant == "msearch"
    v = MSEARCH.resolve("msearch")
    assert v.env_id == "pdomains-multimodal-search-v0"
    assert v.particle_filter is msearch.AgentRelativeMultimodalSearchParticleFilter
    assert MSEARCH.episode_cap("msearch") == 42 == gym.spec(v.env_id).max_episode_steps
    cfg = msearch._env_config("msearch")
    assert float(cfg.arena_half) == msearch.ARENA_HALF == MSEARCH.default_arena_scale("msearch")
    assert cfg.obs_dim == BASE_OBS_DIM + MODE_OBS_DIM * cfg.k_max == 67
    assert isinstance(MSEARCH, Domain) and MSEARCH.particle_dim == 2
    assert MSEARCH.default_num_particles("msearch") == 100 and MSEARCH.default_total_timesteps == 3_000_000
    assert MSEARCH.collection is not None and MSEARCH.pretraining.objectives == {}
    assert MSEARCH.evaluation.reseed_per_episode and MSEARCH.evaluation.default_n_episodes == 300
    assert domains.get("msearch") is MSEARCH and domains.domain_of_variant("msearch") is MSEARCH
    with pytest.raises(ValueError, match="msearch"):
        MSEARCH.resolve("multimodal")
    assert msearch.cgf_t_bound("msearch") == pytest.approx(28.0)
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=MSEARCH.default_total_timesteps)
    MSEARCH.add_arguments(parser)
    args = parser.parse_args([]); args.variant = "msearch"
    assert MSEARCH.resolve_arguments(parser, args) == {"shaping": "info_gain"} and args.shaping == "info_gain"
    extras = MSEARCH.run_config_extras(args)
    assert "shaping" not in extras and extras["shaping_scale"] == 100.0     # `shaping` comes from the flags
    assert extras["episode_cap"] == 42 and extras["agent_obs_dim"] == 7
    assert extras["particle_frame"] == "agent_relative" and extras["env_config"]["k_max"] == 10
    args = parser.parse_args(["--shaping", "none"]); args.variant = "msearch"
    assert MSEARCH.resolve_arguments(parser, args) == {"shaping": "none"}
    assert "shaping_scale" not in MSEARCH.run_config_extras(args)


# --------------------------------------------------------------------------
# What the policy sees, what the filter sees, the frame
# --------------------------------------------------------------------------

def test_policy_sees_seven_entries_and_the_filter_still_builds_its_prior_from_the_modes():
    env = _env(training=False, seed=4)
    obs, _ = env.reset(seed=4)
    assert set(obs) == {"obs", "particles", "weights"}
    assert obs["obs"].shape == (7,) and obs["particles"].shape == (100, 2) and obs["weights"].shape == (100,)
    assert env.observation_space["obs"].shape == (7,)
    np.testing.assert_allclose(obs["weights"], 1 / 100)
    u = env.unwrapped
    np.testing.assert_allclose(obs["obs"][0:2], u.agent_pos, atol=1e-5)
    pf = _find(env, PFDictWithWeightsObservationWrapper).particle_filter
    # the filter read the mode block off the RAW observation
    np.testing.assert_allclose(pf._prior_means, u.mode_means)
    np.testing.assert_allclose(pf._prior_covs, u.mode_covs)
    # equal share of particles per mode (nearest prior mean)
    d = np.linalg.norm(pf.absolute_particles[:, None, :] - u.mode_means[None, :, :], axis=-1)
    counts = np.bincount(d.argmin(1), minlength=len(u.mode_means))
    assert counts.max() - counts.min() <= 1 and counts.sum() == 100
    # the pinned mean: the ABSOLUTE cloud's centroid sits near the origin, the relative one at -agent
    assert np.linalg.norm(pf.absolute_particles.mean(0)) < 2.0
    np.testing.assert_allclose(obs["particles"].mean(0), -u.agent_pos + pf.absolute_particles.mean(0), atol=1e-4)
    env.close()


def test_frame_is_absolute_minus_agent_element_for_element_and_the_seed_is_honoured():
    a = _env(training=False, seed=11)
    obs_a, _ = a.reset(seed=11)
    pf = _find(a, PFDictWithWeightsObservationWrapper).particle_filter
    np.testing.assert_allclose(obs_a["particles"], pf.absolute_particles - a.unwrapped.agent_pos, atol=1e-5)
    obs2, *_ = a.step(np.array([1.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(obs2["particles"], pf.absolute_particles - a.unwrapped.agent_pos, atol=1e-5)
    np.testing.assert_allclose(pf.agent_pos, a.unwrapped.agent_pos)
    # same worker seed -> same prior cloud (the parent class draws from an UNSEEDED generator)
    b = _env(training=False, seed=11)
    obs_b, _ = b.reset(seed=11)
    np.testing.assert_array_equal(obs_a["particles"], obs_b["particles"])
    c = _env(training=False, seed=12)
    obs_c, _ = c.reset(seed=11)                 # same env episode, another filter seed
    assert not np.array_equal(obs_a["particles"], obs_c["particles"])
    for e in (a, b, c):
        e.close()


def test_sweeping_refutes_the_swath_and_a_sighting_collapses_the_cloud():
    env = _env(training=False, seed=2)
    found_once = False
    for ep_seed in range(6):
        obs, _ = env.reset(seed=ep_seed)
        u = env.unwrapped
        pfw = _find(env, PFDictWithWeightsObservationWrapper)
        visited: set[int] = set()
        for _ in range(42):
            modes = np.asarray(u.mode_means)
            remaining = [i for i in range(len(modes)) if i not in visited]
            if not remaining:
                break
            j = min(remaining, key=lambda i: np.linalg.norm(modes[i] - u.agent_pos))
            if np.linalg.norm(modes[j] - u.agent_pos) < 3.0:
                visited.add(j)
            prev = u.agent_pos.copy()
            obs, r, term, trunc, info = env.step(_toward(u, modes[j]))
            pf = pfw.particle_filter
            if info["found"]:
                # a sighting: every particle on the target, relative = target offset
                np.testing.assert_allclose(pf.absolute_particles, np.tile(u.target_pos, (100, 1)), atol=1e-6)
                np.testing.assert_allclose(obs["particles"], np.tile(u.target_pos - u.agent_pos, (100, 1)), atol=1e-5)
                assert term and r == pytest.approx(42.0)          # -1 + 43, the TRUE reward (eval env)
                found_once = True
                break
            # no surviving particle inside the swath just swept
            swept = segment_distance(pf.absolute_particles, prev, u.agent_pos) <= u.config.visibility_radius
            assert not swept.any()
            assert r == pytest.approx(-1.0)
            if term or trunc:
                break
    assert found_once, "the scripted tour never found the target in 6 episodes"
    env.close()


# --------------------------------------------------------------------------
# Shaping: training env only
# --------------------------------------------------------------------------

def test_training_env_is_shaped_with_the_true_reward_in_info_and_the_eval_env_is_not():
    train_env = _env(training=True, seed=3)
    assert isinstance(_find(train_env, PotentialBasedShapingWrapper), PotentialBasedShapingWrapper)
    shaper = _find(train_env, PotentialBasedShapingWrapper)
    assert shaper.gamma == 1.0 and shaper.zero_at_termination is False
    obs, info = train_env.reset(seed=3)
    assert info["potential"] == pytest.approx(0.0)              # nothing ruled out yet
    u = train_env.unwrapped
    # sweep a mode that does NOT hold the target: refuting its particles is what the proxy pays for
    # (a sighting ends the episode through the task reward and pays nothing through the potential)
    j = [i for i in range(len(u.mode_means)) if i != u.target_mode][0]
    total_shaping = 0.0
    for _ in range(8):
        obs, r, term, trunc, info = train_env.step(_toward(u, u.mode_means[j]))
        assert info["true_reward"] == pytest.approx(-1.0)
        assert r == pytest.approx(info["true_reward"] + info["shaping_reward"])
        total_shaping += info["shaping_reward"]
        assert not term and not trunc
    pf = _find(train_env, PFDictWithWeightsObservationWrapper).particle_filter
    assert pf.surviving_mass < 1.0 and total_shaping > 0.0     # the swept mode's share was ruled out
    # gamma 1, not zeroed at termination: the shaping so far IS 100 x the nats gathered
    assert total_shaping == pytest.approx(100.0 * pf.information_gain, rel=1e-6)
    assert info["potential"] == pytest.approx(100.0 * pf.information_gain, rel=1e-6)
    train_env.close()
    eval_env = _env(training=False, seed=3)
    with pytest.raises(AttributeError):
        _find(eval_env, PotentialBasedShapingWrapper)
    _, info = eval_env.reset(seed=3)
    _, r, *_ = eval_env.step(np.zeros(2, np.float32))
    assert "true_reward" not in info and r == pytest.approx(-1.0)
    eval_env.close()
    # --shaping none removes it from the training env too
    off = _env(training=True, seed=3, shaping="none")
    with pytest.raises(AttributeError):
        _find(off, PotentialBasedShapingWrapper)
    off.close()
    with pytest.raises(ValueError, match="shaping"):
        msearch.make_msearch_belief_env(100, shaping="distance")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _episode(n_steps: int, found: bool) -> Episode:
    ep = Episode()
    for i in range(n_steps):
        last = i == n_steps - 1
        ep.rewards.append(42.0 if (last and found) else -1.0)
        ep.infos.append({"found": found and last})
    return ep


def test_report_counts_found_and_timeout(capsys):
    episodes = [_episode(10, True), _episode(42, False), _episode(30, True), _episode(42, False)]
    args = argparse.Namespace(deterministic=True)
    report = msearch._eval_report(episodes, None, args, MSEARCH.resolve("msearch"), 42)
    assert report["successes"] == 2 and report["success_rate"] == pytest.approx(0.5)
    assert report["outcome_counts"] == {"found": 2, "timeout": 2}
    assert report["found_mean_length"] == pytest.approx(20.0)
    assert report["mean_reward"] == pytest.approx((33 + -42 + 13 + -42) / 4)
    assert [e["success"] for e in report["episodes"]] == [True, False, True, False]
    out = capsys.readouterr().out
    assert "Found         : 2/4 (50.0%)" in out and "found 2, timeout 2" in out
    json.dumps(report)


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------

def test_dry_run_from_a_foreign_cwd_writes_under_the_root_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    root = tmp_path / "root"
    train_mod.main(["--domain", "msearch", "--encoder", "cgf", "--feature_mode", "K_grad",
                    "--ent_coef", "0.005", "--separate_extractors", "--output_root", str(root), "--dry_run"])
    [record] = list(root.glob("msearch/msearch/rl/cgf/*_seed0/run_config.json"))
    config = json.loads(record.read_text())
    assert config["env_id"] == "pdomains-multimodal-search-v0"
    assert config["particle_filter_class"] == "AgentRelativeMultimodalSearchParticleFilter"
    assert config["num_particles"] == 100 and config["arena_scale"] == 14.0
    assert config["t_param"] == "polar" and config["t_bound"] == pytest.approx(28.0)
    assert config["t_init_max"] == pytest.approx(22.4) and config["t_init_mode"] == "spread"
    assert config["shaping"] == "info_gain" and config["shaping_scale"] == 100.0
    assert config["episode_cap"] == 42 and config["total_timesteps"] == 3_000_000
    assert config["env_config"]["visibility_radius"] == 1.5
    assert list(elsewhere.iterdir()) == []
    out = capsys.readouterr().out
    assert "CGF t_bound from the sizing rule: 28" in out and "training shaping info_gain" in out


# --------------------------------------------------------------------------
# Collection -> reconstruction pretraining -> round trip
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("msearch_data") / "msearch.npz"
    collect.main(["--domain", "msearch", "--variant", "msearch", "--num_episodes", "12",
                  "--seed", "5", "--output_file", str(path)])
    return path


def test_collected_dataset_carries_relative_particles_and_the_labels(dataset):
    with np.load(dataset, allow_pickle=True) as z:
        files = set(z.files)
        assert {"particles", "weights", "particle_scale", "metadata", "agent", "target", "mode_means",
                "mode_covs", "mode_valid", "num_modes", "target_mode", "step"} <= files
        particles, weights = z["particles"], z["weights"]
        means, valid, nm = z["mode_means"], z["mode_valid"], z["num_modes"]
        target, tmode, steps = z["target"], z["target_mode"], z["step"]
        meta = json.loads(str(z["metadata"]))
        assert float(z["particle_scale"]) == 14.0
    n = len(particles)
    assert particles.shape == (n, 100, 2) and weights.shape == (n, 100)
    np.testing.assert_allclose(weights, 1 / 100)
    assert means.shape == (n, 10, 2) and valid.shape == (n, 10)
    assert np.all(valid.sum(1) == nm) and np.all((2 <= nm) & (nm <= 10))
    assert np.all((0 <= tmode) & (tmode < nm))
    # the target lies in its own mode's neighbourhood (scaled frame; sigma <= 1.0 / 14)
    own = means[np.arange(n), tmode]
    assert np.median(np.linalg.norm(target - own, axis=1)) < 0.25
    # the particles are the relative cloud: raw units, within the arena diameter of the agent
    assert np.abs(particles).max() <= 2 * 14.0 + 1e-3
    assert steps[0] == 0 and steps.max() <= 42
    assert meta["variant"] == "msearch" and meta["particle_scale"] == 14.0
    assert meta["particle_frame"] == "agent_relative" and meta["tour_frac"] == 0.6
    assert meta["label_arrays"][0] == "agent"


@pytest.mark.parametrize("encoder", ["st", "cgf"])
def test_reconstruction_pretraining_round_trips_into_the_rl_extractor(encoder, dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    flags = (["--num_inds", "4", "--dim_hidden", "16", "--num_post_sab", "0"] if encoder == "st"
             else ["--num_cgf_features", "8", "--feature_mode", "K_grad"])
    result = pretrain.main(["--domain", "msearch", "--encoder", encoder, "--objective", "reconstruction",
                            "--data_path", str(dataset), "--loss_type", "sinkhorn", "--sinkhorn_blur", "0.02",
                            "--num_epochs", "1", "--batch_size", "64", "--device", "cpu", "--warmup_epochs", "0",
                            "--output_root", str(tmp_path / "root"), *flags])
    out = capsys.readouterr().out
    assert result.rl_checkpoint.exists()
    assert f"Verified: {encoders.get(encoder).extractor_class.__name__} encoder matches" in out
    assert list((tmp_path / "root").glob("msearch/msearch/pretrain/*/reconstruction/*_seed0/run_status.json"))
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert payload["pretraining_run"]["domain"] == "msearch" and payload["pretraining_run"]["objective"] == "reconstruction"
    found = run_records.latest_pretrain_checkpoint("msearch", "msearch", encoder, "reconstruction",
                                                   root=tmp_path / "root")
    assert Path(found) == result.rl_checkpoint.resolve()
    train_mod.main(["--domain", "msearch", "--encoder", encoder, "--variant", "msearch",
                    "--pretrained_path", str(found), "--frozen", "--output_root", str(tmp_path / "root"),
                    "--dry_run"] + (["--feature_mode", "K_grad"] if encoder == "cgf" else []))


@pytest.mark.slow
def test_smoke_train_st_saves_and_reloads(monkeypatch, tmp_path, capsys):
    from stable_baselines3 import PPO
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    model_path = tmp_path / "models" / "st_agent.zip"
    model = train_mod.main(
        ["--variant", "msearch", "--total_timesteps", "128", "--n_envs", "2",
         "--ppo_n_steps", "32", "--batch_size", "32", "--n_epochs", "1", "--device", "cpu",
         "--eval_freq", "64", "--n_eval_episodes", "1", "--save_freq", "64",
         "--ent_coef", "0.005", "--separate_extractors", "--output_root", str(tmp_path / "root"),
         "--log_dir", str(tmp_path / "logs") + "/", "--model_save_path", str(model_path)],
        domain="msearch", encoder="st")
    out = capsys.readouterr().out
    assert "ST geometry: num_inds=16 dim_hidden=64 num_post_sab=2 (default)" in out
    assert "training shaping info_gain" in out
    assert model_path.exists()
    assert run_records.read_run_status(str(model_path))["status"] == "completed"
    assert list((tmp_path / "root").glob("msearch/msearch/rl/st/*/run_config.json"))
    PPO.load(str(model_path), device="cpu")
