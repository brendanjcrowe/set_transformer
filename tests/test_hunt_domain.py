"""The ``hunt`` domain on the shared harness (plan section 9, batch 9.1, 2026-09-13).

Pins: the registry (three variants, the pass-through filter, caps from the registration);
the FRAME -- the tensor the harness hands the encoder equals what the recorded arms' extractor
computed (``src/hunt_tasks/encoders/extractors.py::_Base.forward``, agent-centric on the env's
scaled observation), element for element; uniform weights; a filter built for another cloud
size is refused; the schedules' endpoints and rounding; the eval env at the final values;
the evaluation rule on hand-built episodes (a wrong hit is NOT a success); ``--dry_run`` from
a foreign cwd writes under the output root only; the Domain record is complete.
"""
from __future__ import annotations

import argparse
import importlib.util
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
pytest.importorskip("pdomains.hunt", reason="needs the pomdp-domains hunt-envs branch")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from set_transformer.rl import domains  # noqa: E402
from set_transformer.rl import encoders  # noqa: E402
from set_transformer.rl import run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.domains import hunt  # noqa: E402
from set_transformer.rl.domains.base import Domain  # noqa: E402
from set_transformer.rl.eval_true_reward import Episode  # noqa: E402
from set_transformer.rl.particle_filters.hunt import EnvEmittedBeliefFilter  # noqa: E402

HUNT = hunt.HUNT


def _env(variant: str, *, training: bool, seed: int = 0, num_particles: int = 100):
    return HUNT.make_env(variant, num_particles=num_particles,
                         particle_filter_class=HUNT.resolve(variant).particle_filter,
                         seed=seed, rank=0, monitor_dir=None, training=training, options={})()


def _schedule_wrapper(env):
    e = env
    while not isinstance(e, hunt.HuntScheduleWrapper):
        e = e.env
    return e


# --------------------------------------------------------------------------
# Registry and record
# --------------------------------------------------------------------------

def test_registry_three_variants_pass_through_filter_and_caps():
    assert sorted(HUNT.variants) == ["cluster_hunt", "least_mass", "most_var"]
    for name in HUNT.variants:
        variant = HUNT.resolve(name)
        assert variant.particle_filter is EnvEmittedBeliefFilter
        assert gym.spec(variant.env_id) is not None
        assert HUNT.episode_cap(name) == 60
        assert HUNT.default_num_particles(name) == 100
    assert HUNT.resolve("cluster_hunt").task == "collect_all"
    assert HUNT.resolve("least_mass").task == HUNT.resolve("most_var").task == "pick_target"
    assert gym.spec("pdomains-most-var-v0").kwargs["target_rule"] == "max_var"
    with pytest.raises(ValueError, match="least_mass"):
        HUNT.resolve("minmass")
    assert domains.get("hunt") is HUNT
    assert domains.domain_of_variant("least_mass") is HUNT
    assert HUNT.run_subdir("st", "most_var") == "hunt_st_most_var"


def test_arena_scale_is_the_env_module_constant():
    assert hunt.ARENA_SCALE == pdomains.hunt.SCALE == HUNT.default_arena_scale("least_mass")


def test_domain_record_is_complete():
    import dataclasses
    for field in dataclasses.fields(Domain):
        if field.name == "collection":
            continue        # the collector arrives with batch 9.2
        assert getattr(HUNT, field.name) is not None, field.name
    assert HUNT.collection is None
    assert HUNT.variants is hunt.VARIANTS and HUNT.resolve is hunt.resolve
    assert HUNT.default_variant in HUNT.variants and HUNT.particle_dim == 2
    assert HUNT.encoder_callbacks("st") == []
    assert HUNT.evaluation.report is hunt._eval_report
    assert HUNT.evaluation.default_n_episodes == 300 and HUNT.evaluation.reseed_per_episode


def test_cgf_t_bound_from_the_sizing_rule_is_50_on_every_variant():
    for name in HUNT.variants:
        assert hunt.cgf_t_bound(name) == pytest.approx(50.0)     # 3 / (0.6 / 10)
    ns = argparse.Namespace(t_param="polar", t_init_mode="spread", t_bound=50.0)
    assert hunt._cgf_t_init_max_default(ns) == pytest.approx(40.0)


# --------------------------------------------------------------------------
# The frame and the weights
# --------------------------------------------------------------------------

def _original_base_forward():
    """``_Base`` from ``src/hunt_tasks/encoders/extractors.py`` (the recorded arms), with the
    identity as the encoder, so ``forward`` returns the agent-centric cloud it hands on."""
    src = _REPO_ROOT / "src"
    extractors = src / "hunt_tasks" / "encoders" / "extractors.py"
    probe = src / "mode_recovery_probe"
    if not extractors.is_file() or not probe.is_dir():
        pytest.skip("the recorded extractor (src/hunt_tasks) is not beside this checkout")
    saved = list(sys.path)
    sys.path[:0] = [str(src / "hunt_tasks"), str(probe)]
    try:
        spec = importlib.util.spec_from_file_location("_hunt_record_extractors", extractors)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved

    class Identity(module._Base):
        def encode(self, particles):
            return particles.reshape(particles.shape[0], -1)

    return Identity


def test_frame_matches_the_recorded_extractors_element_for_element():
    """The recorded arms saw ``(particles - CENTER)/SCALE - (pos - CENTER)/SCALE`` inside the
    extractor; the harness filter emits ``particles - pos`` raw and the extractors divide by
    ``arena_scale`` = SCALE. Same tensor, checked on the live env for several steps."""
    Identity = _original_base_forward()
    env = _env("least_mass", training=True, seed=3)
    unwrapped = env.unwrapped
    obs_space = gym.spaces.Dict({
        "agent": gym.spaces.Box(-1, 1, (2,), np.float32),
        "particles": gym.spaces.Box(-1, 1, (100, 2), np.float32)})
    original = Identity(obs_space, enc_dim=200, agent_centric=True)

    obs, _ = env.reset()
    for _ in range(6):
        # what the recorded stack computed from the env's own (scaled) observation
        raw = unwrapped._obs()
        record = original({k: torch.as_tensor(v)[None] for k, v in raw.items()})
        record_cloud = record[0, :200].reshape(100, 2)
        record_agent = record[0, 200:]
        # what the harness hands the encoder: filter output / arena_scale, beside "obs"
        harness_cloud = torch.as_tensor(obs["particles"]) / HUNT.default_arena_scale("least_mass")
        assert torch.allclose(harness_cloud, record_cloud, atol=1e-6)
        assert torch.equal(torch.as_tensor(obs["obs"]), record_agent)
        # the filter's copy is the env's cloud minus the env's position, raw units
        assert np.allclose(obs["particles"], unwrapped.particles - unwrapped.pos, atol=1e-5)
        obs, _r, term, trunc, _i = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()


def test_weights_are_uniform_and_the_obs_key_is_the_agent_position():
    env = _env("cluster_hunt", training=True)
    obs, _ = env.reset()
    assert set(obs) == {"obs", "particles", "weights"}
    assert obs["obs"].shape == (2,) and obs["particles"].shape == (100, 2)
    assert np.allclose(obs["weights"], 0.01) and obs["weights"].dtype == np.float32
    assert np.allclose(obs["obs"], (env.unwrapped.pos - pdomains.hunt.CENTER) / pdomains.hunt.SCALE)
    obs, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert np.allclose(obs["weights"], 0.01)


def test_filter_refuses_a_cloud_size_that_is_not_the_envs():
    with pytest.raises(ValueError, match="emits 100 particles"):
        _env("least_mass", training=True, num_particles=50)
    base = gym.make("pdomains-least-mass-v0")
    base.reset(seed=0)
    with pytest.raises(ValueError, match="num_particles=64"):
        EnvEmittedBeliefFilter(64, None, env=base.unwrapped)
    with pytest.raises(TypeError):
        EnvEmittedBeliefFilter(100, None, env=base.unwrapped, sigma=1.0)
    pf = EnvEmittedBeliefFilter(100, None, env=base.unwrapped, rng_seed=7)
    assert pf.particle_dim == 2 and pf.particles.shape == (100, 2)
    pf.predict(np.zeros(2))                  # no-op
    before = pf.particles.copy()
    base.step(np.ones(2, dtype=np.float32))  # the env redraws its cloud and moves
    pf.update(None)
    assert not np.array_equal(before, pf.particles)
    assert np.allclose(pf.particles, base.unwrapped.particles - base.unwrapped.pos, atol=1e-5)


def test_agent_obs_wrapper_refuses_the_oracle_key():
    env = gym.make("pdomains-least-mass-v0", include_oracle=True)
    with pytest.raises(ValueError, match="oracle"):
        hunt.HuntAgentObsWrapper(env)


# --------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------

def test_schedule_endpoints_match_the_records():
    lm = hunt.variant_schedules("least_mass")
    assert [s.name for s in lm] == ["n_active"]
    assert lm[0].values_at(0.0) == (2.0,) and lm[0].values_at(0.4) == (5.0,)
    assert lm[0].values_at(0.2) == (3.5,) and lm[0].values_at(1.0) == (5.0,)
    assert hunt.variant_schedules("most_var") == lm
    ch = hunt.variant_schedules("cluster_hunt")
    assert [s.name for s in ch] == ["n_active", "hit_radius"]
    assert ch[0].values_at(0.0) == (1.0,) and ch[0].values_at(0.4) == (5.0,)
    assert ch[1].values_at(0.0) == (1.6,) and ch[1].values_at(0.4) == pytest.approx((0.6,))
    assert ch[1].values_at(0.2) == pytest.approx((1.1,))
    assert hunt.final_values(ch) == {"set_n_active": 5.0, "set_hit_radius": pytest.approx(0.6)}
    assert hunt.make_schedules("none", None) == ()


def test_setters_round_n_active_like_the_recorded_callbacks_and_reach_the_env():
    env = _env("cluster_hunt", training=True)
    wrapper = _schedule_wrapper(env)
    env.set_n_active(2.6)                       # through the router, as the callback calls it
    assert wrapper.n_active == 3 == env.unwrapped.cfg.n_active
    env.set_n_active(2.4)
    assert wrapper.n_active == 2
    env.set_hit_radius(1.1)
    assert wrapper.hit_radius == pytest.approx(1.1) == env.unwrapped.cfg.hit_radius
    env.reset()
    assert env.unwrapped.n_active == 2          # the next episode spawns that many


def test_eval_env_sits_at_the_final_values_which_are_the_registered_defaults():
    for name in HUNT.variants:
        env = _env(name, training=False)
        wrapper = _schedule_wrapper(env)
        assert wrapper.n_active == 5
        assert wrapper.hit_radius == pytest.approx(0.6)
        spec_kwargs = gym.spec(HUNT.resolve(name).env_id).kwargs
        assert wrapper.hit_radius == pytest.approx(spec_kwargs.get("hit_radius", 0.6))


def test_resolve_arguments_fills_the_variant_schedules_and_horizon():
    parser = train_mod.build_parser(HUNT, encoders.get("gaussian"))
    args = parser.parse_args(["--variant", "cluster_hunt"])
    assert HUNT.resolve_arguments(parser, args) == {}
    assert args.n_active_curriculum == "0:1,0.4:5,1:5"
    assert args.hit_radius_curriculum == "0:1.6,0.4:0.6,1:0.6"
    assert args.total_timesteps == 1_500_000
    assert [s.target for s in HUNT.schedules(args)] == ["set_n_active", "set_hit_radius"]
    args = parser.parse_args(["--variant", "least_mass", "--n_active_curriculum", "none",
                              "--total_timesteps", "1000"])
    HUNT.resolve_arguments(parser, args)
    assert args.total_timesteps == 1000 and args.hit_radius_curriculum is None
    assert HUNT.schedules(args) == ()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _episode(n_steps: int, final: dict, reward: float = -0.1) -> Episode:
    ep = Episode()
    for i in range(n_steps):
        ep.rewards.append(reward)
        ep.infos.append(dict(final) if i == n_steps - 1 else {"target": 0, "hit": -1})
    return ep


def test_pick_target_report_reads_solved_not_early_termination(capsys):
    cap = 60
    episodes = [
        _episode(12, dict(target=0, hit=0, solved=True, outcome="correct", episode_steps=12)),
        _episode(9, dict(target=0, hit=2, solved=False, outcome="wrong", episode_steps=9)),
        _episode(60, dict(target=0, hit=-1, solved=False, outcome="timeout", episode_steps=60)),
        _episode(20, dict(target=1, hit=1, solved=True, outcome="correct", episode_steps=20)),
    ]
    # the generic rule would count the WRONG hit (ended at step 9 < cap) as a success
    assert episodes[1].is_success(cap) is True
    args = argparse.Namespace(deterministic=True)
    report = hunt._eval_report(episodes, None, args, HUNT.resolve("least_mass"), cap)
    assert report["success_rate"] == pytest.approx(0.5) and report["successes"] == 2
    assert report["outcome_counts"] == {"correct": 2, "wrong": 1, "timeout": 1}
    assert report["outcome_wrong"] == pytest.approx(0.25)
    assert [e["outcome"] for e in report["episodes"]] == ["correct", "wrong", "timeout", "correct"]
    assert report["mean_length"] == pytest.approx((12 + 9 + 60 + 20) / 4)
    out = capsys.readouterr().out
    assert "Success (correct cluster): 2/4 (50.0%)" in out
    assert "correct 2, wrong 1, timeout 1" in out
    json.dumps(report)   # what the eval script writes


def test_collect_all_report_counts_clusters(capsys):
    cap = 60
    episodes = [
        _episode(30, dict(episode_collected=5, n_active=5, solved=True, episode_steps=30)),
        _episode(60, dict(episode_collected=3, n_active=5, solved=False, episode_steps=60)),
    ]
    args = argparse.Namespace(deterministic=True)
    report = hunt._eval_report(episodes, None, args, HUNT.resolve("cluster_hunt"), cap)
    assert report["success_rate"] == pytest.approx(0.5)
    assert report["mean_collected"] == pytest.approx(4.0)
    assert report["fraction_collected"] == pytest.approx(0.8)
    assert "Collected     : 4.00 of 5.0 clusters (80.0%)" in capsys.readouterr().out


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
    train_mod.main(["--domain", "hunt", "--encoder", "cgf", "--variant", "most_var",
                    "--feature_mode", "K_grad", "--ent_coef", "0.005", "--separate_extractors",
                    "--output_root", str(root), "--dry_run"])
    [record] = list(root.glob("hunt/most_var/rl/cgf/*_seed0/run_config.json"))
    config = json.loads(record.read_text())
    assert config["env_id"] == "pdomains-most-var-v0"
    assert config["particle_filter_class"] == "EnvEmittedBeliefFilter"
    assert config["num_particles"] == 100 and config["arena_scale"] == 10.0
    assert config["t_param"] == "polar" and config["t_bound"] == 50.0
    assert config["t_init_max"] == pytest.approx(40.0) and config["t_init_mode"] == "spread"
    assert config["ent_coef"] == 0.005 and config["separate_extractors"] is True
    assert config["episode_cap"] == 60 and config["task"] == "pick_target"
    assert config["env_kwargs"]["target_rule"] == "max_var"
    assert config["total_timesteps"] == 3_000_000
    assert config["n_active_curriculum"] == "0:2,0.4:5,1:5"
    assert list(elsewhere.iterdir()) == []
    assert "CGF t_bound from the sizing rule: 50" in capsys.readouterr().out


@pytest.mark.slow
def test_smoke_train_least_mass_st_runs_the_schedule_and_saves(monkeypatch, tmp_path, capsys):
    from stable_baselines3 import PPO
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    model_path = tmp_path / "models" / "st_agent.zip"
    model = train_mod.main(
        ["--variant", "least_mass", "--total_timesteps", "128", "--n_envs", "2",
         "--ppo_n_steps", "32", "--batch_size", "32", "--n_epochs", "1", "--device", "cpu",
         "--eval_freq", "64", "--n_eval_episodes", "1", "--save_freq", "64",
         "--ent_coef", "0.005", "--output_root", str(tmp_path / "root"),
         "--log_dir", str(tmp_path / "logs") + "/", "--model_save_path", str(model_path)],
        domain="hunt", encoder="st")
    out = capsys.readouterr().out
    assert "[Curriculum] step=0, progress=0.00, n_active=2.00" in out
    assert "ST geometry: num_inds=16 dim_hidden=64 num_post_sab=2 (default)" in out
    assert model.ent_coef == 0.005
    assert model_path.exists()
    assert run_records.read_run_status(str(model_path))["status"] == "completed"
    assert (tmp_path / "models" / "checkpoints" / "hunt_st_64_steps.zip").exists()
    assert list((tmp_path / "root").glob("hunt/least_mass/rl/st/*/run_config.json"))
    PPO.load(str(model_path), device="cpu")
