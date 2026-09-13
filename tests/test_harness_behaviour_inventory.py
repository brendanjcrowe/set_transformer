"""Behaviour pins for the RL-harness centralisation (refactor_plans.md, section 5).

Written BEFORE any code moved, against the nine arm scripts as they are, so that the
move can be checked mechanically: every test here fails if the behaviour it names
changes, whichever file the code ends up in. The other test files already pin the
extractors, the particle-filter seeding, the variant tables, the reload-after-PPO
step and the finetune fixes; this file covers what they left out:

* how each arm's ``main()`` resolves flags into the training call and the run record
  (since changes 4.3-4.5 every arm is an entry point of ``set_transformer.rl.train``, so the
  shared ``train()`` is intercepted and read back in the flat shape the scripts' own
  ``train_*`` functions took);
* how the training call wires the training env, the eval env, VecNormalize, PPO and the
  callbacks (checked with a fake PPO so nothing trains);
* the curriculum callback's interpolation and how it pushes values into the wrappers;
* the run-record helpers (run-dir name, run_config.json, git provenance);
* the Ant-Tag eval script's cap default and post-load re-seeding;
* the Odd-Even arms' registry-derived defaults, CGF recipe defaults, ST geometry rule.

Loading conventions follow tests/test_ant_tag_shared_pieces_regression.py (Ant-Tag
scripts under their flat names, on sys.path only while importing) and
tests/test_odd_even_pipeline.py (Odd-Even scripts loaded by path under unique keys).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANT_TAG_DIR = _ST_ROOT / "experiments" / "ant_tag"
_ODD_EVEN_DIR = _ST_ROOT / "experiments" / "odd_even"
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco", reason="Ant-Tag needs MuJoCo")
pytest.importorskip("pdomains", reason="envs are registered by pdomains")

import torch  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from set_transformer.rl.encoder_finetune import EncoderLRLoggingCallback  # noqa: E402,F401
from set_transformer.rl.feature_extractors.cgf import (  # noqa: E402
    EncoderDriftLoggingCallback,
    RolloutFeatureNormCallback,
    TNormLoggingCallback,
    WeightedCGFFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.st import (  # noqa: E402
    STFeatureLoggingCallback,
    SetTransformerFeaturesExtractor,
)
from set_transformer.rl.particle_filters.ant_tag import (  # noqa: E402
    AntTagParticleFilter,
    SmartAntTagParticleFilter,
)
from set_transformer.rl.wrappers.particle_filter import (  # noqa: E402
    PFDictWithWeightsObservationWrapper,
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@contextmanager
def _sys_path(*directories):
    saved = list(sys.path)
    try:
        for directory in directories:
            sys.path.insert(0, str(directory))
        yield
    finally:
        sys.path[:] = saved


_ANT_TAG_MODULES = ("variants", "4_train_rl_frozen", "4_train_rl_cgf", "4_train_rl_st",
                    "4_train_rl_gaussian", "4_train_rl_pool")


@pytest.fixture(scope="module")
def ant_tag():
    """Ant-Tag scripts under their flat names (a saved zip resolves its extractor class
    as ``4_train_rl_cgf.WeightedCGFFeaturesExtractor``, so the flat name is the
    contract); dropped from sys.modules at teardown."""
    preexisting = set(sys.modules)
    with _sys_path(_ANT_TAG_DIR):
        modules = {name: importlib.import_module(name) for name in _ANT_TAG_MODULES}
    try:
        yield modules
    finally:
        for name in set(sys.modules) - preexisting:
            module = sys.modules.get(name)
            file = getattr(module, "__file__", None) or ""
            if str(_ST_ROOT / "experiments") in file or name in _ANT_TAG_MODULES:
                sys.modules.pop(name, None)


#: Names that were forwarding files in experiments/odd_even/ until change 5.3a (2026-09-12);
#: the objects live in the package module, which is what these names now resolve to.
_PACKAGE_NAMES = {"variants", "odd_even_belief_env", "st_feature_sentinel", "pretrained_encoder"}


def _load_odd_even(name):
    """experiments/odd_even/<name>.py by path under a unique key (never by flat name: the
    Ant-Tag directory holds same-named scripts); the old forwarding names resolve to the
    package module."""
    if name in _PACKAGE_NAMES:
        from set_transformer.rl.domains import odd_even
        return odd_even
    key = f"_inventory_oe_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, _ODD_EVEN_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    with _sys_path(_ODD_EVEN_DIR):
        spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def odd_even():
    return {name: _load_odd_even(name) for name in
            ("variants", "odd_even_belief_env", "4_train_rl_cgf", "4_train_rl_st",
             "4_train_rl_gaussian")}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _stack(env):
    """The gym wrapper stack from the outside in."""
    e = env
    while e is not None:
        yield e
        e = getattr(e, "env", None)


def _find(env, cls):
    hits = [e for e in _stack(env) if isinstance(e, cls)]
    return hits[0] if hits else None


def _inner_env(vec_env):
    """The single wrapped env inside VecNormalize(DummyVecEnv([...]))."""
    assert isinstance(vec_env, VecNormalize)
    return vec_env.venv.envs[0]


class _FakeModel:
    """Stands in for PPO: records its constructor and learn() arguments, trains nothing."""

    instances: list = []

    def __init__(self, policy, env, **kwargs):
        self.policy_name, self.env, self.kwargs = policy, env, kwargs
        self.num_timesteps = 0
        self.learn_calls = []
        self.saved = []
        # The real features extractor on the env's observation space, so the post-construction
        # steps that read model.policy.features_extractor (the encoder parameter count) run.
        pk = kwargs.get("policy_kwargs") or {}
        cls = pk.get("features_extractor_class")
        extractor = cls(env.observation_space, **pk.get("features_extractor_kwargs", {})) if cls else None
        self.policy = SimpleNamespace(features_extractor=extractor)
        _FakeModel.instances.append(self)

    def learn(self, total_timesteps, callback=None, progress_bar=False, **kw):
        self.learn_calls.append(dict(total_timesteps=total_timesteps, callback=callback, **kw))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"fake")
        self.saved.append(path)


def _recording(cls, store):
    class Recording(cls):
        def __init__(self, *args, **kwargs):
            store.append((args, kwargs))
            super().__init__(*args, **kwargs)
    Recording.__name__ = cls.__name__
    return Recording


def _install_fakes(monkeypatch, module):
    """Replace PPO / EvalCallback / CheckpointCallback on one script module."""
    _FakeModel.instances = []
    evals, checkpoints = [], []
    monkeypatch.setattr(module, "PPO", _FakeModel)
    monkeypatch.setattr(module, "EvalCallback", _recording(EvalCallback, evals))
    monkeypatch.setattr(module, "CheckpointCallback", _recording(CheckpointCallback, checkpoints))
    return evals, checkpoints


_SCHEDULE_KEYS = {"visibility": "curriculum_schedule", "reward": "reward_schedule",
                  "evasion": "evasion_schedule"}


def _flat_ant_tag_train(kw: dict, variant: str, encoder) -> dict:
    """The shared train() call in the FLAT shape the scripts' train_ant_tag_* functions took
    (extractor kwargs, env options and the three schedules' waypoints at top level, plus the
    derived env id), so the assertions written against those functions still read."""
    from set_transformer.rl.domains.ant_tag import resolve
    flat = {k: v for k, v in kw.items()
            if k not in ("features_extractor_kwargs", "env_options", "schedules", "encoder_options")}
    flat.update(kw["features_extractor_kwargs"])
    options = kw.get("env_options") or {}
    for key in ("distance_coeff", "entropy_coeff", "obs_mask_indices", "target_speed_scale"):
        flat[key] = options.get(key)
    for schedule in kw.get("schedules", ()):
        flat[_SCHEDULE_KEYS[schedule.name]] = list(schedule.waypoints)
    flat["env_id"] = resolve(variant).env_id
    flat["variant"] = variant
    flat["encoder"] = encoder if isinstance(encoder, str) else encoder.name
    return flat


def _drive_ant_tag_main(mods, monkeypatch, tmp_path, module_name, argv, encoder=None):
    """Run an Ant-Tag arm's main() through argparse; capture the run record and the
    training call (the shared train(), flattened); create nothing under runs/ and start no
    PPO. The scripts are entry points of set_transformer.rl.train since change 4.5, so the
    run-record helpers are intercepted on the package."""
    from set_transformer.rl import run_records
    from set_transformer.rl import train as train_mod
    module = mods[module_name]
    captured = {}
    monkeypatch.setattr(run_records, "output_root", lambda *a, **k: tmp_path / "runs")
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.setattr(run_records, "write_run_config",
                        lambda run_dir, **cfg: captured.setdefault("config", cfg))

    def fake_train(domain, variant, enc, **kw):
        captured.setdefault("train", _flat_ant_tag_train(kw, variant, enc))
    monkeypatch.setattr(train_mod, "train", fake_train)
    if encoder is None:
        module.main(list(argv))
    else:
        module.main(encoder=encoder, argv=list(argv))
    return captured["config"], captured["train"]


#: Arms whose main() forwards to set_transformer.rl.train.main (switched in change 4.3+).
#: Their training call is the shared train(), intercepted below in the shape the older
#: assertions read.
_SWITCHED_ODD_EVEN_ARMS = {"gaussian", "cgf", "st"}


def _shared_callback_names(encoder: str, train: dict) -> list[str]:
    """The encoder / domain callbacks the shared trainer attaches for a switched arm's
    resolved training call (the pre-switch scripts passed them as extra_callbacks)."""
    from set_transformer.rl.domains.odd_even import ODD_EVEN
    from set_transformer.rl.encoders import ENCODERS
    callbacks = ENCODERS[encoder].callbacks(train["features_extractor_kwargs"],
                                            train.get("encoder_options", {}))
    return [type(cb).__name__ for cb in callbacks + ODD_EVEN.encoder_callbacks(encoder)]


def _drive_switched_odd_even_main(oe, monkeypatch, tmp_path, arm, argv):
    """A switched arm: intercept the shared run-record helpers and train(); return the
    record and the training call with the keys the pre-switch assertions use
    (`variant`, `encoder`, `policy_kwargs`, `run_subdir` beside train()'s own kwargs)."""
    from set_transformer.rl import run_records
    from set_transformer.rl import train as train_mod
    captured = {}
    monkeypatch.setattr(run_records, "output_root", lambda *a, **k: tmp_path / "runs")
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.setattr(run_records, "write_run_config",
                        lambda run_dir, **cfg: captured.setdefault("config", cfg))

    def fake_train(domain, variant, encoder, **kw):
        captured.setdefault("train", dict(
            kw, variant=variant, encoder=encoder if isinstance(encoder, str) else encoder.name,
            policy_kwargs={"features_extractor_kwargs": kw["features_extractor_kwargs"]},
            run_subdir=captured["config"]["run_subdir"]))
    monkeypatch.setattr(train_mod, "train", fake_train)
    oe[f"4_train_rl_{arm}"].main(list(argv))
    return captured["config"], captured["train"]


def _drive_odd_even_main(oe, monkeypatch, tmp_path, arm, argv):
    """Same for an Odd-Even arm. The shared pieces live on the CGF module (the other
    two arms bind them by name), the training call on the arm's own module."""
    if arm in _SWITCHED_ODD_EVEN_ARMS:
        return _drive_switched_odd_even_main(oe, monkeypatch, tmp_path, arm, argv)
    cgf, arm_mod = oe["4_train_rl_cgf"], oe[f"4_train_rl_{arm}"]
    captured = {}
    monkeypatch.setattr(cgf, "_default_run_dir", lambda *a, **k: str(tmp_path / "run"))
    monkeypatch.setattr(cgf, "_git_provenance", lambda: {})
    monkeypatch.setattr(cgf, "_tee_stdout_stderr", lambda path: None)
    monkeypatch.setattr(cgf, "_write_run_config",
                        lambda run_dir, **cfg: captured.setdefault("config", cfg))
    monkeypatch.setattr(arm_mod, "train_odd_even", lambda **kw: captured.setdefault("train", kw))
    monkeypatch.setattr(sys, "argv", [f"4_train_rl_{arm}.py"] + list(argv))
    arm_mod.main()
    return captured["config"], captured["train"]


#: (script module, encoder name, the `encoder=` argument of the pool script's main())
_ANT_TAG_ARMS = [
    ("4_train_rl_cgf", "cgf", None),
    ("4_train_rl_st", "st", None),
    ("4_train_rl_gaussian", "gaussian", None),
    ("4_train_rl_pool", "deepset", "deepset"),
]
_ANT_TAG_IDS = [arm[1] for arm in _ANT_TAG_ARMS]

#: Flags every Ant-Tag arm records in run_config.json for a bare `--variant smart`.
_SHARED_RUN_CONFIG_KEYS = {
    "env_id", "particle_filter_class", "run_subdir", "git", "log_dir", "model_save_path",
    "variant", "seed", "run_tag", "algorithm", "total_timesteps", "n_envs", "learning_rate",
    "batch_size", "ppo_n_steps", "n_epochs", "target_kl", "lr_anneal", "eval_freq", "save_freq",
    "n_eval_episodes", "no_vec_normalize", "net_arch", "device", "num_particles",
    "arena_scale", "curriculum", "reward_schedule", "evasion_curriculum", "distance_coeff",
    "entropy_coeff", "mask_target_obs", "target_speed_scale", "progress_bar",
}


# ==========================================================================
# 1. Ant-Tag main(): flags -> training call and run record  (inventory 5.1, 5.2)
# ==========================================================================

@pytest.mark.parametrize("module_name,encoder,main_encoder", _ANT_TAG_ARMS, ids=_ANT_TAG_IDS)
def test_ant_tag_bare_run_resolves_variant_defaults_and_records_them(
        ant_tag, monkeypatch, tmp_path, module_name, encoder, main_encoder):
    """`--variant smart` alone: env id and filter from the registry, the variant's three
    schedules parsed into waypoints, the target masked, the historical PPO defaults, and
    a run record that carries every flag plus the derived env / filter / subdir."""
    variants = ant_tag["variants"]
    config, train = _drive_ant_tag_main(ant_tag, monkeypatch, tmp_path, module_name,
                                        ["--variant", "smart"], encoder=main_encoder)
    smart = variants.VARIANTS["smart"]
    # derived from the registry
    assert train["env_id"] == smart.env_id == "pdomains-ant-tag-smart-v0"
    assert train["particle_filter_class"] is SmartAntTagParticleFilter
    assert config["env_id"] == smart.env_id
    assert config["particle_filter_class"] == "SmartAntTagParticleFilter"
    assert config["run_subdir"] == variants.run_subdir(encoder, "smart") == f"ant_tag_{encoder}_smart"
    assert "list_variants" not in config
    assert _SHARED_RUN_CONFIG_KEYS <= set(config), _SHARED_RUN_CONFIG_KEYS - set(config)
    # the variant's schedules, parsed
    cgf = ant_tag["4_train_rl_cgf"]
    assert train["curriculum_schedule"] == cgf._parse_curriculum(smart.default_curriculum)
    assert train["evasion_schedule"] == cgf._parse_curriculum(smart.default_evasion_curriculum)
    # Until change 1d (2026-09-12) the Gaussian arm carried a pre-2026-09-08 copy of the
    # parser that padded to FOUR fields and refused a fifth (spread gain), pinned here as
    # a known drift. Every arm now uses set_transformer.rl.curriculum.parse_reward_schedule,
    # so all four parse the recipe to the same five-field waypoints (a missing gain is 0,
    # which is what CurriculumCallback read before, so recorded runs are unaffected).
    parsed = train["reward_schedule"]
    assert parsed == cgf._parse_reward_schedule(smart.default_reward_schedule)
    assert all(len(entry) == 5 for entry in parsed)
    # historical defaults every recorded run used
    assert train["obs_mask_indices"] == [-2, -1]
    assert train["total_timesteps"] == 3_000_000 and train["n_envs"] == 4
    assert train["learning_rate"] == 3e-4 and train["batch_size"] == 64
    assert train["ppo_n_steps"] == 2048 and train["n_epochs"] == 10
    assert train["eval_freq"] == 20_000 and train["save_freq"] == 100_000
    assert train["n_eval_episodes"] == 20 and train["use_vec_normalize"] is True
    assert train["lr_anneal"] is False and train["target_kl"] is None
    assert train["num_particles"] == 100 and train["algorithm"] == "PPO"
    assert train["target_speed_scale"] is None and train["net_arch"] is None
    assert train["device"] == "cuda:1"
    assert train["log_dir"].startswith(str(tmp_path / "run"))


def test_ant_tag_st_geometry_and_start_mode_defaults(ant_tag, monkeypatch, tmp_path):
    """The ST arm's bare defaults are the 8 x 8 = 64-feature encoder with 32 inducing
    points, hidden 128, 4 heads, layer norm and the weight channel on; trained from
    scratch under PPO (no checkpoint, not frozen, shared learning rate, no fork)."""
    _, train = _drive_ant_tag_main(ant_tag, monkeypatch, tmp_path, "4_train_rl_st",
                                   ["--variant", "smart"])
    assert (train["num_encodings"], train["dim_encoder"]) == (8, 8)
    assert (train["num_inds"], train["dim_hidden"], train["num_heads"]) == (32, 128, 4)
    assert train["ln"] is True and train["weight_channel"] is True
    assert train["pretrained_st_model_path"] is None and train["st_frozen"] is False
    assert train["pretrained_path"] is None and train["frozen"] is False
    assert train["encoder_lr_scale"] == 1.0 and train["resume_from"] is None


@pytest.mark.parametrize("module_name,encoder,main_encoder", _ANT_TAG_ARMS, ids=_ANT_TAG_IDS)
def test_ant_tag_ppo_flags_reach_the_training_call(
        ant_tag, monkeypatch, tmp_path, module_name, encoder, main_encoder):
    """`--net_arch a,b` becomes a list, `--lr_anneal`, `--target_kl`, `--no_vec_normalize`,
    `--no_mask_target_obs` and `--target_speed_scale` reach train_* as the values the
    recorded runs used."""
    argv = ["--variant", "smart", "--net_arch", "256,256", "--lr_anneal", "--target_kl", "0.03",
            "--no_vec_normalize", "--no_mask_target_obs", "--target_speed_scale", "0.5",
            "--n_envs", "2", "--seed", "7", "--device", "cpu"]
    config, train = _drive_ant_tag_main(ant_tag, monkeypatch, tmp_path, module_name,
                                        argv, encoder=main_encoder)
    assert train["net_arch"] == [256, 256]
    assert train["lr_anneal"] is True and train["target_kl"] == 0.03
    assert train["use_vec_normalize"] is False
    assert train["obs_mask_indices"] is None
    assert train["target_speed_scale"] == 0.5
    assert train["n_envs"] == 2 and train["seed"] == 7 and train["device"] == "cpu"
    assert config["net_arch"] == "256,256" and config["seed"] == 7


def test_ant_tag_evasion_flags_warn_on_a_non_evading_variant(ant_tag, capsys):
    """`base` has no evading target: --evasion_curriculum / --target_speed_scale would be
    silent no-ops there, so the registry prints a WARNING naming the flags. On an
    evading variant it prints nothing."""
    variants = ant_tag["variants"]
    variants.warn_if_not_evading("base", "0:0,1:1", 0.5)
    out = capsys.readouterr().out
    assert "WARNING" in out and "--evasion_curriculum" in out and "--target_speed_scale" in out
    variants.warn_if_not_evading("smart", "0:0,1:1", 0.5)
    assert capsys.readouterr().out == ""
    variants.warn_if_not_evading("base", None, None)
    assert capsys.readouterr().out == ""


def test_ant_tag_target_speed_scale_is_refused_on_an_env_without_the_knob(ant_tag):
    """The base Ant-Tag target has no cornered-speed knob. Passing the flag there must be
    a hard error, not a silently dropped kwarg the PF would then mirror faithfully."""
    cgf = ant_tag["4_train_rl_cgf"]
    with pytest.raises(ValueError, match="does not support"):
        cgf.make_ant_tag_cgf_env(num_particles=8, env_id="pdomains-ant-tag-v0",
                                 particle_filter_class=AntTagParticleFilter,
                                 target_speed_scale=1.0)()


# ==========================================================================
# 2. Ant-Tag training call: env stack, eval env, VecNormalize, PPO kwargs, callbacks
# ==========================================================================

def _ant_tag_train_kwargs(tmp_path, encoder_options=None, **extractor_kwargs):
    """The shared train() call for a small `smart` run: what the scripts' train_ant_tag_*
    functions received, in the shared shape (extractor kwargs, env options, Schedules)."""
    from set_transformer.rl.domains.ant_tag import make_schedules
    return dict(
        features_extractor_kwargs=dict(arena_scale=4.5, **extractor_kwargs),
        env_options=dict(distance_coeff=1.0, entropy_coeff=0.0, tag_bonus_coeff=0.0,
                         initial_visibility_radius=100.0, obs_mask_indices=None,
                         target_speed_scale=None),
        schedules=make_schedules([(0.0, 100.0), (0.5, 1.5), (1.0, 1.5)],
                                 [(0.0, 1.0, 2.0, 0.0, 0.0), (1.0, 0.15, 2.0, 50.0, 0.0)],
                                 [(0.0, 0.0), (1.0, 1.0)]),
        encoder_options=encoder_options or {},
        total_timesteps=1000, n_envs=1, num_particles=16, device="cpu", seed=3,
        log_dir=str(tmp_path / "logs") + "/",
        model_save_path=str(tmp_path / "models" / "agent.zip"),
        eval_freq=200, save_freq=400, n_eval_episodes=3,
        particle_filter_class=SmartAntTagParticleFilter,
        lr_anneal=True, target_kl=0.03, learning_rate=3e-4,
    )


#: The domain's three schedules ride in ONE ScheduleCallback where the scripts had their
#: CurriculumCallback adapter (same waypoints; tests/test_curriculum.py pins the equivalence).
_WIRING_ARMS = [
    ("cgf", dict(num_cgf_features=8),
     [CheckpointCallback, EvalCallback, "ScheduleCallback", TNormLoggingCallback,
      EncoderDriftLoggingCallback]),
    ("st", dict(num_encodings=2, dim_encoder=4, num_inds=4, dim_hidden=16, num_heads=2),
     [CheckpointCallback, EvalCallback, "ScheduleCallback", STFeatureLoggingCallback]),
    ("gaussian", {},
     [CheckpointCallback, EvalCallback, "ScheduleCallback"]),
]


@pytest.mark.parametrize("encoder,extra,expected_callbacks", _WIRING_ARMS,
                         ids=[a[0] for a in _WIRING_ARMS])
def test_ant_tag_train_wiring(ant_tag, monkeypatch, tmp_path, encoder, extra,
                              expected_callbacks):
    """What the shared train() builds around PPO for an Ant-Tag arm, checked without training:

    * training env: base -> visibility wrapper at the FIRST curriculum radius -> weighted
      PF dict wrapper seeded seed+rank -> reward shaping -> Monitor -> router, inside a
      VecNormalize that normalises ONLY the `obs` key and the reward;
    * eval env: rank n_envs+1, the env's REAL visibility radius, NO shaping, VecNormalize
      with training off and reward un-normalised;
    * PPO gets the historical kwargs, tensorboard under log_dir, and an annealing
      learning-rate schedule when asked;
    * checkpoints every save_freq // n_envs WITH the VecNormalize snapshot, evals every
      eval_freq // n_envs, deterministic, best model under models/;
    * the arm's callback list, in order;
    * the final zip and vecnormalize.pkl are written on exit.
    """
    from set_transformer.rl import train as train_mod
    frozen, cgf = ant_tag["4_train_rl_frozen"], ant_tag["4_train_rl_cgf"]
    evals, checkpoints = _install_fakes(monkeypatch, train_mod)
    kwargs = _ant_tag_train_kwargs(tmp_path, **extra)
    train_mod.train("ant_tag", "smart", encoder, **kwargs)

    (model,) = _FakeModel.instances
    # --- PPO kwargs
    assert model.policy_name == "MultiInputPolicy"
    k = model.kwargs
    assert (k["n_steps"], k["batch_size"], k["n_epochs"]) == (2048, 64, 10)
    assert k["target_kl"] == 0.03 and k["seed"] == 3 and k["device"] == "cpu"
    assert k["tensorboard_log"] == kwargs["log_dir"]
    assert callable(k["learning_rate"]) and k["learning_rate"](0.5) == pytest.approx(1.5e-4)
    # --- training env
    train_vec = model.env
    assert isinstance(train_vec, VecNormalize)
    assert train_vec.norm_obs_keys == ["obs"] and train_vec.norm_reward and train_vec.training
    train_env = _inner_env(train_vec)
    assert isinstance(train_env, frozen._CurriculumRouter)
    order = [type(e).__name__ for e in _stack(train_env)]
    assert order[:5] == ["_CurriculumRouter", "Monitor", "PFRewardShapingWrapper",
                         "PFDictWithWeightsObservationWrapper", "CurriculumVisibilityWrapper"]
    assert _find(train_env, frozen.CurriculumVisibilityWrapper).visibility_radius == 100.0
    assert _find(train_env, PFDictWithWeightsObservationWrapper).particle_filter_seed == 3
    assert _find(train_env, PFDictWithWeightsObservationWrapper).obs_mask_indices is None
    # --- eval env
    ((eval_vec,), eval_kw), = evals
    assert isinstance(eval_vec, VecNormalize)
    assert eval_vec.norm_obs_keys == ["obs"] and not eval_vec.norm_reward and not eval_vec.training
    eval_env = _inner_env(eval_vec)
    assert _find(eval_env, frozen.PFRewardShapingWrapper) is None
    real_radius = cgf.get_env_visible_radius("pdomains-ant-tag-smart-v0")
    assert _find(eval_env, frozen.CurriculumVisibilityWrapper).visibility_radius == real_radius == 3.0
    assert _find(eval_env, PFDictWithWeightsObservationWrapper).particle_filter_seed == 3 + 1 + 1
    assert eval_kw["eval_freq"] == 200 and eval_kw["n_eval_episodes"] == 3
    assert eval_kw["deterministic"] is True and eval_kw["render"] is False
    assert eval_kw["log_path"] == kwargs["log_dir"]
    assert eval_kw["best_model_save_path"] == str(tmp_path / "models" / "best_model")
    # --- checkpoints
    ((), ck_kw), = checkpoints
    assert ck_kw["save_freq"] == 400 and ck_kw["save_vecnormalize"] is True
    assert ck_kw["name_prefix"] == f"ant_tag_{encoder}"
    assert ck_kw["save_path"] == str(tmp_path / "models" / "checkpoints")
    # --- callbacks, in order
    (learn,) = model.learn_calls
    assert learn["total_timesteps"] == 1000
    names = [type(cb).__name__ for cb in learn["callback"]]
    expected = [c if isinstance(c, str) else c.__name__ for c in expected_callbacks]
    assert names == expected
    # --- saved on exit
    assert model.saved == [kwargs["model_save_path"]]
    assert (tmp_path / "models" / "vecnormalize.pkl").exists()


def test_ant_tag_cgf_running_norm_adds_the_rollout_callback_only_when_live(
        ant_tag, monkeypatch, tmp_path):
    """feature_norm=running with running_norm_update=rollout attaches the
    RolloutFeatureNormCallback; feature_norm=none does not; a frozen CGF never does."""
    from set_transformer.rl import train as train_mod

    def callbacks(running_norm_update="rollout", **extra):
        _install_fakes(monkeypatch, train_mod)
        train_mod.train("ant_tag", "smart", "cgf", **_ant_tag_train_kwargs(
            tmp_path, encoder_options=dict(running_norm_update=running_norm_update),
            num_cgf_features=8, **extra))
        (model,) = _FakeModel.instances
        return [type(cb).__name__ for cb in model.learn_calls[0]["callback"]]

    assert "RolloutFeatureNormCallback" in callbacks(feature_norm="running",
                                                     running_norm_update="rollout")
    assert "RolloutFeatureNormCallback" not in callbacks(feature_norm="none")
    assert "RolloutFeatureNormCallback" not in callbacks(feature_norm="running",
                                                         running_norm_update="minibatch")


def test_ant_tag_train_builds_the_requested_extractor(ant_tag, monkeypatch, tmp_path):
    """The arm's extractor class reaches PPO's policy_kwargs with the resolved arena scale,
    which main() defaults to the live env's half-width (4.5 on smart) before the record is
    written; no net_arch unless asked."""
    from set_transformer.rl import train as train_mod
    from set_transformer.rl.encoders import ENCODERS
    for module_name, encoder, cls in (("4_train_rl_cgf", "cgf", WeightedCGFFeaturesExtractor),
                                      ("4_train_rl_st", "st", SetTransformerFeaturesExtractor)):
        # main() with train() intercepted (undone at the end of the block, so the real
        # train() runs below)
        with monkeypatch.context() as mp:
            _, train = _drive_ant_tag_main(ant_tag, mp, tmp_path, module_name,
                                           ["--variant", "smart"])
        assert train["arena_scale"] == pytest.approx(4.5) and train["net_arch"] is None
        assert ENCODERS[encoder].extractor_class is cls
        _install_fakes(monkeypatch, train_mod)
        train_mod.train("ant_tag", "smart", encoder, **_ant_tag_train_kwargs(
            tmp_path, **{"cgf": dict(num_cgf_features=8),
                         "st": dict(num_encodings=2, dim_encoder=4, num_inds=4, dim_hidden=16,
                                    num_heads=2)}[encoder]))
        (model,) = _FakeModel.instances
        pk = model.kwargs["policy_kwargs"]
        assert pk["features_extractor_class"] is cls
        assert pk["features_extractor_kwargs"]["arena_scale"] == pytest.approx(4.5)
        assert "net_arch" not in pk


# ==========================================================================
# 3. Curriculum: interpolation, defaults, and pushing values into the wrappers
# ==========================================================================

def test_curriculum_interpolation_is_linear_between_waypoints_and_clamped_outside(ant_tag):
    frozen = ant_tag["4_train_rl_frozen"]
    cb = frozen.CurriculumCallback(total_timesteps=100,
                                   schedule=[(0.0, 100.0), (0.2, 100.0), (0.5, 3.0), (1.0, 3.0)])
    radius = lambda p: cb._interpolate_schedule(cb.schedule, p)[0]  # noqa: E731
    assert radius(-0.1) == 100.0 and radius(0.0) == 100.0 and radius(0.1) == 100.0
    assert radius(0.35) == pytest.approx(51.5)
    assert radius(0.5) == 3.0 and radius(0.9) == 3.0 and radius(1.5) == 3.0
    # coefficient schedules interpolate every field
    cb2 = frozen.CurriculumCallback(
        total_timesteps=100,
        reward_schedule=[(0.0, 1.0, 2.0, 0.0, 0.0), (1.0, 0.0, 2.0, 50.0, 10.0)])
    assert cb2._interpolate_schedule(cb2.reward_schedule, 0.5) == pytest.approx((0.5, 2.0, 25.0, 5.0))


def test_curriculum_defaults_are_the_historical_script_schedules(ant_tag):
    """No schedule given: visibility 100 -> 3 between 30% and 70%; distance 1 -> 0 and
    tag bonus 0 -> 50 over the same window with PF-entropy 0 throughout; evasion a
    CONSTANT 1.0 (full-strength target, i.e. no curriculum)."""
    frozen = ant_tag["4_train_rl_frozen"]
    cb = frozen.CurriculumCallback(total_timesteps=100)
    assert cb.schedule == [(0.0, 100.0), (0.3, 100.0), (0.7, 3.0), (1.0, 3.0)]
    assert cb.reward_schedule == [(0.0, 1.0, 0.0, 0.0), (0.3, 1.0, 0.0, 0.0),
                                  (0.7, 0.0, 0.0, 50.0), (1.0, 0.0, 0.0, 50.0)]
    assert cb.evasion_schedule == [(0.0, 1.0), (1.0, 1.0)]
    assert cb._interpolate_schedule(cb.evasion_schedule, 0.37) == (1.0,)
    assert cb._interpolate_schedule(cb.reward_schedule, 0.5) == pytest.approx((0.5, 0.0, 25.0))


def test_curriculum_apply_pushes_radius_coefficients_and_evasion_into_the_env(ant_tag):
    """_apply(progress) reaches, through the router, the visibility wrapper (which also
    publishes the live radius onto the base env for the PF mapper), the shaping
    wrapper's four coefficients, and the smart target's evasion_scale."""
    frozen, cgf = ant_tag["4_train_rl_frozen"], ant_tag["4_train_rl_cgf"]
    venv = DummyVecEnv([cgf.make_ant_tag_cgf_env(
        num_particles=8, env_id="pdomains-ant-tag-smart-v0",
        particle_filter_class=SmartAntTagParticleFilter, seed=0)])
    try:
        cb = frozen.CurriculumCallback(
            total_timesteps=100, schedule=[(0.0, 100.0), (1.0, 1.0)],
            reward_schedule=[(0.0, 1.0, 2.0, 0.0, 0.0), (1.0, 0.15, 2.0, 50.0, 10.0)],
            evasion_schedule=[(0.0, 0.0), (1.0, 1.0)], verbose=0)
        cb.model = SimpleNamespace(get_env=lambda: venv)
        cb._apply(0.5)
        env = venv.envs[0]
        vis = _find(env, frozen.CurriculumVisibilityWrapper)
        shaping = _find(env, frozen.PFRewardShapingWrapper)
        assert vis.visibility_radius == pytest.approx(50.5)
        assert env.unwrapped.current_visibility_radius == pytest.approx(50.5)
        assert (shaping.distance_coeff, shaping.entropy_coeff, shaping.tag_bonus_coeff,
                shaping.gain_coeff) == pytest.approx((0.575, 2.0, 25.0, 5.0))
        assert env.unwrapped.evasion_scale == pytest.approx(0.5)
    finally:
        venv.close()


def test_reward_schedule_parser_pads_three_and_four_field_entries_with_zeros(ant_tag):
    cgf = ant_tag["4_train_rl_cgf"]
    parsed = cgf._parse_reward_schedule("0:1:0,0.5:1:0:50,1:0.15:2:50:10")
    assert parsed == [(0.0, 1.0, 0.0, 0.0, 0.0), (0.5, 1.0, 0.0, 50.0, 0.0),
                      (1.0, 0.15, 2.0, 50.0, 10.0)]
    with pytest.raises(ValueError):
        cgf._parse_reward_schedule("0:1")
    assert cgf._parse_curriculum("0:100, 0.4:1.5") == [(0.0, 100.0), (0.4, 1.5)]


def test_reward_schedule_parser_is_one_function_in_every_arm(ant_tag):
    """Until 2026-09-12 this test pinned a KNOWN DRIFT: the Gaussian arm's own copy of the
    reward-schedule parser padded three fields to four and raised on five, so a spread-gain
    schedule could not be passed to that arm. Change 1d replaced every copy with
    set_transformer.rl.curriculum.parse_reward_schedule, so the test is flipped on purpose:
    all four arms expose the SAME function, and the Gaussian arm gains the fifth field.
    Every three- and four-field schedule parses to the same coefficients as before (the
    padding is zeros, and the callback read a missing spread gain as 0), so recorded
    Gaussian runs are unaffected."""
    from set_transformer.rl.curriculum import parse_reward_schedule

    five = "0:1:2:0:0,1:0.15:2:50:10"
    four = "0:1:2:0,1:0.15:2:50"
    arms = [ant_tag[n] for n in ("4_train_rl_cgf", "4_train_rl_st", "4_train_rl_pool", "4_train_rl_gaussian")]
    assert all(arm._parse_reward_schedule is parse_reward_schedule for arm in arms)
    assert parse_reward_schedule(five) == [(0.0, 1.0, 2.0, 0.0, 0.0), (1.0, 0.15, 2.0, 50.0, 10.0)]
    assert parse_reward_schedule(four) == [(0.0, 1.0, 2.0, 0.0, 0.0), (1.0, 0.15, 2.0, 50.0, 0.0)]
    with pytest.raises(ValueError):
        parse_reward_schedule("0:1:2:0:0:7")


# ==========================================================================
# 4. Run records: run-dir name, run_config.json, git provenance (both domains)
# ==========================================================================

def test_run_dir_name_shape_and_tag_sanitising(ant_tag, odd_even):
    pattern = r"runs/ant_tag_cgf_smart/\d{8}_\d{6}_seed3_6M_vis_0\.2$"
    ant = ant_tag["4_train_rl_cgf"]._default_run_dir(3, "ant_tag_cgf_smart", run_tag="6M vis/0.2")
    assert re.fullmatch(pattern, ant), ant
    bare = ant_tag["4_train_rl_cgf"]._default_run_dir(0, "ant_tag_cgf")
    assert re.fullmatch(r"runs/ant_tag_cgf/\d{8}_\d{6}_seed0$", bare), bare
    oe = odd_even["4_train_rl_cgf"]._default_run_dir(5, "odd_even_cgf_oe50", run_tag="a b")
    assert re.fullmatch(r"runs/odd_even_cgf_oe50/\d{8}_\d{6}_seed5_a_b$", oe), oe


def test_write_run_config_is_sorted_json_that_stringifies_odd_values(ant_tag, tmp_path):
    ant_tag["4_train_rl_cgf"]._write_run_config(
        str(tmp_path / "r"), zeta=1, alpha=SmartAntTagParticleFilter, mid=[1, 2])
    text = (tmp_path / "r" / "run_config.json").read_text()
    data = json.loads(text)
    assert list(data) == ["alpha", "mid", "zeta"]
    assert data["alpha"].startswith("<class ") and data["mid"] == [1, 2]


def test_git_provenance_records_both_repos_and_never_raises(ant_tag, odd_even):
    for module in (ant_tag["4_train_rl_cgf"], odd_even["4_train_rl_cgf"]):
        prov = module._git_provenance()
        assert set(prov) == {"set_transformer", "pomdp-domains"}
        for entry in prov.values():
            assert "path" in entry
            assert ("error" in entry) or {"head", "dirty", "diff_sha256", "status"} <= set(entry)


# ==========================================================================
# 5. The Ant-Tag eval script
# ==========================================================================

def _load_ant_tag_eval(ant_tag):
    """The CGF eval entry point (a forwarder since change 5.1) and the shared script it
    forwards to; the fakes are installed on the shared module, which owns the loop."""
    import set_transformer.rl.eval_true_reward as shared
    path = _ANT_TAG_DIR / "eval_scripts" / "eval_true_reward_cgf.py"
    key = "_inventory_eval_true_reward_cgf"
    spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(spec)
    with _sys_path(_ANT_TAG_DIR, path.parent):
        spec.loader.exec_module(module)
    return module, shared


class _ZeroPolicy:
    def predict(self, obs, deterministic=True):
        return np.zeros((1, 8), dtype=np.float32), None


def test_ant_tag_eval_defaults_the_cap_to_the_registry_and_reseeds_after_load(
        ant_tag, monkeypatch, capsys, tmp_path):
    """The eval reads the episode cap off the variant's gym registration (a wrong cap
    counts timeouts as tags), builds the env at the real radius without shaping, and
    re-applies --seed to the vec env AFTER PPO.load (which would otherwise restore the
    training seed and replay one episode set). Ant-Tag seeds ONCE (no per-episode
    re-seeding), as the standalone script did."""
    module, shared = _load_ant_tag_eval(ant_tag)
    seeds = []

    class RecordingDummyVecEnv(DummyVecEnv):
        def seed(self, seed=None):
            seeds.append(seed)
            return super().seed(seed)

    monkeypatch.setattr(shared, "DummyVecEnv", RecordingDummyVecEnv)
    monkeypatch.setattr(shared, "PPO", SimpleNamespace(load=lambda path, env=None: _ZeroPolicy()))
    monkeypatch.setattr(sys, "argv", ["eval_true_reward_cgf.py", "--variant", "smart",
                                      "--model_path", "/nonexistent/agent.zip",
                                      "--num_particles", "16", "--n_episodes", "1", "--seed", "123",
                                      "--output_root", str(tmp_path / "runs")])
    module.main()
    out = capsys.readouterr().out
    assert "episode cap: 400" in out
    assert "Eval episode seed: 123" in out
    assert seeds == [123]
    assert "Success rate  : " in out
    assert "Median length : " in out    # the wave drivers grep this line
    # Change 5.2: the report is also a JSON record under <root>/ant_tag/smart/eval/.
    [summary] = list((tmp_path / "runs" / "ant_tag" / "smart" / "eval").glob("*_agent_seed123_1ep.json"))
    record = json.loads(summary.read_text())
    assert record["variant"] == "smart" and record["seed"] == 123 and record["episode_cap"] == 400
    assert record["report"]["n_episodes"] == 1 and "success_rate" in record["report"]
    assert record["references"] is None and record["run_status"] is None


def test_ant_tag_eval_refuses_a_particle_count_that_contradicts_the_checkpoint(
        ant_tag, monkeypatch):
    module, shared = _load_ant_tag_eval(ant_tag)
    monkeypatch.setattr(shared, "checkpoint_num_particles", lambda path: 100)
    monkeypatch.setattr(sys, "argv", ["eval_true_reward_cgf.py", "--variant", "smart",
                                      "--model_path", "x.zip", "--num_particles", "16"])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2


def test_ant_tag_eval_env_is_the_domain_eval_env(ant_tag):
    """`rl.domains.ant_tag.make_eval_env` (what the diagnostics call) and
    `Domain.make_env(training=False)` (what the shared script calls) build the same stack: real visibility radius, no shaping,
    Monitor on the env's own reward, seeded particle filter."""
    from set_transformer.rl.domains.ant_tag import (
        ANT_TAG, PFRewardShapingWrapper, make_eval_env, resolve)

    def _layers(env):
        names = []
        cur = env
        while cur is not None:
            names.append(type(cur).__name__)
            cur = getattr(cur, "env", None)
        return names

    variant = resolve("smart")
    a = make_eval_env(16, [-2, -1], 5, env_id=variant.env_id,
                      particle_filter_class=variant.particle_filter)()
    b = ANT_TAG.make_env("smart", num_particles=16, particle_filter_class=variant.particle_filter,
                         seed=5, rank=0, monitor_dir=None, training=False,
                         options=ANT_TAG.evaluation.options(SimpleNamespace(no_mask=False)))()
    try:
        assert _layers(a) == _layers(b)
        assert PFRewardShapingWrapper.__name__ not in _layers(a)
        for env in (a, b):
            pf = next(w for w in _walk(env) if hasattr(w, "particle_filter_seed"))
            assert pf.particle_filter_seed == 5
            assert pf.obs_mask_indices == [-2, -1]
    finally:
        a.close()
        b.close()


def _walk(env):
    cur = env
    while cur is not None:
        yield cur
        cur = getattr(cur, "env", None)


def test_bare_entry_points_write_under_the_root_layout(ant_tag, odd_even, monkeypatch, tmp_path):
    """Change 5.2: a bare `4_train_rl_<enc>.py --variant v` writes to
    <root>/<domain>/<variant>/rl/<encoder>/<timestamp>_seed<seed>[_<tag>]/, root = --output_root
    > $RL_BMDP_RUNS > the parent repo's runs/, never the current directory. --run_subdir is a
    raw override of the <variant>/rl/<encoder> part."""
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.chdir(tmp_path / "elsewhere") if (tmp_path / "elsewhere").mkdir() is None else None
    root = tmp_path / "root"
    ant_tag["4_train_rl_cgf"].main(["--variant", "smart", "--seed", "3", "--run_tag", "t",
                                    "--output_root", str(root), "--dry_run"])
    [record] = list(root.glob("ant_tag/smart/rl/cgf/*_seed3_t/run_config.json"))
    config = json.loads(record.read_text())
    assert config["run_directory"] == str(record.parent) and config["output_root"] == str(root.resolve())
    odd_even["4_train_rl_gaussian"].main(["--variant", "oe50_short", "--output_root", str(root), "--dry_run"])
    assert list(root.glob("odd_even/oe50_short/rl/gaussian/*_seed0/run_config.json"))
    ant_tag["4_train_rl_pool"].main(encoder="kmoments", argv=["--variant", "smart", "--run_subdir", "sweep",
                                                              "--output_root", str(root), "--dry_run"])
    assert list(root.glob("ant_tag/sweep/*_seed0/run_config.json"))
    # The environment variable is the second choice, and nothing landed in the cwd.
    monkeypatch.setenv(run_records.OUTPUT_ROOT_ENV, str(tmp_path / "env_root"))
    ant_tag["4_train_rl_st"].main(["--variant", "smart", "--dry_run"])
    assert list((tmp_path / "env_root").glob("ant_tag/smart/rl/st/*_seed0/run_config.json"))
    assert not (tmp_path / "elsewhere" / "runs").exists()


# ==========================================================================
# 6. Odd-Even main(): registry-derived defaults, filter override, encoder recipes
# ==========================================================================

@pytest.mark.parametrize("arm", ["cgf", "st", "gaussian"])
def test_odd_even_bare_run_takes_its_defaults_from_the_registry(odd_even, monkeypatch, tmp_path, arm):
    """`--variant oe50_short` alone: particle count = n_dist_size (an EXACT belief), arena
    scale = (n-1)/2, the exact-support filter, CPU, 1M steps, and a run record naming
    the encoder, env, n, cap and filter."""
    variants = odd_even["variants"]
    config, train = _drive_odd_even_main(odd_even, monkeypatch, tmp_path, arm,
                                         ["--variant", "oe50_short"])
    from set_transformer.rl.particle_filters.odd_even import OddEvenExactSupportParticleFilter
    assert train["variant"] == "oe50_short" and train["encoder"] == arm
    assert train["num_particles"] == 50
    assert train["policy_kwargs"]["features_extractor_kwargs"]["arena_scale"] == pytest.approx(24.5)
    assert train["particle_filter_class"] is OddEvenExactSupportParticleFilter
    assert train["device"] == "cpu" and train["total_timesteps"] == 1_000_000
    assert train["n_envs"] == 4 and train["use_vec_normalize"] is True
    assert train["run_subdir"] == variants.run_subdir(arm, "oe50_short") == f"odd_even_{arm}_oe50_short"
    assert config["encoder"] == arm and config["env_id"] == "pdomains-odd-even-50-short-v0"
    assert config["n_dist_size"] == 50 and config["episode_cap"] == 30
    assert config["particle_filter_class"] == "OddEvenExactSupportParticleFilter"
    assert "list_variants" not in config and "run_subdir" in config


def test_odd_even_bootstrap_filter_is_an_arm_selected_by_flag(odd_even, monkeypatch, tmp_path):
    from set_transformer.rl.particle_filters.odd_even import OddEvenBootstrapParticleFilter
    config, train = _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "gaussian",
                                         ["--variant", "oe50_short", "--particle_filter", "bootstrap",
                                          "--num_particles", "80"])
    assert train["particle_filter_class"] is OddEvenBootstrapParticleFilter
    assert train["num_particles"] == 80
    assert config["particle_filter_class"] == "OddEvenBootstrapParticleFilter"


def test_odd_even_cgf_defaults_are_the_tanh_running_recipe(odd_even, monkeypatch, tmp_path):
    """spread_1d init, tanh bound 50 with init ceiling 40, K features, running per-feature
    norm refreshed per rollout; callbacks t-norm, drift and the rollout norm refresh.
    clamp mode: no bound, ceiling = the clamp (2.0). feature_norm none: no refresh."""
    _, train = _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "cgf", ["--variant", "oe50_short"])
    kw = train["policy_kwargs"]["features_extractor_kwargs"]
    assert kw["t_init_mode"] == "spread_1d" and kw["t_param"] == "tanh"
    assert kw["t_bound"] == 50.0 and kw["t_init_max"] == 40.0
    assert kw["feature_mode"] == "K" and kw["feature_norm"] == "running"
    assert kw["num_cgf_features"] == 64 and kw["pretrained_cgf_model_path"] is None
    assert train["pretrained_path"] is None
    assert _shared_callback_names("cgf", train) == \
        ["TNormLoggingCallback", "EncoderDriftLoggingCallback", "RolloutFeatureNormCallback"]

    _, train = _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "cgf",
                                    ["--variant", "oe50_short", "--t_param", "clamp"])
    kw = train["policy_kwargs"]["features_extractor_kwargs"]
    assert kw["t_bound"] is None and kw["t_init_max"] == 2.0 and kw["t_clamp"] == 2.0

    _, train = _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "cgf",
                                    ["--variant", "oe50_short", "--feature_norm", "none"])
    assert _shared_callback_names("cgf", train) == \
        ["TNormLoggingCallback", "EncoderDriftLoggingCallback"]


def test_odd_even_st_geometry_defaults_and_the_checkpoint_rule(odd_even, monkeypatch, tmp_path):
    """Bare: the small encoder (16 inducing points, hidden 64, 2 post-SAB), weight channel
    on, both feature callbacks. With a checkpoint: geometry flags left at default take the
    checkpoint's values and a reload hook is installed; an explicit flag that disagrees
    with the checkpoint is a parser error, not an override."""
    _, train = _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "st", ["--variant", "oe50_short"])
    kw = train["policy_kwargs"]["features_extractor_kwargs"]
    assert (kw["num_inds"], kw["dim_hidden"], kw["num_post_sab"]) == (16, 64, 2)
    assert (kw["num_encodings"], kw["dim_encoder"], kw["num_heads"]) == (8, 8, 4)
    assert kw["ln"] is True and kw["weight_channel"] is True
    assert kw["pretrained_st_model_path"] is None and kw["st_frozen"] is False
    assert train["pretrained_path"] is None
    assert _shared_callback_names("st", train) == \
        ["STFeatureLoggingCallback", "OddEvenSTFeatureSentinel"]

    ckpt = tmp_path / "st.pt"
    torch.save({"model_state_dict": {}, "config": {"num_inds": 8, "dim_hidden": 32, "num_post_sab": 1}}, ckpt)
    _, train = _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "st",
                                    ["--variant", "oe50_short", "--pretrained_st_model_path", str(ckpt)])
    kw = train["policy_kwargs"]["features_extractor_kwargs"]
    assert (kw["num_inds"], kw["dim_hidden"], kw["num_post_sab"]) == (8, 32, 1)
    assert kw["pretrained_st_model_path"] == str(ckpt) and train["pretrained_path"] == str(ckpt)

    with pytest.raises(SystemExit) as exc:
        _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "st",
                             ["--variant", "oe50_short", "--pretrained_st_model_path", str(ckpt),
                              "--num_inds", "16"])
    assert exc.value.code == 2


def test_odd_even_gaussian_has_no_knobs_and_no_callbacks(odd_even, monkeypatch, tmp_path):
    _, train = _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "gaussian", ["--variant", "oe50_short"])
    assert train["policy_kwargs"]["features_extractor_kwargs"] == {"arena_scale": pytest.approx(24.5)}
    assert "extra_callbacks" not in train and "post_construct" not in train


def test_every_arm_refuses_a_frozen_encoder_without_a_checkpoint(ant_tag, odd_even, monkeypatch, tmp_path):
    """--st_frozen / --cgf_frozen without a pretrained checkpoint would freeze a RANDOM
    encoder and train for hours as a capacity control nobody asked for. All four
    learned-encoder arms refuse it before anything is built. Until change 4.5 (2026-09-12)
    this pinned a KNOWN DRIFT in HOW: the Ant-Tag ST arm raised ValueError from main() while
    the other three used parser.error. Every arm now goes through the shared start-mode
    resolution, so all four are parser errors (SystemExit, code 2)."""
    cases = [
        (SystemExit, lambda: _drive_ant_tag_main(ant_tag, monkeypatch, tmp_path, "4_train_rl_st",
                                                 ["--variant", "smart", "--st_frozen"])),
        (SystemExit, lambda: _drive_ant_tag_main(ant_tag, monkeypatch, tmp_path, "4_train_rl_cgf",
                                                 ["--variant", "smart", "--cgf_frozen"])),
        (SystemExit, lambda: _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "st",
                                                  ["--variant", "oe50_short", "--st_frozen"])),
        (SystemExit, lambda: _drive_odd_even_main(odd_even, monkeypatch, tmp_path, "cgf",
                                                  ["--variant", "oe50_short", "--cgf_frozen"])),
    ]
    for expected, run in cases:
        with pytest.raises(expected) as exc:
            run()
        assert exc.value.code == 2


# ==========================================================================
# 7. Odd-Even train_odd_even(): env stack, eval env, VecNormalize, callbacks, run_status
# ==========================================================================

def test_odd_even_train_wiring(odd_even, monkeypatch, tmp_path):
    """The shared Odd-Even loop, checked without training: training env = base ->
    step-index obs -> weighted PF dict wrapper (seeded seed+rank, no mask) -> particle
    centring -> Monitor, in a VecNormalize on the `obs` key with reward normalisation;
    eval env identical but rank n_envs+1, training off, reward raw; checkpoints with
    prefix odd_even_<encoder>; extra callbacks appended after checkpoint and eval; the
    zip, vecnormalize.pkl and a `completed` run_status.json written on exit."""
    from set_transformer.rl.feature_extractors.gaussian import WeightedGaussianFeaturesExtractor
    from set_transformer.rl import train as train_mod
    belief = odd_even["odd_even_belief_env"]
    # Since change 4.4 the Odd-Even loop IS set_transformer.rl.train.train (the CGF script's
    # train_odd_even is gone); the fakes go on the shared module.
    evals, checkpoints = _install_fakes(monkeypatch, train_mod)
    marker = TNormLoggingCallback()
    log_dir = str(tmp_path / "logs") + "/"
    model_path = str(tmp_path / "models" / "gaussian_agent.zip")
    assert WeightedGaussianFeaturesExtractor is train_mod._encoders.ENCODERS["gaussian"].extractor_class
    train_mod.train(
        "odd_even", "oe50_short", "gaussian",
        features_extractor_kwargs=dict(arena_scale=24.5), total_timesteps=500, n_envs=1,
        num_particles=50, device="cpu", seed=11, log_dir=log_dir, model_save_path=model_path,
        eval_freq=100, save_freq=200, n_eval_episodes=2, lr_anneal=True, target_kl=0.02,
        extra_callbacks=[marker])
    (model,) = _FakeModel.instances
    k = model.kwargs
    assert k["target_kl"] == 0.02 and k["seed"] == 11 and k["tensorboard_log"] == log_dir
    assert callable(k["learning_rate"]) and k["learning_rate"](0.5) == pytest.approx(1.5e-4)
    # training env
    train_vec = model.env
    assert train_vec.norm_obs_keys == ["obs"] and train_vec.norm_reward and train_vec.training
    train_env = _inner_env(train_vec)
    assert [type(e).__name__ for e in _stack(train_env)][:4] == \
        ["Monitor", "ParticleCentringWrapper", "OddEvenPFDictWrapper", "StepIndexObservationWrapper"]
    pf_wrapper = _find(train_env, belief.OddEvenPFDictWrapper)
    assert pf_wrapper.particle_filter_seed == 11 and pf_wrapper.obs_mask_indices is None
    # eval env
    ((eval_vec,), eval_kw), = evals
    assert eval_vec.norm_obs_keys == ["obs"] and not eval_vec.norm_reward and not eval_vec.training
    assert _find(_inner_env(eval_vec), belief.OddEvenPFDictWrapper).particle_filter_seed == 11 + 1 + 1
    assert eval_kw["eval_freq"] == 100 and eval_kw["n_eval_episodes"] == 2 and eval_kw["deterministic"]
    # checkpoints and callbacks
    ((), ck_kw), = checkpoints
    assert ck_kw["save_freq"] == 200 and ck_kw["name_prefix"] == "odd_even_gaussian"
    (learn,) = model.learn_calls
    assert [type(cb).__name__ for cb in learn["callback"]] == \
        ["CheckpointCallback", "EvalCallback", "TNormLoggingCallback"]
    assert learn["callback"][2] is marker
    # written on exit
    assert model.saved == [model_path]
    assert (tmp_path / "models" / "vecnormalize.pkl").exists()
    status = json.loads((tmp_path / "models" / "run_status.json").read_text())
    assert status["status"] == "completed" and status["total_timesteps"] == 500
