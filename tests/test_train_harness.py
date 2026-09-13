"""The shared trainer (change 4 of the harness centralisation, 2026-09-12).

Pins four things about ``rl/train.py``, ``rl/encoders.py`` and the two ``Domain`` records:

* the two Domain records and the encoder table are complete and name the same objects the
  package exports;
* every start-mode alias (``--pretrained_st_model_path`` and friends) stores the same value
  as the generic spelling, and the shared refusals fire;
* the rules the merge tightened or added (a CGF checkpoint fitted at another arena_scale
  is refused on Odd-Even too; the Ant-Tag ST arm takes its geometry off a checkpoint), the
  two run-directory layouts, and the small helpers;
* a tiny training run per domain through the new function completes and saves what the
  eval scripts need (model, VecNormalize, run_status.json, checkpoint + snapshot).

While the numbered scripts still held their own flag resolution (batches 4.1-4.4) this file
also carried a RESOLVER-EQUIVALENCE grid: the run record the shared command line wrote had
to carry every key of the record the arm's own script wrote, with the same value. Every
script is an entry point of the shared command line since change 4.5, so that grid is gone;
the before/after evidence is the parity driver, tests/tools/rl_parity.py, run against master
at each switch (refactor_plans.md, section 6).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ODD_EVEN_DIR = _ST_ROOT / "experiments" / "odd_even"
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
pytest.importorskip("mujoco", reason="Ant-Tag needs MuJoCo")
pytest.importorskip("pdomains", reason="envs are registered by pdomains")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from set_transformer.rl import domains  # noqa: E402
from set_transformer.rl import encoders  # noqa: E402
from set_transformer.rl import run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.curriculum import Schedule  # noqa: E402
from set_transformer.rl.domains import ant_tag as ant_tag_domain  # noqa: E402
from set_transformer.rl.domains import odd_even as odd_even_domain  # noqa: E402
from set_transformer.rl.domains.base import Domain  # noqa: E402
from set_transformer.rl.feature_extractors.cgf import WeightedCGFFeaturesExtractor  # noqa: E402
from set_transformer.rl.feature_extractors.pooled import (  # noqa: E402
    PointNetFeaturesExtractor,
    WeightedDeepSetFeaturesExtractor,
)
from set_transformer.rl.feature_extractors.st import SetTransformerFeaturesExtractor  # noqa: E402


# --------------------------------------------------------------------------
# Loading the Odd-Even sentinel forwarding file
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


def _odd_even_sibling():
    key = "_train_harness_oe_sibling_loader"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, _ODD_EVEN_DIR / "_sibling.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _obs_space(particle_dim: int, num_particles: int, obs_dim: int) -> gym.spaces.Dict:
    inf = np.float32(np.inf)
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-inf, inf, (obs_dim,), np.float32),
        "particles": gym.spaces.Box(-inf, inf, (num_particles, particle_dim), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (num_particles,), np.float32),
    })


ANT_TAG_SPACE = _obs_space(2, 100, 31)
ODD_EVEN_SPACE = _obs_space(1, 50, 1)
ODD_EVEN_SCALE = 24.5          # state_scale("oe50_short") = (50 - 1) / 2
ANT_TAG_SMART_SCALE = 4.5      # the smart env's cage half-width


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Tiny pretraining checkpoints in the two formats the resolvers read."""
    root = tmp_path_factory.mktemp("checkpoints")
    out = {}
    torch.manual_seed(0)
    cgf = WeightedCGFFeaturesExtractor(
        ANT_TAG_SPACE, num_cgf_features=8, arena_scale=ANT_TAG_SMART_SCALE, t_param="polar",
        t_bound=9.0, t_init_mode="spread", t_init_max=7.2, feature_mode="K_grad",
        readout_hidden=16, readout_depth=1)
    torch.save({"model_state_dict": cgf.state_dict(), "config": dict(cgf._cgf_geometry)},
               root / "ant_tag_cgf.pt")
    cgf = WeightedCGFFeaturesExtractor(
        ODD_EVEN_SPACE, num_cgf_features=8, arena_scale=ODD_EVEN_SCALE, t_param="tanh",
        t_bound=50.0, t_init_mode="spread_1d", t_init_max=40.0, feature_norm="running")
    torch.save({"model_state_dict": cgf.state_dict(), "config": dict(cgf._cgf_geometry)},
               root / "odd_even_cgf.pt")
    # The Ant-Tag checkpoint carries the Ant-Tag DEFAULT geometry.
    st = SetTransformerFeaturesExtractor(
        ANT_TAG_SPACE, num_encodings=8, dim_encoder=8, num_inds=32, dim_hidden=128,
        num_heads=4, ln=True, arena_scale=ANT_TAG_SMART_SCALE, weight_channel=True)
    torch.save({"model_state_dict": {f"set_transformer.{k}": v
                                     for k, v in st.encoder.state_dict().items()},
                "config": dict(st._st_geometry)}, root / "ant_tag_st.pt")
    # The Odd-Even one is deliberately NOT the default geometry: the resolver must take it
    # from the checkpoint.
    st = SetTransformerFeaturesExtractor(
        ODD_EVEN_SPACE, num_encodings=8, dim_encoder=8, num_inds=8, dim_hidden=32,
        num_heads=4, ln=True, arena_scale=ODD_EVEN_SCALE, weight_channel=True, num_post_sab=1)
    torch.save({"model_state_dict": {f"set_transformer.{k}": v
                                     for k, v in st.encoder.state_dict().items()},
                "config": dict(st._st_geometry)}, root / "odd_even_st.pt")
    for path in root.iterdir():
        out[path.stem] = str(path)
    return out


def _drive_shared(monkeypatch, tmp_path, domain, encoder, argv, *, legacy_layout=False):
    """The shared command line with --dry_run: the run record it writes, and the training
    call it would have made (captured by replacing train()). The run record lands under a
    temporary output root."""
    captured = {}
    monkeypatch.setattr(run_records, "output_root", lambda *a, **k: tmp_path / "runs")
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.setattr(run_records, "write_run_config",
                        lambda run_dir, **cfg: captured.setdefault("config", cfg))
    train_mod.main(list(argv) + ["--dry_run"], domain=domain, encoder=encoder,
                   legacy_layout=legacy_layout)
    return captured["config"]


# --------------------------------------------------------------------------
# 1. The records: two Domains, six Encoders
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,record,module", [
    ("ant_tag", ant_tag_domain.ANT_TAG, ant_tag_domain),
    ("odd_even", odd_even_domain.ODD_EVEN, odd_even_domain),
])
def test_domain_record_is_complete_and_names_the_module_objects(name, record, module):
    assert isinstance(record, Domain) and record.name == name
    assert domains.get(name) is record and domains.get(record) is record
    for field in dataclasses.fields(Domain):
        assert getattr(record, field.name) is not None, field.name
    assert record.variants is module.VARIANTS
    assert record.resolve is module.resolve and record.episode_cap is module.episode_cap
    assert record.default_variant in record.variants
    assert record.particle_dim in (1, 2)
    assert set(record.encoder_defaults) <= set(encoders.ENCODERS)
    assert record.default_num_particles(record.default_variant) > 0
    # schedules(args) and encoder_callbacks(name) are callable on a bare namespace
    assert record.encoder_callbacks("gaussian") == []


def test_unknown_domain_is_refused_with_the_list():
    with pytest.raises(ValueError, match="ant_tag"):
        domains.get("cluster_hunt")


def test_encoder_table_names_the_package_classes_and_their_checkpoint_kwarg():
    assert sorted(encoders.ENCODERS) == ["cgf", "deepset", "gaussian", "kmoments", "pointnet", "st"]
    assert encoders.ENCODERS["cgf"].extractor_class is WeightedCGFFeaturesExtractor
    assert encoders.ENCODERS["st"].extractor_class is SetTransformerFeaturesExtractor
    assert encoders.ENCODERS["deepset"].extractor_class is WeightedDeepSetFeaturesExtractor
    assert encoders.ENCODERS["pointnet"].extractor_class is PointNetFeaturesExtractor
    for name, enc in encoders.ENCODERS.items():
        if enc.learned:
            # The stored name of the checkpoint path IS the extractor's own constructor
            # argument, which the shared reload blanks in policy_kwargs.
            assert enc.pretrained_dest == enc.extractor_class.PRETRAINED_PATH_KWARG, name
            assert enc.frozen_dest and enc.lr_scale_dest and enc.unfreeze_dest
        else:
            assert enc.pretrained_dest is None
    with pytest.raises(ValueError, match="gaussian"):
        encoders.get("moments")


def test_odd_even_domain_attaches_the_sentinel_to_the_st_arm_only():
    from set_transformer.rl.domains.odd_even import OddEvenSTFeatureSentinel
    st_extra = odd_even_domain.ODD_EVEN.encoder_callbacks("st")
    assert len(st_extra) == 1 and isinstance(st_extra[0], OddEvenSTFeatureSentinel)
    assert odd_even_domain.ODD_EVEN.encoder_callbacks("cgf") == []
    assert ant_tag_domain.ANT_TAG.encoder_callbacks("st") == []


def test_sentinel_forwarding_file_hands_back_the_package_objects():
    with _sys_path(_ODD_EVEN_DIR):
        sentinel = _odd_even_sibling().load("st_feature_sentinel")
    from set_transformer.rl.domains import odd_even as oe
    assert sentinel.OddEvenSTFeatureSentinel is oe.OddEvenSTFeatureSentinel
    assert sentinel.relative_feature_spread is oe.relative_feature_spread
    assert sentinel.COLLAPSE_RELATIVE_SPREAD == oe.COLLAPSE_RELATIVE_SPREAD == 5e-3


def test_ant_tag_make_schedules_matches_the_adapter():
    """The trainer builds its Schedules through make_schedules; the scripts through the
    CurriculumCallback adapter. Same waypoints, same order, for defaults and for given lists."""
    cases = [
        (None, None, None),
        ([(0.0, 100.0), (0.5, 1.0), (1.0, 1.0)],
         [(0.5, 0.0, 0.0, 50.0), (0.0, 1.0, 0.0, 0.0)],      # unsorted on purpose
         [(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)]),
        ([(0.0, 2.0)], None, [(1.0, 0.3), (0.0, 0.3)]),
    ]
    for curriculum, reward, evasion in cases:
        adapter = ant_tag_domain.CurriculumCallback(
            1000, schedule=curriculum, reward_schedule=reward, evasion_schedule=evasion)
        built = ant_tag_domain.make_schedules(curriculum, reward, evasion)
        assert built == adapter.schedules
        assert all(isinstance(s, Schedule) for s in built)
        assert [s.target for s in built] == ["set_curriculum_radius", "set_reward_coeffs",
                                             "set_evasion_scale"]


# --------------------------------------------------------------------------
# 2. Start mode: aliases and refusals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("encoder,generic,aliases", [
    ("st", ["--pretrained_path", "x.pt", "--frozen", "--encoder_lr_scale", "0.1",
            "--unfreeze_at", "7"],
     ["--pretrained_st_model_path", "x.pt", "--st_frozen", "--st_encoder_lr_scale", "0.1",
      "--st_unfreeze_at", "7"]),
    ("cgf", ["--pretrained_path", "x.pt", "--frozen"],
     ["--pretrained_cgf_model_path", "x.pt", "--cgf_frozen"]),
    ("deepset", ["--pretrained_path", "x.pt", "--frozen", "--encoder_lr_scale", "0.5"],
     ["--pretrained_model_path", "x.pt", "--frozen", "--encoder_lr_scale", "0.5"]),
])
def test_start_mode_aliases_store_the_same_values_under_the_historical_names(
        encoder, generic, aliases):
    enc = encoders.ENCODERS[encoder]
    parser = train_mod.build_parser(odd_even_domain.ODD_EVEN, enc)
    a, b = parser.parse_args(generic), parser.parse_args(aliases)
    assert vars(a) == vars(b)
    assert getattr(a, enc.pretrained_dest) == "x.pt" and getattr(a, enc.frozen_dest) is True
    start = enc.start_mode(a)
    assert start.pretrained_path == "x.pt" and start.frozen and start.kind == "frozen"


def test_odd_even_st_layer_norm_and_weight_channel_spellings_share_one_dest():
    parser = train_mod.build_parser(odd_even_domain.ODD_EVEN, encoders.ENCODERS["st"])
    assert parser.parse_args([]).ln is True and parser.parse_args([]).weight_channel is True
    assert parser.parse_args(["--no_layer_norm"]).ln is False
    assert parser.parse_args(["--no_ln"]).ln is False
    assert parser.parse_args(["--no_st_weight_channel"]).weight_channel is False
    assert parser.parse_args(["--no_weight_channel"]).weight_channel is False


@pytest.mark.parametrize("argv", [
    ["--frozen"],                                              # frozen without a checkpoint
    ["--encoder_lr_scale", "0.1"],                             # finetune fix without a checkpoint
    ["--unfreeze_at", "5"],                                    # unfreeze without a checkpoint
    ["--pretrained_path", "CKPT", "--encoder_lr_scale", "0"],  # non-positive scale
    ["--pretrained_path", "CKPT", "--frozen", "--encoder_lr_scale", "0.1"],
    ["--pretrained_path", "CKPT", "--frozen", "--unfreeze_at", "5"],
    ["--pretrained_path", "/nowhere/missing.pt"],              # missing file, refused early
    ["--pretrained_path", "CKPT", "--encoder_lr_scale", "0.1", "--algorithm", "SAC"],
])
def test_start_mode_refusals_are_parser_errors(monkeypatch, tmp_path, checkpoints, argv):
    argv = [checkpoints["odd_even_st"] if a == "CKPT" else a for a in argv]
    with pytest.raises(SystemExit):
        _drive_shared(monkeypatch, tmp_path, "odd_even", "st",
                      ["--variant", "oe50_short"] + argv)


def test_analytic_encoders_offer_no_start_mode_flags(monkeypatch, tmp_path):
    for encoder in ("gaussian", "kmoments"):
        with pytest.raises(SystemExit):
            _drive_shared(monkeypatch, tmp_path, "odd_even", encoder,
                          ["--variant", "oe50_short", "--frozen"])


def test_init_policy_excludes_a_pretrained_encoder_and_resume(monkeypatch, tmp_path, checkpoints):
    donor = tmp_path / "donor.zip"
    donor.write_bytes(b"not a real zip; never opened in a dry run")
    with pytest.raises(SystemExit):
        _drive_shared(monkeypatch, tmp_path, "odd_even", "st",
                      ["--variant", "oe50_short", "--init_policy", str(donor),
                       "--pretrained_path", checkpoints["odd_even_st"]])
    with pytest.raises(SystemExit):
        _drive_shared(monkeypatch, tmp_path, "odd_even", "st",
                      ["--variant", "oe50_short", "--init_policy", "/nowhere/donor.zip"])
    config = _drive_shared(monkeypatch, tmp_path, "odd_even", "st",
                           ["--variant", "oe50_short", "--init_policy", str(donor)])
    assert config["init_policy"] == str(donor)


# --------------------------------------------------------------------------
# 3. Rules the merge tightened or added
# --------------------------------------------------------------------------

# (The resolver-equivalence grid that compared each script's record with the shared one
# left with the arm switches, changes 4.3-4.5: the scripts ARE the shared command line now;
# the parity driver, tests/tools/rl_parity.py, covers them against master.)


def test_odd_even_cgf_checkpoint_at_another_arena_scale_is_now_refused(
        monkeypatch, tmp_path, checkpoints):
    """The one Odd-Even tightening: the script silently took a checkpoint's arena_scale; the
    shared resolver applies Ant-Tag's rule and refuses (PITFALLS.md section 4)."""
    with pytest.raises(SystemExit):
        _drive_shared(monkeypatch, tmp_path, "odd_even", "cgf",
                      ["--variant", "oe10", "--pretrained_path", checkpoints["odd_even_cgf"]])


def test_ant_tag_st_takes_geometry_off_a_checkpoint_and_refuses_disagreement(
        monkeypatch, tmp_path, checkpoints):
    """New on Ant-Tag (the script had fixed 32 / 128 defaults and let the extractor raise
    on a mismatch); the Odd-Even rule, applied to both domains."""
    config = _drive_shared(monkeypatch, tmp_path, "ant_tag", "st",
                           ["--variant", "smart", "--pretrained_path", checkpoints["odd_even_st"]])
    assert (config["num_inds"], config["dim_hidden"], config["num_post_sab"]) == (8, 32, 1)
    assert config["pretrained_config"]["num_inds"] == 8
    with pytest.raises(SystemExit):
        _drive_shared(monkeypatch, tmp_path, "ant_tag", "st",
                      ["--variant", "smart", "--pretrained_path", checkpoints["odd_even_st"],
                       "--num_inds", "32"])


# --------------------------------------------------------------------------
# 4. The run directory: root-level layout and the legacy switch
# --------------------------------------------------------------------------

def test_root_layout_and_run_subdir_override(monkeypatch, tmp_path):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    root = tmp_path / "runs"
    train_mod.main(["--variant", "oe50_short", "--seed", "2", "--run_tag", "a b",
                    "--output_root", str(root), "--dry_run"],
                   domain="odd_even", encoder="gaussian")
    [record] = list(root.glob("odd_even/oe50_short/rl/gaussian/*_seed2_a_b/run_config.json"))
    import json
    config = json.loads(record.read_text())
    assert config["run_directory"] == str(record.parent)
    assert config["output_root"] == str(root.resolve())
    assert config["run_subdir"] == "odd_even_gaussian_oe50_short"
    assert "dry_run" not in config and "list_variants" not in config
    train_mod.main(["--variant", "oe50_short", "--run_subdir", "sweep_x",
                    "--output_root", str(root), "--dry_run"],
                   domain="odd_even", encoder="gaussian")
    assert list(root.glob("odd_even/sweep_x/*_seed0/run_config.json"))


def test_legacy_layout_keeps_the_cwd_relative_run_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.chdir(tmp_path)
    train_mod.main(["--variant", "oe50_short", "--dry_run"], domain="odd_even",
                   encoder="gaussian", legacy_layout=True)
    [record] = list((tmp_path / "runs" / "odd_even_gaussian_oe50_short").glob("*_seed0/run_config.json"))
    import json
    config = json.loads(record.read_text())
    assert config["output_root"] is None and config["run_directory"] == str(record.parent)


def test_module_entry_needs_domain_and_encoder_or_a_listing(capsys):
    with pytest.raises(SystemExit):
        train_mod.main(["--variant", "smart"])
    assert train_mod.main(["--list_encoders"]) is None
    assert "WeightedCGFFeaturesExtractor" in capsys.readouterr().out
    assert train_mod.main(["--domain", "odd_even", "--encoder", "st", "--list_variants"]) is None
    assert "oe50_short" in capsys.readouterr().out


# --------------------------------------------------------------------------
# 5. Small helpers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("zip_name,expected", [
    ("ant_tag_st_100000_steps.zip", "ant_tag_st_vecnormalize_100000_steps.pkl"),
    ("odd_even_cgf_4096_steps.zip", "odd_even_cgf_vecnormalize_4096_steps.pkl"),
    ("st_agent.zip", "vecnormalize.pkl"),
    ("cgf_agent.zip", "vecnormalize.pkl"),
])
def test_resume_vecnormalize_path(zip_name, expected):
    assert run_records.resume_vecnormalize_path(f"/x/models/{zip_name}") == f"/x/models/{expected}"


def test_resume_vecnormalize_path_rejects_unknown():
    with pytest.raises(ValueError, match="resume_vecnormalize"):
        run_records.resume_vecnormalize_path("/x/models/best_model.zip")


@dataclasses.dataclass
class _Cfg:
    """Stands in for a Trainer checkpoint's TrainingConfig (module-level so torch can pickle it)."""
    num_inds: int = 4
    dim_hidden: int = 8


def test_checkpoint_config_reads_dicts_and_dataclasses(tmp_path):
    torch.save({"model_state_dict": {}, "config": _Cfg()}, tmp_path / "dc.pt")
    torch.save({"model_state_dict": {}, "config": {"num_inds": 3}}, tmp_path / "d.pt")
    torch.save({"model_state_dict": {}}, tmp_path / "none.pt")
    assert encoders.checkpoint_config(str(tmp_path / "dc.pt")) == {"num_inds": 4, "dim_hidden": 8}
    assert encoders.checkpoint_config(str(tmp_path / "d.pt")) == {"num_inds": 3}
    assert encoders.checkpoint_config(str(tmp_path / "none.pt")) is None
    assert encoders.checkpoint_config(None) is None


@pytest.mark.parametrize("build", [
    lambda: SetTransformerFeaturesExtractor(ODD_EVEN_SPACE, num_encodings=2, dim_encoder=4,
                                            num_inds=4, dim_hidden=16, num_heads=2,
                                            arena_scale=ODD_EVEN_SCALE),
    lambda: WeightedCGFFeaturesExtractor(ODD_EVEN_SPACE, num_cgf_features=8,
                                         arena_scale=ODD_EVEN_SCALE, t_param="tanh",
                                         t_bound=50.0, feature_norm="running",
                                         readout_hidden=8, readout_depth=1),
    lambda: WeightedDeepSetFeaturesExtractor(ANT_TAG_SPACE, num_encodings=2, dim_encoder=4,
                                             dim_hidden=16, arena_scale=4.5),
    lambda: PointNetFeaturesExtractor(ANT_TAG_SPACE, num_encodings=2, dim_encoder=4,
                                      dim_hidden=16, arena_scale=4.5),
], ids=["st", "cgf", "deepset", "pointnet"])
def test_unfreeze_undoes_freeze_on_every_learned_extractor(build):
    extractor = build()
    params = extractor.encoder_parameters()
    extractor.freeze()
    assert not any(p.requires_grad for p in params)
    released = extractor.unfreeze()
    assert released == len(params) and released > 0
    assert all(p.requires_grad for p in params)
    assert not getattr(extractor, "st_frozen", False) and not getattr(extractor, "_frozen", False)
    # a CGF held in eval mode by freeze() follows the policy's mode again
    extractor.train()
    assert extractor.training


# --------------------------------------------------------------------------
# 6. Smoke trains through the new function (slow; the real gate)
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_smoke_train_odd_even_gaussian_saves_everything_the_eval_needs(tmp_path):
    from set_transformer.rl.run_records import read_run_status
    log_dir = tmp_path / "logs"
    model_path = tmp_path / "models" / "gaussian_agent.zip"
    model = train_mod.train(
        "odd_even", "oe50_short", "gaussian",
        features_extractor_kwargs=dict(arena_scale=ODD_EVEN_SCALE),
        num_particles=50, n_envs=1, seed=0, total_timesteps=64, ppo_n_steps=32,
        batch_size=16, n_epochs=1, device="cpu", eval_freq=32, save_freq=32,
        n_eval_episodes=1, log_dir=str(log_dir) + "/", model_save_path=str(model_path))
    assert model.num_timesteps == 64 and model_path.exists()
    assert (tmp_path / "models" / "vecnormalize.pkl").exists()
    assert read_run_status(str(model_path))["status"] == "completed"
    checkpoints = tmp_path / "models" / "checkpoints"
    assert (checkpoints / "odd_even_gaussian_32_steps.zip").exists()
    # the VecNormalize snapshot beside every checkpoint (new on Odd-Even, PITFALLS section 7)
    assert (checkpoints / "odd_even_gaussian_vecnormalize_32_steps.pkl").exists()
    assert (log_dir / "evaluations.npz").exists()
    PPO.load(str(model_path), device="cpu")


@pytest.mark.slow
def test_smoke_train_odd_even_st_pretrained_frozen_and_unfreeze(monkeypatch, tmp_path,
                                                                checkpoints, capsys):
    """Through main(): reload-after-PPO with verification, the Odd-Even sentinel, the
    unfreeze step -- the post-construction path a frozen and a finetune arm take."""
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    common = ["--variant", "oe50_short", "--total_timesteps", "64", "--n_envs", "1",
              "--ppo_n_steps", "32", "--batch_size", "16", "--n_epochs", "1", "--device", "cpu",
              "--eval_freq", "1000000", "--save_freq", "1000000", "--n_eval_episodes", "1",
              "--output_root", str(tmp_path / "root"),
              "--pretrained_path", checkpoints["odd_even_st"]]
    train_mod.main(common + ["--frozen", "--log_dir", str(tmp_path / "frozen" / "logs") + "/",
                             "--model_save_path", str(tmp_path / "frozen" / "models" / "st_agent.zip")],
                   domain="odd_even", encoder="st")
    out = capsys.readouterr().out
    assert "RE-loaded after PPO construction" in out and "re-frozen" in out
    assert "Verified: SetTransformerFeaturesExtractor encoder matches" in out
    assert "ST geometry: num_inds=8 dim_hidden=32 num_post_sab=1 (checkpoint)" in out
    train_mod.main(common + ["--unfreeze_at", "32", "--encoder_lr_scale", "0.5",
                             "--log_dir", str(tmp_path / "unfreeze" / "logs") + "/",
                             "--model_save_path", str(tmp_path / "unfreeze" / "models" / "st_agent.zip")],
                   domain="odd_even", encoder="st")
    out = capsys.readouterr().out
    assert "encoder frozen until step 32, then finetuned" in out
    assert "encoder UNFROZEN at step" in out
    assert "encoder learning rate scaled by 0.5" in out


@pytest.mark.slow
def test_smoke_train_ant_tag_cgf_through_main_runs_the_schedules(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    from set_transformer.rl.run_records import read_run_status
    model_path = tmp_path / "models" / "cgf_agent.zip"
    train_mod.main(
        ["--variant", "cdens_terminal", "--total_timesteps", "256", "--n_envs", "1",
         "--ppo_n_steps", "128", "--batch_size", "64", "--n_epochs", "1", "--num_particles", "32",
         "--device", "cpu", "--eval_freq", "1000000000", "--save_freq", "128", "--n_eval_episodes", "1",
         "--num_cgf_features", "8", "--t_init_mode", "spread",
         "--curriculum", "0:100,0.5:1,1:1", "--reward_schedule", "0:1:0:0,0.5:0:0:50",
         "--evasion_curriculum", "0:0,0.5:1,1:1", "--target_speed_scale", "0",
         "--output_root", str(tmp_path / "root"),
         "--log_dir", str(tmp_path / "logs") + "/", "--model_save_path", str(model_path)],
        domain="ant_tag", encoder="cgf")
    out = capsys.readouterr().out
    assert "[Curriculum] step=0, progress=0.00, vis_radius=100.00" in out
    assert model_path.exists() and read_run_status(str(model_path))["status"] == "completed"
    assert (tmp_path / "models" / "checkpoints" / "ant_tag_cgf_128_steps.zip").exists()
    assert (tmp_path / "models" / "checkpoints" / "ant_tag_cgf_vecnormalize_128_steps.pkl").exists()
    # the record went to the derived run dir under --output_root, not beside the model
    assert list((tmp_path / "root").glob("ant_tag/cdens_terminal/rl/cgf/*/run_config.json"))
    PPO.load(str(model_path), device="cpu")
