"""The shared pretraining command (``rl/pretrain.py``, batch 7.2 of plan section 7): parser
assembly from domain + encoder + objective, objective tables, refusals, placement, the run
record, and the ``3_train_st.py`` entry point's legacy defaults. The numerical parity with
the pre-7.2 script is checked by ``tests/tools/pretrain_parity.py`` (plan 7e-bis), and the
script's command-line behaviour by ``test_pretrain_collect_behaviour_inventory.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from set_transformer.rl import pretrain
from set_transformer.rl import pretrain_objectives
from set_transformer.rl.domains import get as get_domain
from set_transformer.rl.domains.base import Domain, Objective, Pretraining

_ST_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ST = _ST_ROOT / "experiments" / "ant_tag" / "3_train_st.py"


def _load_entry_point():
    spec = importlib.util.spec_from_file_location("pretrain_cmd_test_3_train_st", TRAIN_ST)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    """24 weighted 2-D clouds recorded as `smart`, plus a copy without a variant."""
    tmp = tmp_path_factory.mktemp("pretrain_cmd")
    rng = np.random.default_rng(0)
    particles = rng.normal(0, 1, size=(24, 100, 2)).astype(np.float32)
    weights = rng.random((24, 100)).astype(np.float32)
    weights /= weights.sum(1, keepdims=True)
    path = tmp / "smart_pf_dataset.npz"
    np.savez(path, particles=particles, weights=weights, particle_scale=np.float32(4.5),
             metadata=np.array(json.dumps({"env_id": "pdomains-ant-tag-smart-v0", "variant": "smart",
                                           "particle_scale": 4.5, "particle_centre": 0.0})))
    variantless = tmp / "variantless.npz"
    np.savez(variantless, particles=particles, weights=weights, particle_scale=np.float32(4.5),
             metadata=np.array(json.dumps({"env_id": "synthetic", "particle_scale": 4.5})))
    return path


SMALL = ["--num_epochs", "1", "--batch_size", "8", "--num_workers", "0", "--device", "cpu"]


# ---------------------------------------------------------------------------
# Objective tables and records
# ---------------------------------------------------------------------------

def test_reconstruction_is_the_generic_default_and_every_domain_gets_it():
    assert pretrain_objectives.DEFAULT_OBJECTIVE == "reconstruction"
    for name in ("ant_tag", "odd_even"):
        domain = get_domain(name)
        assert "reconstruction" in pretrain.objectives_for(domain)
        assert pretrain.default_objective_name(domain) == "reconstruction"
        # Neither domain declares an objective of its own yet (7.3 adds Odd-Even's).
        assert domain.pretraining == Pretraining()


def test_a_domain_declared_objective_joins_the_table_and_may_not_shadow_a_generic_one():
    own = Objective(name="exact_posterior", description="d", add_arguments=lambda p, d: None,
                    run=lambda a, c: None, run_name=lambda a, now: "x")
    base = get_domain("odd_even")
    fields = {f: getattr(base, f) for f in base.__dataclass_fields__}
    fields["pretraining"] = Pretraining(objectives={"exact_posterior": own},
                                        default_objective="exact_posterior")
    domain = Domain(**fields)
    table = pretrain.objectives_for(domain)
    assert set(table) == {"reconstruction", "exact_posterior"}
    assert pretrain.default_objective_name(domain) == "exact_posterior"
    fields["pretraining"] = Pretraining(objectives={"reconstruction": own})
    with pytest.raises(ValueError, match="also a generic objective"):
        pretrain.objectives_for(Domain(**fields))


def test_listings_exit_before_anything_is_built(capsys):
    assert pretrain.main(["--list_encoders"]) is None
    assert "cgf" in capsys.readouterr().out
    assert pretrain.main(["--list_objectives", "--domain", "ant_tag"]) is None
    out = capsys.readouterr().out
    assert "reconstruction" in out and "generic" in out
    assert pretrain.main(["--list_variants", "--domain", "odd_even", "--encoder", "st"]) is None
    assert "oe50_short" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_analytic_encoders_are_refused(dataset, capsys):
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--domain", "ant_tag", "--encoder", "gaussian", "--data_path", str(dataset)])
    assert exc.value.code == 2
    assert "has no parameters to pretrain" in capsys.readouterr().err


def test_encoder_is_required_for_the_package_command(dataset, capsys):
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--domain", "ant_tag", "--data_path", str(dataset)])
    assert exc.value.code == 2 and "--encoder is required" in capsys.readouterr().err


def test_unknown_objective_is_refused(dataset, capsys):
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--domain", "ant_tag", "--encoder", "st", "--objective", "nope",
                       "--data_path", str(dataset)])
    assert exc.value.code == 2 and "unknown objective 'nope'" in capsys.readouterr().err


def test_reconstruction_post_sab_count_is_configurable_and_recorded(dataset, tmp_path, monkeypatch):
    """Any count >= 0 of SAB blocks after the PMA (user decision 2026-09-13; the old script
    always built 2 because TrainingConfig had no field). The checkpoint records it, so the RL
    extractor builds the same shape, and the command's own round-trip check passes."""
    import re
    from set_transformer.rl import run_records
    from set_transformer.rl.encoders import _checkpoint_config_for_flags
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    for count in (0, 3):
        result = pretrain.main(["--domain", "ant_tag", "--encoder", "st", "--data_path", str(dataset), *SMALL,
                                "--num_inds", "8", "--dim_hidden", "16", "--eval_freq", "2", "--save_freq", "1000",
                                "--warmup_epochs", "1", "--num_post_sab", str(count),
                                "--base_dir", str(tmp_path / f"b{count}")])
        state = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
        dec_blocks = {int(m.group(1)) for k in state for m in [re.match(r"set_transformer\.dec\.(\d+)\.", k)] if m}
        assert dec_blocks == set(range(count + 2))            # PMA, `count` SABs, the output Linear
        assert _checkpoint_config_for_flags(str(result.rl_checkpoint))["num_post_sab"] == count
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "ant_tag", "--encoder", "st", "--data_path", str(dataset), *SMALL,
                       "--num_post_sab", "-1", "--base_dir", str(tmp_path / "neg"), "--dry_run"])


def test_a_dataset_without_a_variant_needs_a_domain_or_a_base_dir(dataset, capsys):
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--encoder", "st", "--data_path", str(dataset.parent / "variantless.npz"), *SMALL])
    assert exc.value.code == 2 and "records no variant" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Placement, the run record, the domain located from the dataset
# ---------------------------------------------------------------------------

def test_dry_run_locates_the_domain_from_the_dataset_and_writes_the_record_under_the_root(
        dataset, tmp_path, monkeypatch):
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    root = tmp_path / "root"
    assert pretrain.main(["--encoder", "st", "--data_path", str(dataset), *SMALL,
                          "--output_root", str(root), "--dry_run"]) is None
    [record] = list(root.glob("ant_tag/smart/pretrain/st_reconstruction/sinkhorn_*/run_config.json"))
    config = json.loads(record.read_text())
    assert (config["domain"], config["encoder"], config["objective"]) == ("ant_tag", "st", "reconstruction")
    assert config["variant"] == "smart" and config["experiment_name"] == "st_reconstruction"
    assert config["num_particles"] == 100 and config["dim_particles"] == 2 and config["arena_scale"] == 4.5
    assert (config["num_inds"], config["dim_hidden"], config["num_post_sab"]) == (32, 128, 2)   # Ant-Tag defaults
    assert config["threads"]["torch"] == torch.get_num_threads() and config["device"] == "cpu"
    assert config["run_directory"] == str(record.parent)
    assert not (tmp_path / "elsewhere" / "runs").exists()
    assert not list(record.parent.glob("checkpoints/*"))          # dry run: nothing trained


def test_run_tag_and_experiment_name_shape_the_run_folder(dataset, tmp_path, monkeypatch):
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    pretrain.main(["--domain", "ant_tag", "--encoder", "st", "--data_path", str(dataset), *SMALL,
                   "--base_dir", str(tmp_path / "b"), "--experiment_name", "exp", "--run_tag", "a b/c",
                   "--dry_run"])
    [run_dir] = [p for p in (tmp_path / "b" / "exp").iterdir()]
    assert run_dir.name.startswith("sinkhorn_") and run_dir.name.endswith("_a_b_c")


def test_variant_flag_must_belong_to_the_domain(dataset, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--domain", "odd_even", "--encoder", "st", "--data_path", str(dataset), *SMALL,
                       "--base_dir", str(tmp_path), "--dry_run"])
    assert exc.value.code == 2 and "--variant 'smart' is not a odd_even variant" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The 3_train_st.py entry point: historical defaults
# ---------------------------------------------------------------------------

def test_entry_point_fills_its_historical_defaults_only_where_the_user_said_nothing():
    ep = _load_entry_point()
    argv = ep.with_legacy_defaults(["--data_path", "d.npz", "--num_inds", "16"])
    assert argv[:4] == ["--data_path", "d.npz", "--num_inds", "16"]
    filled = dict(zip(argv[4::2], argv[5::2]))
    assert filled == {"--encoder": "st", "--experiment_name": "ant_tag_st", "--dim_encoder": "2",
                      "--dim_hidden": "128", "--num_post_sab": "2"}
    assert "--t_init_mode" not in argv                    # cgf-only defaults stay out for st


def test_entry_point_cgf_defaults_and_the_bound_rule():
    ep = _load_entry_point()
    argv = ep.with_legacy_defaults(["--data_path", "d.npz", "--encoder", "cgf", "--t_param", "polar",
                                    "--t_bound", "9"])
    filled = dict(zip(argv[6::2], argv[7::2]))
    assert filled["--t_init_mode"] == "spread" and filled["--feature_norm"] == "none"
    assert float(filled["--t_init_max"]) == pytest.approx(0.8 * 9)
    argv = ep.with_legacy_defaults(["--data_path", "d.npz", "--encoder", "cgf", "--t_param", "clamp"])
    assert "--t_init_max" not in argv                     # clamp: the extractor's own ceiling, as before
    with pytest.raises(SystemExit):
        ep.with_legacy_defaults(["--data_path", "d.npz", "--encoder", "cgf", "--t_param", "tanh"])


def test_entry_point_accepts_a_variantless_dataset_placed_with_base_dir(dataset, tmp_path, monkeypatch):
    """The old script pretrained any dataset without an env to ask; --base_dir keeps that
    working (Ant-Tag encoder defaults apply), the package command requires --domain."""
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    ep = _load_entry_point()
    ep.main(["--data_path", str(dataset.parent / "variantless.npz"), *SMALL,
             "--base_dir", str(tmp_path / "b"), "--dry_run"])
    [record] = list((tmp_path / "b" / "ant_tag_st").glob("sinkhorn_*/run_config.json"))
    config = json.loads(record.read_text())
    assert config["domain"] == "ant_tag" and config["variant"] is None
    assert (config["dim_encoder"], config["num_inds"], config["dim_hidden"]) == (2, 32, 128)
