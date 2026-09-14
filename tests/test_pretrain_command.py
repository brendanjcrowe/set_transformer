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
        assert "reconstruction" in pretrain.objectives_for(get_domain(name))
    # Ant-Tag declares nothing of its own: the generic objectives only, reconstruction default.
    ant_tag = get_domain("ant_tag")
    assert ant_tag.pretraining == Pretraining()
    assert pretrain.default_objective_name(ant_tag) == "reconstruction"


def test_odd_even_declares_the_three_exact_posterior_objectives_and_defaults_to_belief_kl(capsys):
    """Batch 7.3: the objectives of 3_pretrain_st_belief.py, declared in rl/domains/odd_even.py
    and NOWHERE else; --objective offers them next to the generic reconstruction, for this
    domain only, and runs belief_kl when --objective is omitted."""
    odd_even = get_domain("odd_even")
    assert set(odd_even.pretraining.objectives) == {"belief_kl", "mode_ce", "state_ce"}
    assert set(pretrain.objectives_for(odd_even)) == {"reconstruction", "belief_kl", "mode_ce", "state_ce"}
    assert pretrain.default_objective_name(odd_even) == "belief_kl"
    assert pretrain.main(["--list_objectives", "--domain", "odd_even"]) is None
    out = capsys.readouterr().out
    for name in ("belief_kl", "mode_ce", "state_ce"):
        assert f"{name:<18} odd_even only" in out, out
    assert "belief_kl" not in pretrain.objectives_for(get_domain("ant_tag"))
    # A domain-declared objective is not reachable without naming its domain.
    with pytest.raises(SystemExit):
        pretrain.main(["--encoder", "st", "--objective", "belief_kl", "--variant", "oe50_short", "--dry_run"])
    assert "a domain-declared objective needs --domain" in capsys.readouterr().err


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
    # 7.5 (decision 4): pretrain/<encoder>/<objective>/<timestamp>_seed<n>/, like an RL run.
    [record] = list(root.glob("ant_tag/smart/pretrain/st/reconstruction/*_seed0/run_config.json"))
    config = json.loads(record.read_text())
    assert (config["domain"], config["encoder"], config["objective"]) == ("ant_tag", "st", "reconstruction")
    assert config["variant"] == "smart" and config["experiment_name"] == "reconstruction" and config["layout"] == "root"
    assert config["checkpoint_dir"] == str(record.parent / "checkpoints")
    assert config["num_particles"] == 100 and config["dim_particles"] == 2 and config["arena_scale"] == 4.5
    assert (config["num_inds"], config["dim_hidden"], config["num_post_sab"]) == (32, 128, 2)   # Ant-Tag defaults
    assert config["threads"]["torch"] == torch.get_num_threads() and config["device"] == "cpu"
    assert config["run_directory"] == str(record.parent)
    assert not (tmp_path / "elsewhere" / "runs").exists()
    assert not list(record.parent.glob("checkpoints/*"))          # dry run: nothing trained


def test_exact_posterior_dry_run_writes_the_record_under_the_root(tmp_path, monkeypatch):
    """The Odd-Even objective through the package command, from a foreign cwd: the geometry the
    objective derives from the variant (50 states, 1-D, scale 24.5), the domain's ST defaults,
    the script's run-folder naming (encoder name inserted for every arm but the ST), the
    default experiment folder <encoder>_belief_pretrain."""
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    root = tmp_path / "root"
    assert pretrain.main(["--domain", "odd_even", "--encoder", "st", "--variant", "oe50_short",
                          "--device", "cpu", "--output_root", str(root), "--dry_run"]) is None
    [record] = list(root.glob("odd_even/oe50_short/pretrain/st/belief_kl/*_seed0/run_config.json"))
    config = json.loads(record.read_text())
    assert (config["domain"], config["encoder"], config["objective"]) == ("odd_even", "st", "belief_kl")
    assert (config["num_particles"], config["dim_particles"], config["arena_scale"]) == (50, 1, 24.5)
    assert (config["num_inds"], config["dim_hidden"], config["num_post_sab"]) == (16, 64, 2)
    assert (config["num_epochs"], config["learning_rate"], config["batch_size"]) == (40, 1e-3, 512)
    assert (config["n_train_episodes"], config["n_val_episodes"], config["data_seed"]) == (4000, 400, 100000)
    assert (config["probe_episodes"], config["probe_seed"], config["probe_splits"]) == (300, 9000, 5)
    assert not (tmp_path / "elsewhere" / "runs").exists()
    assert not list(record.parent.glob("*.pt"))
    assert pretrain.main(["--domain", "odd_even", "--encoder", "cgf", "--objective", "mode_ce", "--variant", "oe50_short",
                          "--match_params", "109448", "--device", "cpu", "--output_root", str(root),
                          "--run_tag", "t", "--dry_run"]) is None
    [record] = list(root.glob("odd_even/oe50_short/pretrain/cgf/mode_ce/*_seed0_t/run_config.json"))
    config = json.loads(record.read_text())
    # Odd-Even's CGF recipe from the encoder table: tanh 50, spread_1d, running norm, t_init_max 40.
    assert (config["t_param"], config["t_bound"], config["t_init_mode"], config["feature_norm"]) == ("tanh", 50.0, "spread_1d", "running")
    assert config["t_init_max"] == 40.0 and config["encoder_params"] > 100_000


def test_exact_posterior_refusals(tmp_path, capsys):
    base = ["--domain", "odd_even", "--variant", "oe50_short", "--dry_run", "--base_dir", str(tmp_path)]
    # 10.4: the pooled arms are accepted (every learned encoder is); an analytic one is refused by
    # the door before the objective sees it
    with pytest.raises(SystemExit):
        pretrain.main(["--encoder", "gaussian", *base])
    assert "has no parameters to pretrain" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "odd_even", "--encoder", "st", "--dry_run", "--base_dir", str(tmp_path)])
    assert "rolls the env itself: pass --variant" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "odd_even", "--encoder", "st", "--dry_run"])
    assert "no dataset was given to read one from" in capsys.readouterr().err     # 10.5: it takes --data_path
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--encoder", "cgf", "--match_params", "2000", "--init_from", "nowhere.pt", *base])
    assert "--init_from is implemented for --encoder st only" in str(exc.value)
    assert not list(tmp_path.iterdir())        # every refusal came before a run folder existed


def test_exact_posterior_package_route_trains_and_round_trips(tmp_path, monkeypatch):
    """A real (tiny) run through `python -m set_transformer.rl.pretrain`: the script's files plus
    run_config.json, the checkpoint in the RL loader's format, the round-trip check passed
    (main raises otherwise), the probe skipped on request."""
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = pretrain.main(["--domain", "odd_even", "--encoder", "st", "--variant", "oe50_short", "--device", "cpu",
                            "--base_dir", str(tmp_path), "--n_train_episodes", "4", "--n_val_episodes", "2",
                            "--num_epochs", "1", "--skip_probe"])
    run_dir = result.run_dir
    assert run_dir.parent == tmp_path / "st_belief_pretrain"
    assert {"args.json", "run_config.json", "checkpoint_best.pt", "checkpoint_last.pt", "history.json"} <= {
        p.name for p in run_dir.iterdir()}
    assert not (run_dir / "probe_results.json").exists()
    assert result.rl_checkpoint == run_dir / "checkpoint_best.pt" and result.summary["best_epoch"] == 1
    ck = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    # 10.4: top-level particle_scale (the Trainer's convention) is written by encoder_checkpoint for every encoder
    assert set(ck) == {"model_state_dict", "head_state_dict", "config", "epoch", "val", "args", "particle_scale",
                       pretrain.CHECKPOINT_RECORD_KEY}
    assert ck["particle_scale"] == 24.5
    assert (ck["config"]["objective"], ck["config"]["encoder"], ck["config"]["arena_scale"]) == ("belief_kl", "st", 24.5)
    args = json.loads((run_dir / "args.json").read_text())
    assert (args["num_epochs"], args["objective"], args["encoder"], args["variant"]) == (1, "belief_kl", "st", "oe50_short")
    assert [row["epoch"] for row in json.loads((run_dir / "history.json").read_text())] == [1]
    # 7.5: run_status.json names the RL-loadable file; the checkpoint carries the run record.
    status = json.loads((run_dir / "run_status.json").read_text())
    assert status["status"] == "completed" and status["rl_checkpoint"] == str(result.rl_checkpoint.resolve())
    record = ck[pretrain.CHECKPOINT_RECORD_KEY]
    assert (record["domain"], record["variant"], record["encoder"], record["objective"], record["layout"]) == (
        "odd_even", "oe50_short", "st", "belief_kl", "legacy")
    assert record["geometry"]["num_inds"] == 16 and record["arena_scale"] == 24.5 and "threads" in record
    assert json.loads((run_dir / "run_config.json").read_text())["layout"] == "legacy"


def test_run_tag_and_experiment_name_shape_the_run_folder(dataset, tmp_path, monkeypatch):
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    pretrain.main(["--domain", "ant_tag", "--encoder", "st", "--data_path", str(dataset), *SMALL,
                   "--base_dir", str(tmp_path / "b"), "--experiment_name", "exp", "--run_tag", "a b/c",
                   "--dry_run"])
    [run_dir] = [p for p in (tmp_path / "b" / "exp").iterdir()]
    assert run_dir.name.startswith("sinkhorn_") and run_dir.name.endswith("_a_b_c")


def test_variant_flag_must_belong_to_the_domain(dataset, tmp_path, capsys):
    # --objective named: since 7.3 Odd-Even's default objective is belief_kl, not reconstruction.
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--domain", "odd_even", "--encoder", "st", "--objective", "reconstruction",
                       "--data_path", str(dataset), *SMALL, "--base_dir", str(tmp_path), "--dry_run"])
    assert exc.value.code == 2 and "--variant 'smart' is not a odd_even variant" in capsys.readouterr().err


def test_a_flag_of_another_objective_is_refused_with_the_default_named(dataset, tmp_path, capsys):
    """A reconstruction flag (--sinkhorn_blur) without --objective on odd_even: the domain's default
    (belief_kl) applies, and the error says so instead of a bare 'unrecognized arguments'. (Until 10.5
    the test used --data_path, which the exact-posterior objectives now take.)"""
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--domain", "odd_even", "--encoder", "st", "--variant", "oe50_short",
                       "--sinkhorn_blur", "0.02", "--base_dir", str(tmp_path), "--dry_run"])
    err = capsys.readouterr().err
    assert exc.value.code == 2 and "unrecognized arguments: --sinkhorn_blur" in err
    assert "odd_even's default 'belief_kl' applies; --list_objectives shows the others" in err
    # With the objective named, the same typo is a plain unrecognized-argument error.
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "odd_even", "--encoder", "st", "--objective", "belief_kl", "--variant",
                       "oe50_short", "--sinkhorn_blur", "0.02", "--base_dir", str(tmp_path), "--dry_run"])
    err = capsys.readouterr().err
    assert "unrecognized arguments: --sinkhorn_blur" in err and "default" not in err


# ---------------------------------------------------------------------------
# The 3_train_st.py entry point: historical defaults
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 7.5: the root layout end to end, the latest-checkpoint helper, the dataset default
# ---------------------------------------------------------------------------

def test_root_layout_run_has_checkpoints_folder_status_and_is_found_by_the_helper(tmp_path, monkeypatch):
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    root = tmp_path / "root"
    result = pretrain.main(["--domain", "odd_even", "--encoder", "st", "--variant", "oe50_short", "--device", "cpu",
                            "--output_root", str(root), "--n_train_episodes", "4", "--n_val_episodes", "2",
                            "--num_epochs", "1", "--skip_probe", "--run_tag", "tag"])
    run_dir = result.run_dir
    assert run_dir.parent == root / "odd_even" / "oe50_short" / "pretrain" / "st" / "belief_kl"
    assert run_dir.name.endswith("_seed0_tag")
    assert {"args.json", "history.json", "run_config.json", "run_status.json", "checkpoints"} <= {
        p.name for p in run_dir.iterdir()}
    assert {p.name for p in (run_dir / "checkpoints").iterdir()} == {"checkpoint_best.pt", "checkpoint_last.pt"}
    assert result.rl_checkpoint == run_dir / "checkpoints" / "checkpoint_best.pt"
    found = run_records.latest_pretrain_checkpoint("odd_even", "oe50_short", "st", "belief_kl", root=root)
    assert found == result.rl_checkpoint.resolve()
    assert run_records.latest_pretrain_checkpoint("odd_even", "oe50_short", "cgf", "belief_kl", root=root) is None
    with pytest.raises(ValueError, match="needs the objective"):
        run_records.latest_pretrain_checkpoint("odd_even", "oe50_short", "st", root=root)


def test_reconstruction_takes_the_variants_dataset_under_the_root_and_exports_the_cgf_arm(dataset, tmp_path, monkeypatch):
    """--data_path omitted: the variant's dataset under the run root (decision 1); the run
    lands under pretrain/cgf/reconstruction/; run_status names the CGF EXPORT as the
    RL-loadable file and the helper returns it."""
    from set_transformer.rl import run_records
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    root = tmp_path / "root"
    target = run_records.dataset_path("ant_tag", "smart", root=root)
    target.parent.mkdir(parents=True)
    target.write_bytes(dataset.read_bytes())
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "ant_tag", "--encoder", "cgf", "--variant", "cdens", *SMALL,
                       "--output_root", str(root), "--dry_run"])          # no dataset for cdens
    result = pretrain.main(["--domain", "ant_tag", "--encoder", "cgf", "--variant", "smart", *SMALL,
                            "--t_param", "clamp", "--output_root", str(root)])
    assert result.run_dir.parent == root / "ant_tag" / "smart" / "pretrain" / "cgf" / "reconstruction"
    assert result.rl_checkpoint.name.endswith("_cgf_arm.pt") and result.rl_checkpoint.parent.name == "checkpoints"
    status = json.loads((result.run_dir / "run_status.json").read_text())
    assert status["rl_checkpoint"] == str(result.rl_checkpoint.resolve())
    assert run_records.latest_pretrain_checkpoint("ant_tag", "smart", "cgf", "reconstruction", root=root) == (
        result.rl_checkpoint.resolve())
    record = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)[pretrain.CHECKPOINT_RECORD_KEY]
    assert record["data_path"] == str(target) and record["encoder"] == "cgf" and record["geometry"]["t_param"] == "clamp"
    # the Trainer's own checkpoint carries the record too; its loader keys are untouched
    trainer_ckpt = torch.load(result.checkpoints["latest"], map_location="cpu", weights_only=False)
    assert pretrain.CHECKPOINT_RECORD_KEY in trainer_ckpt and "model_state_dict" in trainer_ckpt


def test_dataset_default_needs_a_variant(capsys):
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "ant_tag", "--encoder", "st", "--base_dir", "/tmp/x", "--dry_run"])
    assert "--data_path not given: pass it, or --variant" in capsys.readouterr().err


BELIEF = _ST_ROOT / "experiments" / "odd_even" / "3_pretrain_st_belief.py"


def _load_belief_entry_point():
    spec = importlib.util.spec_from_file_location("pretrain_cmd_test_3_pretrain_st_belief", BELIEF)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_belief_entry_point_translates_its_legacy_spellings(capsys):
    """3_pretrain_st_belief.py: --epochs / --lr / --out_dir become the package spellings, the
    script's defaults (--encoder st, --variant oe50_short) are filled in, the RL-side flags are
    refused with the script's message (exit 2), and --help is the package help for the chosen
    encoder with the translations appended."""
    ep = _load_belief_entry_point()
    assert ep.translate(["--epochs", "3", "--lr=0.01", "--out_dir", "/x/y", "--encoder", "cgf", "--skip_probe"]) == [
        "--num_epochs", "3", "--learning_rate=0.01", "--base_dir", "/x", "--experiment_name", "y",
        "--encoder", "cgf", "--skip_probe", "--variant", "oe50_short"]
    # 7.5: without --out_dir the script keeps its historical folder <encoder>_belief_pretrain/.
    assert ep.translate([]) == ["--encoder", "st", "--variant", "oe50_short", "--experiment_name", "st_belief_pretrain"]
    assert ep.translate(["--variant", "oe50", "--objective", "mode_ce", "--encoder", "cgf"]) == [
        "--variant", "oe50", "--objective", "mode_ce", "--encoder", "cgf", "--experiment_name", "cgf_belief_pretrain"]
    assert ep.translate(["--base_dir", "/b"])[-2:] == ["--variant", "oe50_short"]
    with pytest.raises(SystemExit) as exc:
        ep.translate(["--encoder", "cgf", "--cgf_frozen"])
    assert exc.value.code == 2
    assert "RL-side flags; this script PRODUCES the checkpoint" in capsys.readouterr().err
    assert ep.main(["--help"]) is None
    out = " ".join(capsys.readouterr().out.split())        # argparse wraps at the terminal width
    body, _, epilogue = out.partition("This script's historical spellings")
    assert "--num_epochs" in body and "--num_post_sab" in body and "--probe_episodes" in body
    assert "--epochs N" in epilogue and "--lr X" in epilogue and "--out_dir DIR" in epilogue
    assert "--t_param" not in body and "--help --encoder cgf" in epilogue
    assert ep.main(["--help", "--encoder", "cgf"]) is None
    out = " ".join(capsys.readouterr().out.split())
    assert "--t_param" in out and "--match_params" in out and "--help --encoder cgf" not in out
    # The pieces the recorded tools and tests read off the script are still there.
    for name in ("build_extractor", "BeliefBatches", "BeliefEncoderWithHead", "save_belief_checkpoint"):
        assert hasattr(ep, name), name


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
