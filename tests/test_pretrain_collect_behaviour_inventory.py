"""Behaviour inventory for pipeline steps 2 (data collection) and 3 (encoder pretraining),
pinned against the CURRENT scripts before any of their code moves into the package
(refactor_plans.md section 7, batch 7.1, 2026-09-13). Same role as
``test_harness_behaviour_inventory.py`` had for the RL arms: every observable command-line
behaviour of the four scripts gets a test here (or is listed as untested on purpose), and the
later batches must keep these green while the scripts turn into entry points of
``set_transformer.rl.pretrain`` / ``set_transformer.rl.collect``.

Behaviour list (section 7d of the plan). Each line names its test in THIS file, an existing
test elsewhere, or says why it is not tested.

experiments/ant_tag/3_train_st.py (reconstruction pretraining, any domain's dataset)
  T1  --help exits 0 and lists the data / loss / alignment / geometry / placement flags; the CGF
      flags with --help --encoder cgf (per-encoder help since 7.2) -> test_train_st_help_lists_every_flag_group
  T2  set geometry (num_particles, dim_particles) comes from the dataset; a contradicting flag
      is a parser error naming the dataset value -> test_train_st_refuses_a_geometry_flag_that_contradicts_the_dataset
  T3  PF weights are used by default; --loss_type chamfer on a weighted dataset is refused
        -> test_train_st_refuses_chamfer_on_a_weighted_dataset
  T4  --align_lambda > 0 without --emd_matrix_path is refused -> test_train_st_alignment_needs_a_matrix
  T5  a matrix whose sidecar disagrees on blur (or scaling / weightedness / hash / rows) is refused
      before any data is loaded -> test_train_st_refuses_a_matrix_whose_sidecar_disagrees
  T6  --max_samples must be positive -> test_train_st_max_samples_must_be_positive
  T7  --encoder cgf with tanh / polar needs --t_bound -> test_train_st_cgf_tanh_needs_a_bound
  T8  --base_dir default is <root>/<domain>/<variant>/pretrain/<experiment_name>/ with domain and
      variant from the dataset metadata -> test_train_st_default_output_is_the_root_pretrain_layout
  T9  a dataset without a recorded variant and no --base_dir / --variant is refused
        -> test_train_st_default_output_needs_a_variant
  T10 --domain must own the variant -> test_train_st_domain_flag_must_own_the_variant
  T11 run folder <base_dir>/<experiment_name>/<loss>_<timestamp>/{checkpoints,logs}; checkpoints
      checkpoint_best.pt / checkpoint_latest.pt / checkpoint_<step>.pt with the Trainer's keys and
      the training config (geometry, weighted, seed, frame) -> test_train_st_checkpoint_layout_and_keys
  T12 the same seed gives bit-identical checkpoints -> test_train_st_same_seed_gives_identical_checkpoints
  T13 --encoder cgf exports checkpoint_{best,latest}_cgf_arm.pt that load strict into the RL
      extractor -> test_cgf_arm_autoencoder.py::test_end_to_end_script_on_a_tiny_weighted_dataset (existing)
  T14 --device (batch 7.0) -> test_cgf_arm_autoencoder.py::test_script_takes_a_device_flag_... (existing)
  T15 explicit --base_dir wins over the root layout -> test_cgf_arm_autoencoder.py end-to-end test (existing)
  --  wandb logging, the automatic device choice, --ignore_weights numerics, --scheduler_type: not
      tested on purpose (logging side effects; hardware; numerics are the parity runs' job).

experiments/odd_even/3_pretrain_st_belief.py (exact-posterior pretraining, Odd-Even only)
  B1  --help exits 0; --list_variants prints the registry and exits 0 -> test_belief_help_and_list_variants
  B2  the RL-side flags --pretrained_cgf_model_path / --cgf_frozen are refused
        -> test_belief_refuses_the_rl_side_cgf_flags
  B3  --init_from is ST only -> test_belief_init_from_is_st_only
  B4  run folder <out_dir>/<stamp>_[<encoder>_]<objective>_seed<n>[_<tag>] holding args.json,
      checkpoint_best.pt, checkpoint_last.pt, history.json, probe_results.json
        -> test_belief_run_folder_name_and_files, test_belief_cgf_run_names_the_encoder_and_skips_the_probe
  B5  ST checkpoint: model_state_dict keys under `set_transformer.`, head_state_dict, config with the
      geometry + objective + variant + arena_scale + encoder + encoder_params + pretraining (+ the
      top-level pretraining_run record since 7.5)
        -> test_belief_st_checkpoint_is_in_the_rl_loaders_format
  B6  the checkpoint loads into the RL extractor with max|delta| == 0
        -> test_belief_checkpoint_loads_into_the_rl_extractor
  B7  CGF checkpoint: the extractor's whole state_dict, unprefixed (t, norm statistics, readout)
        -> test_belief_cgf_run_names_the_encoder_and_skips_the_probe
  B8  history.json has one row per epoch with train / val loss, lr, head accuracies split
      transient / steady, feature std and effective rank -> test_belief_history_has_one_row_per_epoch
  B9  probe_results.json has the geometry / head / targets blocks over EXACT, GAUSS2 and the
      encoder -> test_belief_probe_results_have_the_three_blocks
  B10 the same seed gives bit-identical checkpoints -> test_belief_same_seed_gives_identical_checkpoints
  B11 default --out_dir is <root>/odd_even/<variant>/pretrain/<encoder>_belief_pretrain/
        -> test_belief_default_out_dir_is_the_root_pretrain_layout
  --  the probe's classifier accuracies, --freeze_encoder / --init_from numerics, data_seed
      disjointness of train and val: not tested here on purpose (results, not behaviour; the
      parity runs compare them bit for bit).

experiments/ant_tag/2_collect_pf_dataset.py (Ant-Tag collection)
  C1  --help exits 0; --list_variants prints the registry -> test_collect_help_and_list_variants
  C2  the .npz contract: particles [S,N,D] float32 raw coords, weights [S,N] rows summing to 1,
      particle_scale = the arena half-width, metadata JSON with variant / env_id /
      particle_filter_class / num_particles / dim_particles / particle_scale / args / git (+ command
      and threads since 7.5) -> test_collect_npz_contract
  C3  the same seed gives the same file -> test_collect_same_seed_gives_the_same_arrays
  C4  a .npy output is refused (cannot hold the weights) -> test_collect_refuses_npy_output
  C5  weighted spread is the weighted std per coordinate averaged; rebalancing with
      upsample=False never duplicates a row, with upsample=True keeps the size
        -> test_collect_spread_and_rebalance_helpers
  C6  default output <root>/<domain>/<variant>/data/<variant>_pf_dataset.npz (decision 1, 7.4; was
      data/<variant>_pf_dataset.npz under the cwd) -> test_collect_default_output_name
  --  the locomotion policy's pursuit action and the visibility-radius draw: not tested on
      purpose (stochastic policy rollouts; the parity runs cover them).

experiments/odd_even/2_collect_pf_dataset.py (Odd-Even collection): already pinned by
  test_odd_even_pomdp_contract.py::test_collected_dataset_contract / test_dataset_covers_the_belief_transient
  and test_odd_even_pipeline.py::test_collected_dataset_loads_in_the_rl_frame /
  test_rebalance_never_duplicates_rows. Nothing added here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ST_ROOT = Path(__file__).resolve().parents[1]
_ANT_TAG = _ST_ROOT / "experiments" / "ant_tag"
_ODD_EVEN = _ST_ROOT / "experiments" / "odd_even"
TRAIN_ST = _ANT_TAG / "3_train_st.py"
BELIEF = _ODD_EVEN / "3_pretrain_st_belief.py"
COLLECT = _ANT_TAG / "2_collect_pf_dataset.py"
LOCOMOTION = _ANT_TAG / "models" / "ant_locomotion_policy.zip"

SCALE = 4.5
N, D, S = 100, 2, 48


def _run(script: Path, args: list[str], tmp: Path, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run a pipeline script as the user would, from its own directory, on CPU, wandb offline
    and writing its side files under `tmp`."""
    env = {**os.environ, "WANDB_MODE": "offline", "WANDB_DIR": str(tmp), "CUDA_VISIBLE_DEVICES": "",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
           "PYTHONWARNINGS": "ignore"}
    return subprocess.run([sys.executable, str(script), *args], cwd=script.parent, env=env,
                          capture_output=True, text=True, timeout=timeout)


def _tail(proc: subprocess.CompletedProcess) -> str:
    return proc.stdout[-3000:] + "\n--- stderr ---\n" + proc.stderr[-3000:]


def _tensors(path: Path, key: str = "model_state_dict") -> dict:
    return {k: v for k, v in torch.load(path, map_location="cpu", weights_only=False)[key].items()
            if torch.is_tensor(v)}


def _assert_identical(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k} differs"


def _load_by_path(script: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ==========================================================================
# Fixtures: one tiny dataset, one run of each script (module-scoped, shared by the assertions)
# ==========================================================================

@pytest.fixture(scope="module")
def tmp(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("inventory_2_3")


@pytest.fixture(scope="module")
def tiny_dataset(tmp) -> Path:
    """48 weighted 2-D clouds in the collector's .npz contract, recorded as `smart`."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-3, 3, size=(S, 1, D))
    particles = (centres + rng.normal(0, 0.7, size=(S, N, D))).astype(np.float32)
    weights = rng.random((S, N)).astype(np.float32)
    weights[:, N // 2:] *= 0.05
    weights /= weights.sum(1, keepdims=True)
    path = tmp / "tiny_smart_pf_dataset.npz"
    meta = json.dumps({"env_id": "pdomains-ant-tag-smart-v0", "variant": "smart",
                       "particle_scale": SCALE, "particle_centre": 0.0})
    np.savez(path, particles=particles, weights=weights, particle_scale=np.float32(SCALE),
             metadata=np.array(meta))
    return path


@pytest.fixture(scope="module")
def variantless_dataset(tmp, tiny_dataset) -> Path:
    z = np.load(tiny_dataset)
    path = tmp / "variantless.npz"
    np.savez(path, particles=z["particles"], weights=z["weights"], particle_scale=z["particle_scale"],
             metadata=np.array(json.dumps({"env_id": "synthetic", "particle_scale": SCALE})))
    return path


ST_FLAGS = ["--num_epochs", "2", "--batch_size", "8", "--num_encodings", "8", "--dim_encoder", "8",
            "--num_inds", "16", "--dim_hidden", "32", "--eval_freq", "3", "--save_freq", "1000",
            "--warmup_epochs", "1", "--seed", "0", "--device", "cpu", "--num_workers", "0",
            "--experiment_name", "inv"]


def _train_st(tiny_dataset: Path, tmp: Path, root: Path) -> Path:
    proc = _run(TRAIN_ST, ["--data_path", str(tiny_dataset), *ST_FLAGS, "--output_root", str(root)], tmp)
    assert proc.returncode == 0, _tail(proc)
    [run_dir] = [p for p in (root / "ant_tag" / "smart" / "pretrain" / "inv").iterdir() if p.is_dir()]
    return run_dir


@pytest.fixture(scope="module")
def st_run(tiny_dataset, tmp) -> Path:
    return _train_st(tiny_dataset, tmp, tmp / "root_a")


@pytest.fixture(scope="module")
def st_run_repeat(tiny_dataset, tmp) -> Path:
    return _train_st(tiny_dataset, tmp, tmp / "root_b")


BELIEF_SMALL = ["--variant", "oe50_short", "--epochs", "2", "--n_train_episodes", "40",
                "--n_val_episodes", "10", "--seed", "0", "--device", "cpu"]


def _belief(tmp: Path, out_dir: Path, extra: list[str]) -> Path:
    proc = _run(BELIEF, [*BELIEF_SMALL, "--out_dir", str(out_dir), *extra], tmp)
    assert proc.returncode == 0, _tail(proc)
    [run_dir] = [p for p in out_dir.iterdir() if p.is_dir()]
    return run_dir


@pytest.fixture(scope="module")
def belief_st_run(tmp) -> Path:
    """ST, belief_kl, WITH the end-of-run probe (12 episodes, 3 folds)."""
    return _belief(tmp, tmp / "belief_st", ["--encoder", "st", "--probe_episodes", "12",
                                            "--probe_splits", "3"])


@pytest.fixture(scope="module")
def belief_st_noprobe_pair(tmp) -> tuple[Path, Path]:
    a = _belief(tmp, tmp / "belief_st_a", ["--encoder", "st", "--skip_probe"])
    b = _belief(tmp, tmp / "belief_st_b", ["--encoder", "st", "--skip_probe"])
    return a, b


@pytest.fixture(scope="module")
def belief_cgf_run(tmp) -> Path:
    return _belief(tmp, tmp / "belief_cgf", ["--encoder", "cgf", "--match_params", "2000",
                                             "--skip_probe", "--run_tag", "inv"])


COLLECT_SMALL = ["--variant", "smart", "--num_trajectories", "2", "--timesteps", "20",
                 "--num_particles", "100", "--seed", "0", "--rebalance_no_upsample",
                 "--locomotion_policy_path", str(LOCOMOTION)]


def _collect(tmp: Path, out: Path, extra: list[str] = ()) -> Path:
    proc = _run(COLLECT, [*COLLECT_SMALL, "--output_file", str(out), *extra], tmp)
    assert proc.returncode == 0, _tail(proc)
    return out


@pytest.fixture(scope="module")
def collect_smart(tmp) -> Path:
    return _collect(tmp, tmp / "smart_a.npz")


@pytest.fixture(scope="module")
def collect_smart_repeat(tmp) -> Path:
    return _collect(tmp, tmp / "smart_b.npz")


# ==========================================================================
# 3_train_st.py
# ==========================================================================

def test_train_st_help_lists_every_flag_group(tmp):
    proc = _run(TRAIN_ST, ["--help"], tmp)
    assert proc.returncode == 0, _tail(proc)
    for flag in ("--data_path", "--ignore_weights", "--loss_type", "--sinkhorn_blur", "--align_lambda",
                 "--emd_matrix_path", "--max_samples", "--num_encodings", "--dim_encoder", "--num_inds",
                 "--dim_hidden", "--encoder", "--base_dir", "--domain", "--variant", "--output_root",
                 "--experiment_name", "--seed", "--device"):
        assert flag in proc.stdout, flag
    # Since batch 7.2 the help page is per encoder, as the RL trainer's is: the CGF flags show
    # with --encoder cgf (the old script listed both groups on one page).
    proc = _run(TRAIN_ST, ["--help", "--encoder", "cgf"], tmp)
    assert proc.returncode == 0, _tail(proc)
    for flag in ("--t_param", "--t_bound", "--feature_mode", "--readout_hidden", "--match_params"):
        assert flag in proc.stdout, flag


def test_train_st_refuses_a_geometry_flag_that_contradicts_the_dataset(tiny_dataset, tmp):
    proc = _run(TRAIN_ST, ["--data_path", str(tiny_dataset), *ST_FLAGS, "--base_dir", str(tmp / "x"),
                           "--num_particles", "7"], tmp)
    assert proc.returncode == 2
    assert "--num_particles 7 contradicts the dataset (100)" in proc.stderr


def test_train_st_refuses_chamfer_on_a_weighted_dataset(tiny_dataset, tmp):
    proc = _run(TRAIN_ST, ["--data_path", str(tiny_dataset), *ST_FLAGS, "--base_dir", str(tmp / "x"),
                           "--loss_type", "chamfer"], tmp)
    assert proc.returncode == 2
    assert "chamfer cannot train on weighted particle sets" in proc.stderr
    assert "--ignore_weights" in proc.stderr


def test_train_st_alignment_needs_a_matrix(tiny_dataset, tmp):
    proc = _run(TRAIN_ST, ["--data_path", str(tiny_dataset), *ST_FLAGS, "--base_dir", str(tmp / "x"),
                           "--align_lambda", "0.2"], tmp)
    assert proc.returncode == 2
    assert "--align_lambda > 0 needs --emd_matrix_path" in proc.stderr


def test_train_st_refuses_a_matrix_whose_sidecar_disagrees(tiny_dataset, tmp):
    matrix = tmp / "fake_emd.npy"
    np.save(matrix, np.zeros((S, S), dtype=np.float32))
    matrix.with_suffix(".json").write_text(json.dumps(
        {"blur": 0.05, "scaling": 0.5, "weighted": True, "n_samples": S}))
    proc = _run(TRAIN_ST, ["--data_path", str(tiny_dataset), *ST_FLAGS, "--base_dir", str(tmp / "x"),
                           "--align_lambda", "0.2", "--emd_matrix_path", str(matrix),
                           "--sinkhorn_blur", "0.01"], tmp)
    assert proc.returncode == 2
    assert "--emd_matrix_path does not match this run" in proc.stderr
    assert "blur: matrix 0.05, this run 0.01" in proc.stderr


def test_train_st_max_samples_must_be_positive(tiny_dataset, tmp):
    proc = _run(TRAIN_ST, ["--data_path", str(tiny_dataset), *ST_FLAGS, "--base_dir", str(tmp / "x"),
                           "--max_samples", "0"], tmp)
    assert proc.returncode == 2 and "--max_samples must be positive" in proc.stderr


def test_train_st_cgf_tanh_needs_a_bound(tiny_dataset, tmp):
    proc = _run(TRAIN_ST, ["--data_path", str(tiny_dataset), *ST_FLAGS, "--base_dir", str(tmp / "x"),
                           "--encoder", "cgf", "--t_param", "tanh"], tmp)
    assert proc.returncode == 2 and "--t_param tanh needs --t_bound > 0" in proc.stderr


def test_train_st_default_output_is_the_root_pretrain_layout(st_run, tmp):
    root = tmp / "root_a"
    assert st_run.parent == root / "ant_tag" / "smart" / "pretrain" / "inv"
    assert re.fullmatch(r"sinkhorn_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", st_run.name), st_run.name
    assert (st_run / "checkpoints").is_dir() and (st_run / "logs").is_dir()


def test_train_st_default_output_needs_a_variant(variantless_dataset, tmp):
    proc = _run(TRAIN_ST, ["--data_path", str(variantless_dataset), *ST_FLAGS], tmp)
    assert proc.returncode == 2
    assert "--base_dir not given and the dataset records no variant" in proc.stderr


def test_train_st_domain_flag_must_own_the_variant(tiny_dataset, tmp):
    proc = _run(TRAIN_ST, ["--data_path", str(tiny_dataset), *ST_FLAGS, "--domain", "odd_even"], tmp)
    assert proc.returncode == 2
    assert "--variant 'smart' is not a odd_even variant" in proc.stderr


def test_train_st_checkpoint_layout_and_keys(st_run):
    names = sorted(p.name for p in (st_run / "checkpoints").iterdir())
    assert "checkpoint_best.pt" in names and "checkpoint_latest.pt" in names
    assert any(re.fullmatch(r"checkpoint_\d+\.pt", n) for n in names), names
    ck = torch.load(st_run / "checkpoints" / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "config", "epoch",
            "global_step", "best_epoch", "best_val_loss", "particle_scale", "particle_centre"} <= set(ck)
    config = ck["config"] if isinstance(ck["config"], dict) else vars(ck["config"])
    assert (config["num_particles"], config["dim_particles"]) == (N, D)
    assert (config["num_encodings"], config["dim_encoder"], config["num_inds"], config["dim_hidden"]) == (8, 8, 16, 32)
    assert config["weighted_particles"] is True and config["loss_type"] == "sinkhorn" and config["seed"] == 0
    assert float(ck["particle_scale"]) == SCALE and float(ck["particle_centre"]) == 0.0
    # Encoder keys carry the model's own prefix; the RL ST extractor strips `set_transformer.`.
    assert any(k.startswith("set_transformer.") for k in ck["model_state_dict"])


def test_train_st_same_seed_gives_identical_checkpoints(st_run, st_run_repeat):
    for name in ("checkpoint_best.pt", "checkpoint_latest.pt"):
        _assert_identical(_tensors(st_run / "checkpoints" / name), _tensors(st_run_repeat / "checkpoints" / name))
    a = torch.load(st_run / "checkpoints" / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    b = torch.load(st_run_repeat / "checkpoints" / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert a["best_val_loss"] == b["best_val_loss"] and a["best_epoch"] == b["best_epoch"]


# ==========================================================================
# 3_pretrain_st_belief.py
# ==========================================================================

def test_belief_help_and_list_variants(tmp):
    proc = _run(BELIEF, ["--help"], tmp)
    assert proc.returncode == 0, _tail(proc)
    for flag in ("--variant", "--encoder", "--objective", "--epochs", "--lr", "--num_inds", "--num_post_sab",
                 "--t_param", "--feature_mode", "--match_params", "--probe_episodes", "--skip_probe",
                 "--out_dir", "--output_root", "--run_tag", "--init_from", "--freeze_encoder", "--device"):
        assert flag in proc.stdout, flag
    proc = _run(BELIEF, ["--list_variants"], tmp)
    assert proc.returncode == 0 and "oe50_short" in proc.stdout and "oe50_long" in proc.stdout


def test_belief_refuses_the_rl_side_cgf_flags(tmp):
    proc = _run(BELIEF, [*BELIEF_SMALL, "--encoder", "cgf", "--cgf_frozen", "--out_dir", str(tmp / "x")], tmp)
    assert proc.returncode == 2
    assert "RL-side flags; this script PRODUCES the checkpoint" in proc.stderr


def test_belief_init_from_is_st_only(tmp):
    proc = _run(BELIEF, [*BELIEF_SMALL, "--encoder", "cgf", "--match_params", "2000", "--skip_probe",
                         "--init_from", "nowhere.pt", "--out_dir", str(tmp / "y")], tmp)
    assert proc.returncode == 1
    assert "--init_from is implemented for --encoder st only" in proc.stderr


def test_belief_run_folder_name_and_files(belief_st_run):
    assert re.fullmatch(r"\d{8}_\d{6}_belief_kl_seed0", belief_st_run.name), belief_st_run.name
    assert {"args.json", "checkpoint_best.pt", "checkpoint_last.pt", "history.json",
            "probe_results.json"} <= {p.name for p in belief_st_run.iterdir()}
    args = json.loads((belief_st_run / "args.json").read_text())
    # 7.3: args.json records the package command's spelling (--epochs is translated to
    # --num_epochs by the entry point; decision 2 of plan section 7).
    assert args["variant"] == "oe50_short" and args["encoder"] == "st" and args["num_epochs"] == 2


def test_belief_st_checkpoint_is_in_the_rl_loaders_format(belief_st_run):
    ck = torch.load(belief_st_run / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    # 7.5 added the top-level `pretraining_run` record (additive; the loaders' keys are unchanged).
    # 7.5 added `pretraining_run`; 10.4 added top-level `particle_scale` (additive; the loaders' keys are unchanged).
    assert set(ck) == {"model_state_dict", "head_state_dict", "config", "epoch", "val", "args", "pretraining_run",
                       "particle_scale"}
    assert ck["particle_scale"] == 24.5
    assert ck["model_state_dict"] and all(k.startswith("set_transformer.") for k in ck["model_state_dict"])
    assert set(ck["head_state_dict"]) == {"weight", "bias"} and ck["head_state_dict"]["weight"].shape == (50, 64)
    c = ck["config"]
    assert (c["num_encodings"], c["dim_encoder"], c["num_inds"], c["dim_hidden"], c["num_post_sab"]) == (8, 8, 16, 64, 2)
    assert c["objective"] == "belief_kl" and c["variant"] == "oe50_short" and c["encoder"] == "st"
    assert c["arena_scale"] == 24.5 and c["weighted_particles"] is True and c["encoder_params"] > 0
    assert c["pretraining"] == "3_pretrain_st_belief.py"


def test_belief_checkpoint_loads_into_the_rl_extractor(belief_st_run):
    """The file is what `4_train_rl_st.py --pretrained_st_model_path` reads: build the RL arm's
    extractor from the recorded args, load it, compare every encoder tensor."""
    import gymnasium as gym
    script = _load_by_path(BELIEF, "inventory_3_pretrain_st_belief")
    ck = torch.load(belief_st_run / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ck["args"])
    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (50, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (50,), np.float32),
    })
    extractor = script.build_extractor(args, space, 24.5, pretrained_path=str(belief_st_run / "checkpoint_best.pt"))
    live = extractor.encoder.state_dict()
    for k, v in ck["model_state_dict"].items():
        assert torch.equal(v, live[k[len("set_transformer."):]].cpu()), k


def test_belief_history_has_one_row_per_epoch(belief_st_run):
    history = json.loads((belief_st_run / "history.json").read_text())
    assert [row["epoch"] for row in history] == [1, 2]
    for row in history:
        assert {"train_loss", "loss", "lr", "seconds", "head_mode_acc", "head_true_state_acc",
                "feature_abs_std", "feature_eff_rank"} <= set(row)
        assert set(row["head_mode_acc"]) == {"transient", "steady"}


def test_belief_probe_results_have_the_three_blocks(belief_st_run):
    results = json.loads((belief_st_run / "probe_results.json").read_text())
    assert set(results) == {"geometry", "head", "targets"}
    assert set(results["geometry"]) == {"EXACT", "GAUSS2", "ST_BELIEF"}
    assert set(results["head"]) == {"B_posterior_mode", "A_true_state"}
    for target in results["targets"].values():
        assert set(target) == {"EXACT", "GAUSS2", "ST_BELIEF"}
        assert set(next(iter(target.values()))) == {"logreg_raw", "logreg_z", "mlp_raw", "mlp_z"}


def test_belief_cgf_run_names_the_encoder_and_skips_the_probe(belief_cgf_run):
    assert re.fullmatch(r"\d{8}_\d{6}_cgf_belief_kl_seed0_inv", belief_cgf_run.name), belief_cgf_run.name
    assert not (belief_cgf_run / "probe_results.json").exists()
    ck = torch.load(belief_cgf_run / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    keys = set(ck["model_state_dict"])
    assert not any(k.startswith("set_transformer.") for k in keys)
    # The probe parameter is `raw_t` under tanh / polar (the script's default t_param) and
    # `t_values` under clamp; the running-norm statistics and the readout ride along.
    assert ("raw_t" in keys) or ("t_values" in keys), sorted(keys)
    assert {"feature_norm.running_mean", "feature_norm.running_var"} <= keys and any(k.startswith("readout.") for k in keys)
    assert ck["config"]["encoder"] == "cgf" and ck["config"]["objective"] == "belief_kl"


def test_belief_same_seed_gives_identical_checkpoints(belief_st_noprobe_pair):
    a, b = belief_st_noprobe_pair
    for name in ("checkpoint_best.pt", "checkpoint_last.pt"):
        _assert_identical(_tensors(a / name), _tensors(b / name))
        _assert_identical(_tensors(a / name, "head_state_dict"), _tensors(b / name, "head_state_dict"))
    assert json.loads((a / "history.json").read_text())[-1]["loss"] == json.loads((b / "history.json").read_text())[-1]["loss"]


def test_belief_default_out_dir_is_the_root_pretrain_layout(tmp):
    """--out_dir unset: <root>/odd_even/<variant>/pretrain/<encoder>_belief_pretrain/ (change 5.2)."""
    root = tmp / "belief_root"
    proc = _run(BELIEF, [*BELIEF_SMALL, "--encoder", "st", "--skip_probe", "--output_root", str(root)], tmp)
    assert proc.returncode == 0, _tail(proc)
    runs = list((root / "odd_even" / "oe50_short" / "pretrain" / "st_belief_pretrain").iterdir())
    assert len(runs) == 1 and (runs[0] / "checkpoint_best.pt").exists()


# ==========================================================================
# 2_collect_pf_dataset.py (Ant-Tag)
# ==========================================================================

def test_collect_help_and_list_variants(tmp):
    proc = _run(COLLECT, ["--help"], tmp)
    assert proc.returncode == 0, _tail(proc)
    for flag in ("--variant", "--num_trajectories", "--timesteps", "--num_particles", "--pursuit_fraction",
                 "--visibility_radius_min", "--visibility_radius_max", "--locomotion_policy_path", "--seed",
                 "--max_snapshots", "--no_rebalance", "--rebalance_no_upsample", "--collapsed_threshold",
                 "--diffuse_threshold", "--output_file"):
        assert flag in proc.stdout, flag
    proc = _run(COLLECT, ["--list_variants"], tmp)
    assert proc.returncode == 0 and "cdens_terminal" in proc.stdout and "smart_mid_slow_v15" in proc.stdout


def test_collect_npz_contract(collect_smart):
    with np.load(collect_smart) as z:
        assert set(z.files) == {"particles", "weights", "particle_scale", "metadata"}
        particles, weights = z["particles"], z["weights"]
        assert particles.dtype == np.float32 and weights.dtype == np.float32
        assert particles.ndim == 3 and particles.shape[1:] == (100, 2)
        assert weights.shape == particles.shape[:2]
        assert np.allclose(weights.sum(1), 1.0, atol=1e-5)
        assert float(z["particle_scale"]) == SCALE
        meta = json.loads(str(z["metadata"]))
    # 7.5 added command + threads (the run records' provenance) to the metadata.
    assert set(meta) == {"variant", "env_id", "particle_filter_class", "num_particles", "dim_particles",
                         "particle_scale", "args", "command", "threads", "git"}
    assert meta["variant"] == "smart" and meta["env_id"] == "pdomains-ant-tag-smart-v0"
    assert meta["particle_filter_class"] == "SmartAntTagParticleFilter"
    assert (meta["num_particles"], meta["dim_particles"], meta["particle_scale"]) == (100, 2, SCALE)
    assert meta["args"]["rebalance_no_upsample"] is True and meta["args"]["seed"] == 0


def test_collect_same_seed_gives_the_same_arrays(collect_smart, collect_smart_repeat):
    with np.load(collect_smart) as a, np.load(collect_smart_repeat) as b:
        assert np.array_equal(a["particles"], b["particles"])
        assert np.array_equal(a["weights"], b["weights"])


def test_collect_refuses_npy_output(tmp):
    proc = _run(COLLECT, [*COLLECT_SMALL, "--num_trajectories", "1", "--timesteps", "3",
                          "--output_file", str(tmp / "bad.npy")], tmp)
    assert proc.returncode == 1
    assert "Output must be .npz" in proc.stderr
    assert not (tmp / "bad.npy").exists()


def test_collect_spread_and_rebalance_helpers(monkeypatch):
    # The script imports its sibling `variants` by flat name off sys.path[0] (CLAUDE.md, the
    # sys.path convention), so loading it by path needs its own directory first.
    monkeypatch.syspath_prepend(str(_ANT_TAG))
    collector = _load_by_path(COLLECT, "inventory_ant_tag_collector")
    # Weighted spread: two clouds, one tight and one wide; weights concentrated on one particle
    # make the spread of the BELIEF far smaller than that of the cloud.
    rng = np.random.default_rng(0)
    tight = rng.normal(0, 0.1, size=(1, 50, 2)); wide = rng.normal(0, 3.0, size=(1, 50, 2))
    uniform = np.full((1, 50), 1 / 50)
    spread = collector._weighted_spread(np.concatenate([tight, wide]), np.concatenate([uniform, uniform]))
    assert spread[0] < 0.3 < 2.0 < spread[1]
    peaked = np.zeros((1, 50)); peaked[0, 0] = 1.0
    assert collector._weighted_spread(wide, peaked)[0] == pytest.approx(0.0, abs=1e-6)
    # Rebalancing: 60 clouds with spreads spanning the three buckets.
    scales = np.concatenate([np.full(40, 0.1), np.full(10, 1.5), np.full(10, 6.0)])
    particles = (rng.normal(size=(60, 50, 2)) * scales[:, None, None]).astype(np.float32)
    weights = np.full((60, 50), 1 / 50, dtype=np.float32)
    p_no, _ = collector._rebalance_by_spread(particles, weights, collapsed_threshold=0.5,
                                             diffuse_threshold=4.0, seed=0, upsample=False)
    rows = {row.tobytes() for row in p_no}
    assert len(rows) == len(p_no) and len(p_no) < 60          # nothing duplicated, dataset shrank
    p_up, _ = collector._rebalance_by_spread(particles, weights, collapsed_threshold=0.5,
                                             diffuse_threshold=4.0, seed=0, upsample=True)
    assert len(p_up) == 60                                    # historical behaviour keeps the size
    assert len({row.tobytes() for row in p_up}) < 60          # ... by duplicating rows


def test_collect_default_output_name(tmp):
    """C6, CHANGED by decision 1 of plan section 7 (2026-09-13): the default output is
    <root>/<domain>/<variant>/data/<variant>_pf_dataset.npz under the shared run root, no longer
    data/<variant>_pf_dataset.npz under the current folder; --output_file still overrides."""
    from set_transformer.rl import run_records
    assert run_records.dataset_path("ant_tag", "smart", root=tmp) == (
        tmp / "ant_tag" / "smart" / "data" / "smart_pf_dataset.npz")
    proc = _run(COLLECT, ["--help"], tmp)
    # argparse wraps (and may break) the long path token, so compare without whitespace.
    assert "<root>/<domain>/<variant>/data/<variant>_pf_dataset" in "".join(proc.stdout.split())
