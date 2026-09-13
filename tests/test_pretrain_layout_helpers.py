"""The 7.5 run-record helpers of ``rl/run_records.py`` (plan section 7, decision 4): the
pretraining status file and ``latest_pretrain_checkpoint`` over the root layout AND the old
folder shapes (read-only), plus the step-2b entry point of ``rl/precompute_emd.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from set_transformer.rl import run_records

_ST_ROOT = Path(__file__).resolve().parents[1]
EMD_SCRIPT = _ST_ROOT / "experiments" / "ant_tag" / "2b_precompute_emd.py"


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_pretrain_status_names_the_rl_loadable_file(tmp_path):
    ck = _touch(tmp_path / "checkpoints" / "checkpoint_best.pt")
    status = run_records.write_pretrain_status(tmp_path, completed=True, error=None, rl_checkpoint=ck,
                                               checkpoints={"best": ck}, summary={"best_epoch": 3})
    on_disk = json.loads((tmp_path / "run_status.json").read_text())
    assert on_disk == status
    assert status["status"] == "completed" and status["rl_checkpoint"] == str(ck.resolve())
    assert status["checkpoints"] == {"best": str(ck.resolve())} and status["summary"] == {"best_epoch": 3}
    failed = run_records.write_pretrain_status(tmp_path, completed=False, error=RuntimeError("boom"))
    assert failed["status"] == "failed" and failed["error"] == "RuntimeError: boom" and failed["rl_checkpoint"] is None


def test_latest_checkpoint_reads_the_root_layout_newest_completed_first(tmp_path):
    root = tmp_path
    base = root / "odd_even" / "oe50" / "pretrain" / "st" / "belief_kl"
    older = _touch(base / "20260901_000000_seed0" / "checkpoints" / "checkpoint_best.pt")
    run_records.write_pretrain_status(older.parent.parent, completed=True, error=None, rl_checkpoint=older)
    newest_failed = base / "20260903_000000_seed0"
    newest_failed.mkdir(parents=True)
    run_records.write_pretrain_status(newest_failed, completed=False, error=RuntimeError("x"))
    no_status = _touch(base / "20260902_000000_seed0" / "checkpoints" / "checkpoint_best.pt")   # crashed before status
    assert run_records.latest_pretrain_checkpoint("odd_even", "oe50", "st", "belief_kl", root=root) == older.resolve()
    newer = _touch(base / "20260904_000000_seed1" / "checkpoints" / "checkpoint_best.pt")
    run_records.write_pretrain_status(newer.parent.parent, completed=True, error=None, rl_checkpoint=newer)
    assert run_records.latest_pretrain_checkpoint("odd_even", "oe50", "st", "belief_kl", root=root) == newer.resolve()
    # a status whose file has disappeared is skipped
    newer.unlink()
    assert run_records.latest_pretrain_checkpoint("odd_even", "oe50", "st", "belief_kl", root=root) == older.resolve()
    assert no_status.exists()   # never touched: the helper is read-only


def test_latest_checkpoint_recognises_the_old_folder_shapes_read_only(tmp_path):
    root = tmp_path
    # Odd-Even's 3_pretrain_st_belief.py shape: <experiment>/<stamp>_<objective>_seed<n>/checkpoint_best.pt
    oe = _touch(root / "odd_even" / "oe50_short" / "pretrain" / "st_belief_pretrain"
                / "20260905_120000_belief_kl_seed0" / "checkpoint_best.pt")
    _touch(root / "odd_even" / "oe50_short" / "pretrain" / "st_belief_pretrain"
           / "20260904_120000_belief_kl_seed0" / "checkpoint_best.pt")
    assert run_records.latest_pretrain_checkpoint("odd_even", "oe50_short", "st", root=root,
                                                  experiment_name="st_belief_pretrain") == oe
    # Ant-Tag's 3_train_st.py shape, ST and the CGF export (preferred when present)
    at = root / "ant_tag" / "smart" / "pretrain" / "smart_cgf_recon" / "sinkhorn_2026-09-11_10-00-00" / "checkpoints"
    _touch(at / "checkpoint_best.pt")
    export = _touch(at / "checkpoint_best_cgf_arm.pt")
    assert run_records.latest_pretrain_checkpoint("ant_tag", "smart", "cgf", root=root,
                                                  experiment_name="smart_cgf_recon") == export
    # an old experiment folder anywhere on disk, by base_dir
    legacy = _touch(tmp_path / "elsewhere" / "exp" / "sinkhorn_2026-09-04_01-18-26" / "checkpoints" / "checkpoint_best.pt")
    assert run_records.latest_pretrain_checkpoint("ant_tag", "x", "st", base_dir=tmp_path / "elsewhere",
                                                  experiment_name="exp") == legacy
    assert run_records.latest_pretrain_checkpoint("ant_tag", "smart", "st", root=root, experiment_name="nope") is None


def test_precompute_emd_entry_point_and_module_share_one_main(tmp_path):
    from set_transformer.rl import precompute_emd
    spec = importlib.util.spec_from_file_location("layout_test_2b", EMD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.default_out_path("a/b_pf_dataset.npz") == Path("a/b_pf_dataset_emd.npy")
    assert module.DEGENERATE_OFFDIAG_STD == precompute_emd.DEGENERATE_OFFDIAG_STD
    parser = precompute_emd.build_parser()
    for flag in ("--data_path", "--domain", "--variant", "--out_path", "--sinkhorn_blur", "--max_samples",
                 "--ignore_weights", "--allow_degenerate"):
        assert flag in parser.format_help(), flag
    # a tiny weighted 2-D dataset: matrix beside it, sidecar with the run provenance
    rng = np.random.default_rng(0)
    particles = rng.normal(0, 1, size=(6, 20, 2)).astype(np.float32)
    weights = rng.random((6, 20)).astype(np.float32)
    weights /= weights.sum(1, keepdims=True)
    data = tmp_path / "tiny_pf_dataset.npz"
    np.savez(data, particles=particles, weights=weights, particle_scale=np.float32(4.5),
             metadata=np.array(json.dumps({"variant": "smart", "particle_scale": 4.5, "particle_centre": 0.0})))
    out = module.main(["--data_path", str(data), "--sinkhorn_blur", "0.05", "--device", "cpu", "--no_verify"])
    assert out == tmp_path / "tiny_pf_dataset_emd.npy" and out.exists()
    sidecar = json.loads((tmp_path / "tiny_pf_dataset_emd.json").read_text())
    assert sidecar["n_samples"] == 6 and sidecar["weighted"] is True
    assert sidecar["command"].startswith("2b_precompute_emd.py --data_path") and "torch" in sidecar["threads"]
    with pytest.raises(SystemExit):
        precompute_emd.main(["--sinkhorn_blur", "0.05"])        # neither --data_path nor --domain/--variant
