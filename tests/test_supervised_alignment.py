"""Latent metric alignment with ONLINE targets on the supervised objectives (2026-09-19;
``change_mds/online_alignment_2026-09-19.md``).

Pins: the hunt ``task`` objective and Odd-Even's ``belief_kl`` take ``--align_lambda``; with it the
history carries the term, its per-batch correlation and a held-out correlation on a fixed pair sample,
the ramp is honoured, the checkpoint config records the target, and the RL extractor still loads the
checkpoint (the round trip every pretraining ends with). At the default lambda 0 nothing is added: no
record, no keys, no message (the old-vs-new bit-identity of the loop itself was measured on the real
command, see the change note).
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
pytest.importorskip("pdomains.hunt", reason="needs the pomdp-domains hunt envs")

import torch  # noqa: E402

from set_transformer.rl import collect, pretrain, run_records  # noqa: E402

#: The schedule the driver's `*_aligned` conditions carry, shrunk to three epochs, a small pair budget
#: and a small held-out sample.
ALIGN = ["--align_lambda", "0.2", "--align_warmup_epochs", "1", "--align_ramp_epochs", "1",
         "--align_pairs", "64", "--align_val_pairs", "300", "--sinkhorn_blur", "0.02"]


@pytest.fixture(scope="module")
def least_mass_npz(tmp_path_factory):
    path = tmp_path_factory.mktemp("hunt_data") / "least_mass.npz"
    collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "12",
                  "--seed", "5", "--output_file", str(path)])
    return path


def _check_history(hist):
    assert [h["align_lambda"] for h in hist] == pytest.approx([0.0, 0.2, 0.2])   # warm-up 1, ramp 1
    for h in hist:
        assert np.isfinite(h["align"]) and np.isfinite(h["align_r"]) and np.isfinite(h["val_align_r"])


def test_hunt_task_objective_aligns_online_and_round_trips(least_mass_npz, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = pretrain.main(["--domain", "hunt", "--encoder", "deepset", "--data_path", str(least_mass_npz),
                            "--num_epochs", "3", "--batch_size", "64", "--device", "cpu", "--dim_hidden", "16",
                            "--output_root", str(tmp_path / "root"), *ALIGN])
    out = capsys.readouterr().out
    assert "latent alignment ON: lambda=0.2" in out and "Verified: " in out
    metrics = json.loads((Path(result.run_dir) / "metrics.json").read_text())
    _check_history(metrics["history"])
    al = metrics["alignment"]
    assert al["target"] == "online" and al["pairs"] == 64 and al["blur"] == 0.02 and al["scaling"] == 0.5
    assert 0 < al["val_pairs"] <= 300 and "lambda_at_best_epoch" in al and "val_align_r_at_best_epoch" in al
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["task"] == "pick_target" and payload["config"]["alignment"]["target"] == "online"
    assert payload["config"]["alignment"]["lambda"] == 0.2
    assert "identify_acc" in result.summary


def test_hunt_task_objective_default_is_untouched(least_mass_npz, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = pretrain.main(["--domain", "hunt", "--encoder", "deepset", "--data_path", str(least_mass_npz),
                            "--num_epochs", "1", "--batch_size", "64", "--device", "cpu", "--dim_hidden", "16",
                            "--output_root", str(tmp_path / "root")])
    assert "latent alignment ON" not in capsys.readouterr().out
    metrics = json.loads((Path(result.run_dir) / "metrics.json").read_text())
    assert metrics["alignment"] is None
    assert set(metrics["history"][0]) == {"epoch", "train", "val"}
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert "alignment" not in payload["config"]


def test_odd_even_belief_kl_aligns_online_and_stays_off_by_default(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    common = ["--domain", "odd_even", "--variant", "oe50_short", "--encoder", "st", "--objective", "belief_kl",
              "--data_source", "rolled", "--n_train_episodes", "40", "--n_val_episodes", "8", "--skip_probe",
              "--device", "cpu", "--num_inds", "4", "--dim_hidden", "16", "--num_post_sab", "0",
              "--output_root", str(tmp_path / "root")]
    result = pretrain.main([*common, "--num_epochs", "3", *ALIGN])
    out = capsys.readouterr().out
    assert "latent alignment ON: lambda=0.2" in out and "= 0.490 states" in out     # blur 0.02 x scale 24.5
    hist = json.loads((Path(result.run_dir) / "history.json").read_text())
    _check_history(hist)
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    al = payload["config"]["alignment"]
    assert al["target"] == "online" and al["pairs"] == 64 and al["lambda"] == 0.2
    assert "lambda_at_best_epoch" in al and np.isfinite(al["val_align_r_at_best_epoch"])
    last = torch.load(result.checkpoints["last"], map_location="cpu", weights_only=False)
    assert last["config"]["alignment"]["lambda_at_best_epoch"] == al["lambda_at_best_epoch"]
    # default: off, nothing added
    result = pretrain.main([*common, "--num_epochs", "1", "--run_tag", "plain"])
    assert "latent alignment ON" not in capsys.readouterr().out
    hist = json.loads((Path(result.run_dir) / "history.json").read_text())
    assert "align" not in hist[0] and "val_align_r" not in hist[0]
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert "alignment" not in payload["config"]


def test_omitted_blur_takes_the_domain_default(least_mass_npz, tmp_path, monkeypatch, capsys):
    """2026-09-19 (user OK): --sinkhorn_blur omitted -> Domain.default_sinkhorn_blur (hunt 0.02), and the
    run says so; the checkpoint records the resolved value."""
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    flags = [f for f in ALIGN if f not in ("--sinkhorn_blur", "0.02")]
    result = pretrain.main(["--domain", "hunt", "--encoder", "deepset", "--data_path", str(least_mass_npz),
                            "--num_epochs", "2", "--batch_size", "64", "--device", "cpu", "--dim_hidden", "16",
                            "--output_root", str(tmp_path / "root"), *flags])
    out = capsys.readouterr().out
    assert "blur 0.02 [domain default]" in out
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["alignment"]["blur"] == 0.02
    assert json.loads((Path(result.run_dir) / "args.json").read_text())["sinkhorn_blur_source"] == "domain default"
