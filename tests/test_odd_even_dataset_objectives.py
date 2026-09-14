"""Odd-Even's exact-posterior objectives read the collected dataset (batch 10.5, decision 6, 2026-09-14).

The Odd-Even collection labels every recorded belief (true state, optimal prediction, episode) read
off the live env, its rebalance returns the kept rows so the labels follow, and `belief_kl` /
`mode_ce` / `state_ce` take `--data_path` (default: the variant's dataset under the run root when it
exists; `--data_source rolled` keeps the recorded runs' path). The tests pin what would silently
change a pretrained arm: the rows equal the rolling path's rows exactly (same factory, same seeding,
the never-acted-on belief after the last step dropped, the step index shifted to 1-based), the split
is episode-disjoint, and the reconstruction objective runs on the same file.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains")

from set_transformer.rl import collect, pretrain, run_records  # noqa: E402
from set_transformer.rl.domains import odd_even as oe  # noqa: E402

VARIANT, EPISODES, SEED = "oe50_short", 6, 5
CAP = oe.variants.episode_cap(VARIANT)
CENTRE = oe.variants.state_centre(VARIANT)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """Six labelled, unbalanced episodes under a run root (the recipe's collect stage, small)."""
    root = tmp_path_factory.mktemp("oe105") / "root"
    collect.main(["--domain", "odd_even", "--variant", VARIANT, "--num_episodes", str(EPISODES),
                  "--seed", str(SEED), "--no_rebalance", "--output_root", str(root)])
    return root


def _dataset(root):
    return run_records.dataset_path("odd_even", VARIANT, root=root)


def test_labels_are_written_and_the_rows_equal_the_rolling_path(root):
    with np.load(_dataset(root), allow_pickle=True) as z:
        assert {"true_state", "optimal_prediction", "episode", "steps"} <= set(z.files)
        d = {k: np.asarray(z[k]) for k in ("particles", "weights", "steps", "true_state", "optimal_prediction", "episode")}
    assert len(d["steps"]) == EPISODES * (CAP + 1)            # reset + cap steps per episode
    keep = d["steps"] < CAP                                    # the belief after the last step is never acted on
    ref = oe.collect_rollouts(VARIANT, EPISODES, SEED)
    assert np.array_equal(d["particles"][keep] - np.float32(CENTRE), ref["particles"])
    assert np.array_equal(d["weights"][keep], ref["weights"])
    assert np.array_equal(d["true_state"][keep], ref["true_state"])
    assert np.array_equal(d["optimal_prediction"][keep], ref["mode"])
    assert np.array_equal(d["episode"][keep], ref["group"]) and np.array_equal(d["steps"][keep] + 1, ref["step"])
    # the exact-support rows: one particle per state, in order
    assert np.array_equal(d["particles"][:, :, 0], np.broadcast_to(np.arange(1, 51, dtype=np.float32), d["particles"].shape[:2]))


def test_the_default_rebalance_keeps_the_labels_aligned(tmp_path):
    out = tmp_path / "rebalanced.npz"
    collect.main(["--domain", "odd_even", "--variant", VARIANT, "--num_episodes", str(EPISODES),
                  "--seed", str(SEED), "--output_file", str(out)])
    with np.load(out, allow_pickle=True) as z:
        w, mode, ts, ep = (np.asarray(z[k]) for k in ("weights", "optimal_prediction", "true_state", "episode"))
    assert 0 < len(w) < EPISODES * (CAP + 1)                   # rows were dropped, none duplicated
    assert np.array_equal(w.argmax(1) + 1, mode)               # the label still belongs to its row
    for e in np.unique(ep):
        assert len(set(ts[ep == e].tolist())) == 1             # one hidden state per episode


def test_loader_drops_the_last_belief_shifts_the_step_and_splits_by_episode(root, tmp_path):
    train, val = oe.load_posterior_snapshots(_dataset(root), VARIANT, 0.2)
    n_val = math.ceil(0.2 * EPISODES)
    assert set(val["group"]) == set(range(EPISODES - n_val, EPISODES)) and not (set(train["group"]) & set(val["group"]))
    assert len(train["weights"]) + len(val["weights"]) == EPISODES * CAP
    assert train["step"].min() == 1 and train["step"].max() == CAP and train["n"] == 50 and train["cap"] == CAP
    with pytest.raises(ValueError, match="collected on variant"):
        oe.load_posterior_snapshots(_dataset(root), "oe50", 0.2)
    with pytest.raises(ValueError, match="leaves no training episode"):
        oe.load_posterior_snapshots(_dataset(root), VARIANT, 0.99)
    # a dataset written before 10.5 carries no labels and is refused with the collect command named
    with np.load(_dataset(root), allow_pickle=True) as z:
        old = tmp_path / "old.npz"
        np.savez(old, particles=z["particles"], weights=z["weights"], steps=z["steps"], metadata=z["metadata"])
    with pytest.raises(ValueError, match="missing arrays"):
        oe.load_posterior_snapshots(old, VARIANT, 0.2)


TINY = ["--num_epochs", "1", "--skip_probe", "--dim_hidden", "16", "--device", "cpu"]


@pytest.mark.parametrize("encoder", ["deepset", "st"])
def test_belief_kl_reads_the_variants_dataset_under_the_root_and_records_the_source(encoder, root, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    result = pretrain.main(["--domain", "odd_even", "--variant", VARIANT, "--encoder", encoder, "--objective", "belief_kl",
                            "--output_root", str(root), *TINY])
    out = capsys.readouterr().out
    assert "Beliefs: dataset" in out and f"loaded {(EPISODES - 1) * CAP} train rows" in out and "Verified:" in out
    config = json.loads((result.run_dir / "run_config.json").read_text())
    assert config["data_source"] == "dataset" and config["data_path"] == str(_dataset(root))
    assert result.rl_checkpoint.exists()


def test_rolled_on_request_and_a_required_dataset_is_refused_when_missing(root, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    pretrain.main(["--domain", "odd_even", "--variant", VARIANT, "--encoder", "st", "--objective", "mode_ce",
                   "--data_source", "rolled", "--n_train_episodes", "3", "--n_val_episodes", "2",
                   "--output_root", str(root), *TINY])
    out = capsys.readouterr().out
    assert "Beliefs: rolled" in out and "collected 90 train rows / 60 val rows" in out
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--domain", "odd_even", "--variant", VARIANT, "--encoder", "st", "--objective", "belief_kl",
                       "--data_source", "dataset", "--output_root", str(tmp_path / "empty"), "--dry_run", *TINY])
    assert exc.value.code == 2 and "--data_source dataset: no collected dataset" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "odd_even", "--variant", VARIANT, "--encoder", "st", "--objective", "belief_kl",
                       "--data_source", "rolled", "--data_path", str(_dataset(root)), "--dry_run", *TINY])
    assert "contradict" in capsys.readouterr().err
    # without --variant the labelled file names it, as a reconstruction dataset does (the domain is
    # still needed: a domain-declared objective cannot be found without it)
    pretrain.main(["--domain", "odd_even", "--encoder", "st", "--objective", "belief_kl", "--data_path", str(_dataset(root)),
                   "--output_root", str(root), "--dry_run", *TINY])
    out = capsys.readouterr().out
    assert "Beliefs: dataset" in out and f"variant {VARIANT}" in out


def test_reconstruction_runs_on_the_same_labelled_file(root, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.setenv("WANDB_MODE", "offline")
    result = pretrain.main(["--domain", "odd_even", "--variant", VARIANT, "--encoder", "st", "--objective", "reconstruction",
                            "--loss_type", "sinkhorn", "--sinkhorn_blur", "0.02", "--num_epochs", "1", "--batch_size", "8",
                            "--num_workers", "0", "--device", "cpu", "--num_inds", "4", "--dim_hidden", "16",
                            "--output_root", str(root)])
    assert "Verified: SetTransformerFeaturesExtractor encoder matches" in capsys.readouterr().out
    assert result.run_dir.parent == root / "odd_even" / VARIANT / "pretrain" / "st" / "reconstruction"
