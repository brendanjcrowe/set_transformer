"""Ant-Tag's ``task`` pretraining objective (batch 10.9, 2026-09-14; design in refactor_plans.md 10.9).

Pins: the registry's heads (position on the smart family, den_mass on the counterweighted dens,
none on ghost / dens); the collector's labels and their alignment with the particles THROUGH the
spread rebalance (the kept-rows contract of 10.5); the den-share function on hand-built clouds;
a tiny collect -> pretrain -> reload per learned encoder on ``smart`` and the den head on
``cdens_terminal`` through the package commands; the refusals; and that hunt's task objective
still runs on the moved loop (``rl/pretrain_objectives/task_head.py``).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ST_ROOT = Path(__file__).resolve().parents[1]
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains")
pytest.importorskip("mujoco")

from set_transformer.rl import collect, encoders, pretrain, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.domains import ant_tag, hunt  # noqa: E402
from set_transformer.rl.pretrain_objectives import task_head  # noqa: E402

ANT_TAG = ant_tag.ANT_TAG
TINY_COLLECT = ["--num_episodes", "6", "--timesteps", "25", "--num_particles", "30", "--seed", "0",
                "--pursuit_fraction", "0", "--fully_observed_fraction", "0.3"]


# --------------------------------------------------------------------------
# Registry and the pure functions
# --------------------------------------------------------------------------

def test_registry_declares_the_heads_per_family():
    heads = {name: v.task_heads for name, v in ant_tag.VARIANTS.items()}
    for name in ("base", "smart", "smart_hard", "smart_mid", "smart_hard_slow", "smart_mid_slow", "smart_mid_slow_v15"):
        assert heads[name] == ("position",), name
    for name in ("cdens", "cdens_hard", "cdens_terminal", "cdens_nospook"):
        assert heads[name] == ("den_mass",), name
    assert heads["ghost"] == () and heads["dens"] == ()
    assert set(ant_tag.TASK_HEADS) == {"position", "den_mass"}
    assert set(ANT_TAG.pretraining.objectives) == {"task"} and ANT_TAG.pretraining.default_objective is None
    assert sorted(pretrain.objectives_for(ANT_TAG)) == ["reconstruction", "task"]


def test_den_shares_on_hand_built_clouds():
    d = np.array([1.0, 1.0]) / np.sqrt(2)
    cand = np.stack([-2.4 * d, 2.4 * d, -6.75 * d, 6.75 * d])   # the env's [-h, +h, -f, +f]
    heavy, light = cand[1], cand[2]
    # all mass in the +h den -> [0, 1, 0, 0, 0]; uniform weights; static [K, 2] centres
    p = heavy[None, None] + 0.3 * np.array([[[1, 0], [-1, 0], [0, 1], [0, -1]]]) / np.sqrt(2)
    assert np.allclose(ant_tag.den_shares(p, np.full((1, 4), 0.25), cand, 0.4), [[0, 1, 0, 0, 0]])
    # weights decide the shares, not counts: one particle in +h, one in -f, two outside
    p = np.array([[heavy, light, [0.0, 0.0], [4.0, -4.0]]])
    w = np.array([[0.6, 0.3, 0.05, 0.05]])
    s = ant_tag.den_shares(p, w, cand, 0.4)
    assert np.allclose(s, [[0, 0.6, 0.3, 0, 0.1]], atol=1e-6) and np.allclose(s.sum(1), 1.0)
    # unnormalised weights are normalised per row; a particle just inside the rim counts; per-row
    # centres ([S, K, 2]) give the same answer as the static table
    rim = (heavy + np.array([0.4, 0.0])).astype(np.float32).astype(np.float64)   # a float32 rim point: 0.4000001 off
    assert np.linalg.norm(rim - heavy) > 0.4 - 1e-7          # ON the rim, not inside it: the leash puts strays here
    s = ant_tag.den_shares(np.array([[rim, light]]), np.array([[2.0, 2.0]]), cand, 0.4)
    assert np.allclose(s, [[0, 0.5, 0.5, 0, 0]], atol=1e-6)
    assert np.array_equal(s, ant_tag.den_shares(np.array([[rim, light]]), np.array([[2.0, 2.0]]), cand[None], 0.4))


def test_task_loss_and_metrics_per_head():
    heads = ("position",)
    y = torch.tensor([[0.5, 0.0], [0.0, 0.0]])
    batch = {"target_scaled": torch.tensor([[0.5, 0.0], [0.0, 1.0]])}
    assert float(ant_tag.task_loss(y, batch, heads)) == pytest.approx(0.25)     # mean over 4 entries
    m = ant_tag.task_metrics(y, batch, heads, scale=4.5, tag_radius=1.5)
    assert m["position_mae"] == pytest.approx(4.5 * 0.5) and m["within_tag_radius"] == 0.5
    heads = ("den_mass",)
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 10.0]])
    batch = {"den_shares": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.6, 0.0, 0.4]]),
             "den_occupied": torch.tensor([0, 2])}
    assert float(ant_tag.task_loss(logits, batch, heads)) > 0
    m = ant_tag.task_metrics(logits, batch, heads, scale=7.0, tag_radius=0.6)
    assert m["den_kl"] > 0 and m["den_acc_belief"] == 0.5 and m["den_acc_truth"] == 0.5
    # the KL of a perfect prediction is zero (0 log 0 = 0)
    perfect = torch.log(batch["den_shares"].clamp_min(1e-12))
    assert ant_tag.task_metrics(perfect, batch, heads, 7.0, 0.6)["den_kl"] == pytest.approx(0.0, abs=1e-5)


# --------------------------------------------------------------------------
# The collector: labels, aligned through the spread rebalance
# --------------------------------------------------------------------------

def _collect(variant, tmp_path, monkeypatch, *flags):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    out = tmp_path / f"{variant}.npz"
    collect.main(["--domain", "ant_tag", "--variant", variant, *TINY_COLLECT, *flags, "--output_file", str(out)])
    return out


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ant_tag_task")
    out = {}
    for variant in ("smart", "cdens_terminal"):
        path = tmp / f"{variant}.npz"
        collect.main(["--domain", "ant_tag", "--variant", variant, *TINY_COLLECT, "--rebalance_no_upsample",
                      "--output_file", str(path)])
        out[variant] = path
    return out


def test_labels_ride_along_and_stay_aligned_through_the_rebalance(tmp_path, monkeypatch):
    plain = _collect("smart", tmp_path / "a", monkeypatch, "--no_rebalance")
    with np.load(plain, allow_pickle=True) as z:
        P, T, A, S = z["particles"], z["target"], z["ant"], z["step"]
        meta = json.loads(str(z["metadata"]))
    assert T.shape == (len(P), 2) and A.shape == (len(P), 2) and S.dtype == np.int32 and S[0] == 0
    assert np.abs(T).max() <= 4.5 + 1e-6 and meta["label_arrays"] == ["ant", "target", "step"]
    assert meta["task_heads"] == ["position"] and meta["tag_radius"] == 1.5 and meta["visible_radius"] == 3.0
    # a fully-observed episode collapses the cloud onto the target: on those rows the weighted
    # mean sits within the tag radius of the label, which is the alignment check
    with np.load(plain, allow_pickle=True) as z:
        W = z["weights"]
    mean = (W[:, :, None] * P).sum(1)
    spread = np.sqrt((W[:, :, None] * (P - mean[:, None]) ** 2).sum(1)).mean(1)
    tight = spread < 0.2
    assert tight.any()
    assert np.all(np.linalg.norm(mean[tight] - T[tight], axis=1) < 1.5)
    # the rebalance hook returns the kept rows (10.5 contract), so the collector applies the same
    # selection to the labels: the four-value form, and the selected particles are the raw rows
    parser = collect.build_parser(ANT_TAG, selectors=False)
    args = parser.parse_args(["--variant", "smart", *TINY_COLLECT])
    options = ANT_TAG.collection.resolve_arguments(parser, args, ANT_TAG)
    P2, W2, _, kept = ANT_TAG.collection.rebalance(args, options, P, W, None)
    assert len(P2) == len(kept) and np.array_equal(P2, P[kept]) and np.array_equal(W2, W[kept])
    assert set(kept.tolist()) <= set(range(len(P))) and len(set(kept.tolist())) < len(kept)   # upsampled: repeats
    # and the entry point's two-array form is the same selection
    a, b = ant_tag._rebalance_by_spread(P, W, collapsed_threshold=args.collapsed_threshold,
                                        diffuse_threshold=args.diffuse_threshold, seed=args.seed)
    assert np.array_equal(a, P2) and np.array_equal(b, W2)


def test_den_family_stores_the_episode_dens_and_the_truth(datasets):
    with np.load(datasets["cdens_terminal"], allow_pickle=True) as z:
        D, occ, P, W = z["den_positions"], z["den_occupied"], z["particles"], z["weights"]
        meta = json.loads(str(z["metadata"]))
    assert D.shape == (len(P), 2, 2) and set(np.unique(occ)) <= {0, 1, 2, 3}
    cand = np.asarray(meta["den_candidates"])
    assert cand.shape == (4, 2) and np.allclose(np.linalg.norm(cand, axis=1), [2.4, 2.4, 6.75, 6.75], atol=1e-4)
    # the truth index names a candidate at the occupied den's position (heavy or light of the episode)
    occupied_pos = cand[occ]
    assert np.all(np.isclose(np.linalg.norm(occupied_pos - D[:, 0], axis=1), 0, atol=1e-4)
                  | np.isclose(np.linalg.norm(occupied_pos - D[:, 1], axis=1), 0, atol=1e-4))
    # heavy at 2.4 out on a diagonal, light at 6.75 on the opposite side, per episode
    assert np.allclose(np.linalg.norm(D[:, 0], axis=1), 2.4, atol=1e-4)
    assert np.allclose(np.linalg.norm(D[:, 1], axis=1), 6.75, atol=1e-4)
    assert np.all(np.sign(D[:, 0, 0]) == -np.sign(D[:, 1, 0]))
    assert meta["den_radius"] == 0.4 and meta["label_arrays"][-2:] == ["den_positions", "den_occupied"]
    assert meta["task_heads"] == ["den_mass"] and meta["tag_radius"] == 0.6
    shares = ant_tag.den_shares(P, W, cand, meta["den_radius"])
    assert shares.shape == (len(P), 5) and np.allclose(shares.sum(1), 1.0, atol=1e-5) and shares.min() >= 0.0
    # the head's radius (1.0) folds the resampling strays back: no less den mass than the env's 0.4 disc
    wide = ant_tag.den_shares(P, W, cand, ant_tag.DEN_SHARE_RADIUS)
    assert np.all(wide[:, :4].sum(1) >= shares[:, :4].sum(1) - 1e-6) and wide[:, 4].mean() <= shares[:, 4].mean()
    # the step-0 rows hold the filter's four-candidate prior: mass on all four discs, little outside
    with np.load(datasets["cdens_terminal"], allow_pickle=True) as z:
        step0 = z["step"] == 0
    assert shares[step0][:, :4].mean(0).min() > 0.05 and shares[step0][:, 4].mean() < 0.1


# --------------------------------------------------------------------------
# The objective through the package command
# --------------------------------------------------------------------------

ENCODER_FLAGS = {
    "st": ["--num_inds", "4", "--dim_hidden", "16", "--num_post_sab", "0"],
    "cgf": ["--num_cgf_features", "8", "--t_param", "polar", "--feature_mode", "K_grad"],
    "deepset": ["--dim_hidden", "16"],
    "pointnet": ["--dim_hidden", "16"],
}


@pytest.mark.parametrize("encoder", sorted(ENCODER_FLAGS))
def test_position_head_round_trips_into_the_rl_extractor(encoder, datasets, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    root = tmp_path / "root"
    result = pretrain.main(["--domain", "ant_tag", "--encoder", encoder, "--objective", "task",
                            "--data_path", str(datasets["smart"]), "--num_epochs", "2", "--batch_size", "32",
                            "--device", "cpu", "--output_root", str(root), *ENCODER_FLAGS[encoder]])
    out = capsys.readouterr().out
    assert f"Verified: {encoders.get(encoder).extractor_class.__name__} encoder matches" in out
    assert {"position_mae", "within_tag_radius", "best_val_loss"} <= set(result.summary)
    assert list(root.glob(f"ant_tag/smart/pretrain/{encoder}/task/*_seed0/run_status.json"))
    ck = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    c = ck["config"]
    assert (c["objective"], c["task"], c["task_heads"], c["variant"], c["arena_scale"]) == (
        "task", "position", ["position"], "smart", 4.5)
    assert c["encoder"] == encoder and c["encoder_params"] > 0
    assert ck["head_state_dict"]["4.weight"].shape[0] == 2
    assert ck["pretraining_run"]["objective"] == "task" and ck["pretraining_run"]["domain"] == "ant_tag"
    metrics = json.loads((result.run_dir / "metrics.json").read_text())
    assert set(metrics["val_metrics"]) == {"position_mae", "within_tag_radius"}
    found = run_records.latest_pretrain_checkpoint("ant_tag", "smart", encoder, "task", root=root)
    assert Path(found) == result.rl_checkpoint.resolve()
    # the trainer takes it frozen, through the arm's own historical flag spelling
    train_mod.main(["--domain", "ant_tag", "--encoder", encoder, "--variant", "smart", "--pretrained_path",
                    str(found), "--frozen", "--output_root", str(root), "--dry_run", *ENCODER_FLAGS[encoder]])
    [record] = list(root.glob(f"ant_tag/smart/rl/{encoder}/*_seed0/run_config.json"))
    config = json.loads(record.read_text())
    assert str(found) in config.values() and any(k.endswith("frozen") and v is True for k, v in config.items())


def test_den_mass_head_on_cdens_terminal(datasets, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    root = tmp_path / "root"
    # --variant omitted: the file's metadata names it, and the heads come from the registry
    result = pretrain.main(["--domain", "ant_tag", "--encoder", "deepset", "--objective", "task",
                            "--data_path", str(datasets["cdens_terminal"]), "--num_epochs", "2",
                            "--batch_size", "32", "--device", "cpu", "--dim_hidden", "16", "--output_root", str(root)])
    assert "Verified: WeightedDeepSetFeaturesExtractor encoder matches" in capsys.readouterr().out
    assert {"den_kl", "den_acc_belief", "den_acc_truth"} <= set(result.summary)
    assert 0.0 <= result.summary["den_acc_belief"] <= 1.0 and result.summary["den_kl"] >= 0.0
    ck = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert (ck["config"]["task"], ck["config"]["variant"], ck["config"]["arena_scale"]) == ("den_mass", "cdens_terminal", 7.0)
    assert ck["head_state_dict"]["4.weight"].shape[0] == 5           # four candidates + outside
    assert ck["config"]["den_share_radius"] == 1.0                   # the head counts strays back into their den
    assert list(root.glob("ant_tag/cdens_terminal/pretrain/deepset/task/*_seed0/checkpoints/checkpoint_best.pt"))


def test_refusals(datasets, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    root = str(tmp_path / "root")
    # an analytic encoder has nothing to train
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "ant_tag", "--encoder", "gaussian", "--objective", "task",
                       "--data_path", str(datasets["smart"]), "--output_root", root])
    assert "has no parameters to" in capsys.readouterr().err      # the door refuses it first
    # a variant without heads
    monkeypatch.setitem(ant_tag.VARIANTS, "smart", ant_tag.VARIANTS["smart"].__class__(
        **{**ant_tag.VARIANTS["smart"].__dict__, "task_heads": ()}))
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "ant_tag", "--encoder", "deepset", "--objective", "task",
                       "--data_path", str(datasets["smart"]), "--dim_hidden", "16", "--output_root", root])
    assert "declares no task heads" in capsys.readouterr().err
    monkeypatch.undo()
    # the file of another variant
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "ant_tag", "--encoder", "deepset", "--objective", "task", "--variant", "cdens_terminal",
                       "--data_path", str(datasets["smart"]), "--dim_hidden", "16", "--output_root", root])
    assert "was collected on 'smart'" in capsys.readouterr().err
    # a recorded-style file without labels names the collect command
    bare = tmp_path / "bare.npz"
    with np.load(datasets["smart"], allow_pickle=True) as z:
        np.savez(bare, particles=z["particles"], weights=z["weights"], particle_scale=z["particle_scale"],
                 metadata=z["metadata"])
    with pytest.raises(SystemExit):        # the door turns the loader's ValueError into a parser error
        pretrain.main(["--domain", "ant_tag", "--encoder", "deepset", "--objective", "task",
                       "--data_path", str(bare), "--dim_hidden", "16", "--output_root", root])
    err = capsys.readouterr().err
    assert "missing arrays ['ant', 'target', 'step']" in err and "collect it with" in err


def test_hunt_keeps_its_names_on_the_shared_loop():
    assert issubclass(hunt.TaskData, task_head.TaskData)
    assert hunt.TaskEncoderWithHead is task_head.TaskEncoderWithHead
    assert hunt.build_task_extractor is task_head.build_task_extractor
    assert hunt._task_run_name is task_head.task_run_name
    assert hunt.TASK_OBJECTIVE.run_name is task_head.task_run_name
    assert ant_tag.ANT_TAG_TASK_OBJECTIVE.locate is task_head.locate_from_dataset
