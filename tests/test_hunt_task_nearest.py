"""Hunt's ``task_nearest`` objective (batch 10.10, 2026-09-14; plan 10.10): the nearest-live label on
hand-built states (ties -> lowest index; all-dead rows masked), agreement with the collector's own
``target`` on a Cluster-Hunt file, the objective end to end through the package command, the refusal
on a pick-a-target variant, and the objective table."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ST_ROOT = Path(__file__).resolve().parents[1]
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains.hunt", reason="needs the pomdp-domains hunt-envs branch")

from set_transformer.rl import collect, pretrain, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.domains import hunt  # noqa: E402


def test_nearest_live_offset_on_hand_built_states():
    centers = torch.tensor([
        [[0.3, 0.0], [0.1, 0.0], [0.0, -0.2], [5.0, 5.0], [0.0, 0.0]],     # nearest live = index 2 (index 1 dead)
        [[0.2, 0.0], [0.0, 0.2], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],      # tie 0 vs 1 -> lowest index 0
        [[0.5, 0.0], [0.6, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],      # nothing alive
    ])
    alive = torch.tensor([[1, 0, 1, 1, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.float32)
    offset, index, live = hunt.nearest_live_offset(centers, alive)
    assert index.tolist()[:2] == [2, 0] and live.tolist() == [True, True, False]
    assert torch.equal(offset[0], centers[0, 2]) and torch.equal(offset[1], centers[1, 0])
    # the loss and the metrics use live rows only; a perfect prediction scores zero / 1.0
    y = offset.clone()
    y[2] = torch.tensor([9.0, 9.0])                                         # garbage on the dead row is ignored
    batch = {"centers": centers, "alive": alive}
    assert float(hunt.task_nearest_loss(y, batch)) == 0.0
    m = hunt.task_nearest_metrics(y, batch, 10.0)
    assert m == {"loc_mae": 0.0, "identify_acc": 1.0, "within_1.0": 1.0}
    # a prediction nearer to another live centre is misidentified; error reported in arena units
    y2 = y.clone()
    y2[0] = torch.tensor([0.3, 0.0])                                        # sits on centre 0, not the nearest (2)
    m = hunt.task_nearest_metrics(y2, batch, 10.0)
    assert m["identify_acc"] == 0.5 and m["loc_mae"] == pytest.approx(10.0 * float(torch.linalg.norm(y2[0] - offset[0])) / 2)
    # an all-dead batch gives a zero loss with a gradient path, not an error
    dead = {"centers": centers[2:], "alive": alive[2:]}
    assert float(hunt.task_nearest_loss(y[2:].requires_grad_(), dead)) == 0.0


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hunt_nearest")
    out = {}
    for variant in ("cluster_hunt", "least_mass"):
        path = tmp / f"{variant}.npz"
        collect.main(["--domain", "hunt", "--variant", variant, "--num_episodes", "12", "--seed", "5",
                      "--output_file", str(path)])
        out[variant] = path
    return out


def test_label_agrees_with_the_collectors_target_on_a_cluster_hunt_file(datasets):
    with np.load(datasets["cluster_hunt"], allow_pickle=True) as z:
        centers, alive, target, index = (torch.as_tensor(z[k]) for k in ("centers", "alive", "target", "target_index"))
    offset, idx, live = hunt.nearest_live_offset(centers, alive)
    assert bool(live.any()) and bool((~live).any())                        # both kinds of rows are in the file
    assert torch.equal(idx[live], index[live]) and torch.allclose(offset[live], target[live])
    assert torch.all(index[~live] == -1)                                     # the collector's mark for "no live cluster"


def test_task_nearest_end_to_end_and_the_objective_table(datasets, tmp_path, monkeypatch, capsys):
    assert sorted(pretrain.objectives_for(hunt.HUNT)) == ["reconstruction", "task", "task_nearest"]
    assert pretrain.default_objective_name(hunt.HUNT) == "task"
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    root = tmp_path / "root"
    result = pretrain.main(["--domain", "hunt", "--encoder", "deepset", "--objective", "task_nearest",
                            "--data_path", str(datasets["cluster_hunt"]), "--num_epochs", "2", "--batch_size", "64",
                            "--device", "cpu", "--dim_hidden", "16", "--output_root", str(root)])
    assert "Verified: WeightedDeepSetFeaturesExtractor encoder matches" in capsys.readouterr().out
    assert {"loc_mae", "identify_acc", "within_1.0"} <= set(result.summary)
    ck = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert (ck["config"]["objective"], ck["config"]["task"], ck["config"]["variant"]) == ("task_nearest", "nearest", "cluster_hunt")
    assert ck["head_state_dict"]["4.weight"].shape[0] == 2 and ck["pretraining_run"]["objective"] == "task_nearest"
    assert list(root.glob("hunt/cluster_hunt/pretrain/deepset/task_nearest/*_seed0/run_status.json"))   # the root layout
    found = run_records.latest_pretrain_checkpoint("hunt", "cluster_hunt", "deepset", "task_nearest", root=root)
    assert Path(found) == result.rl_checkpoint.resolve()
    train_mod.main(["--domain", "hunt", "--encoder", "deepset", "--variant", "cluster_hunt", "--dim_hidden", "16",
                    "--pretrained_path", str(found), "--frozen", "--output_root", str(root), "--dry_run"])


def test_task_nearest_refuses_a_pick_target_variant(datasets, tmp_path, capsys):
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "hunt", "--encoder", "deepset", "--objective", "task_nearest",
                       "--data_path", str(datasets["least_mass"]), "--dim_hidden", "16",
                       "--output_root", str(tmp_path / "root")])
    assert "use --objective task" in capsys.readouterr().err
