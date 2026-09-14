"""Hunt collection and the ``task`` objective (plan section 9, batch 9.2, 2026-09-13).

Pins: the shared collector's new per-snapshot label hook (stacked, truncated, refused with
rebalancing); collector PARITY with the recorded ``src/hunt_tasks/pretrain/collect.py`` for the
same seed (every particle and label, terminal rows dropped); the slot loss is permutation
invariant; a tiny collect -> pretrain -> reload run per learned encoder through the package
commands; the refusals.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_ST_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_ST_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains", reason="envs are registered by pdomains")
pytest.importorskip("pdomains.hunt", reason="needs the pomdp-domains hunt-envs branch")

import torch  # noqa: E402

from set_transformer.rl import collect, encoders, pretrain, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.domains import hunt  # noqa: E402
from set_transformer.rl.domains.base import Collection  # noqa: E402
from set_transformer.rl.pretrained_encoder import verify_matches_checkpoint  # noqa: E402

HUNT = hunt.HUNT


def _collect(variant: str, episodes: int, seed: int, **extra_flags):
    parser = collect.build_parser(HUNT, selectors=False)
    argv = ["--variant", variant, "--num_episodes", str(episodes), "--seed", str(seed)]
    for k, v in extra_flags.items():
        argv += [f"--{k}", str(v)]
    args = parser.parse_args(argv)
    options = HUNT.collection.resolve_arguments(parser, args, HUNT)
    extras = {}
    particles, weights, steps = collect.collect_arrays(HUNT, args, options, progress=False,
                                                       extras_out=extras)
    return particles, weights, steps, extras


def _drop_terminal_rows(steps: np.ndarray) -> np.ndarray:
    """Mask of the rows the recorded collector also produced: it stopped BEFORE the state after
    the last step of an episode, the shared loop records it."""
    keep = np.ones(len(steps), dtype=bool)
    ends = np.flatnonzero(np.diff(steps) < 0)          # a row whose successor restarts at 0
    keep[ends] = False
    keep[-1] = False
    return keep


# --------------------------------------------------------------------------
# The hook in the shared collector
# --------------------------------------------------------------------------

def test_collection_record_and_defaults():
    assert isinstance(HUNT.collection, Collection)
    assert HUNT.collection.defaults == {"seed": 7, "num_episodes": 4000}
    a = collect.build_parser(HUNT).parse_args([])
    assert (a.seed, a.num_episodes, a.variant, a.pursuit_frac) == (7, 4000, "least_mass", 0.6)
    assert HUNT.pretraining.default_objective == "task"
    assert sorted(pretrain.objectives_for(HUNT)) == ["reconstruction", "task", "task_nearest"]   # 10.10


def test_snapshot_extras_are_stacked_aligned_and_truncated():
    particles, weights, steps, extras = _collect("least_mass", 3, 11)
    n = len(particles)
    assert set(extras) == {"agent", "centers", "alive", "counts", "sigmas", "target",
                           "target_index", "step"}
    assert extras["centers"].shape == (n, 5, 2) and extras["target"].shape == (n, 2)
    assert extras["target_index"].shape == (n,) and extras["step"].tolist() == steps.tolist()
    assert np.allclose(weights, 0.01)
    # the label refers to the same state as the cloud: the target's centre sits inside the
    # cloud's bounding box, agent-relative, once scaled back to raw units
    lo, hi = particles.min(1), particles.max(1)
    t = extras["target"] * hunt.ARENA_SCALE
    assert np.all(t >= lo - 3.0) and np.all(t <= hi + 3.0)
    # truncation applies to the labels too
    _, _, s2, e2 = _collect("least_mass", 3, 11, max_snapshots=7)
    assert len(s2) == 7 and all(len(v) == 7 for v in e2.values())
    assert np.array_equal(e2["target"], extras["target"][:7])


def test_extras_with_rebalancing_are_refused_but_the_domain_does_not_rebalance(tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    out = tmp_path / "d.npz"
    collect.main(["--domain", "hunt", "--variant", "most_var", "--num_episodes", "2", "--seed", "3",
                  "--output_file", str(out)])
    with np.load(out, allow_pickle=True) as z:
        assert "target_index" in z.files and "sigmas" in z.files
        meta = json.loads(str(z["metadata"]))
        assert meta["task"] == "pick_target" and meta["particle_scale"] == 10.0
        assert meta["env_kwargs"]["target_rule"] == "max_var"
        # most_var: the target is the widest live cluster and the counts are equal
        sig, idx, alive, counts = z["sigmas"], z["target_index"], z["alive"], z["counts"]
        for row in range(len(idx)):
            live = alive[row] > 0
            assert idx[row] == int(np.argmax(np.where(live, sig[row], -1)))
            assert counts[row][live].max() - counts[row][live].min() <= 1
    # a domain that rebalances AND records labels is refused (synthetic record)
    rebalancing = Collection(**{**{f.name: getattr(HUNT.collection, f.name)
                                   for f in Collection.__dataclass_fields__.values()},
                                "rebalance": lambda a, o, p, w, s: (p, w, s)})
    import dataclasses
    domain = dataclasses.replace(HUNT, collection=rebalancing)
    monkeypatch.setattr(collect._domains, "get", lambda name: domain)
    with pytest.raises(SystemExit):
        collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "1",
                      "--output_file", str(tmp_path / "e.npz")])
    collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "1",
                  "--no_rebalance", "--output_file", str(tmp_path / "e.npz")])
    assert (tmp_path / "e.npz").exists()


# --------------------------------------------------------------------------
# Parity with the recorded collector
# --------------------------------------------------------------------------

def _recorded_collector():
    src = _REPO_ROOT / "src" / "hunt_tasks"
    if not (src / "pretrain" / "collect.py").is_file():
        pytest.skip("the recorded collector (src/hunt_tasks) is not beside this checkout")
    saved = list(sys.path)
    sys.path.insert(0, str(src))
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = importlib.util.spec_from_file_location("_hunt_record_collect", src / "pretrain" / "collect.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved
    return module


@pytest.mark.parametrize("variant,recorded,choices", [
    ("least_mass", "collect_minmass", [2, 3, 4, 5, 5, 5]),
    ("cluster_hunt", "collect_cluster_hunt", [1, 2, 3, 4, 5, 5, 5]),
])
def test_collector_parity_with_the_record(variant, recorded, choices):
    """Same seed -> the same behaviour, the same states, the same labels. The record stored
    the env's SCALED world-frame particles beside the scaled agent; the harness stores the
    agent-relative raw cloud, so the comparison rescales."""
    rec = _recorded_collector()
    if variant == "least_mass":
        cfg = rec.MinMassHuntConfig()
    else:
        cfg = rec.ClusterHuntConfig(hit_radius=0.6, min_sep=2.5, max_steps=60)
    old = getattr(rec, recorded)(4, cfg, 7, choices, 0.6)
    particles, weights, steps, extras = _collect(variant, 4, 7)
    keep = _drop_terminal_rows(steps)
    assert keep.sum() == len(old["particles"]), "row counts differ beyond the terminal rows"
    relative_raw = (old["particles"] - old["agent"][:, None, :]) * 10.0
    assert np.allclose(particles[keep], relative_raw, atol=1e-4)
    assert np.array_equal(extras["agent"][keep], old["agent"])
    assert np.array_equal(extras["centers"][keep], old["centers"])
    assert np.array_equal(extras["alive"][keep], old["alive"])
    assert np.array_equal(extras["target"][keep], old["target"])
    if "counts" in old:
        assert np.array_equal(extras["counts"][keep], old["counts"])


# --------------------------------------------------------------------------
# The objective
# --------------------------------------------------------------------------

def test_matched_slot_loss_is_permutation_invariant_and_zero_at_the_target():
    torch.manual_seed(0)
    off = torch.randn(8, 5, 2)
    al = torch.ones(8, 5)
    perm = torch.randperm(5)
    a = hunt.matched_slot_loss(off[:, perm], torch.zeros(8, 5), off, al)
    b = hunt.matched_slot_loss(off, torch.zeros(8, 5), off, al)
    assert abs(float(a) - float(b)) < 1e-5
    exact = hunt.matched_slot_loss(off, torch.full((8, 5), 50.0), off, al)
    assert float(exact) < 1e-6
    assert hunt.perms(5, "cpu").shape == (120, 5)


def test_task_metrics_identify_by_nearest_live_centre():
    centers = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])
    batch = {"centers": centers, "alive": torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]),
             "target": torch.tensor([[1.0, 0.0]]), "target_index": torch.tensor([1])}
    right = hunt.task_metrics(torch.tensor([[0.9, 0.0]]), batch, "pick_target", 5, 10.0)
    assert right["identify_acc"] == 1.0 and right["loc_mae"] == pytest.approx(1.0)
    wrong = hunt.task_metrics(torch.tensor([[0.2, 0.0]]), batch, "pick_target", 5, 10.0)
    assert wrong["identify_acc"] == 0.0


@pytest.fixture(scope="module")
def datasets(tmp_path_factory):
    """Tiny least-mass and cluster-hunt datasets through the package command."""
    tmp = tmp_path_factory.mktemp("hunt_data")
    out = {}
    for variant in ("least_mass", "cluster_hunt"):
        path = tmp / f"{variant}.npz"
        collect.main(["--domain", "hunt", "--variant", variant, "--num_episodes", "12",
                      "--seed", "5", "--output_file", str(path)])
        out[variant] = path
    return out


ENCODER_FLAGS = {
    "st": ["--num_inds", "4", "--dim_hidden", "16", "--num_post_sab", "0"],
    "cgf": ["--num_cgf_features", "8", "--feature_mode", "K_grad"],
    "deepset": ["--dim_hidden", "16"],
    "pointnet": ["--dim_hidden", "16"],
}


@pytest.mark.parametrize("encoder", sorted(ENCODER_FLAGS))
def test_task_pretraining_round_trips_into_the_rl_extractor(encoder, datasets, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = pretrain.main(["--domain", "hunt", "--encoder", encoder, "--data_path", str(datasets["least_mass"]),
                            "--num_epochs", "2", "--batch_size", "64", "--device", "cpu",
                            "--output_root", str(tmp_path / "root"), *ENCODER_FLAGS[encoder]])
    out = capsys.readouterr().out
    assert result.rl_checkpoint.exists() and "identify_acc" in result.summary
    assert f"Verified: {encoders.get(encoder).extractor_class.__name__} encoder matches" in out
    assert list((tmp_path / "root").glob("hunt/least_mass/pretrain/*/task/*_seed0/run_status.json"))
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["task"] == "pick_target" and payload["config"]["arena_scale"] == 10.0
    assert payload["pretraining_run"]["objective"] == "task"
    # the checkpoint is read by run_records and accepted by the trainer's dry run
    found = run_records.latest_pretrain_checkpoint("hunt", "least_mass", encoder, "task", root=tmp_path / "root")
    assert Path(found) == result.rl_checkpoint.resolve()
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    train_mod.main(["--domain", "hunt", "--encoder", encoder, "--variant", "least_mass",
                    "--pretrained_path", str(found), "--frozen", "--output_root", str(tmp_path / "root"),
                    "--dry_run"] + (["--feature_mode", "K_grad"] if encoder == "cgf" else []))


def test_cluster_hunt_uses_the_matched_slot_objective(datasets, tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = pretrain.main(["--domain", "hunt", "--encoder", "deepset", "--data_path", str(datasets["cluster_hunt"]),
                            "--num_epochs", "1", "--batch_size", "64", "--device", "cpu", "--dim_hidden", "16",
                            "--output_root", str(tmp_path / "root")])
    assert "centre_mae" in result.summary and "alive_acc" in result.summary
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["task"] == "collect_all"
    assert payload["head_state_dict"]["4.weight"].shape[0] == 15       # 5 slots x (dx, dy, alive)


def test_task_objective_refusals(datasets, tmp_path, capsys):
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "hunt", "--encoder", "gaussian", "--data_path", str(datasets["least_mass"])])
    with pytest.raises(SystemExit):     # dataset from another variant
        pretrain.main(["--domain", "hunt", "--encoder", "st", "--variant", "most_var",
                       "--data_path", str(datasets["least_mass"]), "--output_root", str(tmp_path)])
    with pytest.raises(SystemExit):     # no dataset under the root
        pretrain.main(["--domain", "hunt", "--encoder", "st", "--variant", "least_mass",
                       "--output_root", str(tmp_path / "empty")])
    # the reconstruction control reads the same file (Chamfer needs --ignore_weights)
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "hunt", "--encoder", "st", "--objective", "reconstruction",
                       "--data_path", str(datasets["least_mass"]), "--loss_type", "chamfer",
                       "--num_epochs", "1", "--device", "cpu", "--output_root", str(tmp_path)])
