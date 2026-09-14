"""Policy-driven collection and dataset merging (plan 10.8b, 2026-09-14): the DAgger step on the harness.

`python -m set_transformer.rl.collect --behaviour policy --policy_path <agent.zip>` rolls the domain's
own belief env with a trained agent acting (the domain still resets and labels), and
`python -m set_transformer.rl.merge_datasets` concatenates harness datasets with a per-row source label.
Both are generic over domains by construction: the tests run them on hunt (labels) and Odd-Even (labels
since 10.5), after a handful of PPO steps produce an agent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("pdomains")

from set_transformer.rl import collect, merge_datasets, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402


def _agent(root, domain, variant, encoder, monkeypatch, **extra):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    argv = ["--domain", domain, "--variant", variant, "--encoder", encoder, "--total_timesteps", "64",
            "--n_envs", "1", "--ppo_n_steps", "32", "--batch_size", "32", "--n_epochs", "1",
            "--n_eval_episodes", "1", "--eval_freq", "100000", "--save_freq", "100000", "--device", "cpu",
            "--seed", "0", "--output_root", str(root), "--dim_hidden", "16", "--run_tag", "pol"]
    for k, v in extra.items():
        argv += [f"--{k}", str(v)]
    train_mod.main(argv)
    run = run_records.find_rl_run(domain, variant, encoder, 0, "pol", root=root)
    zip_path = run / "models" / f"{encoder}_agent.zip"
    assert zip_path.exists() and (run / "models" / "vecnormalize.pkl").exists()
    return zip_path


def test_hunt_policy_rollout_has_the_scripted_rows_labels_and_frame(tmp_path, monkeypatch, capsys):
    root = tmp_path / "root"
    agent = _agent(root, "hunt", "least_mass", "pointnet", monkeypatch)
    scripted = collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "3", "--seed", "1",
                             "--output_root", str(root), "--run_tag", "scripted"])
    policy = collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "3", "--seed", "1",
                           "--behaviour", "policy", "--policy_path", str(agent),
                           "--output_root", str(root), "--run_tag", "policy"])
    out = capsys.readouterr().out
    assert "Behaviour: policy" in out and "stochastic" in out and "normalised obs: yes" in out
    with np.load(scripted, allow_pickle=True) as a, np.load(policy, allow_pickle=True) as b:
        assert set(a.files) == set(b.files)
        for k in ("particles", "agent", "centers", "target", "target_index", "step", "alive"):
            assert a[k].shape[1:] == b[k].shape[1:] and a[k].dtype == b[k].dtype
        assert float(a["particle_scale"]) == float(b["particle_scale"]) == 10.0
        assert np.abs(b["particles"]).max() > 1.0 and np.abs(b["agent"]).max() <= 1.0     # raw cloud, /10 labels
        assert b["step"].min() == 0 and len(b["step"]) >= 3
        meta = json.loads(str(b["metadata"]))
    assert meta["behaviour"] == "policy" and meta["policy"]["policy_path"] == str(agent.resolve())
    assert meta["policy"]["vecnormalize_path"].endswith("vecnormalize.pkl") and meta["policy"]["deterministic"] is False
    assert json.loads(str(np.load(scripted, allow_pickle=True)["metadata"]))["behaviour"] == "scripted"
    # refusals: the agent's particle count, a policy path with the scripted behaviour, a missing zip
    with pytest.raises(SystemExit):
        collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "1", "--behaviour", "policy",
                      "--policy_path", str(agent), "--num_particles", "50", "--output_root", str(root)])
    assert "contradicts the agent" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "1",
                      "--policy_path", str(agent), "--output_root", str(root)])
    assert "belong to --behaviour policy" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "1", "--behaviour", "policy",
                      "--policy_path", str(tmp_path / "nowhere.zip"), "--output_root", str(root)])
    # deterministic on request
    det = collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "1", "--seed", "1",
                        "--behaviour", "policy", "--policy_path", str(agent), "--deterministic",
                        "--output_root", str(root), "--run_tag", "det"])
    assert json.loads(str(np.load(det, allow_pickle=True)["metadata"]))["policy"]["deterministic"] is True


def test_odd_even_policy_rollout_keeps_the_exact_posterior_and_its_labels(tmp_path, monkeypatch):
    root = tmp_path / "root"
    agent = _agent(root, "odd_even", "oe50_short", "deepset", monkeypatch)
    out = collect.main(["--domain", "odd_even", "--variant", "oe50_short", "--num_episodes", "2", "--seed", "3",
                        "--behaviour", "policy", "--policy_path", str(agent), "--no_rebalance",
                        "--output_root", str(root), "--run_tag", "policy"])
    with np.load(out, allow_pickle=True) as z:
        assert {"true_state", "optimal_prediction", "episode"} <= set(z.files)
        assert len(z["steps"]) == 2 * 31 and np.array_equal(z["weights"].argmax(1) + 1, z["optimal_prediction"])
        assert json.loads(str(z["metadata"]))["behaviour"] == "policy"


def _tiny(path, n, scale=10.0, variant="least_mass", with_round=None, episode_start=0, keys=("a", "b")):
    rng = np.random.default_rng(n)
    arrays = dict(particles=rng.normal(size=(n, 4, 2)).astype(np.float32), weights=np.full((n, 4), 0.25, np.float32),
                  step=np.arange(n, dtype=np.int32), episode=np.repeat(np.arange(episode_start, episode_start + n // 2), 2))
    for k in keys:
        arrays[k] = rng.normal(size=(n, 3)).astype(np.float32)
    if with_round is not None:
        arrays["source_round"] = np.full(n, with_round, np.int8)
    meta = {"variant": variant, "env_id": "pdomains-least-mass-v0", "particle_scale": scale, "label_arrays": list(keys)}
    np.savez(path, **arrays, particle_scale=np.float32(scale), metadata=json.dumps(meta))
    return path


def test_merge_concatenates_labels_rounds_and_episodes_and_refuses_mismatches(tmp_path, capsys):
    a = _tiny(tmp_path / "a.npz", 6, with_round=0)
    b = _tiny(tmp_path / "b.npz", 4)                       # no source_round: the next UNUSED number (c holds 1) = 2
    c = _tiny(tmp_path / "c.npz", 2, with_round=1)         # keeps its own 1
    out = merge_datasets.main([str(a), str(b), str(c), "--output_file", str(tmp_path / "m.npz"), "--no_shuffle"])
    assert "Merged 3 datasets" in capsys.readouterr().out
    with np.load(out, allow_pickle=True) as z:
        assert len(z["particles"]) == 12 and float(z["particle_scale"]) == 10.0
        assert z["source_round"].tolist() == [0] * 6 + [2] * 4 + [1] * 2
        assert z["source_file"].tolist() == [0] * 6 + [1] * 4 + [2] * 2
        # episode ids offset per input so they stay unique and contiguous pairs stay together
        ep = z["episode"].tolist()
        assert ep == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        meta = json.loads(str(z["metadata"]))
    assert meta["n_samples"] == 12 and meta["merged"] is True and [i["rows"] for i in meta["inputs"]] == [6, 4, 2]
    assert meta["row_order"] == "input order"
    # the default shuffles UNITS (episodes stay contiguous), so a tail split is not the last input
    shuffled = merge_datasets.main([str(a), str(b), str(c), "--output_file", str(tmp_path / "s.npz"), "--seed", "3"])
    with np.load(shuffled, allow_pickle=True) as z:
        ep, src = z["episode"].tolist(), z["source_round"].tolist()
        assert sorted(ep) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5] and ep != [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        assert all(ep[i] == ep[i + 1] for i in range(0, 12, 2))            # pairs intact
        assert sorted(src) == sorted([0] * 6 + [2] * 4 + [1] * 2) and len(set(src[-4:])) > 1
        assert "shuffled with seed 3" in json.loads(str(z["metadata"]))["row_order"]
    # without an `episode` array the units come from the step resets
    arrays = {"particles": np.zeros((5, 4, 2), np.float32), "weights": np.full((5, 4), 0.25, np.float32),
              "step": np.array([0, 1, 2, 0, 1], np.int32)}
    assert merge_datasets.unit_ids(arrays).tolist() == [0, 0, 0, 1, 1]
    assert merge_datasets.unit_ids({"particles": np.zeros((3, 4, 2))}).tolist() == [0, 1, 2]
    assert meta["inputs"][1]["source_rounds"] == [2] and "source_round" in meta["label_arrays"]
    assert meta["variant"] == "least_mass" and all(len(i["sha256"]) == 64 for i in meta["inputs"])
    # the default place under the root
    out2 = merge_datasets.main([str(a), str(b), "--domain", "hunt", "--variant", "least_mass", "--run_tag", "dagger1",
                                "--output_root", str(tmp_path / "root")])
    assert out2 == run_records.dataset_path("hunt", "least_mass", tag="dagger1", root=tmp_path / "root")
    # refusals
    with pytest.raises(ValueError, match="particle_scale"):
        merge_datasets.merge([a, _tiny(tmp_path / "s.npz", 2, scale=1.0)])
    with pytest.raises(ValueError, match="arrays"):
        merge_datasets.merge([a, _tiny(tmp_path / "k.npz", 2, keys=("a",))])
    with pytest.raises(ValueError, match="variant"):
        merge_datasets.merge([a, _tiny(tmp_path / "v.npz", 2, variant="most_var")])
    with pytest.raises(SystemExit):
        merge_datasets.main([str(a), str(b), "--output_file", str(a)])
    with pytest.raises(ValueError, match="at least two"):
        merge_datasets.merge([a])
