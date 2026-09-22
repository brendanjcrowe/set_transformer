"""Fresh-layout pretraining on the hunt task objective (2026-09-21;
``change_mds/fresh_layout_pretraining_2026-09-21.md`` section 1.5).

Pins, in the plan's order: (a) each variant's generator against the REAL ``env.reset()`` -- exact
invariants as assertions and a two-sample KS test on every continuous marginal, per variant per
cluster count; (b) the constants come off the live ``cfg``, so a changed registration changes the
generator's output; (c) ``--data_source file`` never reaches a generator and is reproducible;
(d) the ``fresh`` and ``mixed`` runs produce an RL-loadable checkpoint with its provenance, which
the trainer's dry run accepts; (e) the refusals.

(c) is also checked OUTSIDE this file, against a pre-change baseline: the three commands of
``runs/verification/baseline/chain.log`` were re-run after the change and their ``history.json``
files diffed byte for byte against the baseline's. That comparison needs the baseline run
directory, which is gitignored, so what is pinned here is the property a committed test can own.
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
pytest.importorskip("pdomains.hunt", reason="needs the pomdp-domains hunt-envs branch")

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from set_transformer.rl import collect, pretrain, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.domains import hunt  # noqa: E402
from set_transformer.rl.pretrain_objectives import task_head  # noqa: E402

#: Rows per side of the comparison in (a). 3,000 is the plan's number: enough that a KS statistic
#: of 0.05 is already significant, cheap enough to run per variant per k on CPU.
N_COMPARE = 3000
#: The KS threshold. Loose on purpose (the point is to catch a generator that samples a DIFFERENT
#: distribution, not to fail on sampling noise); the statistics are printed either way, so drift
#: shows up in the log long before it crosses this.
KS_MAX = 0.10


# --------------------------------------------------------------------------
# The env as the oracle: the same label extraction the collector performs
# --------------------------------------------------------------------------

def _env_rows(variant: str, k: int, n: int, seed0: int = 0) -> dict:
    """``n`` rows from the real env: ``reset(seed=...)`` per row, then ``_collect_snapshot_extras``'
    own arithmetic on the live env (kept here rather than called, because the collector's hook
    wants the wrapped belief env and its ``obs`` dict; the lines below are its body)."""
    record = hunt.resolve(variant)
    env = gym.make(record.env_id)
    u = env.unwrapped
    K = int(u.cfg.n_clusters)
    out = {key: [] for key in hunt.GENERATED_ARRAYS if key != "weights"}
    for i in range(n):
        u.set_n_active(k)
        obs, _ = env.reset(seed=seed0 + i)
        centers = np.zeros((K, 2), np.float32)
        alive = np.zeros(K, np.float32)
        counts = np.zeros(K, np.float32)
        sigmas = np.zeros(K, np.float32)
        if record.task == "collect_all":
            centers[:] = (u.centers - u.pos) / hunt.ARENA_SCALE
            alive[:] = u.alive.astype(np.float32)
            sigmas[:] = u.sigmas
            counts[:] = u._particle_counts()
            live = np.flatnonzero(u.alive)
            j = int(live[np.argmin(np.linalg.norm(u.centers[live] - u.pos, axis=-1))])
        else:
            kk = int(u.k)
            centers[:kk] = (u.centers - u.pos) / hunt.ARENA_SCALE
            alive[:kk] = 1.0
            counts[:kk] = u.counts
            sigmas[:kk] = u.sigmas
            j = int(u.target)
        out["agent"].append(np.asarray(obs["agent"], np.float32))
        out["particles"].append((u.particles - u.pos).astype(np.float32))
        out["centers"].append(centers)
        out["alive"].append(alive)
        out["counts"].append(counts)
        out["sigmas"].append(sigmas)
        out["target"].append(centers[j])
        out["target_index"].append(np.int64(j))
    env.close()
    return {key: np.asarray(value) for key, value in out.items()}


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (scipy is not a hard dependency of the suite)."""
    a, b = np.sort(np.asarray(a, float).ravel()), np.sort(np.asarray(b, float).ravel())
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / len(a)
                               - np.searchsorted(b, grid, "right") / len(b))))


def _marginals(rows: dict) -> dict:
    live = rows["alive"] > 0.5
    return {"agent_x": rows["agent"][:, 0], "agent_y": rows["agent"][:, 1],
            "centre_x": rows["centers"][live][:, 0], "centre_y": rows["centers"][live][:, 1],
            "sigma": rows["sigmas"][live],
            "cloud_spread": rows["particles"].std(1).mean(1)}


# --------------------------------------------------------------------------
# (a) the generator against the env, per variant per k
# --------------------------------------------------------------------------

@pytest.mark.parametrize("variant,k", [(v, k)
                                       for v in ("least_mass", "most_var")
                                       for k in (2, 3, 4, 5)]
                         + [("cluster_hunt", k) for k in (1, 2, 3, 4, 5)])
def test_generator_matches_the_env_per_variant_per_k(variant, k, capsys):
    cfg = hunt._env_config(variant)
    generated = hunt.resolve(variant).layout_generator(cfg, (k,), N_COMPARE,
                                                       np.random.default_rng(0))
    sampled = _env_rows(variant, k, N_COMPARE)
    K, N = int(cfg.n_clusters), int(cfg.n_particles)

    # -- exact invariants (the rules, not the distribution) --------------------------------
    live = generated["alive"] > 0.5
    assert generated["alive"].shape == (N_COMPARE, K)
    assert np.array_equal(generated["alive"], sampled["alive"]), "the alive PATTERN must match"
    assert (live.sum(1) == k).all()
    assert (generated["counts"].sum(1) == N).all()
    assert (generated["counts"][live] >= 1).all()
    assert np.allclose(generated["target"],
                       generated["centers"][np.arange(N_COMPARE), generated["target_index"]])
    assert np.allclose(generated["weights"], 1.0 / N)
    assert (generated["counts"][~live] == 0).all()
    # a dead slot's width: zero on pick_target (the env never drew it), the env's own draw on
    # collect_all (every cluster is sampled and labelled there) -- the same as the env's rows
    assert ((generated["sigmas"][~live] == 0).all()
            == (sampled["sigmas"][~live] == 0).all())
    # centre separation: the same rejection rule, and the generator may not fail more often than
    # the env's own 2000-try fallback does
    # collect_all draws ALL n_clusters centres (the dead ones too) and separates all of them;
    # pick_target draws and separates the k live ones
    drawn = slice(None) if hunt.resolve(variant).task == "collect_all" else slice(0, k)

    def _too_close(rows):
        centres = rows["centers"][:, drawn] * hunt.ARENA_SCALE
        if centres.shape[1] < 2:
            return 0
        d = np.linalg.norm(centres[:, :, None] - centres[:, None], axis=-1)
        d += 1e9 * np.eye(centres.shape[1])[None]
        return int((d.min(axis=(1, 2)) < cfg.min_sep - 1e-6).sum())
    assert _too_close(generated) <= _too_close(sampled)

    if hunt.resolve(variant).task == "pick_target":
        ordered = np.sort(np.where(live, generated["counts"], np.inf), 1)
        sigma_ordered = np.sort(np.where(live, generated["sigmas"], -np.inf), 1)
        if cfg.target_rule == "min_mass":
            assert (ordered[:, 0] >= cfg.n_min_lo).all() and (ordered[:, 0] <= cfg.n_min_hi).all()
            assert (ordered[:, 1] - ordered[:, 0] >= cfg.mass_margin).all()
            assert (generated["target_index"]
                    == np.where(live, generated["counts"], np.inf).argmin(1)).all()
        else:
            assert (sigma_ordered[:, -1] - sigma_ordered[:, -2] >= cfg.sigma_margin - 1e-6).all()
            assert (generated["target_index"]
                    == np.where(live, generated["sigmas"], -np.inf).argmax(1)).all()
            assert (ordered[:, k - 1] - ordered[:, 0] <= 1).all()          # counts as equal as possible
    else:
        # collect_all: equal counts over the live clusters, remainder to the FIRST live ones, and
        # the collector's nearest-live target
        share, remainder = divmod(N, k)
        expected = np.zeros(K, int)
        expected[:k] = share
        expected[:remainder] += 1
        assert (generated["counts"] == expected.astype(np.float32)).all()
        d = np.linalg.norm(generated["centers"], axis=-1)
        d[~live] = np.inf
        assert (generated["target_index"] == d.argmin(1)).all()

    # -- continuous marginals, printed either way ------------------------------------------
    gen_marginals, env_marginals = _marginals(generated), _marginals(sampled)
    with capsys.disabled():
        print(f"\n  [{variant} k={k}] KS generator vs env.reset():", end="")
        for name in gen_marginals:
            print(f" {name}={_ks(gen_marginals[name], env_marginals[name]):.4f}", end="")
        print("")
    for name in gen_marginals:
        statistic = _ks(gen_marginals[name], env_marginals[name])
        assert statistic < KS_MAX, f"{variant} k={k}: {name} KS={statistic:.4f}"


def test_the_default_k_draw_is_the_collectors():
    """The generator's cluster-count mix is the collector's, and `--k_choices` overrides it."""
    from argparse import Namespace
    for variant, expected in (("least_mass", (2, 3, 4, 5, 5, 5)),
                              ("cluster_hunt", (1, 2, 3, 4, 5, 5, 5))):
        task = hunt.resolve(variant).task
        assert hunt.N_ACTIVE_CHOICES[task] == expected
        assert task_head.resolve_k_choices(Namespace(k_choices=None), expected) == expected
    assert task_head.resolve_k_choices(Namespace(k_choices="5,5,2"), (1,)) == (5, 5, 2)
    assert task_head.resolve_k_choices(Namespace(), (2, 3)) == (2, 3)


# --------------------------------------------------------------------------
# (b) the constants come from cfg, not from literals
# --------------------------------------------------------------------------

def test_the_generator_follows_a_changed_registration(monkeypatch):
    """most_var's ``sigma_margin`` 0.2 -> 0.4 and ``min_sep`` 3.0 -> 6.0 in the REGISTRATION: the
    generator's widths and centres must follow. A generator with the constants written into it
    (both standalone scripts) passes every other test here and fails this one."""
    spec = gym.spec("pdomains-most-var-v0")
    before = hunt._env_config("most_var")
    assert before.sigma_margin == 0.2 and before.min_sep == 3.0

    generated = hunt.resolve("most_var").layout_generator(before, (4,), 400,
                                                          np.random.default_rng(0))
    ordered = np.sort(generated["sigmas"][:, :4], 1)
    assert (ordered[:, -1] - ordered[:, -2] < 0.4).any(), "0.2 margin: some gap is below 0.4"

    monkeypatch.setitem(spec.kwargs, "sigma_margin", 0.4)
    monkeypatch.setitem(spec.kwargs, "min_sep", 6.0)
    after = hunt._env_config("most_var")
    assert after.sigma_margin == 0.4 and after.min_sep == 6.0
    generated = hunt.resolve("most_var").layout_generator(after, (4,), 400,
                                                          np.random.default_rng(0))
    ordered = np.sort(generated["sigmas"][:, :4], 1)
    assert (ordered[:, -1] - ordered[:, -2] >= 0.4 - 1e-6).all()
    centres = generated["centers"][:, :4] * hunt.ARENA_SCALE
    d = np.linalg.norm(centres[:, :, None] - centres[:, None], axis=-1) + 1e9 * np.eye(4)[None]
    assert (d.min(axis=(1, 2)) >= 6.0 - 1e-6).all()


def test_the_env_config_is_recorded_in_full(tmp_path, monkeypatch):
    """Plan section 3: a run record carries the CONSTRUCTED config, so a dataclass default is
    visible and not only the registration's overrides (``env_kwargs``, kept beside it)."""
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    for variant, key, value in (("least_mass", "timeout_penalty", 40.0),
                                ("most_var", "target_rule", "max_var"),
                                ("cluster_hunt", "min_sep", 2.5)):
        train_mod.main(["--domain", "hunt", "--encoder", "gaussian", "--variant", variant,
                        "--device", "cpu", "--output_root", str(tmp_path), "--dry_run"])
        run = sorted((tmp_path / "hunt" / variant / "rl" / "gaussian").iterdir())[-1]
        config = json.loads((run / "run_config.json").read_text())
        assert config["env_config"][key] == value
        assert "env_kwargs" in config                      # the older record is still there
    # the collector writes it too
    out = tmp_path / "lm.npz"
    collect.main(["--domain", "hunt", "--variant", "least_mass", "--num_episodes", "1",
                  "--output_file", str(out)])
    with np.load(out, allow_pickle=True) as z:
        meta = json.loads(str(z["metadata"]))
    assert meta["env_config"]["timeout_penalty"] == 40.0 and meta["env_config"]["n_particles"] == 100


# --------------------------------------------------------------------------
# (c) --data_source file is what it was
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """A tiny most_var dataset through the package command (the pattern of test_hunt_pretrain)."""
    path = tmp_path_factory.mktemp("fresh_data") / "most_var.npz"
    collect.main(["--domain", "hunt", "--variant", "most_var", "--num_episodes", "12",
                  "--seed", "5", "--output_file", str(path)])
    return path


ST_SMALL = ["--num_inds", "4", "--dim_hidden", "16", "--num_post_sab", "0"]


def _pretrain(tmp_path, *flags, encoder="st"):
    return pretrain.main(["--domain", "hunt", "--encoder", encoder, "--device", "cpu",
                          "--batch_size", "64", "--output_root", str(tmp_path / "root"),
                          *ST_SMALL, *flags])


def test_the_file_path_never_reaches_a_generator_and_is_reproducible(tiny_dataset, tmp_path,
                                                                     monkeypatch):
    """The default path must not have changed. Both generators are replaced by one that raises, and
    the same command is run twice: it completes, and the two histories are equal to the digit."""
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})

    def _refuse(*args, **kwargs):
        raise AssertionError("--data_source file must not call a layout generator")

    monkeypatch.setattr(hunt, "generate_pick_target_layouts", _refuse)
    monkeypatch.setattr(hunt, "generate_collect_all_layouts", _refuse)
    for name, variant in hunt.VARIANTS.items():
        monkeypatch.setitem(hunt.VARIANTS, name,
                            __import__("dataclasses").replace(variant, layout_generator=_refuse))
    histories = []
    for run in ("a", "b"):
        result = _pretrain(tmp_path / run, "--data_path", str(tiny_dataset), "--num_epochs", "2")
        histories.append((Path(result.run_dir) / "history.json").read_text())
        payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
        assert "data" not in payload["config"], "a file-backed checkpoint gains no new key"
        assert "data" not in payload["pretraining_run"]
        # 2026-09-22 (plan 11.3 item 2): nor does its metrics.json; the recorded layout ends in `history`
        metrics = json.loads((Path(result.run_dir) / "metrics.json").read_text())
        assert "data" not in metrics and list(metrics)[-1] == "history"
    assert histories[0] == histories[1]
    assert json.loads(histories[0])[0].keys() == {"epoch", "train", "val"}


# --------------------------------------------------------------------------
# (d) fresh and mixed, end to end
# --------------------------------------------------------------------------

@pytest.mark.parametrize("encoder", ["st", "deepset"])
@pytest.mark.parametrize("variant", ["most_var", "least_mass", "cluster_hunt"])
def test_fresh_pretraining_round_trips_into_the_rl_extractor(encoder, variant, tmp_path,
                                                             monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    flags = ST_SMALL if encoder == "st" else ["--dim_hidden", "16"]
    result = pretrain.main(["--domain", "hunt", "--encoder", encoder, "--variant", variant,
                            "--data_source", "fresh", "--fresh_rows_per_epoch", "512",
                            "--fresh_val_rows", "256", "--num_epochs", "3", "--batch_size", "64",
                            "--device", "cpu", "--output_root", str(tmp_path / "root"), *flags])
    out = capsys.readouterr().out
    assert result.rl_checkpoint.exists()
    assert "Verified:" in out and "max|delta| = 0.0" in out
    assert "512 rows/epoch (8 steps at batch 64)" in out           # the rows/epoch line (plan 1.5)

    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    record = payload["pretraining_run"]["data"]
    assert record["data_source"] == "fresh" and record["val_select"] == "generated"
    assert record["fresh_rows_per_epoch"] == 512 and record["fresh_val_rows"] == 256
    assert record["fresh_seed"] == 0
    assert record["k_choices"] == list(hunt.N_ACTIVE_CHOICES[hunt.resolve(variant).task])
    assert record["generator_constants"] == hunt._env_config(variant).to_dict()
    assert payload["config"]["data"]["data_source"] == "fresh"
    # metrics.json says so too (a generated run only; a file run's has no such key, test (c))
    assert json.loads((Path(result.run_dir) / "metrics.json").read_text())["data"]["data_source"] == "fresh"
    # run_config.json carries the flags, and the trainer accepts the checkpoint
    config = json.loads((Path(result.run_dir) / "run_config.json").read_text())
    assert config["data_source"] == "fresh" and config["fresh_rows_per_epoch"] == 512
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    train_mod.main(["--domain", "hunt", "--encoder", encoder, "--variant", variant,
                    "--pretrained_path", str(result.rl_checkpoint), "--frozen",
                    "--output_root", str(tmp_path / "root"), "--dry_run", *flags])


def test_task_nearest_round_trips_on_fresh_cluster_hunt_layouts(tmp_path, monkeypatch, capsys):
    """2026-09-22 (plan 11.3 item 3): Cluster-Hunt's campaign objective shares the fresh-layout code
    with `task` and was never run on generated rows. Its loss reads the generated `centers` / `alive`
    (the nearest live centre is recomputed per batch), so the same round trip must hold."""
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = pretrain.main(["--domain", "hunt", "--encoder", "st", "--objective", "task_nearest",
                            "--variant", "cluster_hunt", "--data_source", "fresh",
                            "--fresh_rows_per_epoch", "512", "--fresh_val_rows", "256", "--num_epochs", "3",
                            "--batch_size", "64", "--device", "cpu", "--output_root", str(tmp_path / "root"),
                            *ST_SMALL])
    out = capsys.readouterr().out
    assert result.rl_checkpoint.exists() and "Verified:" in out and "max|delta| = 0.0" in out
    assert {"identify_acc", "loc_mae", "within_1.0"} <= set(result.summary)     # the pick-a-target metrics
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert payload["config"]["objective"] == "task_nearest" and payload["config"]["task"] == "nearest"
    assert payload["pretraining_run"]["data"]["data_source"] == "fresh"
    assert payload["pretraining_run"]["objective"] == "task_nearest"
    assert payload["head_state_dict"]["4.weight"].shape[0] == 2                 # one offset, not K slots
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    train_mod.main(["--domain", "hunt", "--encoder", "st", "--variant", "cluster_hunt",
                    "--pretrained_path", str(result.rl_checkpoint), "--frozen",
                    "--output_root", str(tmp_path / "root"), "--dry_run", *ST_SMALL])


def test_fresh_epoch_seed_honours_an_explicit_zero(tiny_dataset, tmp_path, monkeypatch):
    """2026-09-22 (plan 11.3 item 1): `fresh_seed or seed` read an explicit --fresh_seed 0 as "not
    given" and seeded the epoch stream from --seed, while the held-out set and the record used 0."""
    from argparse import Namespace
    assert task_head.fresh_epoch_seed(Namespace(fresh_seed=0, seed=3)) == 2       # the bug gave 5
    assert task_head.fresh_epoch_seed(Namespace(fresh_seed=None, seed=3)) == 5    # default: seed
    assert task_head.fresh_epoch_seed(Namespace(fresh_seed=7, seed=3)) == 9
    assert task_head.fresh_epoch_seed(Namespace(seed=4)) == 6                     # Ant-Tag: no flag at all
    # and the loop uses it: the stream handed to the first refresh_epoch is default_rng(0 + 2)'s
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    seen = []
    original = task_head.GeneratedTaskData.refresh_epoch

    def spy(self, rng):
        seen.append(rng.bit_generator.state)
        return original(self, rng)

    monkeypatch.setattr(task_head.GeneratedTaskData, "refresh_epoch", spy)
    result = _pretrain(tmp_path, "--variant", "most_var", "--data_source", "fresh", "--fresh_seed", "0",
                       "--seed", "3", "--fresh_rows_per_epoch", "128", "--fresh_val_rows", "64",
                       "--num_epochs", "1")
    assert seen[0] == np.random.default_rng(2).bit_generator.state
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert payload["pretraining_run"]["data"]["fresh_seed"] == 0


def test_fresh_rows_differ_every_epoch(tmp_path, monkeypatch):
    """The point of the whole change: epoch 1 does not see epoch 0's layouts."""
    cfg = hunt._env_config("most_var")
    data = task_head.GeneratedTaskData(
        hunt.generate_pick_target_layouts, cfg, variant="most_var", env_id="pdomains-most-var-v0",
        obs_key="agent", k_choices=(5,), rows_per_epoch=64, val_rows=32, seed=0,
        particle_scale=hunt.ARENA_SCALE, device=torch.device("cpu"),
        label_arrays=hunt.GENERATED_ARRAYS)
    rng = np.random.default_rng(7)
    data.refresh_epoch(rng)
    first = data.train["centers"].clone()
    held_out = data.val["centers"].clone()
    data.refresh_epoch(rng)
    assert not torch.equal(first, data.train["centers"])
    assert torch.equal(held_out, data.val["centers"]), "the held-out set is drawn ONCE"
    assert data.n_train == 64 and data.n_val == 32


def test_mixed_trains_on_the_file_and_the_generated_rows(tiny_dataset, tmp_path, monkeypatch,
                                                         capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    with np.load(tiny_dataset) as z:
        n = len(z["particles"])
    n_train = n - int(0.1 * n)
    result = _pretrain(tmp_path, "--variant", "most_var", "--data_source", "mixed",
                       "--data_path", str(tiny_dataset), "--fresh_rows_per_epoch", "256",
                       "--fresh_val_rows", "128", "--num_epochs", "2")
    out = capsys.readouterr().out
    assert f"train={n_train + 256:,}" in out                      # rows/epoch = file train + fresh
    assert "data_source=mixed" in out and "--val_select mean" in out
    payload = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    record = payload["pretraining_run"]["data"]
    assert record["data_source"] == "mixed" and record["file_rows_per_epoch"] == n_train
    assert record["val_select"] == "mean"
    # both held-out sets are reported, and both feed the selection loss
    assert "file_identify_acc" in result.summary and "identify_acc" in result.summary
    history = json.loads((Path(result.run_dir) / "history.json").read_text())
    assert {"val_file", "val_generated"} <= history[0].keys()
    assert history[0]["val"] == pytest.approx(
        0.5 * (history[0]["val_file"] + history[0]["val_generated"]))


def test_mixed_default_rows_per_epoch_is_the_files_own(tiny_dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    with np.load(tiny_dataset) as z:
        n = len(z["particles"])
    n_train = n - int(0.1 * n)
    _pretrain(tmp_path, "--variant", "most_var", "--data_source", "mixed",
              "--data_path", str(tiny_dataset), "--fresh_val_rows", "64", "--num_epochs", "1")
    out = capsys.readouterr().out
    assert f"the file's own {n_train:,} training rows" in out
    assert f"train={2 * n_train:,}" in out


# --------------------------------------------------------------------------
# (e) the refusals
# --------------------------------------------------------------------------

def test_fresh_refusals(tiny_dataset, tmp_path, monkeypatch):
    import dataclasses

    base = ["--domain", "hunt", "--encoder", "st", "--device", "cpu", "--num_epochs", "1",
            "--output_root", str(tmp_path / "root"), *ST_SMALL]
    with pytest.raises(SystemExit):        # no --variant: nothing says which env's rules to use
        pretrain.main([*base, "--data_source", "fresh"])
    with pytest.raises(SystemExit):        # a file AND fresh is `mixed`, spelled out
        pretrain.main([*base, "--variant", "most_var", "--data_source", "fresh",
                       "--data_path", str(tiny_dataset)])
    with pytest.raises(SystemExit):        # --val_select file needs a file
        pretrain.main([*base, "--variant", "most_var", "--data_source", "fresh",
                       "--val_select", "file"])
    with pytest.raises(SystemExit):
        pretrain.main([*base, "--variant", "most_var", "--data_source", "fresh",
                       "--fresh_rows_per_epoch", "0"])
    # a variant with no generator
    monkeypatch.setitem(hunt.VARIANTS, "most_var",
                        dataclasses.replace(hunt.VARIANTS["most_var"], layout_generator=None))
    with pytest.raises(SystemExit):
        pretrain.main([*base, "--variant", "most_var", "--data_source", "fresh"])


def test_ant_tags_task_objective_has_no_fresh_flags():
    """The group is offered per domain: Ant-Tag's variants have no generator, so its command line
    is the one it had (an unknown flag is an error there, not a silently ignored one)."""
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "ant_tag", "--encoder", "st", "--objective", "task",
                       "--variant", "smart", "--data_source", "fresh"])
