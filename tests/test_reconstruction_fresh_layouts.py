"""Fresh layouts on the generic RECONSTRUCTION objective (2026-09-22;
``change_mds/fresh_layout_reconstruction_2026-09-22.md``; plan
``change_mds/fresh_layout_reconstruction_plan_2026-09-21.md`` section 6).

Pins, in the plan's order: (a) the dataset's rows are the generator's particles in the collector's
frame with uniform weights; (b) the training block is regenerated once per epoch, before the first
item, and the held-out block never moves -- through a bare DataLoader and through the Trainer;
(c) ``--data_source file`` never reaches a generator and is bit-reproducible, and the four other
domains' reconstruction command lines have no such flags; (d) ``fresh`` and ``mixed`` runs round-trip
into the RL extractor for every learned encoder, with their provenance, and the file half of a mixed
run is the plain run's split; (e) online alignment on fresh rows reads the held-out block only;
(f) the refusals; (g) a file run's checkpoint record is unchanged.

The generators themselves are checked against ``env.reset()`` in ``test_hunt_fresh_layouts.py``;
this file consumes their ``particles`` / ``weights`` only.
"""
from __future__ import annotations

import dataclasses
import sys
from argparse import Namespace
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

from set_transformer.data.dataset import POMDPDataset, get_data_loader, get_dataset  # noqa: E402
from set_transformer.rl import collect, pretrain, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl import domains as domains_pkg  # noqa: E402
from set_transformer.rl import encoders as rl_encoders  # noqa: E402
from set_transformer.rl.domains import hunt  # noqa: E402
from set_transformer.rl.pretrain_objectives import fresh_layouts, reconstruction, task_head  # noqa: E402
from set_transformer.training.trainer import Trainer  # noqa: E402

ST_SMALL = ["--num_inds", "4", "--dim_hidden", "16", "--num_post_sab", "0"]
#: The RL door's geometry flags per arm (the ST-only spellings are unknown to the pooled and CGF parsers).
RL_GEOMETRY = {"st": ST_SMALL, "deepset": ["--dim_hidden", "16"], "pointnet": ["--dim_hidden", "16"], "cgf": []}
TINY = ["--fresh_rows_per_epoch", "128", "--fresh_val_rows", "64", "--eval_freq", "2", "--save_freq", "100000"]


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    """A tiny most_var dataset through the package command (the pattern of test_hunt_fresh_layouts)."""
    path = tmp_path_factory.mktemp("recon_fresh_data") / "most_var.npz"
    collect.main(["--domain", "hunt", "--variant", "most_var", "--num_episodes", "12",
                  "--seed", "5", "--output_file", str(path)])
    return path


def _pretrain(tmp_path, *flags, encoder="st"):
    return pretrain.main(["--domain", "hunt", "--encoder", encoder, "--objective", "reconstruction",
                          "--device", "cpu", "--batch_size", "64", "--sinkhorn_blur", "0.02",
                          "--output_root", str(tmp_path / "root"), *ST_SMALL, *flags])


def _spec(variant="most_var"):
    spec = hunt.HUNT.fresh_layouts(variant)
    assert spec is not None
    return spec


def _payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


# --------------------------------------------------------------------------
# (a) the dataset
# --------------------------------------------------------------------------

def test_dataset_rows_are_the_generators_particles_in_the_collectors_frame():
    spec = _spec()
    base = fresh_layouts.GeneratedParticleDataset(
        spec.generator, spec.k_choices, n_val=32, n_train=48, fresh_seed=3,
        particle_scale=spec.particle_scale, particle_centre=spec.particle_centre, weighted=True)
    assert isinstance(base, POMDPDataset) and len(base) == 80
    assert base.particle_scale == 10.0 and base.particle_centre == 0.0
    assert base.num_particles == 100 and base.particle_dim == 2 and base.is_weighted
    held_out = spec.generator(spec.k_choices, 32, np.random.default_rng(3 + 1))
    expected = torch.from_numpy(held_out["particles"]).float() / 10.0
    assert torch.equal(base.data[:32], expected), "held-out block = generator(seed + 1) / scale"
    assert torch.allclose(base.weights, torch.full((80, 100), 0.01))
    first = spec.generator(spec.k_choices, 48, np.random.default_rng(3 + 2))
    assert torch.equal(base.data[32:], torch.from_numpy(first["particles"]).float() / 10.0), \
        "the first training block is the epoch stream's (seed + 2) first draw"
    assert list(base.val_positions) == list(range(0, 32))
    assert list(base.train_positions) == list(range(32, 80))
    particles, weights = base[5]
    assert particles.shape == (100, 2) and weights.shape == (100,)


def test_unweighted_dataset_returns_bare_tensors():
    spec = _spec("least_mass")
    base = fresh_layouts.GeneratedParticleDataset(
        spec.generator, spec.k_choices, n_val=8, n_train=8, fresh_seed=0,
        particle_scale=10.0, particle_centre=0.0, weighted=False)
    assert not base.is_weighted and base.weights is None
    assert isinstance(base[0], torch.Tensor)
    base.refresh_training_rows()
    assert base.refreshes == 1


# --------------------------------------------------------------------------
# (b) the per-epoch hook
# --------------------------------------------------------------------------

def test_the_sampler_refreshes_once_per_epoch_before_the_first_item_and_keeps_the_held_out_block():
    spec = _spec()
    train_loader, val_loader, n_train, n_val, base = fresh_layouts.build_generated_loaders(
        generator=spec.generator, k_choices=spec.k_choices, source="fresh", val_select="generated",
        rows_per_epoch=96, val_rows=40, fresh_seed=0, seed=0, particle_scale=10.0,
        particle_centre=0.0, weighted=True, device="cpu", batch_size=32)
    assert (n_train, n_val) == (96, 40) and len(base) == 136
    assert list(val_loader.dataset.indices) == list(range(0, 40)), "val indices are BASE rows of the held-out block"
    held_out = base.data[:40].clone()
    first_batches = []
    for epoch in range(3):
        for i, (particles, weights) in enumerate(train_loader):
            if i == 0:
                # the refresh happened when the loader asked its sampler for an iterator,
                # i.e. before this first batch's __getitem__ calls
                assert base.refreshes == epoch + 1
                first_batches.append(particles.clone())
        assert base.refreshes == epoch + 1, "exactly one refresh per epoch"
    assert not torch.equal(first_batches[0], first_batches[1])
    assert not torch.equal(first_batches[1], first_batches[2])
    assert torch.equal(base.data[:40], held_out), "the held-out block is drawn ONCE"
    # the validation loader sees the same rows every time
    a = torch.cat([p for p, _ in val_loader]); b = torch.cat([p for p, _ in val_loader])
    assert torch.equal(a, b) and torch.equal(a, held_out)


def test_the_trainer_gets_a_new_training_block_every_epoch(tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    calls = []
    original = fresh_layouts.GeneratedParticleDataset.refresh_training_rows

    def spy(self, rng=None):
        calls.append(self.train_positions.start)
        return original(self, rng)

    monkeypatch.setattr(fresh_layouts.GeneratedParticleDataset, "refresh_training_rows", spy)
    result = _pretrain(tmp_path, "--variant", "most_var", "--data_source", "fresh", *TINY, "--num_epochs", "3")
    assert len(calls) == 3, "one refresh per epoch (the constructor's first block is drawn directly)"
    assert result.rl_checkpoint is not None


# --------------------------------------------------------------------------
# (c) + (g) the file path is untouched
# --------------------------------------------------------------------------

def test_the_file_path_never_reaches_a_generator_and_is_bit_reproducible(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})

    def _refuse(*args, **kwargs):
        raise AssertionError("--data_source file must not build generated loaders")

    monkeypatch.setattr(reconstruction, "_generated_loaders", _refuse)
    monkeypatch.setattr(fresh_layouts, "build_generated_loaders", _refuse)
    for name, variant in hunt.VARIANTS.items():
        monkeypatch.setitem(hunt.VARIANTS, name, dataclasses.replace(variant, layout_generator=_refuse))
    states = []
    for run in ("a", "b"):
        result = _pretrain(tmp_path / run, "--data_path", str(tiny_dataset), "--num_epochs", "2",
                           "--eval_freq", "3", "--save_freq", "100000")
        payload = _payload(result.rl_checkpoint)
        assert "data" not in payload["pretraining_run"], "(g) a file run's record keeps exactly its keys"
        states.append(payload["model_state_dict"])
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key


@pytest.mark.parametrize("domain_name", ["ant_tag", "odd_even", "car_flag", "msearch"])
def test_the_other_domains_reconstruction_command_lines_have_no_fresh_flags(domain_name):
    domain = domains_pkg.get(domain_name)
    assert getattr(domain, "fresh_layouts", None) is None
    parser = pretrain.build_parser(domain, rl_encoders.get("st"), reconstruction.OBJECTIVE, selectors=True)
    for flag in ("--data_source", "--fresh_rows_per_epoch", "--fresh_val_rows", "--fresh_seed",
                 "--k_choices", "--val_select"):
        assert flag not in parser._option_string_actions, f"{domain_name}: {flag}"
    hunt_parser = pretrain.build_parser(hunt.HUNT, rl_encoders.get("st"), reconstruction.OBJECTIVE, selectors=True)
    assert "--data_source" in hunt_parser._option_string_actions


def test_the_task_door_shares_the_group_and_reexports_the_helpers():
    assert task_head.DATA_SOURCES is fresh_layouts.DATA_SOURCES
    assert task_head.DEFAULT_FRESH_ROWS_PER_EPOCH == fresh_layouts.DEFAULT_FRESH_ROWS_PER_EPOCH == 100_000
    assert task_head._resolve_fresh_arguments is fresh_layouts.resolve_fresh_arguments
    assert task_head.resolve_k_choices is fresh_layouts.resolve_k_choices
    assert task_head.fresh_epoch_seed is fresh_layouts.fresh_epoch_seed
    assert task_head.data_source_of(Namespace()) == "file"
    task_parser = pretrain.build_parser(hunt.HUNT, rl_encoders.get("st"), hunt.TASK_OBJECTIVE, selectors=True)
    recon_parser = pretrain.build_parser(hunt.HUNT, rl_encoders.get("st"), reconstruction.OBJECTIVE, selectors=True)
    for flag in ("--data_source", "--fresh_rows_per_epoch", "--fresh_val_rows", "--fresh_seed", "--k_choices", "--val_select"):
        a, b = task_parser._option_string_actions[flag], recon_parser._option_string_actions[flag]
        assert (a.choices, a.default, a.type) == (b.choices, b.default, b.type), flag


# --------------------------------------------------------------------------
# (d) round trips and provenance
# --------------------------------------------------------------------------

@pytest.mark.parametrize("encoder,variant", [("st", "most_var"), ("deepset", "most_var"),
                                             ("pointnet", "cluster_hunt"), ("cgf", "least_mass")])
def test_fresh_pretraining_round_trips_into_the_rl_extractor(encoder, variant, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = _pretrain(tmp_path, "--variant", variant, "--data_source", "fresh", *TINY, "--num_epochs", "2",
                       encoder=encoder)
    out = capsys.readouterr().out
    assert "Fresh layouts (fresh): variant " + variant in out and "max|delta| = 0.0" in out
    assert Path(result.rl_checkpoint).exists()
    payload = _payload(result.rl_checkpoint)
    record = payload["pretraining_run"]
    assert record["data_path"] is None and record["arena_scale"] == 10.0
    data = record["data"]
    assert data["data_source"] == "fresh" and data["val_select"] == "generated"
    assert data["fresh_rows_per_epoch"] == 128 and data["fresh_val_rows"] == 64 and data["fresh_seed"] == 0
    assert data["k_choices"] == list(hunt.N_ACTIVE_CHOICES[hunt.resolve(variant).task])
    assert data["generator_constants"] == hunt._env_config(variant).to_dict()
    assert data["env_id"] == hunt.resolve(variant).env_id
    assert (data["particle_scale"], data["particle_centre"]) == (10.0, 0.0)
    # the RL side accepts the file
    assert train_mod.main(["--domain", "hunt", "--encoder", encoder, "--variant", variant, "--device", "cpu",
                           "--output_root", str(tmp_path / "root"), "--pretrained_path", str(result.rl_checkpoint),
                           "--frozen", "--dry_run", *RL_GEOMETRY[encoder]]) is None


def test_arm_export_records_the_generated_source_as_its_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = _pretrain(tmp_path, "--variant", "most_var", "--data_source", "fresh", *TINY, "--num_epochs", "1",
                       encoder="deepset")
    payload = _payload(result.rl_checkpoint)

    def values(node):
        if isinstance(node, dict):
            for v in node.values():
                yield from values(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                yield from values(v)
        else:
            yield node

    strings = {v for v in values({k: v for k, v in payload.items() if k != "model_state_dict"}) if isinstance(v, str)}
    assert "generated:most_var" in strings, "the arm export's dataset field names the generated source, not None"
    assert "None" not in strings


def test_mixed_trains_on_the_file_and_the_generated_rows_and_keeps_the_files_split(tiny_dataset, tmp_path,
                                                                                    monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    with np.load(tiny_dataset) as z:
        n = len(z["particles"])
    n_file_train = int(0.8 * n)
    result = _pretrain(tmp_path, "--variant", "most_var", "--data_source", "mixed", "--data_path", str(tiny_dataset),
                       *TINY, "--num_epochs", "2")
    out = capsys.readouterr().out
    assert f"128 generated training rows per epoch + the file's {n_file_train:,} = {n_file_train + 128:,} rows" in out
    assert "--val_select mean" in out
    data = _payload(result.rl_checkpoint)["pretraining_run"]["data"]
    assert data["data_source"] == "mixed" and data["val_select"] == "mean"
    assert data["file_rows_per_epoch"] == n_file_train and data["file_val_rows"] == n - n_file_train
    assert data["file"] == str(tiny_dataset)
    # the file half of a mixed run is the plain run's split, row for row
    plain_train, plain_val, _, _ = get_data_loader(batch_size=64, data_path=str(tiny_dataset), device="cpu",
                                                   train_split=0.8, load_weights=True, seed=0)
    file_dataset = get_dataset(str(tiny_dataset), "cpu", load_weights=True)
    spec = _spec()
    for select, expected_val in (("mean", None), ("file", None), ("generated", None)):
        train_loader, val_loader, _, _, base = fresh_layouts.build_generated_loaders(
            generator=spec.generator, k_choices=spec.k_choices, source="mixed", val_select=select,
            rows_per_epoch=16, val_rows=8, fresh_seed=0, seed=0, particle_scale=file_dataset.particle_scale,
            particle_centre=file_dataset.particle_centre, weighted=True, device="cpu", batch_size=64,
            file_dataset=file_dataset, train_split=0.8)
        assert base.n_file == n
        assert list(train_loader.dataset.indices) == list(plain_train.dataset.indices) + list(range(n + 8, n + 24))
        file_val = list(plain_val.dataset.indices)
        generated_val = list(range(n, n + 8))
        expected = {"mean": file_val + generated_val, "file": file_val, "generated": generated_val}[select]
        assert list(val_loader.dataset.indices) == expected, select
        assert torch.equal(base.data[:n], file_dataset.data), "the file rows are the plain run's tensors"


def test_mixed_default_rows_per_epoch_is_the_files_own(tiny_dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    with np.load(tiny_dataset) as z:
        n = len(z["particles"])
    result = _pretrain(tmp_path, "--variant", "most_var", "--data_source", "mixed", "--data_path", str(tiny_dataset),
                       "--fresh_val_rows", "32", "--eval_freq", "4", "--save_freq", "100000", "--num_epochs", "1")
    out = capsys.readouterr().out
    assert f"--fresh_rows_per_epoch not given; the file's own {int(0.8 * n):,} training rows" in out
    assert _payload(result.rl_checkpoint)["pretraining_run"]["data"]["fresh_rows_per_epoch"] == int(0.8 * n)


# --------------------------------------------------------------------------
# (e) online alignment on fresh rows
# --------------------------------------------------------------------------

def test_online_alignment_on_fresh_rows_reads_the_held_out_block_only(tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    seen, values = [], []
    original_rows, original_r = Trainer._rows_tensors, Trainer._validation_alignment_r

    def rows_spy(self, rows):
        seen.extend(int(i) for i in rows)
        return original_rows(self, rows)

    def r_spy(self):
        values.append(original_r(self))
        return values[-1]

    monkeypatch.setattr(Trainer, "_rows_tensors", rows_spy)
    monkeypatch.setattr(Trainer, "_validation_alignment_r", r_spy)
    result = _pretrain(tmp_path, "--variant", "most_var", "--data_source", "fresh", *TINY, "--num_epochs", "2",
                       "--align_lambda", "0.05", "--align_target", "online", "--align_val_pairs", "100")
    assert seen and max(seen) < 64 and min(seen) >= 0, "every row read for val/align_r lies in the held-out block"
    assert values and all(np.isfinite(v) for v in values)
    payload = _payload(result.rl_checkpoint)
    assert payload["alignment"]["target"] == "online"


# --------------------------------------------------------------------------
# (f) refusals
# --------------------------------------------------------------------------

def test_fresh_refusals(tiny_dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    cases = {
        "data_path under fresh": ["--variant", "most_var", "--data_source", "fresh", "--data_path", str(tiny_dataset)],
        "matrix alignment under fresh": ["--variant", "most_var", "--data_source", "fresh", "--align_lambda", "0.1"],
        "matrix path under mixed": ["--variant", "most_var", "--data_source", "mixed", "--data_path", str(tiny_dataset),
                                    "--emd_matrix_path", str(tmp_path / "none.npy")],
        "max_samples under fresh": ["--variant", "most_var", "--data_source", "fresh", "--max_samples", "10"],
        "workers under fresh": ["--variant", "most_var", "--data_source", "fresh", "--num_workers", "1"],
        "workers under mixed": ["--variant", "most_var", "--data_source", "mixed", "--data_path", str(tiny_dataset),
                                "--num_workers", "2"],
        "no variant under fresh": ["--data_source", "fresh"],
        "val_select file under fresh": ["--variant", "most_var", "--data_source", "fresh", "--val_select", "file"],
        "zero rows": ["--variant", "most_var", "--data_source", "fresh", "--fresh_rows_per_epoch", "0"],
        "mixed without a file under the root": ["--variant", "most_var", "--data_source", "mixed"],
    }
    for label, flags in cases.items():
        with pytest.raises(SystemExit):
            _pretrain(tmp_path / label.replace(" ", "_"), *TINY, "--num_epochs", "1", *flags)   # the case LAST: argparse keeps the last value
    # a domain without the group: the flag itself is unrecognized
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "ant_tag", "--encoder", "st", "--objective", "reconstruction", "--device", "cpu",
                       "--variant", "smart", "--data_source", "fresh", "--output_root", str(tmp_path / "at")])


def test_a_variant_without_a_generator_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    variant = hunt.VARIANTS["most_var"]
    monkeypatch.setitem(hunt.VARIANTS, "most_var", dataclasses.replace(variant, layout_generator=None))
    assert hunt.HUNT.fresh_layouts("most_var") is None
    with pytest.raises(SystemExit):
        _pretrain(tmp_path, "--variant", "most_var", "--data_source", "fresh", *TINY, "--num_epochs", "1")


def test_mixed_refuses_a_file_in_another_frame(tiny_dataset):
    spec = _spec()
    file_dataset = get_dataset(str(tiny_dataset), "cpu", load_weights=True, particle_scale=5.0)
    with pytest.raises(ValueError, match="particle_scale"):
        fresh_layouts.GeneratedParticleDataset(
            spec.generator, spec.k_choices, n_val=4, n_train=4, fresh_seed=0, particle_scale=10.0,
            particle_centre=0.0, weighted=True, file_dataset=file_dataset)
