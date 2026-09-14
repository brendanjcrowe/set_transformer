"""Reconstruction pretraining of the ARMS' DeepSet and PointNet encoders (batch 10.11, 2026-09-14).

`PooledArmAutoencoder` (models/pooled_arm_ae.py) puts a PFDecoder behind the pooled RL extractor's
own `encode` and trains under the existing Trainer; the generic `export_arm_checkpoint`
(models/arm_export.py) writes what `rl/train.py --encoder deepset|pointnet --pretrained_path`
loads. The tests pin what would silently invalidate a pretrained arm -- the frame (the encoder must
see under pretraining exactly what it sees under PPO), the weights (mass in the measure and in the
weight channel), the export (strict keys + the loader's geometry check) -- then run the package
command for real on a tiny weighted dataset WITH latent alignment (the Trainer branch this batch
opened for the arm autoencoders) and hand the export to the trainer's frozen dry run.
"""

from __future__ import annotations

import json
import pickletools
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("stable_baselines3")

import gymnasium as gym  # noqa: E402

from set_transformer.models.arm_export import export_arm_checkpoint  # noqa: E402
from set_transformer.models.pooled_arm_ae import PooledArmAutoencoder, make_pooled_arm_extractor  # noqa: E402
from set_transformer.rl import encoders, precompute_emd, pretrain, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.pretrain_objectives import reconstruction  # noqa: E402
from set_transformer.training.config import TrainingConfig  # noqa: E402

B, N, D, SCALE = 4, 100, 2, 10.0
SMALL = dict(num_encodings=4, dim_encoder=4, dim_hidden=16)
POOLED = ["deepset", "pointnet"]


def _norm_batch(seed=0):
    """Loader-frame particles (already / SCALE) and PF weights with half the mass nearly dead."""
    g = torch.Generator().manual_seed(seed)
    x_raw = torch.randn(B, N, D, generator=g) * 3.0
    w = torch.rand(B, N, generator=g)
    w[:, N // 2:] *= 0.01
    w = w / w.sum(1, keepdim=True)
    return x_raw / SCALE, w, x_raw


def _trainer_input(x_norm, w):
    """What Trainer._model_input hands the model in weighted mode."""
    return torch.cat([x_norm, (w * N).unsqueeze(-1)], dim=-1)


def _ae(name, weighted=True, **kw):
    cls = encoders.get(name).extractor_class
    ext = make_pooled_arm_extractor(cls, N, D, arena_scale=SCALE, weight_channel=weighted, **{**SMALL, **kw})
    return PooledArmAutoencoder(ext, N, D, particle_scale=SCALE, dim_hidden=16, weighted=weighted)


def _obs(x_raw, w):
    return {"obs": torch.zeros(x_raw.shape[0], 1), "particles": x_raw, "weights": w}


def test_the_objective_pretrains_every_learned_encoder_of_the_table():
    learned = {n for n, e in encoders.ENCODERS.items() if e.learned}
    assert set(reconstruction.ENCODERS) == learned == {"st", "cgf", "deepset", "pointnet"}
    assert set(reconstruction.POOLED_ENCODERS) == {"deepset", "pointnet"}


@pytest.mark.parametrize("name", POOLED)
def test_shapes_and_code_geometry(name):
    torch.manual_seed(0)
    ae = _ae(name)
    x_norm, w, _ = _norm_batch()
    X = _trainer_input(x_norm, w)
    assert ae.encode(X).shape == (B, 4, 4)          # the extractor's own num_encodings x dim_encoder
    assert ae(X).shape == (B, N, D)


@pytest.mark.parametrize("name", POOLED)
def test_encoder_sees_the_ppo_frame(name):
    """Pretraining feeds loader-normalised particles and the mass channel; PPO feeds raw particles
    and probabilities. The extractor's code must be the same, or the encoder is trained in a
    frame the policy never uses."""
    torch.manual_seed(0)
    ae = _ae(name)
    x_norm, w, x_raw = _norm_batch()
    code = ae.encode(_trainer_input(x_norm, w))
    direct = ae.extractor.encode(_obs(x_raw, w))                 # the RL-time call
    assert torch.allclose(code, direct, atol=1e-5)


@pytest.mark.parametrize("name", POOLED)
def test_weights_reach_the_encoder(name):
    torch.manual_seed(0)
    ae = _ae(name)
    x_norm, w, _ = _norm_batch()
    uniform = torch.full_like(w, 1.0 / N)
    assert not torch.allclose(ae.encode(_trainer_input(x_norm, w)),
                              ae.encode(_trainer_input(x_norm, uniform)), atol=1e-3)


def test_unweighted_mode_feeds_uniform_weights_and_no_mass_channel():
    torch.manual_seed(0)
    ae = _ae("deepset", weighted=False)
    x_norm, _, x_raw = _norm_batch()
    assert ae.extractor.dim_input == D
    code = ae.encode(x_norm)
    uniform = torch.full((B, N), 1.0 / N)
    assert torch.allclose(code, ae.extractor.encode(_obs(x_raw, uniform)), atol=1e-5)


def test_mismatches_are_refused():
    cls = encoders.get("deepset").extractor_class
    with pytest.raises(ValueError, match="arena_scale"):
        PooledArmAutoencoder(make_pooled_arm_extractor(cls, N, D, arena_scale=1.0, **SMALL), N, D,
                             particle_scale=SCALE)
    with pytest.raises(ValueError, match="weight_channel"):
        PooledArmAutoencoder(make_pooled_arm_extractor(cls, N, D, arena_scale=SCALE, weight_channel=False,
                                                       **SMALL), N, D, particle_scale=SCALE, weighted=True)
    with pytest.raises(ValueError, match="-D particles"):
        PooledArmAutoencoder(make_pooled_arm_extractor(cls, N, 3, arena_scale=SCALE, **SMALL), N, D,
                             particle_scale=SCALE)


@pytest.mark.parametrize("name", POOLED)
def test_gradients_reach_the_encoder(name):
    torch.manual_seed(0)
    ae = _ae(name)
    x_norm, w, _ = _norm_batch()
    ae(_trainer_input(x_norm, w)).sum().backward()
    grads = [p.grad for p in ae.extractor.encoder.parameters()]
    assert all(g is not None for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0


@pytest.mark.parametrize("name", POOLED)
def test_export_loads_strict_into_the_rl_extractor(name, tmp_path):
    """The exported file is what --pretrained_path reads: it reproduces the encoder bit for bit,
    passes the geometry check, is pure Python + tensors, and another geometry is refused."""
    torch.manual_seed(0)
    ae = _ae(name)
    with torch.no_grad():
        next(ae.extractor.encoder.parameters()).add_(0.3)        # move the encoder off its init
    trainer_ckpt = tmp_path / "checkpoint_best.pt"
    torch.save({"model_state_dict": ae.state_dict(), "epoch": 3, "global_step": 99,
                "best_val_loss": 0.123, "alignment": {"lambda": 0.2},
                "config": TrainingConfig(model_type="pooled_arm_ae", num_particles=N, dim_particles=D)},
               trainer_ckpt)
    out = export_arm_checkpoint(trainer_ckpt, tmp_path / f"checkpoint_best_{name}_arm.pt", ae.extractor,
                                encoder_name=name, particle_centre=0.0,
                                objective="reconstruction_sinkhorn_aligned", data_path="x.npz",
                                extra_config={"weighted_pretraining": True})
    payload = torch.load(out, map_location="cpu", weights_only=False)
    assert set(payload["model_state_dict"]) == set(ae.extractor.checkpoint_state())
    assert all(k.startswith("encoder.") for k in payload["model_state_dict"])
    c = payload["config"]
    assert (c["encoder"], c["objective"], c["pretraining"]) == (name, "reconstruction_sinkhorn_aligned", "3_train_st.py")
    assert c["weighted_particles"] is True and c["weighted_pretraining"] is True and c["arena_scale"] == SCALE
    assert c["encoder_params"] == ae.extractor.encoder_parameter_count() and c["dim_hidden"] == 16
    assert payload["particle_scale"] == SCALE and payload["epoch"] == 3 and payload["alignment"] == {"lambda": 0.2}
    assert isinstance(payload["trainer_config"], dict) and payload["trainer_config"]["model_type"] == "pooled_arm_ae"
    with zipfile.ZipFile(out) as z:
        pkl = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    globals_used = {arg for op, arg, _ in pickletools.genops(pkl) if op.name in ("GLOBAL", "STACK_GLOBAL") and arg}
    assert not any("set_transformer" in str(g) for g in globals_used), globals_used

    cls = encoders.get(name).extractor_class
    fresh = make_pooled_arm_extractor(cls, N, D, arena_scale=SCALE, **SMALL, pretrained_model_path=str(out))
    _, w, x_raw = _norm_batch()
    fresh.eval(); ae.extractor.eval()
    assert torch.equal(fresh.encode(_obs(x_raw, w)), ae.extractor.encode(_obs(x_raw, w)))
    with pytest.raises(RuntimeError):
        make_pooled_arm_extractor(cls, N, D, arena_scale=SCALE, **{**SMALL, "dim_hidden": 32},
                                  pretrained_model_path=str(out))
    with pytest.raises(RuntimeError, match="weight_channel"):
        make_pooled_arm_extractor(cls, N, D, arena_scale=SCALE, weight_channel=False, **SMALL,
                                  pretrained_model_path=str(out))
    with pytest.raises(RuntimeError, match="arena_scale"):
        make_pooled_arm_extractor(cls, N, D, arena_scale=9.0, **SMALL, pretrained_model_path=str(out))


def test_a_checkpoint_of_another_geometry_is_refused_at_export(tmp_path):
    ae, other = _ae("deepset"), _ae("deepset", dim_hidden=32)
    trainer_ckpt = tmp_path / "checkpoint_best.pt"
    torch.save({"model_state_dict": other.state_dict(), "config": None}, trainer_ckpt)
    # same key set, other shapes: the strict load into the extractor's copy refuses it
    with pytest.raises(RuntimeError, match="size mismatch"):
        export_arm_checkpoint(trainer_ckpt, tmp_path / "out.pt", ae.extractor, encoder_name="deepset",
                              particle_centre=0.0, objective="reconstruction_sinkhorn", data_path="x.npz")


# ---------------------------------------------------------------------------
# The package command, for real, on a tiny weighted dataset with alignment
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> tuple[Path, Path]:
    """48 weighted 2-D clouds recorded as hunt `least_mass` (scale 10), plus their debiased-Sinkhorn
    matrix over the first 32 rows (the aligned condition's input)."""
    tmp = tmp_path_factory.mktemp("pooled_recon")
    rng = np.random.default_rng(0)
    S = 48
    centres = rng.uniform(-3, 3, size=(S, 1, D))
    particles = (centres + rng.normal(0, 0.7, size=(S, N, D))).astype(np.float32)
    weights = rng.random((S, N)).astype(np.float32)
    weights[:, N // 2:] *= 0.05
    weights /= weights.sum(1, keepdims=True)
    path = tmp / "least_mass_pf_dataset.npz"
    np.savez(path, particles=particles, weights=weights, particle_scale=np.float32(SCALE),
             metadata=np.array(json.dumps({"env_id": "pdomains-least-mass-v0", "variant": "least_mass",
                                           "particle_scale": SCALE, "particle_centre": 0.0})))
    matrix = precompute_emd.main(["--data_path", str(path), "--sinkhorn_blur", "0.02", "--max_samples", "32",
                                  "--device", "cpu"])
    return path, Path(matrix)


TINY = ["--objective", "reconstruction", "--loss_type", "sinkhorn",     # hunt's default objective is `task`
         "--sinkhorn_blur", "0.02", "--num_epochs", "2", "--batch_size", "8",
        "--num_workers", "0", "--eval_freq", "3", "--device", "cpu", "--dim_hidden", "16",
        "--num_encodings", "4", "--dim_encoder", "4"]


@pytest.mark.parametrize("name", POOLED)
def test_package_command_aligns_exports_round_trips_and_the_trainer_takes_it_frozen(
        name, dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.setenv("WANDB_MODE", "offline")
    data, matrix = dataset
    root = tmp_path / "root"
    result = pretrain.main(["--domain", "hunt", "--variant", "least_mass", "--encoder", name,
                            "--data_path", str(data), "--emd_matrix_path", str(matrix),
                            "--align_lambda", "0.2", "--align_warmup_epochs", "0", "--align_ramp_epochs", "1",
                            "--output_root", str(root), *TINY])
    out = capsys.readouterr().out
    cls = encoders.get(name).extractor_class.__name__
    assert f"Exported {name} arm encoder" in out and f"Verified: {cls} encoder matches" in out
    assert "Latent alignment: lambda=0.2" in out
    assert result.rl_checkpoint.name == f"checkpoint_best_{name}_arm.pt"
    assert result.run_dir.parent == root / "hunt" / "least_mass" / "pretrain" / name / "reconstruction"
    ck = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    c = ck["config"]
    assert (c["encoder"], c["objective"]) == (name, "reconstruction_sinkhorn_aligned")
    assert c["weighted_particles"] is True and c["weighted_pretraining"] is True and c["arena_scale"] == SCALE
    assert c["num_encodings"] == 4 and c["dim_encoder"] == 4 and c["dim_hidden"] == 16
    assert ck["particle_scale"] == SCALE and ck["alignment"]["lambda"] == 0.2 and "lambda_at_best_epoch" in ck["alignment"]
    assert ck["trainer_config"]["model_type"] == "pooled_arm_ae"
    record = ck[pretrain.CHECKPOINT_RECORD_KEY]
    assert record["encoder"] == name and record["geometry"]["weighted_particles"] is True
    status = json.loads((result.run_dir / "run_status.json").read_text())
    assert status["rl_checkpoint"] == str(result.rl_checkpoint.resolve())
    assert run_records.latest_pretrain_checkpoint("hunt", "least_mass", name, "reconstruction", root=root) == (
        result.rl_checkpoint.resolve())
    # the trainer's dry run takes the export frozen (the geometry flags spelled as at pretraining)
    train_mod.main(["--domain", "hunt", "--variant", "least_mass", "--encoder", name, "--dim_hidden", "16",
                    "--num_encodings", "4", "--dim_encoder", "4", "--pretrained_path", str(result.rl_checkpoint),
                    "--frozen", "--output_root", str(root), "--dry_run"])
    [rc] = list(root.glob(f"hunt/least_mass/rl/{name}/*_seed0/run_config.json"))
    config = json.loads(rc.read_text())
    assert str(result.rl_checkpoint) in config.values()
    assert any(k.endswith("frozen") and v is True for k, v in config.items())


def test_no_weight_channel_trains_the_unweighted_encoder(dataset, tmp_path, monkeypatch, capsys):
    """--no_weight_channel on a pooled arm means --ignore_weights (the ST's --no_st_weight_channel
    rule): the encoder reads D channels and the export says so, so the RL side must say it too."""
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.setenv("WANDB_MODE", "offline")
    data, _ = dataset
    root = tmp_path / "root"
    result = pretrain.main(["--domain", "hunt", "--variant", "least_mass", "--encoder", "deepset",
                            "--data_path", str(data), "--no_weight_channel", "--output_root", str(root), *TINY])
    out = capsys.readouterr().out
    ck = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert ck["config"]["weighted_particles"] is False and ck["config"]["weighted_pretraining"] is False
    assert ck["config"]["dim_input"] == D and "--no_weight_channel" in out
    cls = encoders.get("deepset").extractor_class
    with pytest.raises(RuntimeError, match="weight_channel"):
        make_pooled_arm_extractor(cls, N, D, arena_scale=SCALE, weight_channel=True, **SMALL,
                                  pretrained_model_path=str(result.rl_checkpoint))


def test_analytic_encoders_have_no_reconstruction_path(dataset, capsys):
    data, _ = dataset
    with pytest.raises(SystemExit) as exc:
        pretrain.main(["--domain", "hunt", "--variant", "least_mass", "--encoder", "kmoments",
                       "--data_path", str(data), *TINY])
    assert exc.value.code == 2 and "has no parameters to pretrain" in capsys.readouterr().err
