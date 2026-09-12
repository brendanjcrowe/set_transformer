"""Reconstruction pretraining of the ARM's CGF (3_train_st.py --encoder cgf).

`CGFArmAutoencoder` puts a PFDecoder behind `WeightedCGFFeaturesExtractor.cgf_block` and
trains under the existing Trainer; `export_arm_checkpoint` writes what
`4_train_rl_cgf.py --pretrained_cgf_model_path` loads. The tests pin the three things
that would silently invalidate a pretrained arm: the frame (the block must see under
pretraining exactly what it sees under PPO), the weights (mass in the measure, not
uniform), and the export (strict keys + the loader's geometry check), then run the real
script end to end on a tiny weighted dataset.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from set_transformer.training.config import TrainingConfig
from set_transformer.models.cgf_arm_ae import (
    CGFArmAutoencoder,
    export_arm_checkpoint,
    make_arm_extractor,
)
from set_transformer.rl.feature_extractors.cgf import WeightedCGFFeaturesExtractor

B, N, D, SCALE = 4, 100, 2, 4.5
POLAR = dict(num_cgf_features=64, t_init_mode="spread", t_param="polar", t_bound=13.5,
             t_init_max=0.8 * 13.5, feature_mode="K_grad", feature_norm="none")


def _norm_batch(seed=0):
    """Loader-frame particles (already / SCALE) and PF weights, ESS ~ 50."""
    g = torch.Generator().manual_seed(seed)
    x_raw = torch.randn(B, N, D, generator=g) * 3.0
    w = torch.rand(B, N, generator=g)
    w[:, N // 2:] *= 0.01
    w = w / w.sum(1, keepdim=True)
    return x_raw / SCALE, w, x_raw


def _trainer_input(x_norm, w):
    """What Trainer._model_input hands the model in weighted mode."""
    return torch.cat([x_norm, (w * N).unsqueeze(-1)], dim=-1)


def _ae(**kw):
    ext = make_arm_extractor(N, D, arena_scale=SCALE, **{**POLAR, **kw})
    return CGFArmAutoencoder(ext, N, D, particle_scale=SCALE, num_encodings=8,
                             dim_hidden=32, weighted=True)


def test_shapes_and_code_geometry():
    torch.manual_seed(0)
    ae = _ae()
    x_norm, w, _ = _norm_batch()
    X = _trainer_input(x_norm, w)
    assert ae.encode(X).shape == (B, 8, 128 // 8)       # K_grad in 2-D: 64 * 2 wide
    assert ae(X).shape == (B, N, D)


def test_block_sees_the_ppo_frame():
    """Pretraining feeds loader-normalised particles; PPO feeds raw ones. The block's
    output must be identical, or t is learned in a frame the policy never uses."""
    torch.manual_seed(0)
    ae = _ae()
    x_norm, w, x_raw = _norm_batch()
    code = ae.encode(_trainer_input(x_norm, w)).reshape(B, -1)
    direct = ae.extractor.cgf_block(x_raw, w)             # the RL-time call
    assert torch.allclose(code, direct, atol=1e-5)


def test_weights_are_in_the_measure():
    torch.manual_seed(0)
    ae = _ae()
    x_norm, w, _ = _norm_batch()
    uniform = torch.full_like(w, 1.0 / N)
    assert not torch.allclose(ae.encode(_trainer_input(x_norm, w)),
                              ae.encode(_trainer_input(x_norm, uniform)), atol=1e-3)


def test_frame_mismatch_is_refused():
    ext = make_arm_extractor(N, D, arena_scale=1.0, **POLAR)
    with pytest.raises(ValueError, match="arena_scale"):
        CGFArmAutoencoder(ext, N, D, particle_scale=SCALE)


def test_gradients_reach_the_probes():
    torch.manual_seed(0)
    ae = _ae()
    x_norm, w, _ = _norm_batch()
    ae(_trainer_input(x_norm, w)).sum().backward()
    assert ae.extractor.raw_a.grad is not None and ae.extractor.raw_a.grad.abs().sum() > 0
    assert ae.extractor.raw_v.grad is not None


def test_export_loads_strict_into_the_rl_extractor(tmp_path):
    """The whole point: the exported file is what --pretrained_cgf_model_path reads,
    reproduces the block bit for bit, passes the geometry check, and a different
    geometry is refused."""
    torch.manual_seed(0)
    ae = _ae(feature_norm="layernorm", readout_hidden=32, readout_depth=1, readout_dim=64)
    with torch.no_grad():
        ae.extractor.raw_a.add_(0.3)                       # move t off its init
    trainer_ckpt = tmp_path / "checkpoint_best.pt"
    torch.save({"model_state_dict": ae.state_dict(), "epoch": 3, "global_step": 99,
                "best_val_loss": 0.123, "alignment": None,
                "config": TrainingConfig(model_type="cgf_arm_ae", num_particles=N,
                                         dim_particles=D)}, trainer_ckpt)
    out = export_arm_checkpoint(trainer_ckpt, tmp_path / "checkpoint_best_cgf_arm.pt",
                                ae.extractor, particle_centre=0.0,
                                objective="reconstruction_sinkhorn", data_path="x.npz")
    payload = torch.load(out, map_location="cpu", weights_only=False)
    # The export must be pure Python + tensors: the Trainer's dataclass is flattened,
    # so it unpickles in an interpreter that cannot import set_transformer (the
    # 2026-09-11 smart wave died on exactly this, between pretraining and RL).
    assert isinstance(payload["trainer_config"], dict)
    assert payload["trainer_config"]["model_type"] == "cgf_arm_ae"
    import pickletools, zipfile
    with zipfile.ZipFile(out) as z:
        pkl = z.read(next(n for n in z.namelist() if n.endswith("data.pkl")))
    globals_used = {arg for op, arg, _ in pickletools.genops(pkl)
                    if op.name in ("GLOBAL", "STACK_GLOBAL") and arg}
    assert not any("set_transformer" in str(g) for g in globals_used), globals_used
    assert set(payload["model_state_dict"]) == set(ae.extractor.state_dict())
    assert payload["config"]["arena_scale"] == SCALE and payload["config"]["t_param"] == "polar"
    assert payload["particle_scale"] == SCALE

    kw = dict(POLAR, feature_norm="layernorm", readout_hidden=32, readout_depth=1, readout_dim=64)
    fresh = make_arm_extractor(N, D, arena_scale=SCALE, **kw, pretrained_cgf_model_path=str(out))
    _, w, x_raw = _norm_batch()
    fresh.eval(); ae.extractor.eval()
    assert torch.equal(fresh.cgf_block(x_raw, w), ae.extractor.cgf_block(x_raw, w))

    with pytest.raises((RuntimeError, ValueError)):
        make_arm_extractor(N, D, arena_scale=SCALE, **{**kw, "feature_mode": "K"},
                           pretrained_cgf_model_path=str(out))
    with pytest.raises((RuntimeError, ValueError)):
        make_arm_extractor(N, D, arena_scale=9.0, **kw, pretrained_cgf_model_path=str(out))


@pytest.mark.parametrize("weighted", [True, False])
def test_end_to_end_script_on_a_tiny_weighted_dataset(tmp_path, weighted):
    """Run 3_train_st.py --encoder cgf for real: 2 epochs on 48 synthetic clouds,
    then load the export into the RL extractor."""
    rng = np.random.default_rng(0)
    S = 48
    centres = rng.uniform(-3, 3, size=(S, 1, D))
    particles = (centres + rng.normal(0, 0.7, size=(S, N, D))).astype(np.float32)
    weights = rng.random((S, N)).astype(np.float32)
    weights[:, N // 2:] *= 0.05
    weights /= weights.sum(1, keepdims=True)
    data = tmp_path / "tiny_pf_dataset.npz"
    meta = json.dumps({"env_id": "synthetic", "particle_scale": SCALE, "particle_centre": 0.0})
    np.savez(data, particles=particles, weights=weights, metadata=np.array(meta))

    script = Path(__file__).resolve().parents[1] / "experiments" / "ant_tag" / "3_train_st.py"
    cmd = [sys.executable, str(script), "--data_path", str(data), "--encoder", "cgf",
           "--t_param", "polar", "--t_bound", "13.5", "--feature_mode", "K_grad",
           "--num_epochs", "2", "--batch_size", "8", "--particle_scale", str(SCALE),
           "--base_dir", str(tmp_path / "runs"), "--experiment_name", "smoke",
           "--eval_freq", "3", "--save_freq", "1000", "--num_workers", "0",
           "--warmup_epochs", "1", "--seed", "0", "--dim_hidden", "32"]
    if not weighted:
        cmd.append("--ignore_weights")
    env = {**os.environ, "WANDB_MODE": "offline", "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(cmd, cwd=script.parent, env=env, capture_output=True, text=True,
                          timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + "\n" + proc.stderr[-3000:]
    exports = list((tmp_path / "runs" / "smoke").glob("*/checkpoints/checkpoint_best_cgf_arm.pt"))
    assert len(exports) == 1, proc.stdout[-2000:]
    payload = torch.load(exports[0], map_location="cpu", weights_only=False)
    assert payload["config"]["weighted_pretraining"] is weighted
    assert payload["config"]["arena_scale"] == SCALE
    ext = make_arm_extractor(N, D, arena_scale=SCALE, num_cgf_features=64, t_param="polar",
                             t_bound=13.5, t_init_max=0.8 * 13.5, feature_mode="K_grad",
                             pretrained_cgf_model_path=str(exports[0]))
    assert isinstance(ext, WeightedCGFFeaturesExtractor)
    assert "Exported CGF arm encoder" in proc.stdout


def test_rl_script_accepts_the_export_and_reloads_it(tmp_path):
    """The consumer, for real: 4_train_rl_cgf.py --variant smart --pretrained_cgf_model_path
    <export> must pass its parser-level geometry check (flags at default take the
    checkpoint's polar / 13.5 / K_grad; arena_scale must equal the variant's 4.5),
    build PPO, re-load the encoder after construction and verify it against the file.
    A handful of PPO steps on one CPU env; cwd is tmp so runs/ lands there."""
    rng = np.random.default_rng(1)
    S = 24
    particles = (rng.uniform(-3, 3, size=(S, 1, D))
                 + rng.normal(0, 0.7, size=(S, N, D))).astype(np.float32)
    weights = rng.random((S, N)).astype(np.float32)
    weights /= weights.sum(1, keepdims=True)
    data = tmp_path / "tiny_pf_dataset.npz"
    np.savez(data, particles=particles, weights=weights,
             metadata=np.array(json.dumps({"particle_scale": SCALE, "particle_centre": 0.0})))
    ant_tag = Path(__file__).resolve().parents[1] / "experiments" / "ant_tag"
    env = {**os.environ, "WANDB_MODE": "offline", "CUDA_VISIBLE_DEVICES": ""}

    pre = subprocess.run(
        [sys.executable, str(ant_tag / "3_train_st.py"), "--data_path", str(data),
         "--encoder", "cgf", "--t_param", "polar", "--t_bound", "13.5",
         "--feature_mode", "K_grad", "--num_epochs", "1", "--batch_size", "8",
         "--particle_scale", str(SCALE), "--base_dir", str(tmp_path / "pre"),
         "--experiment_name", "smoke", "--eval_freq", "3", "--save_freq", "1000",
         "--num_workers", "0", "--warmup_epochs", "1", "--dim_hidden", "32"],
        cwd=ant_tag, env=env, capture_output=True, text=True, timeout=900)
    assert pre.returncode == 0, pre.stdout[-2000:] + pre.stderr[-2000:]
    export = next((tmp_path / "pre" / "smoke").glob("*/checkpoints/checkpoint_best_cgf_arm.pt"))

    rl = subprocess.run(
        [sys.executable, str(ant_tag / "4_train_rl_cgf.py"), "--variant", "smart",
         "--pretrained_cgf_model_path", str(export),
         "--total_timesteps", "32", "--n_envs", "1", "--ppo_n_steps", "16",
         "--batch_size", "16", "--n_epochs", "1", "--n_eval_episodes", "1",
         "--eval_freq", "100000", "--save_freq", "100000", "--device", "cpu", "--seed", "0",
         "--log_dir", str(tmp_path / "rl" / "logs") + "/",
         "--model_save_path", str(tmp_path / "rl" / "models" / "agent.zip")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1500)
    out = rl.stdout + rl.stderr
    assert rl.returncode == 0, out[-4000:]
    assert "encoder RE-loaded after PPO construction" in out, out[-3000:]
    # Flags left at default took the checkpoint's geometry (polar / 13.5 / K_grad).
    assert "polar" in out and "K_grad" in out, out[-3000:]
    # The derived runs/ tree is relative to cwd, i.e. under tmp -- not in the repo.
    assert (tmp_path / "runs").exists() or (tmp_path / "rl").exists()
    assert (tmp_path / "rl" / "models" / "agent.zip").exists()
