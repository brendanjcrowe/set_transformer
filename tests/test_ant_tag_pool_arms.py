"""The pooling / moment arms (4_train_rl_{deepset,pointnet,kmoments}.py, 2026-09-11).

Extractor contract (frame, weights, permutation invariance, pooling semantics, pretrained
load and freeze), then the real scripts on `--variant smart` for a few PPO steps each,
including a DeepSet run that loads a DeepSetAE checkpoint (the benchmark's format) and
must show the post-construction reload + verification line.
"""

import os
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from set_transformer.models import DeepSetAE, PointNetAE
from set_transformer.rl.feature_extractors.pooled import (
    PointNetFeaturesExtractor,
    WeightedDeepSetFeaturesExtractor,
    WeightedKMomentsFeaturesExtractor,
)

B, N, D, SCALE, OBS = 4, 100, 2, 4.5, 31


def _space():
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (N, D), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (N,), np.float32),
    })


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(B, N, generator=g)
    w[:, N // 2:] *= 0.01
    w[0, 3] = 0.0                                  # an exactly refuted particle
    return {"obs": torch.randn(B, OBS, generator=g),
            "particles": torch.randn(B, N, D, generator=g) * 3.0,
            "weights": w / w.sum(1, keepdim=True)}


def _quiet(fn):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


@pytest.mark.parametrize("cls,width", [(WeightedDeepSetFeaturesExtractor, 64),
                                       (PointNetFeaturesExtractor, 64)])
def test_learned_arms_output_obs_plus_code_and_are_permutation_invariant(cls, width):
    torch.manual_seed(0)
    e = _quiet(lambda: cls(_space(), arena_scale=SCALE)).eval()
    b = _batch()
    out = e(b)
    assert out.shape == (B, OBS + width)
    assert torch.equal(out[:, :OBS], b["obs"])
    perm = torch.randperm(N)
    shuffled = {**b, "particles": b["particles"][:, perm], "weights": b["weights"][:, perm]}
    assert torch.allclose(e(b), e(shuffled), atol=1e-5)
    assert e.encoder_parameter_count() > 0


def test_deepset_weighted_pool_uses_the_mass_and_mean_pool_does_not():
    torch.manual_seed(0)
    weighted = _quiet(lambda: WeightedDeepSetFeaturesExtractor(_space(), arena_scale=SCALE, weight_channel=False)).eval()
    mean = _quiet(lambda: WeightedDeepSetFeaturesExtractor(_space(), arena_scale=SCALE, weight_channel=False, pooling="mean")).eval()
    mean.encoder.load_state_dict(weighted.encoder.state_dict())
    b = _batch()
    uniform = {**b, "weights": torch.full((B, N), 1.0 / N)}
    # same network: equal under uniform weights ...
    assert torch.allclose(weighted.encode(uniform), mean.encode(uniform), atol=1e-5)
    # ... and under skewed weights the POOLED vectors differ materially (the random-init
    # dec MLP then compresses that to ~1e-4 on the code, so the code check is inequality
    # at float precision, not a magnitude claim)
    x, w = weighted._prepare(b)
    phi = weighted.encoder.enc(x)
    pooled_w, pooled_m = weighted._pool(phi, w), mean._pool(phi, w)
    # ~4 % on a random-init net whose phi is bias-dominated; the point is nonzero vs the
    # mean pool's exact zero below, not the size
    assert (pooled_w - pooled_m).norm() / pooled_m.norm() > 0.01
    assert torch.equal(mean._pool(phi, w), mean._pool(phi, torch.full_like(w, 1.0 / N)))
    assert not torch.allclose(weighted.encode(b), mean.encode(b), atol=1e-6, rtol=0.0)
    assert torch.allclose(mean.encode(b), mean.encode(uniform), atol=1e-5)    # mean pool is weight-blind


def test_pointnet_masked_max_ignores_a_refuted_particle_and_plain_max_does_not():
    torch.manual_seed(0)
    masked = _quiet(lambda: PointNetFeaturesExtractor(_space(), arena_scale=SCALE, weight_channel=False)).eval()
    plain = _quiet(lambda: PointNetFeaturesExtractor(_space(), arena_scale=SCALE, weight_channel=False, pooling="max")).eval()
    plain.encoder.load_state_dict(masked.encoder.state_dict())
    b = _batch()
    # push particle 3 of row 0 (weight exactly 0) far away so it would dominate a max
    far = {**b, "particles": b["particles"].clone()}
    far["particles"][0, 3] = 40.0
    assert torch.allclose(masked.encode(far)[0], masked.encode(b)[0], atol=1e-5)
    assert not torch.allclose(plain.encode(far)[0], plain.encode(b)[0], atol=1e-3)


def test_kmoments_k2_is_the_weighted_mean_and_variance_on_the_scaled_frame():
    e = WeightedKMomentsFeaturesExtractor(_space(), k=2, arena_scale=SCALE)
    b = _batch()
    out = e(b)
    assert out.shape == (B, OBS + 2 * D) and e.encoder_parameter_count() == 0
    x = b["particles"] / SCALE
    w = b["weights"].unsqueeze(-1)
    mean = (w * x).sum(1)
    var = (w * (x - mean.unsqueeze(1)) ** 2).sum(1)
    assert torch.allclose(out[:, OBS:OBS + D], mean, atol=1e-5)
    assert torch.allclose(out[:, OBS + D:], var, atol=1e-5)
    e4 = WeightedKMomentsFeaturesExtractor(_space(), k=4, arena_scale=SCALE)
    assert e4(b).shape == (B, OBS + 4 * D)


@pytest.mark.parametrize("cls,ae_cls", [(WeightedDeepSetFeaturesExtractor, DeepSetAE),
                                        (PointNetFeaturesExtractor, PointNetAE)])
def test_pretrained_ae_checkpoint_loads_freezes_and_mismatches_are_refused(tmp_path, cls, ae_cls):
    """The benchmark's autoencoders are unweighted (D inputs): they load with
    weight_channel=False, reproduce the encoder, freeze; a weight-channel run refuses
    them with the flag to fix, and a wrong geometry is refused."""
    torch.manual_seed(0)
    ae = ae_cls(num_particles=N, dim_particles=D, num_encodings=8, dim_encoder=8, dim_hidden=128)
    ckpt = tmp_path / "ae.pt"
    torch.save(ae.state_dict(), ckpt)
    e = _quiet(lambda: cls(_space(), arena_scale=SCALE, weight_channel=False,
                           pretrained_model_path=str(ckpt), frozen=True))
    assert all(torch.equal(a, b) for a, b in zip(e.encoder.state_dict().values(), ae.encoder.state_dict().values()))
    assert all(not p.requires_grad for p in e.encoder.parameters())
    e.train(True)
    assert not e.encoder.training
    with pytest.raises(RuntimeError, match="no_weight_channel"):
        _quiet(lambda: cls(_space(), arena_scale=SCALE, weight_channel=True, pretrained_model_path=str(ckpt)))
    with pytest.raises(RuntimeError):
        _quiet(lambda: cls(_space(), arena_scale=SCALE, weight_channel=False, dim_encoder=2,
                           pretrained_model_path=str(ckpt)))
    with pytest.raises(ValueError, match="frozen"):
        _quiet(lambda: cls(_space(), arena_scale=SCALE, frozen=True))


def test_trainer_checkpoint_frame_check(tmp_path):
    """A Trainer checkpoint carries particle_scale; a different arena_scale is refused."""
    torch.manual_seed(0)
    ae = DeepSetAE(num_particles=N, dim_particles=D, num_encodings=8, dim_encoder=8, dim_hidden=128)
    ckpt = tmp_path / "trainer.pt"
    torch.save({"model_state_dict": ae.state_dict(), "particle_scale": 4.5, "config": None}, ckpt)
    _quiet(lambda: WeightedDeepSetFeaturesExtractor(_space(), arena_scale=4.5, weight_channel=False,
                                                    pretrained_model_path=str(ckpt)))
    with pytest.raises(RuntimeError, match="particle_scale"):
        _quiet(lambda: WeightedDeepSetFeaturesExtractor(_space(), arena_scale=9.0, weight_channel=False,
                                                        pretrained_model_path=str(ckpt)))


# --- the real scripts ---------------------------------------------------------------

ANT_TAG = Path(__file__).resolve().parents[1] / "experiments" / "ant_tag"
TINY = ["--variant", "smart", "--total_timesteps", "32", "--n_envs", "1", "--ppo_n_steps", "16",
        "--batch_size", "16", "--n_epochs", "1", "--n_eval_episodes", "1", "--eval_freq", "100000",
        "--save_freq", "100000", "--device", "cpu", "--seed", "0"]


def _run(script, extra, tmp_path, tag):
    env = {**os.environ, "WANDB_MODE": "offline", "CUDA_VISIBLE_DEVICES": ""}
    out_dir = tmp_path / tag
    cmd = [sys.executable, str(ANT_TAG / script), *TINY, *extra,
           "--log_dir", str(out_dir / "logs") + "/", "--model_save_path", str(out_dir / "models" / "agent.zip")]
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1500)
    text = proc.stdout + proc.stderr
    assert proc.returncode == 0, text[-4000:]
    assert (out_dir / "models" / "agent.zip").exists()
    return text


@pytest.mark.parametrize("encoder,marker", [
    ("deepset", "WeightedDeepSetFeaturesExtractor"),
    ("pointnet", "PointNetFeaturesExtractor"),
    ("kmoments", "WeightedKMomentsFeaturesExtractor"),
])
def test_arm_script_trains_and_saves(tmp_path, encoder, marker):
    text = _run(f"4_train_rl_{encoder}.py", [], tmp_path, encoder)
    assert marker in text
    cfg = next(Path(tmp_path).glob("runs/ant_tag_*_smart/*/run_config.json"), None)
    assert cfg is not None and f"ant_tag_{encoder}_smart" in str(cfg)


def test_deepset_script_loads_a_pretrained_ae_and_verifies_the_reload(tmp_path):
    torch.manual_seed(0)
    ae = DeepSetAE(num_particles=N, dim_particles=D, num_encodings=8, dim_encoder=8, dim_hidden=128)
    ckpt = tmp_path / "deepset_ae.pt"
    torch.save(ae.state_dict(), ckpt)
    text = _run("4_train_rl_deepset.py",
                ["--pretrained_model_path", str(ckpt), "--no_weight_channel", "--frozen"],
                tmp_path, "deepset_pre")
    assert "RE-loaded after PPO construction" in text and "re-frozen" in text
    text2 = _run("4_train_rl_deepset.py",
                 ["--pretrained_model_path", str(ckpt), "--no_weight_channel", "--encoder_lr_scale", "0.1"],
                 tmp_path, "deepset_ft")
    assert "learning rate scaled by 0.1" in text2
