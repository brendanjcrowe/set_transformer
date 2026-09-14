"""``--separate_extractors`` and ``--ent_coef`` on the shared trainer (batch 9.1, 2026-09-13).

Every recorded hunt PPO run gave the value network its own features extractor
(``share_features_extractor=False``); the harness default shares one. With the flag on, the
policy holds TWO extractor objects, and every post-construction step that touches "the
encoder" has to touch both: the pretrained reload and its verification, the freeze, the
scaled learning rate, the unfreeze. ``rl/pretrained_encoder.policy_extractors`` is the one
enumeration; these tests pin that it finds both and that the steps use it.
"""
from __future__ import annotations

import contextlib
import io
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

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from set_transformer.rl import pretrained_encoder as pe  # noqa: E402
from set_transformer.rl import run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402
from set_transformer.rl.encoder_finetune import (  # noqa: E402
    UnfreezeEncoderCallback,
    scale_encoder_learning_rate,
)
from set_transformer.rl.feature_extractors.st import SetTransformerFeaturesExtractor  # noqa: E402

ODD_EVEN_SPACE = gym.spaces.Dict({
    "obs": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
    "particles": gym.spaces.Box(-np.inf, np.inf, (50, 1), np.float32),
    "weights": gym.spaces.Box(0.0, 1.0, (50,), np.float32),
})
ODD_EVEN_SCALE = 24.5


@pytest.fixture(scope="module")
def st_checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    st = SetTransformerFeaturesExtractor(
        ODD_EVEN_SPACE, num_encodings=8, dim_encoder=8, num_inds=8, dim_hidden=32,
        num_heads=4, ln=True, arena_scale=ODD_EVEN_SCALE, weight_channel=True, num_post_sab=1)
    path = tmp_path_factory.mktemp("ckpt") / "odd_even_st.pt"
    torch.save({"model_state_dict": {f"set_transformer.{k}": v
                                     for k, v in st.encoder.state_dict().items()},
                "config": dict(st._st_geometry)}, path)
    return str(path)


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def _drive_shared(monkeypatch, tmp_path, argv, domain="odd_even", encoder="st"):
    captured = {}
    monkeypatch.setattr(run_records, "output_root", lambda *a, **k: tmp_path / "runs")
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.setattr(run_records, "write_run_config",
                        lambda run_dir, **cfg: captured.setdefault("config", cfg))
    train_mod.main(list(argv) + ["--dry_run"], domain=domain, encoder=encoder)
    return captured["config"]


# --------------------------------------------------------------------------
# The enumeration
# --------------------------------------------------------------------------

def test_policy_extractors_finds_the_value_networks_own_extractor_once():
    class _P:
        pass
    pi, vf = object(), object()
    policy = _P()
    policy.features_extractor = pi
    policy.pi_features_extractor = pi
    policy.vf_features_extractor = vf
    model = _P(); model.policy = policy
    found = pe.policy_extractors(model)
    assert found[0] is pi and len(found) == 2 and found[1] is vf
    # shared: the same object under both names is listed once
    policy.vf_features_extractor = pi
    assert pe.policy_extractors(model) == [pi]


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------

def test_flags_default_off_and_are_recorded(monkeypatch, tmp_path):
    config = _drive_shared(monkeypatch, tmp_path, ["--variant", "oe50_short"])
    assert config["ent_coef"] == 0.0 and config["separate_extractors"] is False
    config = _drive_shared(monkeypatch, tmp_path,
                           ["--variant", "oe50_short", "--ent_coef", "0.005", "--separate_extractors"])
    assert config["ent_coef"] == 0.005 and config["separate_extractors"] is True


def test_flags_are_refused_with_sac(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _drive_shared(monkeypatch, tmp_path,
                      ["--variant", "oe50_short", "--algorithm", "SAC", "--ent_coef", "0.01"])
    with pytest.raises(SystemExit):
        _drive_shared(monkeypatch, tmp_path,
                      ["--variant", "oe50_short", "--algorithm", "SAC", "--separate_extractors"])
    with pytest.raises(ValueError, match="PPO options"):
        train_mod.train("odd_even", "oe50_short", "gaussian",
                        features_extractor_kwargs=dict(arena_scale=ODD_EVEN_SCALE),
                        num_particles=50, total_timesteps=8, algorithm="SAC",
                        separate_extractors=True)


# --------------------------------------------------------------------------
# Through train(): two extractors, both reloaded, verified, frozen, scaled, released
# --------------------------------------------------------------------------

def _tiny(st_checkpoint, tmp_path, **kwargs):
    common = dict(
        features_extractor_kwargs=dict(
            num_encodings=8, dim_encoder=8, num_inds=8, dim_hidden=32, num_heads=4, ln=True,
            arena_scale=ODD_EVEN_SCALE, weight_channel=True, num_post_sab=1),
        num_particles=50, n_envs=1, seed=0, total_timesteps=64, ppo_n_steps=32, batch_size=16,
        n_epochs=1, device="cpu", eval_freq=1_000_000, save_freq=1_000_000, n_eval_episodes=1,
        log_dir=str(tmp_path / "logs") + "/",
        model_save_path=str(tmp_path / "models" / "st_agent.zip"))
    common.update(kwargs)
    captured = {}

    def post_construct(model):
        captured["extractors"] = pe.policy_extractors(model)
        captured["frozen"] = [not any(p.requires_grad for p in e.encoder_parameters())
                              for e in captured["extractors"]]
        captured["groups"] = [list(g["params"]) for g in model.policy.optimizer.param_groups]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = train_mod.train("odd_even", "oe50_short", "st", post_construct=post_construct,
                                **common)
    return model, captured, buf.getvalue()


@pytest.mark.slow
def test_separate_extractors_two_objects_both_reloaded_verified_and_frozen(st_checkpoint, tmp_path):
    model, cap, out = _tiny(st_checkpoint, tmp_path, pretrained_path=st_checkpoint, frozen=True,
                            separate_extractors=True, ent_coef=0.005)
    ex = cap["extractors"]
    assert len(ex) == 2 and ex[0] is not ex[1]
    assert model.policy.share_features_extractor is False
    assert ex[0] is model.policy.pi_features_extractor and ex[1] is model.policy.vf_features_extractor
    assert cap["frozen"] == [True, True]
    assert out.count("Verified: SetTransformerFeaturesExtractor encoder [") == 2
    assert "encoder [1/2] matches" in out and "encoder [2/2] matches" in out
    assert "2 extractors (separate actor / critic)" in out
    assert "(x2: separate actor / critic extractors)" in out
    assert model.ent_coef == 0.005
    # both encoders still equal the checkpoint after training: frozen means frozen in both
    reference = ex[0].reference_state(st_checkpoint)
    for extractor in ex:
        worst, _n = pe.max_abs_delta(reference, extractor.encoder_state_dict())
        assert worst == 0.0


@pytest.mark.slow
def test_separate_extractors_lr_scale_puts_both_encoders_in_the_scaled_group(st_checkpoint, tmp_path):
    model, cap, out = _tiny(st_checkpoint, tmp_path, pretrained_path=st_checkpoint,
                            encoder_lr_scale=0.1, separate_extractors=True)
    ex = cap["extractors"]
    encoder_ids = {id(p) for e in ex for p in e.encoder_parameters()}
    assert len(encoder_ids) == 2 * len(list(ex[0].encoder_parameters()))
    assert {id(p) for p in cap["groups"][0]} == encoder_ids
    assert not ({id(p) for p in cap["groups"][1]} & encoder_ids)
    assert f"({len(encoder_ids)} encoder tensors at" in out


def test_shared_default_is_one_extractor(st_checkpoint, tmp_path):
    model, cap, out = _tiny(st_checkpoint, tmp_path, pretrained_path=st_checkpoint, frozen=True,
                            total_timesteps=32)
    assert len(cap["extractors"]) == 1 and model.policy.share_features_extractor is True
    assert out.count("Verified: SetTransformerFeaturesExtractor encoder matches") == 1
    assert "separate" not in out


def test_unfreeze_callback_releases_both_extractors(st_checkpoint, tmp_path):
    model, cap, out = _tiny(st_checkpoint, tmp_path, pretrained_path=st_checkpoint,
                            unfreeze_at=32, separate_extractors=True)
    assert "encoder UNFROZEN at step" in out
    for extractor in cap["extractors"]:
        assert not extractor.st_frozen
        assert all(p.requires_grad for p in extractor.encoder_parameters())
    cb = UnfreezeEncoderCallback(0)
    cb.model = model
    assert cb._encoders() == pe.policy_extractors(model)


def test_scale_encoder_learning_rate_dedupes_a_shared_extractor(st_checkpoint, tmp_path):
    """With the default shared extractor the enumeration yields one object; the scaled group
    holds each encoder tensor once (the pre-9.1 behaviour, unchanged)."""
    model, cap, out = _tiny(st_checkpoint, tmp_path, pretrained_path=st_checkpoint,
                            encoder_lr_scale=0.5, total_timesteps=32)
    encoder = list(model.policy.features_extractor.encoder_parameters())
    assert {id(p) for p in cap["groups"][0]} == {id(p) for p in encoder}
    assert len(cap["groups"][0]) == len(encoder)
