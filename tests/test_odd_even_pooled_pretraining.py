"""The exact-posterior objectives on the DeepSet and PointNet arms (batch 10.4, 2026-09-14).

Through the package command: belief_kl on both pooled arms pretrains one epoch, the door's
round-trip check passes, the checkpoint is in the pooled loader's format (``encoder.`` keys,
``weighted_particles`` true, ``particle_scale`` 24.5 top-level), the trainer's dry run accepts it
frozen, an extractor built without the weight channel refuses it naming the flag; mode_ce runs on
a pooled arm; an analytic encoder is refused with the new message.
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

import gymnasium as gym  # noqa: E402

from set_transformer.rl import encoders, pretrain, run_records  # noqa: E402
from set_transformer.rl import train as train_mod  # noqa: E402

TINY = ["--variant", "oe50_short", "--device", "cpu", "--n_train_episodes", "4", "--n_val_episodes", "2",
        "--num_epochs", "1", "--skip_probe", "--dim_hidden", "16"]


@pytest.mark.parametrize("encoder", ["deepset", "pointnet"])
def test_belief_kl_on_a_pooled_arm_round_trips_and_loads_frozen(encoder, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    root = tmp_path / "root"
    result = pretrain.main(["--domain", "odd_even", "--encoder", encoder, "--objective", "belief_kl",
                            "--output_root", str(root), *TINY])
    out = capsys.readouterr().out
    cls = encoders.get(encoder).extractor_class.__name__
    assert f"Verified: {cls} encoder matches" in out
    assert result.rl_checkpoint.exists() and result.summary["best_epoch"] == 1
    ck = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert all(k.startswith("encoder.") for k in ck["model_state_dict"]) and ck["model_state_dict"]
    c = ck["config"]
    assert (c["encoder"], c["objective"], c["variant"]) == (encoder, "belief_kl", "oe50_short")
    assert c["weighted_particles"] is True and c["dim_hidden"] == 16 and c["arena_scale"] == 24.5
    assert ck["particle_scale"] == 24.5 and c["encoder_params"] > 0
    assert ck["head_state_dict"]["weight"].shape == (50, c["num_encodings"] * c["dim_encoder"])
    assert list(root.glob(f"odd_even/oe50_short/pretrain/{encoder}/belief_kl/*_seed0/run_status.json"))
    found = run_records.latest_pretrain_checkpoint("odd_even", "oe50_short", encoder, "belief_kl", root=root)
    assert Path(found) == result.rl_checkpoint.resolve()
    # the trainer's dry run takes it frozen
    train_mod.main(["--domain", "odd_even", "--encoder", encoder, "--variant", "oe50_short", "--dim_hidden", "16",
                    "--pretrained_path", str(found), "--frozen", "--output_root", str(root), "--dry_run"])
    [record] = list(root.glob(f"odd_even/oe50_short/rl/{encoder}/*_seed0/run_config.json"))
    config = json.loads(record.read_text())
    # the start mode is recorded under the encoder's own historical key names
    assert str(found) in config.values()
    assert any(k.endswith("frozen") and v is True for k, v in config.items())
    # an extractor built WITHOUT the weight channel refuses the weighted checkpoint, naming the flag
    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (50, 1), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (50,), np.float32),
    })
    cls = encoders.get(encoder).extractor_class
    with pytest.raises(RuntimeError, match="weight_channel"):
        cls(space, dim_hidden=16, arena_scale=24.5, weight_channel=False, pretrained_model_path=str(found))


def test_mode_ce_on_a_pooled_arm_and_analytic_encoders_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    result = pretrain.main(["--domain", "odd_even", "--encoder", "deepset", "--objective", "mode_ce",
                            "--output_root", str(tmp_path / "root"), *TINY])
    ck = torch.load(result.rl_checkpoint, map_location="cpu", weights_only=False)
    assert ck["config"]["objective"] == "mode_ce" and ck["pretraining_run"]["encoder"] == "deepset"
    with pytest.raises(SystemExit):
        pretrain.main(["--domain", "odd_even", "--encoder", "gaussian", "--objective", "belief_kl", *TINY,
                       "--output_root", str(tmp_path / "root")])
    assert "has no parameters to pretrain" in capsys.readouterr().err      # the door refuses it first


def test_a_frozen_pooled_agent_reloads_from_its_zip(tmp_path, monkeypatch, capsys):
    """Found by the w1p smoke: the pooled constructor refused frozen=True without a checkpoint path,
    which is exactly the pair SB3 rebuilds the extractor with at PPO.load (the reload blanks the path
    in the pickled kwargs). Pretrain, train 64 frozen steps, then load the zip and act."""
    from stable_baselines3 import PPO
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    root = tmp_path / "root"
    result = pretrain.main(["--domain", "odd_even", "--encoder", "deepset", "--objective", "belief_kl",
                            "--output_root", str(root), *TINY])
    model_path = tmp_path / "models" / "deepset_agent.zip"
    train_mod.main(["--domain", "odd_even", "--encoder", "deepset", "--variant", "oe50_short", "--dim_hidden", "16",
                    "--pretrained_path", str(result.rl_checkpoint), "--frozen", "--total_timesteps", "64",
                    "--n_envs", "1", "--ppo_n_steps", "32", "--batch_size", "32", "--n_epochs", "1",
                    "--eval_freq", "1000000", "--n_eval_episodes", "1", "--save_freq", "1000000", "--device", "cpu",
                    "--output_root", str(root), "--log_dir", str(tmp_path / "logs") + "/",
                    "--model_save_path", str(model_path)])
    assert model_path.exists()
    capsys.readouterr()
    model = PPO.load(str(model_path), device="cpu")          # rebuilds the extractor with frozen=True, path None
    assert "frozen=True without a checkpoint path" in capsys.readouterr().out
    extractor = model.policy.features_extractor
    assert not any(p.requires_grad for p in extractor.encoder_parameters())
    # the trained (= pretrained, frozen) encoder tensors are the checkpoint's
    from set_transformer.rl.pretrained_encoder import verify_matches_checkpoint
    verify_matches_checkpoint(extractor.reference_state(str(result.rl_checkpoint)), extractor.encoder_state_dict(),
                              str(result.rl_checkpoint))
    obs = model.observation_space.sample()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == ()  or action.shape == (1,) or action is not None
