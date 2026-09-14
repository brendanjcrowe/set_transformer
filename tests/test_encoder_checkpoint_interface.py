"""The inverse of `load_pretrained` on the shared encoder interface (batch 10.4, 2026-09-14).

Every learned extractor (ST, CGF, DeepSet, PointNet) exposes ``checkpoint_state()`` and
``checkpoint_config()``, and ``rl/pretrained_encoder.encoder_checkpoint`` assembles the file a
pretraining objective writes. Pins: the file round-trips into a fresh extractor of the same class
bit for bit; the state keys carry the prefix each loader strips; ``particle_scale`` is top-level
and a frame mismatch is refused; a geometry mismatch is refused; the pooled weight-channel
mismatch names the flag; the two objectives' writers (Odd-Even belief, hunt task) produce the same
encoder tensors and geometry for the same extractor -- the point of having one writer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

_ST_ROOT = Path(__file__).resolve().parents[1]
if str(_ST_ROOT) not in sys.path:
    sys.path.insert(0, str(_ST_ROOT))

pytest.importorskip("stable_baselines3")

from set_transformer.rl import encoders  # noqa: E402
from set_transformer.rl.pretrained_encoder import encoder_checkpoint, verify_matches_checkpoint  # noqa: E402

N, D, SCALE = 20, 2, 7.0


def _space(d=D):
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (N, d), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (N,), np.float32),
    })


SMALL = {
    "st": dict(num_encodings=4, dim_encoder=4, num_inds=4, dim_hidden=16, num_heads=2, num_post_sab=0),
    "cgf": dict(num_cgf_features=8, feature_mode="K_grad", t_param="tanh", t_bound=3.0,
                readout_hidden=16, readout_depth=1, readout_dim=8),
    "deepset": dict(num_encodings=4, dim_encoder=4, dim_hidden=16),
    "pointnet": dict(num_encodings=4, dim_encoder=4, dim_hidden=16),
}
LEARNED = sorted(name for name, e in encoders.ENCODERS.items() if e.learned)
PREFIX = {"st": "set_transformer.", "deepset": "encoder.", "pointnet": "encoder."}


def _build(name, path=None, **override):
    cls = encoders.get(name).extractor_class
    kwargs = {**SMALL[name], "arena_scale": SCALE, **override}
    kwargs[cls.PRETRAINED_PATH_KWARG] = path
    return cls(_space(), **kwargs)


def _obs(seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(2, N, generator=g)
    return {"obs": torch.zeros(2, 3), "particles": torch.randn(2, N, D, generator=g) * SCALE,
            "weights": w / w.sum(1, keepdim=True)}


def test_the_table_has_the_four_learned_encoders():
    assert LEARNED == ["cgf", "deepset", "pointnet", "st"]


@pytest.mark.parametrize("name", LEARNED)
def test_checkpoint_round_trips_bit_for_bit_and_carries_the_loaders_prefix(name, tmp_path):
    torch.manual_seed(1)
    a = _build(name)
    path = tmp_path / f"{name}.pt"
    payload = encoder_checkpoint(a, config={"objective": "x"}, epoch=3)
    assert set(payload) == {"model_state_dict", "config", "particle_scale", "epoch"}
    assert payload["particle_scale"] == SCALE and payload["epoch"] == 3
    assert payload["config"]["objective"] == "x" and payload["config"]["arena_scale"] == SCALE
    if name in PREFIX:
        assert all(k.startswith(PREFIX[name]) for k in payload["model_state_dict"])
    else:
        assert set(payload["model_state_dict"]) == set(a.state_dict())      # the CGF: the whole extractor
    assert all(v.device.type == "cpu" for v in payload["model_state_dict"].values())
    torch.save(payload, path)
    torch.manual_seed(2)
    b = _build(name, path=str(path))
    verify_matches_checkpoint(b.reference_state(str(path)), b.encoder_state_dict(), str(path))
    with torch.no_grad():
        torch.testing.assert_close(a(_obs()), b(_obs()))
    # a fresh random extractor does NOT match: the check is not vacuous
    torch.manual_seed(3)
    with pytest.raises(AssertionError, match="does not match"):
        verify_matches_checkpoint(b.reference_state(str(path)), _build(name).encoder_state_dict(), str(path))


@pytest.mark.parametrize("name", LEARNED)
def test_frame_and_geometry_mismatches_are_refused(name, tmp_path):
    path = tmp_path / f"{name}.pt"
    torch.save(encoder_checkpoint(_build(name)), path)
    with pytest.raises(RuntimeError, match="scale"):
        _build(name, path=str(path), arena_scale=SCALE * 2)
    wrong = {"st": dict(dim_hidden=32), "cgf": dict(num_cgf_features=16),
             "deepset": dict(dim_hidden=32), "pointnet": dict(dim_hidden=32)}[name]
    with pytest.raises(RuntimeError):
        _build(name, path=str(path), **wrong)


@pytest.mark.parametrize("name", ["st", "deepset", "pointnet"])
def test_weight_channel_mismatch_is_refused_with_the_flag_named(name, tmp_path):
    path = tmp_path / f"{name}.pt"
    a = _build(name)
    payload = encoder_checkpoint(a)
    assert payload["config"]["weighted_particles"] is True
    torch.save(payload, path)
    flag = {"st": "no_st_weight_channel", "deepset": "no_weight_channel", "pointnet": "no_weight_channel"}[name]
    kw = {"st": dict(weight_channel=False), "deepset": dict(weight_channel=False), "pointnet": dict(weight_channel=False)}[name]
    with pytest.raises(RuntimeError, match="weight"):
        _build(name, path=str(path), **kw)


@pytest.mark.parametrize("name", LEARNED)
def test_both_objectives_write_the_same_encoder_for_the_same_extractor(name, tmp_path):
    """Odd-Even's belief writer and hunt's task writer both call encoder_checkpoint: same tensors,
    same geometry record, same top-level particle_scale; they differ in their own config fields."""
    from set_transformer.rl.domains import hunt, odd_even
    torch.manual_seed(5)
    extractor = _build(name)
    obs_dim = 3
    belief = odd_even.BeliefEncoderWithHead.__new__(odd_even.BeliefEncoderWithHead)
    torch.nn.Module.__init__(belief)
    belief.extractor = extractor
    belief.head = torch.nn.Linear(extractor.features_dim - 1, 5)
    task = hunt.TaskEncoderWithHead(extractor, obs_dim=obs_dim, out_dim=2, hidden=8)
    args = argparse.Namespace(objective="belief_kl", variant="v", encoder=name, encoder_params=7)
    odd_even.save_belief_checkpoint(belief, tmp_path / "belief.pt", args, 1, {"loss": 0.0},
                                    odd_even.extractor_geometry(extractor))
    hunt.save_task_checkpoint(task, tmp_path / "task.pt", args, 1, {"loss": 0.0},
                              hunt.extractor_geometry(extractor), "pick_target")
    b = torch.load(tmp_path / "belief.pt", map_location="cpu", weights_only=False)
    t = torch.load(tmp_path / "task.pt", map_location="cpu", weights_only=False)
    assert set(b["model_state_dict"]) == set(t["model_state_dict"])
    for k in b["model_state_dict"]:
        torch.testing.assert_close(b["model_state_dict"][k], t["model_state_dict"][k])
    assert b["particle_scale"] == t["particle_scale"] == SCALE
    geometry = extractor.checkpoint_config()
    assert {k: b["config"][k] for k in geometry} == geometry == {k: t["config"][k] for k in geometry}
    assert b["config"]["objective"] == "belief_kl" and t["config"]["objective"] == "task"
    assert b["config"]["pretraining"] == "3_pretrain_st_belief.py" and t["config"]["task"] == "pick_target"
    assert set(b) == {"model_state_dict", "config", "particle_scale", "head_state_dict", "epoch", "val", "args"}
    # and each loads back through the class's own loader
    for f in ("belief.pt", "task.pt"):
        c = _build(name, path=str(tmp_path / f))
        verify_matches_checkpoint(c.reference_state(str(tmp_path / f)), c.encoder_state_dict(), str(tmp_path / f))
