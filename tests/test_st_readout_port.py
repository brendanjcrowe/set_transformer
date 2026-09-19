"""The least-mass record's encoder as ST options (2026-09-19): `readout="flatten"` (+ `num_pma_seeds`) and
`input_embed`. Defaults must rebuild the old module exactly; the record-like configuration must have the
record's parameter count; a checkpoint built one way must not load into an extractor built the other way."""
import argparse
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from set_transformer.rl.feature_extractors.st import SetTransformerFeaturesExtractor
from set_transformer.rl.pretrained_encoder import encoder_checkpoint
from set_transformer.rl import encoders as enc_table

N, D = 100, 2
SPACE = gym.spaces.Dict({"obs": gym.spaces.Box(-np.inf, np.inf, (2,), np.float32),
                         "particles": gym.spaces.Box(-np.inf, np.inf, (N, D), np.float32),
                         "weights": gym.spaces.Box(0, 1, (N,), np.float32)})
BASE = dict(num_encodings=8, dim_encoder=8, num_inds=16, dim_hidden=64, num_heads=4, arena_scale=10.0)
RECORD_LIKE = dict(BASE, ln=False, weight_channel=False, num_post_sab=0, readout="flatten", num_pma_seeds=5,
                   input_embed=True)


def _build(path=None, **kw):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return SetTransformerFeaturesExtractor(SPACE, pretrained_st_model_path=path, **{**BASE, **kw})


def _obs(b=3):
    g = torch.Generator().manual_seed(0)
    return {"obs": torch.randn(b, 2, generator=g), "particles": torch.randn(b, N, D, generator=g) * 5,
            "weights": torch.full((b, N), 1.0 / N)}


def test_defaults_rebuild_the_old_module_exactly():
    torch.manual_seed(0); old = _build()
    torch.manual_seed(0); explicit = _build(readout="per_seed", num_pma_seeds=None, input_embed=False)
    assert list(old.encoder.state_dict()) == list(explicit.encoder.state_dict())
    assert not any(k.startswith(("embed.", "readout_linear.")) for k in old.encoder.state_dict())
    cfg = old.checkpoint_config()
    assert cfg["readout"] == "per_seed" and cfg["num_pma_seeds"] == 8 and cfg["input_embed"] is False
    with torch.no_grad():
        assert torch.equal(old(_obs()), explicit(_obs()))


def test_record_like_configuration_has_the_records_parameter_count_and_feature_width():
    """src/hunt_tasks/encoders/extractors.py::SetTransformerExtractor(num_seeds=5, ln=False) on 2-D
    particles: Linear(2->64) + 2 ISAB(64, 16 inducing) + PMA(5 seeds) + Linear(320->64) = 106,304."""
    ext = _build(**RECORD_LIKE)
    assert sum(p.numel() for p in ext.encoder_parameters()) == 106_304
    out = ext(_obs())
    assert out.shape == (3, 2 + 64)                      # agent obs + 8 x 8 = 64 features, as before
    assert ext.encoder.dec[0].S.shape == (1, 5, 64)      # five pooling seeds
    assert ext.encoder.readout_linear.weight.shape == (64, 320)
    assert ext.encoder.embed.weight.shape == (64, 2)


def test_flatten_output_is_gelu_of_the_mixing_linear():
    ext = _build(readout="flatten", num_pma_seeds=5, num_post_sab=0).eval()
    x = torch.randn(2, N, 3)
    with torch.no_grad():
        pooled = ext.encoder.dec(ext.encoder.enc(x)).reshape(2, -1)
        expected = torch.nn.functional.gelu(ext.encoder.readout_linear(pooled)).reshape(2, 8, 8)
        assert torch.allclose(ext.encoder(x), expected)


def test_invalid_combinations_are_refused():
    with pytest.raises(ValueError, match="readout"):
        _build(readout="mixing")
    with pytest.raises(ValueError, match="num_pma_seeds"):
        _build(readout="per_seed", num_pma_seeds=5)


def test_checkpoint_mismatch_is_refused_and_a_legacy_checkpoint_means_defaults(tmp_path):
    flat, dflt, legacy = (tmp_path / f"{k}.pt" for k in ("flat", "dflt", "legacy"))
    torch.save(encoder_checkpoint(_build(**RECORD_LIKE)), flat)
    torch.save(encoder_checkpoint(_build()), dflt)
    old = encoder_checkpoint(_build())
    old["config"] = {k: v for k, v in old["config"].items() if k not in ("readout", "num_pma_seeds", "input_embed")}
    torch.save(old, legacy)
    with pytest.raises(RuntimeError, match="readout"):
        _build(path=str(flat))                                   # record-like into the default extractor
    with pytest.raises(RuntimeError, match="readout|input_embed"):
        _build(path=str(dflt), **RECORD_LIKE)                    # default into a record-like extractor
    with pytest.raises(RuntimeError, match="readout|input_embed"):
        _build(path=str(legacy), **RECORD_LIKE)                  # pre-field checkpoint = defaults, refused
    _build(path=str(flat), **RECORD_LIKE)                        # matching: loads
    _build(path=str(legacy))                                     # pre-field checkpoint into default: loads


def test_encoder_table_resolves_the_three_fields_from_the_checkpoint(tmp_path):
    """No flag given -> the checkpoint's values; a disagreeing flag -> a parser error; no checkpoint -> defaults."""
    path = tmp_path / "flat.pt"
    torch.save(encoder_checkpoint(_build(**RECORD_LIKE)), path)
    domain = SimpleNamespace(encoder_defaults={"st": {"num_inds": 16, "dim_hidden": 64, "num_post_sab": 2}})
    parser = argparse.ArgumentParser()

    def ns(**kw):
        base = dict(num_inds=None, dim_hidden=None, num_post_sab=None, readout=None, num_pma_seeds=None,
                    input_embed=None, pretrained_st_model_path=str(path))
        return argparse.Namespace(**{**base, **kw})

    args = ns()
    enc_table._st_resolve_arguments(parser, args, domain)
    assert (args.readout, args.num_pma_seeds, args.input_embed, args.num_post_sab) == ("flatten", 5, True, 0)
    with pytest.raises(SystemExit):
        enc_table._st_resolve_arguments(parser, ns(readout="per_seed"), domain)
    fresh = ns(pretrained_st_model_path=None)
    enc_table._st_resolve_arguments(parser, fresh, domain)
    assert (fresh.readout, fresh.num_pma_seeds, fresh.input_embed) == ("per_seed", None, False)
    kw = enc_table._st_extractor_kwargs(argparse.Namespace(
        num_encodings=8, dim_encoder=8, num_inds=16, dim_hidden=64, num_heads=4, num_post_sab=0, ln=False,
        arena_scale=10.0, weight_channel=False, pretrained_st_model_path=None, st_frozen=False,
        readout="flatten", num_pma_seeds=5, input_embed=True))
    assert (kw["readout"], kw["num_pma_seeds"], kw["input_embed"]) == ("flatten", 5, True)
