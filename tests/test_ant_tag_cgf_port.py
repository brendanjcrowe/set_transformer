"""The 2026-09-10 Ant-Tag CGF port: polar t, the registry bound, flag plumbing.

What this pins (change_mds/ant_tag_cgf_port_2026-09-10.md):

* ``t_param="polar"``: reproduces the spread init exactly, stays inside a
  BALL of radius t_bound, gives direction and magnitude separate gradients,
  freezes to buffers, is refused on 1-D particles, round-trips through the
  kwargs SB3 stores, and its K / K' agree with a float64 reference.
* Clamp mode refuses a ``t_init_max`` above the clamp, and the legacy 2-D
  ``spread`` init at clamp 2.0 really does flatten 4 of 64 probes (the
  behaviour every recorded Ant-Tag `spread` run had).
* ``variants.cgf_t_bound`` = 3 / (tag_radius / arena half-width) per variant.
* ``4_train_rl_cgf.py``'s argparse path: legacy defaults unchanged, the
  registry bound and init ceiling fill in for polar / tanh, contradictions
  are parser errors, ``arena_scale`` and ``encoder_params`` land in
  run_config, a checkpoint's geometry is taken and its frame checked.

The legacy numerics stay pinned by test_ant_tag_shared_pieces_regression.py.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from set_transformer.rl.feature_extractors.cgf import (
    EncoderDriftLoggingCallback,
    WeightedCGFFeaturesExtractor,
)

_ST_ROOT = Path(__file__).resolve().parents[1]
_ANT_TAG_DIR = _ST_ROOT / "experiments" / "ant_tag"
_MODULE_NAMES = ("variants", "4_train_rl_frozen", "4_train_rl_cgf")

N_PARTICLES, OBS_DIM, SCALE = 100, 31, 4.5


# ---------------------------------------------------------------- fixtures

@contextlib.contextmanager
def _sys_path(directory: Path):
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(str(directory))


@pytest.fixture(scope="module")
def ant_tag():
    """The Ant-Tag modules under their flat names, dropped again at teardown
    so `variants` cannot leak into the Odd-Even tests (Gap 12)."""
    preexisting = set(sys.modules)
    with _sys_path(_ANT_TAG_DIR):
        modules = {name: importlib.import_module(name) for name in _MODULE_NAMES}
    try:
        yield modules
    finally:
        for name in set(sys.modules) - preexisting:
            module = sys.modules.get(name)
            file = getattr(module, "__file__", None) or ""
            if str(_ST_ROOT / "experiments") in file or name in _MODULE_NAMES:
                sys.modules.pop(name, None)


def _space(particle_dim=2):
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (N_PARTICLES, particle_dim), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (N_PARTICLES,), np.float32),
    })


def _batch(seed=0, b=3):
    g = torch.Generator().manual_seed(seed)
    particles = (torch.rand(b, N_PARTICLES, 2, generator=g) * 2 - 1) * SCALE
    weights = torch.rand(b, N_PARTICLES, generator=g)
    weights = weights / weights.sum(dim=1, keepdim=True)
    return {"obs": torch.randn(b, OBS_DIM, generator=g),
            "particles": particles, "weights": weights}


def _spread_grid(rho_hi: float, num=64) -> torch.Tensor:
    """The 'spread' init as the extractor defines it: 8 directions x log-spaced norms."""
    angles = torch.arange(8, dtype=torch.float32) * (2 * torch.pi / 8)
    dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    norms = torch.tensor(np.geomspace(0.25, rho_hi, num // 8), dtype=torch.float32)
    return (norms[None, :, None] * dirs[:, None, :]).reshape(-1, 2)


def _polar(t_bound=13.5, t_init_max=10.8, **kw):
    return WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread",
        t_param="polar", t_bound=t_bound, t_init_max=t_init_max, **kw)


def _reference(batch, t, mode):
    x = batch["particles"].double() / SCALE
    w = batch["weights"].double()
    scores = torch.einsum("bnd,td->bnt", x, t.double()) + torch.log(w)[:, :, None]
    k = torch.logsumexp(scores, dim=1)
    kgrad = torch.einsum("bnt,bnd->btd", torch.softmax(scores, dim=1), x).reshape(x.shape[0], -1)
    return {"K": k, "K_grad": kgrad, "both": torch.cat([k, kgrad], dim=-1)}[mode]


# ------------------------------------------------------------------- polar

def test_polar_reproduces_the_spread_init_exactly():
    ext = _polar()
    torch.testing.assert_close(ext.effective_t(), _spread_grid(10.8), rtol=1e-4, atol=1e-5)
    assert set(ext.state_dict()) == {"raw_v", "raw_a"}
    assert isinstance(ext.raw_v, torch.nn.Parameter) and isinstance(ext.raw_a, torch.nn.Parameter)


def test_polar_is_a_ball_not_a_box():
    ext = _polar()
    with torch.no_grad():
        ext.raw_a.fill_(50.0)                       # magnitude saturated
        ext.raw_v.copy_(torch.randn_like(ext.raw_v) * 7)
    norms = torch.linalg.norm(ext.effective_t(), dim=1).detach()
    # sigmoid(50) is exactly 1.0 in float32, so the norm lands ON the bound.
    assert float(norms.max()) <= 13.5 * (1 + 1e-6) and float(norms.min()) >= 13.5 * (1 - 1e-6)
    # tanh at the same bound reaches t_bound * sqrt(2) on the diagonal
    box = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread",
        t_param="tanh", t_bound=13.5, t_init_max=10.8)
    with torch.no_grad():
        box.raw_t.fill_(50.0)
    assert float(torch.linalg.norm(box.effective_t(), dim=1).detach().max()) > 13.5 * 1.4


def test_polar_direction_and_magnitude_get_separate_gradients():
    ext = _polar()
    out = ext(_batch())[:, OBS_DIM:]
    out.sum().backward()
    assert ext.raw_a.grad is not None and float(ext.raw_a.grad.abs().max()) > 0
    assert ext.raw_v.grad is not None and float(ext.raw_v.grad.abs().max()) > 0
    # The direction gradient lives on the unit sphere: orthogonal to v row-wise,
    # so the magnitude cannot leak into the direction parameter.
    radial = (ext.raw_v.grad * ext.raw_v.detach()).sum(dim=1)
    assert float(radial.abs().max()) < 1e-5 * float(ext.raw_v.grad.abs().max())


def test_polar_magnitude_gradient_scales_with_r():
    """dr/da = r (1 - r/T): small probes move multiplicatively, near-bound ones slow."""
    ext = _polar()
    r = torch.linalg.norm(ext.effective_t(), dim=1)
    r.sum().backward()
    expected = (r * (1 - r / 13.5)).detach()
    torch.testing.assert_close(ext.raw_a.grad, expected, rtol=1e-4, atol=1e-6)


def test_polar_frozen_registers_buffers_and_forward_has_no_grad():
    ext = _polar(t_frozen=True)
    assert set(dict(ext.named_buffers())) == {"raw_v", "raw_a"}
    assert list(ext.parameters()) == []
    torch.testing.assert_close(ext.effective_t(), _spread_grid(10.8), rtol=1e-4, atol=1e-5)


def test_polar_is_refused_on_1d_particles_and_at_the_bound():
    with pytest.raises(ValueError, match="particle_dim >= 2"):
        WeightedCGFFeaturesExtractor(
            _space(particle_dim=1), num_cgf_features=8, arena_scale=SCALE,
            t_init_mode="spread_1d", t_param="polar", t_bound=50.0, t_init_max=40.0)
    with pytest.raises(ValueError, match="sigmoid's flat region"):
        _polar(t_bound=10.0, t_init_max=10.0)
    with pytest.raises(ValueError, match="positive t_bound"):
        WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread",
            t_param="polar")


def test_polar_handles_a_zero_init_row():
    """linspace_all_dims places one probe at t = 0 (65 features -> odd linspace)."""
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=65, arena_scale=SCALE, t_init_mode="linspace_all_dims",
        t_init_scale=1.0, t_param="polar", t_bound=13.5)
    t = ext.effective_t()
    assert torch.isfinite(t).all()
    assert float(torch.linalg.norm(t, dim=1).min()) < 1e-3   # tiny, not NaN


@pytest.mark.parametrize("mode", ["K", "K_grad", "both"])
def test_polar_features_match_a_float64_reference(mode):
    ext = _polar(feature_mode=mode)
    batch = _batch()
    with torch.no_grad():
        out = ext(batch)[:, OBS_DIM:].double()
    torch.testing.assert_close(out, _reference(batch, ext.effective_t().detach(), mode),
                               rtol=1e-4, atol=1e-5)


def test_polar_kwargs_round_trip_the_way_sb3_reloads_them():
    kwargs = dict(num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread",
                  t_param="polar", t_bound=13.5, t_init_max=10.8, feature_mode="K_grad")
    a = WeightedCGFFeaturesExtractor(_space(), **kwargs)
    with torch.no_grad():
        a.raw_v.add_(torch.randn_like(a.raw_v) * 0.3)
        a.raw_a.add_(torch.randn_like(a.raw_a) * 0.3)
    b = WeightedCGFFeaturesExtractor(_space(), **kwargs)
    b.load_state_dict(a.state_dict(), strict=True)
    batch = _batch(seed=1)
    with torch.no_grad():
        torch.testing.assert_close(a(batch), b(batch))
    assert a._cgf_geometry["t_param"] == "polar" and a._cgf_geometry["t_bound"] == 13.5


def test_drift_callback_groups_polar_parameters_separately():
    groups = EncoderDriftLoggingCallback._groups(_polar())
    assert set(groups) == {"raw_v", "raw_a"}


# -------------------------------------------------------- clamp-mode facts

def test_clamp_mode_refuses_t_init_max_above_the_clamp():
    with pytest.raises(ValueError, match="exceeds t_clamp"):
        WeightedCGFFeaturesExtractor(
            _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread",
            t_param="clamp", t_clamp=2.0, t_init_max=5.0)
    # At or below the clamp is fine (the Odd-Even clamp arms pass t_init_max == t_clamp).
    WeightedCGFFeaturesExtractor(
        _space(particle_dim=1), num_cgf_features=8, arena_scale=SCALE,
        t_init_mode="spread_1d", t_param="clamp", t_clamp=2.0, t_init_max=2.0)


@pytest.mark.parametrize("t_frozen", [False, True])
def test_legacy_spread_at_clamp_two_flattens_the_four_axis_probes(t_frozen):
    """The behaviour every recorded Ant-Tag `spread` run had: the norm-2.8
    probes on the 4 axis directions have a component of 2.8 and are clamped
    to 2.0 in the forward pass (zero gradient there when learned)."""
    ext = WeightedCGFFeaturesExtractor(
        _space(), num_cgf_features=64, arena_scale=SCALE, t_init_mode="spread",
        t_frozen=t_frozen)
    raw = ext.t_values.detach()
    eff = ext.effective_t().detach()
    changed = (raw - eff).abs().sum(dim=1) > 1e-6
    assert int(changed.sum()) == 4
    assert torch.allclose(torch.linalg.norm(raw[changed], dim=1), torch.full((4,), 2.8), atol=1e-5)
    assert torch.allclose(torch.linalg.norm(eff[changed], dim=1), torch.full((4,), 2.0), atol=1e-5)
    # axis-aligned: one component is (numerically) zero
    assert bool((raw[changed].abs().min(dim=1).values < 1e-5).all())
    if not t_frozen:
        ext.effective_t().sum().backward()
        assert float(ext.t_values.grad[changed].abs().max(dim=1).values.min()) == 0.0 or \
            bool((ext.t_values.grad[changed].abs() < 1e-12).any(dim=1).all())


# ------------------------------------------------------------- registry

@pytest.mark.parametrize("name,scale,bound", [
    ("smart", 4.5, 9.0), ("smart_mid_slow_v15", 4.5, 13.5), ("cdens_terminal", 7.0, 35.0),
])
def test_registry_bound_is_three_over_normalised_tag_radius(ant_tag, name, scale, bound):
    variants = ant_tag["variants"]
    assert variants.arena_scale(name) == pytest.approx(scale)
    assert variants.cgf_t_bound(name) == pytest.approx(bound)
    assert variants.CGF_TILT_TARGET == 3.0


# -------------------------------------------------------- argparse plumbing

def _drive_main(ant_tag, monkeypatch, tmp_path, argv):
    """The script's main() through argparse; the run record and the shared train() call
    (the script is an entry point of set_transformer.rl.train since change 4.5), with the
    extractor kwargs at top level as the script's own train_ant_tag_cgf took them."""
    from set_transformer.rl import run_records
    from set_transformer.rl import train as train_mod
    module = ant_tag["4_train_rl_cgf"]
    captured = {}
    monkeypatch.setattr(run_records, "default_run_dir", lambda *a, **k: str(tmp_path / "run"))
    monkeypatch.setattr(run_records, "git_provenance", lambda: {})
    monkeypatch.setattr(run_records, "tee_stdout_stderr", lambda path: None)
    monkeypatch.setattr(run_records, "write_run_config",
                        lambda run_dir, **cfg: captured.setdefault("config", cfg))
    monkeypatch.setattr(train_mod, "train",
                        lambda domain, variant, encoder, **kw: captured.setdefault(
                            "train", {**kw, **kw["features_extractor_kwargs"],
                                      **(kw.get("encoder_options") or {})}))
    module.main(list(argv))
    return captured["config"], captured["train"]


_PORTED = ("t_param", "t_bound", "t_init_max", "feature_mode", "feature_norm",
           "running_norm_update", "readout_hidden", "readout_depth", "readout_dim",
           "pretrained_cgf_model_path", "cgf_frozen", "x_embed_dim", "x_embed_hidden",
           "x_embed_depth")


def _extractor_from_train_kwargs(train):
    """Build the extractor exactly as the shared trainer's policy_kwargs would."""
    return WeightedCGFFeaturesExtractor(
        _space(),
        num_cgf_features=train["num_cgf_features"], arena_scale=train["arena_scale"],
        t_init_mode=train["t_init_mode"], t_init_scale=train["t_init_scale"],
        t_clamp=train["t_clamp"], exp_arg_clamp=train["exp_arg_clamp"],
        t_frozen=train["t_frozen"], t_param=train["t_param"],
        t_bound=train["t_bound"] if train["t_param"] != "clamp" else None,
        t_init_max=train["t_init_max"], feature_mode=train["feature_mode"],
        feature_norm=train["feature_norm"], readout_hidden=train["readout_hidden"],
        readout_depth=train["readout_depth"], readout_dim=train["readout_dim"],
        pretrained_cgf_model_path=train["pretrained_cgf_model_path"],
        cgf_frozen=train["cgf_frozen"], x_embed_dim=train["x_embed_dim"],
        x_embed_hidden=train["x_embed_hidden"], x_embed_depth=train["x_embed_depth"])


def test_bare_run_keeps_every_legacy_default_and_records_arena_scale(ant_tag, monkeypatch, tmp_path):
    config, train = _drive_main(ant_tag, monkeypatch, tmp_path, ["--variant", "smart_mid_slow_v15"])
    assert all(k in train for k in _PORTED)
    assert train["t_param"] == "clamp" and train["t_bound"] is None
    assert train["t_init_mode"] == "linspace_all_dims" and train["t_init_max"] is None
    assert train["feature_mode"] == "K" and train["feature_norm"] == "none"
    assert train["readout_depth"] == 0 and train["x_embed_dim"] == 0
    assert train["pretrained_cgf_model_path"] is None and train["cgf_frozen"] is False
    # run_config.json: a number, not None; and the encoder size on record.
    assert config["arena_scale"] == pytest.approx(4.5) and train["arena_scale"] == pytest.approx(4.5)
    assert config["encoder_params"] == 64 * 2
    ext = _extractor_from_train_kwargs(train)
    assert ext.features_dim == OBS_DIM + 64 and set(ext.state_dict()) == {"t_values"}


def test_legacy_spread_keeps_the_implicit_ceiling_and_its_clamped_probes(ant_tag, monkeypatch, tmp_path):
    """A bare `--t_init_mode spread` must reproduce the recorded runs: t_init_max
    stays unset (the extractor's built-in 2.8) and the 4 axis probes are clamped."""
    config, train = _drive_main(ant_tag, monkeypatch, tmp_path,
                                ["--variant", "smart_hard", "--t_init_mode", "spread"])
    assert train["t_init_max"] is None and train["t_param"] == "clamp"
    assert config["t_init_max"] is None
    ext = _extractor_from_train_kwargs(train)
    changed = (ext.t_values.detach() - ext.effective_t().detach()).abs().sum(dim=1) > 1e-6
    assert int(changed.sum()) == 4


def test_polar_recipe_takes_bound_and_ceiling_from_the_registry(ant_tag, monkeypatch, tmp_path):
    config, train = _drive_main(ant_tag, monkeypatch, tmp_path,
                                ["--variant", "smart_mid_slow_v15", "--t_param", "polar",
                                 "--t_init_mode", "spread", "--feature_mode", "K_grad"])
    assert train["t_bound"] == pytest.approx(13.5)
    assert train["t_init_max"] == pytest.approx(0.8 * 13.5)
    assert config["t_bound"] == pytest.approx(13.5) and config["t_init_max"] == pytest.approx(10.8)
    assert config["encoder_params"] == 64 * 2 + 64          # v (64 x 2) + a (64)
    ext = _extractor_from_train_kwargs(train)
    assert ext.features_dim == OBS_DIM + 128
    torch.testing.assert_close(ext.effective_t(), _spread_grid(10.8), rtol=1e-4, atol=1e-5)


def test_explicit_bound_wins_and_tanh_uses_the_same_rule(ant_tag, monkeypatch, tmp_path):
    _, train = _drive_main(ant_tag, monkeypatch, tmp_path,
                           ["--variant", "smart", "--t_param", "tanh", "--t_init_mode", "spread",
                            "--t_bound", "20"])
    assert train["t_bound"] == 20.0 and train["t_init_max"] == pytest.approx(16.0)
    _, train = _drive_main(ant_tag, monkeypatch, tmp_path,
                           ["--variant", "smart", "--t_param", "tanh", "--t_init_mode", "spread"])
    assert train["t_bound"] == pytest.approx(9.0)


@pytest.mark.parametrize("argv", [
    ["--t_param", "clamp", "--t_init_mode", "spread", "--t_init_max", "5"],   # above the clamp
    ["--t_param", "polar", "--t_init_mode", "spread", "--t_bound", "10", "--t_init_max", "10"],
    ["--cgf_frozen"],                                                           # no checkpoint
    ["--readout_hidden", "64"],                                                 # depth missing
])
def test_contradictions_are_parser_errors(ant_tag, monkeypatch, tmp_path, argv):
    with pytest.raises((SystemExit, ValueError)):
        config, train = _drive_main(ant_tag, monkeypatch, tmp_path,
                                    ["--variant", "smart_mid_slow_v15"] + argv)
        _extractor_from_train_kwargs(train)


def test_match_params_sizes_a_readout(ant_tag, monkeypatch, tmp_path):
    config, train = _drive_main(ant_tag, monkeypatch, tmp_path,
                                ["--variant", "smart_mid_slow_v15", "--t_param", "polar",
                                 "--t_init_mode", "spread", "--feature_mode", "K_grad",
                                 "--match_params", "109448"])
    assert train["readout_depth"] == 2 and train["readout_hidden"] > 0
    assert abs(config["encoder_params"] - 109448) / 109448 < 0.01
    ext = _extractor_from_train_kwargs(train)
    assert ext.features_dim == OBS_DIM + 64
    assert ext.encoder_parameter_count() == config["encoder_params"]


def _write_checkpoint(path, **kwargs):
    ext = WeightedCGFFeaturesExtractor(_space(), **kwargs)
    with torch.no_grad():
        for p in ext.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    torch.save({"model_state_dict": ext.state_dict(), "config": dict(ext._cgf_geometry)}, path)
    return ext


def test_checkpoint_geometry_is_taken_and_its_frame_checked(ant_tag, monkeypatch, tmp_path):
    ckpt = tmp_path / "cgf.pt"
    ref = _write_checkpoint(ckpt, num_cgf_features=32, arena_scale=4.5, t_init_mode="spread",
                            t_param="polar", t_bound=13.5, t_init_max=10.8, feature_mode="K_grad",
                            readout_hidden=50, readout_depth=2)
    config, train = _drive_main(ant_tag, monkeypatch, tmp_path,
                                ["--variant", "smart_mid_slow_v15",
                                 "--pretrained_cgf_model_path", str(ckpt), "--cgf_frozen"])
    for key in ("num_cgf_features", "t_param", "t_bound", "feature_mode",
                "readout_hidden", "readout_depth", "t_init_max"):
        assert train[key] == ref._cgf_geometry[key], key
    assert train["cgf_frozen"] is True
    ext = _extractor_from_train_kwargs(train)          # loads + freezes
    assert list(ext.parameters()) == [] or all(not p.requires_grad for p in ext.parameters())
    for key, value in ref.state_dict().items():
        torch.testing.assert_close(ext.state_dict()[key], value)
    # an explicit disagreement is an error (a value EQUAL to the parser default
    # is indistinguishable from "not given" and takes the checkpoint's -- the
    # Odd-Even rule; so the disagreeing value must not be the default 64)
    with pytest.raises(SystemExit):
        _drive_main(ant_tag, monkeypatch, tmp_path,
                    ["--variant", "smart_mid_slow_v15", "--pretrained_cgf_model_path", str(ckpt),
                     "--num_cgf_features", "48"])
    # and so is a checkpoint fitted in another frame (cdens_terminal's arena is 7.0)
    with pytest.raises(SystemExit):
        _drive_main(ant_tag, monkeypatch, tmp_path,
                    ["--variant", "cdens_terminal", "--pretrained_cgf_model_path", str(ckpt)])


def test_reload_helper_lives_in_the_package_and_the_odd_even_shim_re_exports_it():
    from set_transformer.rl import pretrained_encoder as pkg
    spec = importlib.util.spec_from_file_location(
        "odd_even_pretrained_encoder_shim",
        _ST_ROOT / "experiments" / "odd_even" / "pretrained_encoder.py")
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    assert shim.reload_pretrained_cgf is pkg.reload_pretrained_cgf
    assert shim.verify_matches_checkpoint is pkg.verify_matches_checkpoint
