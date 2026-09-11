"""CGFEncoder is now an adapter over the arm kernel; its outputs must not have moved.

On 2026-09-11 `models/cgf_encoder.py` stopped carrying its own CGF (a July 2026 port of
the arm extractor) and became: scale, uniform-if-no-weights, clamp t, call
`rl.feature_extractors.cgf.cgf_log_mgf`, project. Everything the benchmark recorded was
computed by the old code at the legacy bound `t_clamp = 2.0`, where the old
`exp_arg_clamp = 20` never fired -- so on that domain the new encoder must reproduce the
old one to float32, weights or not, 1-D or 2-D, at every `particle_scale` the registry
uses. Outside it (||t|| >~ 7) the old code was wrong and the new one is exact; the last
test pins that the difference is the intended one and not a regression.

`_legacy_forward` is the old forward, verbatim from 638c74d, so the reference cannot
drift with the code under test.
"""

import numpy as np
import pytest
import torch

from set_transformer.models import CGFAutoencoder, CGFEncoder

B, N = 5, 100
SCALES = (1.0, 4.5, 10.0, 14.0)      # registry: car_flag / ant_tag / odd_even / msearch


def _legacy_forward(enc: CGFEncoder, particles, weights=None, exp_arg_clamp=20.0):
    """`CGFEncoder.forward` as it stood before 2026-09-11 (638c74d), on the module's
    own t_values / cgf_proj so only the kernel differs."""
    particles = particles / enc.particle_scale
    particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)
    t = enc.t_values
    if enc.t_clamp is not None:
        t = torch.clamp(t, -enc.t_clamp, enc.t_clamp)
    exp_arg = torch.einsum("md,bnd->bmn", t, particles)
    exp_arg = torch.clamp(exp_arg, -exp_arg_clamp, exp_arg_clamp)
    if weights is None:
        n = particles.shape[1]
        cgf = torch.logsumexp(exp_arg, dim=2) - float(np.log(n))
    else:
        w = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        cgf = torch.logsumexp(exp_arg + torch.log(w + 1e-20).unsqueeze(1), dim=2)
    code = enc.cgf_proj(cgf)
    return code.reshape(code.shape[0], enc.num_outputs, enc.dim_output)


def _particles(d, seed=0, spread=3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, d, generator=g) * spread


def _skewed_weights(seed=1):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(B, N, generator=g)
    w[:, N // 2:] *= 0.01                     # half the set near-dead, ESS ~ 50
    w[0, 7] = 0.0                             # an exactly refuted particle
    return w / w.sum(dim=1, keepdim=True)


@pytest.mark.parametrize("d", [1, 2])
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("weights", ["none", "uniform", "skewed"])
def test_new_encoder_reproduces_legacy_at_the_legacy_bound(d, scale, weights):
    torch.manual_seed(0)
    enc = CGFEncoder(dim_input=d, num_outputs=8, dim_output=2, num_t=64,
                     t_clamp=2.0, particle_scale=scale, t_init_mode="spread")
    x = _particles(d, spread=scale)           # particles fill the arena
    w = {"none": None, "uniform": torch.full((B, N), 1.0 / N),
         "skewed": _skewed_weights()}[weights]
    new = enc(x, w)
    old = _legacy_forward(enc, x, w)
    assert new.shape == old.shape == (B, 8, 2)
    assert torch.allclose(new, old, atol=2e-5, rtol=0.0), (new - old).abs().max().item()


def test_autoencoder_round_trip_is_unchanged():
    """Same encoder inside CGFAutoencoder: decoder output identical to the legacy path."""
    torch.manual_seed(0)
    ae = CGFAutoencoder(num_particles=N, dim_particles=2, num_encodings=8, dim_encoder=2,
                        dim_hidden=128, num_t=64, t_init_mode="spread", particle_scale=4.5)
    x = _particles(2, spread=4.5)
    with torch.no_grad():
        new = ae(x)
        old = ae.decoder(_legacy_forward(ae.encoder, x, None))
    assert torch.allclose(new, old, atol=1e-4, rtol=0.0)


def test_legacy_checkpoints_still_load():
    """Constructor and state_dict keys are the contract old .pt files were saved under."""
    torch.manual_seed(0)
    enc = CGFEncoder(dim_input=2, num_outputs=8, dim_output=2, num_t=64)
    assert set(enc.state_dict()) == {"t_values", "cgf_proj.weight", "cgf_proj.bias"}
    legacy = {"t_values": torch.randn(64, 2), "cgf_proj.weight": torch.randn(16, 64),
              "cgf_proj.bias": torch.randn(16)}
    enc.load_state_dict(legacy)               # strict: no missing / unexpected keys
    assert torch.equal(enc.t_values.detach(), legacy["t_values"])
    frozen = CGFEncoder(dim_input=2, num_outputs=8, dim_output=2, num_t=64, t_frozen=True)
    frozen.load_state_dict(legacy)
    assert not any(p is frozen.t_values for p in frozen.parameters())
    # the removed guard is still accepted by old configs
    CGFEncoder(dim_input=2, exp_arg_clamp=20.0)


def test_spread_init_unchanged_in_both_dimensions():
    """8 dirs x geomspace(0.25, 2.8, 8) in 2-D; +-geomspace(0.25, 2.8, 32) in 1-D."""
    e2 = CGFEncoder(dim_input=2, num_t=64, t_init_mode="spread")
    norms = torch.linalg.norm(e2.t_values.detach(), dim=1)
    assert torch.allclose(torch.unique(norms.round(decimals=4)),
                          torch.tensor(np.geomspace(0.25, 2.8, 8), dtype=torch.float32).round(decimals=4))
    e1 = CGFEncoder(dim_input=1, num_t=64, t_init_mode="spread")
    expected = torch.tensor(np.geomspace(0.25, 2.8, 32), dtype=torch.float32)
    assert torch.allclose(e1.t_values.detach().reshape(-1), torch.cat([expected, -expected]))


@pytest.mark.parametrize("bound", [9.0, 13.5, 35.0])   # variants.cgf_t_bound
def test_outside_the_legacy_domain_new_is_exact_and_legacy_was_not(bound):
    """The intended divergence. Probes at the PITFALLS sec. 9 bounds: the old exp_arg
    clamp corrupted K by whole nats; the kernel path matches a float64 reference."""
    torch.manual_seed(0)
    enc = CGFEncoder(dim_input=2, num_outputs=8, dim_output=8, num_t=64,
                     t_clamp=None, particle_scale=4.5)          # dim_output=8 -> Identity proj
    with torch.no_grad():
        t = torch.randn(64, 2)
        enc.t_values.copy_(bound * t / t.norm(dim=1, keepdim=True))
    x, w = _particles(2, spread=3.0), _skewed_weights()
    scores = torch.einsum("md,bnd->bmn", enc.t_values.double(), (x / 4.5).double())
    exact = torch.logsumexp(scores + torch.log(w.double()).unsqueeze(1), dim=2)
    new = enc(x, w).reshape(B, -1).double()
    old = _legacy_forward(enc, x, w).reshape(B, -1).double()
    assert (new - exact).abs().max() < 1e-4
    assert (old - exact).abs().max() > 1.0
