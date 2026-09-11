"""The CGF kernel: probe init, mass rule and log-MGF, in one leaf module.

    K(t_j) = log sum_i w_i exp(<t_j, x_i>)

This is THE implementation of the CGF in the repository (2026-09-11). Both consumers
import from here and add only their own contract on top:

- :class:`set_transformer.rl.feature_extractors.cgf.WeightedCGFFeaturesExtractor` --
  the ``experiments/{ant_tag,odd_even}`` arms: polar / tanh / clamp probe bounds, K and
  K', feature standardisers, optional readout, weights required. That module re-exports
  every name here, and is where the history and the design notes live.
- :class:`set_transformer.models.CGFEncoder` -- the benchmark and its
  :class:`~set_transformer.models.CGFAutoencoder`: per-env ``particle_scale``, uniform
  weights when none are given, a ``Linear(num_t -> stat_dim)`` readout.

It is a leaf on purpose: it imports nothing from the package, so ``set_transformer.models``
can depend on it without importing ``set_transformer.rl`` (which imports ``models`` back).
``tests/test_cgf_kernel_parity.py`` pins that both consumers route through it.
"""

from __future__ import annotations

import math

import numpy as np
import torch
def init_t_values(num_t: int, particle_dim: int, t_init_mode: str, *,
                  t_init_scale: float = 0.1,
                  t_init_max: float | None = None) -> torch.Tensor:
    """Initial CGF probe matrix ``[num_t, particle_dim]`` for every ``t_init_mode``.

    Factored out of :class:`WeightedCGFFeaturesExtractor` (2026-09-11) so the
    benchmark's :class:`~set_transformer.models.CGFEncoder` starts its probes
    from the same rows -- one init, not two copies that agree until one is
    edited. ``spread`` is the 2-D geometry recorded in every Ant-Tag `spread`
    run (8 directions x geomspace(0.25, rho_hi)); ``spread_1d`` mirrors the
    magnitudes into both signs. Error messages are part of the contract:
    callers match on "divisible" and "Unknown t_init_mode".
    """
    if t_init_mode == "linspace_all_dims":
        # Round-robin the linspace directions across every particle
        # dimension so each coordinate (not just dim 0) gets a nontrivial
        # initial CGF sensitivity.
        t_values = torch.zeros(num_t, particle_dim)
        linspace_vals = torch.linspace(-t_init_scale, t_init_scale, num_t)
        dim_assignment = torch.arange(num_t) % particle_dim
        for d in range(particle_dim):
            mask = dim_assignment == d
            t_values[mask, d] = linspace_vals[mask]
    elif t_init_mode == "linspace_first_dim":
        t_values = torch.zeros(num_t, particle_dim)
        t_values[:, 0] = torch.linspace(
            -t_init_scale,
            t_init_scale,
            num_t,
        )
        if particle_dim > 1:
            t_values[:, 1:] = 0.01 * torch.randn(
                num_t,
                particle_dim - 1,
            )
    elif t_init_mode == "spread":
        # 8 directions x (num//8) log-spaced norms, matching the probe's
        # CGF_SPREAD64 feature geometry. rho_hi=2.8 ~ the max norm
        # reachable under the elementwise t_clamp=2.0 on the DIAGONAL
        # (2*sqrt(2)). On the 4 axis-aligned directions a norm-2.8 probe
        # has a component of 2.8, which clamp mode flattens to 2.0 in the
        # forward pass, so 4 of 64 probes sit on the clamp with zero
        # gradient and near-duplicate the norm-1.98 ring (measured
        # 2026-09-09; pinned by tests/test_ant_tag_cgf_port.py). Every
        # recorded Ant-Tag `spread` run has this; tanh / polar do not.
        # The den diagonal (45 deg) is one of the 8 directions exactly.
        # Unlike the legacy 0.1-scale linspace init, this starts where
        # the signal actually is, so no ~10x growth of ||t_j|| is
        # required for the encoder to see it.
        if particle_dim != 2:
            raise ValueError("t_init_mode='spread' assumes 2D particles")
        num_dirs = 8
        if num_t % num_dirs != 0:
            raise ValueError(
                "num_t must be divisible by 8 for 'spread'")
        num_norms = num_t // num_dirs
        angles = torch.arange(num_dirs, dtype=torch.float32) * (
            2 * torch.pi / num_dirs)
        dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        # t_init_max overrides the 2.8 (e.g. 0.8 * t_bound in tanh mode,
        # so the init spans the range the bound allows).
        rho_hi = 2.8 if t_init_max is None else float(t_init_max)
        norms = torch.tensor(np.geomspace(0.25, rho_hi, num_norms),
                             dtype=torch.float32)
        t_values = (norms[None, :, None] * dirs[:, None, :]).reshape(
            -1, particle_dim)
    elif t_init_mode == "spread_1d":
        # The 1-D analogue of "spread", for a scalar state space such as
        # the Odd-Even POMDP's integer hidden state. A SEPARATE mode
        # rather than a widening of "spread": that mode's geometry is
        # intrinsically planar (8 directions from angles 2*pi*k/8) and
        # its rho_hi=2.8 is the largest norm the ELEMENTWISE t_clamp=2.0
        # permits in 2-D (the diagonal, 2*sqrt(2)). In 1-D there are
        # exactly two directions and that same clamp bounds ||t|| at
        # 2.0, so neither number carries over. Overloading one flag name
        # across two unrelated geometries would also make run_config.json
        # ambiguous about what a run actually built, and would silently
        # change a mode that hundreds of saved 2-D checkpoints were
        # built with.
        if particle_dim != 1:
            raise ValueError(
                "t_init_mode='spread_1d' assumes 1D particles; use "
                "'spread' for the 2D case")
        if num_t % 2 != 0:
            raise ValueError(
                "num_t must be even for 'spread_1d' (the "
                "magnitudes are mirrored into both signs)")
        num_norms = num_t // 2
        # 2.0, not 2.8: in 1-D the elementwise clamp IS the norm bound.
        # t_init_max overrides it (e.g. 40 with t_bound=50 in tanh mode:
        # the log-spaced init then covers both the mean/variance regime
        # at small t and the support-edge regime at large t).
        rho_hi = 2.0 if t_init_max is None else float(t_init_max)
        norms = torch.tensor(np.geomspace(0.25, rho_hi, num_norms),
                             dtype=torch.float32)
        # Both signs, so the CGF sees the belief's mass on either side of
        # the origin. Like "spread", this starts where the signal already
        # is, so PPO needs no ~10x growth of ||t_j|| to resolve it.
        t_values = torch.cat([norms, -norms]).reshape(-1, particle_dim)
    elif t_init_mode == "random":
        t_values = t_init_scale * torch.randn(num_t, particle_dim)
    else:
        raise ValueError(f"Unknown t_init_mode: {t_init_mode}")
    return t_values


#: What a row with NO mass reports for every K feature: the old floor's value,
#: kept so dead beliefs look the same as they always did.
DEAD_ROW_VALUE = math.log(1e-8)


def normalize_log_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(log_w [B, N], dead [B, 1])`` from raw PF weights.

    NaN/inf -> 0, negatives -> 0, then normalised by the exact-sum rule with the
    load-bearing ``+ 1e-8`` (the regression gate pins its magnitude as
    ``near_epsilon_mass``; do not "tidy" it). ``log 0 = -inf`` for a refuted
    particle is correct and intended: ``exp(-inf)`` is exactly 0 inside
    logsumexp and so is its gradient. The one case that must not reach
    logsumexp is a row with no mass at all -- every entry -inf, backward 0/0 --
    so those rows get uniform stand-in weights (finite gradient) and are
    flagged in ``dead`` for the caller to overwrite the output.
    """
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, min=0.0)
    mass = weights.sum(dim=1, keepdim=True)                           # [B, 1]
    weights = weights / (mass + 1e-8)
    dead = mass <= 0.0                                                # [B, 1]
    num_particles = weights.shape[1]
    safe_weights = torch.where(
        dead, torch.full_like(weights, 1.0 / num_particles), weights)
    return safe_weights.log(), dead


def cgf_scores(particles: torch.Tensor, log_w: torch.Tensor,
               t: torch.Tensor) -> torch.Tensor:
    """``score[b, i, j] = <t_j, x_bi> + log w_bi`` -- the one tensor K and K' share."""
    return torch.matmul(particles, t.transpose(0, 1)) + log_w.unsqueeze(-1)


def log_mgf_from_scores(scores: torch.Tensor, dead: torch.Tensor,
                        dead_row_value: float = DEAD_ROW_VALUE) -> torch.Tensor:
    """``K(t_j) = logsumexp_i score[., i, j]``, dead rows replaced by the constant.

    The maximum is subtracted before anything is exponentiated, so no exponent
    can overflow or underflow and the exponent needs no clamp. The old
    ``exp_arg_clamp`` silently flattened every feature once |<t, x>| passed 20
    -- at |t| = 50 on [-1, 1] particles the two paths differed by 31.5
    (domain_mds/oddeven.md, 2026-09-05), and at the PITFALLS sec. 9 bounds on
    Ant-Tag (9 / 13.5 / 35) by 5 to 78 -- and is not applied anywhere any more.
    """
    k = torch.logsumexp(scores, dim=1)                                # [B, T]
    return torch.where(dead, torch.full_like(k, dead_row_value), k)


def cgf_log_mgf(particles: torch.Tensor, weights: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
    """The CGF kernel: ``K(t_j) = log sum_i w_i exp(<t_j, x_i>)`` for ``[B, N, D]``
    particles (already in the caller's normalised frame), ``[B, N]`` weights and
    ``[T, D]`` probes -> ``[B, T]``. This is THE implementation; the benchmark's
    :class:`~set_transformer.models.CGFEncoder` and the arm extractor's K
    feature both come from here (``tests/test_cgf_kernel_parity.py``)."""
    particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)
    log_w, dead = normalize_log_weights(weights)
    return log_mgf_from_scores(cgf_scores(particles, log_w, t), dead)

