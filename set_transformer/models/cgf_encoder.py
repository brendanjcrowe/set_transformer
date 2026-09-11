"""CGF set encoder for the benchmark and its autoencoder -- a thin adapter over the arm kernel.

The empirical cumulant generating function of a particle set at ``num_t`` probes,
projected to a fixed-width code:

    K(t_m) = log( sum_i w_i exp(t_m . x_i) )

Where the CGF is evaluated decides what it measures. Writing ``t = s*u`` for a unit ``u``:
small ``s`` makes the Taylor jet at the origin the cumulants (a reparameterisation of the
k-moments baseline), while large ``s`` drives ``logsumexp`` toward ``max_i (u . x_i)`` --
the support function of the set, i.e. the PointNet max-pool regime.

**There is one CGF in this repository (2026-09-11).** The probes' init, the mass rule and
the log-MGF itself come from :mod:`set_transformer.rl.feature_extractors.cgf`
(:func:`init_t_values`, :func:`cgf_log_mgf`), the implementation the
``experiments/{ant_tag,odd_even}`` arms train with. This class adds only what the
benchmark's contract needs on top: a per-env ``particle_scale``, uniform weights when
the wrapper supplies none, the legacy elementwise ``t_clamp``, and the learned
``Linear(num_t -> num_outputs * dim_output)`` readout that matches every benchmark
method's bottleneck. The previous standalone copy (a July 2026 port of the arm
extractor) additionally clamped ``t . x`` to ``+-exp_arg_clamp = 20``; that guard is
unnecessary under logsumexp and made K wrong once ``||t||`` passed ~7 -- by 5 to 78
nats at the probe bounds PITFALLS sec. 9 prescribes for Ant-Tag (9 / 13.5 / 35). At the
legacy bound 2.0 the two implementations agree to float32 precision, so every
recorded benchmark run is unaffected (``tests/test_cgf_encoder_matches_legacy.py``).

Constructor signature and ``state_dict`` keys (``t_values``, ``cgf_proj.*``) are
unchanged, so existing checkpoints load. ``exp_arg_clamp`` is still accepted and
recorded, and ignored.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from set_transformer.cgf_kernel import cgf_log_mgf, init_t_values


class CGFEncoder(nn.Module):
    """Particle set ``[B, N, d]`` -> code ``[B, num_outputs, dim_output]``.

    The output is reshaped to the same rank as the other set encoders in this repo
    (Set Transformer, DeepSet, PointNet) so a single ``PFDecoder`` and a single alignment
    loss work against all of them unchanged.
    """

    def __init__(
        self,
        dim_input: int,
        num_outputs: int = 8,
        dim_output: int = 2,
        num_t: int = 64,
        t_init_mode: str = "spread",
        t_init_scale: float = 0.1,
        t_clamp: float | None = 2.0,
        exp_arg_clamp: float | None = None,
        t_frozen: bool = False,
        particle_scale: float = 1.0,
        **_,
    ) -> None:
        super().__init__()
        if num_t < 1:
            raise ValueError(f"num_t must be >= 1, got {num_t}.")
        if particle_scale <= 0:
            raise ValueError(f"particle_scale must be > 0, got {particle_scale}.")
        self.particle_dim = dim_input
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.stat_dim = num_outputs * dim_output
        self.num_t = num_t
        self.t_init_mode = t_init_mode
        self.t_init_scale = t_init_scale
        self.t_clamp = t_clamp
        #: Accepted for old configs and the run record; not applied (see module doc).
        self.exp_arg_clamp = exp_arg_clamp
        self.t_frozen = bool(t_frozen)
        self.particle_scale = particle_scale

        # The arm kernel spells the 1-D case ``spread_1d``; the benchmark has always
        # accepted ``spread`` for 1-D particles with geomspace(0.25, 2.8) magnitudes
        # mirrored into both signs, which is exactly ``spread_1d`` at that ceiling.
        mode, t_init_max = t_init_mode, None
        if mode == "spread" and dim_input == 1:
            mode, t_init_max = "spread_1d", 2.8
        t_values = init_t_values(num_t, dim_input, mode,
                                 t_init_scale=t_init_scale, t_init_max=t_init_max)
        if self.t_frozen:
            # A buffer takes no gradient (so no optimizer can move it) while still
            # saving/loading and moving across devices exactly like a Parameter.
            self.register_buffer("t_values", t_values)
        else:
            self.t_values = nn.Parameter(t_values)
        # Learned readout of the sampled CGF curve. This decouples sampling resolution
        # (num_t) from the bottleneck width the consumer sees (stat_dim).
        self.cgf_proj = nn.Identity() if self.num_t == self.stat_dim \
            else nn.Linear(self.num_t, self.stat_dim)

    def effective_t(self) -> torch.Tensor:
        """The ``[num_t, d]`` probe matrix ``forward`` uses: ``t_values`` under the clamp."""
        t = self.t_values
        if self.t_clamp is not None:
            t = torch.clamp(t, -self.t_clamp, self.t_clamp)
        return t

    def forward(self, particles: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        particles = particles / self.particle_scale
        if weights is None:
            batch, n = particles.shape[0], particles.shape[1]
            weights = particles.new_full((batch, n), 1.0 / n)
        cgf = cgf_log_mgf(particles, weights, self.effective_t())        # [B, num_t]
        code = self.cgf_proj(cgf)
        return code.reshape(code.shape[0], self.num_outputs, self.dim_output)

    @torch.no_grad()
    def t_value_norms(self) -> torch.Tensor:
        """Per-point L2 norms ``||t_m||``, POST-clamp -- what ``forward`` actually uses."""
        return torch.linalg.norm(self.effective_t().detach(), dim=1)
