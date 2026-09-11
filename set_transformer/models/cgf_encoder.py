"""Shared CGF set encoder — used by both the RL feature extractor and the autoencoder.

The empirical cumulant generating function of a particle set, evaluated at ``num_t``
sampling points and projected to a fixed-width code:

    K(t_m) = log( sum_i w_i exp(t_m . x_i) )

Where the CGF is evaluated decides what it measures. Writing ``t = s*u`` for a unit ``u``:
small ``s`` makes the Taylor jet at the origin the cumulants (a reparameterisation of the
k-moments baseline), while large ``s`` drives ``logsumexp`` toward ``max_i (u . x_i)`` --
the support function of the set, i.e. the PointNet max-pool regime.

This module exists so the CGF used for *pretraining* and the CGF used inside the *policy*
are the same code. A separately written copy in either place would silently produce
checkpoints whose weights no longer mean what the other expects.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class CGFEncoder(nn.Module):
    """Particle set ``[B, N, d]`` -> code ``[B, num_encodings, dim_encoder]``.

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
        exp_arg_clamp: float = 20.0,
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
        self.exp_arg_clamp = exp_arg_clamp
        self.t_frozen = bool(t_frozen)
        self.particle_scale = particle_scale

        t_values = self._init_t_values()
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

    def _init_t_values(self) -> torch.Tensor:
        """Initial CGF sampling points ``[num_t, d]``.

        ``spread`` starts where the signal is (log-spaced norms over evenly spread
        directions) rather than requiring ``||t||`` to grow ~10x from a small init before
        the encoder can see anything.
        """
        num_t, d, scale = self.num_t, self.particle_dim, self.t_init_scale
        mode = self.t_init_mode

        if mode == "spread":
            if d < 2:
                norms = torch.tensor(np.geomspace(0.25, 2.8, max(num_t // 2, 1)),
                                     dtype=torch.float32)
                signed = torch.cat([norms, -norms])[:num_t]
                return signed.reshape(num_t, 1)
            num_dirs = 8
            if num_t % num_dirs != 0:
                raise ValueError(
                    f"t_init_mode='spread' needs num_t divisible by {num_dirs}, got {num_t}")
            if d != 2:
                raise ValueError("t_init_mode='spread' assumes 1-D or 2-D particles")
            angles = torch.arange(num_dirs, dtype=torch.float32) * (2 * torch.pi / num_dirs)
            dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            norms = torch.tensor(np.geomspace(0.25, 2.8, num_t // num_dirs),
                                 dtype=torch.float32)
            return (norms[None, :, None] * dirs[:, None, :]).reshape(-1, d)

        if mode == "linspace_all_dims":
            t_values = torch.zeros(num_t, d)
            vals = torch.linspace(-scale, scale, num_t)
            assignment = torch.arange(num_t) % d
            for dim in range(d):
                mask = assignment == dim
                t_values[mask, dim] = vals[mask]
            return t_values

        if mode == "linspace_first_dim":
            t_values = torch.zeros(num_t, d)
            t_values[:, 0] = torch.linspace(-scale, scale, num_t)
            if d > 1:
                t_values[:, 1:] = 0.01 * torch.randn(num_t, d - 1)
            return t_values

        if mode == "random":
            return scale * torch.randn(num_t, d)

        raise ValueError(f"Unknown t_init_mode: {mode}")

    def forward(self, particles: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
        particles = particles / self.particle_scale
        particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)

        t = self.t_values
        if self.t_clamp is not None:
            t = torch.clamp(t, -self.t_clamp, self.t_clamp)
        exp_arg = torch.einsum("md,bnd->bmn", t, particles)
        exp_arg = torch.clamp(exp_arg, -self.exp_arg_clamp, self.exp_arg_clamp)

        if weights is None:
            n = particles.shape[1]
            cgf = torch.logsumexp(exp_arg, dim=2) - float(np.log(n))
        else:
            w = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            cgf = torch.logsumexp(exp_arg + torch.log(w + 1e-20).unsqueeze(1), dim=2)
        code = self.cgf_proj(cgf)
        return code.reshape(code.shape[0], self.num_outputs, self.dim_output)

    @torch.no_grad()
    def t_value_norms(self) -> torch.Tensor:
        """Per-point L2 norms ``||t_m||``, POST-clamp — what ``forward`` actually uses."""
        t = self.t_values.detach()
        if self.t_clamp is not None:
            t = torch.clamp(t, -self.t_clamp, self.t_clamp)
        return torch.linalg.norm(t, dim=1)
