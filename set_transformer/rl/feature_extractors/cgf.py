"""Weighted empirical-CGF belief encoder for SB3, plus its logging callback.

    CGF_j = log(sum_i w_i * exp(<t_j, x_i>))

where x_i is a belief particle (normalized by ``arena_scale``) and w_i is its
particle-filter weight. The t_j directions are learned under PPO by default.

**Domain-independent.** Nothing here knows about any particular POMDP: the
extractor reads the ``{"obs", "particles", "weights"}`` Dict observation
produced by
:class:`set_transformer.rl.wrappers.particle_filter.PFDictWithWeightsObservationWrapper`
and works for any particle dimension. Two init modes are dimension-specific:
``t_init_mode="spread"`` asserts 2-D particles and ``"spread_1d"`` asserts 1-D
ones. They are separate names on purpose -- see ``"spread_1d"`` below.

Moved out of ``experiments/ant_tag/4_train_rl_cgf.py`` so a second domain can
use it without importing an Ant-Tag script. That script still re-exports both
names: SB3 pickles a features-extractor CLASS into the saved zip by module
path, so an existing checkpoint loads via
``getattr(import_module("4_train_rl_cgf"), "WeightedCGFFeaturesExtractor")``
and the re-export is what keeps thousands of saved runs loadable.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TNormLoggingCallback(BaseCallback):
    """Log the distribution of ||t_j|| every rollout.

    The Twin-Den probe found CGF's advantage needs ||t_j|| ~ 10x the 0.1
    legacy init scale, so whether PPO actually grows t is a first-class
    experimental question. Quantiles land in TensorBoard as
    cgf/t_norm_q{0,25,50,75,90,100}. No-op (and free) for encoders without
    t_values, e.g. the Gaussian arm; a FLAT line is the expected, built-in
    sanity check for the frozen arm.
    """

    def _on_step(self) -> bool:  # required abstract method
        return True

    def _on_rollout_end(self) -> None:
        extractor = getattr(self.model.policy, "features_extractor", None)
        t = getattr(extractor, "t_values", None)
        if t is None:
            return
        with torch.no_grad():
            norms = torch.linalg.norm(t.detach(), dim=1).cpu().numpy()
        for q in (0, 25, 50, 75, 90, 100):
            self.logger.record(f"cgf/t_norm_q{q}",
                               float(np.percentile(norms, q)))


class WeightedCGFFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor for weighted empirical CGF particle features."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_cgf_features: int = 64,
        arena_scale: float = 4.5,
        t_init_mode: str = "linspace_first_dim",
        t_init_scale: float = 0.1,
        t_clamp: float = 2.0,
        exp_arg_clamp: float = 20.0,
        t_frozen: bool = False,
    ):
        obs_dim = observation_space["obs"].shape[0]
        particle_dim = observation_space["particles"].shape[1]
        super().__init__(observation_space, features_dim=obs_dim + num_cgf_features)

        self.num_cgf_features = num_cgf_features
        self.particle_dim = particle_dim
        self.arena_scale = arena_scale
        self.t_clamp = t_clamp
        self.exp_arg_clamp = exp_arg_clamp

        if t_init_mode == "linspace_all_dims":
            # Round-robin the linspace directions across every particle
            # dimension so each coordinate (not just dim 0) gets a nontrivial
            # initial CGF sensitivity.
            t_values = torch.zeros(num_cgf_features, particle_dim)
            linspace_vals = torch.linspace(-t_init_scale, t_init_scale, num_cgf_features)
            dim_assignment = torch.arange(num_cgf_features) % particle_dim
            for d in range(particle_dim):
                mask = dim_assignment == d
                t_values[mask, d] = linspace_vals[mask]
        elif t_init_mode == "linspace_first_dim":
            t_values = torch.zeros(num_cgf_features, particle_dim)
            t_values[:, 0] = torch.linspace(
                -t_init_scale,
                t_init_scale,
                num_cgf_features,
            )
            if particle_dim > 1:
                t_values[:, 1:] = 0.01 * torch.randn(
                    num_cgf_features,
                    particle_dim - 1,
                )
        elif t_init_mode == "spread":
            # 8 directions x (num//8) log-spaced norms, matching the probe's
            # CGF_SPREAD64 feature geometry. rho_hi=2.8 ~ the max norm
            # reachable under the elementwise t_clamp=2.0 (diagonal norm
            # 2*sqrt(2)). The den diagonal (45 deg) is one of the 8
            # directions exactly. Unlike the legacy 0.1-scale linspace init,
            # this starts where the signal actually is, so no ~10x growth of
            # ||t_j|| is required for the encoder to see it.
            if particle_dim != 2:
                raise ValueError("t_init_mode='spread' assumes 2D particles")
            num_dirs = 8
            if num_cgf_features % num_dirs != 0:
                raise ValueError(
                    "num_cgf_features must be divisible by 8 for 'spread'")
            num_norms = num_cgf_features // num_dirs
            angles = torch.arange(num_dirs, dtype=torch.float32) * (
                2 * torch.pi / num_dirs)
            dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            norms = torch.tensor(np.geomspace(0.25, 2.8, num_norms),
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
            if num_cgf_features % 2 != 0:
                raise ValueError(
                    "num_cgf_features must be even for 'spread_1d' (the "
                    "magnitudes are mirrored into both signs)")
            num_norms = num_cgf_features // 2
            # 2.0, not 2.8: in 1-D the elementwise clamp IS the norm bound.
            norms = torch.tensor(np.geomspace(0.25, 2.0, num_norms),
                                 dtype=torch.float32)
            # Both signs, so the CGF sees the belief's mass on either side of
            # the origin. Like "spread", this starts where the signal already
            # is, so PPO needs no ~10x growth of ||t_j|| to resolve it.
            t_values = torch.cat([norms, -norms]).reshape(-1, particle_dim)
        elif t_init_mode == "random":
            t_values = t_init_scale * torch.randn(num_cgf_features, particle_dim)
        else:
            raise ValueError(f"Unknown t_init_mode: {t_init_mode}")

        self.t_frozen = bool(t_frozen)
        if self.t_frozen:
            # A buffer gets no gradient (so PPO cannot move it) while still
            # saving/loading and moving across devices exactly like the
            # Parameter. forward() is untouched: on the spread init the
            # elementwise clamp is a no-op by construction.
            self.register_buffer("t_values", t_values)
        else:
            self.t_values = nn.Parameter(t_values)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        base_obs = obs_dict["obs"]
        particles = obs_dict["particles"] / self.arena_scale
        weights = obs_dict["weights"]

        particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.clamp(weights, min=0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        t = torch.clamp(self.t_values, -self.t_clamp, self.t_clamp)
        exp_arg = torch.matmul(particles, t.transpose(0, 1))
        exp_arg = torch.clamp(exp_arg, -self.exp_arg_clamp, self.exp_arg_clamp)

        weighted_mgf = torch.sum(
            weights.unsqueeze(-1) * torch.exp(exp_arg),
            dim=1,
        )
        cgf = torch.log(torch.clamp(weighted_mgf, min=1e-8))
        return torch.cat([base_obs, cgf], dim=-1)
