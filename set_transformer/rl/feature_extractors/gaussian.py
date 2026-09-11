"""Weighted Gaussian (mean + covariance) belief encoder for SB3.

    mean_d      = sum_i w_i * x_i[d]
    cov_{d,d'}  = sum_i w_i * (x_i[d] - mean_d) * (x_i[d'] - mean_d')

for particle dimension d, d' in {0, ..., D-1}. The policy receives
[mean, var, off-diagonal covariance]. Unlike the CGF and Set Transformer
encoders this extractor has no learnable parameters — it is a fixed,
closed-form summary of the particle set.

**Domain-independent.** It reads the ``{"obs", "particles", "weights"}`` Dict
observation produced by
:class:`set_transformer.rl.wrappers.particle_filter.PFDictWithWeightsObservationWrapper`
and derives its feature count from the particle dimension, so it works for any
D (5 features for D=2, 2 features for the 1-D case).

Moved out of ``experiments/ant_tag/4_train_rl_gaussian.py`` so a second domain
can use it without importing an Ant-Tag script. That script still re-exports
the name: SB3 pickles a features-extractor CLASS into the saved zip by module
path, so an existing checkpoint loads via
``getattr(import_module("4_train_rl_gaussian"), "WeightedGaussianFeaturesExtractor")``
and the re-export is what keeps thousands of saved runs loadable.
"""

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def weighted_mean_cov(particles: torch.Tensor,
                      weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted mean ``[B, D]`` and covariance ``[B, D, D]`` of a particle set.

        mean_d      = sum_i w_i * x_i[d]
        cov_{d,d'}  = sum_i w_i * (x_i[d] - mean_d) * (x_i[d'] - mean_d')

    Weights are cleaned (NaN/inf -> 0, negatives -> 0) and normalised by the
    exact-sum rule with ``+ 1e-8``; uniform weights give the biased (divide by
    N) covariance exactly. Module-level (2026-09-11) so the benchmark's
    :class:`~set_transformer.rl.feature_extractors.statistical.GaussianExtractor`
    computes the same statistic instead of an unweighted copy.
    """
    particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, min=0.0)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    w = weights.unsqueeze(-1)                                         # [B, N, 1]
    mean = torch.sum(w * particles, dim=1)                            # [B, D]
    centered = particles - mean.unsqueeze(1)                          # [B, N, D]
    cov = torch.einsum("bni,bnj->bij", w * centered, centered)        # [B, D, D]
    return mean, torch.nan_to_num(cov, nan=0.0)


class WeightedGaussianFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor for weighted Gaussian (mean + covariance) particle features.

    Computes the weighted mean and covariance of the particle set (normalized
    by arena_scale, matching WeightedCGFFeaturesExtractor's particle scaling),
    then exposes [mean, var, off-diagonal covariance] as features. For
    particle_dim=D this is D + D + D*(D-1)/2 features (5 for D=2).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        arena_scale: float = 4.5,
    ):
        obs_dim = observation_space["obs"].shape[0]
        particle_dim = observation_space["particles"].shape[1]
        num_gaussian_features = particle_dim + particle_dim + particle_dim * (particle_dim - 1) // 2
        super().__init__(observation_space, features_dim=obs_dim + num_gaussian_features)

        self.particle_dim = particle_dim
        self.arena_scale = arena_scale

        triu_indices = torch.triu_indices(particle_dim, particle_dim, offset=1)
        self.register_buffer("triu_rows", triu_indices[0])
        self.register_buffer("triu_cols", triu_indices[1])

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        base_obs = obs_dict["obs"]
        mean, cov = weighted_mean_cov(obs_dict["particles"] / self.arena_scale,
                                      obs_dict["weights"])

        # Weighted variance can dip slightly below zero from floating-point
        # cancellation (mean subtracted then squared back out); clamp so the
        # policy never sees a negative "variance" feature.
        var = torch.clamp(torch.diagonal(cov, dim1=-2, dim2=-1), min=0.0)  # [B, D]
        off_diag_cov = cov[:, self.triu_rows, self.triu_cols]  # [B, D*(D-1)/2]

        gaussian_features = torch.cat([mean, var, off_diag_cov], dim=-1)
        return torch.cat([base_obs, gaussian_features], dim=-1)
