"""Pooling-based particle-set feature extractors for SB3 policies.

Two learned, permutation-invariant encoders that sit between the analytic baselines
(Gaussian / k-moments / CGF) and the Set Transformer in the belief-encoder benchmark:

- :class:`DeepSetExtractor`  — mean-pool DeepSet encoder (Zaheer et al., 2017).
- :class:`PointNetExtractor` — max-pool PointNet encoder (Qi et al., 2017).

Both share a per-element MLP + symmetric pooling + output MLP, differing *only* in the
pooling operator (mean vs. max), so the benchmark isolates aggregation as the sole
difference between them. Like the from-scratch Set Transformer and the CGF extractor, the
particle encoder is trained end-to-end with the policy (no pretraining).

They subclass the same :class:`~set_transformer.rl.feature_extractors.statistical.
_BasePFStatExtractor` plumbing as every other benchmark method, so all methods share an
identical obs-MLP + concat + projection head and differ only in the particle statistic.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from set_transformer.models import DeepSet, PointNet
from set_transformer.rl.feature_extractors.statistical import _BasePFStatExtractor


class _PooledSetExtractor(_BasePFStatExtractor):
    """Shared plumbing for the DeepSet / PointNet pooling encoders.

    Subclasses set :attr:`encoder_cls` to a set-encoder module with the signature
    ``(dim_input, num_outputs, dim_output, dim_hidden)`` returning ``[B, num_outputs,
    dim_output]``. The flattened encoding (``num_encodings * dim_encoder`` features) is the
    particle-side statistic.
    """

    encoder_cls: type  # set by subclass

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_encodings: int = 8,
        dim_encoder: int = 16,
        dim_hidden: int = 128,
        obs_mlp_hidden_dims: list[int] = [64, 64],
        features_dim: int = 128,
    ):
        self.num_encodings = num_encodings
        self.dim_encoder = dim_encoder
        self.dim_hidden = dim_hidden
        super().__init__(observation_space, obs_mlp_hidden_dims, features_dim)

    def _build_particle_stat(self) -> None:
        self.encoder = type(self).encoder_cls(
            dim_input=self.particle_dim,
            num_outputs=self.num_encodings,
            dim_output=self.dim_encoder,
            dim_hidden=self.dim_hidden,
        )

    def _particle_stat_dim(self) -> int:
        return self.num_encodings * self.dim_encoder

    def _particle_features(self, particles: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(particles)  # [B, num_encodings, dim_encoder]
        return enc.reshape(enc.size(0), -1)


class DeepSetExtractor(_PooledSetExtractor):
    """Mean-pool DeepSet particle encoder (Zaheer et al., 2017), trained from scratch."""

    encoder_cls = DeepSet


class PointNetExtractor(_PooledSetExtractor):
    """Max-pool PointNet particle encoder (Qi et al., 2017), trained from scratch."""

    encoder_cls = PointNet
