"""Pooling-based particle-set feature extractors for SB3 policies.

Two learned, permutation-invariant encoders that sit between the analytic baselines
(Gaussian / k-moments / CGF) and the Set Transformer in the belief-encoder benchmark:

- :class:`DeepSetExtractor`  — mean-pool DeepSet encoder (Zaheer et al., 2017).
- :class:`PointNetExtractor` — max-pool PointNet encoder (Qi et al., 2017).

Both share a per-element MLP + symmetric pooling + output MLP, differing *only* in the
pooling operator (mean vs. max), so the benchmark isolates aggregation as the sole
difference between them.

Like :class:`~set_transformer.rl.feature_extractors.st.SetTransformerExtractor`, each
covers three flavors behind one class — from-scratch, frozen-pretrained and
fine-tuned-pretrained — selected by ``pretrained_model_path`` / ``freeze``. The encoder is
the ``encoder`` half of a :class:`~set_transformer.models.DeepSetAE` /
:class:`~set_transformer.models.PointNetAE` autoencoder, so pretraining checkpoints load
directly.

They subclass the same :class:`~set_transformer.rl.feature_extractors.statistical.
_BasePFStatExtractor` plumbing as every other benchmark method, so all methods share an
identical obs-MLP + concat + projection head and differ only in the particle statistic.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from set_transformer.models import DeepSetAE, PointNetAE
from set_transformer.rl.feature_extractors.statistical import _BasePFStatExtractor


class _PooledSetExtractor(_BasePFStatExtractor):
    """Shared plumbing for the DeepSet / PointNet pooling encoders.

    Subclasses set :attr:`autoencoder_cls` to a set autoencoder exposing ``.encoder``
    returning ``[B, num_outputs, dim_output]``. The flattened encoding
    (``num_encodings * dim_encoder`` features) is the particle-side statistic.
    """

    autoencoder_cls: type  # set by subclass

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pretrained_model_path: str | None = None,
        freeze: bool = False,
        num_encodings: int = 8,
        dim_encoder: int = 2,
        dim_hidden: int = 128,
        obs_mlp_hidden_dims: list[int] = [64, 64],
        features_dim: int = 128,
        **_: Any,  # num_inds / num_heads / ln: accepted for API parity with the ST arch
    ):
        self._check_freeze(pretrained_model_path, freeze)
        self.pretrained_model_path = pretrained_model_path
        self.freeze = freeze
        self.num_encodings = num_encodings
        self.dim_encoder = dim_encoder
        self.dim_hidden = dim_hidden
        super().__init__(observation_space, obs_mlp_hidden_dims, features_dim)

    def _build_particle_stat(self) -> None:
        self.autoencoder = self._apply_pretrained(
            type(self).autoencoder_cls(
                num_particles=self.num_particles,
                dim_particles=self.particle_dim,
                num_encodings=self.num_encodings,
                dim_encoder=self.dim_encoder,
                dim_hidden=self.dim_hidden,
            ),
            self.pretrained_model_path,
            self.freeze,
        )
        # Only the encoder half feeds forward(); the decoder stays attached so that
        # checkpoints round-trip unchanged.
        self.encoder = self.autoencoder.encoder

    def _particle_stat_dim(self) -> int:
        return self.num_encodings * self.dim_encoder

    def _particle_features(self, particles: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(particles)  # [B, num_encodings, dim_encoder]
        return enc.reshape(enc.size(0), -1)


class DeepSetExtractor(_PooledSetExtractor):
    """Mean-pool DeepSet particle encoder (Zaheer et al., 2017)."""

    autoencoder_cls = DeepSetAE


class PointNetExtractor(_PooledSetExtractor):
    """Max-pool PointNet particle encoder (Qi et al., 2017)."""

    autoencoder_cls = PointNetAE
