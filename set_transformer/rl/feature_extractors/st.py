"""Unified Set Transformer feature extractor for the belief-encoder benchmark.

:class:`SetTransformerExtractor` covers all three ST method flavors behind one class:

- **st_frozen**    — ``pretrained_model_path=<ckpt>, freeze=True``
- **st_finetune**  — ``pretrained_model_path=<ckpt>, freeze=False``
- **st_scratch**   — ``pretrained_model_path=None``

It subclasses the same :class:`~set_transformer.rl.feature_extractors.statistical.
_BasePFStatExtractor` plumbing as the statistical baselines, so **every method in the
benchmark shares an identical obs-MLP + concat + projection head** and differs only in
the particle-set statistic — the fairness requirement of the paper.

The particle encoder is the ``set_transformer`` half of a
:class:`~set_transformer.models.PFSetTransformer` autoencoder, so pretraining checkpoints
produced by ``set_transformer.training`` (either raw ``state_dict`` or Trainer checkpoint
dicts with a ``model_state_dict`` key) load directly — the same convention used by the
inline ``FineTunableSTFeaturesExtractor``/``AntTagPretrainedProcessor`` in
``experiments/ant_tag/4_train_rl_*.py``, which this class supersedes for the benchmark.
"""

from __future__ import annotations

import gymnasium as gym
import torch

from set_transformer.models import PFSetTransformer
from set_transformer.rl.feature_extractors.statistical import (
    _BasePFStatExtractor,
    load_pretrained_state_dict,
)

#: Backwards-compatible alias; checkpoint loading is now shared with the pooling encoders.
load_pf_st_state_dict = load_pretrained_state_dict


class SetTransformerExtractor(_BasePFStatExtractor):
    """Set Transformer particle-set encoder (pretrained/frozen/fine-tuned/from-scratch).

    ``num_particles`` and ``dim_particles`` are inferred from the ``particles`` entry of
    the observation space; the remaining architecture args must match the pretraining
    run when loading a checkpoint.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        pretrained_model_path: str | None = None,
        freeze: bool = False,
        num_encodings: int = 8,
        dim_encoder: int = 2,
        num_inds: int = 32,
        dim_hidden: int = 64,
        num_heads: int = 4,
        ln: bool = True,
        obs_mlp_hidden_dims: list[int] = [64, 64],
        features_dim: int = 128,
    ):
        self._check_freeze(pretrained_model_path, freeze)
        self.pretrained_model_path = pretrained_model_path
        self.freeze = freeze
        self.num_encodings = num_encodings
        self.dim_encoder = dim_encoder
        self._st_arch = dict(
            num_encodings=num_encodings,
            dim_encoder=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        )
        super().__init__(observation_space, obs_mlp_hidden_dims, features_dim)

    def _build_particle_stat(self) -> None:
        self.pf_st = self._apply_pretrained(
            PFSetTransformer(
                num_particles=self.num_particles,
                dim_particles=self.particle_dim,
                **self._st_arch,
            ),
            self.pretrained_model_path,
            self.freeze,
        )
        # Only the encoder half is used for feature extraction; the decoder stays
        # attached so checkpoints round-trip, but contributes nothing to forward.
        self.encoder = self.pf_st.set_transformer

    def _particle_stat_dim(self) -> int:
        return self.num_encodings * self.dim_encoder

    def _particle_features(self, particles: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(particles)  # [B, num_encodings, dim_encoder]
        return enc.reshape(enc.size(0), -1)
