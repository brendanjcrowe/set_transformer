from typing import Any

import torch
import torch.nn as nn

from set_transformer.modules import PFDecoder
from set_transformer.models.cgf_encoder import CGFEncoder


class CGFAutoencoder(nn.Module):
    """CGF autoencoder — the pretraining counterpart of the CGF feature extractor.

    Same shape as `PFSetTransformer` / `DeepSetAE` / `PointNetAE`: an encoder producing a
    `(batch, num_encodings, dim_encoder)` code and a `PFDecoder` reconstructing the set.
    The encoder is the shared :class:`CGFEncoder`, so a checkpoint trained here loads
    directly into the policy's extractor.
    """

    def __init__(
        self,
        num_particles: int,
        dim_particles: int,
        num_encodings: int,
        dim_encoder: int,
        dim_hidden: int = 128,
        **cgf_kwargs: Any,
    ) -> None:
        super().__init__()
        self.encoder = CGFEncoder(
            dim_input=dim_particles,
            num_outputs=num_encodings,
            dim_output=dim_encoder,
            **cgf_kwargs,
        )
        self.decoder = PFDecoder(dim_encoder, dim_hidden, num_particles, dim_particles)

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """Encode a particle set to its `(batch, num_encodings, dim_encoder)` code."""
        return self.encoder(X)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(X))
