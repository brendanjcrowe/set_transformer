from typing import Any

import torch
import torch.nn as nn

from set_transformer.modules import PFDecoder
from set_transformer.models.deep_set import DeepSet


class DeepSetAE(nn.Module):
    """Deep Sets autoencoder — analogous to `PFSetTransformer` but with a
    mean-pool `DeepSet` encoder instead of a Set Transformer. Extra kwargs
    (`num_inds`, `num_heads`, `ln`) are accepted for API parity with the ST
    variants but silently ignored.
    """

    def __init__(
        self,
        num_particles: int,
        dim_particles: int,
        num_encodings: int,
        dim_encoder: int,
        dim_hidden: int = 128,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder = DeepSet(
            dim_input=dim_particles,
            num_outputs=num_encodings,
            dim_output=dim_encoder,
            dim_hidden=dim_hidden,
        )
        self.decoder = PFDecoder(dim_encoder, dim_hidden, num_particles, dim_particles)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(X))
