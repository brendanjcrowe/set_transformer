from typing import Any, Dict

import torch
import torch.nn as nn

from set_transformer.modules import PFDecoder
from set_transformer.models.deep_set import DeepSet
from set_transformer.models.set_vqvae import VectorQuantizerEMA


class DeepSetVQVAE(nn.Module):
    """DeepSet + VQ-VAE bottleneck + PFDecoder. Same forward-return contract
    as `SetVQVAE`.
    """

    def __init__(
        self,
        num_particles: int,
        dim_particles: int,
        num_encodings: int,
        dim_encoder: int,
        codebook_size: int,
        dim_hidden: int = 128,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        **_: Any,
    ) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight
        self.encoder = DeepSet(
            dim_input=dim_particles,
            num_outputs=num_encodings,
            dim_output=dim_encoder,
            dim_hidden=dim_hidden,
        )
        self.quantizer = VectorQuantizerEMA(
            num_codes=codebook_size,
            dim=dim_encoder,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay,
        )
        self.decoder = PFDecoder(dim_encoder, dim_hidden, num_particles, dim_particles)

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encoder(X)
        q = self.quantizer(z_e)
        recon = self.decoder(q["z_q"])
        return {
            "recon": recon,
            "z_e": z_e,
            "z_q": q["z_q"],
            "indices": q["indices"],
            "commitment_loss": q["commitment_loss"],
            "perplexity": q["perplexity"],
        }
