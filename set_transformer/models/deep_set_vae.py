from typing import Any, Dict

import torch
import torch.nn as nn

from set_transformer.modules import PFDecoder
from set_transformer.models.deep_set import DeepSet
from set_transformer.models.set_vae import SetVAE


class DeepSetVAE(nn.Module):
    """Variational DeepSet autoencoder — DeepSet encoder + Gaussian
    bottleneck + PFDecoder. Same forward-return contract as `SetVAE`.
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
        self.fc_mu = nn.Linear(dim_encoder, dim_encoder)
        self.fc_logvar = nn.Linear(dim_encoder, dim_encoder)
        self.decoder = PFDecoder(dim_encoder, dim_hidden, num_particles, dim_particles)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(X)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "kl": SetVAE.kl_loss(mu, logvar),
        }
