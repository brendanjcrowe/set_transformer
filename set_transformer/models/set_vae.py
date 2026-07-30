from typing import Dict

import torch
import torch.nn as nn

from set_transformer.modules import PFDecoder
from set_transformer.models.set_transformer import SetTransformer


class SetVAE(nn.Module):
    """Variational Set Transformer autoencoder.

    Identical encoder/decoder skeleton to PFSetTransformer; the bottleneck
    `[B, num_encodings, dim_encoder]` is reinterpreted as a Gaussian posterior
    via two linear heads producing `mu` and `logvar`. A reparameterized sample
    `z` is passed to PFDecoder. Forward returns a dict including the KL
    divergence to the standard normal prior (averaged over batch, summed over
    latent dims).
    """

    def __init__(
        self,
        num_particles: int,
        dim_particles: int,
        num_encodings: int,
        dim_encoder: int,
        num_inds: int = 32,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = False,
    ) -> None:
        super().__init__()
        self.set_transformer = SetTransformer(
            dim_particles,
            num_outputs=num_encodings,
            dim_output=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        )
        self.fc_mu = nn.Linear(dim_encoder, dim_encoder)
        self.fc_logvar = nn.Linear(dim_encoder, dim_encoder)
        self.decoder = PFDecoder(dim_encoder, dim_hidden, num_particles, dim_particles)

    @staticmethod
    def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # KL[N(mu, sigma^2) || N(0, I)] summed over latent dims, averaged over batch.
        kl_per_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return kl_per_elem.sum(dim=tuple(range(1, kl_per_elem.dim()))).mean()

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.set_transformer(X)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "kl": self.kl_loss(mu, logvar),
        }
