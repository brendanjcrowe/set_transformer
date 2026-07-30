from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from set_transformer.modules import PFDecoder
from set_transformer.models.set_transformer import SetTransformer


class VectorQuantizerEMA(nn.Module):
    """Vector-quantizer with EMA codebook updates (van den Oord et al., 2017).

    Quantizes each vector in a tensor of shape `[..., dim]` against a codebook
    of `num_codes` learned entries, returning the quantized tensor (same shape),
    the assignment indices, the commitment loss, and the batch perplexity.
    """

    def __init__(
        self,
        num_codes: int,
        dim: int,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.commitment_weight = commitment_weight
        self.decay = ema_decay
        self.eps = eps

        embedding = torch.randn(num_codes, dim) * (1.0 / (dim ** 0.5))
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", embedding.clone())

    def forward(self, z_e: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat = z_e.reshape(-1, self.dim)  # [N, D]
        # Squared L2 distances to codes.
        dists = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + self.embedding.pow(2).sum(dim=1)
        )
        indices = dists.argmin(dim=1)  # [N]
        onehot = F.one_hot(indices, self.num_codes).type(flat.dtype)  # [N, K]

        q_flat = self.embedding[indices]  # [N, D]
        q = q_flat.view_as(z_e)

        if self.training:
            with torch.no_grad():
                self.ema_cluster_size.mul_(self.decay).add_(
                    onehot.sum(dim=0), alpha=1 - self.decay
                )
                dw = onehot.t() @ flat  # [K, D]
                self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.num_codes * self.eps)
                    * n
                )
                self.embedding.copy_(self.ema_w / smoothed.unsqueeze(1))

        commitment = F.mse_loss(z_e, q.detach())

        # Straight-through estimator.
        q_st = z_e + (q - z_e).detach()

        avg_probs = onehot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())
        indices_shaped = indices.view(z_e.shape[:-1])

        return {
            "z_q": q_st,
            "indices": indices_shaped,
            "commitment_loss": commitment,
            "perplexity": perplexity,
        }


class SetVQVAE(nn.Module):
    """VQ-VAE Set Transformer autoencoder.

    Identical encoder/decoder skeleton to PFSetTransformer; each of the
    `num_encodings` latent slots `[B, num_encodings, dim_encoder]` is
    independently quantized against a shared codebook with EMA updates.
    Straight-through gradient lets the encoder train via reconstruction loss.
    """

    def __init__(
        self,
        num_particles: int,
        dim_particles: int,
        num_encodings: int,
        dim_encoder: int,
        codebook_size: int,
        num_inds: int = 32,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = False,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.commitment_weight = commitment_weight
        self.set_transformer = SetTransformer(
            dim_particles,
            num_outputs=num_encodings,
            dim_output=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        )
        self.quantizer = VectorQuantizerEMA(
            num_codes=codebook_size,
            dim=dim_encoder,
            commitment_weight=commitment_weight,
            ema_decay=ema_decay,
        )
        self.decoder = PFDecoder(dim_encoder, dim_hidden, num_particles, dim_particles)

    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.set_transformer(X)
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
