import torch
import torch.nn as nn

from set_transformer.modules import PFDecoder
from set_transformer.models.set_transformer import SetTransformer


class PFSetTransformer(nn.Module):
    """Particle Filter Set Transformer model.

    This model combines a Set Transformer encoder with a particle filter decoder
    to process sets of particles. It first encodes the particle set into a fixed-size
    representation, then decodes it back into a set of particles.

    Args:
        num_particles (int): Number of particles to output
        dim_particles (int): Dimension of each particle
        num_encodings (int): Number of encodings to generate from the set transformer
        dim_encoder (int): Dimension of each encoding
        num_inds (int, optional): Number of inducing points. Defaults to 32.
        dim_hidden (int, optional): Dimension of hidden layers. Defaults to 128.
        num_heads (int, optional): Number of attention heads. Defaults to 4.
        ln (bool, optional): Whether to use layer normalization. Defaults to False.
        dim_output_particles (int, optional): Dimension of each RECONSTRUCTED
            particle. Defaults to None, meaning `dim_particles` — input and
            output shapes match, the ordinary autoencoder.

            Set it smaller than `dim_particles` for weighted particle sets: the
            encoder reads D coordinates plus a mass channel (dim_particles=D+1)
            while the decoder emits D coordinates only (dim_output_particles=D).
            The mass belongs in the reconstruction LOSS, as the target measure's
            weights, not in the reconstructed points — asking the decoder to
            regress a weight would put probability mass inside the geometric
            ground metric and let it predict negative, unnormalized "weights".
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
        dim_output_particles: int | None = None,
    ) -> None:
        super(PFSetTransformer, self).__init__()
        if dim_output_particles is None:
            dim_output_particles = dim_particles
        self.dim_particles = dim_particles
        self.dim_output_particles = dim_output_particles
        self.set_transformer = SetTransformer(
            dim_particles,
            num_outputs=num_encodings,
            dim_output=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        )
        self.decoder = PFDecoder(
            dim_encoder, dim_hidden, num_particles, dim_output_particles
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PFSetTransformer.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim_particles)

        Returns:
            torch.Tensor: Output tensor of shape
                (batch_size, num_particles, dim_output_particles)
        """
        return self.decoder(self.set_transformer(X))
