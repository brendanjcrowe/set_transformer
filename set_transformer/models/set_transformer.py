import torch
import torch.nn as nn
import torch.nn.functional as F

from set_transformer.modules import ISAB, PMA, SAB


class SetTransformer(nn.Module):
    """A Set Transformer model implementation for processing sets of varying size.

    This model implements the Set Transformer architecture (Lee et al., 2019) which uses
    attention mechanisms to process sets. It consists of Induced Set Attention Blocks (ISAB)
    for encoding and Pooling by Multihead Attention (PMA) followed by Set Attention Blocks (SAB)
    for decoding.

    Args:
        dim_input (int): Dimension of input features for each set element
        num_outputs (int): Number of outputs to generate
        dim_output (int): Dimension of each output
        num_inds (int, optional): Number of inducing points. Defaults to 32.
        dim_hidden (int, optional): Dimension of hidden layers. Defaults to 128.
        num_heads (int, optional): Number of attention heads. Defaults to 4.
        ln (bool, optional): Whether to use layer normalization. Defaults to False.
        num_post_sab (int, optional): Number of SAB blocks between the PMA and
            the output Linear. Defaults to 2 (Lee et al., 2019). 0 gives the
            PMA -> Linear head used by the ClusterHunt/LeastMass encoders:
            the seed vectors then never attend to each other, so every output
            feature is a fixed linear combination of per-seed pooled statistics.
    """

    def __init__(
        self,
        dim_input: int,
        num_outputs: int,
        dim_output: int,
        num_inds: int = 32,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = False,
        num_post_sab: int = 2,
        output_norm: bool = False,
    ) -> None:
        super(SetTransformer, self).__init__()
        if num_post_sab < 0:
            raise ValueError(f"num_post_sab must be >= 0, got {num_post_sab}")
        self.num_post_sab = int(num_post_sab)
        # 2026-09-19 (debug_plans/ch_fixes.md, change A1): when on, the flattened
        # num_outputs x dim_output code is layer-normalised WITHOUT learned gain or bias as
        # the encoder's LAST operation, so the features a policy reads always have mean 0 and
        # length sqrt(num_outputs * dim_output), whatever the weights do. The reconstruction
        # objective's softmax decoder otherwise rewards large codes (Cluster-Hunt: length ~87,
        # saturating the PPO policy's tanh layer). Default OFF: an old checkpoint or zip
        # rebuilds the encoder without it and computes exactly what it did before.
        self.output_norm = bool(output_norm)
        self.num_outputs = int(num_outputs)
        self.dim_output = int(dim_output)
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            *[SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
              for _ in range(self.num_post_sab)],
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Process a batch of sets through the Set Transformer model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim_input)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs, dim_output)
        """
        out = self.dec(self.enc(X))
        if self.output_norm:
            flat = out.reshape(out.size(0), -1)
            flat = F.layer_norm(flat, (flat.size(-1),))
            out = flat.reshape(out.size(0), self.num_outputs, self.dim_output)
        return out
