from typing import Tuple

import torch
import torch.nn as nn

from .modules import ISAB, PMA, SAB


class DeepSet(nn.Module):
    """A Deep Set model implementation that processes sets of varying size.

    This model implements the Deep Sets architecture (Zaheer et al., 2017) which consists
    of an encoder network that processes each set element independently followed by a 
    permutation-invariant pooling operation (mean) and a decoder network.

    Args:
        dim_input (int): Dimension of input features for each set element
        num_outputs (int): Number of outputs to generate
        dim_output (int): Dimension of each output
        dim_hidden (int, optional): Dimension of hidden layers. Defaults to 128.
    """

    def __init__(
        self, dim_input: int, num_outputs: int, dim_output: int, dim_hidden: int = 128
    ) -> None:
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Process a batch of sets through the Deep Set model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim_input)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs, dim_output)
        """
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


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
    ) -> None:
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Process a batch of sets through the Set Transformer model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim_input)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs, dim_output)
        """
        return self.dec(self.enc(X))
