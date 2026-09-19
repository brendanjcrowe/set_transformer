import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self, dim_input: int, num_outputs: int, dim_output: int, dim_hidden: int = 128,
        output_norm: bool = False,
    ) -> None:
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        # 2026-09-19 (debug_plans/ch_fixes.md, A3): layer-normalise the flattened code (no
        # learned gain / bias) as the last operation; default OFF = today's forward exactly.
        self.output_norm = bool(output_norm)
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
        X = self.dec(X)
        if self.output_norm:
            X = F.layer_norm(X, (X.size(-1),))
        X = X.reshape(-1, self.num_outputs, self.dim_output)
        return X
