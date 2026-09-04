import torch
import torch.nn as nn


class PointNet(nn.Module):
    """A PointNet-style set encoder (Qi et al., 2017).

    Structurally identical to :class:`~set_transformer.models.deep_set.DeepSet` — a shared
    per-element MLP followed by a permutation-invariant pooling operation and an output
    MLP — but the symmetric aggregation is **max** pooling rather than DeepSet's **mean**
    pooling. Holding everything else fixed, this isolates the pooling operator as the only
    difference between the two encoders.

    Args:
        dim_input (int): Dimension of input features for each set element
        num_outputs (int): Number of outputs to generate
        dim_output (int): Dimension of each output
        dim_hidden (int, optional): Dimension of hidden layers. Defaults to 128.
    """

    def __init__(
        self, dim_input: int, num_outputs: int, dim_output: int, dim_hidden: int = 128
    ) -> None:
        super(PointNet, self).__init__()
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
        """Process a batch of sets through the PointNet encoder.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim_input)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs, dim_output)
        """
        X = self.enc(X).max(-2)[0]  # max pooling over the set dimension (permutation-invariant)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X
