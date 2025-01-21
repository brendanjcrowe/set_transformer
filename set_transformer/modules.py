import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block (MAB) module.
    
    This is a core building block of the Set Transformer that implements multihead attention
    with preprocessing and postprocessing networks. It computes attention between a set of 
    queries Q and a set of key-value pairs K (where K is also used as values).

    Args:
        dim_Q (int): Dimension of the query features
        dim_K (int): Dimension of the key features
        dim_V (int): Dimension of the value features (output dimension)
        num_heads (int): Number of attention heads
        ln (bool, optional): Whether to use layer normalization. Defaults to False.
    """

    def __init__(
        self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False
    ) -> None:
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MAB module.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, num_queries, dim_Q)
            K (torch.Tensor): Key tensor of shape (batch_size, num_keys, dim_K)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_queries, dim_V)
        """
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    """Self-Attention Block (SAB) module.
    
    This module applies self-attention to a set of inputs. It's essentially a MAB where
    the input set acts as both the query and key-value set, enabling the elements to
    attend to each other.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_heads (int): Number of attention heads
        ln (bool, optional): Whether to use layer normalization. Defaults to False.
    """

    def __init__(
        self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False
    ) -> None:
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SAB module.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, set_size, dim_out)
        """
        return self.mab(X, X)


class ISAB(nn.Module):
    """Induced Set Attention Block (ISAB) module.
    
    This module implements a more efficient version of SAB by using a set of inducing points.
    Instead of computing attention between all pairs of elements (O(n²) complexity),
    it computes attention through a smaller set of inducing points (O(n) complexity).

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_heads (int): Number of attention heads
        num_inds (int): Number of inducing points
        ln (bool, optional): Whether to use layer normalization. Defaults to False.
    """

    def __init__(
        self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, ln: bool = False
    ) -> None:
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ISAB module.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim_in)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, set_size, dim_out)
        """
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    """Pooling by Multihead Attention (PMA) module.
    
    This module learns to pool a set of features into a fixed number of outputs using
    multihead attention. It uses a set of learned seed vectors to attend to the input set,
    producing a fixed-size output regardless of the input set size.

    Args:
        dim (int): Feature dimension
        num_heads (int): Number of attention heads
        num_seeds (int): Number of seed vectors (output size)
        ln (bool, optional): Whether to use layer normalization. Defaults to False.
    """

    def __init__(
        self, dim: int, num_heads: int, num_seeds: int, ln: bool = False
    ) -> None:
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PMA module.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_seeds, dim)
        """
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
