import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
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
    def __init__(
        self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False
    ) -> None:
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, ln: bool = False
    ) -> None:
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, num_seeds: int, ln: bool = False
    ) -> None:
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
