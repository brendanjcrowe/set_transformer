import torch
import torch.nn as nn


from modules import ISAB, PMA, SAB, PFDecoder


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
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

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
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

    def forward(self, X):
        return self.dec(self.enc(X))


class PFSetTransformer(nn.Module):
    def __init__(
        self,
        num_particles: int,
        dim_partciles: int,
        num_encodings: int,
        dim_encoder: int,
        num_inds: int = 32,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = False,
    ) -> None:
        super(PFSetTransformer, self).__init__()
        self.set_transformer = SetTransformer(
            dim_partciles,
            num_outputs=num_encodings,
            dim_output=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        )
        self.decoder = PFDecoder(dim_encoder, dim_hidden, num_particles, dim_partciles)


    def forward(self, X):
        return self.decoder(self.set_transformer(X))