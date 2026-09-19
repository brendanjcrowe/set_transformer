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
        readout: str = "per_seed",
        num_pma_seeds: int | None = None,
        input_embed: bool = False,
    ) -> None:
        super(SetTransformer, self).__init__()
        if num_post_sab < 0:
            raise ValueError(f"num_post_sab must be >= 0, got {num_post_sab}")
        if readout not in ("per_seed", "flatten"):
            raise ValueError(f"readout must be 'per_seed' or 'flatten', got {readout!r}")
        if num_pma_seeds is not None and readout == "per_seed" and int(num_pma_seeds) != int(num_outputs):
            raise ValueError("num_pma_seeds can differ from num_outputs only with readout='flatten' "
                             f"(got num_pma_seeds={num_pma_seeds}, num_outputs={num_outputs})")
        self.num_post_sab = int(num_post_sab)
        # 2026-09-19 (least-mass gap, src/scripts/least_mass_gap/NOTES.md): the record's encoder
        # (src/hunt_tasks/encoders/extractors.py::SetTransformerExtractor) differs from this one in
        # two structural ways that no flag reached. `readout="flatten"`: the pooled seed vectors are
        # flattened and mixed by ONE Linear(num_pma_seeds * dim_hidden -> num_outputs * dim_output)
        # followed by a GELU, instead of every seed being compressed by a shared
        # Linear(dim_hidden -> dim_output); `num_pma_seeds` then decouples the number of pooling
        # seeds from the output shape (the record: 5 seeds -> 64 features). `input_embed`: a
        # Linear(dim_input -> dim_hidden) before the first ISAB. Defaults reproduce the old module
        # exactly (same state_dict keys), so every existing checkpoint and zip still loads strictly.
        self.readout = str(readout)
        self.input_embed = bool(input_embed)
        self.num_pma_seeds = int(num_pma_seeds) if num_pma_seeds is not None else int(num_outputs)
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
        if self.input_embed:
            self.embed = nn.Linear(dim_input, dim_hidden)
        self.enc = nn.Sequential(
            ISAB(dim_hidden if self.input_embed else dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        # Construction order = parameter-initialisation order (PMA, then the SABs, then the output
        # Linear), kept exactly as before so a seeded default encoder still matches its golden values
        # (tests/test_ant_tag_shared_pieces_regression.py).
        pma = PMA(dim_hidden, num_heads, self.num_pma_seeds if self.readout == "flatten" else num_outputs, ln=ln)
        post = [SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(self.num_post_sab)]
        if self.readout == "flatten":
            self.dec = nn.Sequential(pma, *post)
            self.readout_linear = nn.Linear(self.num_pma_seeds * dim_hidden, self.num_outputs * self.dim_output)
        else:
            self.dec = nn.Sequential(pma, *post, nn.Linear(dim_hidden, dim_output))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Process a batch of sets through the Set Transformer model.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, set_size, dim_input)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_outputs, dim_output)
        """
        if self.input_embed:
            X = self.embed(X)
        out = self.dec(self.enc(X))
        if self.readout == "flatten":
            out = F.gelu(self.readout_linear(out.reshape(out.size(0), -1)))
            out = out.reshape(out.size(0), self.num_outputs, self.dim_output)
        if self.output_norm:
            flat = out.reshape(out.size(0), -1)
            flat = F.layer_norm(flat, (flat.size(-1),))
            out = flat.reshape(out.size(0), self.num_outputs, self.dim_output)
        return out
