"""Weighted empirical-CGF belief encoder for SB3, plus its logging callback.

    CGF_j = log(sum_i w_i * exp(<t_j, x_i>))

where x_i is a belief particle (normalized by ``arena_scale``) and w_i is its
particle-filter weight. The t_j directions are learned under PPO by default.

Three opt-in extensions (2026-09-05), all defaulting to the legacy behaviour
so every saved Ant-Tag checkpoint and the regression goldens are unchanged:

``t_param="tanh"``
    ``t_j = t_bound * tanh(raw_t_j)`` instead of a hard ``torch.clamp`` on the
    parameter. The hard clamp has zero gradient beyond the bound, so a probe
    that crosses it is frozen for the rest of training; tanh is smooth
    everywhere. This is the parameterisation the ClusterHunt / LeastMass CGF
    arms used. Pick ``t_bound`` by the dimensionless rule
    ``t_bound * (decision-relevant length in normalized units) ~ 2..4``: on
    Odd-Even the same-parity spacing is 2 / 24.5 = 0.0816, so ``t_bound=50``.
``feature_mode="K_grad"`` / ``"both"``
    Also (or only) emit K'(t_j) = sum_i softmax_i(<t_j, x_i> + log w_i) x_i,
    the tilted mean, read off the same score tensor. K' beat K on both
    ClusterHunt and LeastMass and on the Odd-Even mode probe at every t range.
    Width becomes ``T * D`` (K_grad) or ``T * (1 + D)`` (both).
``feature_norm="running"`` / ``"layernorm"``
    Standardise the CGF feature block before it reaches the policy MLP. At
    wide t the raw K values are dominated by ``t_j * mean`` and reach
    magnitudes of tens, which the policy's default init and learning rate
    handle badly. ``"running"`` is per-feature z-scoring with running
    statistics (recommended here); ``"layernorm"`` normalises ACROSS the
    feature vector per sample, which on near-rank-1 features divides out the
    posterior mean itself -- see :class:`RunningFeatureNorm`.
``readout_hidden`` / ``readout_depth``
    An MLP between the (normalised) CGF block and the policy:
    ``Linear(raw -> hidden), GELU, [Linear(hidden -> hidden), GELU] x (depth-1),
    Linear(hidden -> readout_dim)``. The last layer is a bare Linear, like the
    Set Transformer's. This is where a parameter-matched CGF arm keeps its
    budget: the CGF proper is ``T x D`` numbers (64 on Odd-Even) against ~109k
    for the small ST, so without a readout "same-size encoders" is impossible
    (ClusterHunt's `hidden=246, depth=2`; oddeven.md 2026-09-05).
    :func:`matched_readout_hidden` picks the width that lands on a target
    parameter count. Depth 0 (default) is no MLP at all -- today's extractor.
``pretrained_cgf_model_path`` / ``cgf_frozen``
    Load a checkpoint written by ``experiments/odd_even/3_pretrain_st_belief.py
    --encoder cgf`` (t, norm statistics and readout together), and optionally
    freeze the WHOLE encoder: t stops moving, the running norm stops updating
    (``train()`` is overridden to keep the module in eval mode), the readout
    weights are fixed. Only PPO's heads then learn -- the same arm the ST's
    ``st_frozen`` gives. A frozen extractor has no trainable parameters at
    all, so ``policy.parameters()`` shrinks accordingly.

``x_embed_dim`` / ``x_embed_hidden`` / ``x_embed_depth``
    A learned per-particle embedding phi: R^D -> R^d (MLP, GELU) applied
    BEFORE the CGF, with t then living in R^d: feature_j = log sum_i w_i
    exp(<phi(x_i), t_j>). With phi = identity this is the plain CGF. On a
    fixed support (Odd-Even) phi is effectively a learned table over the
    atoms, and at d >= N the family contains the identity map log b -- so
    this is a LEARNED encoder in the Deep Set family (log-sum-exp pooling over
    per-particle embeddings), not a parameter-free statistic. It is kept
    here, as an option on the same class, so it shares the readout, the
    matching, the freeze and the checkpoint path. Default 0 = off.

The constructor defaults stay at the legacy values (``feature_norm="none"``,
no readout) because SB3 rebuilds an extractor from the kwargs stored in each
saved policy, and a changed default would add state-dict keys the ~1250 saved
Ant-Tag checkpoints do not carry. The Odd-Even scripts default to
``feature_norm="running"`` themselves.

The extractor's parameters and buffers ARE the encoder -- there is nothing
else in it -- so ``state_dict()`` is the checkpoint payload and the state-dict
keys are unchanged from before the readout existed (``readout`` is an
``nn.Identity`` at depth 0, which owns no state). Every saved policy loads.

**Domain-independent.** Nothing here knows about any particular POMDP: the
extractor reads the ``{"obs", "particles", "weights"}`` Dict observation
produced by
:class:`set_transformer.rl.wrappers.particle_filter.PFDictWithWeightsObservationWrapper`
and works for any particle dimension. Two init modes are dimension-specific:
``t_init_mode="spread"`` asserts 2-D particles and ``"spread_1d"`` asserts 1-D
ones. They are separate names on purpose -- see ``"spread_1d"`` below.

Moved out of ``experiments/ant_tag/4_train_rl_cgf.py`` so a second domain can
use it without importing an Ant-Tag script. That script still re-exports both
names: SB3 pickles a features-extractor CLASS into the saved zip by module
path, so an existing checkpoint loads via
``getattr(import_module("4_train_rl_cgf"), "WeightedCGFFeaturesExtractor")``
and the re-export is what keeps thousands of saved runs loadable.
"""

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TNormLoggingCallback(BaseCallback):
    """Log the distribution of ||t_j|| every rollout.

    The Twin-Den probe found CGF's advantage needs ||t_j|| ~ 10x the 0.1
    legacy init scale, so whether PPO actually grows t is a first-class
    experimental question. Quantiles land in TensorBoard as
    cgf/t_norm_q{0,25,50,75,90,100}. No-op (and free) for encoders without
    t_values, e.g. the Gaussian arm; a FLAT line is the expected, built-in
    sanity check for the frozen arm.
    """

    def _on_step(self) -> bool:  # required abstract method
        return True

    def _on_rollout_end(self) -> None:
        extractor = getattr(self.model.policy, "features_extractor", None)
        if hasattr(extractor, "effective_t"):
            # The t the forward pass actually uses -- after the clamp, or
            # after the tanh bound. Logging the raw parameter would show a
            # tanh-mode extractor's unbounded pre-activation instead.
            with torch.no_grad():
                t = extractor.effective_t()
        else:
            t = getattr(extractor, "t_values", None)
        if t is None:
            return
        with torch.no_grad():
            norms = torch.linalg.norm(t.detach(), dim=1).cpu().numpy()
        for q in (0, 25, 50, 75, 90, 100):
            self.logger.record(f"cgf/t_norm_q{q}",
                               float(np.percentile(norms, q)))


class RunningFeatureNorm(nn.Module):
    """Per-feature z-scoring with running statistics and no learned affine.

    Why this and not ``nn.LayerNorm`` for CGF features. LayerNorm normalises
    ACROSS the feature vector of each sample. On beliefs where the CGF block
    is close to rank 1 -- every feature ~ ``t_j * mean`` plus small terms,
    which the Odd-Even mode probe measured (domain_mds/oddeven.md,
    2026-09-05) -- the per-sample std across features is proportional to
    ``|mean|``, so LayerNorm divides the posterior mean out of the encoding
    and hands the policy only its sign. Per-feature standardisation keeps
    every feature's information and only rescales it, which is what the
    ClusterHunt fixed-t arms did with statistics calibrated on random
    rollouts. Running statistics do the same without a calibration pass and
    keep tracking as a learned ``t`` moves.

    Statistics update from the (detached) batch only in training mode -- SB3
    puts the policy in training mode during the PPO update and in eval mode
    during rollout collection -- and the RUNNING statistics are used for the
    output in both modes, so the same input maps to the same output within
    an update. The first update copies the batch statistics outright rather
    than lerping from (0, 1), so the features are on scale from the first
    gradient step instead of after ``1 / momentum`` updates.
    """

    def __init__(self, num_features: int, momentum: float = 0.01,
                 eps: float = 1e-5):
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] > 1:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                if int(self.num_updates) == 0:
                    self.running_mean.copy_(batch_mean)
                    self.running_var.copy_(batch_var)
                else:
                    self.running_mean.lerp_(batch_mean, self.momentum)
                    self.running_var.lerp_(batch_var, self.momentum)
                self.num_updates += 1
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)


def readout_param_count(raw_dim: int, hidden: int, depth: int, out_dim: int) -> int:
    """Parameters in the readout MLP ``raw_dim -> hidden x depth -> out_dim``."""
    if depth <= 0 or hidden <= 0:
        return 0
    dims = [raw_dim] + [hidden] * depth + [out_dim]
    return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def cgf_raw_dim(num_cgf_features: int, particle_dim: int, feature_mode: str) -> int:
    """Width of the CGF block before any readout: T, T*D or T*(1+D)."""
    per_probe = {"K": 1, "K_grad": particle_dim, "both": 1 + particle_dim}[feature_mode]
    return num_cgf_features * per_probe


def x_embed_param_count(particle_dim: int, hidden: int, depth: int, embed_dim: int) -> int:
    """Parameters in the particle embedding MLP ``D -> hidden x depth -> d``."""
    if embed_dim <= 0:
        return 0
    return readout_param_count(particle_dim, hidden, depth, embed_dim)


def non_readout_param_count(num_cgf_features: int, particle_dim: int,
                            feature_mode: str, t_frozen: bool,
                            feature_norm: str, x_embed_dim: int = 0,
                            x_embed_hidden: int = 0, x_embed_depth: int = 0) -> int:
    """Encoder parameters outside the readout: the learned t (unless frozen),
    LayerNorm's affine pair (the running norm owns buffers only) and the
    particle embedding when one is used. ``cgf_raw_dim`` must be called with
    the embedded dimension when ``x_embed_dim > 0``."""
    t_dim = x_embed_dim if x_embed_dim > 0 else particle_dim
    count = 0 if t_frozen else num_cgf_features * t_dim
    if feature_norm == "layernorm":
        count += 2 * cgf_raw_dim(num_cgf_features, t_dim, feature_mode)
    count += x_embed_param_count(particle_dim, x_embed_hidden, x_embed_depth, x_embed_dim)
    return count


def matched_readout_hidden(target_params: int, raw_dim: int, depth: int,
                           out_dim: int, fixed_params: int = 0,
                           max_hidden: int = 8192) -> tuple[int, int]:
    """The readout width whose encoder total lands closest to ``target_params``.

    Returns ``(hidden, total)`` with ``total = fixed_params + readout``. Depth
    must be >= 1. The count is exact, so the two CGF arms and the ST arm can
    be reported at their true totals rather than "about 100k".
    """
    if depth < 1:
        raise ValueError("matched_readout_hidden needs readout_depth >= 1")
    if target_params <= fixed_params:
        raise ValueError(
            f"target {target_params} is not above the non-readout parameter "
            f"count {fixed_params}; nothing is left for a readout")
    best = min(range(1, max_hidden + 1),
               key=lambda h: abs(fixed_params + readout_param_count(raw_dim, h, depth, out_dim)
                                 - target_params))
    return best, fixed_params + readout_param_count(raw_dim, best, depth, out_dim)


class WeightedCGFFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor for weighted empirical CGF particle features."""

    #: Geometry fields a pretrained checkpoint must agree on. ``t_frozen`` is
    #: deliberately NOT here: a checkpoint whose t was learned under the
    #: supervised objective loads into an extractor that then freezes it --
    #: that is the frozen RL arm -- and the state_dict key is the same either
    #: way. ``arena_scale`` is checked because the t values only mean
    #: anything in the coordinates they were fitted in.
    #: Readout output width when none is given: the ST arm's 8 x 8 = 64.
    DEFAULT_READOUT_DIM = 64

    CHECKED_GEOMETRY = ("num_cgf_features", "particle_dim", "feature_mode",
                        "t_param", "t_bound", "t_clamp", "feature_norm",
                        "readout_hidden", "readout_depth", "readout_dim",
                        "arena_scale", "x_embed_dim", "x_embed_hidden",
                        "x_embed_depth")

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_cgf_features: int = 64,
        arena_scale: float = 4.5,
        t_init_mode: str = "linspace_first_dim",
        t_init_scale: float = 0.1,
        t_clamp: float = 2.0,
        exp_arg_clamp: float = 20.0,
        t_frozen: bool = False,
        t_param: str = "clamp",
        t_bound: float | None = None,
        t_init_max: float | None = None,
        feature_mode: str = "K",
        feature_norm: str = "none",
        readout_hidden: int = 0,
        readout_depth: int = 0,
        readout_dim: int | None = None,
        pretrained_cgf_model_path: str | None = None,
        cgf_frozen: bool = False,
        x_embed_dim: int = 0,
        x_embed_hidden: int = 64,
        x_embed_depth: int = 1,
    ):
        obs_dim = observation_space["obs"].shape[0]
        particle_dim = observation_space["particles"].shape[1]
        x_embed_dim = int(x_embed_dim or 0)
        x_embed_hidden = int(x_embed_hidden or 0)
        x_embed_depth = int(x_embed_depth or 0)
        if x_embed_dim > 0 and (x_embed_hidden <= 0 or x_embed_depth <= 0):
            raise ValueError("x_embed_dim > 0 needs x_embed_hidden > 0 and x_embed_depth > 0")
        # t lives in the embedded space when there is one. Everything below
        # that sizes t or the CGF block by "particle_dim" uses t_dim, and the
        # raw particle_dim only feeds the embedding's first layer.
        self.raw_particle_dim = particle_dim
        t_dim = x_embed_dim if x_embed_dim > 0 else particle_dim
        readout_hidden = int(readout_hidden or 0)
        readout_depth = int(readout_depth or 0)
        if (readout_hidden > 0) != (readout_depth > 0):
            raise ValueError(
                "readout_hidden and readout_depth must both be > 0 (an MLP "
                f"readout) or both 0 (none); got {readout_hidden}, {readout_depth}")
        use_readout = readout_depth > 0
        if feature_mode not in ("K", "K_grad", "both"):
            raise ValueError(
                f"feature_mode must be 'K', 'K_grad' or 'both', got {feature_mode!r}")
        if t_param not in ("clamp", "tanh"):
            raise ValueError(f"t_param must be 'clamp' or 'tanh', got {t_param!r}")
        if feature_norm not in ("none", "running", "layernorm"):
            raise ValueError(
                f"feature_norm must be 'none', 'running' or 'layernorm', "
                f"got {feature_norm!r}")
        if t_param == "tanh" and (t_bound is None or t_bound <= 0):
            raise ValueError(
                "t_param='tanh' needs a positive t_bound. Pick it by "
                "t_bound * (decision-relevant length in normalized units) ~ 2..4; "
                "Odd-Even oe50 (spacing 0.0816) -> 50, ClusterHunt (0.3) -> 10.")
        num_encoded = cgf_raw_dim(num_cgf_features, t_dim, feature_mode)
        if readout_dim is None:
            # The ST arm emits num_encodings * dim_encoder = 64 features; a
            # readout defaults to the same width so the policy heads (and the
            # pretraining head) are identical across arms WHATEVER the number
            # of probes. Without a readout the block's own width is what
            # reaches the policy.
            readout_dim = self.DEFAULT_READOUT_DIM if use_readout else num_encoded
        readout_dim = int(readout_dim)
        if not use_readout and readout_dim != num_encoded:
            raise ValueError(
                f"readout_dim={readout_dim} needs a readout MLP; without one the "
                f"CGF block's own width {num_encoded} is the output")
        super().__init__(observation_space,
                         features_dim=obs_dim + (readout_dim if use_readout else num_encoded))
        self.feature_mode = feature_mode
        self.readout_hidden = readout_hidden
        self.readout_depth = readout_depth
        self.readout_dim = readout_dim
        self.num_encoded = num_encoded
        if use_readout:
            dims = [num_encoded] + [readout_hidden] * readout_depth + [readout_dim]
            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.GELU())
            self.readout: nn.Module = nn.Sequential(*layers)
        else:
            self.readout = nn.Identity()
        self.t_param = t_param
        self.t_bound = None if t_bound is None else float(t_bound)
        self.t_init_max = None if t_init_max is None else float(t_init_max)
        self.feature_norm_mode = feature_norm
        if feature_norm == "running":
            self.feature_norm = RunningFeatureNorm(num_encoded)
        elif feature_norm == "layernorm":
            self.feature_norm = nn.LayerNorm(num_encoded)
        else:
            self.feature_norm = nn.Identity()

        self.num_cgf_features = num_cgf_features
        self.particle_dim = t_dim
        self.x_embed_dim = x_embed_dim
        self.x_embed_hidden = x_embed_hidden if x_embed_dim > 0 else 0
        self.x_embed_depth = x_embed_depth if x_embed_dim > 0 else 0
        if x_embed_dim > 0:
            dims = [particle_dim] + [x_embed_hidden] * x_embed_depth + [x_embed_dim]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.GELU())
            self.x_embed: nn.Module = nn.Sequential(*layers)
        else:
            self.x_embed = nn.Identity()
        # The t-init code below sizes t by `particle_dim`; rebind it to the
        # space t actually lives in.
        particle_dim = t_dim
        self.arena_scale = arena_scale
        self.t_clamp = t_clamp
        # DEPRECATED, not applied. The CGF is computed with logsumexp (see
        # forward), which needs no clamp on the exponent. The argument is kept
        # because SB3 stores ``features_extractor_kwargs`` in every saved
        # checkpoint and passes them back to this constructor on load, and
        # because run_config.json records it. Removing it would break loading
        # of ~1250 saved CGF checkpoints.
        self.exp_arg_clamp = exp_arg_clamp

        if t_init_mode == "linspace_all_dims":
            # Round-robin the linspace directions across every particle
            # dimension so each coordinate (not just dim 0) gets a nontrivial
            # initial CGF sensitivity.
            t_values = torch.zeros(num_cgf_features, particle_dim)
            linspace_vals = torch.linspace(-t_init_scale, t_init_scale, num_cgf_features)
            dim_assignment = torch.arange(num_cgf_features) % particle_dim
            for d in range(particle_dim):
                mask = dim_assignment == d
                t_values[mask, d] = linspace_vals[mask]
        elif t_init_mode == "linspace_first_dim":
            t_values = torch.zeros(num_cgf_features, particle_dim)
            t_values[:, 0] = torch.linspace(
                -t_init_scale,
                t_init_scale,
                num_cgf_features,
            )
            if particle_dim > 1:
                t_values[:, 1:] = 0.01 * torch.randn(
                    num_cgf_features,
                    particle_dim - 1,
                )
        elif t_init_mode == "spread":
            # 8 directions x (num//8) log-spaced norms, matching the probe's
            # CGF_SPREAD64 feature geometry. rho_hi=2.8 ~ the max norm
            # reachable under the elementwise t_clamp=2.0 (diagonal norm
            # 2*sqrt(2)). The den diagonal (45 deg) is one of the 8
            # directions exactly. Unlike the legacy 0.1-scale linspace init,
            # this starts where the signal actually is, so no ~10x growth of
            # ||t_j|| is required for the encoder to see it.
            if particle_dim != 2:
                raise ValueError("t_init_mode='spread' assumes 2D particles")
            num_dirs = 8
            if num_cgf_features % num_dirs != 0:
                raise ValueError(
                    "num_cgf_features must be divisible by 8 for 'spread'")
            num_norms = num_cgf_features // num_dirs
            angles = torch.arange(num_dirs, dtype=torch.float32) * (
                2 * torch.pi / num_dirs)
            dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            # t_init_max overrides the 2.8 (e.g. 0.8 * t_bound in tanh mode,
            # so the init spans the range the bound allows).
            rho_hi = 2.8 if t_init_max is None else float(t_init_max)
            norms = torch.tensor(np.geomspace(0.25, rho_hi, num_norms),
                                 dtype=torch.float32)
            t_values = (norms[None, :, None] * dirs[:, None, :]).reshape(
                -1, particle_dim)
        elif t_init_mode == "spread_1d":
            # The 1-D analogue of "spread", for a scalar state space such as
            # the Odd-Even POMDP's integer hidden state. A SEPARATE mode
            # rather than a widening of "spread": that mode's geometry is
            # intrinsically planar (8 directions from angles 2*pi*k/8) and
            # its rho_hi=2.8 is the largest norm the ELEMENTWISE t_clamp=2.0
            # permits in 2-D (the diagonal, 2*sqrt(2)). In 1-D there are
            # exactly two directions and that same clamp bounds ||t|| at
            # 2.0, so neither number carries over. Overloading one flag name
            # across two unrelated geometries would also make run_config.json
            # ambiguous about what a run actually built, and would silently
            # change a mode that hundreds of saved 2-D checkpoints were
            # built with.
            if particle_dim != 1:
                raise ValueError(
                    "t_init_mode='spread_1d' assumes 1D particles; use "
                    "'spread' for the 2D case")
            if num_cgf_features % 2 != 0:
                raise ValueError(
                    "num_cgf_features must be even for 'spread_1d' (the "
                    "magnitudes are mirrored into both signs)")
            num_norms = num_cgf_features // 2
            # 2.0, not 2.8: in 1-D the elementwise clamp IS the norm bound.
            # t_init_max overrides it (e.g. 40 with t_bound=50 in tanh mode:
            # the log-spaced init then covers both the mean/variance regime
            # at small t and the support-edge regime at large t).
            rho_hi = 2.0 if t_init_max is None else float(t_init_max)
            norms = torch.tensor(np.geomspace(0.25, rho_hi, num_norms),
                                 dtype=torch.float32)
            # Both signs, so the CGF sees the belief's mass on either side of
            # the origin. Like "spread", this starts where the signal already
            # is, so PPO needs no ~10x growth of ||t_j|| to resolve it.
            t_values = torch.cat([norms, -norms]).reshape(-1, particle_dim)
        elif t_init_mode == "random":
            t_values = t_init_scale * torch.randn(num_cgf_features, particle_dim)
        else:
            raise ValueError(f"Unknown t_init_mode: {t_init_mode}")

        self.t_frozen = bool(t_frozen)
        if self.t_param == "tanh":
            # Store the pre-activation; effective_t() applies the bound. The
            # init is mapped through atanh so the FIRST forward pass sees
            # exactly the requested t (up to the 1e-5 saturation guard). An
            # init magnitude at or above t_bound would start on the flat part
            # of tanh with ~zero gradient -- that is what t_init_max is for.
            if float(t_values.abs().max()) >= self.t_bound:
                raise ValueError(
                    f"t init reaches {float(t_values.abs().max()):.3g} but "
                    f"t_bound is {self.t_bound}; probes starting at the bound "
                    "sit on tanh's flat region and never move. Lower "
                    "t_init_max / t_init_scale or raise t_bound.")
            scaled = (t_values / self.t_bound).clamp(-1 + 1e-5, 1 - 1e-5)
            raw_t = torch.atanh(scaled)
            if self.t_frozen:
                self.register_buffer("raw_t", raw_t)
            else:
                self.raw_t = nn.Parameter(raw_t)
        elif self.t_frozen:
            # A buffer gets no gradient (so PPO cannot move it) while still
            # saving/loading and moving across devices exactly like the
            # Parameter. forward() is untouched: on the spread init the
            # elementwise clamp is a no-op by construction.
            self.register_buffer("t_values", t_values)
        else:
            self.t_values = nn.Parameter(t_values)

        # What a pretrained checkpoint must agree on (CHECKED_GEOMETRY) plus
        # the fields recorded for the reader. Written into every checkpoint's
        # "config" by the pretraining script.
        self._cgf_geometry = dict(
            num_cgf_features=int(num_cgf_features),
            particle_dim=int(particle_dim),
            feature_mode=feature_mode,
            t_param=t_param,
            t_bound=self.t_bound,
            t_clamp=float(t_clamp),
            t_frozen=self.t_frozen,
            feature_norm=feature_norm,
            readout_hidden=readout_hidden,
            readout_depth=readout_depth,
            readout_dim=readout_dim,
            arena_scale=float(arena_scale),
            t_init_mode=t_init_mode,
            x_embed_dim=self.x_embed_dim,
            x_embed_hidden=self.x_embed_hidden,
            x_embed_depth=self.x_embed_depth,
        )

        self._frozen = False
        self.pretrained_cgf_model_path = pretrained_cgf_model_path
        if pretrained_cgf_model_path:
            self._load_pretrained_encoder(pretrained_cgf_model_path)
        if cgf_frozen:
            if not pretrained_cgf_model_path:
                # Not an error here: SB3 re-runs this constructor when a SAVED
                # policy is loaded, with the pretraining path scrubbed
                # (PITFALLS.md section 7) and the trained weights arriving from
                # the zip a moment later. The CLI guard against freezing a
                # random readout lives in 4_train_rl_cgf.py.
                print("WeightedCGFFeaturesExtractor: cgf_frozen without a "
                      "checkpoint path; freezing the weights the caller loads "
                      "next (a saved policy), not a fresh readout.")
            self.freeze_encoder()

    # ------------------------------------------------------------------
    # the extractor as one "encoder" object: parameters, freeze, checkpoint
    # ------------------------------------------------------------------

    def encoder_parameter_count(self, trainable_only: bool = False) -> int:
        """t (if learned) + norm affine (LayerNorm only) + readout."""
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad or not trainable_only)

    def freeze_encoder(self) -> None:
        """Fix t, the norm statistics and the readout. Idempotent.

        ``requires_grad_(False)`` on every parameter handles t (a Parameter
        in the learned case; a buffer never had a gradient) and the readout.
        The running norm has no parameters but UPDATES ITS BUFFERS in
        training mode, so ``train()`` is overridden below to hold the module
        in eval mode whatever SB3 sets on the policy. Together these are what
        ``cgf_frozen`` means: the extractor is a fixed function.
        """
        self._frozen = True
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        """A frozen encoder stays in eval mode whatever the policy does."""
        return super().train(False if getattr(self, "_frozen", False) else mode)

    def _check_checkpoint_geometry(self, config, path: str) -> None:
        if not isinstance(config, dict):
            return
        mismatches = []
        for field in self.CHECKED_GEOMETRY:
            if field not in config:
                continue
            expected, actual = self._cgf_geometry[field], config[field]
            if isinstance(expected, float) or isinstance(actual, float):
                same = (expected is None) == (actual is None) and (
                    expected is None or math.isclose(float(expected), float(actual),
                                                     rel_tol=1e-6, abs_tol=1e-9))
            else:
                same = expected == actual
            if not same:
                mismatches.append(f"{field}: checkpoint={actual!r}, this run={expected!r}")
        if mismatches:
            raise RuntimeError(
                f"Checkpoint {path} was pretrained with a different CGF encoder "
                "geometry than this run requests:\n  " + "\n  ".join(mismatches)
                + "\nPass the matching --feature_mode / --t_param / --t_bound / "
                "--num_cgf_features / --feature_norm / --readout_* / --arena_scale "
                "flags (4_train_rl_cgf.py takes them from the checkpoint when "
                "the flags are left at their defaults).")

    def _load_pretrained_encoder(self, path: str) -> None:
        """Load t, norm statistics and readout from a pretraining checkpoint.

        Accepts the dict ``3_pretrain_st_belief.py --encoder cgf`` writes
        (``model_state_dict`` keyed exactly like this module's state_dict,
        plus ``config``) or a bare state_dict. Strict: every key must match,
        so a geometry the config check cannot see still cannot load silently.
        """
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
            self._check_checkpoint_geometry(loaded.get("config"), path)
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            raise ValueError(f"Expected a checkpoint dict at {path}, got {type(loaded)}")
        try:
            self.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load the CGF encoder from {path}; its geometry must "
                f"match this run's ({self._cgf_geometry}).\nOriginal error: {exc}"
            ) from exc
        print(f"WeightedCGFFeaturesExtractor: loaded encoder from {path}")

    def effective_t(self) -> torch.Tensor:
        """The [T, D] probe matrix the forward pass uses.

        ``clamp`` mode: ``clamp(t_values, -t_clamp, t_clamp)`` -- the legacy
        behaviour, gradient zero beyond the bound. ``tanh`` mode:
        ``t_bound * tanh(raw_t)`` -- smooth, never reaches the bound.
        """
        if self.t_param == "tanh":
            return self.t_bound * torch.tanh(self.raw_t)
        return torch.clamp(self.t_values, -self.t_clamp, self.t_clamp)

    #: Feature value for a particle set with NO mass at all (every weight zero
    #: after sanitising). This is log(1e-8), the constant the pre-logsumexp
    #: forward produced for such rows through ``log(clamp(sum, min=1e-8))``;
    #: kept so the ``zero_mass`` golden in the Ant-Tag regression gate and the
    #: behaviour of every saved checkpoint on a dead filter are unchanged.
    DEAD_ROW_VALUE = math.log(1e-8)

    def _forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        base_obs = obs_dict["obs"]
        particles = obs_dict["particles"] / self.arena_scale
        weights = obs_dict["weights"]

        particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = torch.clamp(weights, min=0.0)
        mass = weights.sum(dim=1, keepdim=True)                       # [B, 1]
        # The ``+ 1e-8`` is load-bearing: the regression gate pins its
        # magnitude (``near_epsilon_mass``). Do not "tidy" it.
        weights = weights / (mass + 1e-8)

        # K(t_j) = log sum_i w_i exp(<t_j, x_i>), computed as
        #
        #     logsumexp_i ( <t_j, x_i> + log w_i )
        #
        # The same quantity as exp -> weighted sum -> log, but the maximum is
        # subtracted before anything is exponentiated, so no exponent magnitude
        # can overflow or underflow and the exponent needs no clamp. The old
        # ``exp_arg_clamp`` silently flattened every feature once |<t, x>|
        # passed 20 -- at |t| = 50 on [-1, 1] particles the two paths differed
        # by 31.5 (domain_mds/oddeven.md, 2026-09-05) -- and is no longer
        # applied. On the Ant-Tag ranges (|t| <= 2.8, |x| <= ~1.6) neither
        # clamp ever bound, so the golden numerics are unchanged.
        #
        # log(0) = -inf for a zero-weight particle is correct and intended:
        # exp(-inf) is exactly 0 inside logsumexp and its gradient is exactly
        # 0, so a refuted particle contributes nothing, as before. The one case
        # that must not reach logsumexp is a row with NO mass -- every entry
        # -inf, and the backward pass is 0/0. Those rows get uniform weights
        # for the computation (finite, so the gradient is finite) and the
        # constant DEAD_ROW_VALUE as output, which is what the old floor gave.
        dead = mass <= 0.0                                            # [B, 1]
        num_particles = weights.shape[1]
        safe_weights = torch.where(
            dead, torch.full_like(weights, 1.0 / num_particles), weights)
        log_w = safe_weights.log()                                    # [B, N]

        t = self.effective_t()
        # Learned per-particle embedding, identity when x_embed_dim == 0.
        particles = self.x_embed(particles)                             # [B, N, t_dim]
        # One score tensor serves K and K': score_ij = <t_j, x_i> + log w_i.
        scores = torch.matmul(particles, t.transpose(0, 1)) + log_w.unsqueeze(-1)  # [B, N, T]
        parts = []
        if self.feature_mode in ("K", "both"):
            k = torch.logsumexp(scores, dim=1)                        # [B, T]
            k = torch.where(dead, torch.full_like(k, self.DEAD_ROW_VALUE), k)
            parts.append(k)
        if self.feature_mode in ("K_grad", "both"):
            # K'(t_j) = sum_i softmax_i(score_ij) x_i: the belief's mean after
            # tilting it by exp(<t_j, x>). t = 0 gives the plain mean; large
            # |t| walks to the support edge in direction t. Bounded by the
            # particle range whatever t is, unlike K. A dead row has uniform
            # stand-in weights, so this is finite there too.
            tilt = torch.softmax(scores, dim=1)                       # [B, N, T]
            k_grad = torch.einsum("bnt,bnd->btd", tilt, particles)   # [B, T, D]
            parts.append(k_grad.reshape(k_grad.shape[0], -1))         # [B, T*D]
        cgf = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        cgf = self.readout(self.feature_norm(cgf))
        return torch.cat([base_obs, cgf], dim=-1)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: F811
        if self._frozen:
            with torch.no_grad():
                return self._forward(obs_dict)
        return self._forward(obs_dict)
