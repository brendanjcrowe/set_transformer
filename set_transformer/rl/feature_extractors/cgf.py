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
``t_param="polar"`` (2026-09-10, for particle_dim >= 2)
    ``t_j = t_bound * sigmoid(raw_a_j) * raw_v_j / (||raw_v_j|| + 1e-8)``:
    a free direction vector and a scalar magnitude per probe, so the bound
    is a BALL of radius ``t_bound`` and direction and tilt strength get
    separate gradients. The elementwise tanh is a box whose diagonal reaches
    ``t_bound * sqrt(D)`` and mixes the two roles; ``t_bound`` in polar mode
    is therefore the exact largest tilt, which is what the sizing rule above
    assumes (personal_mds_hg/how_to_do_multidimensional_cgf_particle_rl_chatgpt.md
    section 7). ``dr/da = r (1 - r / t_bound)``, so small probes move
    multiplicatively and probes near the bound slow down; the spread init
    caps at ``t_init_max`` below the bound for that reason. In 1-D the
    direction is a sign that ``v / |v|`` cannot flip smoothly, so polar is
    refused there and tanh (the same function in 1-D) is the mode to use.
    The Ant-Tag frame is invariant under this: translating every particle by
    a constant adds ``-t . c`` to K and ``-c`` to K' and leaves the tilted
    weights unchanged, so the bound does not depend on agent-centring.
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

    **Two update modes** (``update``):

    ``"minibatch"`` (constructor default)
        The statistics lerp towards each training-mode batch (momentum
        ``momentum``) and the SAME forward normalises with the updated
        values, as BatchNorm's running buffers do. Right for a supervised
        loop (``3_pretrain_st_belief.py --encoder cgf``), where nothing
        compares a stored output against a recomputed one. WRONG under PPO:
        an update makes ``n_epochs * n_steps * n_envs / batch_size`` = 1280
        train-mode forwards, so the statistics turn over inside the update
        (``0.99 ** 1280`` of the rollout's values survive) and the stored
        ``old_log_prob`` and the recomputed log-probs are on differently
        standardised features -- PPO's ratio then moves without the
        parameters moving (domain_mds/PITFALLS.md section 8 item 5).
    ``"rollout"``
        The module NEVER updates itself; forward only normalises. The owner
        sets the statistics through :meth:`set_statistics`, which
        :class:`RolloutFeatureNormCallback` does exactly once per PPO cycle,
        between an update and the next collection. Within a collection and
        the update trained on it the statistics are constant, so stored and
        recomputed log-probs see the same normalisation -- the discipline
        SB3's ``VecNormalize`` applies to observations. The Odd-Even CGF RL
        arm flips the module into this mode at training start.

    In both modes the RUNNING statistics (not the batch's) normalise the
    output, and eval mode never updates anything, so a saved policy is a
    fixed function whatever mode trained it. The first minibatch update
    copies the batch statistics outright rather than lerping from (0, 1).
    """

    UPDATE_MODES = ("minibatch", "rollout")

    def __init__(self, num_features: int, momentum: float = 0.01,
                 eps: float = 1e-5, update: str = "minibatch"):
        super().__init__()
        if update not in self.UPDATE_MODES:
            raise ValueError(f"update must be one of {self.UPDATE_MODES}, got {update!r}")
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.update = update
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.update == "minibatch" and self.training and x.shape[0] > 1:
            with torch.no_grad():
                self.set_statistics(x.mean(dim=0), x.var(dim=0, unbiased=False),
                                    momentum=self.momentum)
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

    @torch.no_grad()
    def set_statistics(self, mean: torch.Tensor, var: torch.Tensor,
                       momentum: float = 1.0) -> None:
        """Move the running statistics towards (mean, var).

        ``momentum=1.0`` adopts them outright. The very first call always
        copies, whatever the momentum, so the features are on scale from the
        first use instead of after ``1 / momentum`` updates.
        """
        mean = mean.to(self.running_mean)
        var = var.to(self.running_var)
        if int(self.num_updates) == 0 or momentum >= 1.0:
            self.running_mean.copy_(mean)
            self.running_var.copy_(var)
        else:
            self.running_mean.lerp_(mean, float(momentum))
            self.running_var.lerp_(var, float(momentum))
        self.num_updates += 1


def _policy_extractors(policy) -> list:
    """The features extractor(s) a policy owns, without duplicates.

    PPO's ActorCriticPolicy shares one; ``share_features_extractor=False``
    or an off-policy actor/critic pair hold several.
    """
    extractors = []
    for candidate in [getattr(policy, "features_extractor", None)] + [
            getattr(getattr(policy, attr, None), "features_extractor", None)
            for attr in ("pi_features_extractor", "vf_features_extractor",
                         "actor", "critic", "critic_target")]:
        if candidate is not None and all(candidate is not e for e in extractors):
            extractors.append(candidate)
    return extractors


class RolloutFeatureNormCallback(BaseCallback):
    """Refresh a ``RunningFeatureNorm`` once per PPO cycle, never inside one.

    PPO alternates COLLECT (``n_steps`` env steps per worker, policy in eval
    mode, actions' log-probs stored) and UPDATE (``n_epochs`` passes over
    that rollout in minibatches, each recomputing the stored actions'
    log-probs and clipping the ratio to the stored ones). The ratio is only
    meaningful if the two log-probs differ ONLY through the parameters, so
    the standardisation the CGF block goes through must be identical in
    both phases. This callback guarantees that:

    * ``_on_training_start``: every ``RunningFeatureNorm`` in an UNFROZEN
      extractor is switched to ``update="rollout"`` (it stops updating
      itself). A frozen extractor is left alone -- ``freeze_encoder`` already
      pins it in eval mode and its statistics are part of the checkpoint.
    * ``_on_rollout_end`` (after collection, before the update): the raw
      pre-norm CGF block is recomputed over EXACTLY the rollout buffer's
      observations (same re-encoding the ST sentinel does) and its
      per-feature mean / variance are stashed. NOT applied yet: the update
      about to run must use the statistics the rollout was collected under.
    * ``_on_rollout_start`` (after the update): the stashed statistics are
      applied (``momentum`` = 1.0 adopts them outright; < 1 lerps). So each
      cycle runs under statistics estimated from the previous rollout, held
      fixed for the whole cycle.

    The very first rollout runs on the initial (0, 1) statistics, i.e. raw
    features, consistently in both of its phases; the first refresh follows
    its update. Logged per rollout: ``cgf/norm_refreshes`` (count),
    ``cgf/norm_mean_shift`` (mean over features of |new - old mean| / old
    std, how far the standardisation moved), ``cgf/norm_std_ratio_mean``
    (mean over features of new std / old std) and ``cgf/norm_samples``.
    """

    def __init__(self, momentum: float = 1.0, verbose: int = 0):
        super().__init__(verbose)
        if not 0.0 < momentum <= 1.0:
            raise ValueError(f"momentum must be in (0, 1], got {momentum}")
        self.momentum = float(momentum)
        self._norms: list = []
        self._pending = None
        self.refreshes = 0
        self.last_stats: dict = {}

    # -- lifecycle -------------------------------------------------------
    def _on_training_start(self) -> None:
        self._norms = []
        for extractor in _policy_extractors(self.model.policy):
            if getattr(extractor, "_frozen", False):
                continue
            norm = getattr(extractor, "feature_norm", None)
            if isinstance(norm, RunningFeatureNorm):
                norm.update = "rollout"
                self._norms.append((extractor, norm))
        if self.verbose and not self._norms:
            print("RolloutFeatureNormCallback: no unfrozen RunningFeatureNorm "
                  "on this policy; idle")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if not self._norms:
            return
        stats = self._rollout_raw_statistics()
        if stats is not None:
            self._pending = stats

    def _on_rollout_start(self) -> None:
        if self._pending is None:
            return
        mean, var, n = self._pending
        self._pending = None
        for _extractor, norm in self._norms:
            old_mean = norm.running_mean.clone()
            old_std = torch.sqrt(norm.running_var + norm.eps)
            norm.set_statistics(mean, var, momentum=self.momentum)
            shift = float(((norm.running_mean - old_mean).abs() / old_std).mean())
            ratio = float((torch.sqrt(norm.running_var + norm.eps) / old_std).mean())
        self.refreshes += 1
        self.last_stats = {"cgf/norm_refreshes": self.refreshes,
                           "cgf/norm_mean_shift": shift,
                           "cgf/norm_std_ratio_mean": ratio,
                           "cgf/norm_samples": n}
        for key, value in self.last_stats.items():
            self.logger.record(key, value)

    # -- the statistic ---------------------------------------------------
    def _rollout_raw_statistics(self):
        """Mean / var of the PRE-norm CGF block over the rollout buffer.

        Uses the primary extractor; a second (unshared) extractor has the
        same t only if it was constructed identically, and PPO's default
        shares one anyway.
        """
        from stable_baselines3.common.utils import obs_as_tensor

        extractor, _norm = self._norms[0]
        buffer = getattr(self.model, "rollout_buffer", None)
        if buffer is None or not isinstance(buffer.observations, dict):
            return None
        flat = {key: np.asarray(value).reshape(-1, *value.shape[2:])
                for key, value in buffer.observations.items()}
        n = next(iter(flat.values())).shape[0]
        total = None
        total_sq = None
        was_training = extractor.training
        extractor.eval()
        with torch.no_grad():
            for start in range(0, n, 1024):
                batch = {key: value[start:start + 1024] for key, value in flat.items()}
                raw = extractor.raw_cgf_features(
                    obs_as_tensor(batch, self.model.device)).double()
                total = raw.sum(dim=0) if total is None else total + raw.sum(dim=0)
                sq = (raw * raw).sum(dim=0)
                total_sq = sq if total_sq is None else total_sq + sq
        extractor.train(was_training)
        mean = total / n
        var = torch.clamp(total_sq / n - mean * mean, min=0.0)
        return mean.float(), var.float(), int(n)


class EncoderDriftLoggingCallback(BaseCallback):
    """Log how far the extractor's parameters have moved since training start.

    Per rollout: ``cgf/drift_<name>`` = ||p - p_0|| / ||p_0|| (over ||p||
    when p_0 = 0, so an initially-zero buffer reads ~1 once set, not 1e12) for
    each top-level parameter group of the features extractor (``raw_t`` or
    ``t_values``, or ``raw_v`` + ``raw_a`` in polar mode, ``readout``,
    ``x_embed``) and ``cgf/drift_feature_norm``
    for the running-norm mean buffer (its variance and counter are skipped:
    the relative norm of a counter means nothing). A frozen encoder logs
    exactly 0 on every
    line; ``--t_frozen`` alone logs 0 for t and > 0 for the readout. This is
    the direct check that PPO IS updating the encoder weights in the
    finetuned arm, next to :class:`TNormLoggingCallback`'s ||t|| quantiles.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._reference: dict[str, torch.Tensor] = {}
        self.last_stats: dict = {}

    @staticmethod
    def _groups(extractor) -> dict[str, list[torch.Tensor]]:
        groups: dict[str, list[torch.Tensor]] = {}
        for name, tensor in list(extractor.named_parameters()) + list(
                extractor.named_buffers()):
            if name.endswith(("num_updates", "running_var")):
                continue
            key = name.split(".")[0] if "." in name else name
            groups.setdefault(key, []).append(tensor.detach())
        return groups

    def _on_training_start(self) -> None:
        extractor = getattr(self.model.policy, "features_extractor", None)
        if extractor is None:
            return
        self._reference = {
            key: torch.cat([t.flatten().clone().cpu() for t in tensors])
            for key, tensors in self._groups(extractor).items()}

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        extractor = getattr(self.model.policy, "features_extractor", None)
        if extractor is None or not self._reference:
            return
        self.last_stats = {}
        for key, tensors in self._groups(extractor).items():
            ref = self._reference.get(key)
            if ref is None:
                continue
            now = torch.cat([t.flatten().cpu() for t in tensors])
            # Relative to the starting value; a group that STARTS at zero
            # (the running norm's mean before its first refresh) is measured
            # against its current norm instead, so it reads ~1 once it has
            # moved rather than 1e12 -- and 0 if it never moves.
            ref_norm = float(torch.linalg.norm(ref))
            denom = ref_norm if ref_norm > 0 else float(torch.linalg.norm(now))
            value = float(torch.linalg.norm(now - ref)) / denom if denom > 0 else 0.0
            self.last_stats[f"cgf/drift_{key}"] = value
            self.logger.record(f"cgf/drift_{key}", value)


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
        if t_param not in ("clamp", "tanh", "polar"):
            raise ValueError(
                f"t_param must be 'clamp', 'tanh' or 'polar', got {t_param!r}")
        if feature_norm not in ("none", "running", "layernorm"):
            raise ValueError(
                f"feature_norm must be 'none', 'running' or 'layernorm', "
                f"got {feature_norm!r}")
        if t_param in ("tanh", "polar") and (t_bound is None or t_bound <= 0):
            raise ValueError(
                f"t_param={t_param!r} needs a positive t_bound. Pick it by "
                "t_bound * (decision-relevant length in normalized units) ~ 2..4; "
                "Odd-Even oe50 (spacing 0.0816) -> 50, ClusterHunt (0.3) -> 10, "
                "Ant-Tag smart_mid_slow_v15 (tag radius 1.0 / 4.5 = 0.22) -> 13.")
        if t_param == "polar" and t_dim < 2:
            raise ValueError(
                "t_param='polar' needs particle_dim >= 2: in 1-D the direction "
                "is a sign that v / |v| cannot flip smoothly. Use t_param='tanh', "
                "which is the same function in 1-D.")
        if (t_param == "clamp" and t_init_max is not None
                and float(t_init_max) > float(t_clamp)):
            raise ValueError(
                f"t_init_max={t_init_max} exceeds t_clamp={t_clamp} in clamp mode: "
                "every probe beyond the clamp would be silently flattened to "
                "+-t_clamp with zero gradient. Lower t_init_max, raise t_clamp, "
                "or use t_param='tanh' / 'polar' with a t_bound.")
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
            # reachable under the elementwise t_clamp=2.0 on the DIAGONAL
            # (2*sqrt(2)). On the 4 axis-aligned directions a norm-2.8 probe
            # has a component of 2.8, which clamp mode flattens to 2.0 in the
            # forward pass, so 4 of 64 probes sit on the clamp with zero
            # gradient and near-duplicate the norm-1.98 ring (measured
            # 2026-09-09; pinned by tests/test_ant_tag_cgf_port.py). Every
            # recorded Ant-Tag `spread` run has this; tanh / polar do not.
            # The den diagonal (45 deg) is one of the 8 directions exactly.
            # Unlike the legacy 0.1-scale linspace init, this starts where
            # the signal actually is, so no ~10x growth of ||t_j|| is
            # required for the encoder to see it.
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
        if self.t_param == "polar":
            # t_j = t_bound * sigmoid(a_j) * v_j / ||v_j||. Init from the
            # requested t: v_j is its unit direction, a_j = logit(r_j /
            # t_bound), so the first forward pass reproduces the init exactly
            # (up to the 1e-5 saturation guard, as in tanh mode). A zero
            # init row (linspace_* modes place one probe at t = 0) has no
            # direction; it gets e_0 and the smallest representable
            # magnitude, which sigmoid can grow.
            radii = torch.linalg.norm(t_values, dim=1)                  # [T]
            if float(radii.max()) >= self.t_bound:
                raise ValueError(
                    f"t init reaches norm {float(radii.max()):.3g} but t_bound "
                    f"is {self.t_bound}; probes starting at the bound sit on "
                    "sigmoid's flat region and never move. Lower t_init_max / "
                    "t_init_scale or raise t_bound.")
            degenerate = radii < 1e-8
            directions = torch.where(
                degenerate[:, None],
                torch.nn.functional.one_hot(
                    torch.zeros(len(radii), dtype=torch.long), particle_dim).float(),
                t_values / radii.clamp(min=1e-8)[:, None])
            frac = (radii / self.t_bound).clamp(1e-5, 1 - 1e-5)
            raw_a = torch.log(frac) - torch.log1p(-frac)                # logit
            if self.t_frozen:
                self.register_buffer("raw_v", directions)
                self.register_buffer("raw_a", raw_a)
            else:
                self.raw_v = nn.Parameter(directions)
                self.raw_a = nn.Parameter(raw_a)
        elif self.t_param == "tanh":
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
            # Parameter. forward() is untouched, so the elementwise clamp
            # still applies to a frozen t: on the 2-D spread init that
            # flattens the 4 axis-aligned norm-2.8 probes to 2.0 (see the
            # spread init above); on spread_1d (rho_hi 2.0) it is a no-op.
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
            t_init_max=self.t_init_max,
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
        ``t_bound * tanh(raw_t)`` -- smooth, never reaches the bound (a box).
        ``polar`` mode: ``t_bound * sigmoid(raw_a) * raw_v / ||raw_v||`` --
        smooth, never reaches the bound (a ball).
        """
        if self.t_param == "polar":
            direction = self.raw_v / (
                torch.linalg.norm(self.raw_v, dim=1, keepdim=True) + 1e-8)
            return self.t_bound * torch.sigmoid(self.raw_a)[:, None] * direction
        if self.t_param == "tanh":
            return self.t_bound * torch.tanh(self.raw_t)
        return torch.clamp(self.t_values, -self.t_clamp, self.t_clamp)

    #: Feature value for a particle set with NO mass at all (every weight zero
    #: after sanitising). This is log(1e-8), the constant the pre-logsumexp
    #: forward produced for such rows through ``log(clamp(sum, min=1e-8))``;
    #: kept so the ``zero_mass`` golden in the Ant-Tag regression gate and the
    #: behaviour of every saved checkpoint on a dead filter are unchanged.
    DEAD_ROW_VALUE = math.log(1e-8)

    def _raw_cgf(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
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
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def raw_cgf_features(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """The CGF block BEFORE feature_norm and the readout, ``[B, num_encoded]``.

        What :class:`RolloutFeatureNormCallback` takes its statistics on:
        the quantity the norm standardises, not the norm's own output.
        """
        return self._raw_cgf(obs_dict)

    def _forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        cgf = self.readout(self.feature_norm(self._raw_cgf(obs_dict)))
        return torch.cat([obs_dict["obs"], cgf], dim=-1)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: F811
        if self._frozen:
            with torch.no_grad():
                return self._forward(obs_dict)
        return self._forward(obs_dict)
