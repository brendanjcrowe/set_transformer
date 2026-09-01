"""Statistical / analytic particle-set feature extractors for SB3 policies.

These are the non-Set-Transformer methods in the belief-encoder benchmark:

- :class:`GaussianExtractor` — mean + covariance (Gaussian approximation of the belief).
- :class:`KMomentsExtractor` — the first ``k`` (central) moments per dimension.
- :class:`CGFExtractor` — an empirical cumulant-generating-function sampled at learned
  points, the particle-set generalization of ``CGFFeatureExtractor`` in
  ``src/cgf_encoding_odd_even_beliefmdp.py``.

All three consume the Dict observation ``{"obs", "particles": (N, d)}`` produced by
:class:`~set_transformer.rl.wrappers.particle_filter.PFDictObservationWrapper`, exactly
like :class:`~set_transformer.rl.feature_extractors.e2e.CustomSetTransformerExtractor`:
they encode ``observations["obs"]`` with a small MLP, compute a permutation-invariant
statistic over ``observations["particles"]``, concatenate the two, and project to
``features_dim``. Particles are treated as an *unweighted* set (matching the ST extractor
convention; the Dict wrapper does not expose PF weights).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def load_pretrained_state_dict(model_path: str) -> dict:
    """Load an autoencoder state_dict from a raw or Trainer/sweep checkpoint file.

    Accepts the three shapes this repo writes: a bare ``state_dict``, a
    ``set_transformer.training`` checkpoint (``model_state_dict``), and a pretraining
    sweep checkpoint (``state_dict``).
    """
    loaded = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in loaded:
                return loaded[key]
        return loaded
    raise ValueError(
        f"Expected a state_dict or trainer checkpoint dict, got {type(loaded)}"
    )


class _BasePFStatExtractor(BaseFeaturesExtractor):
    """Shared plumbing for the statistical particle-set extractors.

    Subclasses implement :meth:`_particle_stat_dim` (the number of particle-derived
    features) and :meth:`_particle_features` (the permutation-invariant statistic). This
    mirrors the obs-MLP / concat / project structure of ``CustomSetTransformerExtractor``.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        obs_mlp_hidden_dims: list[int] = [64, 64],
        features_dim: int = 128,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(f"{type(self).__name__} expects a Dict observation space.")
        if "obs" not in observation_space.spaces:
            raise ValueError("Observation space Dict must contain an 'obs' key.")
        if "particles" not in observation_space.spaces:
            raise ValueError("Observation space Dict must contain a 'particles' key.")

        super().__init__(observation_space, features_dim)

        self.obs_dim = observation_space["obs"].shape[0]
        self.num_particles = observation_space["particles"].shape[0]
        self.particle_dim = observation_space["particles"].shape[1]

        # Let the subclass configure its particle-side parameters and report its stat width.
        self._build_particle_stat()
        stat_dim = self._particle_stat_dim()

        obs_mlp_layers: list[nn.Module] = []
        current_dim = self.obs_dim
        for hidden_dim in obs_mlp_hidden_dims:
            obs_mlp_layers.append(nn.Linear(current_dim, hidden_dim))
            obs_mlp_layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.obs_net = nn.Sequential(*obs_mlp_layers)

        self.combined_net = nn.Sequential(
            nn.Linear(current_dim + stat_dim, features_dim),
            nn.ReLU(),
        )

    # --- pretrained-encoder plumbing ------------------------------------------
    def _apply_pretrained(
        self, model: nn.Module, pretrained_model_path: str | None, freeze: bool
    ) -> nn.Module:
        """Load pretrained weights into ``model`` and optionally freeze it.

        Shared by the Set Transformer and the pooling encoders so checkpoint handling
        lives in one place. The *whole* autoencoder is kept attached even though only its
        encoder half feeds ``forward`` — that way a checkpoint round-trips unchanged.
        """
        if pretrained_model_path:
            model.load_state_dict(load_pretrained_state_dict(pretrained_model_path))
            print(f"{type(self).__name__}: loaded {pretrained_model_path}")
        else:
            print(f"{type(self).__name__}: training from scratch (no pretrained weights)")
        if freeze:
            model.requires_grad_(False)
            model.eval()
            self._frozen_model = model
        return model

    def train(self, mode: bool = True) -> "_BasePFStatExtractor":
        """Keep a frozen encoder in eval mode even when SB3 flips ``train()``."""
        super().train(mode)
        frozen = getattr(self, "_frozen_model", None)
        if frozen is not None:
            frozen.eval()
        return self

    def particle_encoder_parameters(self) -> int:
        """Learned parameters on the particle side that ``forward`` actually uses.

        The pretrained autoencoders keep their decoder attached so checkpoints round-trip
        unchanged, but it never runs in the policy — counting it would overstate encoder
        capacity in the fairness audit. Analytic baselines return 0.
        """
        if hasattr(self, "encoder"):
            return sum(p.numel() for p in self.encoder.parameters())
        head = {id(p) for p in self.obs_net.parameters()}
        head |= {id(p) for p in self.combined_net.parameters()}
        return sum(p.numel() for p in self.parameters() if id(p) not in head)

    @staticmethod
    def _check_freeze(pretrained_model_path: str | None, freeze: bool) -> None:
        if freeze and not pretrained_model_path:
            raise ValueError(
                "freeze=True requires pretrained_model_path (frozen random weights are "
                "meaningless)."
            )

    # --- subclass hooks -------------------------------------------------------
    def _build_particle_stat(self) -> None:
        """Create any particle-side parameters (default: none)."""

    def _particle_stat_dim(self) -> int:
        raise NotImplementedError

    def _particle_features(self, particles: torch.Tensor) -> torch.Tensor:
        """Map particles ``[B, N, d]`` to a statistic ``[B, stat_dim]``."""
        raise NotImplementedError

    # --- SB3 forward ----------------------------------------------------------
    #: PF weights for the current batch, or None when the observation carries no
    #: ``weights`` key. Stashed rather than threaded through ``_particle_features`` so
    #: subclasses that ignore weights keep their signature. Treating a missing key as
    #: uniform weights makes weight support opt-in per environment: an extractor works
    #: unchanged whether or not the wrapper exposes them.
    _particle_weights: torch.Tensor | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        self._particle_weights = observations.get("weights")
        obs_features = self.obs_net(observations["obs"])
        particle_features = self._particle_features(observations["particles"])
        combined = torch.cat([obs_features, particle_features], dim=1)
        return self.combined_net(combined)


class GaussianExtractor(_BasePFStatExtractor):
    """Gaussian approximation of the belief: empirical mean + covariance.

    Features are ``[mean (d), flattened lower-triangular covariance (d(d+1)/2)]``. No
    learned parameters on the particle side — this is the cheap analytic baseline.
    """

    def _build_particle_stat(self) -> None:
        d = self.particle_dim
        # Indices of the lower triangle (incl. diagonal) of the d x d covariance.
        self.register_buffer("_tril_rows", torch.tril_indices(d, d)[0], persistent=False)
        self.register_buffer("_tril_cols", torch.tril_indices(d, d)[1], persistent=False)

    def _particle_stat_dim(self) -> int:
        d = self.particle_dim
        return d + d * (d + 1) // 2

    def _particle_features(self, particles: torch.Tensor) -> torch.Tensor:
        mean = particles.mean(dim=1)  # [B, d]
        centered = particles - mean.unsqueeze(1)  # [B, N, d]
        # Biased covariance (divide by N); symmetric so the lower triangle is sufficient.
        cov = torch.einsum("bni,bnj->bij", centered, centered) / particles.shape[1]
        cov_flat = cov[:, self._tril_rows, self._tril_cols]  # [B, d(d+1)/2]
        return torch.cat([mean, cov_flat], dim=1)


class KMomentsExtractor(_BasePFStatExtractor):
    """First ``k`` moments per dimension: mean (order 1) + central moments of order 2..k.

    Feature width is ``k * d``. ``k=2`` recovers per-dimension mean+variance (Gaussian
    without cross-covariance). No learned parameters on the particle side.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        k: int = 4,
        obs_mlp_hidden_dims: list[int] = [64, 64],
        features_dim: int = 128,
    ):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        self.k = k
        super().__init__(observation_space, obs_mlp_hidden_dims, features_dim)

    def _particle_stat_dim(self) -> int:
        return self.k * self.particle_dim

    def _particle_features(self, particles: torch.Tensor) -> torch.Tensor:
        mean = particles.mean(dim=1)  # [B, d] — order-1 raw moment
        moments = [mean]
        if self.k >= 2:
            centered = particles - mean.unsqueeze(1)  # [B, N, d]
            for p in range(2, self.k + 1):
                moments.append((centered ** p).mean(dim=1))  # [B, d] central moment
        return torch.cat(moments, dim=1)


class CGFExtractor(_BasePFStatExtractor):
    """Empirical cumulant generating function sampled at ``num_t`` points.

    For sampling points ``t_m in R^d`` the weighted empirical CGF of a particle set is
    ``K(t_m) = log( sum_i w_i exp(t_m . x_i) )``. Where the CGF is evaluated determines
    what it measures: as ``t = s*u`` with unit ``u``, small ``s`` makes the Taylor jet at
    the origin the cumulants (i.e. a reparameterization of the k-moments baseline), while
    large ``s`` drives ``logsumexp`` toward ``max_i (u . x_i)`` — the support function of
    the set, which is the PointNet max-pool regime. ``t_value_norms()`` reports where on
    that continuum a trained encoder actually landed.

    Ported from the collaborator's ``WeightedCGFFeaturesExtractor``
    (``experiments/ant_tag/4_train_rl_cgf.py``), which is the canonical implementation:
    structured ``t`` inits, an elementwise clamp on ``t``, overflow-safe exponent
    clamping, optionally frozen ``t``, and PF weights. Two things differ, both to keep the
    benchmark's fairness contract:

    * **Matched bottleneck.** ``num_t`` CGF evaluations are projected to ``stat_dim``
      features, so the policy sees the same width from every method however finely the
      CGF curve is sampled. Sampling resolution and bottleneck width are then independent
      knobs, where in the original they were the same number.
    * **Shared head.** The projected features go through the same obs-MLP + concat +
      projection as every other method, rather than being concatenated raw to the obs.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_t: int = 64,
        stat_dim: int = 16,
        t_init_mode: str = "spread",
        t_init_scale: float = 0.1,
        t_clamp: float | None = 2.0,
        exp_arg_clamp: float = 20.0,
        t_frozen: bool = False,
        particle_scale: float = 1.0,
        obs_mlp_hidden_dims: list[int] = [64, 64],
        features_dim: int = 128,
    ):
        if num_t < 1:
            raise ValueError(f"num_t must be >= 1, got {num_t}.")
        if stat_dim < 1:
            raise ValueError(f"stat_dim must be >= 1, got {stat_dim}.")
        if particle_scale <= 0:
            raise ValueError(f"particle_scale must be > 0, got {particle_scale}.")
        self.num_t = num_t
        self.stat_dim = stat_dim
        self.t_init_mode = t_init_mode
        self.t_init_scale = t_init_scale
        self.t_clamp = t_clamp
        self.exp_arg_clamp = exp_arg_clamp
        self.t_frozen = bool(t_frozen)
        # exp(t.x) is scale-sensitive, so a t-range tuned for one arena is wrong for
        # another. Registry-driven per env (EnvSpec.particle_scale); 1.0 = no rescaling.
        self.particle_scale = particle_scale
        super().__init__(observation_space, obs_mlp_hidden_dims, features_dim)

    def _init_t_values(self) -> torch.Tensor:
        """Initial CGF sampling points ``[num_t, d]``.

        ``spread`` starts where the signal is (log-spaced norms over evenly spread
        directions) rather than requiring ``||t||`` to grow ~10x from a small init before
        the encoder can see anything; it is the collaborator's default and ours.
        """
        num_t, d, scale = self.num_t, self.particle_dim, self.t_init_scale
        mode = self.t_init_mode

        if mode == "spread":
            if d < 2:
                # The 1-D case has only two directions, so evenly spread them by sign.
                norms = torch.tensor(np.geomspace(0.25, 2.8, max(num_t // 2, 1)),
                                     dtype=torch.float32)
                signed = torch.cat([norms, -norms])[:num_t]
                return signed.reshape(num_t, 1)
            num_dirs = 8
            if num_t % num_dirs != 0:
                raise ValueError(
                    f"t_init_mode='spread' needs num_t divisible by {num_dirs}, got {num_t}")
            if d != 2:
                raise ValueError("t_init_mode='spread' assumes 1-D or 2-D particles")
            angles = torch.arange(num_dirs, dtype=torch.float32) * (2 * torch.pi / num_dirs)
            dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            # rho_hi=2.8 ~ the largest norm reachable under an elementwise clamp of 2.0
            # in 2-D (the diagonal, 2*sqrt(2)).
            norms = torch.tensor(np.geomspace(0.25, 2.8, num_t // num_dirs),
                                 dtype=torch.float32)
            return (norms[None, :, None] * dirs[:, None, :]).reshape(-1, d)

        if mode == "linspace_all_dims":
            # Round-robin the linspace directions across every dimension so each
            # coordinate gets a nontrivial initial CGF sensitivity.
            t_values = torch.zeros(num_t, d)
            vals = torch.linspace(-scale, scale, num_t)
            assignment = torch.arange(num_t) % d
            for dim in range(d):
                mask = assignment == dim
                t_values[mask, dim] = vals[mask]
            return t_values

        if mode == "linspace_first_dim":
            t_values = torch.zeros(num_t, d)
            t_values[:, 0] = torch.linspace(-scale, scale, num_t)
            if d > 1:
                t_values[:, 1:] = 0.01 * torch.randn(num_t, d - 1)
            return t_values

        if mode == "random":
            return scale * torch.randn(num_t, d)

        raise ValueError(f"Unknown t_init_mode: {mode}")

    def _build_particle_stat(self) -> None:
        t_values = self._init_t_values()
        if self.t_frozen:
            # A buffer takes no gradient (so the policy optimizer cannot move it) while
            # still saving/loading and moving across devices exactly like a Parameter.
            self.register_buffer("t_values", t_values)
        else:
            self.t_values = nn.Parameter(t_values)
        # Learned readout of the sampled CGF curve. This is what decouples sampling
        # resolution (num_t) from the bottleneck the policy sees (stat_dim).
        self.cgf_proj = nn.Identity() if self.num_t == self.stat_dim \
            else nn.Linear(self.num_t, self.stat_dim)

    def _particle_stat_dim(self) -> int:
        return self.stat_dim

    def _particle_features(self, particles: torch.Tensor) -> torch.Tensor:
        weights = self._particle_weights
        particles = particles / self.particle_scale
        particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)

        t = self.t_values
        if self.t_clamp is not None:
            t = torch.clamp(t, -self.t_clamp, self.t_clamp)
        exp_arg = torch.einsum("md,bnd->bmn", t, particles)
        exp_arg = torch.clamp(exp_arg, -self.exp_arg_clamp, self.exp_arg_clamp)

        if weights is None:
            # Unweighted set: log(mean_i exp(.)) via logsumexp (overflow-safe).
            n = particles.shape[1]
            cgf = torch.logsumexp(exp_arg, dim=2) - float(np.log(n))
        else:
            w = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
            # log-sum-exp with weights folded in as log w, keeping the stable form.
            cgf = torch.logsumexp(exp_arg + torch.log(w + 1e-20).unsqueeze(1), dim=2)
        return self.cgf_proj(cgf)

    @torch.no_grad()
    def t_value_norms(self) -> torch.Tensor:
        """Per-point L2 norms ``||t_m||`` of the CGF sampling points, post-clamp.

        Diagnostic for *where on the moment<->support-function continuum* the encoder
        operates (see the class docstring). Returns a ``[num_t]`` tensor.
        """
        t = self.t_values.detach()
        if self.t_clamp is not None:
            t = torch.clamp(t, -self.t_clamp, self.t_clamp)
        return torch.linalg.norm(t, dim=1)
