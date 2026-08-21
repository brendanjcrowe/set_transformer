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
    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
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
    """Empirical cumulant generating function sampled at ``num_t`` learned points.

    For learned vectors ``t_m in R^d`` the empirical CGF of the particle set is
    ``K(t_m) = log( (1/N) * sum_i exp(t_m . x_i) )`` (numerically via ``logsumexp``).
    This is the particle-set / multivariate generalization of the 1-D belief-vector
    ``CGFFeatureExtractor`` in ``src/cgf_encoding_odd_even_beliefmdp.py``. The sample
    points ``t`` are learned end-to-end.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_t: int = 16,
        t_init_scale: float = 0.5,
        obs_mlp_hidden_dims: list[int] = [64, 64],
        features_dim: int = 128,
    ):
        if num_t < 1:
            raise ValueError(f"num_t must be >= 1, got {num_t}.")
        self.num_t = num_t
        self.t_init_scale = t_init_scale
        super().__init__(observation_space, obs_mlp_hidden_dims, features_dim)

    def _build_particle_stat(self) -> None:
        # Learned sample points, one d-vector per output feature.
        self.t_values = nn.Parameter(
            torch.randn(self.num_t, self.particle_dim) * self.t_init_scale
        )

    def _particle_stat_dim(self) -> int:
        return self.num_t

    def _particle_features(self, particles: torch.Tensor) -> torch.Tensor:
        n = particles.shape[1]
        # dot[b, m, i] = t_m . x_i
        dot = torch.einsum("md,bnd->bmn", self.t_values, particles)  # [B, num_t, N]
        # CGF = log( mean_i exp(dot) ) = logsumexp_i(dot) - log(N)  (overflow-safe)
        cgf = torch.logsumexp(dot, dim=2) - torch.log(
            torch.tensor(float(n), device=particles.device)
        )
        return cgf  # [B, num_t]

    @torch.no_grad()
    def t_value_norms(self) -> torch.Tensor:
        """Per-point L2 norms ``||t_m||`` of the learned CGF sampling points.

        Diagnostic for *where on the moment<->support-function continuum* the encoder
        operates. As ``t = s*u`` with unit ``u``: for small ``s`` the CGF's Taylor jet at
        the origin is the cumulants (moment-equivalent), so norms collapsing toward 0 mean
        the encoder has degenerated into a reparameterization of the k-moments baseline;
        for large ``s`` the ``logsumexp`` tends to ``max_i (u . x_i)`` — the support
        function of the particle set (the PointNet max-pool regime) — capturing non-local
        structure that finite moments cannot. Returns a ``[num_t]`` tensor.
        """
        return torch.linalg.norm(self.t_values.detach(), dim=1)
