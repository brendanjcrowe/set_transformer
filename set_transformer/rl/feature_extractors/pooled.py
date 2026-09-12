"""Pooling and moment belief encoders for the experiments/ arms (2026-09-11).

Three SB3 features extractors over the ``{"obs", "particles", "weights"}`` Dict
observation of :class:`~set_transformer.rl.wrappers.particle_filter.PFDictWithWeightsObservationWrapper`,
in the arm convention shared by ``WeightedCGFFeaturesExtractor``,
``SetTransformerFeaturesExtractor`` and ``WeightedGaussianFeaturesExtractor``: particles
are divided by ``arena_scale``, PF weights are in the measure, and the encoder output is
concatenated raw to the base observation for the policy's own MLP to read.

- :class:`WeightedDeepSetFeaturesExtractor` -- ``DeepSet`` (per-particle MLP, pool, MLP;
  Zaheer et al. 2017) with the pool a **weighted mean** ``sum_i w_i phi(x_i)`` (``pooling
  ="weighted"``) or the plain mean (``"mean"``, what an unweighted pretraining produced).
- :class:`PointNetFeaturesExtractor` -- the same network with a **max** pool (Qi et al.
  2017). Max has no weighted form; ``pooling="masked_max"`` (default) excludes particles
  of exactly zero weight -- refuted particles must not define the support -- and
  ``"max"`` is the plain operator. Either way the mass channel lets the network see
  each particle's weight.
- :class:`WeightedKMomentsFeaturesExtractor` -- weighted mean and weighted central
  moments of orders 2..k per coordinate. Parameter-free; ``k=2`` is the diagonal of the
  Gaussian arm.

The two learned encoders reuse the benchmark's model classes
(:class:`~set_transformer.models.deep_set.DeepSet`, :class:`~set_transformer.models.point_net.PointNet`),
calling their ``enc`` / ``dec`` halves directly so the pool can be weighted; a
``DeepSetAE`` / ``PointNetAE`` checkpoint (the benchmark's ``3_pretrain_encoder.py`` or a
``Trainer`` run with ``model_type ds_ae``) therefore loads into ``.encoder`` unchanged
(``--pretrained_model_path``), and ``--frozen`` / ``--encoder_lr_scale`` work as on the ST
arm. Both AEs are unweighted (``dim_input = D``), so load them with
``weight_channel=False``; the extractor refuses the mismatch with the flag to fix it.
"""

from __future__ import annotations

import math

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from set_transformer.models.deep_set import DeepSet
from set_transformer.models.point_net import PointNet


def _clean_weights(weights: torch.Tensor) -> torch.Tensor:
    """NaN/inf -> 0, negatives -> 0, exact-sum normalisation with the arms' ``+ 1e-8``."""
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, min=0.0)
    return weights / (weights.sum(dim=1, keepdim=True) + 1e-8)


def _extract_encoder_state(loaded, path: str) -> tuple[dict, object, object]:
    """``(encoder_state, config, particle_scale)`` from any of the checkpoint shapes.

    A ``Trainer`` checkpoint (``model_state_dict`` + ``config`` + top-level
    ``particle_scale``), a ``DeepSetAE`` / ``PointNetAE`` ``state_dict`` (``encoder.*`` +
    ``decoder.*``), or a bare ``DeepSet`` / ``PointNet`` ``state_dict`` (``enc.*`` + ``dec.*``).
    """
    config, scale = None, None
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state, config, scale = loaded["model_state_dict"], loaded.get("config"), loaded.get("particle_scale")
    elif isinstance(loaded, dict):
        state = loaded
    else:
        raise ValueError(f"Expected a state_dict or Trainer checkpoint at {path}, got {type(loaded)}")
    prefix = "encoder."
    encoder_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not encoder_state:
        encoder_state = {k: v for k, v in state.items() if not k.startswith("decoder.")}
    if not any(k.startswith("enc.") for k in encoder_state):
        raise ValueError(f"{path} holds no DeepSet / PointNet encoder parameters "
                         f"(keys: {sorted(state)[:6]} ...)")
    return encoder_state, config, scale


class _PooledFeaturesExtractor(BaseFeaturesExtractor):
    """Shared plumbing: frame, weights, weight channel, pretrained load, freeze."""

    ENCODER_CLS: type
    ENCODER_NAME: str
    POOLINGS: tuple[str, ...]
    #: Constructor argument carrying the pretraining checkpoint path; blanked in
    #: policy_kwargs by the shared reload (PITFALLS.md section 7).
    PRETRAINED_PATH_KWARG = "pretrained_model_path"

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_encodings: int = 8,
        dim_encoder: int = 8,
        dim_hidden: int = 128,
        arena_scale: float = 4.5,
        weight_channel: bool = True,
        pooling: str | None = None,
        pretrained_model_path: str | None = None,
        frozen: bool = False,
    ):
        pooling = pooling or self.POOLINGS[0]
        if pooling not in self.POOLINGS:
            raise ValueError(f"{type(self).__name__}: pooling must be one of "
                             f"{self.POOLINGS}, got {pooling!r}")
        if frozen and not pretrained_model_path:
            raise ValueError(f"{type(self).__name__}: frozen=True without a pretrained "
                             "checkpoint would freeze a random encoder")
        obs_dim = observation_space["obs"].shape[0]
        particle_dim = observation_space["particles"].shape[1]
        self.num_particles = observation_space["particles"].shape[0]
        output_dim = num_encodings * dim_encoder
        super().__init__(observation_space, features_dim=obs_dim + output_dim)
        self.arena_scale = float(arena_scale)
        self.weight_channel = bool(weight_channel)
        self.pooling = pooling
        self.num_encodings, self.dim_encoder, self.dim_hidden = num_encodings, dim_encoder, dim_hidden
        self.dim_input = particle_dim + (1 if self.weight_channel else 0)
        self.encoder = self.ENCODER_CLS(dim_input=self.dim_input, num_outputs=num_encodings,
                                        dim_output=dim_encoder, dim_hidden=dim_hidden)
        self._geometry = dict(
            encoder=self.ENCODER_NAME, num_encodings=int(num_encodings),
            dim_encoder=int(dim_encoder), dim_hidden=int(dim_hidden),
            weight_channel=self.weight_channel, pooling=pooling,
            arena_scale=self.arena_scale, dim_input=int(self.dim_input),
        )
        self._frozen = False
        if pretrained_model_path:
            self._load_pretrained_encoder(pretrained_model_path)
        else:
            print(f"{type(self).__name__}: no pretrained checkpoint given; encoder starts "
                  "from random init (when loading a saved policy its trained weights "
                  "overwrite this init).")
        if frozen:
            self.freeze_encoder()
        print(f"{type(self).__name__}: dim_input={self.dim_input} (weight_channel="
              f"{self.weight_channel}), pooling={pooling}, output_dim={output_dim}, "
              f"features_dim={self.features_dim}")

    # -- frame and weights ------------------------------------------------------------
    def _prepare(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        particles = torch.nan_to_num(obs_dict["particles"] / self.arena_scale,
                                     nan=0.0, posinf=1.0, neginf=-1.0)
        weights = _clean_weights(obs_dict["weights"])                            # [B, N]
        if self.weight_channel:
            # Mass as one extra input channel scaled by N, so a uniform belief feeds
            # 1.0 -- the same convention as the ST arm and Trainer._model_input.
            particles = torch.cat([particles, (weights * weights.shape[1]).unsqueeze(-1)], dim=-1)
        return particles, weights

    def _pool(self, phi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[B, num_encodings, dim_encoder]`` -- the encoder's code for this belief."""
        x, w = self._prepare(obs_dict)
        phi = self.encoder.enc(x)                                                # [B, N, H]
        code = self.encoder.dec(self._pool(phi, w))                              # [B, K*d]
        return code.reshape(code.shape[0], self.num_encodings, self.dim_encoder)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        code = self.encode(obs_dict)
        return torch.cat([obs_dict["obs"], code.reshape(code.shape[0], -1)], dim=-1)

    # -- pretrained / frozen ------------------------------------------------------------
    def encoder_parameter_count(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters())

    def freeze_encoder(self) -> None:
        """Fix every encoder parameter; idempotent. No running statistics to hold."""
        self._frozen = True
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "_frozen", False):
            self.encoder.eval()
        return self

    def _check_checkpoint_geometry(self, config, particle_scale, path: str) -> None:
        get = (lambda k: config.get(k)) if isinstance(config, dict) else (lambda k: getattr(config, k, None))
        if config is not None:
            for field in ("num_encodings", "dim_encoder", "dim_hidden"):
                recorded = get(field)
                if recorded is not None and int(recorded) != self._geometry[field]:
                    raise RuntimeError(
                        f"Checkpoint {path} records {field}={recorded}, this run has "
                        f"{self._geometry[field]}; pass matching --{field}.")
            recorded_wc = get("weighted_particles")
            if recorded_wc is not None and bool(recorded_wc) != self.weight_channel:
                raise RuntimeError(
                    f"Checkpoint {path} was pretrained with weighted_particles="
                    f"{bool(recorded_wc)} (encoder input {'D+1' if recorded_wc else 'D'}), "
                    f"this run has weight_channel={self.weight_channel}. Flip "
                    "--no_weight_channel / --weight_channel to match.")
        if particle_scale is not None and not math.isclose(
                float(particle_scale), self.arena_scale, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"Checkpoint {path} was pretrained on particles scaled by particle_scale="
                f"{float(particle_scale)!r}, this run divides by arena_scale="
                f"{self.arena_scale!r}: the encoder would read another frame "
                "(PITFALLS.md section 4).")

    def pretrained_reference_state(self, path: str) -> dict:
        """The encoder tensors a checkpoint holds, keyed like ``self.encoder.state_dict()``."""
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        return _extract_encoder_state(loaded, path)[0]

    def _load_pretrained_encoder(self, path: str) -> None:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        encoder_state, config, scale = _extract_encoder_state(loaded, path)
        self._check_checkpoint_geometry(config, scale, path)
        try:
            self.encoder.load_state_dict(encoder_state, strict=True)
        except RuntimeError as exc:
            hint = ""
            first = encoder_state.get("enc.0.weight")
            if first is not None and first.shape[1] != self.dim_input:
                hint = (f" The checkpoint's encoder reads {first.shape[1]} input channels, "
                        f"this run feeds {self.dim_input}: "
                        + ("pass --no_weight_channel (the checkpoint is unweighted)."
                           if first.shape[1] < self.dim_input else "pass --weight_channel."))
            raise RuntimeError(f"Could not load the {self.ENCODER_NAME} encoder from {path}; "
                               f"its geometry must match this run's ({self._geometry}).{hint}"
                               f"\nOriginal error: {exc}") from exc
        print(f"{type(self).__name__}: loaded encoder from {path}")

    # -- The shared encoder interface (change 2, 2026-09-12); the contract is spelled out in st.py.

    def encoder_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.encoder.parameters())

    def load_pretrained(self, path: str) -> None:
        self._load_pretrained_encoder(path)

    def freeze(self) -> None:
        self.freeze_encoder()

    def encoder_state_dict(self) -> dict:
        return self.encoder.state_dict()

    def reference_state(self, path: str) -> dict:
        return self.pretrained_reference_state(path)


class WeightedDeepSetFeaturesExtractor(_PooledFeaturesExtractor):
    """DeepSet with a weighted-mean pool (``pooling="weighted"``) or plain mean (``"mean"``)."""

    ENCODER_CLS = DeepSet
    ENCODER_NAME = "deepset"
    POOLINGS = ("weighted", "mean")

    def _pool(self, phi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.pooling == "weighted":
            return torch.sum(weights.unsqueeze(-1) * phi, dim=1)                # sum_i w_i phi(x_i)
        return phi.mean(dim=1)


class PointNetFeaturesExtractor(_PooledFeaturesExtractor):
    """PointNet with a max pool: ``"masked_max"`` (zero-weight particles excluded) or ``"max"``."""

    ENCODER_CLS = PointNet
    ENCODER_NAME = "pointnet"
    POOLINGS = ("masked_max", "max")

    def _pool(self, phi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.pooling == "masked_max":
            dead_particle = (weights <= 0.0).unsqueeze(-1)                       # [B, N, 1]
            all_dead = dead_particle.all(dim=1, keepdim=True)                    # a massless row: keep everything
            masked = phi.masked_fill(dead_particle & ~all_dead, float("-inf"))
            return masked.max(dim=1)[0]
        return phi.max(dim=1)[0]


class WeightedKMomentsFeaturesExtractor(BaseFeaturesExtractor):
    """``[mean_d, m2_d, ..., mk_d]`` per coordinate: weighted mean and weighted central
    moments of orders 2..k, on particles divided by ``arena_scale``. ``k * D`` features,
    no parameters. ``k=2`` is the Gaussian arm without the cross-covariance."""

    def __init__(self, observation_space: gym.spaces.Dict, k: int = 4, arena_scale: float = 4.5):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        obs_dim = observation_space["obs"].shape[0]
        self.particle_dim = observation_space["particles"].shape[1]
        super().__init__(observation_space, features_dim=obs_dim + k * self.particle_dim)
        self.k = int(k)
        self.arena_scale = float(arena_scale)
        self._geometry = dict(encoder="kmoments", k=self.k, arena_scale=self.arena_scale)

    def moments(self, particles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """``[B, k*D]`` for raw-frame particles and PF weights."""
        x = torch.nan_to_num(particles / self.arena_scale, nan=0.0, posinf=1.0, neginf=-1.0)
        w = _clean_weights(weights).unsqueeze(-1)                                # [B, N, 1]
        mean = torch.sum(w * x, dim=1)                                           # [B, D]
        out = [mean]
        if self.k >= 2:
            centered = x - mean.unsqueeze(1)
            for p in range(2, self.k + 1):
                out.append(torch.sum(w * centered ** p, dim=1))
        return torch.cat(out, dim=-1)

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([obs_dict["obs"], self.moments(obs_dict["particles"], obs_dict["weights"])], dim=-1)

    def encoder_parameter_count(self) -> int:
        return 0


def reload_pretrained_pooled(model, path: str, frozen: bool, verify: bool = True) -> None:
    """Post-PPO-construction reload for the pooling arms (PITFALLS.md section 1). Since
    change 2 (2026-09-12) a forwarder to the shared
    :func:`set_transformer.rl.pretrained_encoder.reload_pretrained`, which does the same four
    steps for every learned extractor; kept under this name for the pool script and
    ``feature_extractors.__init__``."""
    from set_transformer.rl.pretrained_encoder import reload_pretrained

    reload_pretrained(model, path, frozen, verify=verify)
