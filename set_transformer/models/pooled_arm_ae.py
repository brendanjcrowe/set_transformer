"""Reconstruction pretraining for the arms' DeepSet and PointNet encoders.

:class:`PooledArmAutoencoder` wraps a pooled RL extractor
(:class:`~set_transformer.rl.feature_extractors.pooled.WeightedDeepSetFeaturesExtractor` or
:class:`~set_transformer.rl.feature_extractors.pooled.PointNetFeaturesExtractor` -- the
per-particle MLP, the WEIGHTED or MASKED pool over the PF weights, the output MLP, i.e. everything
``rl/train.py --encoder deepset|pointnet`` trains and ``--pretrained_model_path`` loads) and puts a
:class:`~set_transformer.modules.PFDecoder` behind it, so the same
:class:`~set_transformer.training.trainer.Trainer` that pretrains the Set Transformer and the CGF
arm (weighted Sinkhorn, latent metric alignment, checkpointing) pretrains the pooled arms. It is
the pooled arms' counterpart of :class:`~set_transformer.models.cgf_arm_ae.CGFArmAutoencoder`
(batch 10.11 of the harness refactor, 2026-09-14) and of the benchmark's ``DeepSetAE`` /
``PointNetAE``: those pool UNWEIGHTED (every particle counts alike, no mass channel), so a
checkpoint of theirs is not the arm's computation; this one runs the extractor's own ``encode``.

Frame. ``Trainer`` feeds particles the loader has already mapped to
``(x - particle_centre) / particle_scale``, with the mass appended as one channel scaled by N
(``Trainer._model_input``). The extractor divides its input by ``arena_scale`` itself and appends
the same mass channel itself (``_PooledFeaturesExtractor._prepare``), because at RL time it
receives raw coordinates and probabilities. So here the normalised particles are multiplied back
by ``arena_scale`` and the mass channel is turned back into probabilities before the extractor,
and ``arena_scale`` must equal the dataset's ``particle_scale`` -- checked at construction -- so
the encoder sees the frame it will see under PPO. The decoder reconstructs in the loader's
normalised frame, which is what the loss compares against.

Export. ``Trainer`` checkpoints hold the whole autoencoder under ``extractor.`` / ``decoder.``
prefixes; :func:`~set_transformer.models.arm_export.export_arm_checkpoint` writes the file the
RL side loads (``encoder.*`` tensors + the geometry record, through the extractor's own
``checkpoint_state()`` / ``checkpoint_config()``).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from set_transformer.modules import PFDecoder
from set_transformer.rl.feature_extractors.pooled import _PooledFeaturesExtractor


class PooledArmAutoencoder(nn.Module):
    """``_PooledFeaturesExtractor.encode`` -> ``PFDecoder``.

    Args:
        extractor: the arm extractor whose encoder is trained (DeepSet or PointNet).
        num_particles / dim_particles: the set the decoder emits.
        particle_scale: the dataset's normalisation; must equal ``extractor.arena_scale``.
        dim_hidden: the decoder MLP width.
        weighted: whether ``Trainer`` appends the mass channel to the input; must equal the
            extractor's ``weight_channel`` (the encoder's input width follows it).

    The code the alignment loss and the decoder read is the extractor's own
    ``[B, num_encodings, dim_encoder]`` output; no reshape is needed (the CGF arm reshapes its
    block width into ``num_encodings`` slots).
    """

    def __init__(self, extractor: _PooledFeaturesExtractor, num_particles: int,
                 dim_particles: int, particle_scale: float, dim_hidden: int = 128,
                 weighted: bool = True, decoder_temperature: bool = False) -> None:
        super().__init__()
        if not np.isclose(float(extractor.arena_scale), float(particle_scale)):
            raise ValueError(
                f"extractor.arena_scale={extractor.arena_scale} but the dataset is "
                f"normalised by particle_scale={particle_scale}. The extractor divides its "
                "input by arena_scale and the RL side hands it raw coordinates, so the "
                "two must be the same number or the encoder is trained in a frame the policy "
                "never uses (PITFALLS.md section 4).")
        raw_dim = int(extractor.dim_input) - (1 if extractor.weight_channel else 0)
        if raw_dim != int(dim_particles):
            raise ValueError(f"extractor was built for {raw_dim}-D particles, dataset has "
                             f"{dim_particles}-D")
        if bool(extractor.weight_channel) != bool(weighted):
            raise ValueError(
                f"extractor.weight_channel={extractor.weight_channel} but the Trainer input is "
                f"{'weighted (mass channel appended)' if weighted else 'unweighted'}: the encoder "
                "reads D+1 channels with the weight channel and D without. Build the extractor "
                "with weight_channel=<weighted> (the reconstruction door does; --no_weight_channel "
                "counts as --ignore_weights).")
        self.extractor = extractor
        self.num_particles = int(num_particles)
        self.dim_particles = int(dim_particles)
        self.particle_scale = float(particle_scale)
        self.num_encodings = int(extractor.num_encodings)
        self.dim_encoder = int(extractor.dim_encoder)
        self.weighted = bool(weighted)
        self.decoder = PFDecoder(self.dim_encoder, dim_hidden, self.num_particles,
                                 self.dim_particles, learn_temperature=decoder_temperature)

    # -- input convention shared with Trainer._model_input (the CGF arm's split_input) ---------
    def split_input(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, N, D(+1)]`` from the Trainer -> raw-frame particles and PF weights."""
        d = self.dim_particles
        if self.weighted:
            if X.shape[-1] != d + 1:
                raise ValueError(f"weighted input must be [B, N, {d + 1}] (coords + mass "
                                 f"channel), got {tuple(X.shape)}")
            particles, mass = X[..., :d], X[..., d]
            # Trainer scales the mass channel by N so a uniform belief feeds 1.0; the
            # extractor renormalises and re-appends it anyway, this returns to probabilities.
            weights = mass / mass.shape[-1]
        else:
            if X.shape[-1] != d:
                raise ValueError(f"unweighted input must be [B, N, {d}], got {tuple(X.shape)}")
            particles = X
            weights = X.new_full(X.shape[:2], 1.0 / X.shape[1])
        # The loader normalised by particle_scale; the extractor divides by arena_scale
        # (== particle_scale, checked in __init__). Undo, so it sees the frame it sees under PPO.
        return particles * self.extractor.arena_scale, weights

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """``[B, num_encodings, dim_encoder]`` code -- the extractor's own ``encode`` on the
        Dict observation it reads under PPO (``obs`` is never used by the encoder)."""
        particles, weights = self.split_input(X)
        obs = particles.new_zeros((particles.shape[0], 1))
        return self.extractor.encode({"obs": obs, "particles": particles, "weights": weights})

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Reconstructed particles ``[B, N, D]`` in the loader's normalised frame."""
        return self.decoder(self.encode(X))

    def encoder_state_dict(self) -> dict:
        """What the RL side loads: the extractor's ``checkpoint_state()`` (``encoder.*``)."""
        return self.extractor.checkpoint_state()


def make_pooled_arm_extractor(extractor_class: type, num_particles: int, dim_particles: int,
                              arena_scale: float, **kwargs) -> _PooledFeaturesExtractor:
    """The arm extractor over a minimal Dict space (a 1-dim dummy ``obs``), the pooled twin of
    :func:`~set_transformer.models.cgf_arm_ae.make_arm_extractor`. Pretraining never touches
    ``obs``; the space is only what the SB3 base class needs to size itself. Every flag is
    passed through by name, spelled as in ``rl/train.py --encoder deepset|pointnet``."""
    import gymnasium as gym

    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (num_particles, dim_particles), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (num_particles,), np.float32),
    })
    return extractor_class(space, arena_scale=arena_scale, **kwargs)
