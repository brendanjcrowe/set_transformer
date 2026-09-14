"""Reconstruction pretraining for the arms' CGF encoder.

:class:`CGFArmAutoencoder` wraps the CGF block of
:class:`~set_transformer.rl.feature_extractors.cgf.WeightedCGFFeaturesExtractor` -- the
probes under their ``t_param`` bound, the optional particle embedding, the feature
standardiser and the readout, i.e. everything ``4_train_rl_cgf.py`` trains and
``--pretrained_cgf_model_path`` loads -- and puts a :class:`~set_transformer.modules.PFDecoder`
behind it, so the same :class:`~set_transformer.training.trainer.Trainer` that pretrains the
Set Transformer (weighted Sinkhorn, latent metric alignment, checkpointing) pretrains the
CGF. This is the arms' counterpart of the benchmark's
:class:`~set_transformer.models.CGFAutoencoder`: that one is the benchmark's contract
(uniform weights, clamp only, K only, a 64 -> 16 projection, ``{t_values, cgf_proj.*}``);
this one is the arm's (PF weights in the measure, polar / tanh / clamp, K or K', norm and
readout, the extractor's own ``state_dict``).

Frame. ``Trainer`` feeds particles the loader has already mapped to
``(x - particle_centre) / particle_scale``, with the mass appended as one channel scaled
by N (``Trainer._model_input``). The extractor divides its input by ``arena_scale``
itself, because at RL time it receives raw coordinates. So here the normalised particles
are multiplied back by ``arena_scale`` before the block, and ``arena_scale`` must equal
the dataset's ``particle_scale`` -- checked at construction -- so the block sees the frame
it will see under PPO and the learned ``t`` mean the same thing in both. The decoder
reconstructs in the loader's normalised frame, which is what the loss compares against.

Export. ``Trainer`` checkpoints hold the whole autoencoder under ``extractor.`` /
``decoder.`` prefixes. :func:`export_arm_checkpoint` writes the file the RL side loads:
``model_state_dict`` = the extractor's unprefixed ``state_dict`` and ``config`` = its
``_cgf_geometry`` (which the loader checks field by field, ``arena_scale`` included) plus
provenance -- the same shape ``experiments/odd_even/3_pretrain_st_belief.py`` writes.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from set_transformer.models.arm_export import plain as _plain  # noqa: F401 - historical name
from set_transformer.modules import PFDecoder
from set_transformer.rl.feature_extractors.cgf import WeightedCGFFeaturesExtractor


def make_arm_extractor(num_particles: int, dim_particles: int, arena_scale: float,
                       **cgf_kwargs) -> WeightedCGFFeaturesExtractor:
    """The arm extractor over a minimal Dict space (a 1-dim dummy ``obs``).

    Pretraining never touches ``obs``; the space is only what the SB3 base class
    needs to size itself. Every CGF flag is passed through by name, spelled as in
    ``4_train_rl_cgf.py``.
    """
    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
        "particles": gym.spaces.Box(-np.inf, np.inf, (num_particles, dim_particles), np.float32),
        "weights": gym.spaces.Box(0.0, 1.0, (num_particles,), np.float32),
    })
    return WeightedCGFFeaturesExtractor(space, arena_scale=arena_scale, **cgf_kwargs)


class CGFArmAutoencoder(nn.Module):
    """``WeightedCGFFeaturesExtractor.cgf_block`` -> ``PFDecoder``.

    Args:
        extractor: the arm extractor whose block is trained.
        num_particles / dim_particles: the set the decoder emits.
        particle_scale: the dataset's normalisation; must equal ``extractor.arena_scale``.
        num_encodings: slots the decoder cross-attends over. The block's width
            (``extractor.readout_dim``: 64 for K, 64*D for K', or the readout's) is
            reshaped to ``[num_encodings, width / num_encodings]``.
        dim_hidden: the decoder MLP width.
        weighted: whether ``Trainer`` appends the mass channel to the input.
    """

    def __init__(self, extractor: WeightedCGFFeaturesExtractor, num_particles: int,
                 dim_particles: int, particle_scale: float, num_encodings: int = 8,
                 dim_hidden: int = 128, weighted: bool = True) -> None:
        super().__init__()
        if not np.isclose(float(extractor.arena_scale), float(particle_scale)):
            raise ValueError(
                f"extractor.arena_scale={extractor.arena_scale} but the dataset is "
                f"normalised by particle_scale={particle_scale}. The block divides its "
                "input by arena_scale and the RL side hands it raw coordinates, so the "
                "two must be the same number or t is learned in a frame the policy "
                "never uses (PITFALLS.md section 4).")
        if int(extractor.raw_particle_dim) != int(dim_particles):
            raise ValueError(f"extractor was built for {extractor.raw_particle_dim}-D "
                             f"particles, dataset has {dim_particles}-D")
        width = int(extractor.readout_dim)
        if width % num_encodings != 0:
            raise ValueError(
                f"the CGF block is {width} wide, not divisible by num_encodings="
                f"{num_encodings}; pick a divisor (8 works for 64 / 128 / 192) or a "
                "readout_dim that is a multiple of it.")
        self.extractor = extractor
        self.num_particles = int(num_particles)
        self.dim_particles = int(dim_particles)
        self.particle_scale = float(particle_scale)
        self.num_encodings = int(num_encodings)
        self.dim_encoder = width // num_encodings
        self.weighted = bool(weighted)
        self.decoder = PFDecoder(self.dim_encoder, dim_hidden, self.num_particles,
                                 self.dim_particles)

    # -- input convention shared with Trainer._model_input --------------------------
    def split_input(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, N, D(+1)]`` from the Trainer -> raw-frame particles and PF weights."""
        d = self.dim_particles
        if self.weighted:
            if X.shape[-1] != d + 1:
                raise ValueError(f"weighted input must be [B, N, {d + 1}] (coords + mass "
                                 f"channel), got {tuple(X.shape)}")
            particles, mass = X[..., :d], X[..., d]
            # Trainer scales the mass channel by N so a uniform belief feeds 1.0;
            # the block renormalises anyway, this just returns to probabilities.
            weights = mass / mass.shape[-1]
        else:
            if X.shape[-1] != d:
                raise ValueError(f"unweighted input must be [B, N, {d}], got {tuple(X.shape)}")
            particles = X
            weights = X.new_full(X.shape[:2], 1.0 / X.shape[1])
        # The loader normalised by particle_scale; the block divides by arena_scale
        # (== particle_scale, checked in __init__). Undo, so the block sees the frame
        # it sees under PPO.
        return particles * self.extractor.arena_scale, weights

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        """``[B, num_encodings, dim_encoder]`` code -- the block output, reshaped.
        The alignment loss and the decoder both read this."""
        particles, weights = self.split_input(X)
        code = self.extractor.cgf_block(particles, weights)              # [B, width]
        return code.reshape(code.shape[0], self.num_encodings, self.dim_encoder)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Reconstructed particles ``[B, N, D]`` in the loader's normalised frame."""
        return self.decoder(self.encode(X))

    def encoder_state_dict(self) -> dict:
        """What the RL side loads: the extractor's own keys, unprefixed."""
        return {k: v.detach().cpu() for k, v in self.extractor.state_dict().items()}


def export_arm_checkpoint(trainer_checkpoint: Path, out_path: Path,
                          extractor: WeightedCGFFeaturesExtractor,
                          particle_centre: float, objective: str,
                          data_path: str, extra_config: dict | None = None) -> Path:
    """Rewrite a ``Trainer`` checkpoint of a :class:`CGFArmAutoencoder` into the file
    ``4_train_rl_cgf.py --pretrained_cgf_model_path`` loads.

    ``model_state_dict`` is the extractor's state under its own key names (the
    ``extractor.`` prefix stripped, ``decoder.*`` dropped); ``config`` is the
    extractor's ``_cgf_geometry`` -- every field
    ``WeightedCGFFeaturesExtractor._check_checkpoint_geometry`` compares, ``arena_scale``
    included -- plus provenance. ``particle_scale`` / ``particle_centre`` are also
    written top-level, as ``Trainer`` checkpoints carry them for the ST arm.

    Since batch 10.11 (2026-09-14) the body is the generic
    :func:`set_transformer.models.arm_export.export_arm_checkpoint`, which does the same steps
    through the extractor's ``checkpoint_state()`` / ``checkpoint_config()`` (for the CGF arm:
    the whole extractor ``state_dict`` and ``_cgf_geometry``, as before) and serves the pooled
    arms too. The file's content is unchanged.
    """
    from set_transformer.models.arm_export import export_arm_checkpoint as _export

    return _export(trainer_checkpoint, out_path, extractor, encoder_name="cgf",
                   particle_centre=particle_centre, objective=objective, data_path=data_path,
                   extra_config=extra_config)
