"""Set Transformer belief encoder for SB3, plus its feature-logging callback.

A :class:`~set_transformer.models.PFSetTransformer`'s encoder (ISAB stack +
PMA/SAB head) consumes the particle set and emits
``num_encodings x dim_encoder`` features, which are concatenated with the base
observation exactly like the CGF features are.

WEIGHTS. The CGF and Gaussian encoders are both *weighted* — they read
``obs_dict["weights"]``. The legacy ST extractors ignore the PF weights, which
throws away real evidence whenever a filter's likelihood step leaves the
weights non-uniform between resamples. So by default the normalized weight is
appended as an extra input channel per particle (scaled by num_particles, so a
uniform belief feeds 1.0). Pass ``weight_channel=False`` for the legacy
unweighted behavior; the arm is then no longer information-matched to CGF.

Pretrained weights are OPTIONAL and off by default, because the CGF baseline
learns its t_values from scratch under PPO and the matched ST run is likewise
trained end-to-end by PPO. ``pretrained_st_model_path`` / ``st_frozen`` exist
for the pretrained-encoder variant, mirroring the CGF arm's ``t_frozen``.

**Domain-independent.** It reads the ``{"obs", "particles", "weights"}`` Dict
observation produced by
:class:`set_transformer.rl.wrappers.particle_filter.PFDictWithWeightsObservationWrapper`
and takes every geometry off that space, so it works for any particle
dimension and set size.

Moved out of ``experiments/ant_tag/4_train_rl_st.py`` so a second domain can
use it without importing an Ant-Tag script. That script still re-exports both
names: SB3 pickles a features-extractor CLASS into the saved zip by module
path, so a checkpoint loads via
``getattr(import_module("4_train_rl_st"), "SetTransformerFeaturesExtractor")``
and the re-export is what keeps saved runs loadable.
"""

import gymnasium as gym
import numpy as np
import math

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from set_transformer.models import PFSetTransformer


class STFeatureLoggingCallback(BaseCallback):
    """Log the distribution of the ST feature vector every rollout.

    The ST analogue of TNormLoggingCallback. An encoder trained inside PPO
    can collapse to a constant — permutation-invariant pooling makes that an
    easy local optimum — and a collapsed encoder still trains happily while
    carrying no belief information at all. st/feat_std_mean is the sentinel:
    it is the per-feature standard deviation across the last forward batch,
    averaged over features. If it decays toward 0 the encoder has collapsed
    and the run is a no-op regardless of what the reward curve does.
    Quantiles of the per-sample feature norm land alongside it.
    No-op (and free) for encoders without cached features.
    """

    def _on_step(self) -> bool:  # required abstract method
        return True

    def _on_rollout_end(self) -> None:
        extractor = getattr(self.model.policy, "features_extractor", None)
        features = getattr(extractor, "last_st_features", None)
        if features is None:
            return
        with torch.no_grad():
            feats = features.detach().float()
            norms = torch.linalg.norm(feats, dim=1).cpu().numpy()
            std_mean = float(feats.std(dim=0).mean().cpu())
        self.logger.record("st/feat_std_mean", std_mean)
        for q in (0, 25, 50, 75, 90, 100):
            self.logger.record(f"st/feat_norm_q{q}",
                               float(np.percentile(norms, q)))


class SetTransformerFeaturesExtractor(BaseFeaturesExtractor):
    """SB3 feature extractor: SetTransformer over the weighted particle set.

    Consumes the same Dict observation as WeightedCGFFeaturesExtractor
    ({"obs", "particles", "weights"}), so every eval / audit script written
    for the CGF arm works unchanged on an ST checkpoint.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        num_encodings: int = 8,
        dim_encoder: int = 8,
        num_inds: int = 32,
        dim_hidden: int = 128,
        num_heads: int = 4,
        ln: bool = True,
        arena_scale: float = 4.5,
        weight_channel: bool = True,
        pretrained_st_model_path: str | None = None,
        st_frozen: bool = False,
        num_post_sab: int = 2,
    ):
        obs_dim = observation_space["obs"].shape[0]
        particle_dim = observation_space["particles"].shape[1]
        num_particles = observation_space["particles"].shape[0]
        st_output_dim = num_encodings * dim_encoder
        super().__init__(observation_space, features_dim=obs_dim + st_output_dim)

        self.arena_scale = arena_scale
        self.weight_channel = bool(weight_channel)
        self.st_frozen = bool(st_frozen)
        self.num_particles = num_particles
        # Cached by forward() for STFeatureLoggingCallback. Not a buffer:
        # it must not enter the checkpoint or move with .to().
        self.last_st_features: torch.Tensor | None = None

        # Kept as an attribute so a post-construction reload (see each
        # arm's train_* function, e.g. 4_train_rl_st.py's train_ant_tag_st)
        # can report the expected geometry.
        dim_input = particle_dim + (1 if self.weight_channel else 0)
        self.dim_input = dim_input

        # Only the encoder is used at RL time, and only the encoder is
        # loaded from a checkpoint. The PFDecoder's shape is deliberately not
        # matched: 3_train_st.py trains weighted sets with a D+1 input and a
        # D-dimensional reconstruction, so its decoder does not have the shape
        # a symmetric PFSetTransformer would build here. Loading encoder keys
        # only keeps this independent of the decoder convention, and keeps the
        # unused decoder out of the PPO optimizer and the saved policy.
        pf_st = PFSetTransformer(
            num_particles=num_particles,
            dim_particles=dim_input,
            num_encodings=num_encodings,
            dim_encoder=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
            num_post_sab=num_post_sab,
        )
        self.encoder = pf_st.set_transformer

        # Compared against the TrainingConfig a 3_train_st.py checkpoint
        # carries. A strict load_state_dict catches every mismatch that
        # changes a parameter shape, but num_heads changes none (the MAB's
        # projections are dim_hidden x dim_hidden however the heads split
        # them), so a 4-head checkpoint loads silently into an 8-head encoder
        # that computes something else. Field names follow TrainingConfig.
        self._st_geometry = dict(
            num_encodings=num_encodings,
            dim_encoder=dim_encoder,
            num_inds=num_inds,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            use_layer_norm=bool(ln),
            weighted_particles=self.weight_channel,
            # Absent from checkpoints written before 2026-09-05; the geometry
            # check skips fields the checkpoint does not carry, and those
            # checkpoints were all built with the default 2.
            num_post_sab=int(num_post_sab),
            # The coordinate frame the encoder was trained in. A checkpoint
            # pretrained at one particle scale loads into an encoder fed
            # another without any shape changing (PITFALLS.md section 4 and
            # section 8 item 2); the CGF extractor already refuses this.
            # 3_pretrain_st_belief.py writes it into config; Trainer
            # checkpoints carry it top-level as particle_scale (checked in
            # _load_pretrained_encoder).
            arena_scale=float(arena_scale),
        )

        if pretrained_st_model_path:
            self._load_pretrained_encoder(pretrained_st_model_path, dim_input)
        else:
            # Also printed when SB3 reconstructs a SAVED policy, whose
            # encoder weights arrive from the checkpoint a moment later, so
            # the wording must not claim the encoder stays random.
            print("SetTransformerFeaturesExtractor: no pretrained checkpoint "
                  "given; encoder starts from random init (this is the arm "
                  "that matches CGF, which also learns its encoder under "
                  "PPO). When loading a saved policy, its trained weights "
                  "overwrite this init.")

        if self.st_frozen:
            # Same intent as the CGF arm's --t_frozen: the encoder is fixed
            # and only the policy/value MLP learns. LayerNorm carries no
            # running statistics, so freezing the parameters is sufficient.
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            print("SetTransformerFeaturesExtractor: encoder FROZEN")

        print(f"SetTransformerFeaturesExtractor: dim_input={dim_input} "
              f"(weight_channel={self.weight_channel}), "
              f"st_output_dim={st_output_dim}, features_dim={self.features_dim}")

    def _check_checkpoint_geometry(self, config, path: str) -> None:
        """Refuse a checkpoint whose recorded geometry contradicts this run's.

        ``config`` is the TrainingConfig (or a dict of it) that
        ``Trainer.save_checkpoint`` stores. Fields it does not carry are
        skipped, so older checkpoints still load and the strict state_dict
        load remains the backstop for shape-changing mismatches.
        """
        if config is None:
            return
        geometry = getattr(self, "_st_geometry", None)
        if not geometry:
            return

        def _get(field):
            if isinstance(config, dict):
                return config.get(field)
            return getattr(config, field, None)

        mismatches = []
        for field, expected in geometry.items():
            actual = _get(field)
            if actual is None:
                continue
            if isinstance(expected, bool):
                actual = bool(actual)
            if isinstance(expected, float):
                same = math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-9)
            else:
                same = actual == expected
            if not same:
                mismatches.append(
                    f"{field}: checkpoint={actual!r}, this run={expected!r}")
        if mismatches:
            raise RuntimeError(
                f"Checkpoint {path} was pretrained with a different encoder "
                "geometry than this run requests:\n  "
                + "\n  ".join(mismatches)
                + "\nPass the matching --num_encodings/--dim_encoder/--num_inds/"
                "--dim_hidden/--num_heads/--ln/--arena_scale flags (or "
                "--no_st_weight_channel for an unweighted checkpoint). A "
                "num_heads or arena_scale mismatch changes no parameter shape "
                "and would otherwise load silently."
            )

    def _load_pretrained_encoder(self, path: str, dim_input: int) -> None:
        """Load the SetTransformer encoder out of a 3_train_st.py checkpoint.

        Accepts a Trainer checkpoint dict, a full PFSetTransformer state_dict,
        or a bare encoder state_dict. Decoder entries are dropped.
        """
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            state_dict = loaded["model_state_dict"]
            self._check_checkpoint_geometry(loaded.get("config"), path)
            # Trainer (3_train_st.py) checkpoints record the dataset frame
            # top-level rather than in config.
            recorded_scale = loaded.get("particle_scale")
            if recorded_scale is not None and not math.isclose(
                    float(recorded_scale), float(self.arena_scale),
                    rel_tol=1e-6, abs_tol=1e-9):
                raise RuntimeError(
                    f"Checkpoint {path} was pretrained on particles scaled by "
                    f"particle_scale={float(recorded_scale)!r}, but this run "
                    f"divides by arena_scale={float(self.arena_scale)!r}. The "
                    "encoder would read inputs in a different frame than it "
                    "was trained on (PITFALLS.md section 4). Pass the matching "
                    "--arena_scale, or pretrain on a dataset in this frame.")
        elif isinstance(loaded, dict):
            state_dict = loaded
        else:
            raise ValueError(
                f"Expected state_dict or trainer checkpoint, got {type(loaded)}"
            )

        prefix = "set_transformer."
        encoder_state = {
            key[len(prefix):]: value for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not encoder_state:
            # Already an encoder-only state_dict.
            encoder_state = {
                key: value for key, value in state_dict.items()
                if not key.startswith("decoder.")
            }
        if not encoder_state:
            raise ValueError(
                f"{path} contains no SetTransformer encoder parameters")

        try:
            self.encoder.load_state_dict(encoder_state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load the encoder from {path}. The checkpoint's "
                "geometry must match this run's: encoder input dim "
                f"{dim_input} (coordinates + 1 when --st_weight_channel is on, "
                "which is the default), and the same --num_encodings, "
                "--dim_encoder, --num_inds, --dim_hidden, --num_heads, --ln. "
                "Pretrain with 3_train_st.py using matching flags, or flip "
                "--no_st_weight_channel if the checkpoint is unweighted.\n"
                f"Original error: {exc}"
            ) from exc
        print(f"SetTransformerFeaturesExtractor: loaded encoder from {path}")

    def forward(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        base_obs = obs_dict["obs"]
        particles = obs_dict["particles"] / self.arena_scale
        particles = torch.nan_to_num(particles, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.weight_channel:
            # Identical sanitizing/renormalization to
            # WeightedCGFFeaturesExtractor.forward, so both arms read the
            # same weight vector. Scaled by N: a uniform belief feeds 1.0
            # instead of 0.01, which keeps the channel on the same scale as
            # the normalized coordinates.
            weights = obs_dict["weights"]
            weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = torch.clamp(weights, min=0.0)
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
            weights = weights * particles.shape[1]
            particle_input = torch.cat([particles, weights.unsqueeze(-1)], dim=-1)
        else:
            particle_input = particles

        if self.st_frozen:
            with torch.no_grad():
                encoded = self.encoder(particle_input)
        else:
            encoded = self.encoder(particle_input)

        st_features = encoded.reshape(encoded.size(0), -1)
        self.last_st_features = st_features.detach()
        return torch.cat([base_obs, st_features], dim=-1)
