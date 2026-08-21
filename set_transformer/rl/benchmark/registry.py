"""Environment and method registries for the unified benchmark trainer.

Adding an env or method is a single ``EnvSpec`` / ``MethodSpec`` entry — the trainer
(``experiments/benchmark/train.py``) is fully driven by these tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import gymnasium as gym

from set_transformer.rl.benchmark import envs
from set_transformer.rl.feature_extractors import (
    CGFExtractor,
    DeepSetExtractor,
    GaussianExtractor,
    KMomentsExtractor,
    PointNetExtractor,
    SetTransformerExtractor,
)
from set_transformer.rl.mappers import (
    ant_tag_pf_interaction_mapper,
    get_ant_tag_pf_kwargs,
)
from set_transformer.rl.particle_filters.ant_tag import AntTagParticleFilter
from set_transformer.rl.particle_filters.car_flag import CarFlagParticleFilter
from set_transformer.rl.particle_filters.odd_even import OddEvenParticleFilter


@dataclass
class EnvSpec:
    """Everything the trainer needs to build one benchmark environment."""

    name: str
    make_base_env: Callable[..., gym.Env]
    particle_filter_class: type
    num_particles: int
    particle_filter_kwargs: dict = field(default_factory=dict)
    # Optional env -> dict hook read off the *live* base env at construction time (arena
    # size, motion-model step, visibility radius, ...). Its result is the base layer;
    # ``particle_filter_kwargs`` above overrides it, so the registry can still pin a value.
    pf_kwargs_from_env: Optional[Callable[[gym.Env], dict]] = None
    pf_mapper: Optional[Callable] = None
    obs_mask_indices: Optional[list[int]] = None
    # Phi(env)->float applied via PotentialBasedShapingWrapper; None => no shaping.
    potential_fn: Optional[Callable[[gym.Env], float]] = None
    gamma: float = 0.99
    default_algo: str = "PPO"
    default_timesteps: int = 2_000_000
    max_ep_steps: Optional[int] = None
    # (true_episode_return, episode_length) -> bool; None if success is not binary.
    success_fn: Optional[Callable[[float, int], bool]] = None
    # Human-readable description of success_fn, recorded in each run's meta.json.
    success_criterion: Optional[str] = None
    base_env_kwargs: dict = field(default_factory=dict)


@dataclass
class MethodSpec:
    """A feature-extractor method.

    ``is_pretrainable`` marks the learned set encoders (ST / DeepSet / PointNet), which
    additionally take the runtime encoder arch and — for the frozen/finetune flavors — a
    pretrained checkpoint. ``encoder_kind`` and ``aligned`` say *which* checkpoint the
    sweep runner should hand them: an encoder pretrained with, or without, the latent
    metric-alignment term.
    """

    name: str
    extractor_class: type
    extractor_kwargs: dict = field(default_factory=dict)
    is_pretrainable: bool = False
    encoder_kind: Optional[str] = None  # "st" | "ds" | "pn"
    aligned: bool = False


# --- Environment registry ------------------------------------------------------

ENV_REGISTRY: dict[str, EnvSpec] = {
    "ant_tag": EnvSpec(
        name="ant_tag",
        make_base_env=envs.make_ant_tag_base_env,
        particle_filter_class=AntTagParticleFilter,
        num_particles=100,
        particle_filter_kwargs={},
        pf_kwargs_from_env=get_ant_tag_pf_kwargs,
        pf_mapper=ant_tag_pf_interaction_mapper,
        obs_mask_indices=[-2, -1],  # hide the true target from the agent (PF still reads it)
        potential_fn=envs.ant_tag_true_state_potential,
        gamma=0.99,
        default_algo="PPO",
        default_timesteps=2_000_000,
        max_ep_steps=400,
        success_fn=lambda ep_ret, ep_len: ep_len < 400,  # tagged => terminated early
        success_criterion="episode_length < max_ep_steps (target tagged before timeout)",
    ),
    "car_flag": EnvSpec(
        name="car_flag",
        make_base_env=envs.make_car_flag_base_env,
        particle_filter_class=CarFlagParticleFilter,
        num_particles=100,
        particle_filter_kwargs={},
        pf_mapper=None,  # direction is part of the observation; no privileged info to bridge
        obs_mask_indices=None,  # nothing to hide: the agent legitimately observes direction
        potential_fn=None,  # hidden-side env: run sparse first (envs.car_flag_belief_potential is the leak-free fallback)
        gamma=0.99,
        default_algo="PPO",
        default_timesteps=1_000_000,
        max_ep_steps=160,
        success_fn=envs.car_flag_success,
        success_criterion="reached heaven (reconstructed terminal reward >= +0.5; corrected reward)",
    ),
    "odd_even": EnvSpec(
        name="odd_even",
        make_base_env=envs.make_odd_even_base_env,
        particle_filter_class=OddEvenParticleFilter,
        num_particles=100,
        particle_filter_kwargs={"n_dist_size": 10},
        pf_mapper=None,  # PF.update(obs) default path; no privileged info to bridge
        obs_mask_indices=None,
        potential_fn=None,  # dense native reward (-squared_error): no shaping
        gamma=0.99,
        default_algo="PPO",
        default_timesteps=500_000,
        max_ep_steps=100,
        success_fn=None,
    ),
}


# --- Method registry -----------------------------------------------------------

# Capacity-matched configuration (2026-08-20). Every *learned* encoder shares
# stat_dim = num_encodings * dim_encoder = 16 and ~100-111k encoder parameters:
# ST at dim_hidden=64/num_inds=32 is 111,106; DeepSet/PointNet at dim_hidden=128,
# dim_encoder=2 are 101,520 (d=2). Gaussian and k-moments stay analytic with zero learned
# parameters — being parameter-free sufficient statistics is their purpose, not a
# confound to correct, so their stat_dim is whatever the statistic itself implies.
# Only the parity knob is pinned here. The bottleneck (num_encodings / dim_encoder) comes
# from the shared CLI arch so every learned method moves together when it is changed;
# dim_hidden is what differs per encoder family to bring their parameter counts in line.
POOLING_ARCH = {"dim_hidden": 128}

#: stat_dim every learned encoder is matched to.
MATCHED_STAT_DIM = 16
#: Learned encoders must land within this fraction of each other's parameter count.
PARAM_PARITY_TOLERANCE = 0.15


def _pretrained_methods() -> dict[str, MethodSpec]:
    """The 12 pretrained cells: {ST, DeepSet, PointNet} x {unaligned, aligned} x
    {frozen, finetune}. ``st_frozen`` / ``st_finetune`` keep their historical names."""
    encoders = {
        "st": (SetTransformerExtractor, {}),
        "ds": (DeepSetExtractor, dict(POOLING_ARCH)),
        "pn": (PointNetExtractor, dict(POOLING_ARCH)),
    }
    out: dict[str, MethodSpec] = {}
    for kind, (cls, arch) in encoders.items():
        for aligned, arm in ((False, ""), (True, "align_")):
            for freeze, flavor in ((True, "frozen"), (False, "finetune")):
                name = f"{kind}_{arm}{flavor}"
                out[name] = MethodSpec(
                    name, cls, {**arch, "freeze": freeze},
                    is_pretrainable=True, encoder_kind=kind, aligned=aligned,
                )
    return out


METHOD_REGISTRY: dict[str, MethodSpec] = {
    # --- analytic baselines (no learned particle-side parameters) ---
    "gaussian": MethodSpec("gaussian", GaussianExtractor),
    "kmoments": MethodSpec("kmoments", KMomentsExtractor, {"k": 4}),
    "cgf": MethodSpec("cgf", CGFExtractor, {"num_t": 16}),
    # --- learned encoders trained from scratch with the policy ---
    "deepset": MethodSpec("deepset", DeepSetExtractor, dict(POOLING_ARCH),
                          is_pretrainable=True, encoder_kind="ds"),
    "pointnet": MethodSpec("pointnet", PointNetExtractor, dict(POOLING_ARCH),
                           is_pretrainable=True, encoder_kind="pn"),
    "st_scratch": MethodSpec("st_scratch", SetTransformerExtractor,
                             is_pretrainable=True, encoder_kind="st"),
    # --- pretrained, unaligned and alignment-trained, frozen and fine-tuned ---
    **_pretrained_methods(),
}

#: Display/plot order: analytic, scratch, then pretrained grouped by encoder.
METHOD_ORDER: list[str] = [
    "gaussian", "kmoments", "cgf",
    "deepset", "pointnet", "st_scratch",
    "ds_frozen", "ds_finetune", "ds_align_frozen", "ds_align_finetune",
    "pn_frozen", "pn_finetune", "pn_align_frozen", "pn_align_finetune",
    "st_frozen", "st_finetune", "st_align_frozen", "st_align_finetune",
]


# --- Lookups / kwargs assembly -------------------------------------------------

def get_env_spec(name: str) -> EnvSpec:
    if name not in ENV_REGISTRY:
        raise KeyError(f"Unknown env '{name}'. Registered: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[name]


def get_method_spec(name: str) -> MethodSpec:
    if name not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Registered: {sorted(METHOD_REGISTRY)}")
    return METHOD_REGISTRY[name]


def build_extractor_kwargs(
    method_spec: MethodSpec,
    features_dim: int,
    obs_mlp_hidden_dims: list[int],
    pretrained_model_path: Optional[str] = None,
    encoder_arch: Optional[dict] = None,
) -> dict:
    """Merge shared + method-specific + (learned-encoder) runtime kwargs.

    ``encoder_arch`` is the ST-shaped arch dict; the pooling extractors accept and ignore
    the keys they have no use for (``num_inds`` / ``num_heads`` / ``ln``), matching how
    ``DeepSetAE`` already handles them. Method-specific kwargs are re-applied last so a
    registry pin (e.g. the pooling ``dim_hidden``) wins over the CLI default.
    """
    kwargs = dict(features_dim=features_dim, obs_mlp_hidden_dims=list(obs_mlp_hidden_dims))
    kwargs.update(method_spec.extractor_kwargs)
    if method_spec.is_pretrainable:
        if encoder_arch:
            kwargs.update(encoder_arch)
            kwargs.update(method_spec.extractor_kwargs)
        if pretrained_model_path:
            kwargs["pretrained_model_path"] = pretrained_model_path
        elif method_spec.extractor_kwargs.get("freeze"):
            raise ValueError(
                f"Method '{method_spec.name}' freezes a pretrained encoder but no "
                "--pretrained_model_path was given."
            )
    return kwargs
