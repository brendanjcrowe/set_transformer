"""Environment and method registries for the unified benchmark trainer.

Adding an env or method is a single ``EnvSpec`` / ``MethodSpec`` entry — the trainer
(``experiments/benchmark/train.py``) is fully driven by these tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import gymnasium as gym

from set_transformer.rl.benchmark import envs
from set_transformer.rl.wrappers import shaping
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
from set_transformer.rl.particle_filters.multimodal_search import (
    MultimodalSearchParticleFilter,
)
from set_transformer.rl.particle_filters.odd_even_parity import (
    ParityAwareOddEvenParticleFilter,
)


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
    # base_env_obs -> origin the particle set is expressed relative to (normally the
    # agent's own position). None leaves particles in world coordinates.
    particle_origin_fn: Optional[Callable] = None
    # Phi(env)->float applied via PotentialBasedShapingWrapper; None => no shaping.
    potential_fn: Optional[Callable[[gym.Env], float]] = None
    gamma: float = 0.99
    # Discount used for the shaping term; defaults to `gamma`. Set to 1.0 together with
    # `shaping_zero_at_termination=False` to turn the potential into a proxy reward whose
    # episode total is exactly the change in potential.
    shaping_gamma: Optional[float] = None
    shaping_zero_at_termination: bool = True
    default_algo: str = "PPO"
    default_timesteps: int = 2_000_000
    max_ep_steps: Optional[int] = None
    # Divisor applied to particles before scale-sensitive statistics (the CGF's
    # exp(t.x)). Set it to the env's characteristic belief scale -- e.g. the arena
    # half-width -- so one t-range is meaningful across envs. 1.0 = no rescaling.
    particle_scale: float = 1.0
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
        particle_scale=4.5,  # arena half-width; matches the collaborator's arena_scale
        success_fn=lambda ep_ret, ep_len: ep_len < 400,  # tagged => terminated early
        success_criterion="episode_length < max_ep_steps (target tagged before timeout)",
    ),
    # Same POMDP as `ant_tag`, with the tag made explicitly worth reaching. The stock
    # reward (-1/step, 0 on tag) produced zero tags across a 2-seed x 8-method x 2M-step
    # pilot -- every run sat at exactly -400.0 -- so `ant_tag` cannot discriminate
    # encoders as shipped. See envs.AntTagRewardWrapper.
    "ant_tag_bonus": EnvSpec(
        name="ant_tag_bonus",
        make_base_env=envs.make_ant_tag_base_env,
        base_env_kwargs={"tag_bonus_reward": True},
        particle_filter_class=AntTagParticleFilter,
        num_particles=100,
        particle_filter_kwargs={},
        pf_kwargs_from_env=get_ant_tag_pf_kwargs,
        pf_mapper=ant_tag_pf_interaction_mapper,
        obs_mask_indices=[-2, -1],
        potential_fn=envs.ant_tag_true_state_potential,
        gamma=0.99,
        default_algo="PPO",
        default_timesteps=2_000_000,
        max_ep_steps=400,
        particle_scale=4.5,
        success_fn=envs.ant_tag_success,
        success_criterion="tagged the target before the horizon (return > 0)",
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
    # Purpose-built to be discriminating: the belief is K random Gaussian modes with the
    # centroid pinned at the origin, so the belief MEAN is a constant carrying zero
    # information while the mode geometry stays random (nothing fixed to sweep instead).
    # Measured: an informed mode tour takes ~77 steps against ~307 for a full lawnmower,
    # and the 250-step horizon sits between them.
    "msearch": EnvSpec(
        name="msearch",
        make_base_env=envs.make_msearch_base_env,
        particle_filter_class=MultimodalSearchParticleFilter,
        num_particles=100,
        particle_filter_kwargs={"k_max": 10, "arena_half": 14.0,
                                "visibility_radius": 1.5},
        pf_mapper=None,       # the PF reads the observation directly; nothing privileged
        # The prior's mode parameters ride in the observation so the filter can build the
        # belief, and are masked from the agent: it is meant to learn the mode structure
        # through its belief encoder, not to be handed it as a vector.
        obs_mask_indices=list(range(7, 67)),
        # Hand the encoder AGENT-RELATIVE particles. The optimal policy is "head for the
        # nearest unvisited mode", which is a relation between the agent and the set; in
        # world coordinates every method has to learn to subtract its own position, which
        # reaches it through a different pathway entirely.
        particle_origin_fn=lambda obs: obs[0:2],
        # Information-gain shaping. The task reward is sparse (one terminal event), and a
        # velocity-controlled point mass explored with iid noise barely moves -- net
        # displacement over a whole episode is ~5 units in a 28-wide arena -- so an
        # unshaped learner sees almost no signal. This potential pays for ruling out prior
        # mass, which is dense, leak-free (it reads only the belief), and still
        # discriminating: collecting it needs to know WHERE the mass is, which a
        # mean-and-covariance summary cannot say. Eval envs are built unshaped, so
        # reported returns remain the true task reward.
        potential_fn=shaping.pf_belief_information_potential(scale=100.0),
        # Information gain as a PROXY REWARD, not policy-invariant shaping: gamma=1 and no
        # terminal zeroing, so an episode's bonus is exactly the information gathered.
        # Scale 100 -- the setting under which st_scratch actually learned. It puts the
        # proxy's ceiling (~2.2 nats * 100 = 220) above the 43 find bonus, which would
        # normally be the trap where surveying pays better than finishing. It does not
        # bite HERE: a mode's information cannot be collected without sweeping it, and
        # sweeping the mode that holds the target finds the target, so the two behaviours
        # are inseparable. An agent can bank at most log(K) nats by clearing the empty
        # modes first, and once those are gone the only remaining source of information is
        # the mode that ends the episode. Evaluation stays on the true sparse reward.
        shaping_gamma=1.0,
        shaping_zero_at_termination=False,
        gamma=0.99,
        # SAC, not PPO. Measured at an identical 600k budget: SAC lifts every method far
        # above PPO here (gaussian 0.12 -> 0.54, st_scratch 0.14 -> 0.89, st_frozen
        # 0.48 -> 0.64) and it INVERTS the ordering -- a from-scratch Set Transformer
        # becomes the best method, essentially matching the informed reference, while the
        # frozen encoder's PPO-era advantage disappears. Much of what looked like an
        # encoder gap under PPO was a PPO optimization artifact; the env itself was fine
        # once the particles were put in the agent's frame.
        default_algo="SAC",
        default_timesteps=600_000,
        max_ep_steps=42,
        particle_scale=14.0,  # arena half-width
        success_fn=envs.msearch_success,
        success_criterion="found the target before the horizon (return > 0)",
    ),
    "odd_even": EnvSpec(
        name="odd_even",
        make_base_env=envs.make_odd_even_base_env,
        # Parity-aware: reproduces the env's own observation model, so the belief is the
        # exact posterior -- a comb over one parity -- rather than a blob smeared across
        # states the observations have already ruled out. std_dev must match the env's.
        particle_filter_class=ParityAwareOddEvenParticleFilter,
        num_particles=100,
        particle_filter_kwargs={"n_dist_size": 10, "std_dev": 2.0},
        pf_mapper=None,  # PF.update(obs) default path; no privileged info to bridge
        obs_mask_indices=None,
        potential_fn=None,  # dense native reward (-squared_error): no shaping
        gamma=0.99,
        default_algo="PPO",
        default_timesteps=500_000,
        max_ep_steps=100,
        particle_scale=10.0,  # latent range [1, n_dist_size]
        success_fn=envs.odd_even_success,
        success_criterion=("mean per-step reward > -2.0, i.e. under ~2.5% of steps spent "
                           "on a wrong-parity (impossible) state"),
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
    "cgf": MethodSpec("cgf", CGFExtractor),
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

def _init_params(cls) -> set:
    import inspect
    return set(inspect.signature(cls.__init__).parameters)


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
    particle_scale: Optional[float] = None,
) -> dict:
    """Merge shared + method-specific + (learned-encoder) runtime kwargs.

    ``encoder_arch`` is the ST-shaped arch dict; the pooling extractors accept and ignore
    the keys they have no use for (``num_inds`` / ``num_heads`` / ``ln``), matching how
    ``DeepSetAE`` already handles them. Method-specific kwargs are re-applied last so a
    registry pin (e.g. the pooling ``dim_hidden``) wins over the CLI default.
    """
    kwargs = dict(features_dim=features_dim, obs_mlp_hidden_dims=list(obs_mlp_hidden_dims))
    kwargs.update(method_spec.extractor_kwargs)
    # Only the CGF is scale-sensitive today; pass it only where accepted so adding the
    # EnvSpec field cannot break the other extractors' signatures.
    if particle_scale is not None and "particle_scale" in _init_params(method_spec.extractor_class):
        kwargs.setdefault("particle_scale", particle_scale)
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
