"""Tests for the benchmark registry and env/wrapper wiring.

Only constructs the Odd-Even env (no MuJoCo); Ant-Tag is checked via its spec only.
Skipped if the ``[rl]`` extra is missing.
"""

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from set_transformer.rl.benchmark.registry import (
    ENV_REGISTRY,
    METHOD_REGISTRY,
    build_extractor_kwargs,
    get_env_spec,
    get_method_spec,
)
from set_transformer.rl.wrappers.shaping import PotentialBasedShapingWrapper


def test_registries_populated():
    assert {"ant_tag", "car_flag", "odd_even"} <= set(ENV_REGISTRY)
    assert {"gaussian", "kmoments", "cgf", "st_frozen", "st_finetune", "st_scratch"} <= set(METHOD_REGISTRY)


def test_car_flag_spec_is_unshaped_and_maskless():
    # Hidden-side env: runs sparse first (no potential), and direction is observed
    # legitimately so nothing is masked and no mapper is needed.
    spec = get_env_spec("car_flag")
    assert spec.potential_fn is None
    assert spec.obs_mask_indices is None
    assert spec.pf_mapper is None
    assert spec.success_fn is not None and spec.success_criterion


def test_unknown_lookups_raise():
    with pytest.raises(KeyError, match="Unknown env"):
        get_env_spec("nope")
    with pytest.raises(KeyError, match="Unknown method"):
        get_method_spec("nope")


def test_st_frozen_requires_path():
    with pytest.raises(ValueError, match="pretrained"):
        build_extractor_kwargs(get_method_spec("st_frozen"), 128, [64, 64])


def test_encoder_arch_and_path_injected_for_learned_encoders_only():
    encoder_arch = dict(num_encodings=4, dim_encoder=2, num_inds=8, dim_hidden=32,
                        num_heads=2, ln=True)
    for method in ("st_finetune", "ds_finetune", "pn_finetune"):
        kw = build_extractor_kwargs(
            get_method_spec(method), 128, [64, 64],
            pretrained_model_path="/tmp/x.pt", encoder_arch=encoder_arch,
        )
        assert kw["pretrained_model_path"] == "/tmp/x.pt"
        assert kw["num_encodings"] == 4
    # Analytic baselines ignore the encoder arch and the checkpoint path entirely.
    kw2 = build_extractor_kwargs(
        get_method_spec("gaussian"), 128, [64, 64],
        pretrained_model_path="/tmp/x.pt", encoder_arch=encoder_arch,
    )
    assert "pretrained_model_path" not in kw2
    assert "num_encodings" not in kw2


def test_registry_pins_win_over_cli_encoder_arch():
    # The pooling encoders pin dim_hidden=128 to hit parameter parity with the ST at
    # dim_hidden=64; a CLI-supplied arch must not silently override that. The bottleneck
    # is deliberately NOT pinned, so --dim_encoder moves every learned method together.
    kw = build_extractor_kwargs(
        get_method_spec("deepset"), 128, [64, 64],
        encoder_arch=dict(num_encodings=8, dim_encoder=2, num_inds=32, dim_hidden=64,
                          num_heads=4, ln=True),
    )
    assert kw["dim_hidden"] == 128
    assert kw["dim_encoder"] == 2  # bottleneck still follows the shared arch


def test_shaping_policy_per_env():
    # Ant-Tag shapes (true-state potential); Odd-Even does not (dense native reward).
    assert get_env_spec("ant_tag").potential_fn is not None
    assert get_env_spec("odd_even").potential_fn is None


def test_odd_even_env_stack_produces_dict_obs():
    from set_transformer.rl.wrappers.particle_filter import PFDictObservationWrapper

    spec = get_env_spec("odd_even")
    env = spec.make_base_env(seed=0)
    env = PFDictObservationWrapper(
        env=env,
        particle_filter_class=spec.particle_filter_class,
        particle_filter_kwargs=spec.particle_filter_kwargs,
        num_particles=spec.num_particles,
        pf_interaction_mapper=spec.pf_mapper,
        obs_mask_indices=spec.obs_mask_indices,
    )
    obs, _ = env.reset()
    assert set(obs.keys()) == {"obs", "particles"}
    assert obs["particles"].shape == (spec.num_particles, 1)
    # Odd-Even is unshaped: no PotentialBasedShapingWrapper needed.
    assert spec.potential_fn is None
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    assert "particles" in obs


def test_shaping_wrapper_reports_true_reward_when_applied():
    """When a potential is present, shaped step exposes the true reward in info."""
    spec = get_env_spec("odd_even")
    base = spec.make_base_env(seed=1)
    # Attach a trivial potential to exercise the wrapper on a non-MuJoCo env.
    wrapped = PotentialBasedShapingWrapper(base, potential_fn=lambda e: 0.0, gamma=spec.gamma)
    wrapped.reset()
    _, shaped, _, _, info = wrapped.step(wrapped.action_space.sample())
    assert info["true_reward"] == pytest.approx(shaped)  # zero potential => no change


# --- capacity fairness (2026-08-20) -------------------------------------------------

def _build_extractor(method: str, particle_dim: int = 2):
    """Instantiate a method's extractor without a checkpoint (drops `freeze`, which
    legitimately refuses to build on random weights)."""
    import gymnasium as gym
    from set_transformer.rl.benchmark.registry import METHOD_REGISTRY

    spec = METHOD_REGISTRY[method]
    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-1.0, 1.0, (8,)),
        "particles": gym.spaces.Box(-9.0, 9.0, (100, particle_dim)),
    })
    kwargs = dict(features_dim=128, obs_mlp_hidden_dims=[64, 64])
    kwargs.update(spec.extractor_kwargs)
    if spec.is_pretrainable:
        kwargs.update(num_encodings=8, dim_encoder=2, num_inds=32, dim_hidden=64,
                      num_heads=4, ln=True)
        kwargs.update(spec.extractor_kwargs)
        kwargs.pop("freeze", None)
    return spec.extractor_class(space, **kwargs)


def test_every_learned_encoder_shares_the_matched_bottleneck():
    from set_transformer.rl.benchmark.registry import (
        MATCHED_STAT_DIM, METHOD_ORDER, METHOD_REGISTRY,
    )
    learned = [m for m in METHOD_ORDER if METHOD_REGISTRY[m].is_pretrainable]
    for method in learned:
        stat = _build_extractor(method)._particle_stat_dim()
        assert stat == MATCHED_STAT_DIM, f"{method} has stat_dim {stat}"


def test_learned_encoder_parameter_counts_are_within_tolerance():
    # The paper's fairness claim: no learned encoder has a large capacity advantage. The
    # analytic baselines are excluded by design -- being parameter-free is their point.
    from set_transformer.rl.benchmark.registry import (
        METHOD_ORDER, METHOD_REGISTRY, PARAM_PARITY_EXEMPT_KINDS,
        PARAM_PARITY_TOLERANCE,
    )
    counts = {
        m: _build_extractor(m).particle_encoder_parameters()
        for m in METHOD_ORDER
        if METHOD_REGISTRY[m].is_pretrainable
        and METHOD_REGISTRY[m].encoder_kind not in PARAM_PARITY_EXEMPT_KINDS
    }
    lo, hi = min(counts.values()), max(counts.values())
    assert lo > 0
    assert (hi - lo) / lo <= PARAM_PARITY_TOLERANCE, counts


def test_analytic_baselines_have_no_particle_side_encoder():
    for method in ("gaussian", "kmoments"):
        assert _build_extractor(method).particle_encoder_parameters() == 0


def test_find_features_extractor_handles_sac_style_policies():
    """Regression: SAC leaves `policy.features_extractor` as None and builds separate
    actor/critic extractors, so reading it unconditionally crashed every SAC run -- after
    training had already been paid for."""
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "experiments" / "benchmark"))
    spec = importlib.util.spec_from_file_location(
        "_bench_train", root / "experiments" / "benchmark" / "train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    sentinel = object()

    class _PPOish:
        features_extractor = sentinel

    class _Sub:
        features_extractor = sentinel

    class _SACish:
        features_extractor = None
        actor = _Sub()

    assert train.find_features_extractor(_PPOish()) is sentinel
    assert train.find_features_extractor(_SACish()) is sentinel

    class _Neither:
        features_extractor = None

    with pytest.raises(AttributeError):
        train.find_features_extractor(_Neither())


def test_cgf_honours_the_matched_bottleneck_even_though_it_is_param_exempt():
    """The exemption is about parameter COUNT only. Every learned encoder, CGF included,
    must still hand the policy the same width, or the comparison is not like-for-like."""
    from set_transformer.rl.benchmark.registry import MATCHED_STAT_DIM
    for method in ("cgf", "cgf_frozen", "cgf_align_finetune"):
        assert _build_extractor(method)._particle_stat_dim() == MATCHED_STAT_DIM


@pytest.mark.parametrize("kind,ae_name", [("cgf", "CGFAutoencoder"),
                                          ("ds", "DeepSetAE"), ("pn", "PointNetAE")])
def test_pretrained_checkpoints_load_into_their_extractor(tmp_path, kind, ae_name):
    """Every pretrainable family must build its autoencoder with the SAME arch the
    pretraining script uses.

    `load_state_dict` is strict, so a decoder width that differs from the checkpoint's
    fails the load even though the decoder never runs in the policy. This is not
    hypothetical: 16 runs of a sweep died this way because CGF took the CLI's
    dim_hidden=64 (chosen for the ST) while its checkpoint was written at 128.
    """
    import importlib
    import torch
    from set_transformer.rl.benchmark.registry import (
        METHOD_REGISTRY, build_extractor_kwargs,
    )
    import set_transformer.models as models

    pretrain = importlib.import_module("importlib.util")
    spec = pretrain.spec_from_file_location(
        "_pre", "experiments/benchmark/pretrain/3_pretrain_encoder.py")
    mod = pretrain.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ae_cls, ae_arch = mod.ENCODERS[kind]
    assert ae_cls is getattr(models, ae_name)
    ae = ae_cls(num_particles=100, dim_particles=2, num_encodings=8, dim_encoder=2,
                **ae_arch)
    ckpt = tmp_path / "enc.pt"
    torch.save(ae.state_dict(), ckpt)

    method = next(m for m, sp in METHOD_REGISTRY.items()
                  if sp.encoder_kind == kind and m.endswith("_frozen")
                  and "align" not in m)
    ms = METHOD_REGISTRY[method]
    kwargs = build_extractor_kwargs(
        ms, 128, [64, 64], pretrained_model_path=str(ckpt),
        # deliberately the ST-shaped CLI arch, which is what bit us
        encoder_arch=dict(num_encodings=8, dim_encoder=2, num_inds=32, dim_hidden=64,
                          num_heads=4, ln=True),
    )
    space = gym.spaces.Dict({
        "obs": gym.spaces.Box(-1.0, 1.0, (8,)),
        "particles": gym.spaces.Box(-9.0, 9.0, (100, 2)),
    })
    ms.extractor_class(space, **kwargs)      # must not raise
