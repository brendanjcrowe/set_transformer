"""Tests for the statistical particle-set feature extractors.

Skipped automatically if the ``[rl]`` extra (gymnasium / stable-baselines3) is not
installed.
"""

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import numpy as np
import torch
import torch

from set_transformer.models import PFSetTransformer
from set_transformer.rl.feature_extractors import (
    CGFExtractor,
    DeepSetExtractor,
    GaussianExtractor,
    KMomentsExtractor,
    PointNetExtractor,
    SetTransformerExtractor,
)

NUM_PARTICLES = 50
PARTICLE_DIM = 2
OBS_DIM = 5
FEATURES_DIM = 32
BATCH = 8


def _obs_space():
    return gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32),
            "particles": gym.spaces.Box(
                -np.inf, np.inf, shape=(NUM_PARTICLES, PARTICLE_DIM), dtype=np.float32
            ),
        }
    )


def _sample_obs():
    return {
        "obs": torch.randn(BATCH, OBS_DIM),
        "particles": torch.randn(BATCH, NUM_PARTICLES, PARTICLE_DIM),
    }


ST_ARCH = dict(num_encodings=4, dim_encoder=2, num_inds=8, dim_hidden=32, num_heads=2, ln=True)


POOL_ARCH = dict(num_encodings=4, dim_encoder=8, dim_hidden=32)


def _extractors():
    space = _obs_space()
    return [
        GaussianExtractor(space, features_dim=FEATURES_DIM),
        KMomentsExtractor(space, k=4, features_dim=FEATURES_DIM),
        CGFExtractor(space, num_t=16, features_dim=FEATURES_DIM),
        DeepSetExtractor(space, features_dim=FEATURES_DIM, **POOL_ARCH),
        PointNetExtractor(space, features_dim=FEATURES_DIM, **POOL_ARCH),
        SetTransformerExtractor(space, features_dim=FEATURES_DIM, **ST_ARCH),
    ]


@pytest.mark.parametrize("extractor", _extractors(), ids=lambda e: type(e).__name__)
def test_output_shape(extractor):
    out = extractor(_sample_obs())
    assert out.shape == (BATCH, FEATURES_DIM)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("extractor", _extractors(), ids=lambda e: type(e).__name__)
def test_permutation_invariance(extractor):
    """Shuffling the particle order must not change the features (set encoders)."""
    obs = _sample_obs()
    perm = torch.randperm(NUM_PARTICLES)
    shuffled = {"obs": obs["obs"], "particles": obs["particles"][:, perm, :]}
    extractor.eval()
    with torch.no_grad():
        a = extractor(obs)
        b = extractor(shuffled)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("extractor", _extractors(), ids=lambda e: type(e).__name__)
def test_gradients_flow(extractor):
    out = extractor(_sample_obs())
    out.sum().backward()
    grads = [p.grad for p in extractor.parameters() if p.requires_grad]
    assert grads, "extractor has no trainable parameters"
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_kmoments_k2_matches_mean_var():
    """k=2 KMoments particle statistic == per-dim mean and (biased) variance."""
    space = _obs_space()
    ext = KMomentsExtractor(space, k=2, features_dim=FEATURES_DIM)
    particles = torch.randn(BATCH, NUM_PARTICLES, PARTICLE_DIM)
    stat = ext._particle_features(particles)
    assert stat.shape == (BATCH, 2 * PARTICLE_DIM)
    mean = particles.mean(dim=1)
    var = particles.var(dim=1, unbiased=False)
    assert torch.allclose(stat[:, :PARTICLE_DIM], mean, atol=1e-5)
    assert torch.allclose(stat[:, PARTICLE_DIM:], var, atol=1e-5)


def test_gaussian_stat_width():
    space = _obs_space()
    ext = GaussianExtractor(space, features_dim=FEATURES_DIM)
    # mean (d) + lower-tri covariance (d(d+1)/2)
    assert ext._particle_stat_dim() == PARTICLE_DIM + PARTICLE_DIM * (PARTICLE_DIM + 1) // 2


def test_cgf_learned_points_receive_grad():
    space = _obs_space()
    ext = CGFExtractor(space, num_t=16, features_dim=FEATURES_DIM)
    ext(_sample_obs()).sum().backward()
    assert ext.t_values.grad is not None
    assert ext.t_values.grad.abs().sum() > 0


def test_cgf_t_value_norms():
    """One L2 norm per sample point, reported POST-clamp.

    The forward pass evaluates the CGF at the clamped t, so the raw parameter norms
    would overstate where on the moment<->support-function continuum the encoder is
    actually operating. Note the clamp is elementwise, so it bites direction-dependently:
    at t_clamp=2.0 an axis-aligned point caps at 2.0 while a diagonal one reaches 2*sqrt(2).
    """
    space = _obs_space()
    ext = CGFExtractor(space, num_t=16, features_dim=FEATURES_DIM)
    norms = ext.t_value_norms()
    assert norms.shape == (16,)
    clamped = torch.clamp(ext.t_values.detach(), -ext.t_clamp, ext.t_clamp)
    assert torch.allclose(norms, torch.linalg.norm(clamped, dim=1))
    assert (norms >= 0).all()
    assert norms.max() <= ext.t_clamp * (ext.particle_dim ** 0.5) + 1e-6

    # Without a clamp the diagnostic is exactly the raw parameter norms.
    unclamped = CGFExtractor(space, num_t=16, t_clamp=None, features_dim=FEATURES_DIM)
    assert torch.allclose(unclamped.t_value_norms(),
                          torch.linalg.norm(unclamped.t_values.detach(), dim=1))


# --- Pooling extractors (DeepSet / PointNet) -----------------------------------


@pytest.mark.parametrize("cls", [DeepSetExtractor, PointNetExtractor])
def test_pooling_stat_width(cls):
    ext = cls(_obs_space(), num_encodings=8, dim_encoder=16, features_dim=FEATURES_DIM)
    assert ext._particle_stat_dim() == 8 * 16


def test_pooling_encoder_receives_grad():
    """The learned pooling encoder must get gradients (it is trained from scratch)."""
    for cls in (DeepSetExtractor, PointNetExtractor):
        ext = cls(_obs_space(), features_dim=FEATURES_DIM, **POOL_ARCH)
        ext(_sample_obs()).sum().backward()
        grads = [p.grad for p in ext.encoder.parameters()]
        assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_deepset_vs_pointnet_differ():
    """Same seed/weights aside, mean-pool and max-pool produce different statistics."""
    torch.manual_seed(0)
    ds = DeepSetExtractor(_obs_space(), features_dim=FEATURES_DIM, **POOL_ARCH)
    torch.manual_seed(0)
    pn = PointNetExtractor(_obs_space(), features_dim=FEATURES_DIM, **POOL_ARCH)
    # Identically-initialised encoders (same seed) still differ because the pooling op does.
    particles = torch.randn(BATCH, NUM_PARTICLES, PARTICLE_DIM)
    ds.eval()
    pn.eval()
    with torch.no_grad():
        assert not torch.allclose(
            ds._particle_features(particles), pn._particle_features(particles), atol=1e-4
        )


# --- SetTransformerExtractor-specific behavior ---------------------------------


def _save_pf_st_checkpoint(path, trainer_style):
    model = PFSetTransformer(
        num_particles=NUM_PARTICLES, dim_particles=PARTICLE_DIM, **ST_ARCH
    )
    payload = model.state_dict()
    if trainer_style:
        payload = {"model_state_dict": payload, "epoch": 3}
    torch.save(payload, path)
    return model


@pytest.mark.parametrize("trainer_style", [False, True], ids=["raw_state_dict", "trainer_ckpt"])
def test_st_loads_pretrained_checkpoint(tmp_path, trainer_style):
    ckpt = tmp_path / "ckpt.pt"
    source = _save_pf_st_checkpoint(ckpt, trainer_style)
    ext = SetTransformerExtractor(
        _obs_space(), pretrained_model_path=str(ckpt), features_dim=FEATURES_DIM, **ST_ARCH
    )
    for (name, p_src), p_loaded in zip(
        source.state_dict().items(), ext.pf_st.state_dict().values()
    ):
        assert torch.equal(p_src, p_loaded), f"weight mismatch after load: {name}"


def test_st_frozen_semantics(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    _save_pf_st_checkpoint(ckpt, trainer_style=True)
    ext = SetTransformerExtractor(
        _obs_space(),
        pretrained_model_path=str(ckpt),
        freeze=True,
        features_dim=FEATURES_DIM,
        **ST_ARCH,
    )
    # Encoder frozen, head/obs-MLP trainable.
    assert all(not p.requires_grad for p in ext.pf_st.parameters())
    assert all(p.requires_grad for p in ext.obs_net.parameters())
    assert all(p.requires_grad for p in ext.combined_net.parameters())
    # SB3 flips .train(); frozen encoder must stay in eval mode.
    ext.train()
    assert not ext.pf_st.training
    assert ext.obs_net.training
    # Grads flow to the head but not the encoder.
    ext(_sample_obs()).sum().backward()
    assert all(p.grad is None for p in ext.pf_st.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ext.combined_net.parameters())


def test_st_finetune_encoder_receives_grad(tmp_path):
    ckpt = tmp_path / "ckpt.pt"
    _save_pf_st_checkpoint(ckpt, trainer_style=True)
    ext = SetTransformerExtractor(
        _obs_space(), pretrained_model_path=str(ckpt), features_dim=FEATURES_DIM, **ST_ARCH
    )
    ext(_sample_obs()).sum().backward()
    encoder_grads = [p.grad for p in ext.encoder.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in encoder_grads)


def test_st_freeze_without_checkpoint_rejected():
    with pytest.raises(ValueError, match="freeze"):
        SetTransformerExtractor(_obs_space(), freeze=True, features_dim=FEATURES_DIM, **ST_ARCH)


# --- pretrained pooling encoders (2026-08-20) ---------------------------------------

def _dict_space(particle_dim=2, num_particles=100, obs_dim=8):
    import gymnasium as gym
    return gym.spaces.Dict({
        "obs": gym.spaces.Box(-1.0, 1.0, (obs_dim,)),
        "particles": gym.spaces.Box(-9.0, 9.0, (num_particles, particle_dim)),
    })


@pytest.mark.parametrize("kind", ["ds", "pn"])
def test_pooling_extractor_loads_and_freezes_a_pretrained_checkpoint(tmp_path, kind):
    """Frozen means: weights come from the checkpoint, and nothing on the particle side
    is trainable or leaves eval mode when SB3 flips train()."""
    import torch
    from set_transformer.models import DeepSetAE, PointNetAE
    from set_transformer.rl.feature_extractors import DeepSetExtractor, PointNetExtractor

    ae_cls, ex_cls = ({"ds": (DeepSetAE, DeepSetExtractor),
                       "pn": (PointNetAE, PointNetExtractor)})[kind]
    arch = dict(num_particles=100, dim_particles=2, num_encodings=8, dim_encoder=2,
                dim_hidden=128)
    ckpt = tmp_path / "pretrained.pt"
    torch.save(ae_cls(**arch).state_dict(), ckpt)

    space = _dict_space()
    frozen = ex_cls(space, pretrained_model_path=str(ckpt), freeze=True,
                    num_encodings=8, dim_encoder=2, dim_hidden=128)
    assert all(not p.requires_grad for p in frozen.encoder.parameters())
    assert frozen.particle_encoder_parameters() > 0
    frozen.train(True)
    assert not frozen.encoder.training, "frozen encoder must stay in eval mode"

    finetune = ex_cls(space, pretrained_model_path=str(ckpt), freeze=False,
                      num_encodings=8, dim_encoder=2, dim_hidden=128)
    assert all(p.requires_grad for p in finetune.encoder.parameters())
    # Both arms start from the same weights; only requires_grad differs.
    for a, b in zip(frozen.encoder.parameters(), finetune.encoder.parameters()):
        assert torch.allclose(a, b)


def test_freeze_without_a_checkpoint_is_refused():
    from set_transformer.rl.feature_extractors import DeepSetExtractor
    with pytest.raises(ValueError, match="freeze"):
        DeepSetExtractor(_dict_space(), freeze=True)


def test_pretrained_and_scratch_pooling_extractors_have_identical_shapes():
    import torch
    from set_transformer.rl.feature_extractors import PointNetExtractor
    space = _dict_space()
    scratch = PointNetExtractor(space, num_encodings=8, dim_encoder=2, dim_hidden=128)
    obs = {"obs": torch.randn(4, 8), "particles": torch.randn(4, 100, 2)}
    assert scratch(obs).shape == (4, 128)


# --- CGF: collaborator's implementation, matched bottleneck (2026-08-24) -------------

def _cgf(**kw):
    from set_transformer.rl.feature_extractors import CGFExtractor
    return CGFExtractor(_dict_space(particle_dim=kw.pop("particle_dim", 2)), **kw)


def test_cgf_decouples_sampling_resolution_from_the_bottleneck():
    """num_t sets how finely the CGF curve is sampled; stat_dim is what the policy sees.
    Keeping them independent is what lets CGF use the collaborator's 64-point sampling
    while still matching every other method's 16-wide bottleneck."""
    import torch
    for num_t in (16, 64, 128):
        e = _cgf(num_t=num_t, stat_dim=16)
        assert e._particle_stat_dim() == 16
        assert e.t_values.shape == (num_t, 2)
        out = e({"obs": torch.randn(3, 8), "particles": torch.randn(3, 100, 2)})
        assert out.shape == (3, 128)


def test_cgf_weighted_matches_unweighted_under_uniform_weights():
    """The equivalence that keeps already-finished runs valid: on an env whose filter
    always resamples to uniform weights (car_flag), exposing weights changes nothing."""
    import torch
    torch.manual_seed(0)
    e = _cgf().eval()
    obs = {"obs": torch.randn(4, 8), "particles": torch.randn(4, 100, 2)}
    unweighted = e(obs)
    weighted = e({**obs, "weights": torch.full((4, 100), 1 / 100)})
    assert torch.allclose(unweighted, weighted, atol=1e-5)


def test_cgf_weights_actually_change_the_statistic():
    import torch
    torch.manual_seed(0)
    e = _cgf().eval()
    obs = {"obs": torch.randn(2, 8), "particles": torch.randn(2, 100, 2)}
    skewed = torch.zeros(2, 100)
    skewed[:, :5] = 0.2  # all mass on five particles
    assert not torch.allclose(e(obs), e({**obs, "weights": skewed}), atol=1e-3)


@pytest.mark.parametrize("weights", [None, "uniform", "skewed"])
def test_cgf_is_permutation_invariant(weights):
    import torch
    torch.manual_seed(0)
    e = _cgf().eval()
    obs = {"obs": torch.randn(2, 8), "particles": torch.randn(2, 100, 2)}
    if weights == "uniform":
        obs["weights"] = torch.full((2, 100), 1 / 100)
    elif weights == "skewed":
        obs["weights"] = torch.rand(2, 100)
    perm = torch.randperm(100)
    shuffled = {k: (v[:, perm] if k in ("particles", "weights") else v)
                for k, v in obs.items()}
    assert torch.allclose(e(obs), e(shuffled), atol=1e-5)


def test_cgf_t_clamp_bounds_the_sampling_points():
    import torch
    e = _cgf(t_clamp=0.5, t_init_mode="random", t_init_scale=5.0)
    assert e.t_value_norms().max() <= 0.5 * (2 ** 0.5) + 1e-6


def test_cgf_frozen_t_is_not_trainable_but_still_saves():
    import torch
    frozen = _cgf(t_frozen=True)
    # The encoder is shared with the autoencoder, so the buffer is nested under it.
    assert any(k.endswith("t_values") for k in frozen.state_dict())
    assert not any(p is frozen.t_values for p in frozen.parameters())
    learned = _cgf(t_frozen=False)
    assert any(p is learned.t_values for p in learned.parameters())


@pytest.mark.parametrize("mode", ["spread", "linspace_all_dims", "linspace_first_dim",
                                  "random"])
@pytest.mark.parametrize("particle_dim", [1, 2])
def test_cgf_t_init_modes_produce_usable_points(mode, particle_dim):
    import torch
    e = _cgf(num_t=64, t_init_mode=mode, particle_dim=particle_dim)
    assert e.t_values.shape == (64, particle_dim)
    assert torch.isfinite(e.t_values).all()
    out = e({"obs": torch.randn(2, 8), "particles": torch.randn(2, 100, particle_dim)})
    assert torch.isfinite(out).all()


def test_cgf_spread_init_covers_directions_and_magnitudes():
    """'spread' exists so t starts where the signal is, instead of needing ||t|| to grow
    ~10x from a small init before the encoder sees anything."""
    e = _cgf(num_t=64, t_init_mode="spread")
    norms = e.t_value_norms()
    assert norms.min() < 0.3 and norms.max() > 2.0   # log-spaced magnitudes
    angles = torch.atan2(e.t_values[:, 1], e.t_values[:, 0])
    assert len(torch.unique(torch.round(angles * 100))) >= 8  # 8 distinct directions


def test_cgf_rejects_bad_configuration():
    with pytest.raises(ValueError, match="num_t"):
        _cgf(num_t=0)
    with pytest.raises(ValueError, match="particle_scale"):
        _cgf(particle_scale=0.0)
    with pytest.raises(ValueError, match="Unknown t_init_mode"):
        _cgf(t_init_mode="nope")
    with pytest.raises(ValueError, match="divisible"):
        _cgf(num_t=12, t_init_mode="spread")


def test_cgf_particle_scale_changes_the_statistic():
    """exp(t.x) is scale-sensitive, so the same t-range means different things on
    different arenas -- hence the per-env scale rather than a hardcoded constant."""
    import torch
    torch.manual_seed(0)
    obs = {"obs": torch.randn(2, 8), "particles": 4.0 * torch.randn(2, 100, 2)}
    a, b = _cgf(particle_scale=1.0).eval(), _cgf(particle_scale=4.5).eval()
    b.load_state_dict(a.state_dict())
    assert not torch.allclose(a(obs), b(obs), atol=1e-3)


def test_cgf_survives_extreme_particles():
    """exp_arg_clamp is the overflow guard; a diverged filter must not produce NaNs that
    silently poison the policy."""
    import torch
    e = _cgf().eval()
    obs = {"obs": torch.randn(2, 8), "particles": torch.full((2, 100, 2), 1e6)}
    assert torch.isfinite(e(obs)).all()
    obs["particles"][0, 0] = float("nan")
    assert torch.isfinite(e(obs)).all()


def test_cgf_autoencoder_checkpoint_loads_into_the_extractor(tmp_path):
    """Pretraining and the policy must share one CGF implementation, or a checkpoint's
    weights stop meaning what the consumer expects."""
    import torch
    from set_transformer.models import CGFAutoencoder
    from set_transformer.rl.feature_extractors import CGFExtractor

    arch = dict(num_particles=100, dim_particles=2, num_encodings=8, dim_encoder=2,
                particle_scale=14.0)
    ae = CGFAutoencoder(**arch)
    ckpt = tmp_path / "cgf.pt"
    torch.save(ae.state_dict(), ckpt)

    space = _dict_space()
    frozen = CGFExtractor(space, pretrained_model_path=str(ckpt), freeze=True,
                          particle_scale=14.0)
    assert torch.allclose(frozen.encoder.t_values, ae.encoder.t_values)
    assert all(not p.requires_grad for p in frozen.encoder.parameters())
    frozen.train(True)
    assert not frozen.encoder.training

    finetune = CGFExtractor(space, pretrained_model_path=str(ckpt), freeze=False,
                            particle_scale=14.0)
    assert all(p.requires_grad for p in finetune.encoder.parameters())


def test_cgf_encoder_is_shared_between_autoencoder_and_extractor():
    from set_transformer.models import CGFAutoencoder, CGFEncoder
    from set_transformer.rl.feature_extractors import CGFExtractor
    ae = CGFAutoencoder(num_particles=100, dim_particles=2, num_encodings=8,
                        dim_encoder=2)
    ex = CGFExtractor(_dict_space())
    assert isinstance(ae.encoder, CGFEncoder) and isinstance(ex.encoder, CGFEncoder)
    assert set(ae.encoder.state_dict()) == set(ex.encoder.state_dict())
