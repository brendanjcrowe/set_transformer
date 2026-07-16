"""Tests for the statistical particle-set feature extractors.

Skipped automatically if the ``[rl]`` extra (gymnasium / stable-baselines3) is not
installed.
"""

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

import numpy as np
import torch

from set_transformer.rl.feature_extractors import (
    CGFExtractor,
    GaussianExtractor,
    KMomentsExtractor,
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


def _extractors():
    space = _obs_space()
    return [
        GaussianExtractor(space, features_dim=FEATURES_DIM),
        KMomentsExtractor(space, k=4, features_dim=FEATURES_DIM),
        CGFExtractor(space, num_t=16, features_dim=FEATURES_DIM),
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
