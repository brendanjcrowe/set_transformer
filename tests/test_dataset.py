"""Test dataset utilities."""

import os
import tempfile
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from set_transformer.data.dataset import POMDPDataset, get_dataset, get_data_loader


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return np.random.randn(100, 50, 4).astype(np.float32)


@pytest.fixture
def temp_data_file(sample_data):
    """Create temporary data file."""
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
        np.save(f, sample_data)
    yield f.name
    os.unlink(f.name)


def test_pomdp_dataset_initialization(sample_data):
    """Test POMDPDataset initialization."""
    dataset = POMDPDataset(sample_data)
    assert isinstance(dataset.data, torch.Tensor)
    assert dataset.data.shape == (100, 50, 4)


def test_pomdp_dataset_getitem(sample_data):
    """Test POMDPDataset __getitem__ method."""
    dataset = POMDPDataset(sample_data)
    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (50, 4)


def test_get_dataset(temp_data_file):
    """Test get_dataset function."""
    dataset = get_dataset(temp_data_file)
    assert isinstance(dataset, POMDPDataset)
    assert len(dataset) == 100


def test_get_data_loader(temp_data_file):
    """Test get_data_loader function."""
    train_loader, eval_loader, train_size, eval_size = get_data_loader(
        batch_size=16,
        data_path=temp_data_file,
        device="cpu"
    )

    assert isinstance(train_loader, DataLoader)
    assert isinstance(eval_loader, DataLoader)
    assert train_size + eval_size == 100


def test_dataset_device_transfer(temp_data_file):
    """Test dataset transfer between devices."""
    if torch.cuda.is_available():
        train_loader, eval_loader, _, _ = get_data_loader(
            batch_size=16,
            data_path=temp_data_file,
            device="cpu"  # Load on CPU first
        )

        # Check that data is on CPU
        train_batch = next(iter(train_loader))
        assert train_batch.device.type == "cpu"

        # Move to GPU
        train_batch = train_batch.cuda()
        assert train_batch.device.type == "cuda"


def test_invalid_split_ratio():
    """Test invalid train/eval split ratios."""
    with pytest.raises(ValueError):
        get_data_loader(
            batch_size=16,
            data_path="dummy.npy",
            device="cpu",
            train_split=1.5  # > 1
        )


def test_empty_dataset():
    """Test handling of empty dataset."""
    empty_data = np.array([]).reshape(0, 50, 4)
    with pytest.raises(ValueError):
        POMDPDataset(empty_data)


def test_dataset_shape_validation(sample_data):
    """Test dataset shape validation."""
    # Test invalid number of dimensions
    invalid_data = sample_data.reshape(100, -1)  # 2D instead of 3D
    with pytest.raises(ValueError):
        POMDPDataset(invalid_data)


# ---------------------------------------------------------------------------
# Weighted particle sets (.npz with particles + weights)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_weights(sample_data):
    """Normalized per-particle mass matching sample_data."""
    rng = np.random.default_rng(0)
    weights = rng.random(sample_data.shape[:2]).astype(np.float32)
    return weights / weights.sum(axis=1, keepdims=True)


@pytest.fixture
def temp_weighted_npz(sample_data, sample_weights):
    """A weighted dataset on disk, plus an unrelated metadata entry."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f, particles=sample_data, weights=sample_weights,
                 metadata="{}")
    yield f.name
    os.unlink(f.name)


def test_unweighted_dataset_returns_bare_tensor(sample_data):
    dataset = POMDPDataset(sample_data)
    assert not dataset.is_weighted
    assert isinstance(dataset[0], torch.Tensor)


def test_weighted_dataset_returns_pair(sample_data, sample_weights):
    dataset = POMDPDataset(sample_data, sample_weights)
    particles, weights = dataset[0]
    assert dataset.is_weighted
    assert particles.shape == torch.Size(sample_data.shape[1:])
    assert weights.shape == torch.Size([sample_data.shape[1]])


def test_weights_must_match_particles(sample_data):
    with pytest.raises(ValueError):
        POMDPDataset(sample_data, np.ones((len(sample_data), 7), dtype=np.float32))


def test_negative_weights_rejected(sample_data, sample_weights):
    bad = sample_weights.copy()
    bad[0, 0] = -1.0
    with pytest.raises(ValueError):
        POMDPDataset(sample_data, bad)


def test_npz_roundtrip_loads_weights(temp_weighted_npz, sample_data):
    dataset = get_dataset(temp_weighted_npz)
    assert dataset.is_weighted
    assert len(dataset) == len(sample_data)
    assert dataset.particle_dim == sample_data.shape[-1]


def test_npz_load_weights_false_ignores_them(temp_weighted_npz):
    """The unweighted ablation must be reachable from a weighted file."""
    dataset = get_dataset(temp_weighted_npz, load_weights=False)
    assert not dataset.is_weighted
    assert isinstance(dataset[0], torch.Tensor)


def test_legacy_npy_still_loads_unweighted(temp_data_file):
    dataset = get_dataset(temp_data_file)
    assert not dataset.is_weighted


def test_weighted_data_loader_yields_pairs(temp_weighted_npz):
    train_loader, _, _, _ = get_data_loader(
        batch_size=8, data_path=temp_weighted_npz, device="cpu"
    )
    particles, weights = next(iter(train_loader))
    assert particles.shape[0] == weights.shape[0]
    assert particles.shape[1] == weights.shape[1]


def test_particle_scale_divides_coordinates(sample_data):
    dataset = POMDPDataset(sample_data, particle_scale=7.0)
    assert dataset.particle_scale == 7.0
    assert float(dataset.data.abs().max()) == pytest.approx(
        float(np.abs(sample_data).max()) / 7.0, rel=1e-5
    )


def test_particle_scale_must_be_positive(sample_data):
    for bad in (0.0, -7.0):
        with pytest.raises(ValueError):
            POMDPDataset(sample_data, particle_scale=bad)


def test_stored_particle_scale_is_used_by_default(sample_data, sample_weights):
    """The RL extractor normalizes by the arena half-width, so pretraining
    must apply the same divisor or the encoder is trained on the wrong range."""
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        np.savez(f, particles=sample_data, weights=sample_weights,
                 particle_scale=np.float32(7.0))
    try:
        assert get_dataset(f.name).particle_scale == 7.0
        assert get_dataset(f.name, particle_scale=1.0).particle_scale == 1.0
    finally:
        os.unlink(f.name)


def test_missing_particle_scale_defaults_to_one(temp_weighted_npz):
    assert get_dataset(temp_weighted_npz).particle_scale == 1.0


def test_massless_weight_row_rejected_at_load(sample_data, sample_weights):
    bad = sample_weights.copy()
    bad[3] = 0.0
    with pytest.raises(ValueError, match="positive total weight"):
        POMDPDataset(sample_data, bad)


def test_nonfinite_weights_rejected_at_load(sample_data, sample_weights):
    bad = sample_weights.copy()
    bad[2, 1] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        POMDPDataset(sample_data, bad)

