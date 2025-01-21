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
