"""Tests for POMDP pretraining module."""

import argparse
import os

import numpy as np
import pytest
import torch

from set_transformer.data.dataset import get_data_loader
from set_transformer.models.pf_set_transformer import PFSetTransformer
from set_transformer.run_pomdp_pretraining import plot, test, train


@pytest.fixture
def setup_and_cleanup():
    """Setup test environment and clean up after tests."""
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not os.path.exists("results"):
        os.makedirs("results")

    yield

    # Cleanup
    if os.path.exists("results"):
        import shutil

        shutil.rmtree("results")


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create a small dataset for testing
    data = np.random.randn(100, 50, 4).astype(np.float32)
    with open("sample_data.npy", "wb") as f:
        np.save(f, data)
    yield "sample_data.npy"
    os.remove("sample_data.npy")


@pytest.mark.slow
def test_train(setup_and_cleanup, sample_data):
    """Test POMDP pretraining."""
    # Override default arguments for faster testing
    import argparse

    args = argparse.Namespace(
        mode="train",
        net="set_transformer",
        B=2,  # Small batch size
        N_min=10,
        N_max=20,
        K=50,  # Number of particles
        gpu="0",
        lr=1e-3,
        run_name="test_run",
        num_steps=5,  # Very few steps for testing
        test_freq=2,
        save_freq=5,
        batch_size=4,  # Small batch size for testing
    )

    # Create data loaders
    train_loader, eval_loader, train_size, eval_size = get_data_loader(
        batch_size=args.batch_size,
        data_path=sample_data,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Train for a few steps
    train(args, train_loader, eval_loader, train_size, eval_size)

    # Check if model was saved
    model_path = os.path.join("results", args.net, args.run_name, "model.tar")
    assert os.path.exists(model_path)

    # Load and verify the model
    checkpoint = torch.load(model_path)
    assert "state_dict" in checkpoint


def test_test(setup_and_cleanup, sample_data):
    """Test model evaluation."""
    # Create data loaders
    train_loader, eval_loader, train_size, eval_size = get_data_loader(
        batch_size=4,
        data_path=sample_data,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run test function
    result = test(eval_loader, eval_size, verbose=True)
    assert isinstance(result, tuple)
    assert len(result) == 2  # Should return a line of text and loss value
    assert isinstance(result[0], str)
    assert isinstance(result[1], float)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_plot(setup_and_cleanup, sample_data):
    """Test plotting functionality."""
    # Create data loaders
    train_loader, eval_loader, train_size, eval_size = get_data_loader(
        batch_size=4,
        data_path=sample_data,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run plot function
    plot(eval_loader)
    # Note: We're not checking the actual plot output, just that it runs without error
