"""Tests for training and evaluation utilities."""

import argparse
import os
import tempfile

import numpy as np
import pytest
import torch

from set_transformer.mixture_of_mvns import MixtureOfMVNs
from set_transformer.models import DeepSet, SetTransformer
from set_transformer.mvn_diag import MultivariateNormalDiag
from set_transformer.run import generate_benchmark, test, train


@pytest.fixture
def model_params():
    """Common model parameters."""
    return {
        "D": 2,  # Data dimension
        "K": 4,  # Number of components
        "dim_output": 4,  # 2*D
    }


@pytest.fixture
def sample_data(model_params):
    """Generate sample data for testing."""
    D = model_params["D"]
    mvn = MultivariateNormalDiag(D)
    mog = MixtureOfMVNs(mvn)
    X = mog.sample(10, 300, model_params["K"])
    return X


@pytest.fixture
def set_transformer(model_params):
    """Initialize SetTransformer model."""
    model = SetTransformer(
        model_params["D"], model_params["K"], model_params["dim_output"]
    ).cuda()
    return model


@pytest.fixture
def deep_set(model_params):
    """Initialize DeepSet model."""
    model = DeepSet(
        model_params["D"], model_params["K"], model_params["dim_output"]
    ).cuda()
    return model


def test_generate_benchmark(tmp_path):
    """Test benchmark dataset generation."""
    # Define N_list before using it
    N_list = [10, 20, 30]  # Example values
    os.environ["BENCHMARK_DIR"] = str(tmp_path)

    generate_benchmark()
    benchmark_file = os.path.join(tmp_path, "mog_4.pkl")

    assert os.path.exists(benchmark_file)
    bench = torch.load(benchmark_file)
    assert len(bench) == 2  # [data, log_likelihood]
    assert isinstance(bench[0], list)  # List of tensors
    assert isinstance(bench[1], float)  # Average log likelihood


def test_test_function(set_transformer, sample_data):
    """Test the test function."""
    bench = [[sample_data], 0.0]  # List of one dataset  # Dummy log likelihood

    result = test(bench, verbose=False)
    assert isinstance(result, str)
    assert "test ll" in result
    assert "oracle" in result


@pytest.mark.slow
def test_train_function(set_transformer, tmp_path):
    """Test the training function."""
    # Override save directory
    os.environ["SAVE_DIR"] = str(tmp_path)

    # Run training for a few steps
    import argparse

    args = argparse.Namespace(
        mode="train",
        net="set_transformer",
        B=2,  # Small batch size
        N_min=10,  # Few points
        N_max=20,
        K=4,
        gpu="0",
        lr=1e-3,
        run_name="test_run",
        num_steps=5,  # Very few steps for testing
        test_freq=2,
        save_freq=5,
    )

    train(args)

    # Check that model was saved
    model_file = os.path.join(tmp_path, "model.tar")
    assert os.path.exists(model_file)

    # Check that log file was created
    log_files = [f for f in os.listdir(tmp_path) if f.endswith(".log")]
    assert len(log_files) > 0


def test_model_forward_pass(set_transformer, deep_set, sample_data):
    """Test forward pass through both model architectures."""
    for model in [set_transformer, deep_set]:
        output = model(sample_data)
        assert isinstance(output, torch.Tensor)
        assert output.shape[1] == 4  # dim_output


def test_model_gradient_flow(set_transformer, sample_data):
    """Test that gradients flow through the model."""
    optimizer = torch.optim.Adam(set_transformer.parameters())

    # Forward pass
    output = set_transformer(sample_data)
    loss = output.mean()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check that gradients exist
    for param in set_transformer.parameters():
        assert param.grad is not None
