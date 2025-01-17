from typing import Iterator

import pytest
import torch
import torch.nn as nn

from set_transformer.modules import ISAB, MAB, PMA, SAB


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def seq_length() -> int:
    return 10


@pytest.fixture
def dim_input() -> int:
    return 64


@pytest.fixture
def dim_output() -> int:
    return 128


@pytest.fixture
def num_heads() -> int:
    return 4


def test_mab(
    batch_size: int, seq_length: int, dim_input: int, dim_output: int, num_heads: int
) -> None:
    # Test both with and without layer normalization
    for ln in [True, False]:
        mab = MAB(dim_input, dim_input, dim_output, num_heads, ln=ln)
        Q = torch.randn(batch_size, seq_length, dim_input)
        K = torch.randn(batch_size, seq_length, dim_input)

        output = mab(Q, K)

        # Check output shape
        assert output.shape == (batch_size, seq_length, dim_output)
        # Check output is not None and contains no NaN values
        assert output is not None
        assert not torch.isnan(output).any()


def test_sab(
    batch_size: int, seq_length: int, dim_input: int, dim_output: int, num_heads: int
) -> None:
    for ln in [True, False]:
        sab = SAB(dim_input, dim_output, num_heads, ln=ln)
        X = torch.randn(batch_size, seq_length, dim_input)

        output = sab(X)

        assert output.shape == (batch_size, seq_length, dim_output)
        assert output is not None
        assert not torch.isnan(output).any()


def test_isab(
    batch_size: int, seq_length: int, dim_input: int, dim_output: int, num_heads: int
) -> None:
    num_inds = 5  # Number of inducing points
    for ln in [True, False]:
        isab = ISAB(dim_input, dim_output, num_heads, num_inds, ln=ln)
        X = torch.randn(batch_size, seq_length, dim_input)

        output = isab(X)

        assert output.shape == (batch_size, seq_length, dim_output)
        assert output is not None
        assert not torch.isnan(output).any()


def test_pma(batch_size: int, seq_length: int, dim_input: int, num_heads: int) -> None:
    num_seeds = 3  # Number of seed vectors
    for ln in [True, False]:
        pma = PMA(dim_input, num_heads, num_seeds, ln=ln)
        X = torch.randn(batch_size, seq_length, dim_input)

        output = pma(X)

        assert output.shape == (batch_size, num_seeds, dim_input)
        assert output is not None
        assert not torch.isnan(output).any()


def test_invalid_inputs() -> None:
    with pytest.raises(RuntimeError):
        # Test with dimensions that will cause a tensor split error
        # dim_V=31 is not divisible by num_heads=4
        mab = MAB(dim_Q=31, dim_K=31, dim_V=31, num_heads=4)
        Q = torch.randn(32, 10, 31)
        K = torch.randn(32, 10, 31)
        mab(Q, K)


def test_device_compatibility(
    batch_size: int, seq_length: int, dim_input: int, dim_output: int, num_heads: int
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mab = MAB(dim_input, dim_input, dim_output, num_heads).to(device)
        Q = torch.randn(batch_size, seq_length, dim_input, device=device)
        K = torch.randn(batch_size, seq_length, dim_input, device=device)

        output = mab(Q, K)

        # Just check the device type, not the specific device object
        assert output.device.type == "cuda"
