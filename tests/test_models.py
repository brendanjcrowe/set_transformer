import pytest
import torch
import torch.nn as nn

from set_transformer.models import DeepSet, SetTransformer


@pytest.fixture
def batch_size() -> int:
    return 16


@pytest.fixture
def seq_length() -> int:
    return 10


@pytest.fixture
def dim_input() -> int:
    return 64


@pytest.fixture
def dim_output() -> int:
    return 32


@pytest.fixture
def num_outputs() -> int:
    return 5


@pytest.fixture
def dim_hidden() -> int:
    return 128


@pytest.fixture
def num_heads() -> int:
    return 4


def test_deepset(
    batch_size: int,
    seq_length: int,
    dim_input: int,
    dim_output: int,
    num_outputs: int,
    dim_hidden: int,
) -> None:
    model = DeepSet(dim_input, num_outputs, dim_output, dim_hidden)
    X = torch.randn(batch_size, seq_length, dim_input)

    output = model(X)

    # Check output shape
    assert output.shape == (batch_size, num_outputs, dim_output)
    # Check output is not None and contains no NaN values
    assert output is not None
    assert not torch.isnan(output).any()


def test_set_transformer(
    batch_size: int,
    seq_length: int,
    dim_input: int,
    dim_output: int,
    num_outputs: int,
    dim_hidden: int,
    num_heads: int,
) -> None:
    # Test both with and without layer normalization
    for ln in [True, False]:
        model = SetTransformer(
            dim_input=dim_input,
            num_outputs=num_outputs,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
            ln=ln,
        )
        X = torch.randn(batch_size, seq_length, dim_input)

        output = model(X)

        assert output.shape == (batch_size, num_outputs, dim_output)
        assert output is not None
        assert not torch.isnan(output).any()


def test_model_device_compatibility(
    batch_size: int,
    seq_length: int,
    dim_input: int,
    dim_output: int,
    num_outputs: int,
    dim_hidden: int,
    num_heads: int,
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")

        # Test DeepSet
        deepset = DeepSet(dim_input, num_outputs, dim_output, dim_hidden).to(device)
        X_deepset = torch.randn(batch_size, seq_length, dim_input, device=device)
        output_deepset = deepset(X_deepset)
        assert output_deepset.device.type == "cuda"

        # Test SetTransformer
        transformer = SetTransformer(
            dim_input=dim_input,
            num_outputs=num_outputs,
            dim_output=dim_output,
            dim_hidden=dim_hidden,
            num_heads=num_heads,
        ).to(device)
        X_transformer = torch.randn(batch_size, seq_length, dim_input, device=device)
        output_transformer = transformer(X_transformer)
        assert output_transformer.device.type == "cuda"


def test_gradient_flow(
    batch_size: int,
    seq_length: int,
    dim_input: int,
    dim_output: int,
    num_outputs: int,
    dim_hidden: int,
    num_heads: int,
) -> None:
    # Test DeepSet
    deepset = DeepSet(dim_input, num_outputs, dim_output, dim_hidden)
    X_deepset = torch.randn(batch_size, seq_length, dim_input, requires_grad=True)
    output_deepset = deepset(X_deepset)
    loss_deepset = output_deepset.sum()
    loss_deepset.backward()
    assert X_deepset.grad is not None
    assert not torch.isnan(X_deepset.grad).any()

    # Test SetTransformer
    transformer = SetTransformer(
        dim_input=dim_input,
        num_outputs=num_outputs,
        dim_output=dim_output,
        dim_hidden=dim_hidden,
        num_heads=num_heads,
    )
    X_transformer = torch.randn(batch_size, seq_length, dim_input, requires_grad=True)
    output_transformer = transformer(X_transformer)
    loss_transformer = output_transformer.sum()
    loss_transformer.backward()
    assert X_transformer.grad is not None
    assert not torch.isnan(X_transformer.grad).any()
