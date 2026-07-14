import pytest
import torch
import torch.nn as nn

from set_transformer.models import (
    DeepSet,
    DeepSetAE,
    DeepSetVAE,
    DeepSetVQVAE,
    PFSetTransformer,
    SetTransformer,
    SetVAE,
    SetVQVAE,
)


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


@pytest.fixture
def num_particles() -> int:
    return 64


@pytest.fixture
def dim_particles() -> int:
    return 2


@pytest.fixture
def num_encodings() -> int:
    return 8


@pytest.fixture
def dim_encoder() -> int:
    return 16


def test_set_vae_shapes(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    model = SetVAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
        ln=True,
    )
    X = torch.randn(batch_size, num_particles, dim_particles)
    out = model(X)

    assert out["recon"].shape == (batch_size, num_particles, dim_particles)
    assert out["mu"].shape == (batch_size, num_encodings, dim_encoder)
    assert out["logvar"].shape == (batch_size, num_encodings, dim_encoder)
    assert out["kl"].dim() == 0
    assert not torch.isnan(out["recon"]).any()
    assert not torch.isnan(out["kl"]).any()


def test_set_vae_eval_uses_mean(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    model = SetVAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
        ln=True,
    )
    model.eval()
    X = torch.randn(batch_size, num_particles, dim_particles)
    with torch.no_grad():
        a = model(X)["recon"]
        b = model(X)["recon"]
    assert torch.allclose(a, b)


def test_set_vae_gradient_flow(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    model = SetVAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
        ln=True,
    )
    X = torch.randn(batch_size, num_particles, dim_particles, requires_grad=True)
    out = model(X)
    (out["recon"].pow(2).mean() + 1e-3 * out["kl"]).backward()
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()


def test_set_vqvae_shapes(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    codebook_size = 32
    model = SetVQVAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
        codebook_size=codebook_size,
        ln=True,
    )
    X = torch.randn(batch_size, num_particles, dim_particles)
    out = model(X)

    assert out["recon"].shape == (batch_size, num_particles, dim_particles)
    assert out["z_e"].shape == (batch_size, num_encodings, dim_encoder)
    assert out["z_q"].shape == (batch_size, num_encodings, dim_encoder)
    assert out["indices"].shape == (batch_size, num_encodings)
    assert out["commitment_loss"].dim() == 0
    assert out["perplexity"].dim() == 0
    assert out["indices"].max().item() < codebook_size
    assert out["indices"].min().item() >= 0


def test_set_vqvae_straight_through_gradient(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    model = SetVQVAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
        codebook_size=32,
        ln=True,
    )
    X = torch.randn(batch_size, num_particles, dim_particles, requires_grad=True)
    out = model(X)
    (out["recon"].pow(2).mean() + 0.25 * out["commitment_loss"]).backward()
    # Gradient must flow through the encoder via the straight-through estimator.
    enc_param = next(model.set_transformer.parameters())
    assert enc_param.grad is not None
    assert enc_param.grad.abs().sum().item() > 0
    assert X.grad is not None
    assert not torch.isnan(X.grad).any()


def test_set_vqvae_codebook_ema_updates_during_training(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    torch.manual_seed(0)
    model = SetVQVAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
        codebook_size=16,
        ln=True,
    )
    model.train()
    before = model.quantizer.embedding.clone()
    X = torch.randn(batch_size, num_particles, dim_particles)
    _ = model(X)
    assert not torch.allclose(before, model.quantizer.embedding)

    model.eval()
    before_eval = model.quantizer.embedding.clone()
    with torch.no_grad():
        _ = model(X)
    assert torch.allclose(before_eval, model.quantizer.embedding)


def test_deep_set_ae_shapes(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    model = DeepSetAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
    )
    X = torch.randn(batch_size, num_particles, dim_particles)
    out = model(X)
    assert out.shape == (batch_size, num_particles, dim_particles)
    assert not torch.isnan(out).any()


def test_deep_set_vae_shapes(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    model = DeepSetVAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
    )
    X = torch.randn(batch_size, num_particles, dim_particles)
    out = model(X)
    assert out["recon"].shape == (batch_size, num_particles, dim_particles)
    assert out["mu"].shape == (batch_size, num_encodings, dim_encoder)
    assert out["logvar"].shape == (batch_size, num_encodings, dim_encoder)
    assert out["kl"].dim() == 0


def test_deep_set_vqvae_shapes_and_grad(
    batch_size: int,
    num_particles: int,
    dim_particles: int,
    num_encodings: int,
    dim_encoder: int,
) -> None:
    model = DeepSetVQVAE(
        num_particles=num_particles,
        dim_particles=dim_particles,
        num_encodings=num_encodings,
        dim_encoder=dim_encoder,
        codebook_size=16,
    )
    X = torch.randn(batch_size, num_particles, dim_particles, requires_grad=True)
    out = model(X)
    assert out["recon"].shape == (batch_size, num_particles, dim_particles)
    assert out["z_e"].shape == (batch_size, num_encodings, dim_encoder)
    assert out["indices"].shape == (batch_size, num_encodings)
    (out["recon"].pow(2).mean() + 0.25 * out["commitment_loss"]).backward()
    enc_param = next(model.encoder.parameters())
    assert enc_param.grad is not None
    assert enc_param.grad.abs().sum().item() > 0
