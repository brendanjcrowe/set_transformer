"""Tests for the shared set-autoencoder training loop.

The properties worth pinning: the alignment arm actually improves latent<->EMD
correlation, the lambda ramp keeps it inert during warmup, and the checkpoints this loop
writes load into the RL feature extractors (the failure that silently retires a whole
pretraining run).
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from set_transformer.data.dataset import IndexedDataset, POMDPDataset
from set_transformer.emd_matrix import compute_matrix
from set_transformer.latent_alignment import (
    flatten_upper_triangle,
    latent_pairwise_distances,
    pearson_r,
)
from set_transformer.models import DeepSetAE, PointNetAE
from set_transformer.training.autoencoder import (
    AlignConfig,
    alignment_correlation,
    build_train_loss,
    train_autoencoder,
)

ARCH = dict(num_particles=24, dim_particles=2, num_encodings=4, dim_encoder=2,
            dim_hidden=32)


@pytest.fixture
def data(tmp_path):
    """Blob clouds whose pairwise EMD structure is dominated by blob location."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(-3, 3, size=(48, 1, 2))
    pts = (centers + 0.2 * rng.standard_normal((48, 24, 2))).astype(np.float32)
    emd = np.array(compute_matrix(pts, tmp_path / "emd.npy", block=16, verify=False,
                                  device="cpu"))
    return pts, emd


def _loaders(pts, indexed):
    ds = POMDPDataset(pts)
    if indexed:
        ds = IndexedDataset(ds)
    return (DataLoader(ds, batch_size=16, shuffle=True),
            DataLoader(POMDPDataset(pts), batch_size=16, shuffle=False))


def test_unknown_loss_type_is_rejected():
    with pytest.raises(ValueError):
        build_train_loss("l2")


def test_training_returns_a_loadable_best_state(data):
    pts, _ = data
    train_loader, val_loader = _loaders(pts, indexed=False)
    model = DeepSetAE(**ARCH)
    state, history, best = train_autoencoder(
        model, train_loader, val_loader, "cpu", num_epochs=2, loss_type="chamfer")
    assert set(state) == set(model.state_dict())
    assert history["val_emd"].shape == (2,)
    assert best == pytest.approx(history["val_emd"].min())
    DeepSetAE(**ARCH).load_state_dict(state)  # round-trips into a fresh model


def test_alignment_requires_a_matrix(data):
    pts, _ = data
    train_loader, val_loader = _loaders(pts, indexed=True)
    with pytest.raises(ValueError, match="emd_matrix"):
        train_autoencoder(DeepSetAE(**ARCH), train_loader, val_loader, "cpu",
                          num_epochs=1, align=AlignConfig())


def test_lambda_stays_zero_through_warmup(data):
    pts, emd = data
    train_loader, val_loader = _loaders(pts, indexed=True)
    _, history, _ = train_autoencoder(
        DeepSetAE(**ARCH), train_loader, val_loader, "cpu", num_epochs=4,
        loss_type="chamfer",
        align=AlignConfig(lam=0.5, warmup_epochs=2, ramp_epochs=2), emd_matrix=emd)
    assert list(history["align_lambda"]) == pytest.approx([0.0, 0.0, 0.25, 0.5])
    # During warmup the total loss is exactly the reconstruction loss.
    assert history["train_loss"][0] == pytest.approx(history["train_recon"][0], abs=1e-5)
    assert history["train_loss"][3] > history["train_recon"][3]


def test_alignment_improves_latent_emd_correlation(data):
    """The point of the whole arm: aligned training should reach a higher latent<->EMD
    correlation than reconstruction alone.

    Measured on the trained-to-completion model rather than the best-by-val-EMD
    checkpoint, so this tests the alignment term itself and not model selection (which
    ``test_warns_when_selection_predates_the_ramp`` covers separately).
    """
    pts, emd = data
    pairs = flatten_upper_triangle(torch.from_numpy(emd.astype(np.float32)))
    points = torch.from_numpy(pts)

    def trained_r(align):
        torch.manual_seed(0)
        np.random.seed(0)
        train_loader, val_loader = _loaders(pts, indexed=align is not None)
        model = PointNetAE(**ARCH)
        train_autoencoder(
            model, train_loader, val_loader, "cpu", num_epochs=20, loss_type="chamfer",
            align=align, emd_matrix=emd if align else None)
        return alignment_correlation(model, points, pairs, "cosine", "cpu")

    plain = trained_r(None)
    aligned = trained_r(AlignConfig(lam=1.0, warmup_epochs=2, ramp_epochs=3))
    assert aligned > plain, f"aligned {aligned:.3f} !> plain {plain:.3f}"


def test_warns_when_selection_predates_the_ramp(data, capsys):
    """Model selection is by val EMD, which is blind to alignment. If the best epoch
    lands before lambda reaches its target, the 'aligned' checkpoint is not actually
    aligned -- a silent mislabel, so it must be reported."""
    pts, emd = data
    train_loader, val_loader = _loaders(pts, indexed=True)
    _, history, _ = train_autoencoder(
        DeepSetAE(**ARCH), train_loader, val_loader, "cpu", num_epochs=3,
        loss_type="chamfer", emd_matrix=emd,
        align=AlignConfig(lam=0.5, warmup_epochs=10, ramp_epochs=5))
    assert int(history["best_epoch"]) < 10
    assert "not fully aligned" in capsys.readouterr().out


def test_best_epoch_indexes_the_recorded_history(data):
    pts, _ = data
    train_loader, val_loader = _loaders(pts, indexed=False)
    _, history, best = train_autoencoder(
        DeepSetAE(**ARCH), train_loader, val_loader, "cpu", num_epochs=4,
        loss_type="chamfer")
    assert history["val_emd"][int(history["best_epoch"])] == pytest.approx(best)


def test_alignment_correlation_matches_a_manual_computation(data):
    pts, emd = data
    model = DeepSetAE(**ARCH).eval()
    points = torch.from_numpy(pts)
    pairs = flatten_upper_triangle(torch.from_numpy(emd.astype(np.float32)))

    got = alignment_correlation(model, points, pairs, "cosine", "cpu")
    with torch.no_grad():
        z = model.encode(points).reshape(len(pts), -1)
    expected = float(pearson_r(latent_pairwise_distances(z, "cosine"), pairs))
    assert got == pytest.approx(expected, abs=1e-5)
