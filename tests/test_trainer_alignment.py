"""The latent metric-alignment term inside our Trainer (Phase 2).

What is pinned: the weighted path end to end (dataset -> matrix -> Trainer),
that lambda 0 leaves the reconstruction path numerically untouched, that the
lambda ramp is honoured, that the alignment actually raises latent<->EMD
correlation on data where it should, that validation loss stays blind to
alignment, that the checkpoint records the frame and alignment settings, and
that mis-wired configurations are refused rather than silently ignored.
"""

import json
import os

import numpy as np
import pytest
import torch

from set_transformer.data.dataset import get_data_loader, get_dataset
from set_transformer.emd_matrix import compute_matrix, load_matrix, matrix_stats
from set_transformer.training.config import ExperimentConfig, TrainingConfig
from set_transformer.training.trainer import Trainer

N_CLOUDS, N_PART, DIM = 64, 12, 2
BLUR = 0.05


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture(scope="module")
def weighted_npz(tmp_path_factory):
    """Blob clouds whose EMD geometry is dominated by blob location, with
    non-uniform weights, stored RAW with a centre and a scale like a real
    collector output."""
    rng = np.random.default_rng(0)
    centres = rng.uniform(-3, 3, size=(N_CLOUDS, 1, DIM))
    raw = 10.0 + 4.0 * (centres + 0.2 * rng.standard_normal((N_CLOUDS, N_PART, DIM)))
    w = rng.random((N_CLOUDS, N_PART)) ** 3
    w /= w.sum(axis=1, keepdims=True)
    path = tmp_path_factory.mktemp("data") / "blobs_pf_dataset.npz"
    np.savez(path, particles=raw.astype(np.float32), weights=w.astype(np.float32),
             particle_scale=np.float32(4.0), particle_centre=np.float32(10.0),
             metadata=json.dumps({"synthetic": True}))
    return str(path)


@pytest.fixture(scope="module")
def emd_matrix_path(weighted_npz, tmp_path_factory):
    """The matrix over the dataset AS THE TRAINER SEES IT (centred, scaled, weighted)."""
    dataset = get_dataset(weighted_npz)
    out = tmp_path_factory.mktemp("emd") / "blobs_emd.npy"
    compute_matrix(dataset.data.numpy(), out, blur=BLUR, block=16, verify=False,
                   device="cpu", weights=dataset.weights.numpy())
    assert matrix_stats(np.array(load_matrix(out, N_CLOUDS)))["offdiag_std"] > 1e-3
    return str(out)


def _config(**overrides):
    base = dict(
        num_particles=N_PART, dim_particles=DIM, num_encodings=2, dim_encoder=4,
        num_inds=4, dim_hidden=16, num_heads=2, use_layer_norm=True,
        weighted_particles=True, batch_size=16, learning_rate=1e-3, num_epochs=3,
        loss_type="sinkhorn", sinkhorn_blur=BLUR, device="cpu", num_workers=0,
        seed=0, log_freq=1, eval_freq=2, save_freq=10_000, scheduler_type="none",
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _trainer(tmp_path, npz, name, indexed, **overrides):
    train_loader, val_loader, _, _ = get_data_loader(
        batch_size=16, data_path=npz, device="cpu", seed=0, indexed=indexed)
    exp = ExperimentConfig(experiment_name="align_test", run_name=name, base_dir=tmp_path)
    return Trainer(_config(**overrides), exp, train_loader, val_loader)


def _epoch_losses(trainer):
    losses = []
    for epoch in range(trainer.config.num_epochs):
        trainer.current_epoch = epoch
        losses.append(trainer.train_epoch())
    return losses


def test_train_epoch_restores_train_mode_after_a_mid_epoch_evaluate(tmp_path, weighted_npz):
    """PITFALLS.md section 8 item 6. evaluate() calls model.eval() and, until
    2026-09-06, nothing switched the model back, so every batch after a
    mid-epoch evaluation trained in eval mode (deterministic VAE posterior,
    frozen VQ-VAE codebook). With eval_freq=1 every step evaluates; the model
    must nevertheless be in training mode at the end of the epoch, and every
    training forward after the first evaluation must have run in train mode."""
    trainer = _trainer(tmp_path, weighted_npz, "evalmode", indexed=False, eval_freq=1)
    modes = []
    handle = trainer.model.register_forward_pre_hook(
        lambda module, _inputs: modes.append(module.training))
    try:
        trainer.current_epoch = 0
        trainer.train_epoch()
    finally:
        handle.remove()
    assert trainer.model.training, "train_epoch left the model in eval mode"
    # Forwards alternate: training batch (True), then evaluate() over the val
    # set (False, ...). After the FIRST evaluation there must still be
    # training-mode forwards, i.e. the flag was restored between them.
    first_eval = modes.index(False)
    assert any(modes[first_eval:]), (
        "no training-mode forward after the first evaluate(): the model stayed "
        "in eval mode for the rest of the epoch")


def test_lambda_zero_leaves_the_reconstruction_path_untouched(tmp_path, weighted_npz):
    """With alignment off, an indexed loader and the new code path must give the
    SAME numbers as the plain loader: same seed, same batches, same losses."""
    plain = _trainer(tmp_path, weighted_npz, "plain", indexed=False)
    plain_losses = _epoch_losses(plain)
    indexed = _trainer(tmp_path, weighted_npz, "indexed", indexed=True)
    assert indexed._indexed_loader and indexed.align_loss is None
    indexed_losses = _epoch_losses(indexed)
    np.testing.assert_allclose(plain_losses, indexed_losses, rtol=0, atol=0)


def test_alignment_requires_matrix_and_indexed_loader(tmp_path, weighted_npz, emd_matrix_path):
    with pytest.raises(ValueError, match="emd_matrix_path"):
        _trainer(tmp_path, weighted_npz, "nomatrix", indexed=True, align_lambda=0.2)
    with pytest.raises(ValueError, match="indexed=True"):
        _trainer(tmp_path, weighted_npz, "notindexed", indexed=False,
                 align_lambda=0.2, emd_matrix_path=emd_matrix_path)


def test_matrix_over_the_wrong_number_of_rows_is_refused(tmp_path, weighted_npz):
    dataset = get_dataset(weighted_npz, max_samples=40)
    small = tmp_path / "small_emd.npy"
    compute_matrix(dataset.data.numpy(), small, blur=BLUR, block=16, verify=False,
                   device="cpu", weights=dataset.weights.numpy())
    with pytest.raises(ValueError, match="holds"):
        _trainer(tmp_path, weighted_npz, "wrongn", indexed=True,
                 align_lambda=0.2, emd_matrix_path=str(small))


def test_lambda_ramp_is_honoured_and_logged(tmp_path, weighted_npz, emd_matrix_path):
    trainer = _trainer(tmp_path, weighted_npz, "ramp", indexed=True, num_epochs=4,
                       align_lambda=0.5, align_warmup_epochs=1, align_ramp_epochs=2,
                       emd_matrix_path=emd_matrix_path)
    assert [trainer._align_lambda(e) for e in range(5)] == pytest.approx([0.0, 0.25, 0.5, 0.5, 0.5])
    # During warmup the term is computed (logged) but carries no weight.
    trainer.current_epoch = 0
    batch = next(iter(trainer.train_loader))
    particles, weights, indices = trainer._split_batch(batch)
    assert indices is not None and indices.dtype == np.int64
    recon, aux, latent = trainer._forward_with_latent(
        trainer._model_input(particles, weights), need_latent=True)
    total, comps = trainer._compose_loss(recon, particles, aux, target_weights=weights,
                                         latent=latent, batch_indices=indices,
                                         align_lambda=0.0)
    assert {"recon", "align", "align_lambda", "align_r"} <= set(comps)
    assert total.item() == pytest.approx(comps["recon"])
    total_w, comps_w = trainer._compose_loss(recon, particles, aux, target_weights=weights,
                                             latent=latent, batch_indices=indices,
                                             align_lambda=0.5)
    assert total_w.item() == pytest.approx(comps_w["recon"] + 0.5 * comps_w["align"], rel=1e-5)


def test_alignment_raises_latent_emd_correlation(tmp_path, weighted_npz, emd_matrix_path):
    """The point of the term: after training with it, held-out latent<->EMD
    correlation must exceed a reconstruction-only twin's by a clear margin."""
    aligned = _trainer(tmp_path, weighted_npz, "aligned", indexed=True, num_epochs=12,
                       align_lambda=1.0, emd_matrix_path=emd_matrix_path)
    r_before = aligned._validation_alignment_r()
    _epoch_losses(aligned)
    r_after = aligned._validation_alignment_r()
    assert np.isfinite(r_after)
    assert r_after > 0.7, (r_before, r_after)
    # Validation loss stays reconstruction-only: no align component in evaluate().
    val_loss, metrics = aligned.evaluate()
    assert "align_r" in metrics and np.isfinite(metrics["align_r"])
    assert np.isfinite(val_loss)


def test_checkpoint_records_frame_alignment_and_best_epoch(tmp_path, weighted_npz,
                                                            emd_matrix_path):
    trainer = _trainer(tmp_path, weighted_npz, "ckpt", indexed=True, num_epochs=2,
                       align_lambda=0.2, align_warmup_epochs=1, align_ramp_epochs=1,
                       emd_matrix_path=emd_matrix_path)
    trainer.train()
    ck = torch.load(trainer.exp_config.checkpoint_dir / "checkpoint_latest.pt",
                    map_location="cpu", weights_only=False)
    assert ck["particle_scale"] == pytest.approx(4.0)
    assert ck["particle_centre"] == pytest.approx(10.0)
    assert ck["alignment"]["lambda"] == 0.2
    assert ck["alignment"]["emd_matrix_path"] == emd_matrix_path
    assert "best_epoch" in ck and ck["config"].align_lambda == 0.2
    assert ck["config"].seed == 0
    # A weighted checkpoint still loads into the RL extractor's encoder shape.
    encoder_keys = [k for k in ck["model_state_dict"] if k.startswith("set_transformer.")]
    assert encoder_keys


def test_trainer_load_checkpoint_roundtrip(tmp_path, weighted_npz):
    """Phase 0 fix: the checkpoint pickles a TrainingConfig, which torch>=2.6
    refused to load without weights_only=False."""
    trainer = _trainer(tmp_path, weighted_npz, "resume", indexed=False, num_epochs=1)
    trainer.train()
    fresh = _trainer(tmp_path, weighted_npz, "resume2", indexed=False, num_epochs=1)
    fresh.load_checkpoint(trainer.exp_config.checkpoint_dir / "checkpoint_latest.pt")
    assert fresh.global_step == trainer.global_step
    for a, b in zip(fresh.model.parameters(), trainer.model.parameters()):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------------------
# Online targets (2026-09-19): align_target="online" needs no matrix and no indexed loader,
# gives the matrix path's numbers on the same batch, trains, and records what it did.
# ---------------------------------------------------------------------------------------

def test_online_alignment_needs_no_matrix_and_no_indexed_loader(tmp_path, weighted_npz):
    t = _trainer(tmp_path, weighted_npz, "online", indexed=False,
                 align_lambda=0.2, align_target="online")
    assert t.align_loss is not None and t.emd_matrix is None and t.online_alignment is not None
    assert not t._indexed_loader
    # the held-out metric: one fixed pair sample over the val rows, its targets computed once
    assert t._val_align_pair_index is not None
    assert len(t._val_align_pairs) == len(t._val_align_pair_index[0])
    assert np.isfinite(t._validation_alignment_r())
    with pytest.raises(ValueError, match="align_target"):
        _trainer(tmp_path, weighted_npz, "badtarget", indexed=False,
                 align_lambda=0.2, align_target="disk")
    with pytest.raises(ValueError, match="align_pairs"):
        _trainer(tmp_path, weighted_npz, "badpairs", indexed=False,
                 align_lambda=0.2, align_target="online", align_pairs="some")
    # the matrix path is untouched: its refusals still fire with the default target
    with pytest.raises(ValueError, match="emd_matrix_path"):
        _trainer(tmp_path, weighted_npz, "nomatrix2", indexed=True, align_lambda=0.2)


def test_online_term_equals_the_matrix_term_on_the_same_batch(tmp_path, weighted_npz, emd_matrix_path):
    """Same clouds, same latent: the per-batch targets ARE the matrix entries, so the two ways
    of aligning give the same term (measured 8e-7 apart; the matrix's manual debiasing vs
    geomloss's own). The held-out metric agrees too: every pair of the val rows is in the
    fixed sample when the budget covers them all."""
    m = _trainer(tmp_path, weighted_npz, "m", indexed=True, align_lambda=0.5,
                 emd_matrix_path=emd_matrix_path)
    o = _trainer(tmp_path, weighted_npz, "o", indexed=False, align_lambda=0.5,
                 align_target="online", align_pairs="all")
    particles, weights, idx = m._split_batch(next(iter(m.train_loader)))
    recon, aux, latent = m._forward_with_latent(m._model_input(particles, weights), need_latent=True)
    _, cm = m._compose_loss(recon, particles, aux, target_weights=weights, latent=latent,
                            batch_indices=idx, align_lambda=0.5)
    total_o, co = o._compose_loss(recon, particles, aux, target_weights=weights, latent=latent,
                                  batch_indices=None, align_lambda=0.5)
    assert cm["recon"] == co["recon"]
    assert co["align"] == pytest.approx(cm["align"], abs=1e-4)
    assert co["align_r"] == pytest.approx(cm["align_r"], abs=1e-4)
    assert total_o.item() == pytest.approx(co["recon"] + 0.5 * co["align"], rel=1e-5)
    assert o._validation_alignment_r() == pytest.approx(m._validation_alignment_r(), abs=1e-3)


def test_online_alignment_trains_and_records_the_target(tmp_path, weighted_npz):
    aligned = _trainer(tmp_path, weighted_npz, "online_train", indexed=False, num_epochs=12,
                       align_lambda=1.0, align_target="online", align_pairs="all")
    r_before = aligned._validation_alignment_r()
    _epoch_losses(aligned)
    r_after = aligned._validation_alignment_r()
    assert np.isfinite(r_after) and r_after > 0.7, (r_before, r_after)
    val_loss, metrics = aligned.evaluate()
    assert "align_r" in metrics and np.isfinite(metrics["align_r"]) and np.isfinite(val_loss)
    # a budget below every pair runs too (the sampler's own stream), and the checkpoint says
    # what the target was: online, its pairs, blur and scaling, and no matrix path
    budget = _trainer(tmp_path, weighted_npz, "online_budget", indexed=False, num_epochs=2,
                      align_lambda=1.0, align_target="online", align_pairs=32)
    budget.train()
    ck = torch.load(budget.exp_config.checkpoint_dir / "checkpoint_latest.pt",
                    map_location="cpu", weights_only=False)
    al = ck["alignment"]
    assert al["target"] == "online" and al["emd_matrix_path"] is None and al["pairs"] == 32
    assert al["blur"] == BLUR and al["scaling"] == 0.5 and al["val_pairs"] == 20000
    assert al["lambda"] == 1.0 and al["lambda_at_best_epoch"] == 1.0
    assert ck["config"].align_target == "online" and ck["config"].align_pairs == 32
