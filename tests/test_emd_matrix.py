"""Tests for the shared pairwise-EMD matrix builder.

The matrix is the reference geometry the alignment loss trains against, so the properties
that matter are: it is a proper divergence (zero diagonal, symmetric), the manual
debiasing agrees with geomloss's own, and an interrupted run resumes to the same answer.
"""

import numpy as np
import pytest
import torch

from set_transformer.emd_matrix import (
    DEFAULT_BLUR,
    compute_matrix,
    load_matrix,
    matrix_stats,
    write_sidecar,
)


@pytest.fixture
def clouds():
    rng = np.random.default_rng(0)
    # Three well-separated blobs, so the pairwise structure is unambiguous.
    centers = rng.uniform(-3, 3, size=(24, 1, 2))
    return (centers + 0.15 * rng.standard_normal((24, 40, 2))).astype(np.float32)


def test_matrix_is_symmetric_with_zero_diagonal(tmp_path, clouds):
    out = tmp_path / "emd.npy"
    m = np.asarray(compute_matrix(clouds, out, block=8, verify=False, device="cpu"))
    assert m.shape == (len(clouds), len(clouds))
    assert np.allclose(m, m.T, atol=1e-6)
    assert np.abs(np.diag(m)).max() == 0.0
    # A debiased divergence: distinct clouds are strictly farther apart than a self-pair.
    assert m[np.triu_indices(len(clouds), k=1)].min() > 0.0


def test_manual_debiasing_matches_geomloss(tmp_path, clouds):
    from geomloss import SamplesLoss

    out = tmp_path / "emd.npy"
    m = np.asarray(compute_matrix(clouds, out, block=8, verify=False, device="cpu"))
    ref = SamplesLoss("sinkhorn", p=2, blur=DEFAULT_BLUR, debias=True)
    pts = torch.from_numpy(clouds)
    a = torch.tensor([0, 1, 2, 3, 4])
    b = torch.tensor([5, 6, 7, 8, 9])
    with torch.no_grad():
        expected = ref(pts[a], pts[b]).numpy()
    assert np.abs(m[a.numpy(), b.numpy()] - expected).max() < 1e-3


def test_resume_reproduces_a_full_run(tmp_path, clouds):
    """Interrupting a long precompute must not silently corrupt the matrix."""
    import json

    full = np.array(compute_matrix(clouds, tmp_path / "full.npy", block=8, verify=False,
                                   device="cpu"))

    partial_path = tmp_path / "partial.npy"
    compute_matrix(clouds, partial_path, block=8, verify=False, device="cpu")
    # Wipe everything the last row block owns and rewind the progress marker.
    m = np.memmap(partial_path, dtype=np.float32, mode="r+", shape=full.shape)
    m[16:, 16:] = -99.0
    m.flush()
    (partial_path.with_suffix(".progress.json")).write_text(
        json.dumps({"n": len(clouds), "block": 8, "next_row_block": 2,
                    "blur": DEFAULT_BLUR, "p": 2}))

    resumed = np.array(compute_matrix(clouds, partial_path, block=8, verify=False,
                                      device="cpu", resume=True))
    assert np.array_equal(resumed, full)


def test_load_matrix_roundtrip_and_stats(tmp_path, clouds):
    out = tmp_path / "emd.npy"
    compute_matrix(clouds, out, block=8, verify=False, device="cpu")
    loaded = load_matrix(out, len(clouds))
    assert loaded.shape == (len(clouds), len(clouds))

    stats = matrix_stats(loaded)
    assert stats["diag_absmax"] == 0.0
    assert stats["offdiag_std"] > 0  # spread, not a collapsed band
    assert stats["offdiag_min"] <= stats["offdiag_mean"] <= stats["offdiag_max"]

    write_sidecar(out, "src.npy", len(clouds), DEFAULT_BLUR, 2, stats)
    import json
    side = json.loads(out.with_suffix(".json").read_text())
    assert side["metric"] == "debiased_sinkhorn"
    assert side["blur"] == DEFAULT_BLUR
    assert side["shape"] == [len(clouds), len(clouds)]
