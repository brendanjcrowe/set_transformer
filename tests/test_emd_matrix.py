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


# ---------------------------------------------------------------------------
# Weighted measures (Phase 2 of the alignment integration)
# ---------------------------------------------------------------------------

from geomloss import SamplesLoss  # noqa: E402

from set_transformer.emd_matrix import (  # noqa: E402
    _prepare_weights,
    _self_terms,
    n_from_size,
    read_sidecar,
    verify_against_geomloss,
    write_sidecar,
)

BLUR_W = 0.05


@pytest.fixture
def wclouds():
    """12 small 2-D clouds with strongly non-uniform PF-like weights."""
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(12, 16, 2)).astype(np.float32)
    w = rng.random((12, 16)).astype(np.float32) ** 4  # a few particles carry most mass
    w /= w.sum(axis=1, keepdims=True)
    return pts, w


def _matrix(tmp_path, name, pts, weights, **kw):
    return np.array(compute_matrix(pts, tmp_path / name, blur=BLUR_W, block=5,
                                   verify=False, device="cpu", weights=weights, **kw))


def test_uniform_weights_reproduce_the_unweighted_matrix(tmp_path, wclouds):
    """Explicit 1/N masses are the same measure as no weights, so the matrix must
    agree to float precision (the two go through different geomloss code paths,
    hence allclose rather than bit equality)."""
    pts, _ = wclouds
    uniform = np.full(pts.shape[:2], 1.0 / pts.shape[1], dtype=np.float32)
    a = _matrix(tmp_path, "none.npy", pts, None)
    b = _matrix(tmp_path, "uniform.npy", pts, uniform)
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)


def test_weighted_manual_debiasing_matches_geomloss(wclouds):
    """The self-term trick must hold for weighted measures too: each cloud's
    OT_eps(alpha, alpha) is computed under its OWN weights."""
    pts, w = wclouds
    t = torch.from_numpy(pts)
    tw = _prepare_weights(t, torch.from_numpy(w))
    ot = SamplesLoss("sinkhorn", p=2, blur=BLUR_W, debias=False)
    self_terms = _self_terms(t, ot, chunk=4, weights=tw)
    err = verify_against_geomloss(t, ot, self_terms, BLUR_W, 2, n_pairs=32, weights=tw)
    assert err < 1e-3, err


def test_weighted_matrix_differs_from_unweighted_when_weights_are_not_uniform(tmp_path, wclouds):
    pts, w = wclouds
    a = _matrix(tmp_path, "none.npy", pts, None)
    b = _matrix(tmp_path, "weighted.npy", pts, w)
    assert np.abs(a - b).max() > 1e-2
    # Still a proper divergence matrix.
    np.testing.assert_allclose(b, b.T, atol=1e-6)
    assert np.abs(np.diag(b)).max() == 0.0
    assert (b[np.triu_indices(len(b), 1)] > -1e-4).all()


def test_identical_support_separates_only_through_the_weights(tmp_path):
    """The Odd-Even exact-support case: every cloud is the SAME 1-D grid of
    states and only the weights differ. Unweighted, the matrix is exactly
    zero (nothing to align against); weighted, two clouds with the same
    weights sit at 0 and two with different weights are strictly apart, and
    mass moved further along the grid costs more."""
    grid = np.arange(1, 21, dtype=np.float32)
    pts = np.repeat(((grid - 10.5) / 9.5)[None, :, None], 5, axis=0)  # 5 identical clouds
    w = np.full((5, 20), 1e-6, dtype=np.float32)
    w[0, 4] = 1.0   # mass on state 5
    w[1, 4] = 1.0   # same as cloud 0
    w[2, 6] = 1.0   # state 7: near
    w[3, 14] = 1.0  # state 15: far
    w[4, 4] = 0.5; w[4, 6] = 0.5  # split between 5 and 7
    w /= w.sum(axis=1, keepdims=True)

    unweighted = _matrix(tmp_path, "u.npy", pts, None)
    assert np.abs(unweighted).max() < 1e-6, "identical supports must give a zero matrix"

    weighted = _matrix(tmp_path, "w.npy", pts, w)
    assert abs(weighted[0, 1]) < 1e-5, "same weights, same support -> 0"
    assert weighted[0, 2] > 1e-3, "different weights on the same support -> > 0"
    assert weighted[0, 3] > weighted[0, 2], "mass moved further costs more"
    assert weighted[0, 2] > weighted[0, 4] > 0, "a half-split sits between"
    stats = matrix_stats(weighted)
    assert stats["offdiag_std"] > 1e-3


def test_reweighting_one_cloud_changes_only_its_row_and_column(tmp_path, wclouds):
    pts, w = wclouds
    base = _matrix(tmp_path, "base.npy", pts, w)
    w2 = w.copy()
    w2[3] = np.roll(w2[3], 5)
    changed = _matrix(tmp_path, "changed.npy", pts, w2)
    diff = np.abs(base - changed)
    mask = np.zeros_like(diff, dtype=bool)
    mask[3, :] = True
    mask[:, 3] = True
    assert diff[~mask].max() < 1e-6
    assert diff[mask].max() > 1e-3


def test_invalid_weights_are_refused(wclouds, tmp_path):
    pts, w = wclouds
    bad = w.copy()
    bad[2] = 0.0
    with pytest.raises(ValueError, match="sum to zero"):
        _matrix(tmp_path, "bad.npy", pts, bad)
    with pytest.raises(ValueError, match="shape"):
        _matrix(tmp_path, "bad2.npy", pts, w[:, :5])


def test_load_matrix_refuses_a_size_mismatch(tmp_path, wclouds):
    pts, w = wclouds
    path = tmp_path / "m.npy"
    _matrix(tmp_path, "m.npy", pts, w)
    load_matrix(path, len(pts))  # exact size loads
    with pytest.raises(ValueError, match="holds"):
        load_matrix(path, len(pts) - 1)  # a SMALLER n would otherwise mmap silently
    with pytest.raises(ValueError):
        load_matrix(path, len(pts) + 1)
    assert n_from_size(4 * 7 * 7) == 7 and n_from_size(4 * 7 * 7 + 4) is None


def test_sidecar_records_weightedness_and_frame(tmp_path):
    out = tmp_path / "m.npy"
    write_sidecar(out, "data.npz", 12, BLUR_W, 2, {"offdiag_std": 0.3}, weighted=True,
                  scaling=0.7, extra={"particle_scale": 24.5, "particle_centre": 25.5,
                                      "data_sha256": "abc"})
    side = read_sidecar(out)
    assert side["weighted"] is True and side["scaling"] == 0.7 and side["blur"] == BLUR_W
    assert side["particle_centre"] == 25.5 and side["data_sha256"] == "abc"
    with pytest.raises(FileNotFoundError):
        read_sidecar(tmp_path / "missing.npy")
