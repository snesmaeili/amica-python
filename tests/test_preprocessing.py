"""Direct tests for jamica.preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

RNG = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cov(data):
    from jamica.preprocessing import compute_covariance

    return np.asarray(compute_covariance(data, data.mean(axis=1)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_mean():
    from jamica.preprocessing import compute_mean

    # Basic correct mean
    data = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    np.testing.assert_allclose(np.asarray(compute_mean(data)), [3.0, 4.0])

    # Shapes and zero-mean check
    data_rand = RNG.randn(8, 1000)
    assert np.asarray(compute_mean(data_rand)).shape == (8,)

    data_zero = RNG.randn(4, 500)
    data_zero -= data_zero.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(np.asarray(compute_mean(data_zero)), np.zeros(4), atol=1e-12)


def test_compute_covariance():
    from jamica.preprocessing import compute_covariance

    data = RNG.randn(6, 500)
    cov = np.asarray(compute_covariance(data, data.mean(axis=1)))

    # Shape and symmetry
    assert cov.shape == (6, 6)
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    # White data near identity
    rng = np.random.RandomState(0)
    white_data = rng.randn(4, 100_000)
    white_cov = np.asarray(compute_covariance(white_data, white_data.mean(axis=1)))
    np.testing.assert_allclose(np.diag(white_cov), np.ones(4), atol=0.05)


def test_compute_sphering_matrix_standard():
    from jamica.preprocessing import compute_dewhitening_matrix, compute_sphering_matrix

    rng = np.random.RandomState(1)
    data = rng.randn(4, 5000)
    data[0] *= 3.0
    cov = _cov(data)

    # Test default (ZCA)
    sphere, eigs, n_comp = compute_sphering_matrix(cov)
    assert n_comp == 4

    # Check whitening property
    sphere_np = np.asarray(sphere)
    cov_white = sphere_np @ cov @ sphere_np.T
    np.testing.assert_allclose(np.diag(cov_white), np.ones(4), atol=0.05)

    # Check dewhitening (left inverse)
    desphere = np.asarray(compute_dewhitening_matrix(sphere, eigs, n_comp))
    np.testing.assert_allclose(sphere_np @ desphere, np.eye(4), atol=1e-8)

    # Test PCA type and reduction
    sphere_pca, _eigs_pca, n_comp_pca = compute_sphering_matrix(cov, sphere_type="pca", pcakeep=2)
    assert n_comp_pca == 2
    assert np.asarray(sphere_pca).shape == (2, 4)


def test_compute_sphering_matrix_exceptions_and_edge_cases(monkeypatch):
    import scipy.linalg as sla

    from jamica.preprocessing import compute_sphering_matrix

    # Near rank 2 dataset -> mineig filtering
    rng = np.random.RandomState(7)
    data = rng.randn(4, 2000)
    data[2:] = data[:2] + 1e-4 * rng.randn(2, 2000)
    cov = _cov(data)
    _, _, n_comp = compute_sphering_matrix(cov, mineig=0.01)
    assert n_comp < 4

    # Flat channel
    data_flat = RNG.randn(4, 2000)
    data_flat[0] = 0.0
    cov_flat = _cov(data_flat)
    with pytest.raises(ValueError, match="zero or near-zero variance channels"):
        compute_sphering_matrix(cov_flat)

    # No eigenvalues above threshold
    data_small = RNG.randn(4, 2000)
    cov_small = _cov(data_small)
    with pytest.raises(ValueError, match="No eigenvalues above threshold"):
        compute_sphering_matrix(cov_small, pcakeep=0)

    # LinAlgError fallback to SVD
    def fake_eigh(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(sla, "eigh", fake_eigh)

    sphere_fallback, _, n_comp_fallback = compute_sphering_matrix(cov_small)
    assert n_comp_fallback == 4
    assert np.asarray(sphere_fallback).shape == (4, 4)


def test_preprocess_data_flags():
    from jamica.preprocessing import preprocess_data

    rng = np.random.RandomState(3)
    data = rng.randn(4, 10_000)

    # Standard run
    white, mean, _sphere, _desphere, _n_comp, eigs = preprocess_data(data)
    assert np.asarray(white).shape == (4, 10_000)
    assert np.asarray(mean).shape == (4,)
    assert np.all(np.asarray(eigs) > 0)

    # Check whitening via covariance
    white_np = np.asarray(white)
    white_cov = white_np @ white_np.T / white_np.shape[1]
    np.testing.assert_allclose(np.diag(white_cov), np.ones(4), atol=0.05)

    # No mean removal
    data_shifted = RNG.randn(4, 1000) + 10.0
    _, mean_out, *_ = preprocess_data(data_shifted, do_mean=False)
    np.testing.assert_allclose(np.asarray(mean_out), np.zeros(4), atol=1e-10)

    # Overrides
    forced_mean = np.ones(4)
    forced_sphere = np.eye(4) * 2.0

    _, mean_over, sphere_over, _, _n_comp_over, _ = preprocess_data(
        data_shifted, init_mean=forced_mean, init_sphere=forced_sphere
    )
    np.testing.assert_allclose(np.asarray(mean_over), forced_mean, atol=1e-10)
    np.testing.assert_allclose(np.asarray(sphere_over), forced_sphere, atol=1e-10)

    # No sphering
    _white_nos, _, sphere_nos, _, _n_comp_nos, eigs_nos = preprocess_data(
        data_shifted, do_sphere=False
    )
    np.testing.assert_allclose(np.asarray(sphere_nos), np.eye(4))
    np.testing.assert_allclose(np.asarray(eigs_nos), np.ones(4))

    # No sphering with pcakeep
    _, _, sphere_nosp_pca, _, n_comp_nosp_pca, _ = preprocess_data(
        data_shifted, do_sphere=False, pcakeep=2
    )
    assert n_comp_nosp_pca == 2
    assert np.asarray(sphere_nosp_pca).shape == (2, 4)

    # init_sphere override but do_sphere=False
    _, _, sphere_force_nos, _, _, eigs_force_nos = preprocess_data(
        data_shifted, init_sphere=forced_sphere, do_sphere=False
    )
    np.testing.assert_allclose(np.asarray(sphere_force_nos), forced_sphere, atol=1e-10)
    np.testing.assert_allclose(np.asarray(eigs_force_nos), np.ones(4))


def test_apply_sphering_and_dewhitening():
    from jamica.preprocessing import apply_sphering, compute_dewhitening_matrix

    data = RNG.randn(4, 1000)
    mean = data.mean(axis=1)
    sphere = np.eye(4) * 2

    # apply_sphering
    res = apply_sphering(data, mean, sphere)
    assert np.asarray(res).shape == (4, 1000)

    # compute_dewhitening_matrix
    sphere_diag = np.array([[2.0, 0.0], [0.0, 3.0]])
    desphere = compute_dewhitening_matrix(sphere_diag, np.array([4.0, 9.0]), 2)
    np.testing.assert_allclose(np.asarray(desphere), [[0.5, 0.0], [0.0, 1 / 3]])
