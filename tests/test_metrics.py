"""Tests for AMICA component metrics."""

from __future__ import annotations

import numpy as np

from jamica import metrics


class MockAmicaResult:
    """Mock result object for fast metric testing."""

    def __init__(self, n_comp=4, n_mix=3, n_models=1):
        if n_models == 1:
            self.alpha_ = np.ones((n_mix, n_comp)) / n_mix
            self.rho_ = np.full((n_mix, n_comp), 1.5)
            self.unmixing_matrix_white_ = np.eye(n_comp)
        else:
            self.alpha_ = np.ones((n_models, n_mix, n_comp)) / n_mix
            self.rho_ = np.full((n_models, n_mix, n_comp), 1.5)
            self.unmixing_matrix_white_ = np.array([np.eye(n_comp) for _ in range(n_models)])

        self.mean_ = np.zeros(n_comp)
        self.whitener_ = np.eye(n_comp)
        self.data_scale = 1.0


def test_rho_mean():
    # Single model
    res1 = MockAmicaResult(n_models=1)
    res1.rho_[0, :] = 1.0
    res1.rho_[1, :] = 2.0
    res1.rho_[2, :] = 1.5
    # alpha is uniform (1/3)
    rm1 = metrics.rho_mean(res1)
    assert rm1.shape == (4,)
    np.testing.assert_allclose(rm1, 1.5)

    # Multi model (should just take first model)
    res2 = MockAmicaResult(n_models=2)
    res2.rho_[0, 0, :] = 1.0
    res2.rho_[0, 1, :] = 2.0
    res2.rho_[0, 2, :] = 1.5
    rm2 = metrics.rho_mean(res2)
    assert rm2.shape == (4,)
    np.testing.assert_allclose(rm2, 1.5)


def test_rho_range():
    res1 = MockAmicaResult(n_models=1)
    res1.rho_[0, :] = 1.0
    res1.rho_[1, :] = 2.0
    res1.rho_[2, :] = 1.5
    rr1 = metrics.rho_range(res1)
    assert rr1.shape == (4,)
    np.testing.assert_allclose(rr1, 1.0)

    res2 = MockAmicaResult(n_models=2)
    res2.rho_[0, 0, :] = 1.0
    res2.rho_[0, 1, :] = 2.0
    res2.rho_[0, 2, :] = 1.5
    rr2 = metrics.rho_range(res2)
    assert rr2.shape == (4,)
    np.testing.assert_allclose(rr2, 1.0)


def test_mixture_entropy():
    # Uniform
    res1 = MockAmicaResult(n_comp=2, n_mix=3, n_models=1)
    ent1 = metrics.mixture_entropy(res1)
    assert ent1.shape == (2,)
    np.testing.assert_allclose(ent1, np.log(3))

    # Multi-model
    res2 = MockAmicaResult(n_comp=2, n_mix=3, n_models=2)
    ent2 = metrics.mixture_entropy(res2)
    assert ent2.shape == (2,)
    np.testing.assert_allclose(ent2, np.log(3))

    # Single mix (should be 0)
    res3 = MockAmicaResult(n_comp=2, n_mix=1, n_models=1)
    ent3 = metrics.mixture_entropy(res3)
    assert ent3.shape == (2,)
    np.testing.assert_allclose(ent3, 0.0, atol=1e-7)


def test_multimodality_flag():
    # Uniform 3-mix -> high entropy -> flag True
    res1 = MockAmicaResult(n_comp=2, n_mix=3, n_models=1)
    flags1 = metrics.multimodality_flag(res1)
    assert flags1.shape == (2,)
    assert np.all(flags1)

    # Multi-model uniform
    res2 = MockAmicaResult(n_comp=2, n_mix=3, n_models=2)
    flags2 = metrics.multimodality_flag(res2)
    assert flags2.shape == (2,)
    assert np.all(flags2)

    # 1-mix -> 0 entropy -> flag False
    res3 = MockAmicaResult(n_comp=2, n_mix=1, n_models=1)
    flags3 = metrics.multimodality_flag(res3)
    assert flags3.shape == (2,)
    assert np.all(~flags3)


def test_source_kurtosis():
    res = MockAmicaResult(n_comp=4, n_mix=1, n_models=1)
    rng = np.random.RandomState(42)

    # Create Laplacian data (positive excess kurtosis)
    data = rng.laplace(size=(4, 1000))
    kurt = metrics.source_kurtosis(res, data)
    assert kurt.shape == (4,)
    assert np.all(kurt > 0)
