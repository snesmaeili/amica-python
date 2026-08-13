"""Tests for jamica.pdf module."""

from __future__ import annotations

import numpy as np

from jamica import pdf

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_inputs(n_comp=4, n_mix=3, n_samp=1000):
    rng = np.random.RandomState(42)
    y = rng.randn(n_comp, n_samp)
    alpha = np.ones((n_mix, n_comp)) / n_mix
    mu = rng.randn(n_mix, n_comp) * 0.1
    beta = np.ones((n_mix, n_comp)) + 0.5
    rho = np.full((n_mix, n_comp), 1.5)
    return y, alpha, mu, beta, rho


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_log_generalized_gaussian():
    from jamica.backend import jnp

    rng = np.random.RandomState(42)
    y = rng.randn(500)
    mu = 0.5
    beta = 1.2

    # Laplacian (rho=1)
    rho_lap = 1.0
    ll_lap = pdf.log_generalized_gaussian(jnp.asarray(y), mu, beta, rho_lap)
    assert np.asarray(ll_lap).shape == (500,)

    # Manual check for Laplace: log(beta) - beta*|y-mu| - log(2)
    y_scaled = beta * (y - mu)
    expected_lap = np.log(beta) - np.abs(y_scaled) - np.log(2.0)
    np.testing.assert_allclose(np.asarray(ll_lap), expected_lap, atol=1e-6)

    # Gaussian (rho=2)
    rho_gau = 2.0
    ll_gau = pdf.log_generalized_gaussian(jnp.asarray(y), mu, beta, rho_gau)
    assert np.asarray(ll_gau).shape == (500,)

    # Manual check for Gaussian: log(beta) - y_scaled^2 - log(sqrt(pi))
    expected_gau = np.log(beta) - (y_scaled**2) - np.log(np.sqrt(np.pi))
    # NOTE: The generalized formulation is gamln(1 + 1/rho) + log(2).
    # gamln(1.5) = log(sqrt(pi)/2). Adding log(2) gives log(sqrt(pi)).
    np.testing.assert_allclose(np.asarray(ll_gau), expected_gau, atol=1e-6)


def test_log_generalized_gaussian_mixture():
    from jamica.backend import jnp

    rng = np.random.RandomState(42)
    y = rng.randn(500)
    n_mix = 3
    alpha = np.array([0.2, 0.5, 0.3])
    mu = rng.randn(n_mix)
    beta = rng.rand(n_mix) + 0.5
    rho = np.full(n_mix, 1.5)

    ll_mix = pdf.log_generalized_gaussian_mixture(
        jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho)
    )

    assert np.asarray(ll_mix).shape == (500,)
    assert np.all(np.isfinite(np.asarray(ll_mix)))


def test_compute_responsibilities():
    from jamica.backend import jnp

    rng = np.random.RandomState(42)
    y = rng.randn(500)
    n_mix = 3
    alpha = np.array([0.2, 0.5, 0.3])
    mu = rng.randn(n_mix)
    beta = rng.rand(n_mix) + 0.5
    rho = np.full(n_mix, 1.5)

    resp = pdf.compute_responsibilities(
        jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho)
    )

    assert np.asarray(resp).shape == (n_mix, 500)

    # Check sum to 1 over mixtures
    np.testing.assert_allclose(np.asarray(resp).sum(axis=0), 1.0, atol=1e-6)

    # Ensure they are bounded > 0
    assert np.all(np.asarray(resp) > 0)


def test_score_functions():
    from jamica.backend import jnp

    rng = np.random.RandomState(42)
    y = rng.randn(500)
    n_mix = 3
    alpha = np.array([0.2, 0.5, 0.3])
    mu = rng.randn(n_mix)
    beta = rng.rand(n_mix) + 0.5
    rho = np.full(n_mix, 1.5)

    # single score function
    fp = pdf.compute_score_function(jnp.asarray(y), mu[0], beta[0], rho[0])
    assert np.asarray(fp).shape == (500,)
    assert np.all(np.isfinite(np.asarray(fp)))

    # weighted score
    resp = pdf.compute_responsibilities(
        jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho)
    )
    g_weighted = pdf.compute_weighted_score(
        jnp.asarray(y), jnp.asarray(resp), jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho)
    )
    assert np.asarray(g_weighted).shape == (500,)
    assert np.all(np.isfinite(np.asarray(g_weighted)))


def test_compute_all_scores_and_likelihood():
    from jamica.backend import jnp

    n_comp, n_mix, n_samp = 4, 3, 500
    y, alpha, mu, beta, rho = _make_mock_inputs(n_comp, n_mix, n_samp)

    # all scores
    g_all = pdf.compute_all_scores(
        jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho)
    )
    assert np.asarray(g_all).shape == (n_comp, n_samp)
    assert np.all(np.isfinite(np.asarray(g_all)))

    # source log likelihood
    ll_all = pdf.compute_source_loglikelihood(
        jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho)
    )
    assert np.asarray(ll_all).shape == (n_samp,)
    assert np.all(np.isfinite(np.asarray(ll_all)))
