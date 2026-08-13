"""Tests for jamica.likelihood module."""

from __future__ import annotations

import numpy as np

RNG = np.random.RandomState(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(n_comp=4, n_mix=3, n_samp=1000, rng=None):
    """Generate valid mock AMICA parameters for testing."""
    if rng is None:
        rng = RNG
    y = rng.randn(n_comp, n_samp)
    alpha = np.ones((n_mix, n_comp)) / n_mix
    mu = rng.randn(n_mix, n_comp) * 0.1
    beta = np.ones((n_mix, n_comp)) + 0.05 * rng.randn(n_mix, n_comp)
    beta = np.abs(beta) + 0.5
    rho = np.full((n_mix, n_comp), 1.5)
    W = np.eye(n_comp) + 0.01 * rng.randn(n_comp, n_comp)
    return y, W, alpha, mu, beta, rho


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_compute_log_det_W():
    from jamica.backend import jnp
    from jamica.likelihood import compute_log_det_W

    # Identity
    W_id = jnp.eye(4)
    assert abs(float(compute_log_det_W(W_id))) < 1e-10

    # Diagonal
    d = np.array([2.0, 3.0, 4.0])
    W_diag = jnp.diag(jnp.asarray(d))
    expected_diag = np.sum(np.log(d))
    assert abs(float(compute_log_det_W(W_diag)) - expected_diag) < 1e-6

    # General random matrix
    rng = np.random.RandomState(1)
    W_rand = rng.randn(5, 5)
    result = float(compute_log_det_W(jnp.asarray(W_rand)))
    expected_rand = np.log(abs(np.linalg.det(W_rand)))
    assert abs(result - expected_rand) < 1e-5


def test_model_and_average_loglikelihood():
    from jamica.backend import jnp
    from jamica.likelihood import (
        compute_average_loglikelihood,
        compute_log_det_W,
        compute_model_loglikelihood,
    )

    n_comp, n_samp = 4, 500
    y, W, alpha, mu, beta, rho = _make_params(n_comp=n_comp, n_samp=n_samp)
    log_det_W = compute_log_det_W(jnp.asarray(W))

    # Compute per-sample model log-likelihood
    sample_ll = compute_model_loglikelihood(
        jnp.asarray(y),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        log_det_W=log_det_W,
        log_det_sphere=0.0,
    )

    # Verify shape and contents
    assert np.asarray(sample_ll).shape == (n_samp,)
    assert np.all(np.isfinite(np.asarray(sample_ll)))

    # Compute average
    avg_ll = compute_average_loglikelihood(sample_ll, n_components=n_comp)
    expected_avg = np.mean(np.asarray(sample_ll)) / n_comp
    assert abs(float(avg_ll) - expected_avg) < 1e-8


def test_total_loglikelihood_and_chunks():
    from jamica.backend import jnp
    from jamica.likelihood import compute_loglik_chunk, compute_total_loglikelihood

    rng = np.random.RandomState(10)
    n_comp, n_samp = 4, 10_000
    y, W, alpha, mu, beta, rho = _make_params(n_comp=n_comp, n_samp=n_samp, rng=rng)

    # Total LL
    args = (
        jnp.asarray(y),
        jnp.asarray(W),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
    )

    ll_full_0 = float(compute_total_loglikelihood(*args, log_det_sphere=0.0))
    ll_full_1 = float(compute_total_loglikelihood(*args, log_det_sphere=1.0))

    assert np.isfinite(ll_full_0)
    assert ll_full_0 < 0
    assert abs((ll_full_1 - ll_full_0) - (1.0 / n_comp)) < 1e-6

    # Test chunk additivity
    ll_h1, n1 = compute_loglik_chunk(
        jnp.asarray(y[:, :5000]),
        jnp.asarray(W),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        log_det_sphere=0.3,
    )
    ll_h2, n2 = compute_loglik_chunk(
        jnp.asarray(y[:, 5000:]),
        jnp.asarray(W),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        log_det_sphere=0.3,
    )
    assert float(n1) == 5000
    assert float(n2) == 5000

    ll_merged = float((ll_h1 + ll_h2) / (n1 + n2) / n_comp)
    ll_full_chunked = float(compute_total_loglikelihood(*args, log_det_sphere=0.3))

    rel_err = abs(ll_full_chunked - ll_merged) / max(abs(ll_full_chunked), 1e-20)
    assert rel_err < 1e-10


def test_multimodel_loglikelihood():
    from jamica.backend import jnp
    from jamica.likelihood import compute_multimodel_loglikelihood

    n_models = 2
    n_comp = 4
    n_mix = 3
    n_samp = 500

    rng = np.random.RandomState(42)
    y_all = rng.randn(n_models, n_comp, n_samp)
    W_all = np.array([np.eye(n_comp) for _ in range(n_models)])
    alpha_all = np.ones((n_models, n_mix, n_comp)) / n_mix
    mu_all = rng.randn(n_models, n_mix, n_comp) * 0.1
    beta_all = np.ones((n_models, n_mix, n_comp)) + 0.5
    rho_all = np.full((n_models, n_mix, n_comp), 1.5)
    gm = np.array([0.4, 0.6])
    c_all = np.zeros((n_models, n_comp))
    data_white = rng.randn(n_comp, n_samp)

    ll = compute_multimodel_loglikelihood(
        jnp.asarray(y_all),
        jnp.asarray(W_all),
        jnp.asarray(alpha_all),
        jnp.asarray(mu_all),
        jnp.asarray(beta_all),
        jnp.asarray(rho_all),
        jnp.asarray(gm),
        jnp.asarray(c_all),
        jnp.asarray(data_white),
        log_det_sphere=0.1,
    )
    assert np.isfinite(float(ll))
