"""Tests for jamica.updates module."""

from __future__ import annotations

import numpy as np

from jamica import updates

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
# Tests for full-batch functions
# ---------------------------------------------------------------------------


def test_compute_newton_terms():
    from jamica.backend import jnp

    n_comp, n_mix, n_samp = 4, 3, 500
    y, alpha, mu, beta, rho = _make_mock_inputs(n_comp, n_mix, n_samp)

    sigma2, kappa, lam = updates.compute_newton_terms(
        jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho)
    )

    assert np.asarray(sigma2).shape == (n_comp,)
    assert np.asarray(kappa).shape == (n_comp,)
    assert np.asarray(lam).shape == (n_comp,)
    assert np.all(np.isfinite(np.asarray(sigma2)))


def test_update_alpha():
    from jamica.backend import jnp

    rng = np.random.RandomState(42)
    resp = rng.rand(3, 100)
    resp /= resp.sum(axis=0, keepdims=True)

    alpha_new = updates.update_alpha(jnp.asarray(resp))
    alpha_np = np.asarray(alpha_new)

    assert alpha_np.shape == (3,)
    np.testing.assert_allclose(alpha_np.sum(), 1.0, atol=1e-7)


def test_update_mu_beta_rho():
    from jamica.backend import jnp

    n_comp, n_mix, n_samp = 4, 3, 500
    y, alpha, mu, beta, rho = _make_mock_inputs(n_comp, n_mix, n_samp)

    y_i = jnp.asarray(y[0])
    alpha_i = jnp.asarray(alpha[:, 0])
    mu_i = jnp.asarray(mu[:, 0])
    beta_i = jnp.asarray(beta[:, 0])
    rho_i = jnp.asarray(rho[:, 0])

    from jamica.pdf import compute_responsibilities

    resp = compute_responsibilities(y_i, alpha_i, mu_i, beta_i, rho_i)

    # test update_mu
    mu_new = updates.update_mu(y_i, resp, mu_i, beta_i, rho_i)
    assert np.asarray(mu_new).shape == (n_mix,)

    # test update_beta
    beta_new = updates.update_beta(y_i, resp, mu_i, rho_i, beta_i, invsigmin=1e-5, invsigmax=1e5)
    assert np.asarray(beta_new).shape == (n_mix,)
    assert np.all(np.asarray(beta_new) >= 1e-5)

    # test update_rho_gradient
    rho_new = updates.update_rho_gradient(
        y_i, resp, mu_i, beta_i, rho_i, rholrate=0.1, minrho=1.0, maxrho=2.0
    )
    assert np.asarray(rho_new).shape == (n_mix,)
    assert np.all((np.asarray(rho_new) >= 1.0) & (np.asarray(rho_new) <= 2.0))


def test_natural_gradient_and_newton_correction():
    from jamica.backend import jnp

    rng = np.random.RandomState(42)
    n_comp, n_samp = 4, 1000
    g = rng.randn(n_comp, n_samp)
    y = rng.randn(n_comp, n_samp)
    W = np.eye(n_comp)
    lrate = 0.1

    W_new, dW = updates.compute_natural_gradient(
        jnp.asarray(g), jnp.asarray(y), jnp.asarray(W), lrate
    )
    assert np.asarray(W_new).shape == (n_comp, n_comp)
    assert np.asarray(dW).shape == (n_comp, n_comp)

    # test newton correction
    sigma2 = jnp.asarray(np.ones(n_comp))
    kappa = jnp.asarray(np.ones(n_comp) * 2.0)
    lam = jnp.asarray(np.ones(n_comp) * 1.5)

    Wtmp, posdef = updates.apply_full_newton_correction(jnp.asarray(dW), sigma2, kappa, lam)
    assert np.asarray(Wtmp).shape == (n_comp, n_comp)
    assert isinstance(posdef, (bool, np.bool_)) or hasattr(posdef, "dtype")


def test_update_all_pdf_params():
    from jamica.backend import jnp
    from jamica.config import AmicaConfig

    n_comp, n_mix, n_samp = 4, 3, 500
    y, alpha, mu, beta, rho = _make_mock_inputs(n_comp, n_mix, n_samp)

    config = AmicaConfig()
    a_new, m_new, b_new, r_new = updates.update_all_pdf_params(
        jnp.asarray(y),
        jnp.asarray(alpha),
        jnp.asarray(mu),
        jnp.asarray(beta),
        jnp.asarray(rho),
        config,
    )

    assert np.asarray(a_new).shape == (n_mix, n_comp)
    assert np.asarray(m_new).shape == (n_mix, n_comp)
    assert np.asarray(b_new).shape == (n_mix, n_comp)
    assert np.asarray(r_new).shape == (n_mix, n_comp)


def test_update_model_weights():
    from jamica.backend import jnp

    rng = np.random.RandomState(42)
    model_logliks = rng.randn(2, 500)
    gm_current = np.array([0.4, 0.6])

    gm_new = updates.update_model_weights(jnp.asarray(model_logliks), jnp.asarray(gm_current))
    assert np.asarray(gm_new).shape == (2,)
    np.testing.assert_allclose(np.asarray(gm_new).sum(), 1.0)


# ---------------------------------------------------------------------------
# Tests for from-stats helpers
# ---------------------------------------------------------------------------


def test_apply_updates_from_stats():
    from jamica.backend import jnp

    n_comp, n_mix = 4, 3
    rng = np.random.RandomState(42)

    # mock some stats
    resp_sum = jnp.asarray(rng.rand(n_mix, n_comp))
    n_total = 1000.0

    a_new = updates.apply_alpha_update_from_stats(resp_sum, n_total)
    assert np.asarray(a_new).shape == (n_mix, n_comp)

    mu_cur = jnp.asarray(rng.randn(n_mix, n_comp))
    mu_numer = jnp.asarray(rng.randn(n_mix, n_comp))
    mu_d1 = jnp.asarray(rng.rand(n_mix, n_comp) + 0.1)
    mu_d2 = jnp.asarray(rng.rand(n_mix, n_comp) + 0.1)
    rho = jnp.asarray(np.full((n_mix, n_comp), 1.5))

    m_new = updates.apply_mu_update_from_stats(mu_cur, mu_numer, mu_d1, mu_d2, rho)
    assert np.asarray(m_new).shape == (n_mix, n_comp)

    beta_cur = jnp.asarray(np.ones((n_mix, n_comp)))
    beta_numer = jnp.asarray(rng.rand(n_mix, n_comp) + 0.1)
    b_new = updates.apply_beta_update_from_stats(beta_cur, beta_numer, mu_d1, mu_d2, rho, 1e-5, 1e5)
    assert np.asarray(b_new).shape == (n_mix, n_comp)

    rho_numer = jnp.asarray(rng.randn(n_mix, n_comp))
    rho_denom = jnp.asarray(rng.rand(n_mix, n_comp) + 0.1)
    r_new = updates.apply_rho_update_from_stats(rho, rho_numer, rho_denom, 0.1, 1.0, 2.0)
    assert np.asarray(r_new).shape == (n_mix, n_comp)

    sigma2_p = jnp.asarray(rng.rand(n_comp))
    kappa_n = jnp.asarray(rng.rand(n_mix, n_comp))
    lambda_n = jnp.asarray(rng.rand(n_mix, n_comp))

    s2, k, lam = updates.compute_newton_terms_from_stats(
        sigma2_p, resp_sum, kappa_n, lambda_n, mu_cur, beta_cur, n_total
    )
    assert np.asarray(s2).shape == (n_comp,)
    assert np.asarray(k).shape == (n_comp,)
    assert np.asarray(lam).shape == (n_comp,)
