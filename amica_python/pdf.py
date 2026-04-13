"""Generalized Gaussian PDF functions for AMICA using JAX/NumPy."""
from __future__ import annotations

from functools import partial

import numpy as np
from scipy.special import gammaln as scipy_gammaln

from .backend import jax, jnp, optional_jit, HAS_JAX

if HAS_JAX:
    from jax.scipy.special import gammaln
else:
    gammaln = scipy_gammaln


@jax.jit
def log_generalized_gaussian(
    y: jnp.ndarray,
    mu: float,
    beta: float,
    rho: float,
) -> jnp.ndarray:
    """Compute log PDF of a single generalized Gaussian.
    
    Following Fortran AMICA z0 computation (lines 1283-1300):
    
    For rho = 1 (Laplacian):
        z0 = log(sbeta) - |y_scaled| - log(2)
    
    For rho = 2 (Gaussian):
        z0 = log(sbeta) - y_scaled² - log(sqrt(pi))
    
    General:
        z0 = log(sbeta) - |y_scaled|^rho - gamln(1 + 1/rho) - log(2)
    
    where y_scaled = sbeta * (b - mu)
    
    Note: The alpha (mixture weight) is added separately in log_generalized_gaussian_mixture.
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_samples,)
        Source values (raw, NOT scaled by beta).
    mu : float
        Location parameter.
    beta : float
        Scale parameter (sbeta in Fortran).
    rho : float
        Shape parameter (1=Laplacian, 2=Gaussian).
        
    Returns
    -------
    log_pdf : jnp.ndarray, shape (n_samples,)
        Log PDF values (without alpha).
    """
    # Fortran: y_scaled = sbeta * (b - mu)
    y_scaled = beta * (y - mu)
    abs_y_scaled = jnp.abs(y_scaled)
    
    # Fortran normalizations:
    # rho=1: log(2) = 0.693...
    # rho=2: log(sqrt(pi)) = log(1.772453851) = 0.572...
    # general: gamln(1 + 1/rho) + log(2)
    
    # Use conditional for numerical stability (matching Fortran exactly)
    # General formula works for all rho, but special cases are more precise
    
    # Safe power: |y_scaled|^rho
    safe_abs = jnp.maximum(abs_y_scaled, 1e-300)
    log_abs = jnp.log(safe_abs)
    abs_y_rho = jnp.exp(rho * log_abs)  # |y_scaled|^rho
    
    # Fortran formula: log(sbeta) - |y_scaled|^rho - gamln(1 + 1/rho) - log(2)
    log_norm = jnp.log(jnp.maximum(beta, 1e-300)) - gammaln(1.0 + 1.0/rho) - jnp.log(2.0)
    
    return log_norm - abs_y_rho



@jax.jit
def log_generalized_gaussian_mixture(
    y: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log PDF of a mixture of generalized Gaussians.
    
    p(y) = Σ_j α_j * p_j(y)
    
    where p_j is a generalized Gaussian with parameters (μ_j, β_j, ρ_j).
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_samples,)
        Source values for a single component.
    alpha : jnp.ndarray, shape (n_mix,)
        Mixture weights (must sum to 1).
    mu : jnp.ndarray, shape (n_mix,)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix,)
        Inverse scale parameters.
    rho : jnp.ndarray, shape (n_mix,)
        Shape parameters.
        
    Returns
    -------
    log_pdf : jnp.ndarray, shape (n_samples,)
        Log PDF values.
    """
    n_mix = alpha.shape[0]
    
    # Compute log PDF for each mixture component
    # Shape: (n_mix, n_samples)
    log_pdfs = jax.vmap(
        lambda a, m, b, r: jnp.log(a) + log_generalized_gaussian(y, m, b, r)
    )(alpha, mu, beta, rho)
    
    # Log-sum-exp for numerical stability
    max_log_pdf = jnp.max(log_pdfs, axis=0)
    log_sum = max_log_pdf + jnp.log(jnp.sum(jnp.exp(log_pdfs - max_log_pdf), axis=0))
    
    return log_sum


@jax.jit
def compute_responsibilities(
    y: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Compute posterior responsibilities of mixture components.
    
    u_j(y) = α_j * p_j(y) / Σ_k α_k * p_k(y)
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_samples,)
        Source values.
    alpha : jnp.ndarray, shape (n_mix,)
        Mixture weights.
    mu : jnp.ndarray, shape (n_mix,)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix,)
        Inverse scale parameters.
    rho : jnp.ndarray, shape (n_mix,)
        Shape parameters.
        
    Returns
    -------
    responsibilities : jnp.ndarray, shape (n_mix, n_samples)
        Posterior responsibility of each mixture for each sample.
    """
    # Compute log(α_j * p_j(y)) for each mixture
    log_weighted_pdfs = jax.vmap(
        lambda a, m, b, r: jnp.log(a) + log_generalized_gaussian(y, m, b, r)
    )(alpha, mu, beta, rho)
    
    # Normalize using log-sum-exp
    log_total = jax.scipy.special.logsumexp(log_weighted_pdfs, axis=0)
    
    # Responsibilities in log space, then exponentiate
    log_responsibilities = log_weighted_pdfs - log_total
    
    # Fortran adds +1e-15 floor then re-normalizes (amica17.f90:1353-1358)
    # to prevent exactly-zero responsibilities from causing 0/0 downstream.
    resp = jnp.exp(log_responsibilities) + 1e-15
    return resp / jnp.sum(resp, axis=0, keepdims=True)


@jax.jit
def compute_score_function(
    y: jnp.ndarray,
    mu: float,
    beta: float,
    rho: float,
) -> jnp.ndarray:
    """Compute score function f'(y) for generalized Gaussian.
    
    Following Fortran AMICA: y_scaled = sbeta * (y - mu)
    Then fp = rho * sign(y_scaled) * |y_scaled|^(rho-1)
    
    For rho=1 (Laplacian): fp = sign(y_scaled)
    For rho=2 (Gaussian): fp = 2 * y_scaled
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_samples,)
        Source values (raw activations, not scaled).
    mu : float
        Location parameter.
    beta : float
        Inverse scale parameter (sbeta in Fortran).
    rho : float
        Shape parameter.
        
    Returns
    -------
    fp : jnp.ndarray, shape (n_samples,)
        Score function values.
    """
    # Fortran: y = sbeta * (b - mu)
    y_scaled = beta * (y - mu)
    abs_y_scaled = jnp.abs(y_scaled)

    # Fortran uses sign(1.0, y_scaled), which returns +1 for y_scaled >= 0.
    sign_y = jnp.where(y_scaled >= 0.0, 1.0, -1.0)

    # General formula works for rho=1 and rho=2 as well.
    fp = rho * sign_y * jnp.power(abs_y_scaled, rho - 1.0)

    return fp


@jax.jit
def compute_weighted_score(
    y: jnp.ndarray,
    responsibilities: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Compute responsibility-weighted score function.
    
    Following Fortran AMICA line 1473:
    g(i) = g(i) + sbeta(j) * ufp
    
    where ufp = u * fp and fp is computed using y_scaled = sbeta * (b - mu).
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_samples,)
        Source values for a single component.
    responsibilities : jnp.ndarray, shape (n_mix, n_samples)
        Posterior responsibilities (u = v * z in Fortran).
    mu : jnp.ndarray, shape (n_mix,)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix,)
        Inverse scale parameters (sbeta in Fortran).
    rho : jnp.ndarray, shape (n_mix,)
        Shape parameters.
        
    Returns
    -------
    g : jnp.ndarray, shape (n_samples,)
        Weighted score function.
    """
    n_mix = mu.shape[0]
    
    # Compute score for each mixture component
    # Now includes beta in score computation
    scores = jax.vmap(
        lambda m, b, r: compute_score_function(y, m, b, r)
    )(mu, beta, rho)
    
    # Weight by responsibilities AND additional sbeta factor (Fortran line 1473)
    # g = sbeta * u * fp
    weighted_scores = responsibilities * beta[:, None] * scores
    
    return jnp.sum(weighted_scores, axis=0)


def compute_all_scores(
    y: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Compute weighted score function for all components.
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_components, n_samples)
        Source activations for all components.
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Inverse scale parameters.
    rho : jnp.ndarray, shape (n_mix, n_components)
        Shape parameters.
        
    Returns
    -------
    g : jnp.ndarray, shape (n_components, n_samples)
        Weighted score function for all components.
    """
    n_components = y.shape[0]
    
    def compute_component_score(i):
        resp = compute_responsibilities(
            y[i], alpha[:, i], mu[:, i], beta[:, i], rho[:, i]
        )
        return compute_weighted_score(
            y[i], resp, mu[:, i], beta[:, i], rho[:, i]
        )
    
    # Vectorize over components
    g = jax.vmap(compute_component_score)(jnp.arange(n_components))
    
    return g


@jax.jit
def compute_source_loglikelihood(
    y: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log-likelihood of sources for all components.
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_components, n_samples)
        Source activations.
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Inverse scale parameters.
    rho : jnp.ndarray, shape (n_mix, n_components)
        Shape parameters.
        
    Returns
    -------
    log_lik : jnp.ndarray, shape (n_samples,)
        Sum of log-likelihoods across components.
    """
    n_components = y.shape[0]
    
    def component_loglik(i):
        return log_generalized_gaussian_mixture(
            y[i], alpha[:, i], mu[:, i], beta[:, i], rho[:, i]
        )
    
    # Sum log-likelihoods across components
    log_liks = jax.vmap(component_loglik)(jnp.arange(n_components))
    
    return jnp.sum(log_liks, axis=0)
