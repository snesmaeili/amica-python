"""Log-likelihood computation for AMICA using JAX/NumPy."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp as scipy_logsumexp

from .backend import jax, jnp, optional_jit, HAS_JAX

from .pdf import log_generalized_gaussian_mixture, compute_source_loglikelihood


@jax.jit
def compute_log_det_W(W: jnp.ndarray) -> float:
    """Compute (log, det)(W)| for unmixing matrix.
    
    Parameters
    ----------
    W : jnp.ndarray, shape (n_components, n_components)
        Unmixing matrix.
        
    Returns
    -------
    log_det : float
        Log absolute determinant.
    """
    q, r = jnp.linalg.qr(W)
    diag = jnp.diag(r)
    return jnp.sum(jnp.log(jnp.abs(diag) + 1e-300))


@jax.jit
def compute_model_loglikelihood(
    y: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    log_det_W: float,
    log_det_sphere: float = 0.0,
) -> jnp.ndarray:
    """Compute log p((x, model)) for a single ICA model.
    
    log p((x, h)) = (log, W_h)| + (log, S)| + Σ_i log p((y_i, h))
    
    where S is the sphering matrix.
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_components, n_samples)
        Source activations (W @ sphere @ (x - mean)).
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Inverse scale parameters.
    rho : jnp.ndarray, shape (n_mix, n_components)
        Shape parameters.
    log_det_W : float
        (Log, det)(W)|.
    log_det_sphere : float
        (Log, det)(S)| (sphering matrix). Default is 0.
        
    Returns
    -------
    log_lik : jnp.ndarray, shape (n_samples,)
        Log-likelihood per sample.
    """
    # Sum of log PDFs across components
    source_ll = compute_source_loglikelihood(y, alpha, mu, beta, rho)
    
    # Add determinant terms
    log_lik = source_ll + log_det_W + log_det_sphere
    
    return log_lik


@jax.jit
def compute_average_loglikelihood(
    sample_logliks: jnp.ndarray,
    n_components: int = 1,
) -> float:
    """Compute average log-likelihood.
    
    Following Fortran AMICA (line 1746-1748):
    LL = sum(P) / (n_samples * n_components)
    
    This gives the per-component-per-sample average.
    
    Parameters
    ----------
    sample_logliks : jnp.ndarray, shape (n_samples,)
        Log-likelihood per sample (sum across components).
    n_components : int
        Number of ICA components. Dividing by this matches Fortran normalization.
        
    Returns
    -------
    avg_ll : float
        Average log-likelihood per component per sample.
    """
    # Fortran divides by (n_samples * n_components)
    # sample_logliks is already summed over components, so we just divide by n_components
    return jnp.mean(sample_logliks) / n_components



def compute_total_loglikelihood(
    y: jnp.ndarray,
    W: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    log_det_sphere: float = 0.0,
) -> float:
    """Compute total average log-likelihood for single model.
    
    Following Fortran normalization: LL / (n_samples * n_components)
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_components, n_samples)
        Source activations.
    W : jnp.ndarray, shape (n_components, n_components)
        Unmixing matrix.
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Inverse scale parameters.
    rho : jnp.ndarray, shape (n_mix, n_components)
        Shape parameters.
    log_det_sphere : float
        (Log, det)(S)|.
        
    Returns
    -------
    avg_ll : float
        Average log-likelihood per component per sample.
    """
    n_components = y.shape[0]
    log_det_W = compute_log_det_W(W)
    sample_lls = compute_model_loglikelihood(
        y, alpha, mu, beta, rho, log_det_W, log_det_sphere
    )
    return compute_average_loglikelihood(sample_lls, n_components)


def compute_multimodel_loglikelihood(
    y_all: jnp.ndarray,
    W_all: jnp.ndarray,
    alpha_all: jnp.ndarray,
    mu_all: jnp.ndarray,
    beta_all: jnp.ndarray,
    rho_all: jnp.ndarray,
    gm: jnp.ndarray,
    c_all: jnp.ndarray,
    data_white: jnp.ndarray,
    log_det_sphere: float = 0.0,
) -> float:
    """Compute log-likelihood with multiple ICA models.
    
    log p(x) = log Σ_h γ_h * p((x, h))
    
    Parameters
    ----------
    y_all : jnp.ndarray, shape (n_models, n_components, n_samples)
        Source activations for each model.
    W_all : jnp.ndarray, shape (n_models, n_components, n_components)
        Unmixing matrices.
    alpha_all : jnp.ndarray, shape (n_models, n_mix, n_components)
        Mixture weights.
    mu_all : jnp.ndarray, shape (n_models, n_mix, n_components)
        Location parameters.
    beta_all : jnp.ndarray, shape (n_models, n_mix, n_components)
        Inverse scale parameters.
    rho_all : jnp.ndarray, shape (n_models, n_mix, n_components)
        Shape parameters.
    gm : jnp.ndarray, shape (n_models,)
        Model weights.
    c_all : jnp.ndarray, shape (n_models, n_components)
        Model centers.
    data_white : jnp.ndarray, shape (n_components, n_samples)
        Whitened data.
    log_det_sphere : float
        (Log, det)(S)|.
        
    Returns
    -------
    avg_ll : float
        Average log-likelihood.
    """
    n_models = gm.shape[0]
    n_components = y_all.shape[1]  # For Fortran-matching normalization
    n_samples = y_all.shape[2]
    
    # Compute log p((x, h)) for each model
    model_logliks = jnp.zeros((n_models, n_samples))
    
    for h in range(n_models):
        log_det_W = compute_log_det_W(W_all[h])
        model_ll = compute_model_loglikelihood(
            y_all[h], alpha_all[h], mu_all[h], beta_all[h], rho_all[h],
            log_det_W, log_det_sphere
        )
        model_logliks = model_logliks.at[h].set(model_ll)
    
    # Compute log Σ_h γ_h * p((x, h)) using log-sum-exp
    log_weighted = model_logliks + jnp.log(gm)[:, None]
    total_ll = jax.scipy.special.logsumexp(log_weighted, axis=0)
    
    # Normalize by n_components to match Fortran
    return jnp.mean(total_ll) / n_components


@jax.jit
def compute_nd(
    dW: jnp.ndarray,
) -> float:
    """Compute normalized change in W (for convergence monitoring).
    
    nd = ||ΔW||_F / ||W||_F (approximated as just ||ΔW||_F)
    
    Parameters
    ----------
    dW : jnp.ndarray, shape (n_components, n_components)
        Change in unmixing matrix.
        
    Returns
    -------
    nd : float
        Normalized change.
    """
    return jnp.linalg.norm(dW, ord='fro')


@jax.jit
def compute_gradient_norm(
    g: jnp.ndarray,
    y: jnp.ndarray,
) -> float:
    """Compute gradient norm for convergence check.
    
    ||I - E[g y^T]||_F
    
    Parameters
    ----------
    g : jnp.ndarray, shape (n_components, n_samples)
        Score function values.
    y : jnp.ndarray, shape (n_components, n_samples)
        Source activations.
        
    Returns
    -------
    grad_norm : float
        Frobenius norm of gradient matrix.
    """
    n_components, n_samples = y.shape
    
    # E[g y^T]
    gy = jnp.dot(g, y.T) / n_samples
    
    # I - E[g y^T]
    I = jnp.eye(n_components)
    grad = I - gy
    
    return jnp.linalg.norm(grad, ord='fro')
