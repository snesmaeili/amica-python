"""Log-likelihood computation for AMICA using JAX/NumPy."""

from __future__ import annotations

from .backend import jax, jnp
from .pdf import compute_source_loglikelihood


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
    _q, r = jnp.linalg.qr(W)
    diag = jnp.diag(r)
    return jnp.sum(jnp.log(jnp.maximum(jnp.abs(diag), jnp.finfo(diag.dtype).tiny)))


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
    sample_lls = compute_model_loglikelihood(y, alpha, mu, beta, rho, log_det_W, log_det_sphere)
    return compute_average_loglikelihood(sample_lls, n_components)


def compute_loglik_chunk(
    y_chunk: jnp.ndarray,
    W: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    log_det_sphere: float = 0.0,
):
    """Compute (sum of per-sample LL, n_chunk) for one time-chunk.

    Companion to compute_total_loglikelihood for chunked accumulation.
    Returns UNNORMALIZED sum so the caller can average across chunks:

        total_ll_avg = (sum_ll_a + sum_ll_b) / (n_a + n_b) / n_components

    Parameters
    ----------
    y_chunk : jnp.ndarray, shape (n_components, n_chunk)
        Source activations for the time chunk.
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
        (Log, det)(S)| added to every per-sample LL. Default is 0.0.

    Returns
    -------
    sum_ll : float
        Unnormalized sum of log-likelihood for the chunk.
    n_chunk : float
        Number of samples in the chunk (returned as an array scalar to match dtype).
    """
    log_det_W = compute_log_det_W(W)
    source_ll = compute_source_loglikelihood(y_chunk, alpha, mu, beta, rho)
    sample_ll = source_ll + log_det_W + log_det_sphere
    return jnp.sum(sample_ll), jnp.asarray(y_chunk.shape[1], dtype=sample_ll.dtype)


@jax.jit
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

    log p(x) = log Σ_h gamma_h * p((x, h))

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
    n_components = y_all.shape[1]  # For Fortran-matching normalization

    def model_ll_fn(W, y, alpha, mu, beta, rho):
        log_det_W = compute_log_det_W(W)
        return compute_model_loglikelihood(y, alpha, mu, beta, rho, log_det_W, log_det_sphere)

    # Vectorize over models (axis 0)
    model_logliks = jax.vmap(model_ll_fn)(W_all, y_all, alpha_all, mu_all, beta_all, rho_all)

    # Compute log Σ_h gamma_h * p((x, h)) using log-sum-exp
    log_weighted = model_logliks + jnp.log(gm)[:, None]
    total_ll = jax.scipy.special.logsumexp(log_weighted, axis=0)

    # Normalize by n_components to match Fortran
    return jnp.mean(total_ll) / n_components
