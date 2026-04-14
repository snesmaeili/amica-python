"""Parameter update equations (M-step) for AMICA using JAX/NumPy."""
from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import numpy as np
from scipy.special import digamma as scipy_digamma

from .backend import jax, jnp, optional_jit, HAS_JAX

if HAS_JAX:
    from jax.scipy.special import digamma
else:
    digamma = scipy_digamma

from .pdf import compute_responsibilities, compute_score_function


@jax.jit
def compute_newton_terms(
    y: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Newton update terms (sigma2, kappa, lambda) for A update.

    This mirrors the Fortran AMICA Newton accumulators for the single-model case.
    Fully vectorized using JAX for GPU acceleration.
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_components, n_samples)
        Source activations.
    alpha : jnp.ndarray, shape (n_mix, n_components)
        Mixture weights.
    mu : jnp.ndarray, shape (n_mix, n_components)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix, n_components)
        Inverse scale parameters (sbeta).
    rho : jnp.ndarray, shape (n_mix, n_components)
        Shape parameters.
        
    Returns
    -------
    sigma2 : jnp.ndarray, shape (n_components,)
    kappa : jnp.ndarray, shape (n_components,)
    lambda_ : jnp.ndarray, shape (n_components,)
    """
    n_mix, n_comp = alpha.shape
    n_samples = y.shape[1]

    # sigma2 = E[y^2] per component
    sigma2 = jnp.mean(y * y, axis=1)
    
    def compute_for_component(i):
        """Compute kappa and lambda for one component."""
        y_i = y[i]  # (n_samples,)
        
        # Compute responsibilities for this component
        resp = compute_responsibilities(
            y_i, alpha[:, i], mu[:, i], beta[:, i], rho[:, i]
        )  # (n_mix, n_samples)
        
        # baralpha = mean responsibility per mixture
        baralpha = jnp.mean(resp, axis=1)  # (n_mix,)
        
        def compute_for_mix(j):
            """Compute kappa/lambda contribution from one mixture."""
            u = resp[j]  # (n_samples,)
            usum = jnp.sum(u)
            
            r = rho[j, i]
            sbeta = beta[j, i]
            mu_j = mu[j, i]
            
            y_scaled = sbeta * (y_i - mu_j)
            sign_y = jnp.where(y_scaled >= 0.0, 1.0, -1.0)
            abs_y_scaled = jnp.abs(y_scaled)
            fp = r * sign_y * jnp.power(abs_y_scaled, r - 1.0)
            
            safe_usum = jnp.maximum(usum, 1e-12)
            dkap = (sbeta ** 2) * jnp.sum(u * fp * fp) / safe_usum
            tmpvec = fp * y_scaled - 1.0
            dlambda = jnp.sum(u * tmpvec * tmpvec) / safe_usum
            
            return dkap, dlambda, mu_j
        
        # Vectorize over mixtures
        dkaps, dlambdas, mu_js = jax.vmap(compute_for_mix)(jnp.arange(n_mix))
        
        # Weighted sum: kappa = sum(baralpha * dkap)
        kappa_i = jnp.sum(baralpha * dkaps)
        # lambda = sum(baralpha * (dlambda + dkap * mu^2))
        lambda_i = jnp.sum(baralpha * (dlambdas + dkaps * mu_js ** 2))
        
        return kappa_i, lambda_i
    
    # Vectorize over components - vmap returns tuple of stacked arrays
    results = jax.vmap(compute_for_component)(jnp.arange(n_comp))
    kappa = results[0]
    lambda_ = results[1]

    return sigma2, kappa, lambda_


@jax.jit
def update_alpha(
    responsibilities: jnp.ndarray,
) -> jnp.ndarray:
    """Update mixture weights α.
    
    α_j = mean(u_j) across samples
    
    Parameters
    ----------
    responsibilities : jnp.ndarray, shape (n_mix, n_samples)
        Posterior responsibilities.
        
    Returns
    -------
    alpha : jnp.ndarray, shape (n_mix,)
        Updated mixture weights.
    """
    alpha = jnp.mean(responsibilities, axis=1)
    # Ensure normalization and minimum value
    alpha = jnp.maximum(alpha, 1e-10)
    alpha = alpha / jnp.sum(alpha)
    return alpha


@jax.jit
def update_mu(
    y: jnp.ndarray,
    responsibilities: jnp.ndarray,
    mu_current: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
) -> jnp.ndarray:
    """Update location parameters μ.
    
    Following Fortran AMICA (line 1796):
    mu = mu + dmu_numer / dmu_denom
    
    Where (lines 1501, 1504, 1506):
    - dmu_numer = sum(ufp) where ufp = u * fp
    - dmu_denom = sbeta * sum(ufp / y) for rho <= 2.0
    - dmu_denom = sbeta * sum(ufp * fp) for rho > 2.0
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_samples,)
        Source values.
    responsibilities : jnp.ndarray, shape (n_mix, n_samples)
        Posterior responsibilities (u = v * z).
    mu_current : jnp.ndarray, shape (n_mix,)
        Current location parameters.
    beta : jnp.ndarray, shape (n_mix,)
        Inverse scale parameters (sbeta).
    rho : jnp.ndarray, shape (n_mix,)
        Shape parameters.
        
    Returns
    -------
    mu : jnp.ndarray, shape (n_mix,)
        Updated location parameters.
    """
    n_mix = mu_current.shape[0]
    
    def update_single_mu(j):
        u = responsibilities[j]
        m = mu_current[j]
        b = beta[j]
        r = rho[j]

        # Fortran: y_scaled = sbeta * (y - mu)
        y_scaled = b * (y - m)
        abs_y_scaled = jnp.abs(y_scaled)
        sign_y = jnp.where(y_scaled >= 0.0, 1.0, -1.0)

        # fp = rho * sign(y_scaled) * |y_scaled|^(rho-1)
        fp = r * sign_y * jnp.power(abs_y_scaled, r - 1.0)
        
        # ufp = u * fp
        ufp = u * fp
        
        # dmu_numer = sum(ufp) (Fortran line 1501)
        numer = jnp.sum(ufp)
        
        # dmu_denom depends on rho (Fortran lines 1504, 1506)
        # For rho <= 2.0: sbeta * sum(ufp / y)
        # For rho > 2.0: sbeta * sum(ufp * fp)
        safe_y_scaled = jnp.where(jnp.abs(y_scaled) < 1e-12, 1e-12, y_scaled)
        denom = jnp.where(
            r <= 2.0,
            b * jnp.sum(ufp / safe_y_scaled),
            b * jnp.sum(ufp * fp),
        )
        safe_denom = jnp.where(jnp.abs(denom) > 1e-12, denom, 1e-12)
        delta = numer / safe_denom
        
        # Fortran: mu = mu + dmu_numer / dmu_denom
        mu_new = m + delta
        
        return mu_new
    
    mu_new = jax.vmap(update_single_mu)(jnp.arange(n_mix))
    return mu_new


@jax.jit
def update_beta(
    y: jnp.ndarray,
    responsibilities: jnp.ndarray,
    mu: jnp.ndarray,
    rho: jnp.ndarray,
    beta_current: jnp.ndarray,
    invsigmin: float,
    invsigmax: float,
) -> jnp.ndarray:
    """Update inverse scale parameters β (sbeta in Fortran).
    
    Following Fortran AMICA (line 1800):
    sbeta = sbeta * sqrt( dbeta_numer / dbeta_denom )
    
    Where:
    - dbeta_numer = sum(u) (line 1514)
    - dbeta_denom = sum(ufp * y) for rho <= 2.0 (line 1517-1519)
    - dbeta_denom = sum(u * |y|^rho) for rho > 2.0 (lines 1543, 1558)
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_samples,)
        Source values.
    responsibilities : jnp.ndarray, shape (n_mix, n_samples)
        Posterior responsibilities (u = v * z in Fortran).
    mu : jnp.ndarray, shape (n_mix,)
        Location parameters.
    rho : jnp.ndarray, shape (n_mix,)
        Shape parameters.
    beta_current : jnp.ndarray, shape (n_mix,)
        Current beta values (sbeta in Fortran).
    invsigmin : float
        Minimum sbeta value.
    invsigmax : float
        Maximum sbeta value.
        
    Returns
    -------
    beta : jnp.ndarray, shape (n_mix,)
        Updated inverse scale parameters.
    """
    n_mix = mu.shape[0]
    
    def update_single_beta(j):
        u = responsibilities[j]
        m = mu[j]
        r = rho[j]
        b = beta_current[j]

        # Fortran: y = sbeta * (b - mu)
        y_scaled = b * (y - m)
        abs_y_scaled = jnp.abs(y_scaled)
        sign_y = jnp.where(y_scaled >= 0.0, 1.0, -1.0)

        # fp = rho * sign(y_scaled) * |y_scaled|^(rho-1)
        fp = r * sign_y * jnp.power(abs_y_scaled, r - 1.0)
        
        # dbeta_numer = sum(u) (Fortran line 1514)
        numer = jnp.sum(u)
        
        # dbeta_denom depends on rho
        # For rho <= 2.0: sum(ufp * y) (Fortran line 1517)
        # For rho > 2.0: sum(u * |y|^rho) (Fortran line 1543)
        ufp = u * fp
        denom = jnp.where(
            r <= 2.0,
            jnp.sum(ufp * y_scaled),
            jnp.sum(u * jnp.power(abs_y_scaled, r)),
        )

        safe_denom = jnp.where(jnp.abs(denom) > 1e-12, denom, 1e-12)
        ratio = numer / safe_denom
        safe_ratio = jnp.maximum(ratio, 1e-12)
        beta_new = b * jnp.sqrt(safe_ratio)
        
        return beta_new
    
    beta_new = jax.vmap(update_single_beta)(jnp.arange(n_mix))
    
    # Clip to reasonable range (Fortran uses invsigmin, invsigmax)
    beta_new = jnp.clip(beta_new, invsigmin, invsigmax)
    
    return beta_new


@jax.jit
def update_rho_gradient(
    y: jnp.ndarray,
    responsibilities: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho_current: jnp.ndarray,
    rholrate: float,
    minrho: float,
    maxrho: float,
) -> jnp.ndarray:
    """Update shape parameters ρ using gradient descent.
    
    Following Fortran AMICA (lines 1808-1809):
    rho(j,k) = rho(j,k) + rholrate * ( 1.0 - 
         (rho(j,k) / psifun(1.0+1.0/rho(j,k))) * drho_numer(j,k) / drho_denom(j,k) )
    
    Where:
    - drho_numer = sum(u * |y|^rho * log(|y|^rho)) (line 1537)
    - drho_denom = sum(u) (line 1540)
    - psifun is the digamma function: Γ'(x)/Γ(x)
    
    Parameters
    ----------
    y : jnp.ndarray, shape (n_samples,)
        Source values.
    responsibilities : jnp.ndarray, shape (n_mix, n_samples)
        Posterior responsibilities (u = v * z).
    mu : jnp.ndarray, shape (n_mix,)
        Location parameters.
    beta : jnp.ndarray, shape (n_mix,)
        Inverse scale parameters (sbeta).
    rho_current : jnp.ndarray, shape (n_mix,)
        Current shape parameters.
    rholrate : float
        Learning rate for rho.
    minrho : float
        Minimum rho value (typically 1.0).
    maxrho : float
        Maximum rho value (typically 2.0).
        
    Returns
    -------
    rho : jnp.ndarray, shape (n_mix,)
        Updated shape parameters.
    """
    n_mix = rho_current.shape[0]
    
    def update_single_rho(j):
        u = responsibilities[j]
        m = mu[j]
        b = beta[j]
        r = rho_current[j]

        # Fortran: y = sbeta * (b - mu)
        y_scaled = b * (y - m)
        abs_y_scaled = jnp.abs(y_scaled)
        safe_abs = jnp.maximum(abs_y_scaled, 1e-300)

        # tmpy = |y|^rho (Fortran lines 1527, 1531)
        log_abs = jnp.log(safe_abs)
        tmpy = jnp.exp(r * log_abs)  # |y_scaled|^rho

        # logab = log(|y|^rho) = rho * log(|y|)
        logab = r * log_abs
        
        # drho_numer = sum(u * |y|^rho * log(|y|^rho)) (Fortran line 1537)
        drho_numer = jnp.sum(u * tmpy * logab)
        
        # drho_denom = sum(u) (Fortran line 1540)
        drho_denom = jnp.sum(u)
        
        # psi = digamma(1 + 1/rho) (Fortran: psifun(1.0+1.0/rho))
        psi = digamma(1.0 + 1.0 / r)
        
        # Fortran formula (line 1808-1809):
        # rho = rho + rholrate * (1.0 - (rho / psi) * (drho_numer / drho_denom))
        safe_denom = jnp.maximum(drho_denom, 1e-12)
        ratio = drho_numer / safe_denom
        
        # psi(1+1/r) is always positive for r > 0, but safeguard anyway
        safe_psi = jnp.maximum(jnp.abs(psi), 1e-12) * jnp.sign(psi + 1e-12)
        
        gradient_term = 1.0 - (r / safe_psi) * ratio
        rho_new = r + rholrate * gradient_term
        
        return rho_new
    
    rho_new = jax.vmap(update_single_rho)(jnp.arange(n_mix))
    
    # Clip to bounds (Fortran lines 1812-1813)
    rho_new = jnp.clip(rho_new, minrho, maxrho)
    
    return rho_new


@jax.jit
def compute_natural_gradient(
    g: jnp.ndarray,
    y: jnp.ndarray,
    W: jnp.ndarray,
    lrate: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute natural gradient update for unmixing matrix W.
    
    The natural gradient update is:
    ΔW = lrate * (I - <g * y^T>) * W
    
    where g is the score function vector and <.> denotes expectation.
    
    Parameters
    ----------
    g : jnp.ndarray, shape (n_components, n_samples)
        Score function values.
    y : jnp.ndarray, shape (n_components, n_samples)
        Source activations.
    W : jnp.ndarray, shape (n_components, n_components)
        Current unmixing matrix.
    lrate : float
        Learning rate.
        
    Returns
    -------
    W_new : jnp.ndarray, shape (n_components, n_components)
        Updated unmixing matrix.
    dW : jnp.ndarray, shape (n_components, n_components)
        Change in W (for monitoring convergence).
    """
    n_components, n_samples = y.shape
    
    # Compute <g * y^T>
    # Shape: (n_components, n_components)
    gy = jnp.dot(g, y.T) / n_samples
    
    # Natural gradient: (I - gy) @ W
    I = jnp.eye(n_components)
    dW = lrate * jnp.dot(I - gy, W)
    
    W_new = W + dW
    
    return W_new, dW


@jax.jit
def compute_newton_correction(
    g: jnp.ndarray,
    y: jnp.ndarray,
    responsibilities: jnp.ndarray,
    dW: jnp.ndarray,
    newtrate: float,
) -> jnp.ndarray:
    """Apply Newton correction to gradient update.
    
    Newton's method uses second-order information for faster convergence.
    
    Parameters
    ----------
    g : jnp.ndarray, shape (n_components, n_samples)
        Score function values.
    y : jnp.ndarray, shape (n_components, n_samples)
        Source activations.
    responsibilities : jnp.ndarray, shape (n_mix, n_components, n_samples)
        Posterior responsibilities for all components.
    dW : jnp.ndarray, shape (n_components, n_components)
        Gradient-based update.
    newtrate : float
        Newton rate (0 to 1).
        
    Returns
    -------
    dW_newton : jnp.ndarray, shape (n_components, n_components)
        Newton-corrected update.
    """
    n_components, n_samples = y.shape
    
    # Compute diagonal Hessian approximation
    # λ = E[u * (f'*y - 1)²]
    # κ = E[u * f'²]
    
    # For now, use simple diagonal scaling
    # This is a simplified Newton step
    
    # Compute expected kurtosis-like quantities per component
    gy_diag = jnp.sum(g * y, axis=1) / n_samples
    g2_diag = jnp.sum(g * g, axis=1) / n_samples
    
    # Hessian diagonal approximation: 1 - κ where κ = E[f'²]
    hess_diag = 1.0 - g2_diag
    
    # Avoid division by zero
    safe_hess = jnp.where(jnp.abs(hess_diag) > 1e-6, hess_diag, 1.0)
    
    # Apply Newton correction to diagonal
    # This is simplified; full Newton would invert full Hessian
    scale = jnp.abs(1.0 / safe_hess)
    scale = jnp.clip(scale, 0.1, 10.0)  # Limit scaling
    
    # Blend with original gradient
    dW_newton = newtrate * (dW * scale[:, None]) + (1.0 - newtrate) * dW
    
    return dW_newton


@jax.jit
def apply_full_newton_correction(
    dA: jnp.ndarray,
    sigma2: jnp.ndarray,
    kappa: jnp.ndarray,
    lambda_: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Apply full pairwise Newton correction to natural gradient update.
    
    This matches the pairwise Hessian inversion logic in Fortran AMICA.
    
    Parameters
    ----------
    dA : jnp.ndarray, shape (n_components, n_components)
        Natural gradient update matrix (I - <g*y^T>).
    sigma2 : jnp.ndarray, shape (n_components,)
        Second moment E[y^2].
    kappa : jnp.ndarray, shape (n_components,)
        Expected curvature term.
    lambda_ : jnp.ndarray, shape (n_components,)
        Diagonal scaling term.
        
    Returns
    -------
    Wtmp : jnp.ndarray
        Newton-corrected update direction.
    posdef : jnp.ndarray (bool scalar)
        Whether the Hessian approximation was positive definite.
    """
    n_components = dA.shape[0]
    
    # Check diagonal positivity condition: lambda > 0
    diag_pos_def = jnp.all(lambda_ > 0)
    
    # Diagonal update: Wtmp[i, i] = dA[i, i] / lambda_[i]
    # We use safe division to avoid NaNs if lambda <= 0 (handled by pos_def flag)
    lambda_safe = jnp.where(lambda_ > 1e-12, lambda_, 1.0)
    diag_update = jnp.diag(jnp.diag(dA) / lambda_safe)
    
    # Off-diagonal update pairs (i, k)
    # sk1[i, k] = sigma2[i] * kappa[k]
    # sk2[i, k] = sigma2[k] * kappa[i] -- this is just sk1.T
    sk1 = sigma2[:, None] * kappa[None, :]
    sk2 = sigma2[None, :] * kappa[:, None]  # equivalent to sk1.T
    
    denom = sk1 * sk2 - 1.0
    numer = sk1 * dA - dA.T
    
    # Pairwise condition: sk1 * sk2 > 1.0 (determinant condition)
    pair_pos_def_all = sk1 * sk2 > 1.0
    
    # Set diagonal of condition to True (handled separately)
    pair_pos_def = pair_pos_def_all | jnp.eye(n_components, dtype=bool)
    
    # Global posdef flag
    posdef = diag_pos_def & jnp.all(pair_pos_def)
    
    # Compute off-diagonal update
    # Use safe denominator
    denom_safe = jnp.where(jnp.abs(denom) > 1e-12, denom, 1.0)
    off_diag_update = numer / denom_safe
    
    # Combine: Diagonal part + Off-diagonal part (masked by I)
    I_mask = jnp.eye(n_components)
    Wtmp = diag_update + off_diag_update * (1.0 - I_mask)
    
    # If not posdef, technically we should fallback to dA. 
    # But caller handles fallback logic based on returned boolean.
    return Wtmp, posdef


def update_all_pdf_params(
    y: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    config,
    rholrate: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update all PDF parameters for all components.
    
    Fully vectorized using JAX vmap for GPU acceleration.
    
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
    config : AmicaConfig
        Configuration object.
    rholrate : Optional[float]
        Optional override for rho learning rate.
        
    Returns
    -------
    alpha_new : jnp.ndarray
    mu_new : jnp.ndarray
    beta_new : jnp.ndarray
    rho_new : jnp.ndarray
    """
    n_components = y.shape[0]
    
    if rholrate is None:
        rholrate = config.rholrate
    
    # Extract config values for use in vectorized function
    invsigmin = config.invsigmin
    invsigmax = config.invsigmax
    minrho = config.minrho
    maxrho = config.maxrho
    do_alpha = config.update_alpha
    do_mu = config.update_mu
    do_beta = config.update_beta
    do_rho = config.update_rho
    
    def update_single_component(i):
        """Update all PDF params for one component."""
        y_i = y[i]  # (n_samples,)
        alpha_i = alpha[:, i]  # (n_mix,)
        mu_i = mu[:, i]  # (n_mix,)
        beta_i = beta[:, i]  # (n_mix,)
        rho_i = rho[:, i]  # (n_mix,)
        
        # Compute responsibilities for this component
        resp = compute_responsibilities(y_i, alpha_i, mu_i, beta_i, rho_i)
        
        # Update alpha if enabled
        alpha_new_i = jnp.where(
            do_alpha,
            update_alpha(resp),
            alpha_i
        )
        
        # Update mu if enabled
        mu_new_i = jnp.where(
            do_mu,
            update_mu(y_i, resp, mu_i, beta_i, rho_i),
            mu_i
        )
        
        # Update beta if enabled
        beta_new_i = jnp.where(
            do_beta,
            update_beta(y_i, resp, mu_i, rho_i, beta_i, invsigmin, invsigmax),
            beta_i
        )
        
        # Update rho if enabled
        rho_new_i = jnp.where(
            do_rho,
            update_rho_gradient(y_i, resp, mu_i, beta_i, rho_i, rholrate, minrho, maxrho),
            rho_i
        )
        
        return alpha_new_i, mu_new_i, beta_new_i, rho_new_i
    
    # Vectorize over all components
    results = jax.vmap(update_single_component)(jnp.arange(n_components))
    
    # results is a tuple of 4 arrays, each with shape (n_components, n_mix)
    # We need to transpose to get (n_mix, n_components)
    alpha_new = results[0].T
    mu_new = results[1].T
    beta_new = results[2].T
    rho_new = results[3].T
    
    return alpha_new, mu_new, beta_new, rho_new


@jax.jit
def update_model_weights(
    model_logliks: jnp.ndarray,
    gm_current: jnp.ndarray,
) -> jnp.ndarray:
    """Update model weights for multiple ICA models.
    
    Parameters
    ----------
    model_logliks : jnp.ndarray, shape (n_models, n_samples)
        Log-likelihood per sample for each model.
    gm_current : jnp.ndarray, shape (n_models,)
        Current model weights.
        
    Returns
    -------
    gm_new : jnp.ndarray, shape (n_models,)
        Updated model weights.
    """
    # Compute model responsibilities
    log_weighted = model_logliks + jnp.log(gm_current)[:, None]
    log_total = jax.scipy.special.logsumexp(log_weighted, axis=0)
    log_resp = log_weighted - log_total
    resp = jnp.exp(log_resp)
    
    # New weights = mean responsibilities
    gm_new = jnp.mean(resp, axis=1)
    gm_new = jnp.maximum(gm_new, 1e-10)
    gm_new = gm_new / jnp.sum(gm_new)
    
    return gm_new


@jax.jit
def update_model_centers(
    data_white: jnp.ndarray,
    model_resp: jnp.ndarray,
    c_current: jnp.ndarray,
) -> jnp.ndarray:
    """Update model centers c.
    
    c_h = Σ_t v_h(t) * x(t) / Σ_t v_h(t)
    
    where v_h is the model responsibility.
    
    Parameters
    ----------
    data_white : jnp.ndarray, shape (n_components, n_samples)
        Whitened data.
    model_resp : jnp.ndarray, shape (n_samples,)
        Model responsibility per sample.
    c_current : jnp.ndarray, shape (n_components,)
        Current model center.
        
    Returns
    -------
    c_new : jnp.ndarray, shape (n_components,)
        Updated model center.
    """
    numer = jnp.dot(data_white, model_resp)
    denom = jnp.sum(model_resp)
    
    c_new = jnp.where(denom > 1e-10, numer / denom, c_current)
    
    return c_new


# -----------------------------------------------------------------------
# From-stats M-step helpers (for chunked E-step path)
# -----------------------------------------------------------------------
# These accept pre-accumulated sufficient statistics from accumulators.py
# and perform the same M-step as the full-batch functions above, but
# without needing the full (T, ...) tensors in memory.
#
# The accumulators are sums (not means). Division by n_total happens
# inside these helpers where appropriate.
# -----------------------------------------------------------------------


@jax.jit
def apply_alpha_update_from_stats(resp_sum: jnp.ndarray, n_total: float) -> jnp.ndarray:
    """Alpha update from pre-accumulated responsibility sum.

    alpha = sum(resp) / n_total, then clipped and normalized per component.
    Same as update_alpha but with `resp_sum = sum(resp, axis=time)` pre-computed.
    """
    alpha = resp_sum / n_total                         # (n_mix, n_comp)
    alpha = jnp.maximum(alpha, 1e-10)
    alpha = alpha / jnp.sum(alpha, axis=0, keepdims=True)
    return alpha


@jax.jit
def apply_mu_update_from_stats(
    mu_current: jnp.ndarray,        # (n_mix, n_comp)
    mu_numer: jnp.ndarray,          # sum(u*fp), shape (n_mix, n_comp)
    mu_denom_le2: jnp.ndarray,      # b*sum(u*fp/y_scaled), shape (n_mix, n_comp)
    mu_denom_gt2: jnp.ndarray,      # b*sum(u*fp*fp), shape (n_mix, n_comp)
    rho: jnp.ndarray,               # (n_mix, n_comp)
) -> jnp.ndarray:
    """mu update from pre-accumulated (u*fp) statistics.

    mu_new = mu + numer/denom where denom is selected by rho <= 2.0
    (same branching as update_mu lines 196-200).
    """
    denom = jnp.where(rho <= 2.0, mu_denom_le2, mu_denom_gt2)
    safe_denom = jnp.where(jnp.abs(denom) > 1e-12, denom, 1e-12)
    return mu_current + mu_numer / safe_denom


@jax.jit
def apply_beta_update_from_stats(
    beta_current: jnp.ndarray,      # (n_mix, n_comp)
    beta_numer: jnp.ndarray,        # sum(u), shape (n_mix, n_comp)
    beta_denom_le2: jnp.ndarray,    # sum(u*fp*y_scaled), shape (n_mix, n_comp)
    beta_denom_gt2: jnp.ndarray,    # sum(u*|y_scaled|^rho), shape (n_mix, n_comp)
    rho: jnp.ndarray,               # (n_mix, n_comp)
    invsigmin: float,
    invsigmax: float,
) -> jnp.ndarray:
    """beta update from pre-accumulated statistics.

    beta_new = beta * sqrt(numer / denom) with branching by rho <= 2.0,
    then clipped. Same as update_beta lines 278-295.
    """
    denom = jnp.where(rho <= 2.0, beta_denom_le2, beta_denom_gt2)
    safe_denom = jnp.where(jnp.abs(denom) > 1e-12, denom, 1e-12)
    ratio = beta_numer / safe_denom
    safe_ratio = jnp.maximum(ratio, 1e-12)
    beta_new = beta_current * jnp.sqrt(safe_ratio)
    return jnp.clip(beta_new, invsigmin, invsigmax)


@jax.jit
def apply_rho_update_from_stats(
    rho_current: jnp.ndarray,       # (n_mix, n_comp)
    rho_numer: jnp.ndarray,         # sum(u*|y|^rho*rho*log|y|), shape (n_mix, n_comp)
    rho_denom: jnp.ndarray,         # sum(u), shape (n_mix, n_comp)
    rholrate: float,
    minrho: float,
    maxrho: float,
) -> jnp.ndarray:
    """rho update from pre-accumulated statistics.

    rho_new = rho + rholrate * (1 - (rho/psi(1+1/rho)) * numer/denom)
    then clipped. Same as update_rho_gradient lines 372-390.
    """
    psi = digamma(1.0 + 1.0 / rho_current)
    safe_denom = jnp.maximum(rho_denom, 1e-12)
    ratio = rho_numer / safe_denom
    safe_psi = jnp.maximum(jnp.abs(psi), 1e-12) * jnp.sign(psi + 1e-12)
    gradient_term = 1.0 - (rho_current / safe_psi) * ratio
    rho_new = rho_current + rholrate * gradient_term
    return jnp.clip(rho_new, minrho, maxrho)


@jax.jit
def compute_newton_terms_from_stats(
    sigma2_partial: jnp.ndarray,    # (n_comp,) = sum(y*y)
    resp_sum: jnp.ndarray,          # (n_mix, n_comp) = sum(u)
    kappa_numer: jnp.ndarray,       # (n_mix, n_comp) = sum(u*fp*fp)
    lambda_numer: jnp.ndarray,      # (n_mix, n_comp) = sum(u*(fp*y-1)^2)
    mu: jnp.ndarray,                # (n_mix, n_comp)
    beta: jnp.ndarray,              # (n_mix, n_comp)
    n_total: float,
):
    """Compute Newton (sigma2, kappa, lambda) from accumulated stats.

    Matches compute_newton_terms (updates.py:20-106) line-for-line:
      sigma2 = mean(y*y)
      dkap_j = beta_j^2 * sum(u*fp*fp) / sum(u)
      dlambda_j = sum(u*(fp*y-1)^2) / sum(u)
      baralpha_j = mean(resp_j) = sum(u) / n_total
      kappa = sum_j(baralpha * dkap)
      lambda = sum_j(baralpha * (dlambda + dkap * mu^2))
    """
    # sigma2 = mean(y*y)
    sigma2 = sigma2_partial / n_total             # (n_comp,)

    # Per-mixture: dkap_j = sbeta_j^2 * sum(u*fp*fp) / sum(u)
    safe_u = jnp.maximum(resp_sum, 1e-12)         # (n_mix, n_comp)
    dkap = (beta ** 2) * kappa_numer / safe_u     # (n_mix, n_comp)
    dlambda = lambda_numer / safe_u               # (n_mix, n_comp)

    # baralpha = mean(u) = sum(u) / n_total
    baralpha = resp_sum / n_total                 # (n_mix, n_comp)

    # kappa = sum_j(baralpha * dkap), per component
    kappa = jnp.sum(baralpha * dkap, axis=0)                          # (n_comp,)
    lambda_ = jnp.sum(baralpha * (dlambda + dkap * mu ** 2), axis=0)  # (n_comp,)

    return sigma2, kappa, lambda_
