"""Chunked E-step accumulators for CPU memory scalability.

The AMICA E-step materializes (n_comp, n_samples) tensors for y, g,
responsibilities, u*fp etc. On real EEG this exceeds RAM on CPU nodes.

Every quantity the M-step needs is a *sample sum* along the time axis
(verified in the audit trace). Splitting the time axis into chunks,
accumulating partial sums, and dividing by the total sample count once
at the end is an algebraic identity — identical fixed point to full-batch
within O(eps*T) float64 rounding.

This module provides the chunk-level accumulator. The outer loop in
solver.py sums the accumulators across chunks and hands the totals to
the M-step.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .likelihood import compute_source_loglikelihood, compute_log_det_W
from .pdf import compute_responsibilities


class ChunkStats(NamedTuple):
    """Sufficient statistics accumulated for one time-chunk.

    All fields are sums (not means) — the outer loop divides by the
    total n_samples AFTER summing all chunk contributions.

    Shapes reference n_comp = number of components, n_mix = num_mix_comps.
    """
    gy_partial: jnp.ndarray        # (n_comp, n_comp)  = g_chunk @ y_chunk.T
    sigma2_partial: jnp.ndarray    # (n_comp,)         = sum(y^2, axis=time)
    data_sum: jnp.ndarray          # (n_comp,)         = sum(data_white_chunk, axis=time) [for c update]

    resp_sum: jnp.ndarray          # (n_mix, n_comp)   = sum(resp, axis=time) [for alpha]

    mu_numer: jnp.ndarray          # (n_mix, n_comp)   = sum(u*fp)
    mu_denom_le2: jnp.ndarray      # (n_mix, n_comp)   = sum(u*fp / y_scaled)    [for rho <= 2.0]
    mu_denom_gt2: jnp.ndarray      # (n_mix, n_comp)   = sum(u*fp * fp)          [for rho > 2.0]

    beta_denom_le2: jnp.ndarray    # (n_mix, n_comp)   = sum(u*fp * y_scaled)    [for rho <= 2.0]
    beta_denom_gt2: jnp.ndarray    # (n_mix, n_comp)   = sum(u * |y_scaled|^rho) [for rho > 2.0]

    rho_numer: jnp.ndarray         # (n_mix, n_comp)   = sum(u * |y|^rho * rho*log|y|)

    kappa_numer: jnp.ndarray       # (n_mix, n_comp)   = sum(u * fp^2)
    lambda_numer: jnp.ndarray      # (n_mix, n_comp)   = sum(u * (fp*y_scaled - 1)^2)

    ll_sum: jnp.ndarray            # scalar            = sum of per-sample source_ll
    n_chunk: jnp.ndarray           # scalar            = y_chunk.shape[1]


@jax.jit
def _chunk_stats_one_component(i, y_chunk, alpha, mu, beta, rho):
    """Compute per-component partial stats for one chunk.

    Returns a tuple of (n_mix,)-shaped arrays for one component i.
    The outer caller vmaps this over components.
    """
    y_i = y_chunk[i]                      # (n_chunk,)
    alpha_i = alpha[:, i]                 # (n_mix,)
    mu_i = mu[:, i]
    beta_i = beta[:, i]
    rho_i = rho[:, i]

    # Responsibilities for this component (n_mix, n_chunk)
    resp = compute_responsibilities(y_i, alpha_i, mu_i, beta_i, rho_i)

    n_mix = alpha_i.shape[0]

    def per_mix(j):
        u = resp[j]                       # (n_chunk,)
        m = mu_i[j]
        b = beta_i[j]
        r = rho_i[j]

        y_scaled = b * (y_i - m)          # (n_chunk,)
        abs_y = jnp.abs(y_scaled)
        sign_y = jnp.where(y_scaled >= 0.0, 1.0, -1.0)
        fp = r * sign_y * jnp.power(abs_y, r - 1.0)

        ufp = u * fp

        # mu numer/denom
        mu_n = jnp.sum(ufp)
        safe_y = jnp.where(jnp.abs(y_scaled) < 1e-12, 1e-12, y_scaled)
        mu_d_le2 = b * jnp.sum(ufp / safe_y)
        mu_d_gt2 = b * jnp.sum(ufp * fp)

        # beta numer/denom
        u_sum = jnp.sum(u)
        beta_d_le2 = jnp.sum(ufp * y_scaled)
        beta_d_gt2 = jnp.sum(u * jnp.power(abs_y, r))

        # rho numer (denom is u_sum)
        safe_abs = jnp.maximum(abs_y, 1e-300)
        log_abs = jnp.log(safe_abs)
        tmpy = jnp.exp(r * log_abs)       # |y|^rho
        logab = r * log_abs
        rho_n = jnp.sum(u * tmpy * logab)

        # Newton accumulators
        kappa_n = jnp.sum(ufp * fp)
        lambda_tmp = fp * y_scaled - 1.0
        lambda_n = jnp.sum(u * lambda_tmp * lambda_tmp)

        return (u_sum, mu_n, mu_d_le2, mu_d_gt2,
                beta_d_le2, beta_d_gt2, rho_n,
                kappa_n, lambda_n)

    outs = jax.vmap(per_mix)(jnp.arange(n_mix))
    return outs  # tuple of 9 arrays, each (n_mix,)


@jax.jit
def compute_chunk_stats(
    data_chunk: jnp.ndarray,       # (n_comp, n_chunk) - pre-centered (data - c)
    W: jnp.ndarray,                # (n_comp, n_comp)
    alpha: jnp.ndarray,            # (n_mix, n_comp)
    mu: jnp.ndarray,               # (n_mix, n_comp)
    beta: jnp.ndarray,             # (n_mix, n_comp)
    rho: jnp.ndarray,              # (n_mix, n_comp)
    log_det_sphere: float,
) -> ChunkStats:
    """Compute all sufficient statistics for one time-chunk.

    Parameters
    ----------
    data_chunk : (n_comp, n_chunk) — the chunk slice of (data_white - c).
        The caller subtracts c; this avoids keeping it as separate argument.
    W, alpha, mu, beta, rho : current model parameters.
    log_det_sphere : scalar, added to per-sample LL.

    Returns
    -------
    ChunkStats with all partial sums (not divided by n).
    """
    n_comp = W.shape[0]
    n_chunk = data_chunk.shape[1]
    n_mix = alpha.shape[0]

    # Sources
    y = jnp.dot(W, data_chunk)            # (n_comp, n_chunk)

    # Compute score function g = sum_j beta_j * resp_j * fp_j per component
    # Reuse the existing compute_all_scores to keep score semantics identical.
    from .pdf import compute_all_scores
    g = compute_all_scores(y, alpha, mu, beta, rho)     # (n_comp, n_chunk)

    # Natural-gradient numerator (sum over time — NOT mean yet)
    gy_partial = jnp.dot(g, y.T)          # (n_comp, n_comp)

    # sigma2 partial (sum of y^2 over time)
    sigma2_partial = jnp.sum(y * y, axis=1)  # (n_comp,)

    # data sum for c update (sum of data_chunk over time)
    # data_chunk here is (data_white - c); we need sum(data_white, axis=time).
    # We'll pass in data_white directly to avoid re-adding c. So caller must
    # pass data_white_chunk (not centered). Fix: we handle this in solver.py
    # by tracking data sum separately in the outer loop.
    data_sum = jnp.sum(data_chunk, axis=1)    # placeholder; caller provides true data_white sum

    # Per-component, per-mixture sufficient statistics
    comp_outs = jax.vmap(
        lambda i: _chunk_stats_one_component(i, y, alpha, mu, beta, rho)
    )(jnp.arange(n_comp))
    # comp_outs is tuple of 9 arrays, each (n_comp, n_mix) — we need (n_mix, n_comp)
    (u_sum, mu_n, mu_d_le2, mu_d_gt2,
     beta_d_le2, beta_d_gt2, rho_n,
     kappa_n, lambda_n) = [a.T for a in comp_outs]  # each (n_mix, n_comp)

    # Per-sample log-likelihood sum
    source_ll = compute_source_loglikelihood(y, alpha, mu, beta, rho)  # (n_chunk,)
    log_det_W = compute_log_det_W(W)
    ll_per_sample = source_ll + log_det_W + log_det_sphere
    ll_sum = jnp.sum(ll_per_sample)

    return ChunkStats(
        gy_partial=gy_partial,
        sigma2_partial=sigma2_partial,
        data_sum=data_sum,
        resp_sum=u_sum,
        mu_numer=mu_n,
        mu_denom_le2=mu_d_le2,
        mu_denom_gt2=mu_d_gt2,
        beta_denom_le2=beta_d_le2,
        beta_denom_gt2=beta_d_gt2,
        rho_numer=rho_n,
        kappa_numer=kappa_n,
        lambda_numer=lambda_n,
        ll_sum=ll_sum,
        n_chunk=jnp.asarray(n_chunk, dtype=jnp.float64),
    )


def zero_stats(n_comp: int, n_mix: int, dtype=jnp.float64) -> ChunkStats:
    """Zero-initialized accumulator matching the ChunkStats shapes."""
    z_cc = jnp.zeros((n_comp, n_comp), dtype=dtype)
    z_c = jnp.zeros((n_comp,), dtype=dtype)
    z_mc = jnp.zeros((n_mix, n_comp), dtype=dtype)
    z_s = jnp.asarray(0.0, dtype=dtype)
    return ChunkStats(
        gy_partial=z_cc,
        sigma2_partial=z_c,
        data_sum=z_c,
        resp_sum=z_mc,
        mu_numer=z_mc,
        mu_denom_le2=z_mc,
        mu_denom_gt2=z_mc,
        beta_denom_le2=z_mc,
        beta_denom_gt2=z_mc,
        rho_numer=z_mc,
        kappa_numer=z_mc,
        lambda_numer=z_mc,
        ll_sum=z_s,
        n_chunk=z_s,
    )


def add_stats(a: ChunkStats, b: ChunkStats) -> ChunkStats:
    """Element-wise sum of two ChunkStats (for accumulating across chunks)."""
    return ChunkStats(*(
        getattr(a, f) + getattr(b, f) for f in ChunkStats._fields
    ))
