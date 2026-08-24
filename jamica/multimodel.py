"""Multi-model AMICA core math (num_models > 1).

AMICA can model data as a mixture of M ICA models, each with its own unmixing
matrix W_h, center c_h, source-density parameters, and prior gamma_h. Each sample
is softly assigned to models via the model posterior

    P[h, t] = log p(x_t | model h) + log gamma_h          (per-model log-lik)
    v[h, t] = softmax_h(P[:, t])                           (model responsibility)

and every M-step sufficient statistic is weighted by v. The total log-likelihood
is a log-sum-exp over models. This is the mechanism for non-stationary data
(Palmer 2011; Hsu et al. 2018, NeuroImage).

Design: this module reuses the single-model machinery wherever possible —
``compute_model_loglikelihood`` for the per-model LL sweep, the
``*_from_stats`` M-step helpers and ``apply_full_newton_correction`` (vmapped
over the model axis), and a v-weighted copy of the per-component accumulator from
``accumulators._chunk_stats_one_component`` (the only change is ``u = v_h * resp``
instead of ``u = resp``). Every accumulator is linear in ``u``, so this single
factor reproduces amica's joint responsibility ``u = v * z`` exactly.

**M=1 reduction (the parity guarantee):** with one model, ``v ≡ 1`` and
``Nv = n_total``; the v-weighted means become ordinary means, the per-model
normalizer ``Nv`` becomes ``n_total``, ``log gamma = log 1 = 0``, and the
log-sum-exp over a single model is the identity. Every quantity below then
matches the single-model fused step. The solver keeps the single-model path
entirely separate (this module is only used for num_models > 1), so this is a
second, defense-in-depth guarantee.

Shapes (leading model axis M; n = n_components, J = num_mix_comps):
    W_all, A_all          (M, n, n)
    c_all                 (M, n)
    alpha/mu/beta/rho_all (M, J, n)
    gm                    (M,)
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

from .backend import jax, jnp
from .likelihood import compute_log_det_W, compute_model_loglikelihood
from .pdf import compute_responsibilities_with_loglik
from .updates import (
    apply_alpha_update_from_stats,
    apply_beta_update_from_stats,
    apply_full_newton_correction,
    apply_mu_update_from_stats,
    apply_rho_update_from_stats,
    compute_newton_terms_from_stats,
)


class MMChunkStats(NamedTuple):
    """Per-model sufficient statistics for one time-chunk (leading model axis M).

    Mirrors ``accumulators.ChunkStats`` field-for-field with an extra leading
    model axis, plus ``Nv`` (the v-mass per model). All accumulators are sums
    over time (and v-weighted); the M-step divides by ``Nv[h]`` per model.
    """

    gy_partial: jnp.ndarray  # (M, n, n)   = g_h @ y_h.T   (g is v-weighted)
    sigma2_partial: jnp.ndarray  # (M, n)      = sum_t v_h * y_h^2
    data_sum: jnp.ndarray  # (M, n)      = sum_t v_h * data_white  [for c]
    Nv: jnp.ndarray  # (M,)        = sum_t v_h               [for gamma]

    resp_sum: jnp.ndarray  # (M, J, n)   = sum(u)            [alpha/beta/rho denom]
    mu_numer: jnp.ndarray  # (M, J, n)   = sum(u*fp)
    mu_denom_le2: jnp.ndarray  # (M, J, n)
    mu_denom_gt2: jnp.ndarray  # (M, J, n)
    beta_denom_le2: jnp.ndarray  # (M, J, n)
    beta_denom_gt2: jnp.ndarray  # (M, J, n)
    rho_numer: jnp.ndarray  # (M, J, n)
    kappa_numer: jnp.ndarray  # (M, J, n)
    lambda_numer: jnp.ndarray  # (M, J, n)

    ll_sum: jnp.ndarray  # scalar      = sum_t logsumexp_h(P)   (multimodel LL)
    n_chunk: jnp.ndarray  # scalar      = n samples in chunk


def _weighted_component_stats(i, y_chunk, w, alpha, mu, beta, rho):
    """Per-component v-weighted accumulators for one model.

    Identical to ``accumulators._chunk_stats_one_component`` except the joint
    responsibility is ``u = w * resp[j]`` (w = model posterior v_h, shape
    (n_chunk,)) instead of ``resp[j]``. Returns the 9 stat arrays (each (n_mix,))
    plus the v-weighted score ``g_i`` (n_chunk,). Source-LL is NOT returned here —
    the LL/v sweep is done separately via ``compute_model_loglikelihood``.
    """
    y_i = y_chunk[i]
    alpha_i = alpha[:, i]
    mu_i = mu[:, i]
    beta_i = beta[:, i]
    rho_i = rho[:, i]

    resp, _ = compute_responsibilities_with_loglik(y_i, alpha_i, mu_i, beta_i, rho_i)
    n_mix = alpha_i.shape[0]

    def per_mix(j):
        u = w * resp[j]  # (n_chunk,)  <-- the single multimodel change
        m = mu_i[j]
        b = beta_i[j]
        r = rho_i[j]

        y_scaled = b * (y_i - m)
        abs_y = jnp.abs(y_scaled)
        sign_y = jnp.where(y_scaled >= 0.0, 1.0, -1.0)
        fp = r * sign_y * jnp.power(abs_y, r - 1.0)
        ufp = u * fp
        g_contrib = b * ufp

        mu_n = jnp.sum(ufp)
        safe_y = jnp.where(jnp.abs(y_scaled) < 1e-12, 1e-12, y_scaled)
        mu_d_le2 = b * jnp.sum(ufp / safe_y)
        mu_d_gt2 = b * jnp.sum(ufp * fp)

        u_sum = jnp.sum(u)
        beta_d_le2 = jnp.sum(ufp * y_scaled)
        beta_d_gt2 = jnp.sum(u * jnp.power(abs_y, r))

        safe_abs = jnp.maximum(abs_y, jnp.finfo(abs_y.dtype).tiny)
        log_abs = jnp.log(safe_abs)
        tmpy = jnp.exp(r * log_abs)
        logab = r * log_abs
        rho_n = jnp.sum(u * tmpy * logab)

        kappa_n = jnp.sum(ufp * fp)
        lambda_tmp = fp * y_scaled - 1.0
        lambda_n = jnp.sum(u * lambda_tmp * lambda_tmp)

        return (
            u_sum,
            mu_n,
            mu_d_le2,
            mu_d_gt2,
            beta_d_le2,
            beta_d_gt2,
            rho_n,
            kappa_n,
            lambda_n,
            g_contrib,
        )

    outs = jax.vmap(per_mix)(jnp.arange(n_mix))
    stats9 = outs[:9]  # tuple of 9 arrays, each (n_mix,)
    g_i = jnp.sum(outs[9], axis=0)  # (n_chunk,)
    return stats9, g_i


def _model_accumulate(y_h, v_h, alpha_h, mu_h, beta_h, rho_h):
    """All per-component v-weighted accumulators for ONE model.

    Returns ``(gy, resp_sum, mu_numer, mu_denom_le2, mu_denom_gt2,
    beta_denom_le2, beta_denom_gt2, rho_numer, kappa_numer, lambda_numer)`` with
    ``gy`` shape (n, n) and the rest (n_mix, n).
    """
    n_comp = y_h.shape[0]
    stats9, g = jax.vmap(
        lambda i: _weighted_component_stats(i, y_h, v_h, alpha_h, mu_h, beta_h, rho_h)
    )(jnp.arange(n_comp))
    # stats9: tuple of 9 arrays each (n_comp, n_mix) -> transpose to (n_mix, n_comp)
    (u_sum, mu_n, mu_d_le2, mu_d_gt2, beta_d_le2, beta_d_gt2, rho_n, kappa_n, lambda_n) = [
        a.T for a in stats9
    ]
    gy = jnp.dot(g, y_h.T)  # (n_comp, n_comp), g already v-weighted
    return (gy, u_sum, mu_n, mu_d_le2, mu_d_gt2, beta_d_le2, beta_d_gt2, rho_n, kappa_n, lambda_n)


def _model_posteriors_from_data(
    data_white, W_all, c_all, alpha_all, mu_all, beta_all, rho_all, gm, log_det_sphere
):
    """E-step model log-lik P and posteriors v for a (chunk of) data.

    Returns ``(P, LL_t, v, y_all)``:
        P     (M, T)  per-model log p(x|h) + log gamma_h
        LL_t  (T,)    logsumexp_h(P)  (the multimodel per-sample LL)
        v     (M, T)  softmax_h(P)    (model posteriors)
        y_all (M, n, T) per-model sources
    """
    y_all = jax.vmap(lambda W, c: jnp.dot(W, data_white - c[:, None]))(W_all, c_all)
    log_det_W_all = jax.vmap(compute_log_det_W)(W_all)  # (M,)
    model_ll = jax.vmap(
        lambda y, a, m, b, r, ldW: compute_model_loglikelihood(y, a, m, b, r, ldW, log_det_sphere)
    )(y_all, alpha_all, mu_all, beta_all, rho_all, log_det_W_all)  # (M, T)
    P = model_ll + jnp.log(gm)[:, None]
    LL_t = jax.scipy.special.logsumexp(P, axis=0)  # (T,)
    v = jnp.exp(P - LL_t[None, :])  # (M, T)
    return P, LL_t, v, y_all


@jax.jit
def compute_chunk_stats_mm(
    data_white_chunk: jnp.ndarray,
    W_all: jnp.ndarray,
    c_all: jnp.ndarray,
    alpha_all: jnp.ndarray,
    mu_all: jnp.ndarray,
    beta_all: jnp.ndarray,
    rho_all: jnp.ndarray,
    gm: jnp.ndarray,
    log_det_sphere: float,
    sample_weight: jnp.ndarray | None = None,
) -> MMChunkStats:
    """Multimodel E-step sufficient statistics for one time-chunk.

    Two sweeps: (1) per-model per-sample LL via ``compute_model_loglikelihood``
    -> log-sum-exp over models -> posteriors v; (2) v-weighted per-model
    accumulators. ``data_white_chunk`` is the UNCENTERED whitened data (each
    model subtracts its own ``c_h``).
    """
    n_chunk = data_white_chunk.shape[1]
    dtype = W_all.dtype

    _P, LL_t, v, y_all = _model_posteriors_from_data(
        data_white_chunk, W_all, c_all, alpha_all, mu_all, beta_all, rho_all, gm, log_det_sphere
    )
    # Likelihood sample rejection (single global mask across models): fold the 0/1
    # sample_weight into the model posteriors so rejected samples drop out of every
    # per-model M-step sum AND the effective counts (Nv / n_chunk / ll_sum). None on
    # the no-rejection path -> taken at trace time, so that graph is byte-identical.
    if sample_weight is None:
        vw = v
        ll_sum = jnp.sum(LL_t)
        n_eff = jnp.asarray(n_chunk, dtype=dtype)
    else:
        vw = v * sample_weight[None, :]
        ll_sum = jnp.sum(LL_t * sample_weight)
        n_eff = jnp.sum(sample_weight).astype(dtype)
    Nv = jnp.sum(vw, axis=1)  # (M,) — effective per-model good count

    # v-weighted center numerator and second moment (per model)
    data_sum = jax.vmap(lambda vh: jnp.sum(data_white_chunk * vh[None, :], axis=1))(vw)  # (M, n)
    sigma2_partial = jax.vmap(lambda y, vh: jnp.sum(y * y * vh[None, :], axis=1))(
        y_all, vw
    )  # (M, n)

    accum = jax.vmap(_model_accumulate)(y_all, vw, alpha_all, mu_all, beta_all, rho_all)
    (
        gy,
        resp_sum,
        mu_numer,
        mu_denom_le2,
        mu_denom_gt2,
        beta_denom_le2,
        beta_denom_gt2,
        rho_numer,
        kappa_numer,
        lambda_numer,
    ) = accum

    return MMChunkStats(
        gy_partial=gy,
        sigma2_partial=sigma2_partial,
        data_sum=data_sum,
        Nv=Nv,
        resp_sum=resp_sum,
        mu_numer=mu_numer,
        mu_denom_le2=mu_denom_le2,
        mu_denom_gt2=mu_denom_gt2,
        beta_denom_le2=beta_denom_le2,
        beta_denom_gt2=beta_denom_gt2,
        rho_numer=rho_numer,
        kappa_numer=kappa_numer,
        lambda_numer=lambda_numer,
        ll_sum=ll_sum,
        n_chunk=n_eff,
    )


def zero_stats_mm(n_models: int, n_comp: int, n_mix: int, dtype=jnp.float64) -> MMChunkStats:
    """Zero-initialized multimodel accumulator (for the chunked outer loop)."""
    z_mnn = jnp.zeros((n_models, n_comp, n_comp), dtype=dtype)
    z_mn = jnp.zeros((n_models, n_comp), dtype=dtype)
    z_m = jnp.zeros((n_models,), dtype=dtype)
    z_mjn = jnp.zeros((n_models, n_mix, n_comp), dtype=dtype)
    z_s = jnp.asarray(0.0, dtype=dtype)
    return MMChunkStats(
        gy_partial=z_mnn,
        sigma2_partial=z_mn,
        data_sum=z_mn,
        Nv=z_m,
        resp_sum=z_mjn,
        mu_numer=z_mjn,
        mu_denom_le2=z_mjn,
        mu_denom_gt2=z_mjn,
        beta_denom_le2=z_mjn,
        beta_denom_gt2=z_mjn,
        rho_numer=z_mjn,
        kappa_numer=z_mjn,
        lambda_numer=z_mjn,
        ll_sum=z_s,
        n_chunk=z_s,
    )


def add_stats_mm(a: MMChunkStats, b: MMChunkStats) -> MMChunkStats:
    """Element-wise sum of two MMChunkStats (accumulate across chunks)."""
    return MMChunkStats(*(getattr(a, f) + getattr(b, f) for f in MMChunkStats._fields))


@partial(
    jax.jit,
    static_argnames=["do_newton", "update_c", "doscaling"],
)
def m_step_mm(
    totals: MMChunkStats,
    W_all,
    A_all,
    c_all,
    alpha_all,
    mu_all,
    beta_all,
    rho_all,
    gm,
    n_total,
    lrate_step,
    rholrate,
    iteration,
    newt_start_iter,
    invsigmin,
    invsigmax,
    minrho,
    maxrho,
    do_newton: bool,
    update_c: bool,
    doscaling: bool,
):
    """Multimodel M-step: per-model parameter update (vmapped) + gamma update.

    Per-model normalizer is ``Nv[h]`` (= n_total at M=1). Reuses the single-model
    ``*_from_stats`` helpers and ``apply_full_newton_correction`` vmapped over M.
    Returns updated per-model params plus ``gm``, the multimodel ``ll``, and
    scalar ``is_good`` / ``newton_used`` flags.
    """
    n_comp = W_all.shape[1]
    dtype = W_all.dtype
    eye = jnp.eye(n_comp, dtype=dtype)

    def per_model(
        gy_h,
        sig_h,
        ds_h,
        Nv_h,
        rs_h,
        mn_h,
        mdle_h,
        mdgt_h,
        bdle_h,
        bdgt_h,
        rn_h,
        kn_h,
        ln_h,
        W_h,
        A_h,
        c_h,
        alpha_h,
        mu_h,
        beta_h,
        rho_h,
    ):
        safe_Nv = jnp.maximum(Nv_h, 1e-30)
        gy = (gy_h / safe_Nv).astype(dtype)
        dA_local = (eye - gy).astype(dtype)

        def _apply_newton(_):
            sigma2, kappa, lambda_ = compute_newton_terms_from_stats(
                sig_h, rs_h, kn_h, ln_h, mu_h, beta_h, safe_Nv
            )
            lambda_pos = jnp.all(lambda_ > 0)
            Wtmp_n, posdef_n = apply_full_newton_correction(dA_local, sigma2, kappa, lambda_)
            is_valid = lambda_pos & posdef_n
            return jnp.where(is_valid, Wtmp_n, dA_local).astype(dtype), is_valid

        def _skip(_):
            return dA_local.astype(dtype), jnp.array(False)

        if do_newton:
            Wtmp, newton_used = jax.lax.cond(
                iteration >= newt_start_iter, _apply_newton, _skip, None
            )
        else:
            Wtmp = dA_local
            newton_used = jnp.array(False)

        dAk = A_h @ Wtmp
        A_new = A_h - lrate_step * dAk
        A_ok = jnp.all(jnp.isfinite(A_new))
        W_new = jnp.where(A_ok, jnp.linalg.pinv(A_new).astype(dtype), W_h)
        is_good = A_ok & jnp.all(jnp.isfinite(W_new))

        alpha_new = apply_alpha_update_from_stats(rs_h, safe_Nv)
        mu_new = apply_mu_update_from_stats(mu_h, mn_h, mdle_h, mdgt_h, rho_h)
        beta_new = apply_beta_update_from_stats(
            beta_h, rs_h, bdle_h, bdgt_h, rho_h, invsigmin, invsigmax
        )
        rho_new = apply_rho_update_from_stats(rho_h, rn_h, rs_h, rholrate, minrho, maxrho)

        c_new = ds_h / safe_Nv if update_c else c_h

        if doscaling:
            col_norms = jnp.linalg.norm(A_new, axis=0)
            col_norms = jnp.where(col_norms > 0.0, col_norms, 1.0)
            A_new = A_new / col_norms
            mu_new = mu_new * col_norms[None, :]
            beta_new = beta_new / col_norms[None, :]
            W_new = W_new * col_norms[:, None]

        return W_new, A_new, c_new, alpha_new, mu_new, beta_new, rho_new, is_good, newton_used

    (W_new, A_new, c_new, alpha_new, mu_new, beta_new, rho_new, is_good_all, newton_all) = jax.vmap(
        per_model
    )(
        totals.gy_partial,
        totals.sigma2_partial,
        totals.data_sum,
        totals.Nv,
        totals.resp_sum,
        totals.mu_numer,
        totals.mu_denom_le2,
        totals.mu_denom_gt2,
        totals.beta_denom_le2,
        totals.beta_denom_gt2,
        totals.rho_numer,
        totals.kappa_numer,
        totals.lambda_numer,
        W_all,
        A_all,
        c_all,
        alpha_all,
        mu_all,
        beta_all,
        rho_all,
    )

    # gamma update: gm_h = Nv_h / n_total (already sums to 1 since sum_h Nv = n_total)
    gm_new = jnp.maximum(totals.Nv / n_total, 1e-30)
    gm_new = gm_new / jnp.sum(gm_new)

    ll = totals.ll_sum / n_total / n_comp
    is_good = jnp.all(is_good_all)
    newton_used = jnp.any(newton_all)

    return (
        W_new,
        A_new,
        c_new,
        alpha_new,
        mu_new,
        beta_new,
        rho_new,
        gm_new,
        ll,
        is_good,
        newton_used,
    )


def _blocked_posteriors(data_white, args, log_det_sphere, chunk_size, pick):
    """Run ``_model_posteriors_from_data`` over time blocks and concatenate ``pick``.

    The returned quantities are per-sample, but the intermediate per-model sources
    are (M, n_components, n_samples); blocking bounds that to the chunk length.
    Each sample depends only on itself, so blocked and full-batch results agree.
    """
    n_samples = data_white.shape[1]
    if chunk_size is None or chunk_size >= n_samples:
        return pick(_model_posteriors_from_data(data_white, *args, log_det_sphere))
    blocks = [
        pick(_model_posteriors_from_data(data_white[:, s : s + chunk_size], *args, log_det_sphere))
        for s in range(0, n_samples, chunk_size)
    ]
    return jnp.concatenate(blocks, axis=-1)


def compute_model_sample_loglik(
    data_white,
    W_all,
    c_all,
    alpha_all,
    mu_all,
    beta_all,
    rho_all,
    gm,
    log_det_sphere: float = 0.0,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Per-sample mixture log-likelihood LL_t (n_samples,) for fitted parameters.

    Used by likelihood-based sample rejection, which evaluates it once per
    rejection round inside the EM loop. See ``chunk_size`` on
    :func:`compute_model_posteriors`.
    """
    args = (W_all, c_all, alpha_all, mu_all, beta_all, rho_all, gm)
    return _blocked_posteriors(data_white, args, log_det_sphere, chunk_size, lambda r: r[1])


def compute_model_posteriors(
    data_white,
    W_all,
    c_all,
    alpha_all,
    mu_all,
    beta_all,
    rho_all,
    gm,
    log_det_sphere: float = 0.0,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """Model posterior time-course v (M, n_samples) for fitted parameters.

    Called once after fitting to populate ``AmicaResult.model_posteriors_`` —
    the p(h|t) curve that tracks which regime is active (Hsu 2018).

    Parameters
    ----------
    chunk_size : int | None
        Process the time axis in blocks of this many samples. ``None`` computes
        in one pass. The returned posteriors are (M, n_samples) either way, but
        the intermediate per-model sources are (M, n_components, n_samples), so
        a full-batch pass on a long recording can dominate peak memory even when
        the fit itself was chunked. Chunking bounds that intermediate to
        (M, n_components, chunk_size); results are identical because each sample's
        posterior depends only on that sample.
    """
    args = (W_all, c_all, alpha_all, mu_all, beta_all, rho_all, gm)
    return _blocked_posteriors(data_white, args, log_det_sphere, chunk_size, lambda r: r[2])
