"""Tests for multi-model AMICA (num_models > 1).

Stage 1 (this file, first block): the core math in ``jamica.multimodel``
reduces EXACTLY to the single-model accumulator at M=1, and its LL / gamma match
the existing (previously-unused) scaffolding. Later stages add solver-level
parity and the Hsu-style synthetic recovery experiment.
"""

from __future__ import annotations

import numpy as np
import pytest

from jamica import multimodel as mm
from jamica.accumulators import compute_chunk_stats
from jamica.backend import jax, jnp
from jamica.likelihood import compute_multimodel_loglikelihood
from jamica.updates import update_model_weights

ATOL = 1e-11


def _rng(seed=0):
    return np.random.default_rng(seed)


def _make_params(rng, n, J):
    W = rng.standard_normal((n, n))
    alpha = rng.random((J, n)) + 0.2
    alpha = alpha / alpha.sum(axis=0, keepdims=True)
    mu = rng.standard_normal((J, n)) * 0.1
    beta = 1.0 + 0.2 * rng.random((J, n))
    rho = 1.5 + 0.4 * rng.random((J, n))
    return (
        jnp.asarray(W, dtype=jnp.float64),
        jnp.asarray(alpha, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(beta, dtype=jnp.float64),
        jnp.asarray(rho, dtype=jnp.float64),
    )


def test_mm_chunk_stats_M1_equals_single():
    """Every MMChunkStats field at M=1 matches the single-model ChunkStats."""
    rng = _rng(1)
    n, J, T = 5, 3, 400
    W, alpha, mu, beta, rho = _make_params(rng, n, J)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)
    log_det_sphere = 0.37  # nonzero, to confirm it propagates identically

    single = compute_chunk_stats(data, W, alpha, mu, beta, rho, log_det_sphere)

    c0 = jnp.zeros((1, n), dtype=jnp.float64)
    multi = mm.compute_chunk_stats_mm(
        data,
        W[None],
        c0,
        alpha[None],
        mu[None],
        beta[None],
        rho[None],
        jnp.asarray([1.0], dtype=jnp.float64),
        log_det_sphere,
    )

    # per-component / per-mix accumulators
    assert np.allclose(multi.gy_partial[0], single.gy_partial, atol=ATOL)
    assert np.allclose(multi.sigma2_partial[0], single.sigma2_partial, atol=ATOL)
    assert np.allclose(multi.data_sum[0], single.data_sum, atol=ATOL)
    assert np.allclose(multi.resp_sum[0], single.resp_sum, atol=ATOL)
    assert np.allclose(multi.mu_numer[0], single.mu_numer, atol=ATOL)
    assert np.allclose(multi.mu_denom_le2[0], single.mu_denom_le2, atol=ATOL)
    assert np.allclose(multi.mu_denom_gt2[0], single.mu_denom_gt2, atol=ATOL)
    assert np.allclose(multi.beta_denom_le2[0], single.beta_denom_le2, atol=ATOL)
    assert np.allclose(multi.beta_denom_gt2[0], single.beta_denom_gt2, atol=ATOL)
    assert np.allclose(multi.rho_numer[0], single.rho_numer, atol=ATOL)
    assert np.allclose(multi.kappa_numer[0], single.kappa_numer, atol=ATOL)
    assert np.allclose(multi.lambda_numer[0], single.lambda_numer, atol=ATOL)
    # scalars + Nv
    assert np.allclose(float(multi.ll_sum), float(single.ll_sum), atol=ATOL)
    assert np.allclose(float(multi.Nv[0]), float(T), atol=ATOL)
    assert np.allclose(float(multi.n_chunk), float(single.n_chunk), atol=ATOL)


def test_mm_v_sums_to_one():
    """Model posteriors v sum to 1 across models at every sample."""
    rng = _rng(2)
    n, J, T, M = 4, 3, 300, 3
    W_all = jnp.stack([_make_params(rng, n, J)[0] for _ in range(M)])
    c_all = jnp.asarray(rng.standard_normal((M, n)) * 0.05, dtype=jnp.float64)
    alpha = jnp.stack([_make_params(rng, n, J)[1] for _ in range(M)])
    muu = jnp.stack([_make_params(rng, n, J)[2] for _ in range(M)])
    beta = jnp.stack([_make_params(rng, n, J)[3] for _ in range(M)])
    rho = jnp.stack([_make_params(rng, n, J)[4] for _ in range(M)])
    gm = jnp.asarray([0.5, 0.3, 0.2], dtype=jnp.float64)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)

    v = mm.compute_model_posteriors(data, W_all, c_all, alpha, muu, beta, rho, gm, 0.0)
    assert v.shape == (M, T)
    assert np.allclose(np.asarray(v).sum(axis=0), 1.0, atol=ATOL)
    assert np.all(np.asarray(v) >= 0.0)


def test_mm_ll_matches_scaffold():
    """compute_chunk_stats_mm ll matches the existing compute_multimodel_loglikelihood."""
    rng = _rng(3)
    n, J, T, M = 4, 3, 250, 2
    W_all = jnp.stack([_make_params(rng, n, J)[0] for _ in range(M)])
    c_all = jnp.zeros((M, n), dtype=jnp.float64)
    alpha = jnp.stack([_make_params(rng, n, J)[1] for _ in range(M)])
    muu = jnp.stack([_make_params(rng, n, J)[2] for _ in range(M)])
    beta = jnp.stack([_make_params(rng, n, J)[3] for _ in range(M)])
    rho = jnp.stack([_make_params(rng, n, J)[4] for _ in range(M)])
    gm = jnp.asarray([0.6, 0.4], dtype=jnp.float64)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)

    stats = mm.compute_chunk_stats_mm(data, W_all, c_all, alpha, muu, beta, rho, gm, 0.0)
    ll_from_stats = float(stats.ll_sum) / T / n

    y_all = jnp.stack([W_all[h] @ data for h in range(M)])
    ll_scaffold = float(
        compute_multimodel_loglikelihood(y_all, W_all, alpha, muu, beta, rho, gm, c_all, data, 0.0)
    )
    assert np.allclose(ll_from_stats, ll_scaffold, atol=ATOL)


def test_mm_gm_equals_update_model_weights():
    """The Nv/n_total gamma update equals update_model_weights on the per-model LL."""
    rng = _rng(4)
    n, J, T, M = 4, 3, 500, 3
    W_all = jnp.stack([_make_params(rng, n, J)[0] for _ in range(M)])
    c_all = jnp.zeros((M, n), dtype=jnp.float64)
    alpha = jnp.stack([_make_params(rng, n, J)[1] for _ in range(M)])
    muu = jnp.stack([_make_params(rng, n, J)[2] for _ in range(M)])
    beta = jnp.stack([_make_params(rng, n, J)[3] for _ in range(M)])
    rho = jnp.stack([_make_params(rng, n, J)[4] for _ in range(M)])
    gm = jnp.asarray([0.4, 0.35, 0.25], dtype=jnp.float64)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)

    stats = mm.compute_chunk_stats_mm(data, W_all, c_all, alpha, muu, beta, rho, gm, 0.0)
    gm_from_Nv = np.asarray(stats.Nv) / T
    gm_from_Nv = gm_from_Nv / gm_from_Nv.sum()

    # reference: per-model per-sample LL fed to update_model_weights
    P, _LL, _v, _y = mm._model_posteriors_from_data(
        data, W_all, c_all, alpha, muu, beta, rho, gm, 0.0
    )
    model_ll = np.asarray(P) - np.log(np.asarray(gm))[:, None]  # strip the prior back out
    gm_ref = np.asarray(update_model_weights(jnp.asarray(model_ll), gm))
    assert np.allclose(gm_from_Nv, gm_ref, atol=1e-9)


# ---------------------------------------------------------------------------
# Stage 3 — M=1 parity: the multimodel step reduces EXACTLY to the fused step
# ---------------------------------------------------------------------------

from jamica.solver import _amica_step_fused, _amica_step_multimodel  # noqa: E402

_NAMES = ["W", "A", "c", "alpha", "mu", "beta", "rho", "gm", "ll", "is_good", "newton_used"]


@pytest.mark.parametrize("iteration,newt_start", [(10, 5), (2, 5)])  # newton on / off
def test_M1_step_matches_fused(iteration, newt_start):
    rng = _rng(7)
    n, J, T = 5, 3, 600
    W, alpha, mu, beta, rho = _make_params(rng, n, J)
    A = jnp.linalg.pinv(W)
    c = jnp.zeros(n, dtype=jnp.float64)
    gm = jnp.asarray([1.0], dtype=jnp.float64)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)

    args_common = (
        0.1,
        0.05,
        data,
        0.0,  # lrate_step, rholrate, data_white, log_det_sphere
        newt_start,
        iteration,
        1e-4,
        1e4,
        1.0,
        2.0,
        True,
        True,
        True,
        True,  # do_newton, do_mean, do_sphere, doscaling
        True,
        True,
        True,
        True,  # update_alpha/mu/beta/rho
    )
    # _amica_step_fused donates its state buffers (W,A,c,alpha,mu,beta,rho;
    # Stage 3F donate_argnums) for in-place reuse — safe in the fit loop, which
    # always accepts the returned state, but this comparison test reuses the same
    # params for the second (multimodel) call, so snapshot copies BEFORE the call.
    Wm, Am, cm = jnp.array(W), jnp.array(A), jnp.array(c)
    am, mm_, bm, rm = jnp.array(alpha), jnp.array(mu), jnp.array(beta), jnp.array(rho)
    out_s = _amica_step_fused(W, A, c, alpha, mu, beta, rho, gm, *args_common)
    out_m = _amica_step_multimodel(
        Wm[None], Am[None], cm[None], am[None], mm_[None], bm[None], rm[None], gm, *args_common
    )

    for name, s, m in zip(_NAMES, out_s, out_m, strict=False):
        s = np.asarray(s)
        m = np.asarray(m)
        if name == "gm":
            assert np.allclose(s, m, atol=1e-11), f"gm: {s} vs {m}"
        elif name in ("ll", "is_good", "newton_used"):
            assert np.allclose(s, m, atol=1e-11), f"{name}: {s} vs {m}"
        else:
            assert np.allclose(s, m[0], atol=1e-11), (
                f"{name} mismatch, max abs diff {np.abs(s - m[0]).max():.2e}"
            )


# ---------------------------------------------------------------------------
# Stage 4 — chunked multimodel == full-batch multimodel
# ---------------------------------------------------------------------------

from jamica.solver import _amica_step_multimodel_chunked  # noqa: E402


def test_mm_chunked_matches_fullbatch_step():
    """One chunked multimodel step equals the full-batch step (summation order)."""
    rng = _rng(11)
    n, J, T, M = 4, 3, 900, 2
    W = jnp.stack([_make_params(rng, n, J)[0] for _ in range(M)])
    A = jax.vmap(jnp.linalg.pinv)(W)
    c = jnp.asarray(rng.standard_normal((M, n)) * 0.05, dtype=jnp.float64)
    alpha = jnp.stack([_make_params(rng, n, J)[1] for _ in range(M)])
    muu = jnp.stack([_make_params(rng, n, J)[2] for _ in range(M)])
    beta = jnp.stack([_make_params(rng, n, J)[3] for _ in range(M)])
    rho = jnp.stack([_make_params(rng, n, J)[4] for _ in range(M)])
    gm = jnp.asarray([0.55, 0.45], dtype=jnp.float64)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)

    common = (
        0.1,
        0.05,
        data,
        0.0,
        5,
        10,
        1e-4,
        1e4,
        1.0,
        2.0,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    )
    out_full = _amica_step_multimodel(W, A, c, alpha, muu, beta, rho, gm, *common)
    out_chunk = _amica_step_multimodel_chunked(
        W, A, c, alpha, muu, beta, rho, gm, *common, chunk_size=250
    )
    for name, f, ch in zip(_NAMES, out_full, out_chunk, strict=False):
        assert np.allclose(np.asarray(f), np.asarray(ch), atol=1e-6, rtol=1e-5), (
            f"{name}: chunked != full (max {np.abs(np.asarray(f) - np.asarray(ch)).max():.2e})"
        )


def test_add_stats_mm_additivity():
    """add_stats_mm over two halves equals one full-chunk accumulation."""
    rng = _rng(12)
    n, J, T, M = 4, 3, 600, 2
    W = jnp.stack([_make_params(rng, n, J)[0] for _ in range(M)])
    c = jnp.zeros((M, n), dtype=jnp.float64)
    alpha = jnp.stack([_make_params(rng, n, J)[1] for _ in range(M)])
    muu = jnp.stack([_make_params(rng, n, J)[2] for _ in range(M)])
    beta = jnp.stack([_make_params(rng, n, J)[3] for _ in range(M)])
    rho = jnp.stack([_make_params(rng, n, J)[4] for _ in range(M)])
    gm = jnp.asarray([0.5, 0.5], dtype=jnp.float64)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)

    full = mm.compute_chunk_stats_mm(data, W, c, alpha, muu, beta, rho, gm, 0.0)
    a = mm.compute_chunk_stats_mm(data[:, : T // 2], W, c, alpha, muu, beta, rho, gm, 0.0)
    b = mm.compute_chunk_stats_mm(data[:, T // 2 :], W, c, alpha, muu, beta, rho, gm, 0.0)
    summed = mm.add_stats_mm(a, b)
    for fname in mm.MMChunkStats._fields:
        assert np.allclose(
            np.asarray(getattr(full, fname)), np.asarray(getattr(summed, fname)), atol=1e-7
        ), f"add_stats_mm mismatch in {fname}"


# ---------------------------------------------------------------------------
# Stage 5 — Hsu-style recovery: M=2 recovers two concatenated ICA regimes
# ---------------------------------------------------------------------------


def _matched_mean_r(A, B):
    """Hungarian-matched mean |corr| between rows of A and B (n, T each)."""
    from scipy.optimize import linear_sum_assignment

    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v = np.corrcoef(A[i], B[j])[0, 1]
            C[i, j] = 1.0 - (abs(v) if np.isfinite(v) else 0.0)
    ri, ci = linear_sum_assignment(C)
    return float(np.mean(1.0 - C[ri, ci]))


def test_multimodel_recovers_two_regimes():
    """Concatenate two distinct ICA mixtures; M=2 recovers both, posteriors
    track the switch, gamma ~ segment fractions, and M=2 LL >= M=1 LL.

    This replicates the Hsu et al. (2018) simulation logic: unsupervised
    multi-model AMICA should match supervised per-segment ICA on non-stationary
    data, whereas a single model cannot.
    """
    from jamica import Amica, AmicaConfig

    rng = np.random.default_rng(20)
    n = 4
    Tseg = 4000  # > 25*H*N^2 = 25*2*16 = 800 per the data-length rule
    # two well-conditioned, distinct mixings
    A1 = rng.standard_normal((n, n))
    A2 = rng.standard_normal((n, n))
    S1 = rng.laplace(size=(n, Tseg))
    S2 = rng.laplace(size=(n, Tseg))
    X = np.concatenate([A1 @ S1, A2 @ S2], axis=1)  # (n, 2*Tseg)

    cfg1 = AmicaConfig(
        num_models=1, max_iter=400, num_mix_comps=3, do_newton=True, do_sphere=True, do_mean=True
    )
    r1 = Amica(cfg1, random_state=1).fit(X)

    ll1 = float(r1.log_likelihood[-1])

    segs = {0: (S1, slice(0, Tseg)), 1: (S2, slice(Tseg, 2 * Tseg))}

    cfg2 = AmicaConfig(
        num_models=2, max_iter=400, num_mix_comps=3, do_newton=True, do_sphere=True, do_mean=True
    )
    attempts = []
    accepted = None
    for seed in (0, 1, 2, 3):
        r2 = Amica(cfg2, random_state=seed).fit(X)
        ll2 = float(r2.log_likelihood[-1])

        # recovered sources per (model, segment)
        Wsens = np.asarray(r2.unmixing_matrix_sensor_)  # (M, n, n_channels)
        mean_ = np.asarray(r2.mean_)
        v = np.asarray(r2.model_posteriors_)  # (M, 2*Tseg)

        # best matched |r| of each segment's true sources to each model's sources
        match = np.zeros((2, 2))  # [model, segment]
        for h in range(2):
            srcs_h = Wsens[h] @ (X - mean_[:, None])
            for k, (Strue, sl) in segs.items():
                match[h, k] = _matched_mean_r(srcs_h[:, sl], Strue)

        # mean posterior per (model, segment)
        post = np.zeros((2, 2))
        for h in range(2):
            for k, (_S, sl) in segs.items():
                post[h, k] = v[h, sl].mean()

        best_for_seg = post.argmax(axis=0)  # which model dominates each segment
        gm = np.asarray(r2.gm_)
        ok = (
            ll2 >= ll1 - 1e-3
            and best_for_seg[0] != best_for_seg[1]
            and all(match[best_for_seg[k], k] > 0.85 for k in range(2))
            and np.allclose(np.sort(gm), [0.5, 0.5], atol=0.15)
        )
        attempts.append((seed, ll2, gm, match, post, best_for_seg, ok))
        if ok:
            accepted = attempts[-1]
            break

    for seed, ll2, gm, match, post, _best_for_seg, ok in attempts:
        print(f"\n[recovery seed={seed} ok={ok}] LL: M=2 {ll2:.4f} vs M=1 {ll1:.4f}")
        print("[recovery] gm =", np.round(gm, 3))
        print("[recovery] matched|r| [model x segment]=\n", np.round(match, 3))
        print("[recovery] mean posterior [model x segment]=\n", np.round(post, 3))

    assert accepted is not None, "no deterministic M=2 initialization recovered the two regimes"


def test_multimodel_rejection():
    """M>1 likelihood sample rejection: planted spikes are flagged in sample_mask_
    (one global mask across models, thresholding the mixture LL), and do_reject=False
    leaves the mask unset (the multimodel anchor)."""
    from jamica import Amica, AmicaConfig

    rng = np.random.RandomState(0)
    data = rng.laplace(size=(5, 1500))
    spikes = [100, 500, 900]
    data[:, spikes] *= 60.0

    common = {"num_models": 2, "max_iter": 40, "do_newton": False}
    res = Amica(
        AmicaConfig(**common, do_reject=True, rejstart=5, rejint=5, numrej=2, rejsig=3.0),
        random_state=0,
    ).fit(data)
    assert res.sample_mask_ is not None
    assert res.sample_mask_.shape == (1500,) and res.sample_mask_.dtype == bool
    assert res.n_rejected_ == int((~res.sample_mask_).sum()) > 0
    assert (~res.sample_mask_)[spikes].all()  # global mask rejects the spikes
    assert np.asarray(res.model_posteriors_).shape == (2, 1500)

    # do_reject=False leaves the mask unset (multimodel no-rejection anchor).
    res_no = Amica(AmicaConfig(**common), random_state=0).fit(data)
    assert res_no.sample_mask_ is None and res_no.n_rejected_ == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-s"]))


def test_mm_posteriors_chunked_matches_full_batch():
    """Chunking the post-fit posterior pass changes memory, not results.

    The posteriors are (M, T), but the intermediate per-model sources are
    (M, n, T), so a full-batch pass dominated peak memory on long recordings
    even when the fit itself was chunked. Each sample's posterior depends only
    on that sample, so blocks must agree with the single pass.
    """
    rng = _rng(11)
    n, J, T, M = 4, 3, 257, 3  # T deliberately not a multiple of any chunk size
    W_all = jnp.stack([_make_params(rng, n, J)[0] for _ in range(M)])
    c_all = jnp.asarray(rng.standard_normal((M, n)) * 0.05, dtype=jnp.float64)
    alpha = jnp.stack([_make_params(rng, n, J)[1] for _ in range(M)])
    muu = jnp.stack([_make_params(rng, n, J)[2] for _ in range(M)])
    beta = jnp.stack([_make_params(rng, n, J)[3] for _ in range(M)])
    rho = jnp.stack([_make_params(rng, n, J)[4] for _ in range(M)])
    gm = jnp.asarray([0.5, 0.3, 0.2], dtype=jnp.float64)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)

    full = np.asarray(
        mm.compute_model_posteriors(data, W_all, c_all, alpha, muu, beta, rho, gm, 0.0)
    )
    assert full.shape == (M, T)

    for chunk in (16, 64, 256, T, T + 10):
        chunked = np.asarray(
            mm.compute_model_posteriors(
                data, W_all, c_all, alpha, muu, beta, rho, gm, 0.0, chunk_size=chunk
            )
        )
        assert chunked.shape == full.shape, f"shape changed at chunk_size={chunk}"
        assert np.allclose(chunked, full, rtol=0, atol=ATOL), (
            f"chunk_size={chunk} disagrees with the full-batch pass"
        )


def test_mm_sample_loglik_chunked_matches_full_batch():
    """compute_model_sample_loglik is the rejection path's entry point and must
    agree with the full-batch pass at every chunk size, including its 1-D
    concatenate branch.

    Added after review noted the helper shipped with no direct test.
    """
    rng = _rng(13)
    n, J, T, M = 4, 3, 199, 2  # T prime, so no chunk size divides it
    W_all = jnp.stack([_make_params(rng, n, J)[0] for _ in range(M)])
    c_all = jnp.asarray(rng.standard_normal((M, n)) * 0.05, dtype=jnp.float64)
    alpha = jnp.stack([_make_params(rng, n, J)[1] for _ in range(M)])
    muu = jnp.stack([_make_params(rng, n, J)[2] for _ in range(M)])
    beta = jnp.stack([_make_params(rng, n, J)[3] for _ in range(M)])
    rho = jnp.stack([_make_params(rng, n, J)[4] for _ in range(M)])
    gm = jnp.asarray([0.6, 0.4], dtype=jnp.float64)
    data = jnp.asarray(rng.standard_normal((n, T)), dtype=jnp.float64)

    full = np.asarray(
        mm.compute_model_sample_loglik(data, W_all, c_all, alpha, muu, beta, rho, gm, 0.0)
    )
    assert full.shape == (T,)  # per-sample, 1-D

    for chunk in (7, 32, 128, T, T + 5):
        chunked = np.asarray(
            mm.compute_model_sample_loglik(
                data, W_all, c_all, alpha, muu, beta, rho, gm, 0.0, chunk_size=chunk
            )
        )
        assert chunked.shape == full.shape, f"shape changed at chunk_size={chunk}"
        assert np.allclose(chunked, full, rtol=0, atol=ATOL), (
            f"chunk_size={chunk} disagrees with the full-batch pass"
        )

    # it is the mixture LL: logsumexp over models of the per-model P
    assert np.all(np.isfinite(full))


def test_solver_forwards_chunk_size_to_rejection_loglik(monkeypatch):
    """The solver must forward its resolved chunk size to the rejection-path
    log-likelihood, not just to the E-step.

    Covers the wiring rather than the helper: review noted that equivalence was
    tested but the forwarding was not.
    """
    from jamica import Amica, AmicaConfig

    real_fn = mm.compute_model_sample_loglik  # capture before patching
    seen = {}

    def _spy(*args, **kwargs):
        seen["chunk_size"] = kwargs.get("chunk_size", "MISSING")
        return real_fn(*args, **kwargs)

    monkeypatch.setattr(mm, "compute_model_sample_loglik", _spy)

    rng = np.random.default_rng(5)
    data = rng.standard_normal((3, 400))
    cfg = AmicaConfig(
        num_models=2,
        num_mix_comps=2,
        max_iter=8,
        do_reject=True,
        rejstart=2,
        rejint=1,
        numrej=1,
        chunk_size=64,
        do_newton=False,
    )
    Amica(cfg, random_state=0).fit(data)

    assert "chunk_size" in seen, "rejection path never called compute_model_sample_loglik"
    assert seen["chunk_size"] == 64, (
        f"solver forwarded chunk_size={seen['chunk_size']!r}; expected 64"
    )


# ---------------------------------------------------------------------------
# Model-centre update semantics (do_mean vs update_c)
#
# ``do_mean`` controls whether preprocessing removes the GLOBAL data mean.
# Whether the EM step re-estimates each model's CENTRE is a separate question:
# for a mixture, the per-model centres are part of what distinguishes the
# models and are generally non-zero even on globally centred data.
# ---------------------------------------------------------------------------


def _two_cluster_data(rng, n=3, tseg=1500, offset=4.0):
    """Two Laplace clusters with opposite mean offsets; global mean ~ 0.

    Mimics the MNE path, which centres and whitens externally and therefore
    fits with ``do_mean=False`` / ``do_sphere=False``.
    """
    u = np.zeros(n)
    u[0] = offset
    A1 = rng.standard_normal((n, n))
    A2 = rng.standard_normal((n, n))
    X1 = A1 @ rng.laplace(size=(n, tseg)) + u[:, None]
    X2 = A2 @ rng.laplace(size=(n, tseg)) - u[:, None]
    X = np.concatenate([X1, X2], axis=1)
    X = X - X.mean(axis=1, keepdims=True)  # external centring, as MNE does
    return X


def test_multimodel_learns_distinct_centers_on_externally_centered_data():
    """M>1 must still learn per-model centres when the mean was removed outside.

    Regression test: the EM centre update used to be gated on ``do_mean``, so
    the MNE path (``do_mean=False``) froze every model centre at its zero init
    and the models could only differ by their unmixing matrix.
    """
    from jamica import Amica, AmicaConfig

    rng = _rng(0)
    X = _two_cluster_data(rng)

    cfg = AmicaConfig(
        num_models=2,
        max_iter=150,
        num_mix_comps=2,
        do_sphere=False,
        do_mean=False,
    )
    res = Amica(cfg, random_state=0).fit(X)

    c = np.asarray(res.c_)
    assert c.shape == (2, X.shape[0])
    assert np.any(np.abs(c) > 1e-6), f"all model centres frozen at init: {c}"
    sep = float(np.linalg.norm(c[0] - c[1]))
    assert sep > 1e-3, f"model centres not distinct (separation {sep:.3e})"


def test_single_model_center_frozen_when_do_mean_false():
    """M=1 with ``do_mean=False`` keeps c at its zero init.

    Guards the existing single-model contract, which the checkpoint-resume
    bit-exactness test relies on (c is not restored by ``init_params``).
    """
    from jamica import Amica, AmicaConfig

    rng = _rng(1)
    X = rng.laplace(size=(3, 1200)) + np.array([5.0, -2.0, 1.0])[:, None]

    cfg = AmicaConfig(num_models=1, max_iter=30, num_mix_comps=2, do_sphere=False, do_mean=False)
    res = Amica(cfg, random_state=0).fit(X)

    assert np.allclose(np.asarray(res.c_), 0.0), "single-model centre must stay frozen"


def test_single_model_center_tracks_data_mean_when_do_mean_true():
    """M=1 with ``do_mean=True`` leaves c ~ 0 because preprocessing centred the data."""
    from jamica import Amica, AmicaConfig

    rng = _rng(2)
    X = rng.laplace(size=(3, 1200)) + np.array([5.0, -2.0, 1.0])[:, None]

    cfg = AmicaConfig(num_models=1, max_iter=30, num_mix_comps=2, do_sphere=False, do_mean=True)
    res = Amica(cfg, random_state=0).fit(X)

    assert np.allclose(np.asarray(res.c_), 0.0, atol=1e-6)
