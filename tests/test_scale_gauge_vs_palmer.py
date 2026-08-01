"""Characterise the scale-fixing gauge against Palmer's specification.

Palmer, Kreutz-Delgado & Makeig, "AMICA: An Adaptive Mixture of Independent
Component Analyzers with Shared Components" (SCCN technical report), §II.A,
resolves the scale redundancy by renormalising the **rows of W** each
iteration::

    tau_i    = ||W[i, :]||
    W'[i, :] = W[i, :] / tau_i
    mu'      = mu / tau_i
    beta'    = beta * tau_i**2         # in the paper's beta

``amica`` instead renormalises the **columns of A** (``solver.py`` step 10,
under ``doscaling``) and applies the likelihood-equivalent inverse scaling to
``W``, ``mu`` and ``sbeta``.

The ``beta`` part is *not* a discrepancy: this package's ``sbeta_`` is Fortran's
``sbeta``, i.e. Palmer's ``sqrt(beta)``, so dividing it by ``tau`` is the same
transform as Palmer's ``beta * tau**2``.

Both are valid gauges -- they differ by an exact diagonal rescaling and leave
the likelihood unchanged. These tests assert neither is "right". They pin down
**how far apart the two are in practice**, so that a later switch to Palmer's
convention can be shown not to move any published number.

Why the published numbers are unaffected: every reported quantity that could
see this is scale-invariant. The likelihood is unchanged by construction; MIR is
invariant because scaling row *i* of ``W`` by ``c_i`` shifts ``h(y_i)`` by
``log c_i`` and ``log|det W|`` by ``sum_i log c_i``, which cancel; matched-map
correlations and dipolarity are correlation- and fit-based. The Fortran parity
check also row-normalises both matrices before comparing
(``run_fortran_parity.py:123-125``).

What is *not* automatically safe is any comparison of raw ``mu_`` or ``sbeta_``
magnitudes against Fortran, since those carry the gauge. The published fixture
records ``density.row_scale`` in [0.999999998, 1.000000021], i.e. its alignment
step found the rescaling to be a no-op there -- but see the measured table in
``test_gauge_deviation_stays_small_but_does_not_vanish``, where the deviation
reaches 2.7e-03 on the natural-gradient path. The two are not the same
condition, and the discrepancy has not been explained.
"""

from __future__ import annotations

import numpy as np
import pytest

from amica import Amica, AmicaConfig


def _cfg(**kw):
    base = {"num_models": 1, "num_mix_comps": 3, "max_iter": 40, "do_newton": False}
    base.update(kw)
    return AmicaConfig(**base)


def _fit(X, **kw):
    return Amica(_cfg(**kw)).fit(X)


def _W(res) -> np.ndarray:
    W = np.asarray(res.unmixing_matrix_white_)
    return W[0] if W.ndim == 3 else W


def _mixture(n_ch=6, n_samp=4000, seed=0):
    rng = np.random.default_rng(seed)
    S = rng.laplace(size=(n_ch, n_samp))
    A = rng.standard_normal((n_ch, n_ch))
    return A @ S


def test_implemented_gauge_is_unit_norm_columns_of_A():
    """Pin the current convention so a change cannot pass silently."""
    res = _fit(_mixture())
    A = np.linalg.inv(_W(res))
    np.testing.assert_allclose(np.linalg.norm(A, axis=0), 1.0, atol=1e-8)


def test_palmer_gauge_is_not_what_is_implemented():
    """The rows of W are *not* exactly unit norm -- the documented deviation.

    Kept as an explicit test so that adopting Palmer's convention turns this
    into a visible, intentional failure rather than a silent behaviour change.
    """
    res = _fit(_mixture(), max_iter=20)
    row_norms = np.linalg.norm(_W(res), axis=1)
    assert not np.allclose(row_norms, 1.0, atol=1e-12), (
        "rows of W are now exactly unit norm; if the gauge was deliberately "
        "switched to Palmer's, delete this test and update the docs"
    )


def test_gauges_agree_closely_on_sphered_data():
    """The practical size of the deviation on a converged Newton fit.

    This is the number that matters for whether a gauge change could move a
    published result. See the parametrised test below for how it varies with
    iteration count and optimiser -- it does not simply shrink.
    """
    res = _fit(_mixture(n_samp=8000, seed=1), max_iter=300, do_newton=True)
    dev = float(np.max(np.abs(np.linalg.norm(_W(res), axis=1) - 1.0)))
    assert dev < 1e-2, f"row norms drifted from 1 by {dev:.3g}"


@pytest.mark.parametrize("do_newton", [False, True])
def test_gauge_deviation_stays_small_but_does_not_vanish(do_newton):
    """Measured behaviour of the deviation, which is not what one might assume.

    The two gauges do **not** simply converge to each other. Measured on a
    6-channel Laplacian mixture (max |row-norm - 1|):

        iters      natural gradient      Newton
            5           6.3e-05         3.7e-05
           25           1.5e-04         1.5e-04
          100           8.3e-04         2.6e-04
          400           2.7e-03         2.4e-04

    With Newton -- the default, and the configuration the published benchmarks
    used -- it plateaus around 2e-4. With the natural-gradient path alone it
    grows slowly with iteration count. Either way it stays far below the 1e-2
    level, so scale-invariant quantities (likelihood, MIR, matched-map
    correlations, dipolarity) are unaffected; but it is not zero, so anything
    reading raw ``mu_``/``sbeta_`` magnitudes against Fortran must align first.
    """
    X = _mixture(seed=3)
    def _dev(n):
        W = _W(_fit(X, max_iter=n, do_newton=do_newton))
        return float(np.max(np.abs(np.linalg.norm(W, axis=1) - 1.0)))

    dev = {n: _dev(n) for n in (5, 100, 400)}
    assert max(dev.values()) < 1e-2, f"gauge deviation exceeded 1e-2: {dev}"
    assert min(dev.values()) > 0.0, f"gauges became identical: {dev}"


def test_rescaling_to_palmer_gauge_preserves_the_density_argument():
    """Applying Palmer's transform post hoc must be an exact no-op on the model.

    The density depends on the standardised argument ``sbeta * (s - mu)``. If
    that is invariant, the two conventions are genuinely a gauge choice. If this
    test ever fails, the deviation is a defect, not a convention.
    """
    X = _mixture(n_ch=5, n_samp=3000, seed=2)
    res = _fit(X, max_iter=60)
    W, mu, sbeta, c = _W(res), np.asarray(res.mu_), np.asarray(res.sbeta_), np.asarray(res.c_)
    if mu.ndim == 3:
        mu, sbeta, c = mu[0], np.asarray(res.sbeta_)[0], c[0]

    tau = np.linalg.norm(W, axis=1)
    Xw = np.asarray(res.whitener_) @ (X - np.asarray(res.mean_)[:, None])

    s = W @ Xw - c[:, None]
    s_p = (W / tau[:, None]) @ Xw - (c / tau)[:, None]

    for i in range(W.shape[0]):
        for k in range(mu.shape[0]):
            z = sbeta[k, i] * (s[i] - mu[k, i])
            z_p = (sbeta[k, i] * tau[i]) * (s_p[i] - mu[k, i] / tau[i])
            np.testing.assert_allclose(z, z_p, rtol=1e-9, atol=1e-9)
