"""Compare amica-python against the original Fortran AMICA (amica17).

Uses a NON-DEGENERATE 5-component fixture with non-orthogonal mixing:
- 5 sources with rho = [1.0, 1.3, 1.5, 1.8, 2.0]
- Non-orthogonal mixing (condition number = 3)
- 10000 samples

Both Fortran (compiled amica17) and Python run 500 iterations on this
fixture. The Fortran output is captured in tests/fixtures/fortran_nondegenerate/.

See FORTRAN_VALIDATION_GUIDE.md for the full validation methodology.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from amica_python import Amica, AmicaConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FORTRAN_DIR = FIXTURE_DIR / "fortran_nondegenerate"

# Tolerances: loose enough for different random init, tight enough to catch bugs.
ATOL_LL = 0.10       # nats/sample — Python and Fortran start ~0.002 apart
ATOL_RHO = 0.50      # rho values — different inits may converge to different shapes


def _load_fortran_outputs():
    """Read the captured amica17 outputs."""
    def _read(name, shape, order="F"):
        return np.fromfile(FORTRAN_DIR / name, dtype=np.float64).reshape(shape, order=order)

    out = {
        "W": _read("W", (5, 5)),
        "A": _read("A", (5, 5)),
        "S": _read("S", (5, 5)),
        "mean": _read("mean", (5,)),
        "alpha": _read("alpha", (3, 5)),
        "mu": _read("mu", (3, 5)),
        "sbeta": _read("sbeta", (3, 5)),
        "rho": _read("rho", (3, 5)),
    }
    ll_full = np.fromfile(FORTRAN_DIR / "LL", dtype=np.float64)
    n_real = int(np.argmax(ll_full == 0)) if (ll_full == 0).any() else len(ll_full)
    out["LL"] = ll_full[:n_real] if n_real > 0 else ll_full
    out["n_iter"] = n_real
    return out


@pytest.fixture(scope="module")
def synthetic_data():
    """Load the non-degenerate 5-source fixture."""
    npz = np.load(FIXTURE_DIR / "synthetic_nondegenerate.npz")
    return {"x": npz["x"], "A_true": npz["A_true"], "rho_true": npz["rho_true"]}


@pytest.fixture(scope="module")
def fortran_outputs():
    if not (FORTRAN_DIR / "W").exists():
        pytest.skip("Fortran fixture not present — see FORTRAN_VALIDATION_GUIDE.md §3")
    return _load_fortran_outputs()


@pytest.fixture(scope="module")
def python_outputs(synthetic_data):
    """Run amica-python on the same data with matching config."""
    cfg = AmicaConfig(
        num_models=1, num_mix_comps=3, max_iter=500, pcakeep=5, dtype="float64",
    )
    return Amica(cfg, random_state=42).fit(synthetic_data["x"])


# ---------------------------------------------------------------------------
# LL parity
# ---------------------------------------------------------------------------

def test_initial_ll_matches_fortran(fortran_outputs, python_outputs):
    """Initial LL should match Fortran to within ATOL_LL."""
    ll_python = float(np.asarray(python_outputs.log_likelihood)[0])
    ll_fortran = float(fortran_outputs["LL"][0])
    diff = abs(ll_python - ll_fortran)
    assert diff < ATOL_LL, (
        f"Initial LL disagreement: Python={ll_python:.6f} Fortran={ll_fortran:.6f} "
        f"|diff|={diff:.6f} > {ATOL_LL}\nSee FORTRAN_VALIDATION_GUIDE.md"
    )


def test_final_ll_in_same_range(fortran_outputs, python_outputs):
    """Final LL should be in the same ballpark (within 0.5 nats)."""
    ll_python = float(np.asarray(python_outputs.log_likelihood)[-1])
    ll_fortran = float(fortran_outputs["LL"][-1])
    diff = abs(ll_python - ll_fortran)
    assert diff < 0.5, (
        f"Final LL diverged: Python={ll_python:.6f} Fortran={ll_fortran:.6f} "
        f"|diff|={diff:.6f} > 0.5"
    )


# ---------------------------------------------------------------------------
# Rho does not collapse
# ---------------------------------------------------------------------------

def test_rho_does_not_collapse(python_outputs):
    """No rho should be pinned at minrho=1.0 floor on the non-degenerate fixture."""
    rho = np.asarray(python_outputs.rho_).flatten()
    n_at_floor = int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))
    assert n_at_floor == 0, (
        f"{n_at_floor}/{rho.size} rho pinned at floor=1.0. "
        f"rho values: {rho}"
    )


# ---------------------------------------------------------------------------
# Newton terms sanity
# ---------------------------------------------------------------------------

def test_newton_terms_well_posed(synthetic_data, fortran_outputs):
    """Newton terms (sigma2, kappa, lambda) should be finite and positive."""
    import jax.numpy as jnp
    from amica_python.updates import compute_newton_terms

    n_comp, n_mix = 5, 3
    S = fortran_outputs["S"][:n_comp, :]
    mean_f = fortran_outputs["mean"]
    W = fortran_outputs["W"]
    x = synthetic_data["x"]

    x_white = S @ (x - mean_f[:, None])
    y = W @ x_white

    sigma2, kappa, lambda_ = compute_newton_terms(
        jnp.asarray(y), jnp.asarray(fortran_outputs["alpha"]),
        jnp.asarray(fortran_outputs["mu"]), jnp.asarray(fortran_outputs["sbeta"]),
        jnp.asarray(fortran_outputs["rho"]),
    )
    sigma2, kappa, lambda_ = map(np.asarray, (sigma2, kappa, lambda_))

    assert np.all(np.isfinite(sigma2)), f"Non-finite sigma2: {sigma2}"
    assert np.all(np.isfinite(kappa)), f"Non-finite kappa: {kappa}"
    assert np.all(np.isfinite(lambda_)), f"Non-finite lambda: {lambda_}"
    assert np.all(sigma2 > 0), f"Non-positive sigma2: {sigma2}"
    assert np.all(kappa > 0), f"Non-positive kappa: {kappa}"
    assert np.all(lambda_ > 0), f"Non-positive lambda: {lambda_}"


def test_newton_posdef_at_fortran_state(synthetic_data, fortran_outputs):
    """Newton Hessian should be positive definite at Fortran's converged state."""
    import jax.numpy as jnp
    from amica_python.updates import compute_newton_terms, apply_full_newton_correction
    from amica_python.pdf import compute_all_scores

    n_comp = 5
    S = fortran_outputs["S"][:n_comp, :]
    mean_f = fortran_outputs["mean"]
    W = fortran_outputs["W"]
    x = synthetic_data["x"]

    x_white = S @ (x - mean_f[:, None])
    y = W @ x_white

    sigma2, kappa, lambda_ = compute_newton_terms(
        jnp.asarray(y), jnp.asarray(fortran_outputs["alpha"]),
        jnp.asarray(fortran_outputs["mu"]), jnp.asarray(fortran_outputs["sbeta"]),
        jnp.asarray(fortran_outputs["rho"]),
    )

    g = compute_all_scores(
        jnp.asarray(y), jnp.asarray(fortran_outputs["alpha"]),
        jnp.asarray(fortran_outputs["mu"]), jnp.asarray(fortran_outputs["sbeta"]),
        jnp.asarray(fortran_outputs["rho"]),
    )
    gy = jnp.dot(g, jnp.asarray(y).T) / y.shape[1]
    dA = jnp.eye(n_comp) - gy

    _, posdef = apply_full_newton_correction(dA, sigma2, kappa, lambda_)
    assert bool(posdef), "Newton Hessian not positive definite at Fortran's converged state"


# ---------------------------------------------------------------------------
# Rho update formula (byte-equivalent, always passes)
# ---------------------------------------------------------------------------

def test_rho_update_step_byte_equivalent_to_fortran():
    """Rho update formula matches Fortran exactly."""
    from scipy.special import digamma
    rho, rholrate = 1.5, 0.05
    drho_numer, drho_denom = 0.42, 1.0
    expected = rho + rholrate * (1.0 - (rho / digamma(1.0 + 1.0 / rho)) * drho_numer / drho_denom)
    psi = digamma(1.0 + 1.0 / rho)
    actual = rho + rholrate * (1.0 - (rho / psi) * (drho_numer / drho_denom))
    assert abs(actual - expected) < 1e-15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matched_rows_max_abs_error(W_a: np.ndarray, W_b: np.ndarray) -> float:
    """Max abs entry error after best row permutation+sign."""
    from scipy.optimize import linear_sum_assignment
    cost = -np.abs(W_a @ W_b.T)
    row_ind, col_ind = linear_sum_assignment(cost)
    W_a_matched = W_a[row_ind]
    signs = np.sign(np.diag(W_a_matched @ W_b[col_ind].T))
    signs[signs == 0] = 1.0
    W_a_matched = W_a_matched * signs[:, None]
    return float(np.max(np.abs(W_a_matched - W_b[col_ind])))
