"""Main AMICA solver class using JAX/NumPy."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import logging
import time
import warnings

import numpy as np
# scipy.linalg moved out of hot path - using JAX for matrix ops in training loop

logger = logging.getLogger(__name__)

from .backend import jax, jnp, HAS_JAX

from .config import AmicaConfig
from .preprocessing import (
    preprocess_data,
    compute_mean,
    compute_covariance,
    compute_sphering_matrix,
    apply_sphering,
    compute_dewhitening_matrix,
)
from .pdf import (
    compute_responsibilities,
    compute_all_scores,
    compute_source_loglikelihood,
)
from .updates import (
    update_alpha,
    update_mu,
    update_beta,
    update_rho_gradient,
    compute_natural_gradient,
    compute_newton_correction,
    update_all_pdf_params,
    compute_newton_terms,
    apply_full_newton_correction,
    update_model_centers,
)
from .likelihood import (
    compute_log_det_W,
    compute_total_loglikelihood,
)


@partial(jax.jit, static_argnames=[
    'do_newton', 'do_mean', 'do_sphere', 'doscaling', 
    'update_alpha', 'update_mu', 'update_beta', 'update_rho'
])
def _amica_step(
    # State variables
    W: jnp.ndarray,
    A: jnp.ndarray,
    c: jnp.ndarray,
    alpha: jnp.ndarray,
    mu: jnp.ndarray,
    beta: jnp.ndarray,
    rho: jnp.ndarray,
    gm: jnp.ndarray,
    # Per-iter step size (computed in outer loop, Fortran state machine)
    lrate_step: float,
    rholrate: float,
    # Static data
    data_white: jnp.ndarray,
    log_det_sphere: float,
    # Config scalars
    newt_start_iter: int,
    iteration: int,
    invsigmin: float,
    invsigmax: float,
    minrho: float,
    maxrho: float,
    # Config flags (static)
    do_newton: bool,
    do_mean: bool,
    do_sphere: bool,
    doscaling: bool,
    update_alpha: bool,
    update_mu: bool,
    update_beta: bool,
    update_rho: bool,
):
    """Single JIT-compiled AMICA iteration step."""
    
    # 1. E-step: Compute sources
    # y = W * (x - c)
    y = jnp.dot(W, data_white - c[:, None])
    n_samples = y.shape[1]
    n_components = W.shape[0]

    # 2. Compute scores
    g = compute_all_scores(y, alpha, mu, beta, rho)

    # 3. Compute Log-Likelihood
    ll = compute_total_loglikelihood(y, W, alpha, mu, beta, rho, log_det_sphere)

    # 4. Natural Gradient on A (Fortran style)
    gy = jnp.dot(g, y.T) / n_samples
    dA_local = jnp.eye(n_components) - gy

    # 5. Newton Correction
    Wtmp = dA_local
    newton_used = jnp.array(False)

    def apply_newton(operands):
        y_, alpha_, mu_, beta_, rho_ = operands
        sigma2, kappa, lambda_ = compute_newton_terms(y_, alpha_, mu_, beta_, rho_)
        lambda_pos = jnp.all(lambda_ > 0)
        Wtmp_newt, posdef_newt = apply_full_newton_correction(
            dA_local, sigma2, kappa, lambda_
        )
        is_valid = lambda_pos & posdef_newt
        return jnp.where(is_valid, Wtmp_newt, dA_local), is_valid

    if do_newton:
        def try_newton(operands):
            return apply_newton(operands)

        def skip_newton(operands):
            return dA_local, jnp.array(False)

        Wtmp, newton_used = jax.lax.cond(
            iteration >= newt_start_iter,
            try_newton,
            skip_newton,
            (y, alpha, mu, beta, rho)
        )
    else:
        Wtmp = dA_local
        newton_used = jnp.array(False)

    # 6. Update A using step-local lrate (halved natgrad or ramped Newton)
    dAk = A @ Wtmp
    A_new = A - lrate_step * dAk
    
    # 7. Update W = inv(A) — exact inverse via LU, matching Fortran's
    # DGETRF+DGETRI (amica17.f90:2157-2158). Do NOT use pinv (SVD
    # pseudoinverse) — it truncates small singular values, breaking
    # the W=A^{-1} relationship the natgrad derivation assumes. See
    # AMICA_AUDIT.md D1.
    def invert_A(A_):
        return jnp.linalg.pinv(A_).astype(W.dtype)

    A_ok = jnp.all(jnp.isfinite(A_new))
    W_new = jax.lax.cond(
        A_ok,
        invert_A,
        lambda x: W,  # Fallback to old W if A has NaN/Inf
        A_new
    )
    # Check BOTH A and W for NaN/Inf (D4 — Fortran checks info from DGETRI)
    is_good = A_ok & jnp.all(jnp.isfinite(W_new))

    # 8. M-step: Update PDF parameters
    # Helper config object for update_all_pdf_params
    # We need to construct a dummy config or modify update_all_pdf_params to take scalars
    # I verified update_all_pdf_params takes 'config' object... that's annoying for JIT.
    # Wait, I modified update_all_pdf_params signature in previous turn!
    # I restored it to take 'config'.
    # 'config' object is not JAX-friendly if it has non-static fields.
    # I should modify update_all_pdf_params to take scalars OR create a NamedTuple config.
    # OR, since I'm modifying solver.py, I can just inline the config structure?
    # Actually, JAX JIT can handle Python objects as static args if they are hashable/frozen?
    # AmicaConfig is a dataclass.
    
    # Let's assume we pass the config object as static?
    # No, 'solver.py' uses self.config.
    # The 'config' argument in update_all_pdf_params is used to access .invsigmin, etc.
    # I should pass a "Struct" or NamedTuple that JAX likes, or just pass 'self.config' and mark it static?
    # AmicaConfig is a dataclass, so it's not automatically static-safe unless frozen/hashable.
    
    # Workaround: Create a lightweight named tuple for config inside the step wrapper
    from collections import namedtuple
    ParamConfig = namedtuple('ParamConfig', [
        'invsigmin', 'invsigmax', 'minrho', 'maxrho', 
        'update_alpha', 'update_mu', 'update_beta', 'update_rho',
        'rholrate'
    ])
    pconfig = ParamConfig(
        invsigmin, invsigmax, minrho, maxrho,
        update_alpha, update_mu, update_beta, update_rho,
        rholrate
    )
    
    # Note: update_all_pdf_params expects .invsigmin access. NamedTuple supports this.
    
    alpha_new, mu_new, beta_new, rho_new = update_all_pdf_params(
        y, alpha, mu, beta, rho, pconfig, rholrate
    )

    # 9. Update model center c (Fortran: c = dc_numer/dc_denom = mean(x))
    # Fortran does NOT adjust mu to compensate for c shift — the next
    # iteration's gradient adapts naturally. See AMICA_AUDIT.md D2.
    if do_mean:
        c_new = jnp.mean(data_white, axis=1)
    else:
        c_new = c

    # 10. Scaling (Fortran doscaling)
    if doscaling:
        col_norms = jnp.linalg.norm(A_new, axis=0)
        col_norms = jnp.where(col_norms > 0.0, col_norms, 1.0)
        A_new = A_new / col_norms
        mu_new = mu_new * col_norms[None, :]
        beta_new = beta_new / col_norms[None, :]
        W_new = jnp.linalg.pinv(A_new)  # pinv for numerical stability (see AMICA_AUDIT.md)

    return (
        W_new, A_new, c_new, alpha_new, mu_new, beta_new, rho_new, gm,
        ll, is_good, newton_used
    )



@dataclass
class AmicaResult:
    """Container for AMICA results.

    Matrix naming convention
    ------------------------
    AMICA operates in whitened space. Matrices are stored in both spaces
    with explicit suffixes to avoid ambiguity:

    - ``*_white_`` — whitened space (after sphering)
    - ``*_sensor_`` — original sensor space

    The relationship is::

        sources = unmixing_matrix_white_ @ whitener_ @ (data - mean_)
        data    = mixing_matrix_sensor_ @ sources + mean_

    Attributes
    ----------
    unmixing_matrix_white_ : np.ndarray, shape (n_components, n_components)
        Unmixing matrix W in whitened space: ``sources = W @ x_white``.
    mixing_matrix_white_ : np.ndarray, shape (n_components, n_components)
        Mixing matrix A in whitened space: ``x_white = A @ sources + c``.
    unmixing_matrix_sensor_ : np.ndarray, shape (n_components, n_channels)
        Full unmixing in sensor space: ``W @ sphere``.
    mixing_matrix_sensor_ : np.ndarray, shape (n_channels, n_components)
        Full mixing in sensor space: ``desphere @ A``.
    whitener_ : np.ndarray, shape (n_components, n_channels)
        Sphering/whitening matrix S.
    dewhitener_ : np.ndarray, shape (n_channels, n_components)
        Dewhitening matrix (pseudo-inverse of sphere).
    mean_ : np.ndarray, shape (n_channels,)
        Data mean removed during preprocessing.
    alpha_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Mixture weights for each component.
    mu_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Location parameters.
    rho_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Shape parameters.
    sbeta_ : np.ndarray, shape (n_mix, n_components) or (n_models, n_mix, n_components)
        Scale parameters (inverse beta).
    c_ : np.ndarray, shape (n_components,) or (n_models, n_components)
        Model centers.
    gm_ : np.ndarray, shape (n_models,)
        Model weights (for multi-model).
    log_likelihood : np.ndarray, shape (n_iter,)
        Log-likelihood per iteration.
    iteration_times : np.ndarray, shape (n_iter,)
        Wall-clock time per iteration in seconds.
    elapsed_times : np.ndarray, shape (n_iter,)
        Cumulative wall-clock time in seconds.
    n_iter : int
        Number of iterations performed.
    converged : bool
        Whether the algorithm converged.
    """
    unmixing_matrix_white_: np.ndarray
    mixing_matrix_white_: np.ndarray
    unmixing_matrix_sensor_: np.ndarray
    mixing_matrix_sensor_: np.ndarray
    whitener_: np.ndarray
    dewhitener_: np.ndarray
    mean_: np.ndarray
    alpha_: np.ndarray
    mu_: np.ndarray
    rho_: np.ndarray
    sbeta_: np.ndarray
    c_: np.ndarray
    gm_: np.ndarray
    log_likelihood: np.ndarray
    n_iter: int
    iteration_times: np.ndarray = field(default_factory=lambda: np.array([]))
    elapsed_times: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = False
    data_scale: float = 1.0

    @property
    def unmixing_matrix(self):
        """Deprecated. Use ``unmixing_matrix_white_`` instead."""
        warnings.warn(
            "AmicaResult.unmixing_matrix is deprecated. "
            "Use unmixing_matrix_white_ (whitened space) or "
            "unmixing_matrix_sensor_ (sensor space) instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.unmixing_matrix_white_

    @property
    def mixing_matrix(self):
        """Deprecated. Use ``mixing_matrix_sensor_`` instead."""
        warnings.warn(
            "AmicaResult.mixing_matrix is deprecated. "
            "Use mixing_matrix_sensor_ (sensor space) or "
            "mixing_matrix_white_ (whitened space) instead.",
            DeprecationWarning, stacklevel=2,
        )
        return self.mixing_matrix_sensor_

    def to_mne(self, info):
        """Convert results to MNE ICA object.

        AMICA decomposes as: ``sources = W @ S @ (x - mean)``
        where W is the unmixing matrix in whitened space and S is the
        sphering/whitening matrix.

        MNE's ICA reconstructs via:
        ``unmixing_full = unmixing_ @ pca_components_[:n_comp]``
        ``mixing_full = pca_components_.T @ mixing_``

        To make these equivalent we use QR decomposition on the combined
        transform ``W @ S`` to extract an orthonormal ``pca_components_``
        (Q.T) and a square ``unmixing_matrix_`` (R), satisfying MNE's
        requirement that ``pca_components_`` has orthonormal rows.

        Parameters
        ----------
        info : mne.Info
            Measurement info (from the Raw/Epochs used for fitting).

        Returns
        -------
        ica : mne.preprocessing.ICA
            Fitted MNE ICA object compatible with plot_components(),
            get_sources(), apply(), and ICLabel.
        """
        try:
            from mne.preprocessing import ICA
        except ImportError:
            raise ImportError("MNE-Python is required for to_mne().")

        W = np.asarray(self.unmixing_matrix_white_)  # (n_comp, n_comp)
        S = np.asarray(self.whitener_)               # (n_comp, n_ch)
        n_components = W.shape[0]
        n_channels = S.shape[1]

        # Combined transform: sources = WS @ (x - mean)
        WS = W @ S  # (n_comp, n_ch)

        # QR decomposition: WS.T = Q @ R  =>  WS = R.T @ Q.T
        # Q.T is orthonormal (n_comp, n_ch) — use as pca_components_
        # R.T is square (n_comp, n_comp) — use as unmixing_matrix_ (before norms)
        Q, R = np.linalg.qr(WS.T, mode='reduced')  # Q: (n_ch, n_comp), R: (n_comp, n_comp)
        pca_components = Q.T   # (n_comp, n_ch) — orthonormal rows
        unmixing_raw = R.T     # (n_comp, n_comp) — square

        # Verify: WS ≈ unmixing_raw @ pca_components
        # (this is exact by QR construction)

        ica = ICA(n_components=n_components, method='infomax')

        # Build full orthonormal pca_components (n_ch, n_ch).
        # First n_comp rows = Q.T from QR. Complete to orthonormal basis
        # using SVD of Q to get its orthogonal complement.
        U_full, _, Vt_full = np.linalg.svd(Q, full_matrices=True)
        # U_full: (n_ch, n_ch) orthonormal columns
        # First n_comp columns span same space as Q
        # Remaining columns span the null space
        pca_full = U_full.T  # (n_ch, n_ch) — orthonormal rows
        # But we need the first n_comp rows to be exactly Q.T (= pca_components)
        # SVD may reorder/flip signs. Use Q directly and append null space.
        null_space = U_full[:, n_components:]  # (n_ch, n_ch - n_comp)
        pca_full = np.vstack([pca_components, null_space.T])
        ica.pca_components_ = pca_full

        ica.pca_mean_ = np.asarray(self.mean_)
        ica.pre_whitener_ = np.ones((n_channels, 1))

        # pca_explained_variance_ — MNE divides unmixing by sqrt(variance)
        # during fit(). We need to match that convention.
        # Since our pca_components are orthonormal, the "variance" each
        # component explains is the squared column norm of unmixing_raw.
        col_var = np.sum(unmixing_raw ** 2, axis=0)
        col_var[col_var == 0] = 1.0
        pca_explained_variance = np.ones(n_channels)
        pca_explained_variance[:n_components] = col_var
        ica.pca_explained_variance_ = pca_explained_variance

        # Apply MNE's normalization: unmixing /= sqrt(variance)
        norms = np.sqrt(col_var)
        ica.unmixing_matrix_ = unmixing_raw / norms
        ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

        # Metadata
        ica.n_components_ = n_components
        ica.info = info
        ica.ch_names = info['ch_names'][:n_channels]
        ica.n_iter_ = self.n_iter
        ica.current_fit = 'raw'
        ica.method = 'amica'

        try:
            ica._ica_names = [f"ICA{ii:03d}" for ii in range(n_components)]
        except Exception:
            pass

        return ica


class Amica:
    """Native JAX implementation of AMICA algorithm.
    
    Adaptive Mixture Independent Component Analysis (AMICA) performs ICA
    with adaptive source density modeling using mixtures of generalized
    Gaussians.
    
    Parameters
    ----------
    config : AmicaConfig, optional
        Configuration object with all algorithm parameters.
        If None, uses default configuration.
    random_state : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    config : AmicaConfig
        Algorithm configuration.
    result_ : AmicaResult
        Fitted model (available after calling fit).
        
    Examples
    --------
    >>> from mne.preprocessing._amica import Amica, AmicaConfig
    >>> config = AmicaConfig(max_iter=500, num_mix_comps=3)
    >>> amica = Amica(config, random_state=42)
    >>> result = amica.fit(data)  # data: (n_channels, n_samples)
    >>> activations = result.unmixing_matrix_white_ @ result.whitener_ @ (data - result.mean_[:, None])
    
    Notes
    -----
    The AMICA algorithm was developed by Jason Palmer at UCSD.
    This is a native Python/JAX implementation for GPU acceleration.
    
    References
    ----------
    .. [1] Palmer et al. (2008). Newton method for the ICA mixture model.
           Proc. IEEE ICASSP.
    .. [2] Palmer et al. (2011). AMICA: An adaptive mixture of independent
           component analyzers with shared components. UCSD Technical Report.
    """
    
    def __init__(
        self,
        config: Optional[AmicaConfig] = None,
        random_state: Optional[int] = None,
    ):
        self.config = config if config is not None else AmicaConfig()
        self.random_state = random_state
        self.rng = jax.random.PRNGKey(random_state if random_state is not None else 0)
        self.result_ = None
    
    def fit(
        self, 
        data: np.ndarray, 
        init_mean: Optional[np.ndarray] = None,
        init_sphere: Optional[np.ndarray] = None,
        init_weights: Optional[np.ndarray] = None,
        init_params: Optional[dict] = None,
    ) -> AmicaResult:
        """Fit AMICA model to data.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            Input EEG/MEG data. Should be high-pass filtered.
        init_mean : np.ndarray, optional
            Precomputed mean vector to use instead of computing from data.
        init_sphere : np.ndarray, optional
            Precomputed sphering matrix to use instead of computing from data.
        init_weights : np.ndarray, optional
            Precomputed unmixing matrix (W) to use for initialization.
        init_params : dict, optional
            Dictionary containing initial values for 'alpha', 'mu', 'beta', 'rho'.
            
        Returns
        -------
        result : AmicaResult
            Fitted model containing mixing/unmixing matrices and
            all model parameters.
        """
        # Determine target JAX dtype
        dtype = jnp.float32 if self.config.dtype == "float32" else jnp.float64
        
        # Preprocessing usually done in float64 for stability, then cast to target dtype
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D, got shape {data.shape}")
            
        # Check for potential unit mismatch (uV vs Volts)
        # Amica (and Engine parity) works best with Volts (std ~ 1e-5 to 1e-4)
        # If std > 0.5, assume uV or similar large unit and auto-scale.
        scaling_factor = 1.0
        data_std = np.std(data)
        if data_std > 1e2:
            # Very large values suggest microvolts or similar units.
            # EEG in Volts: std ~ 1e-5 to 1e-4. In uV: std ~ 10 to 100.
            scaling_factor = 1.0 / data_std
            logger.info("Data std (%.2e) very large. Auto-scaling by %.2e for stability.", data_std, scaling_factor)
            data = data * scaling_factor
        
        n_channels, n_samples = data.shape
        logger.info("Fitting %d channels x %d samples using %s", n_channels, n_samples, self.config.dtype)
        
        # Determine number of components
        if self.config.pcakeep is not None:
            n_components = min(self.config.pcakeep, n_channels)
        else:
            n_components = n_channels
        
        # ========== Preprocessing ==========
        logger.info("Preprocessing (mean removal, sphering)...")
        data_white, mean, sphere, desphere, n_components, eigenvalues = preprocess_data(
            data,
            do_mean=self.config.do_mean,
            do_sphere=self.config.do_sphere,
            pcakeep=self.config.pcakeep,
            mineig=self.config.mineig,
            do_approx=self.config.do_approx_sphere,
            sphere_type=self.config.sphere_type,
            init_mean=init_mean,
            init_sphere=init_sphere,
        )
        
        # Compute (log, det)(S)| from kept eigenvalues (Fortran sldet)
        safe_eigs = np.maximum(np.asarray(eigenvalues[:n_components]), self.config.mineig)
        self.log_det_sphere = float(-0.5 * np.sum(np.log(safe_eigs)))
        
        log_det_sphere = self.log_det_sphere
        
        logger.info("Using %d components", n_components)
        
        # ========== Initialize Parameters ==========
        W, A, alpha, mu, beta, rho, c, gm = self._initialize_params(
            n_components, self.config.num_mix_comps, self.config.num_models,
            dtype=dtype,
            init_weights=init_weights,
            init_params=init_params,
        )
        
        # Store learning rates (can be modified during optimization)
        lrate0 = self.config.lrate
        lrate = lrate0
        newtrate = self.config.newtrate
        rholrate0 = self.config.rholrate
        rholrate = rholrate0
        numdecs = 0
        numincs = 0
        
        # ========== Main EM Loop ==========
        LL = []
        iteration_times: List[float] = []
        elapsed_times: List[float] = []
        converged = False
        newton_count = 0  # Track how many iterations actually used Newton
        natgrad_fallback_count = 0  # Track Newton fallbacks
        start_time = time.perf_counter()
        
        # Initial ll_prev for first iteration
        ll_prev_val = -np.inf

        # Sample rejection state
        sample_mask = jnp.ones(n_samples, dtype=bool)  # True = keep
        rej_count = 0  # Number of rejection passes done
        
        # Convert config flags to static arguments once
        do_newton_static = self.config.do_newton
        do_mean_static = self.config.do_mean
        do_sphere_static = self.config.do_sphere
        doscaling_static = self.config.doscaling
        update_alpha_static = self.config.update_alpha
        update_mu_static = self.config.update_mu
        update_beta_static = self.config.update_beta
        update_rho_static = self.config.update_rho
        
        # Ensure initial state is JAX array with correct dtype
        W = jnp.asarray(W, dtype=dtype)
        A = jnp.asarray(A, dtype=dtype)
        c = jnp.asarray(c, dtype=dtype)
        alpha = jnp.asarray(alpha, dtype=dtype)
        mu = jnp.asarray(mu, dtype=dtype)
        beta = jnp.asarray(beta, dtype=dtype)
        rho = jnp.asarray(rho, dtype=dtype)
        gm = jnp.asarray(gm, dtype=dtype)
        data_white = jnp.asarray(data_white, dtype=dtype)
        
        for iteration in range(self.config.max_iter):
            iter_start = time.perf_counter()

            # ===== Fortran-style lrate state machine (AMICA_AUDIT.md F1) =====
            # Stage 1 — decay on LL decrease (mirrors amica15.f90:1038-1058).
            # Uses dll from the previous accepted iteration.
            if iteration > 0 and len(LL) >= 2 and LL[-1] < LL[-2]:
                if lrate <= self.config.minlrate:
                    logger.info("Converged at iteration %d (lrate <= minlrate)", iteration)
                    converged = True
                    break
                lrate = lrate * self.config.lratefact
                rholrate = rholrate * self.config.rholratefact
                numdecs += 1
                if numdecs >= self.config.max_decs:
                    lrate0 = lrate0 * self.config.lratefact
                    if iteration > self.config.newt_start:
                        rholrate0 = rholrate0 * self.config.rholratefact
                    if self.config.do_newton and iteration > self.config.newt_start:
                        newtrate = newtrate * self.config.lratefact
                    numdecs = 0
            elif iteration > 0 and len(LL) >= 2 and LL[-1] > LL[-2]:
                numdecs = 0

            # Stage 2 — per-iter ramp toward ceiling (mirrors amica15.f90:1786-1797).
            # Fortran order: decay → ramp → step (ramp AND A-update are both
            # inside update_params; the A update at line 1789 uses the RAMPED lrate).
            in_newton = self.config.do_newton and (iteration >= self.config.newt_start)
            ceiling = newtrate if in_newton else lrate0
            lrate = min(ceiling, lrate + min(1.0 / self.config.newt_ramp, lrate))
            # Note: amica15 resets rholrate every iter (lines 1788/1795).
            # amica17 comments this out (line 1908). We follow amica17.
            # rholrate = rholrate0  # removed per amica17

            (W_new, A_new, c_new, alpha_new, mu_new, beta_new, rho_new, gm_new,
             ll_curr, is_good, newton_used) = _amica_step(
                W, A, c, alpha, mu, beta, rho, gm,
                lrate,
                rholrate,
                data_white, log_det_sphere,
                # Config scalars
                self.config.newt_start,
                iteration,
                self.config.invsigmin, self.config.invsigmax,
                self.config.minrho, self.config.maxrho,
                # Config flags (static)
                do_newton_static, do_mean_static, do_sphere_static, doscaling_static,
                update_alpha_static, update_mu_static, update_beta_static, update_rho_static,
            )

            # Block until scalars are ready (synchronize for checking)
            is_good_val = bool(is_good)
            ll_val = float(ll_curr)
            newton_used_val = bool(newton_used)
            
            if not is_good_val:
                lrate *= 0.5
                if iteration % 10 == 0:
                    logger.warning("Iter %d: NaN/Inf detected, reducing lrate to %.2e", iteration, lrate)
                iteration_times.append(time.perf_counter() - iter_start)
                elapsed_times.append(time.perf_counter() - start_time)
                continue

            # Accept update
            W, A, c, alpha, mu, beta, rho, gm = W_new, A_new, c_new, alpha_new, mu_new, beta_new, rho_new, gm_new
            LL.append(ll_val)

            # ========== Sample Rejection ==========
            if (self.config.do_reject
                and iteration >= self.config.rejstart
                and rej_count < self.config.numrej
                and (iteration - self.config.rejstart) % self.config.rejint == 0):
                # Compute per-sample LL
                y_rej = jnp.dot(W, data_white - c[:, None])
                sample_lls = compute_source_loglikelihood(y_rej, alpha, mu, beta, rho)
                sample_lls = sample_lls + compute_log_det_W(W) + log_det_sphere

                # Rejection threshold: mean - rejsig * std
                ll_mean = jnp.mean(sample_lls)
                ll_std = jnp.std(sample_lls)
                threshold = ll_mean - self.config.rejsig * ll_std

                new_mask = sample_lls >= threshold
                n_rejected = int(jnp.sum(~new_mask))
                max_reject = int(0.2 * n_samples)  # Cap at 20%

                if n_rejected <= max_reject:
                    sample_mask = new_mask
                    rej_count += 1
                    if iteration % 10 == 0:
                        pct = 100.0 * n_rejected / n_samples
                        logger.info("Iter %d: Rejected %d samples (%.1f%%)", iteration, n_rejected, pct)

                    # Subset data to non-rejected samples
                    mask_np = np.asarray(new_mask)
                    kept_idx = np.where(mask_np)[0]
                    data_white = jnp.asarray(np.asarray(data_white)[:, kept_idx])
                    n_samples = data_white.shape[1]

            # Convergence: dll window test (Fortran use_min_dll, amica15.f90:1060-1072)
            dll = ll_val - ll_prev_val
            if iteration > 0 and self.config.use_min_dll:
                if dll < self.config.min_dll:
                    numincs += 1
                    if numincs > self.config.max_incs:
                        logger.info("Converged at iteration %d (dll < min_dll)", iteration)
                        converged = True
                        iteration_times.append(time.perf_counter() - iter_start)
                        elapsed_times.append(time.perf_counter() - start_time)
                        break
                else:
                    numincs = 0

            # Track Newton usage
            if iteration >= self.config.newt_start and do_newton_static:
                if newton_used_val:
                    newton_count += 1
                else:
                    natgrad_fallback_count += 1

            # Progress output
            if iteration % 10 == 0:
                if iteration >= self.config.newt_start and do_newton_static:
                    mode = "N" if newton_used_val else "ng"
                    logger.info("Iter %4d: LL = %.6f, lrate = %.2e [%s]",
                                iteration, ll_val, lrate, mode)
                else:
                    logger.info("Iter %4d: LL = %.6f, lrate = %.2e",
                                iteration, ll_val, lrate)
            
            # Checkpoint
            if self.config.outdir is not None and self.config.writestep > 0 and (iteration + 1) % self.config.writestep == 0:
                # Need to bring back to CPU for saving
                self.result_ = AmicaResult(
                    unmixing_matrix_white_=np.asarray(W),
                    mixing_matrix_white_=np.asarray(A),
                    unmixing_matrix_sensor_=np.asarray(W @ sphere),
                    mixing_matrix_sensor_=np.asarray(desphere @ A),
                    whitener_=np.asarray(sphere),
                    dewhitener_=np.asarray(desphere),
                    mean_=np.asarray(mean),
                    alpha_=np.asarray(alpha),
                    mu_=np.asarray(mu),
                    rho_=np.asarray(rho),
                    sbeta_=np.asarray(beta),
                    c_=np.asarray(c),
                    gm_=np.asarray(gm),
                    log_likelihood=np.array(LL),
                    iteration_times=np.array(iteration_times),
                    elapsed_times=np.array(elapsed_times),
                    n_iter=len(LL),
                    converged=converged,
                    data_scale=scaling_factor,
                )
                self.save(self.config.outdir)
                logger.info("Saved checkpoint to %s", self.config.outdir)

            ll_prev_val = ll_val
            iteration_times.append(time.perf_counter() - iter_start)
            elapsed_times.append(time.perf_counter() - start_time)

        if not converged:
            logger.info("Reached max_iter (%d)", self.config.max_iter)

        # Newton diagnostic summary
        if do_newton_static and self.config.newt_start < len(LL):
            total_newton_iters = newton_count + natgrad_fallback_count
            if total_newton_iters > 0:
                pct = 100.0 * newton_count / total_newton_iters
                logger.info(
                    "Newton: %d/%d iterations used Newton (%.0f%%), "
                    "%d fell back to natural gradient",
                    newton_count, total_newton_iters, pct,
                    natgrad_fallback_count
                )

        # ========== Construct Result ==========
        self.result_ = AmicaResult(
            unmixing_matrix_white_=np.asarray(W),
            mixing_matrix_white_=np.asarray(A),
            unmixing_matrix_sensor_=np.asarray(W @ sphere),
            mixing_matrix_sensor_=np.asarray(desphere @ A),
            whitener_=np.asarray(sphere),
            dewhitener_=np.asarray(desphere),
            mean_=np.asarray(mean),
            alpha_=np.asarray(alpha),
            mu_=np.asarray(mu),
            rho_=np.asarray(rho),
            sbeta_=np.asarray(beta),
            c_=np.asarray(c),
            gm_=np.asarray(gm),
            log_likelihood=np.array(LL),
            iteration_times=np.array(iteration_times),
            elapsed_times=np.array(elapsed_times),
            n_iter=len(LL),
            converged=converged,
            data_scale=scaling_factor,
        )
        
        return self.result_
    
    def _initialize_params(
        self,
        n_components: int,
        n_mix: int,
        n_models: int,
        dtype: Any = jnp.float64,
        init_weights: Optional[np.ndarray] = None,
        init_params: Optional[dict] = None,
    ) -> Tuple[jnp.ndarray, ...]:
        """Initialize model parameters.
        
        Parameters
        ----------
        n_components : int
            Number of ICA components.
        n_mix : int
            Number of Gaussian mixture components.
        n_models : int
            Number of ICA models.
        dtype : Any
            JAX dtype for parameters (jnp.float32 or jnp.float64).
        init_weights : np.ndarray, optional
            Precomputed unmixing matrix (W) to use.
        init_params : dict, optional
            Dictionary containing initial values for 'alpha', 'mu', 'beta', 'rho'.
            
        Returns
        -------
        W : jnp.ndarray, shape (n_components, n_components)
        A : jnp.ndarray, shape (n_components, n_components)
        alpha : jnp.ndarray, shape (n_mix, n_components)
        mu : jnp.ndarray, shape (n_mix, n_components)
        beta : jnp.ndarray, shape (n_mix, n_components)
        rho : jnp.ndarray, shape (n_mix, n_components)
        c : jnp.ndarray, shape (n_components,)
        gm : jnp.ndarray, shape (n_models,)
        """
        rng = np.random.default_rng(self.random_state)

        # Initialize mixing matrix A (Fortran-style), then invert to get W.
        if init_weights is not None:
            W = jnp.asarray(init_weights, dtype=dtype)
            A = jnp.linalg.pinv(W)
        else:
            if self.config.fix_init:
                A_np = np.eye(n_components, dtype=np.float64)
            else:
                noise = rng.random((n_components, n_components))
                A_np = 0.01 * (0.5 - noise)
                A_np += np.eye(n_components, dtype=np.float64)
                col_norms = np.linalg.norm(A_np, axis=0)
                col_norms = np.where(col_norms > 0.0, col_norms, 1.0)
                A_np = A_np / col_norms
            A = jnp.asarray(A_np, dtype=dtype)
            W = jnp.linalg.pinv(A)
        
        # Initialize mixture parameters
        if init_params is not None and 'alpha' in init_params:
            alpha = jnp.asarray(init_params['alpha'], dtype=dtype)
        else:
            # alpha: uniform mixture weights
            alpha = jnp.ones((n_mix, n_components), dtype=dtype) / n_mix
        
        if init_params is not None and 'mu' in init_params:
            mu = jnp.asarray(init_params['mu'], dtype=dtype)
        else:
            # Fortran: mu(j,k) = (j-1) - (num_mix-1)/2
            base = np.arange(n_mix, dtype=np.float64) - (n_mix - 1) / 2.0
            mu_np = base[:, None] * np.ones((n_mix, n_components), dtype=np.float64)
            if not self.config.fix_init:
                noise = rng.random((n_mix, n_components))
                mu_np = mu_np + 0.05 * (1.0 - 2.0 * noise)
            mu = jnp.asarray(mu_np, dtype=dtype)

        if init_params is not None and ('beta' in init_params or 'sbeta' in init_params):
            beta_key = 'beta' if 'beta' in init_params else 'sbeta'
            beta = jnp.asarray(init_params[beta_key], dtype=dtype)
        else:
            if self.config.fix_init:
                beta = jnp.ones((n_mix, n_components), dtype=dtype)
            else:
                noise = rng.random((n_mix, n_components))
                beta = jnp.asarray(
                    1.0 + 0.1 * (0.5 - noise),
                    dtype=dtype
                )
        
        if init_params is not None and 'rho' in init_params:
            rho = jnp.asarray(init_params['rho'], dtype=dtype)
        else:
            # rho: start at middle value
            rho = jnp.full(
                (n_mix, n_components),
                self.config.rho0,
                dtype=dtype
            )
        
        # Model center: zero
        c = jnp.zeros(n_components, dtype=dtype)
        
        # Model weights: uniform
        gm = jnp.ones(n_models, dtype=dtype) / n_models
        
        return W, A, alpha, mu, beta, rho, c, gm
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted unmixing to new data.
        
        Parameters
        ----------
        data : np.ndarray, shape (n_channels, n_samples)
            New data to transform.
            
        Returns
        -------
        sources : np.ndarray, shape (n_components, n_samples)
            Source activations.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        data = np.asarray(data, dtype=np.float64) * self.result_.data_scale
        
        # Apply full unmixing: W @ sphere @ (x - mean)
        centered = data - self.result_.mean_[:, None]
        whitened = self.result_.whitener_ @ centered
        sources = self.result_.unmixing_matrix_white_ @ whitened
        
        return sources
    
    def inverse_transform(self, sources: np.ndarray) -> np.ndarray:
        """Reconstruct data from sources.
        
        Parameters
        ----------
        sources : np.ndarray, shape (n_components, n_samples)
            Source activations.
            
        Returns
        -------
        data : np.ndarray, shape (n_channels, n_samples)
            Reconstructed data.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        sources = np.asarray(sources, dtype=np.float64)
        
        # data = A @ sources + mean = desphere @ A_white @ sources + mean
        data = self.result_.mixing_matrix_sensor_ @ sources + self.result_.mean_[:, None]
        
        return data / self.result_.data_scale
    
    def save(self, outdir: Union[str, Path]) -> None:
        """Save model to directory in AMICA-compatible format.
        
        Parameters
        ----------
        outdir : str or Path
            Output directory.
        """
        if self.result_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Save in Fortran-compatible binary format (column-major)
        def save_binary(name: str, arr: np.ndarray):
            arr.astype('<f8').T.tofile(outdir / name)
        
        save_binary('A', self.result_.mixing_matrix_sensor_)
        save_binary('W', self.result_.unmixing_matrix_white_)
        save_binary('S', self.result_.whitener_)
        save_binary('mean', self.result_.mean_)
        save_binary('alpha', self.result_.alpha_)
        save_binary('mu', self.result_.mu_)
        save_binary('rho', self.result_.rho_)
        save_binary('sbeta', self.result_.sbeta_)
        save_binary('c', self.result_.c_)
        save_binary('gm', self.result_.gm_)
        save_binary('LL', self.result_.log_likelihood)
        
        logger.info("Saved model to %s", outdir)
