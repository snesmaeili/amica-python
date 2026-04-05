"""Data preprocessing functions for AMICA using JAX/NumPy."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .backend import jax, jnp, optional_jit


@jax.jit
def compute_mean(data: jnp.ndarray) -> jnp.ndarray:
    """Compute data mean across time (samples).
    
    Parameters
    ----------
    data : jnp.ndarray, shape (n_channels, n_samples)
        Input data matrix.
        
    Returns
    -------
    mean : jnp.ndarray, shape (n_channels,)
        Mean value per channel.
    """
    return jnp.mean(data, axis=1)


@jax.jit
def compute_covariance(data: jnp.ndarray, mean: jnp.ndarray) -> jnp.ndarray:
    """Compute data covariance matrix.
    
    Parameters
    ----------
    data : jnp.ndarray, shape (n_channels, n_samples)
        Input data matrix.
    mean : jnp.ndarray, shape (n_channels,)
        Data mean per channel.
        
    Returns
    -------
    cov : jnp.ndarray, shape (n_channels, n_channels)
        Covariance matrix.
    """
    centered = data - mean[:, None]
    n_samples = data.shape[1]
    return jnp.dot(centered, centered.T) / n_samples


def compute_sphering_matrix(
    cov: jnp.ndarray,
    pcakeep: int = None,
    mineig: float = 1e-12,
    do_approx: bool = True,
    sphere_type: str = "pca",
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Compute whitening/sphering matrix via eigendecomposition.
    
    The sphering matrix S transforms data such that:
    cov(S @ data) = I
    
    Parameters
    ----------
    cov : jnp.ndarray, shape (n_channels, n_channels)
        Data covariance matrix.
    pcakeep : int, optional
        Number of components to keep. If None, keeps all above mineig.
    mineig : float
        Minimum eigenvalue threshold. Default is 1e-12.
    do_approx : bool
        If True, use approximate sphering (orthogonal). Default is True.
        
    Returns
    -------
    sphere : jnp.ndarray, shape (n_keep, n_channels)
        Sphering/whitening matrix.
    eigenvalues : jnp.ndarray, shape (n_channels,)
        Eigenvalues of covariance (sorted descending).
    n_keep : int
        Number of components kept.
    """
    # Use scipy for more robust eigendecomposition
    import scipy.linalg as sla

    cov_np = np.asarray(cov)

    # Ensure symmetric (Fortran uses DSYEV on symmetric covariance)
    cov_np = (cov_np + cov_np.T) / 2.0

    try:
        eigenvalues, eigenvectors = sla.eigh(cov_np)
    except np.linalg.LinAlgError:
        # Fallback to svd-based approach
        U, s, _ = sla.svd(cov_np, full_matrices=False)
        eigenvalues = s
        eigenvectors = U
    
    # Convert back to jnp
    eigenvalues = jnp.asarray(eigenvalues)
    eigenvectors = jnp.asarray(eigenvectors)
    
    # Reverse to descending order
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    # Ensure positive eigenvalues
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    
    # Determine number of components to keep
    n_channels = cov.shape[0]
    n_above_thresh = int(jnp.sum(eigenvalues > mineig))
    
    if pcakeep is None:
        n_keep = n_above_thresh
    else:
        n_keep = min(pcakeep, n_above_thresh)
    
    if n_keep < 1:
        raise ValueError(f"No eigenvalues above threshold {mineig}. Max eigenvalue: {jnp.max(eigenvalues)}")
    
    # Compute sphering matrix following Fortran AMICA logic.
    eigenvalues_kept = np.asarray(eigenvalues[:n_keep])
    eigenvectors_kept = np.asarray(eigenvectors[:, :n_keep])

    scaling = 1.0 / np.sqrt(np.maximum(eigenvalues_kept, mineig))
    
    # Standard PCA Sphering: S = D^(-1/2) * V^T
    S_pca = np.diag(scaling) @ eigenvectors_kept.T
    
    if sphere_type == "zca":
         # Symmetric/Polar Sphering
         # If n_keep < n_channels, Fortran uses an orthogonal rotation of the 
         # PCA components to align them with the first n_keep sensors.
         # Logic: orth = UV^T from SVD of V[0:n_keep, 0:n_keep]
         # S = orth @ S_pca
         
         # V_kept is (n_keep, n_channels).
         # We take the top-left square block (n_keep, n_keep).
         # Note: eigenvectors_kept is (n_channels, n_keep) [Columns are eigenvectors]
         # So V^T (rows are eigenvectors) is eigenvectors_kept.T
         # V_block = eigenvectors_kept.T[:, :n_keep] ? No, top-left of V.
         # eigenvectors is (channels, components).
         # Fortran V is (components, channels).
         # V_block in Fortran is V(1:k, 1:k).
         # Equivalent to eigenvectors_kept.T[:n_keep, :n_keep]?
         # eigenvectors_kept.T is (n_keep, n_channels).
         # So yes, slicing columns 0..k.
         
         v_block = eigenvectors_kept.T[:, :n_keep]
         u, _, vt = sla.svd(v_block, full_matrices=False)
         orth = u @ vt # Fortran uses vt.T @ u.T ?
         # Fortran: call dgesvd( 'A', 'A', ... v(1:n,1:n) ... ) -> s, u, vt
         # orth = matmul(transpose(vt), transpose(u))
         # orth = V * U^T = (U * V^T)^T ?
         # If A = U S V^T. Polar Q = U V^T.
         # Fortran computes Transpose(U V^T)? Or something.
         # Let's trust my previous transcription or check logic.
         # Previous code: orth = vt.T @ u.T.
         # Scipy svd: A = U @ S @ Vt.
         # If Fortran gets U, Vt from Lapack.
         # Fortran SVD of A returns A = U * S * VT.
         # Fortran code: orth = VT^T * U^T = V * U^T.
         # Scipy: A = U S Vt. (Vt is V^T).
         # So orth = Vt.T @ U.T.
         # This matches Fortran.
         
         orth = vt.T @ u.T
         sphere_np = orth @ S_pca
         
         # Note: If n_keep == n_channels, this converges to standard ZCA (E @ S_pca)?
         # Only if eigenvectors is symmetric?
         # S_pca is (n, n). E is (n, n). v_block is E^T.
         # SVD of E^T -> U, S, Vt. E^T is orthogonal. S=I.
         # E^T = U Vt.
         # Orth = Vt.T @ U.T = (U Vt)^T = E.
         # sphere_np = E @ S_pca. YES.
         # So this logic works for both cases.
         
    else:
         # Default PCA: S = D^(-1/2) * V^T
         sphere_np = S_pca

    sphere = jnp.asarray(sphere_np)
    return sphere, jnp.asarray(eigenvalues), n_keep


@jax.jit
def apply_sphering(
    data: jnp.ndarray,
    mean: jnp.ndarray,
    sphere: jnp.ndarray,
) -> jnp.ndarray:
    """Apply mean removal and sphering to data.
    
    Parameters
    ----------
    data : jnp.ndarray, shape (n_channels, n_samples)
        Input data matrix.
    mean : jnp.ndarray, shape (n_channels,)
        Data mean per channel.
    sphere : jnp.ndarray, shape (n_components, n_channels)
        Sphering matrix.
        
    Returns
    -------
    data_white : jnp.ndarray, shape (n_components, n_samples)
        Whitened data.
    """
    centered = data - mean[:, None]
    return jnp.dot(sphere, centered)


def compute_dewhitening_matrix(
    sphere: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    n_keep: int,
) -> jnp.ndarray:
    """Compute dewhitening matrix (pseudo-inverse of sphere).
    
    Parameters
    ----------
    sphere : jnp.ndarray, shape (n_keep, n_channels)
        Sphering matrix.
    eigenvalues : jnp.ndarray, shape (n_channels,)
        Eigenvalues from sphering computation.
    n_keep : int
        Number of components kept.
        
    Returns
    -------
    desphere : jnp.ndarray, shape (n_channels, n_keep)
        Dewhitening matrix (pseudo-inverse of sphere).
    """
    return jnp.linalg.pinv(sphere, rcond=1e-12)


@jax.jit
def reject_outliers(
    data: jnp.ndarray,
    threshold: float = 3.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Identify outlier samples based on deviation from mean.
    
    Parameters
    ----------
    data : jnp.ndarray, shape (n_channels, n_samples)
        Input data.
    threshold : float
        Number of standard deviations for rejection. Default is 3.0.
        
    Returns
    -------
    good_mask : jnp.ndarray, shape (n_samples,)
        Boolean mask of good (non-outlier) samples.
    outlier_indices : jnp.ndarray
        Indices of rejected samples.
    """
    # Compute sample-wise statistics
    sample_norms = jnp.linalg.norm(data, axis=0)
    mean_norm = jnp.mean(sample_norms)
    std_norm = jnp.std(sample_norms)
    
    # Mark samples within threshold as good
    good_mask = jnp.abs(sample_norms - mean_norm) <= threshold * std_norm
    outlier_indices = jnp.where(~good_mask)[0]
    
    return good_mask, outlier_indices


def preprocess_data(
    data: np.ndarray,
    do_mean: bool = True,
    do_sphere: bool = True,
    pcakeep: int = None,
    mineig: float = 1e-12,
    do_approx: bool = True,
    sphere_type: str = "pca",
    init_mean: np.ndarray = None,
    init_sphere: np.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """Full preprocessing pipeline for AMICA."""
    # ... docstring ... lines 234-290 omitted ...
    
    data = jnp.asarray(data, dtype=jnp.float64)
    n_channels, n_samples = data.shape
    
    # Compute mean
    if init_mean is not None:
        mean = jnp.asarray(init_mean, dtype=jnp.float64)
    elif do_mean:
        mean = compute_mean(data)
    else:
        mean = jnp.zeros(n_channels, dtype=jnp.float64)
    
    # Compute sphering
    if init_sphere is not None:
        sphere = jnp.asarray(init_sphere, dtype=jnp.float64)
        n_components = sphere.shape[0]
        
        # When using injected sphere, we still compute eigenvalues from covariance
        # to support log-likelihood calculation, ensuring compatibility.
        if do_sphere:
            cov = compute_covariance(data, mean)
            _, eigenvalues, _ = compute_sphering_matrix(
                cov, pcakeep, mineig, do_approx, sphere_type=sphere_type
            )
            eigenvalues_kept = eigenvalues[:n_components]
        else:
            eigenvalues_kept = jnp.ones(n_components, dtype=jnp.float64)
            
        desphere = compute_dewhitening_matrix(sphere, eigenvalues_kept, n_components)

    elif do_sphere:
        cov = compute_covariance(data, mean)
        sphere, eigenvalues, n_components = compute_sphering_matrix(
            cov, pcakeep=pcakeep, mineig=mineig, do_approx=do_approx, sphere_type=sphere_type
        )

        desphere = compute_dewhitening_matrix(sphere, eigenvalues, n_components)
        eigenvalues_kept = eigenvalues[:n_components]
    else:
        if pcakeep is not None:
            n_components = min(pcakeep, n_channels)
        else:
            n_components = n_channels
        sphere = jnp.eye(n_components, n_channels, dtype=jnp.float64)
        desphere = jnp.eye(n_channels, n_components, dtype=jnp.float64)
        eigenvalues_kept = jnp.ones(n_components, dtype=jnp.float64)
    
    # Apply preprocessing
    data_white = apply_sphering(data, mean, sphere)
    
    return data_white, mean, sphere, desphere, n_components, eigenvalues_kept

