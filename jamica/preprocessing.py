"""Data preprocessing functions for AMICA using JAX/NumPy."""

from __future__ import annotations

import numpy as np

from .backend import jax, jnp


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
    pcakeep: int | None = None,
    mineig: float = 1e-12,
    do_approx: bool = True,
    sphere_type: str = "zca",
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
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
    sphere_type : str
        Type of sphering: "pca" or "zca". Default is "zca".

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

    # Guard against zero-variance channels (flat channels)
    if np.any(np.diag(cov_np) <= mineig):
        raise ValueError(
            "Covariance matrix contains zero or near-zero variance channels. "
            "Please remove flat channels before fitting AMICA."
        )

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
    n_above_thresh = int(jnp.sum(eigenvalues > mineig))

    n_keep = n_above_thresh if pcakeep is None else min(pcakeep, n_above_thresh)

    if n_keep < 1:
        raise ValueError(
            f"No eigenvalues above threshold {mineig}. Max eigenvalue: {jnp.max(eigenvalues)}"
        )

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
        v_block = eigenvectors_kept.T[:, :n_keep]
        u, _, vt = sla.svd(v_block, full_matrices=False)

        # Fortran equivalence: orth = vt.T @ u.T
        orth = vt.T @ u.T
        sphere_np = orth @ S_pca
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


def preprocess_data(
    data: np.ndarray,
    do_mean: bool = True,
    do_sphere: bool = True,
    pcakeep: int | None = None,
    mineig: float = 1e-12,
    do_approx: bool = True,
    sphere_type: str = "pca",
    init_mean: np.ndarray | None = None,
    init_sphere: np.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, jnp.ndarray]:
    """Full preprocessing pipeline for AMICA.

    Performs mean removal and sphering (whitening/PCA/ZCA) on input data.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Input data matrix.
    do_mean : bool
        Whether to remove the channel-wise mean.
    do_sphere : bool
        Whether to compute and apply the sphering matrix.
    pcakeep : int or None
        Number of PCA components to keep. If None, uses all components above mineig.
    mineig : float
        Minimum eigenvalue threshold for PCA.
    do_approx : bool
        Whether to use approximate sphering (discard small eigenvalues).
    sphere_type : str
        Type of sphering: "pca" or "zca".
    init_mean : np.ndarray or None
        Pre-computed mean vector. If provided, overrides do_mean logic.
    init_sphere : np.ndarray or None
        Pre-computed sphering matrix. If provided, overrides do_sphere logic.

    Returns
    -------
    data_white : jnp.ndarray, shape (n_components, n_samples)
        Preprocessed and whitened data.
    mean : jnp.ndarray, shape (n_channels,)
        Removed mean vector.
    sphere : jnp.ndarray, shape (n_components, n_channels)
        Sphering matrix.
    desphere : jnp.ndarray, shape (n_channels, n_components)
        Dewhitening matrix.
    n_components : int
        Number of components retained.
    eigenvalues_kept : jnp.ndarray, shape (n_components,)
        Eigenvalues corresponding to the retained components.
    """

    data = jnp.asarray(data, dtype=jnp.float64)
    n_channels, _n_samples = data.shape

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
        n_components = min(pcakeep, n_channels) if pcakeep is not None else n_channels
        sphere = jnp.eye(n_components, n_channels, dtype=jnp.float64)
        desphere = jnp.eye(n_channels, n_components, dtype=jnp.float64)
        eigenvalues_kept = jnp.ones(n_components, dtype=jnp.float64)

    # Apply preprocessing
    data_white = apply_sphering(data, mean, sphere)

    return data_white, mean, sphere, desphere, n_components, eigenvalues_kept
