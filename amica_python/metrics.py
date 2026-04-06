"""Per-component metrics unique to AMICA's mixture model.

These functions extract statistics from AMICA's generalized Gaussian
mixture parameters that are not available from standard ICA methods.
"""
from __future__ import annotations

import numpy as np

from .solver import AmicaResult


def rho_mean(result: AmicaResult) -> np.ndarray:
    """Mean shape parameter per component across mixture components.

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.

    Returns
    -------
    rho_mean : ndarray, shape (n_components,)
        Weighted mean of rho across mixture components.
        rho=1 is Laplacian (super-Gaussian), rho=2 is Gaussian.
    """
    alpha = np.asarray(result.alpha_)
    rho = np.asarray(result.rho_)
    # Handle multi-model: take first model
    if alpha.ndim == 3:
        alpha, rho = alpha[0], rho[0]
    return np.sum(alpha * rho, axis=0)


def rho_range(result: AmicaResult) -> np.ndarray:
    """Range of shape parameter per component across mixture components.

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.

    Returns
    -------
    rho_range : ndarray, shape (n_components,)
        max(rho) - min(rho) across mixtures for each component.
        Large range indicates multimodal source density.
    """
    rho = np.asarray(result.rho_)
    if rho.ndim == 3:
        rho = rho[0]
    return np.ptp(rho, axis=0)


def mixture_entropy(result: AmicaResult) -> np.ndarray:
    """Entropy of mixture weights per component.

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.

    Returns
    -------
    entropy : ndarray, shape (n_components,)
        Shannon entropy of alpha (mixture weights) in nats.
        Higher entropy means more uniform mixture (more complex density).
        Zero entropy means single dominant mixture component.
    """
    alpha = np.asarray(result.alpha_)
    if alpha.ndim == 3:
        alpha = alpha[0]
    # Clip to avoid log(0)
    alpha_safe = np.clip(alpha, 1e-15, None)
    return -np.sum(alpha_safe * np.log(alpha_safe), axis=0)


def multimodality_flag(
    result: AmicaResult, threshold: float = 0.5
) -> np.ndarray:
    """Flag components whose source density is likely multimodal.

    A component is flagged if its mixture entropy exceeds ``threshold``
    times the maximum possible entropy (uniform weights).

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.
    threshold : float
        Fraction of max entropy above which a component is flagged.

    Returns
    -------
    is_multimodal : ndarray, shape (n_components,), dtype=bool
    """
    alpha = np.asarray(result.alpha_)
    if alpha.ndim == 3:
        alpha = alpha[0]
    n_mix = alpha.shape[0]
    max_entropy = np.log(n_mix)
    ent = mixture_entropy(result)
    return ent > threshold * max_entropy


def source_kurtosis(result: AmicaResult, data: np.ndarray) -> np.ndarray:
    """Excess kurtosis of each source activation.

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.
    data : ndarray, shape (n_channels, n_samples)
        Data in sensor space (same as was fitted).

    Returns
    -------
    kurtosis : ndarray, shape (n_components,)
        Excess kurtosis per component (0 for Gaussian).
    """
    from scipy.stats import kurtosis as _kurt

    data = np.asarray(data, dtype=np.float64)
    centered = data - result.mean_[:, None]
    whitened = result.whitener_ @ centered
    sources = result.unmixing_matrix_white_ @ whitened
    return _kurt(sources, axis=1, fisher=True)
