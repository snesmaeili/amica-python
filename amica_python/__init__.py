"""amica-python: Native Python AMICA for MNE-Python.

Adaptive Mixture Independent Component Analysis (AMICA) with JAX
acceleration. Designed as a drop-in ICA method for MNE-Python,
following the Picard integration pattern.

References
----------
.. [1] Palmer, J.A., Kreutz-Delgado, K., & Makeig, S. (2012).
       AMICA: An Adaptive Mixture of Independent Component Analyzers
       with Shared Components. Technical Report, UCSD.

.. [2] Palmer, J.A., Makeig, S., Kreutz-Delgado, K., & Rao, B.D.
       (2008). Newton Method for the ICA Mixture Model. ICASSP 2008.
"""

from .config import AmicaConfig
from .solver import Amica, AmicaResult

__all__ = ["Amica", "AmicaConfig", "AmicaResult", "amica", "fit_ica", "viz", "metrics"]


def fit_ica(inst, n_components=None, max_iter=2000, num_mix=3,
            random_state=None, picks=None, reject=None, flat=None,
            decim=None, fit_params=None, verbose=None, **kwargs):
    """Fit ICA using AMICA on MNE Raw or Epochs data.

    See :func:`amica_python.mne_integration.fit_ica` for full docs.
    """
    from .mne_integration import fit_ica as _fit_ica
    return _fit_ica(
        inst, n_components=n_components, max_iter=max_iter,
        num_mix=num_mix, random_state=random_state, picks=picks,
        reject=reject, flat=flat, decim=decim, fit_params=fit_params,
        verbose=verbose, **kwargs,
    )
__version__ = "0.1.0"


def amica(X, n_components=None, whiten=False, return_n_iter=False,
          random_state=None, max_iter=2000, num_mix=3, **kwargs):
    """Adaptive Mixture ICA (AMICA).

    Compatible with MNE-Python's ``ICA(method='amica')``.
    Follows the Picard integration pattern.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_components)
        Pre-whitened data (MNE convention: samples x components).
    n_components : int or None
        Number of components. If None, uses X.shape[1].
    whiten : bool
        If True, whiten the data. MNE passes False (pre-whitened).
    return_n_iter : bool
        If True, return (W, n_iter) tuple.
    random_state : int or None
        Random seed for reproducibility.
    max_iter : int
        Maximum number of EM iterations.
    num_mix : int
        Number of generalized Gaussian mixture components per source.
    **kwargs
        Additional parameters passed to AmicaConfig.

    Returns
    -------
    W : ndarray, shape (n_components, n_components)
        Unmixing matrix.
    n_iter : int
        Number of iterations (only if return_n_iter=True).
    """
    import numpy as np

    # random_state is passed to Amica solver, which uses jax.random.PRNGKey

    n_comp = n_components or X.shape[1]

    # Build config from kwargs
    cfg_kwargs = {
        "max_iter": max_iter,
        "num_mix_comps": num_mix,
        "do_sphere": whiten,
        "do_mean": whiten,
    }
    cfg_kwargs.update(kwargs)
    config = AmicaConfig(**cfg_kwargs)

    # AMICA expects (n_channels, n_samples), MNE passes (n_samples, n_components)
    data = X.T  # (n_components, n_samples)

    solver = Amica(config, random_state=random_state)
    result = solver.fit(data)

    W = result.unmixing_matrix_white_  # (n_components, n_components)

    if return_n_iter:
        return W, result.n_iter
    return W
