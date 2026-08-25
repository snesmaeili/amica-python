"""jamica: Native Python AMICA for MNE-Python.

Adaptive Mixture Independent Component Analysis (AMICA) [1]_ [2]_ with JAX
acceleration. Designed as a native ICA method for MNE-Python
(via ``mne.preprocessing.ICA``).

References
----------
.. [1] Palmer, J.A., Kreutz-Delgado, K., & Makeig, S. (2012).
       AMICA: An Adaptive Mixture of Independent Component Analyzers
       with Shared Components. Technical Report, UCSD.

.. [2] Palmer, J.A., Makeig, S., Kreutz-Delgado, K., & Rao, B.D.
       (2008). Newton Method for the ICA Mixture Model. ICASSP 2008.
"""

from .amica_ica import AmicaICA, read_amica_ica
from .config import AmicaConfig
from .metrics import (
    mixture_entropy,
    multimodality_flag,
    rho_mean,
    rho_range,
    source_kurtosis,
)
from .mne_integration import fit_ica, get_model_ica
from .solver import Amica, AmicaResult, amica
from .viz import (
    plot_component_metrics,
    plot_convergence,
    plot_mixture_weights,
    plot_model_responsibilities,
    plot_parameter_summary,
    plot_shape_parameters,
    plot_source_densities,
)

__all__ = [
    "Amica",
    "AmicaConfig",
    "AmicaICA",
    "AmicaResult",
    "amica",
    "fit_ica",
    "get_model_ica",
    "mixture_entropy",
    "multimodality_flag",
    "plot_component_metrics",
    "plot_convergence",
    "plot_mixture_weights",
    "plot_model_responsibilities",
    "plot_parameter_summary",
    "plot_shape_parameters",
    "plot_source_densities",
    "read_amica_ica",
    "rho_mean",
    "rho_range",
    "source_kurtosis",
]
