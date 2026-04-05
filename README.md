# amica-python

Native Python implementation of AMICA (Adaptive Mixture Independent Component Analysis) for MNE-Python.

AMICA is the highest-performing ICA algorithm for EEG/MEG data, achieving the best mutual information reduction and source dipolarity among 22 tested algorithms ([Delorme et al., 2012](https://doi.org/10.1371/journal.pone.0030135)).

## Installation

```bash
pip install amica-python

# With JAX acceleration (recommended)
pip install "amica-python[jax]"

# With MNE-Python integration
pip install "amica-python[mne]"

# Everything
pip install "amica-python[all]"
```

For development:

```bash
git clone https://github.com/snesmaeili/amica-python.git
cd amica-python
pip install -e ".[dev]"
```

## Quick Start

### Standalone

```python
from amica_python import Amica, AmicaConfig

config = AmicaConfig(max_iter=2000, num_mix_comps=3)
model = Amica(config, random_state=42)
result = model.fit(data)  # data: (n_channels, n_samples)

sources = model.transform(data)
```

### With MNE-Python

```python
from amica_python import amica

# Picard-compatible functional API
W, n_iter = amica(X, return_n_iter=True, max_iter=2000)
```

### Planned: MNE ICA Integration

```python
from mne.preprocessing import ICA
ica = ICA(n_components=20, method='amica')
ica.fit(raw)
```

## Features

- Full AMICA algorithm: ICA mixture model with adaptive generalized Gaussian source densities
- Newton optimization with quadratic convergence
- EM-based parameter updates with GEM convergence guarantee
- Sample rejection for artifact robustness
- JAX acceleration (GPU/TPU) with NumPy fallback
- scikit-learn-compatible API

## Default Parameters

| Parameter | Default | Reference |
|-----------|---------|-----------|
| `max_iter` | 2000 | Frank et al. (2023) |
| `num_mix_comps` | 3 | Frank et al. (2023) |
| `do_newton` | True | Palmer et al. (2008) |
| `newt_start` | 50 | Palmer et al. (2008) |
| `rejstart` | 2 | Klug et al. (2024) |
| `rejint` | 3 | Klug et al. (2024) |
| `rejsig` | 3.0 | Klug et al. (2024) |

## References

- Palmer, Kreutz-Delgado, Makeig (2011). AMICA: An Adaptive Mixture of Independent Component Analyzers with Shared Components. SCCN Technical Report.
- Palmer, Makeig, Kreutz-Delgado, Rao (2008). Newton Method for the ICA Mixture Model. ICASSP.
- Palmer, Kreutz-Delgado, Makeig (2006). Super-Gaussian Mixture Source Model for ICA. ICA.
- Delorme et al. (2012). Independent EEG Sources Are Dipolar. PLoS ONE.
- Frank et al. (2023). Optimal Parameters for AMICA. IEEE BIBE.
- Klug, Berg, Gramann (2024). Optimizing EEG ICA decomposition. Scientific Reports.

## License

BSD-3-Clause
