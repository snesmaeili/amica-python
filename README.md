# amica-python

Python implementation of AMICA (Adaptive Mixture ICA) with optional JAX acceleration. Designed for EEG/MEG preprocessing with MNE-Python.

AMICA models source distributions as mixtures of generalized Gaussians and uses Newton optimization for fast convergence. It ranked first among 22 ICA algorithms on EEG data in terms of mutual information reduction and dipole fit quality (Delorme et al., 2012).

## Installation

```bash
git clone https://github.com/snesmaeili/amica-python.git
cd amica-python
pip install -e ".[all]"
```

Or install specific extras:

```bash
pip install -e ".[jax]"   # JAX backend
pip install -e ".[mne]"   # MNE-Python integration
pip install -e ".[dev]"   # testing
```

## Usage

```python
from amica_python import Amica, AmicaConfig

config = AmicaConfig(max_iter=2000, num_mix_comps=3)
model = Amica(config, random_state=42)
result = model.fit(data)  # (n_channels, n_samples)
sources = model.transform(data)
```

With MNE-Python:

```python
from amica_python import fit_ica

ica = fit_ica(raw, n_components=20, max_iter=2000)
ica.plot_components()
ica.apply(raw)
```

Picard-style functional API:

```python
from amica_python import amica

W, n_iter = amica(X, return_n_iter=True)  # X: (n_samples, n_components)
```

## What AMICA does differently

Standard ICA algorithms (Infomax, FastICA, Picard) assume fixed source distributions. AMICA learns them: each source gets a mixture of generalized Gaussians with adaptive shape (rho), scale (beta), and location (mu) parameters. The shape parameter ranges from 1 (Laplacian, super-Gaussian) to 2 (Gaussian), so each component adapts to the actual statistics of the data.

The Newton method (Palmer et al., 2008) gives quadratic convergence once the natural gradient phase gets close to the solution. Sample rejection (Klug et al., 2024) downweights outlier time points to improve robustness on noisy data.

## Validation

Numerical parity with MATLAB AMICA 1.7 has been verified:

| Configuration | LL difference | W correlation |
| --- | --- | --- |
| 1 model, Newton | 0.0002% | > 0.9999 |
| 3 mixtures, Newton | 0.00008% | > 0.9999 |
| 3 mixtures, natural gradient | 0.08% | > 0.9995 |

On synthetic sources (3 channels, 5000 samples, Laplacian), the Amari index is 0.008 compared to 0.010 for FastICA and 0.195 for Infomax.

## Parameters

Defaults follow the recommendations in Frank et al. (2023) and Klug et al. (2024):

| Parameter | Default | What it does |
| --- | --- | --- |
| `max_iter` | 2000 | EM iterations |
| `num_mix_comps` | 3 | Gaussian mixture components per source |
| `do_newton` | True | Newton optimization after iteration 50 |
| `do_reject` | False | Outlier sample rejection |
| `rejsig` | 3.0 | Rejection threshold (standard deviations) |
| `rho0` | 1.5 | Initial shape (between 1=Laplacian and 2=Gaussian) |

## References

Palmer, J.A., Kreutz-Delgado, K., & Makeig, S. (2011). AMICA: An Adaptive Mixture of Independent Component Analyzers with Shared Components. SCCN Technical Report.

Palmer, J.A., Makeig, S., Kreutz-Delgado, K., & Rao, B.D. (2008). Newton Method for the ICA Mixture Model. Proc. ICASSP.

Delorme, A., Palmer, J., Onton, J., Oostenveld, R., & Makeig, S. (2012). Independent EEG Sources Are Dipolar. PLoS ONE, 7(2), e30135.

Frank, R.M., Heppner, A., & Borghese, F. (2023). Optimal Parameters for AMICA. IEEE BIBE.

Klug, M., Berg, J., & Gramann, K. (2024). Optimizing EEG independent component analysis. Scientific Reports.

## License

BSD-3-Clause
