# amica

[![CI](https://img.shields.io/github/actions/workflow/status/snesmaeili/amica/tests.yml?branch=main&label=CI)](https://github.com/snesmaeili/amica/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/snesmaeili/amica/docs.yml?branch=main&label=docs)](https://snesmaeili.github.io/amica/)
[![Codecov](https://img.shields.io/codecov/c/github/snesmaeili/amica)](https://codecov.io/gh/snesmaeili/amica)
[![PyPI - Version](https://img.shields.io/pypi/v/amica.svg)](https://pypi.org/project/amica/)
[![Python Versions](https://img.shields.io/pypi/pyversions/amica.svg)](https://pypi.org/project/amica/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

> **amica** is a native Python implementation of **AMICA (Adaptive Mixture Independent Component Analysis)**, one of the highest-performing ICA algorithms for EEG source separation.

The canonical implementation is a Fortran program from UCSD, typically driven through MATLAB- or EEGLAB-based workflows. amica provides an open, extensible Python implementation with optional **JAX acceleration**, seamless **MNE-Python integration**, and a modern Python API for reproducible neuroimaging workflows.

> **Status:** amica reproduces the Fortran AMICA 1.7 reference on the tested single-model configurations. Validation scope, the exact reference build used, and known limitations are described under [Validation](#validation).

______________________________________________________________________

# Highlights

- Native Python implementation of the AMICA algorithm
- Numerical agreement with the Fortran AMICA 1.7 reference on the tested configurations
- Optional **JAX** backend for CPU and GPU acceleration
- Native integration with **MNE-Python**
- Support for **multi-model AMICA**
- Modern scientific Python API
- Extensive testing and continuous integration
- Fully open source (BSD-3-Clause)

______________________________________________________________________

# Installation

```bash
pip install amica
```

The core install depends only on NumPy and SciPy. Everything else is an optional
extra, so a CPU-only NumPy install stays small:

```bash
pip install "amica[jax]"        # JAX backend, JIT-compiled CPU
pip install "amica[gpu]"        # JAX with CUDA 12 (Linux only)
pip install "amica[mne]"        # MNE-Python integration, fit_ica()
pip install "amica[icalabel]"   # ICLabel component classification
pip install "amica[viz]"        # plotting and density diagnostics
pip install "amica[all]"        # everything above
```

## From source

For development, or to run the test suite:

```bash
git clone https://github.com/snesmaeili/amica.git
cd amica
pip install -e ".[dev]"
```

With `uv`:

```bash
git clone https://github.com/snesmaeili/amica.git
cd amica
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

______________________________________________________________________

# Quick Start

```python
from amica import Amica, AmicaConfig

config = AmicaConfig(
    max_iter=2000,
    num_mix_comps=3,
)

model = Amica(config, random_state=42)

result = model.fit(data)

sources = model.transform(data)
```

For MNE-Python:

```python
from amica import fit_ica

ica = fit_ica(raw)

ica.plot_components()
ica.apply(raw)
```

______________________________________________________________________

# Examples

Example scripts are available in the `examples/` directory, including:

- MNE-Python integration
- Native AMICA API
- JAX acceleration
- Multi-model AMICA
- HPC / SLURM execution

______________________________________________________________________

# Documentation

Full documentation, API reference, validation experiments, and tutorials are available at

**https://snesmaeili.github.io/amica/**

______________________________________________________________________

# Validation

amica has been validated against the **Fortran AMICA 1.7** reference implementation.

Scope of that validation, stated precisely so it is not over-read:

- **Single-model fits.** Six-channel Laplacian fixtures with `K=1` and `K=3` adaptive-density terms,
  under Newton and natural-gradient updates, plus a 100-iteration audit on real EEG. Final
  log-likelihoods, unmixing matrices and adaptive-density parameters agree closely.
- **The reference was a locally patched build.** Stock AMICA 1.7 does not converge on these fixtures;
  three corrections were required, including a generalized-Gaussian score exponent fix. The patched
  source and build recipe are included in the validation archive accompanying the manuscript; that
  archive is not yet deposited, so the patch is not currently redistributable from this repository.
  Comparisons against an unpatched upstream build will not reproduce these numbers.
- **Not covered by the parity fixtures:** multi-model agreement with Fortran, long high-dimensional
  optimisation runs, and likelihood-based sample rejection. Rejection follows the reference procedure
  but its equivalence was not measured against the reference build.

Backend agreement (JAX-GPU / JAX-CPU / NumPy-CPU) is close in aggregate, but component-level agreement
is not guaranteed on every recording: fits that reach the same likelihood can still differ in
individual component subspaces. Check component identity if you switch backends mid-analysis.

The documentation contains:

- validation experiments
- numerical parity analyses
- performance benchmarks
- reproducibility instructions

______________________________________________________________________

# Contributing

Contributions are welcome!

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

______________________________________________________________________

# Citation

If amica contributes to your research, please cite the original AMICA publications.

Citation metadata is available in
[CITATION.cff](CITATION.cff).

______________________________________________________________________

# License

amica is distributed under the terms of the BSD 3-Clause License.
