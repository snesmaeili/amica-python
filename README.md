<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/snesmaeili/jamica/main/docs/_static/logo-dark.png?v=3">
    <img src="https://raw.githubusercontent.com/snesmaeili/jamica/main/docs/_static/logo.png?v=3" alt="jamica - Adaptive Mixture Independent Component Analysis, powered by JAX" width="420">
  </picture>
</p>

<p align="center"><strong>JAX-accelerated Adaptive Mixture Independent Component Analysis for Python.</strong></p>

[![CI](https://img.shields.io/github/actions/workflow/status/snesmaeili/jamica/tests.yml?branch=main&label=CI)](https://github.com/snesmaeili/jamica/actions/workflows/tests.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/snesmaeili/jamica/docs.yml?branch=main&label=docs)](https://snesmaeili.github.io/jamica/)
[![Codecov](https://img.shields.io/codecov/c/github/snesmaeili/jamica)](https://codecov.io/gh/snesmaeili/jamica)
[![PyPI - Version](https://img.shields.io/pypi/v/jamica.svg)](https://pypi.org/project/jamica/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/jamica.svg)](https://anaconda.org/conda-forge/jamica)
[![Python Versions](https://img.shields.io/pypi/pyversions/jamica.svg)](https://pypi.org/project/jamica/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21817485.svg)](https://doi.org/10.5281/zenodo.21817485)

jamica is a JAX implementation of **AMICA** (Adaptive Mixture Independent Component Analysis), an algorithm for blind source separation. It is aimed mainly at EEG.

AMICA is one of the strongest methods in the ICA family for EEG decomposition. The original implementation is a Fortran program from UCSD, usually run through MATLAB or EEGLAB. jamica rewrites it in Python on top of JAX, so the same code JIT-compiles and runs on either CPU or GPU, and it works directly with **MNE-Python**.

______________________________________________________________________

# Why jamica?

JAX + AMICA = jamica. The name also describes the problem AMICA solves.

Record a jam session with a few microphones. Each one picks up a different blend of the same players. Getting the individual instruments back out of those recordings is blind source separation, which is what ICA does for EEG: electrodes pick up mixtures of cortical, muscular and ocular activity, and the job is to pull them apart again.

A jam is rarely one fixed mixture, though. Players drop in and out. Someone takes a solo. The statistics of what the microphones hear keep shifting. AMICA handles this by fitting several mixture models instead of one, and by learning the shape of each source distribution rather than assuming it. That makes it a good match for data a single stationary ICA model does not describe well.

jamica runs that algorithm on JAX, on CPU or GPU, inside the usual Python scientific stack.

> **Status:** jamica reproduces the Fortran AMICA 1.7 reference on the tested single-model configurations. Validation scope, the exact reference build used, and known limitations are described under [Validation](#validation).

______________________________________________________________________

# Highlights

- The AMICA algorithm in Python, JIT-compiled through **JAX**
- Runs on CPU or GPU without changing your code
- Numerical agreement with the Fortran AMICA 1.7 reference on the tested configurations
- Native integration with **MNE-Python**
- Support for **multi-model AMICA**
- Modern scientific Python API
- Extensive testing and continuous integration
- Fully open source (BSD-3-Clause)

______________________________________________________________________

# Installation

```bash
pip install "jamica[jax]"
```

or, from conda-forge:

```bash
conda install -c conda-forge jamica jax
```

> **Renamed from `amica`.** Releases up to 0.1.0 were published as `amica`.
> That name installed a top-level `amica` module, which collided with
> `amica-python` — an independent implementation of the same algorithm by
> another author — so the two could not coexist in one environment. Since
> 0.2.0 this project installs as `jamica`, and the two can be installed side
> by side.

For NVIDIA GPUs, take the CUDA build of JAX instead:

```bash
pip install "jamica[gpu]"        # JAX with CUDA 12 (Linux only)
```

The other extras are separate, so you only install what you need:

```bash
pip install "jamica[mne]"        # MNE-Python integration, fit_ica()
pip install "jamica[icalabel]"   # ICLabel component classification
pip install "jamica[viz]"        # plotting and density diagnostics
pip install "jamica[all]"        # everything above
```

## From source

For development, or to run the test suite:

```bash
git clone https://github.com/snesmaeili/jamica.git
cd jamica
pip install -e ".[dev]"
```

With `uv`:

```bash
git clone https://github.com/snesmaeili/jamica.git
cd jamica
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

______________________________________________________________________

# Quick Start

```python
from jamica import Amica, AmicaConfig

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
from jamica import fit_ica

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

**https://snesmaeili.github.io/jamica/**

______________________________________________________________________

# Validation

jamica has been validated against the **Fortran AMICA 1.7** reference implementation.

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

Agreement across devices is close in aggregate, but component-level agreement is not guaranteed on
every recording: fits that reach the same likelihood can still differ in individual component
subspaces. Check component identity if you move an analysis between CPU and GPU part-way through.

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

If jamica contributes to your research, please cite the original AMICA publications.

Citation metadata is available in
[CITATION.cff](CITATION.cff).

______________________________________________________________________

# License

jamica is distributed under the terms of the BSD 3-Clause License.
