# Understanding jamica

jamica is a native Python implementation of **Adaptive Mixture Independent Component Analysis (AMICA)**, an ICA algorithm originally developed at the University of California, San Diego (UCSD) for blind source separation of EEG and other multichannel signals.

Unlike many ICA algorithms that assume a single statistical model for all sources, AMICA models each source using a **mixture of generalized Gaussian distributions** and can optionally fit **multiple ICA models** to account for non-stationary recordings.

______________________________________________________________________

## Independent Component Analysis

Independent Component Analysis (ICA) seeks to recover statistically independent latent sources from observed mixtures.

Given an observed signal

```text
X = AS
```

where

- X is the observed data,
- A is an unknown mixing matrix,
- S contains the latent independent sources,

ICA estimates an unmixing matrix

```text
W ≈ A⁻¹
```

such that

```text
S = WX.
```

In EEG, ICA is commonly used to separate brain activity from artifacts such as eye blinks, muscle activity, and cardiac signals.

______________________________________________________________________

## What makes AMICA different?

Most ICA algorithms assume that each source follows a fixed probability distribution.

AMICA instead models each source as a mixture of generalized Gaussian distributions, allowing it to adapt to a much wider range of source characteristics.

Key features include:

- adaptive source density estimation
- natural-gradient optimization
- optional Newton optimization for faster convergence
- optional outlier rejection
- support for multiple ICA models

These capabilities have made AMICA one of the strongest-performing ICA methods for EEG source separation.

______________________________________________________________________

## jamica

jamica brings AMICA into the modern scientific Python ecosystem.

It provides:

- a fully native Python implementation
- optional JAX acceleration for CPU and GPU execution
- integration with MNE-Python
- reproducible APIs for research workflows
- compatibility with the broader Scientific Python ecosystem

The goal is not only to reproduce the original algorithm but also to make it easier to understand, extend, and integrate into modern neuroimaging pipelines.

______________________________________________________________________

## Choosing an interface

jamica provides two primary interfaces.

### MNE-Python interface

For EEG analysis with MNE, use:

```python
from jamica import fit_ica
```

This returns a standard `mne.preprocessing.ICA` object that integrates directly with existing MNE workflows.

### Native jamica interface

For NumPy arrays or custom pipelines, use:

```python
from jamica import Amica, AmicaConfig
```

This provides direct access to the AMICA algorithm and all configuration options.

______________________________________________________________________

## Numerical validation

jamica reproduces the Fortran AMICA 1.7 reference on single-model fits. Final log-likelihoods, unmixing matrices and adaptive-density parameters agree closely across `K=1` and `K=3` density terms, under both Newton and natural-gradient updates, and on a 100-iteration audit on real EEG.

### Reproducing the parity numbers

The reference was a **patched** AMICA 1.7. Stock 1.7 does not converge on these fixtures; three corrections were required, including a generalized-Gaussian score exponent fix. Comparing against an unpatched upstream build will not reproduce these numbers, so start from the patched source and its Docker build in [`jamica-benchmark/fortran/`](https://github.com/snesmaeili/jamica-benchmark/tree/main/fortran).

### What the parity fixtures cover

They cover single-model fits. Multi-model agreement with Fortran, long high-dimensional runs, and likelihood-based sample rejection are not measured against the reference build; rejection follows the reference procedure, but its equivalence was not verified numerically.

Agreement across devices is close in aggregate rather than component by component. Two fits reaching the same likelihood can still differ in individual component subspaces, so check component identity if you move an analysis between CPU and GPU part-way through.

Protocols, cross-implementation comparisons, and the manuscript's figures and tables are in [jamica-benchmark](https://github.com/snesmaeili/jamica-benchmark).

______________________________________________________________________

## References

- Palmer JA, Kreutz-Delgado K, Makeig S. AMICA: An Adaptive Mixture of Independent Component Analyzers with Shared Components. 2011.
- Palmer JA, Makeig S, Kreutz-Delgado K, Rao BD. Newton Method for the ICA Mixture Model. ICASSP, 2008.
- Delorme A, et al. Independent EEG Sources Are Dipolar. PLOS ONE, 2012.
