# Frequently Asked Questions

## Is this the same project as `amica-python`?

No.

`amica-python` is an independent implementation of the AMICA algorithm by a different author. Both projects install a top-level `amica` module, so only one can be present in an environment at a time: the conda-forge packages are declared mutually exclusive, and on PyPI installing both overwrites files in each.

This documentation describes the distribution published as [`amica`](https://pypi.org/project/amica/). To check which one is installed:

```bash
pip show amica amica-python
```

If behaviour does not match what you read here, confirm this first — an `import amica` that succeeds does not by itself tell you which implementation you are running.

______________________________________________________________________

## Which Python versions are supported?

amica supports **Python 3.10 and newer**.

______________________________________________________________________

## Does amica require JAX?

No.

amica runs out of the box using the NumPy backend. Installing the `jax` extra enables hardware acceleration on supported systems.

```bash
pip install "amica[jax]"
```

______________________________________________________________________

## Does amica require a GPU?

No.

The NumPy backend runs on any machine. If JAX with GPU support is installed, amica will automatically use the GPU.

______________________________________________________________________

## Can I use amica with MNE-Python?

Yes.

amica provides a high-level `fit_ica` function that returns a standard `mne.preprocessing.ICA` object, allowing you to use the full MNE visualization and artifact-rejection workflow.

______________________________________________________________________

## What input format does amica expect?

The core API expects a NumPy array with shape

```text
(n_channels, n_samples)
```

When using the MNE interface, simply pass an `mne.io.Raw` object.

______________________________________________________________________

## Can amica fit multiple ICA models?

Yes.

AMICA supports fitting multiple ICA models (`num_models > 1`) to capture non-stationary data. See the examples and API documentation for details.

______________________________________________________________________

## How does amica compare with the reference AMICA implementation?

amica is designed to reproduce the original AMICA algorithm while providing a native Python implementation, optional JAX acceleration, and seamless integration with the scientific Python ecosystem.

Validation experiments comparing amica with the Fortran AMICA 1.7 reference are available in the documentation. Agreement is established for single-model configurations against a locally patched build of the reference; multi-model parity is not covered.

______________________________________________________________________

## Where can I find examples?

See the **Examples** section of the documentation for:

- MNE-Python integration
- Pure NumPy/JAX workflows
- Validation examples
- HPC/Slurm execution

______________________________________________________________________

## I found a bug. Where should I report it?

Please open an issue on GitHub:

**https://github.com/snesmaeili/amica/issues**

When possible, include:

- your operating system
- Python version
- amica version
- backend (NumPy or JAX)
- a minimal reproducible example

______________________________________________________________________

## Can I contribute?

Absolutely!

Please read the [Contributing Guide](contributing.md) before opening a pull request.
