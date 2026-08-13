# Frequently Asked Questions

## What happened to the `amica` package on PyPI?

This project was published as `amica` up to version 0.1.0. Since 0.2.0 it is published as [`jamica`](https://pypi.org/project/jamica/) and installs a top-level `jamica` module.

```bash
pip install jamica
```

The old name installed a top-level `amica` module, which collided with `amica-python` (see below), so the two could not be installed together. The rename removes that conflict. The exported names are unchanged — `from jamica import amica, Amica, AmicaConfig` works exactly as `from amica import ...` did.

______________________________________________________________________

## Is this the same project as `amica-python`?

No.

`amica-python` is an independent implementation of the AMICA algorithm by a different author. Before the rename both projects installed a top-level `amica` module, so only one could be present in an environment at a time. Since 0.2.0 they no longer share a module name and both can be installed together.

______________________________________________________________________

## Which Python versions are supported?

jamica supports **Python 3.10 and newer**.

______________________________________________________________________

## Does jamica require JAX?

No.

jamica runs out of the box using the NumPy backend. Installing the `jax` extra enables hardware acceleration on supported systems.

```bash
pip install "jamica[jax]"
```

______________________________________________________________________

## Does jamica require a GPU?

No.

The NumPy backend runs on any machine. If JAX with GPU support is installed, jamica will automatically use the GPU.

______________________________________________________________________

## Can I use jamica with MNE-Python?

Yes.

jamica provides a high-level `fit_ica` function that returns a standard `mne.preprocessing.ICA` object, allowing you to use the full MNE visualization and artifact-rejection workflow.

______________________________________________________________________

## What input format does jamica expect?

The core API expects a NumPy array with shape

```text
(n_channels, n_samples)
```

When using the MNE interface, simply pass an `mne.io.Raw` object.

______________________________________________________________________

## Can jamica fit multiple ICA models?

Yes.

AMICA supports fitting multiple ICA models (`num_models > 1`) to capture non-stationary data. See the examples and API documentation for details.

______________________________________________________________________

## How does jamica compare with the reference AMICA implementation?

jamica is designed to reproduce the original AMICA algorithm while providing a native Python implementation, optional JAX acceleration, and seamless integration with the scientific Python ecosystem.

Validation experiments comparing jamica with the Fortran AMICA 1.7 reference are available in the documentation. Agreement is established for single-model configurations against a locally patched build of the reference; multi-model parity is not covered.

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

**https://github.com/snesmaeili/jamica/issues**

When possible, include:

- your operating system
- Python version
- jamica version
- backend (NumPy or JAX)
- a minimal reproducible example

______________________________________________________________________

## Can I contribute?

Absolutely!

Please read the [Contributing Guide](contributing.md) before opening a pull request.
