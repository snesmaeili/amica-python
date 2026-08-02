# Examples

amica provides examples covering the most common workflows, from fitting ICA on EEG recordings to using the core Python API directly.

## Getting Started

If you are new to amica, we recommend following the examples in this order:

1. **MNE-Python integration** – fit AMICA on an EEG recording and obtain a standard `mne.preprocessing.ICA` object.
1. **Native amica API** – fit AMICA directly on a NumPy array using the core `Amica` class.
1. **Validation example** – reproduce the numerical validation experiments on the MNE sample dataset.

______________________________________________________________________

## Example Gallery

The examples below are generated automatically from the scripts in the
`examples/` directory.

```{toctree}
---
maxdepth: 2
---
auto_examples/index
```

______________________________________________________________________

## Running Examples Locally

Clone the repository and install the optional dependencies:

```bash
git clone https://github.com/snesmaeili/amica.git
cd amica

pip install -e ".[all]"
```

or with `uv`:

```bash
uv pip install -e ".[all]"
```

You can then execute any example directly, for example:

```bash
python examples/01_mne_integration.py
```

or

```bash
python examples/02_pure_jax_fitting.py
```

______________________________________________________________________

## Advanced Examples

The repository also includes additional examples that are not part of the online gallery:

- Validation experiments comparing amica with the Fortran AMICA 1.7 reference implementation.
- HPC / Slurm templates for running AMICA on computing clusters.
- Jupyter notebooks demonstrating complete EEG workflows.

These can be found in the `examples/validation/`, `examples/cluster/`, and notebook files in the repository.
