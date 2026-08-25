# Contributing to jamica

Thank you for your interest in contributing to **jamica**!

We welcome contributions of all kinds, including bug fixes, new features, documentation improvements, tests, benchmarks, and examples.

______________________________________________________________________

# Getting Started

## 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/jamica.git
cd jamica
```

## 2. Create a Virtual Environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

or using `uv`:

```bash
uv venv
source .venv/bin/activate
```

## 3. Install jamica

Using pip:

```bash
pip install -e ".[dev]"
```

Using uv:

```bash
uv pip install -e ".[dev]"
```

`[dev]` pulls `[test]`, which brings JAX and the plotting stack, so `pytest`
runs the suite against the shipped backend.

The MNE-Python integration tests skip themselves when MNE is absent, so a plain
`[dev]` install reports a clean run while leaving roughly twenty of them
unexecuted. Since that integration is a headline feature, install those extras
before trusting a green suite on changes that could touch it:

```bash
pip install -e ".[dev,mne,icalabel]"
pytest --run-slow
```

## 4. Install Pre-commit Hooks

```bash
pre-commit install
```

This enables automatic formatting, linting, and repository consistency checks before every commit.

______________________________________________________________________

# Development Workflow

Create a feature branch:

```bash
git checkout -b feature/my-new-feature
```

Make your changes and add or update tests where appropriate.

Run the full pre-commit suite:

```bash
pre-commit run --all-files
```

Run the test suite:

```bash
pytest
```

______________________________________________________________________

# Backend Testing

jamica supports multiple computational backends.

## NumPy

```bash
pytest
```

## JAX (CPU)

```bash
pytest tests/ --backend=cpu
```

## JAX (GPU)

Requires CUDA and the GPU dependencies.

```bash
pytest tests/ --backend=gpu
```

To include slow tests:

```bash
pytest tests/ --backend=gpu --run-slow
```

GPU tests are recommended whenever modifying the optimization algorithm or JAX backend.

______________________________________________________________________

# Documentation

If your changes affect the documentation, ensure it builds successfully.

```bash
cd docs
make html
```

______________________________________________________________________

# Nox

For maintainers and advanced contributors, Nox provides reproducible development sessions.

```bash
nox -s tests
nox -s lint
nox -s docs
```

______________________________________________________________________

# Code Style

jamica uses:

- **Ruff** for linting and formatting
- **pre-commit** for automated quality checks
- **NumPy-style docstrings** for public APIs

Before opening a pull request, make sure:

- all tests pass
- pre-commit passes
- documentation builds successfully (if affected)

______________________________________________________________________

# Pull Requests

Please:

- write clear commit messages
- include tests for new functionality
- update documentation when appropriate
- reference related issues (for example `Fixes #42`)

Pull requests should target the **main** branch.

## AI assistance

You are welcome to use AI coding assistants. Two requests: understand what you
are submitting well enough to defend it in review, and note the assistance with
a `Co-authored-by:` trailer on the commit so the record stays accurate.

[AI_USAGE.md](https://github.com/snesmaeili/jamica/blob/main/AI_USAGE.md) describes how assistance has been used in this
project and what it does not change about how the package is verified.

______________________________________________________________________

# Bug Reports

When reporting a bug, please include:

- a minimal reproducible example
- expected behavior
- actual behavior
- Python version
- operating system
- backend (NumPy/JAX CPU/JAX GPU)

______________________________________________________________________

# Feature Requests

Feature requests are welcome.

Please describe:

- the motivation
- the proposed API or behavior
- relevant papers or references, if applicable

______________________________________________________________________

# Benchmarks and Validation

Contributions that compare jamica against other ICA implementations are especially valuable, including:

- Fortran AMICA 1.7
- Picard
- FastICA
- Infomax

Benchmarking on new EEG or MEG datasets is also encouraged.

______________________________________________________________________

# Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.

______________________________________________________________________

# Questions

If you have questions, feel free to open a GitHub issue or discussion.
