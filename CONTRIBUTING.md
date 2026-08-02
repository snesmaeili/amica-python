# Contributing to amica

Thank you for your interest in contributing to **amica**!

We welcome contributions of all kinds, including bug fixes, new features, documentation improvements, tests, benchmarks, and examples.

______________________________________________________________________

# Getting Started

## 1. Fork and Clone

```bash
git clone https://github.com/<your-username>/amica.git
cd amica
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

## 3. Install amica

Using pip:

```bash
pip install -e ".[dev]"
```

Using uv:

```bash
uv pip install -e ".[dev]"
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

amica supports multiple computational backends.

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

amica uses:

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

Contributions that compare amica against other ICA implementations are especially valuable, including:

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
