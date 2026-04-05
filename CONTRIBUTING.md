# Contributing to amica-python

Thank you for your interest in contributing to amica-python! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/amica-python.git
   cd amica-python
   ```
3. Create a virtual environment and install in development mode:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -e ".[dev,mne,jax]"
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running Tests

```bash
python -m pytest tests/ -v
```

### Code Style

- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings in NumPy format for all public functions and classes

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in imperative mood (e.g., "Add", "Fix", "Update")
- Reference issue numbers where applicable (e.g., "Fix #42")

## Types of Contributions

### Bug Reports

Open an issue on GitHub with:
- A minimal reproducible example
- Expected vs. actual behavior
- Your environment (Python version, OS, JAX version if applicable)

### Feature Requests

Open an issue describing:
- The use case / motivation
- Proposed API or behavior
- Any relevant references (papers, other implementations)

### Code Contributions

1. Ensure your changes pass all existing tests
2. Add tests for new functionality
3. Update docstrings and documentation as needed
4. Submit a pull request against the `main` branch

### Validation and Benchmarks

We especially welcome contributions that:
- Test AMICA against MATLAB reference outputs on new configurations
- Benchmark on new EEG/MEG datasets
- Compare with other ICA methods (Infomax, FastICA, Picard)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Questions?

Open an issue or contact Sina Esmaeili at sina.esmaeili@umontreal.ca.
