```{raw} html
<p align="center">
  <img src="_static/logo.png" class="only-light" alt="jamica - Adaptive Mixture Independent Component Analysis, powered by JAX" width="420">
  <img src="_static/logo-dark.png" class="only-dark" alt="jamica - Adaptive Mixture Independent Component Analysis, powered by JAX" width="420">
</p>
```

# jamica

**Native Python AMICA for scientific EEG workflows.**

jamica provides an open Python implementation of **Adaptive Mixture Independent Component Analysis (AMICA)** with a modern scientific Python API, optional JAX acceleration, and MNE-Python integration.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Quick Start
:link: examples
:link-type: doc

Install jamica, fit AMICA on MNE data, and run the core examples.
:::

:::{grid-item-card} Background
:link: explanation
:link-type: doc

Understand AMICA, mixture ICA, optimization, and the design of jamica.
:::

:::{grid-item-card} API Reference
:link: api
:link-type: doc

Browse the public Python API, classes, functions, and configuration objects.
:::

:::{grid-item-card} FAQ
:link: faq
:link-type: doc

Find troubleshooting notes for installation, JAX, MNE integration, and validation.
:::
::::

## Why jamica?

- Native Python implementation of AMICA
- Optional JAX CPU/GPU acceleration
- Drop-in MNE-Python integration
- Multi-model AMICA support
- Numerical validation against the Fortran AMICA 1.7 reference implementation
- Designed for reproducible EEG and neuroimaging workflows

## Minimal Example

```python
from jamica import Amica, AmicaConfig

config = AmicaConfig(max_iter=2000, num_mix_comps=3)
model = Amica(config, random_state=42)

result = model.fit(data)
sources = model.transform(data)
```

```{toctree}
---
maxdepth: 2
caption: User Guide
hidden: true
---
examples
explanation
faq
```

```{toctree}
---
maxdepth: 2
caption: Reference
hidden: true
---
api
auto_examples/index
contributing
```

## Project Links

- [GitHub repository](https://github.com/snesmaeili/jamica)
- [PyPI package](https://pypi.org/project/jamica/)
- [Issue tracker](https://github.com/snesmaeili/jamica/issues)
