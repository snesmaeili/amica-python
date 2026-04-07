# MNE-Python Issue: Add AMICA as ICA Method

## Title

Add AMICA (Adaptive Mixture ICA) as ICA method

---

## Describe the new feature or enhancement

AMICA (Adaptive Mixture ICA) ranked #1 out of 22 ICA algorithms for EEG source separation in Delorme et al. (2012, PLoS ONE), producing the most near-dipolar independent components. Despite this, it has never been available in MNE-Python because it only existed as a GPL-licensed Fortran binary (discussed in #1715).

I've now built a **pure Python/JAX, BSD-3-licensed** implementation that removes this blocker:

- **Package:** [amica-python](https://github.com/snesmaeili/amica-python)
- **License:** BSD-3-Clause (MNE-compatible)
- **Install:** `pip install amica-python`

### What AMICA adds over existing methods

- **Adaptive mixture source models** — each source modeled as mixture of generalized Gaussians, capturing both super- and sub-Gaussian distributions (Infomax/Picard assume fixed density)
- **Newton optimization** — quadratic convergence after natural gradient warm-up (Palmer et al. 2008)
- **Sample rejection** — automatic outlier downweighting for robust decomposition (Klug et al. 2024)
- **Optional GPU acceleration** via JAX — 12.5s for 30 components on 118-channel EEG on A100 (vs ~170s for Picard on CPU)
- **Multi-model ICA** — can learn multiple unmixing matrices for non-stationary data

### Validation

**Numerical parity with MATLAB AMICA 1.7:**

| Config | LL diff | W correlation |
|--------|---------|---------------|
| m=1, Newton | 0.0002% | >0.999 |
| m=3, Newton | 0.00008% | >0.999 |

**Source separation (Amari index, synthetic):**

| Method | Amari Index |
|--------|------------|
| **AMICA** | **0.008** |
| FastICA | 0.010 |
| Infomax | 0.195 |

**Real EEG (ds004505, 118ch dual-layer):** Reconstruction error ~1e-15, ICLabel-compatible, full MNE ICA object returned (plot_components, apply, get_sources all work).

### Current MNE integration

Already working as standalone package:

```python
from amica_python import fit_ica

ica = fit_ica(raw, n_components=20, max_iter=2000)
ica.plot_components()  # standard MNE ICA object
ica.apply(raw)
```

And a Picard-compatible functional API ready for direct integration:

```python
from amica_python import amica

W, n_iter = amica(X, whiten=False, return_n_iter=True, random_state=42)
```

---

## Describe your proposed implementation

Following the exact Picard integration pattern in `mne/preprocessing/ica.py`. The change is ~15 lines:

```python
elif self.method == "amica":
    from amica_python import amica

    W, n_iter = amica(
        data[:, sel].T,
        whiten=False,
        return_n_iter=True,
        random_state=random_state,
        **self.fit_params,
    )
    self.unmixing_matrix_ = W
    self.n_iter_ = n_iter
```

Users would use:

```python
ica = mne.preprocessing.ICA(method='amica', fit_params=dict(num_mix=3))
ica.fit(raw)
```

`amica-python` would be an optional dependency (like `python-picard`), installed separately via `pip install amica-python`. JAX is optional within amica-python — it falls back to NumPy automatically if JAX isn't installed.

I'm happy to submit a PR implementing this if the maintainers are open to it. I'd include:

- Method dispatch in `ica.py`
- Tests following the existing ICA test patterns
- Documentation updates (docstrings, what's new entry)
- Example comparing AMICA vs Picard vs Infomax

---

## Describe possible alternatives

1. **External-only package (current state):** Users install `amica-python` and use `fit_ica(raw)` which returns a standard `mne.preprocessing.ICA` object. This already works but requires users to know about the package — it won't appear in MNE's documentation or `ICA(method=...)` interface.

2. **MNE plugin/extension system:** If MNE develops a plugin system for ICA methods in the future, AMICA could register through that. But no such system exists currently.

3. **Contribution to mne-icalabel or mne-connectivity:** AMICA could live as an MNE-affiliated package rather than being integrated into core MNE. However, ICA methods are core functionality, and the Picard precedent shows direct integration is preferred.

The direct integration (option in the main proposal) is preferred because:

- It follows the established Picard pattern exactly
- Users discover it through MNE's standard API
- Only ~15 lines of code in MNE itself (minimal maintenance)
- The heavy lifting stays in the external `amica-python` package

---

## Additional context

### Background

- **Author:** Sina Esmaeili, PhD student at Universite de Montreal (CoCo Lab, Karim Jerbi). M/EEG signal processing, also author of mne-denoise.
- **Why now:** The original issue #1715 (2014) was blocked by GPL licensing. This pure Python reimplementation with BSD-3 license resolves that.
- **Ongoing validation:** Currently running multi-subject benchmarks on ds004505 (Studnicki 2022, 25 subjects, 118-channel dual-layer EEG) comparing AMICA vs Picard vs Infomax vs FastICA with multiple metrics (ICLabel, kurtosis, MIR, PSD analysis).

### Key references

- Palmer et al. (2011) — AMICA technical report
- Delorme et al. (2012) — AMICA ranked #1 of 22 ICA algorithms (PLoS ONE)
- Frank et al. (2023) — Optimal AMICA parameters (IEEE BIBE)
- Klug et al. (2024) — Sample rejection (Scientific Reports)

### Questions for maintainers

1. Are you open to adding AMICA given the licensing issue from #1715 is resolved?
2. Should I follow the exact Picard pattern (optional dependency)?
3. Any specific tests or benchmarks you'd want before a PR?
4. Concerns about the optional JAX dependency? (NumPy fallback always available)
