# MNE single-model solver contract

The public integration point for ICA frameworks that already perform their
own centering, whitening, and dimension reduction is {func}`jamica.amica`.
It exposes one conventional ICA decomposition while the full adaptive-mixture
API remains available through {class}`jamica.Amica`,
{class}`jamica.AmicaConfig`, and {class}`jamica.AmicaICA`.

This is the boundary intended for `mne.preprocessing.ICA(method="jamica")`.
It deliberately does not expose multi-model AMICA through MNE's single-model
`ICA` object.

## Call

```python
from jamica import amica

K, W, Y, n_iter = amica(
    X,
    n_components=None,
    whiten=False,
    return_n_iter=True,
    random_state=random_state,
    max_iter=max_iter,
    num_models=1,
)
```

Here `X` has shape `(n_components, n_samples)`. It must be a finite, real,
two-dimensional numeric array with at least two components and at least as
many samples as components. JAMICA converts it to `float64` before fitting.

With `whiten=False`, which is the MNE integration path:

- JAMICA does not center, whiten, sphere, or PCA-reduce `X`;
- JAMICA always fits exactly one model;
- `n_components` must be `None` or equal to `X.shape[0]`;
- `K` is `None`;
- `W` has shape `(n_components, n_components)` and operates directly on the
  caller's `X`;
- `Y` has shape `(n_components, n_samples)` and is computed as `W @ X`;
- `n_iter` is the number of solver updates attempted.

If the optimizer applies an emergency scalar rescaling internally, that scale
is composed into the returned `W`. There is therefore no hidden transform
between the input and the returned operator. In equations,

```text
Y = W X
A = pinv(W)
X ≈ A Y
```

The approximation in the last line is the ordinary numerical inverse
relationship. The function rejects a non-finite or singular final `W` rather
than returning an operator that an adapter cannot reconstruct with.

## Ownership of preprocessing

The adapter owns preprocessing on this path:

```text
sensor data
  -> MNE channel selection
  -> MNE pre-whitener / noise-covariance transform
  -> MNE PCA centering, projection, and rank selection
  -> MNE selected-component variance normalization
  -> X, shaped components x samples
  -> jamica.amica(..., whiten=False)
  -> W and Y = W @ X
```

Calling with `whiten=True` is supported for standalone use, but an MNE adapter
must not do so because MNE has already whitened and PCA-reduced the data.

## Controlled options

The stable functional signature exposes only:

- `max_iter`, `min_dll`, `do_newton`, and `newt_start` for optimization;
- `num_mix` for the adaptive source-density model;
- `chunk_size` for bounded-memory CPU/JAX execution;
- `random_state` for initialization.

The keyword-only `num_models` parameter accepts only `1`. A different value
raises an actionable `ValueError` directing the caller to
{class}`jamica.AmicaICA`. This lets errors from an MNE `fit_params` dictionary
propagate without an MNE-specific guard.

`random_state` accepts a non-negative integer, `numpy.random.RandomState`,
`numpy.random.Generator`, or `None`. Integer seeds create the same fresh
generator for each fit. RNG objects are consumed in place.

There is no `**kwargs` escape hatch. Other parameters that would enable
internal centering, sphering, PCA, a different dtype, or backend selection are
not accepted. Passing one raises Python's normal unexpected-keyword `TypeError`.
This prevents an adapter's `fit_params` from silently violating the
single-model or preprocessed-data contract.

The package uses JAX when it is installed and otherwise uses its NumPy
implementation. JAX and GPU support remain optional; the MNE integration does
not need to install JAX or expose backend controls.

## Termination and errors

`n_iter` counts attempted updates, including a numerically guarded update whose
likelihood could not be accepted. Consequently it can exceed the number of
entries in the lower-level `AmicaResult.log_likelihood` history.

Reaching `max_iter` with a finite, invertible last operator returns that
operator and emits {class}`jamica.JamicaConvergenceWarning`. Invalid inputs raise
`TypeError` or `ValueError`; an invalid final operator raises `RuntimeError`.
Adapters should let these exceptions and warnings propagate.

## Full JAMICA functionality

For multiple adaptive ICA models, model posterior probabilities, JAMICA model
views, and other advanced controls, use the `jamica` package directly. Those
features are intentionally outside MNE's `ICA(method="jamica")` contract.
