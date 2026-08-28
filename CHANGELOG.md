# Changelog

<!-- towncrier release notes start -->

## [0.3.0](https://github.com/snesmaeili/jamica/releases/tag/v0.3.0) (2026-08-28)

### Enhancements

- `AmicaICA.save()` writes a whole multi-model fit to HDF5 and `read_amica_ica()` reads it back, restoring the per-model `mne.preprocessing.ICA` views along with each child's `exclude` and `labels_`. A FIF file holds one unmixing matrix and so cannot represent a mixture; HDF5 is used through the same MNE helper that `EOGRegression` uses, and needs the optional `h5io` dependency (`pip install "jamica[hdf5]"`). `AmicaICA.export_model_fifs()` additionally writes each model as an ordinary `-ica.fif` that `mne.preprocessing.read_ica()` opens on a machine without jamica installed, so a decomposition never becomes readable only through this package.
- `AmicaICA` exposes a multi-model AMICA fit as one ordinary `mne.preprocessing.ICA` per model, so MNE's existing review tools work unchanged on each model while the mixture — the model priors and the posterior time course — stays on the parent. `apply()` raises for more than one model rather than silently reconstructing from the highest-weight one; name the model with `model_idx=` or use `models_[h].apply()`. Posteriors are reported on the input sampling grid for both Raw and Epochs, and `fit_sample_mask_` records which samples drove the fit. `fit_ica()` is unchanged.

### Bug Fixes

- Multi-model fits froze every model centre at zero whenever the caller had already centred the data, because the M-step gated its centre update on `do_mean` — which asks whether preprocessing removes the global mean, a separate question. The mixture could then only distinguish its models by their unmixing matrices. The centre update is now controlled by `AmicaConfig.update_c`, which defaults to following `do_mean` for a single model and to always updating for a mixture; single-model behaviour is unchanged.

### API Changes

- `jamica.amica` is now a stable, single-model solver boundary for ICA frameworks
  that already preprocess their data. With `whiten=False` it disables JAMICA
  centering, sphering, and PCA, composes any internal scalar rescaling into the
  returned unmixing matrix, and guarantees `Y == W @ X`. The function no longer
  accepts arbitrary configuration keywords, so callers cannot enable multiple
  models or hidden preprocessing through adapter parameters. It accepts both
  NumPy random-state APIs, reports attempted iterations accurately, and warns
  with `JamicaConvergenceWarning` when a finite last iterate has not converged.
  The top-level package now exposes `jamica.__version__` so optional-dependency
  version checks can enforce this contract.

### Documentation

- Added a project logo and rewrote the README opening around it. The README now
  leads with what jamica is and where the name comes from, and `docs/_static/`
  carries the generated logo, navbar mark and favicon. All of the assets are
  derived from a single master by `scripts/make_logo_assets.py`.
- Added an AI usage statement, and shortened the README's validation section to a
  summary plus a pointer at
  [jamica-benchmark](https://github.com/snesmaeili/jamica-benchmark), which holds
  the protocols, the cross-implementation comparisons and the manuscript figures.
  The patched Fortran reference is public there, which the README previously said
  it was not.
- Replaced the logo with the line-art mark and corrected how the assets are
  derived from it. The background removal kept anti-aliased edges opaque in
  their blended-with-white colour, which showed as a pale rim on any dark
  surface and became a bright halo in the dark variant; it now solves for
  coverage and colour separately. The dark variant also decided what counted
  as greyscale using HLS saturation, which is unstable for near-black pixels
  and left holes in the wordmark, and now uses absolute chroma.
- Restored the solid-fill logo artwork, and stopped downscaling the navbar mark
  to a fixed 512px. The mark is now capped rather than resized, so it keeps the
  master's native resolution when that is smaller than the cap instead of
  throwing away detail retina displays can show.
- The documentation build works again. `docs/contributing.md` includes `CONTRIBUTING.md`, so its document-relative link to `AI_USAGE.md` was resolved against `docs/` rather than the repository root and raised a warning, which the build treats as an error. The link is now absolute, which resolves for both GitHub and Sphinx.
- Trimmed the README's validation section to the result and a pointer. The scope
  detail and the instructions for reproducing the parity numbers now live in the
  documentation, under Numerical validation.
- `AmicaICA`, `read_amica_ica` and `get_model_ica` now appear in the API reference, and the introduction points multi-model users at `AmicaICA` rather than only at `fit_ica`.

## [0.2.0](https://github.com/snesmaeili/jamica/releases/tag/v0.2.0) (2026-08-13)

### Enhancements

- Bound the expectation step's temporaries by a block rather than by the recording. Those intermediates are `(n_comp, n_mix, n_samples)` tensors -- about eleven live at once, 8160 bytes per sample at 30 components and 3 mixtures -- so at full batch peak memory scaled with the recording. Blocking the time axis inside the compiled graph bounds them by the block instead, which cuts peak memory roughly fivefold and is also faster, because a block fits in cache where a whole recording never did. Measured across six EEG recordings, peak process memory falls 79% at every one of them (11.4--19.4 GiB to 2.4--4.1 GiB). ([#21](https://github.com/snesmaeili/jamica/pull/21))

### Bug Fixes

- `amica(X, whiten=True)` returned incorrect sources. With `whiten=True` the solver centres and spheres the data internally, so its unmixing matrix operates on whitened data, but the returned `Y` applied that matrix to the raw `X` — on a three-source mixture the sources correlated with ground truth at 0.67–0.75 instead of 1.00. `K` was also always `None`, so the sphering matrix could not be recovered to correct the result. `Y` is now computed as `W @ K @ (X - mean)` and `K` returns the sphering matrix. Only the `whiten=True` path was affected: MNE passes `whiten=False` on data it has pre-whitened itself, and `fit_ica` drives the solver directly with internal whitening disabled, so neither reached this. ([#20](https://github.com/snesmaeili/jamica/pull/20))
- Fixed a `NaN` in the generalized-Gaussian E-step when fitting with `dtype="float32"`. The small-value floor `jnp.maximum(|y|, 1e-300)` guarding `log|y|` used a literal that underflows to `0.0` in float32, so at an exact-zero source activation with `rho == 1` (the Laplacian endpoint) the shape term became `(rho-1)*log(0) = 0*-inf = NaN`, poisoning the sufficient statistics. The floor is now `jnp.finfo(dtype).tiny`, representable in both float32 and float64. Float64 results are unchanged (bit-identical), since the floor only differs for `|y| < 1e-300`, which no real fit reaches.

### API Changes

- `chunk_size` now defaults to `"auto"` rather than `None`. On a single-model CPU fit this blocks the expectation step at 4096 samples, which measures both smaller and faster than full batch at every problem size tested, so it is no longer something to opt into. Pass `chunk_size=None` to force the previous full-batch behaviour; regrouping the sums changes results by roughly 5e-10 relative per step, five orders inside the tolerance the full-batch and chunked paths are held to. ([#21](https://github.com/snesmaeili/jamica/pull/21))
- The distribution and import package are renamed from `amica` to `jamica`: install with `pip install jamica` and import with `from jamica import ...`. The old name installed a top-level `amica` module, which collided with `amica-python`, an independent implementation of the same algorithm by another author, so the two could not be present in one environment — the conda-forge packages had to be declared mutually exclusive. Under the new name they share no paths and can be installed together, which is what the benchmark suite comparing them needs. Exported names are unchanged: `amica`, `Amica`, `AmicaConfig`, `AmicaResult`, `fit_ica` and the rest keep their spelling, and `ica.method` is still `"amica"`, so migration is a one-line change to the import statement. Releases up to 0.1.0 remain available on PyPI under the old name.

### Miscellaneous

- Removed redundant transcendental evaluations from the density and accumulator hot loops: `logsumexp` already forms an exponential that was then recomputed, and the per-mixture loop took three logarithms and three exponentials for two distinct values. Added profiling tools for where a fit's time (`jamica.benchmark.profile_cpu`), memory (`profile_memory`) and scaling (`profile_scaling`) go, and `scripts/regression_vs_ref.py`, which fits identical data under two checkouts and reports the matched unmixing correlation and log-likelihood against a chosen baseline commit. ([#21](https://github.com/snesmaeili/jamica/pull/21)) ## [0.1.0](https://github.com/snesmaeili/jamica/releases/tag/v0.1.0) (2026-08-06)

### Enhancements

- Multi-model fits no longer materialise the full `(n_models, n_components, n_samples)` source array when computing model posteriors or when likelihood-based sample rejection is enabled. Both paths now honour `chunk_size`, bounding peak memory on long recordings. Results are unchanged.

### API Changes

- `fit_ica` now validates `n_components` instead of silently reinterpreting it. Requesting more components than there are selected channels raises `ValueError` (matching `sklearn.decomposition.PCA` and `mne.preprocessing.ICA`, which previously differed from this function), and requesting more than the estimated numerical rank of the data also raises. With `n_components` unset the estimated rank is used and a `RuntimeWarning` reports the value chosen. Previously the default kept every channel: on average-referenced EEG the trailing PCA direction has near-zero variance, which made the unmixing matrix numerically singular, collapsed `mixing_matrix_`, and caused `ICA.apply()` to return near-zero data with no warning.

## [0.0.1](https://github.com/snesmaeili/jamica/releases/tag/v0.0.1) (2026-05-07)

### Enhancements

- Initial release of amica, a native Python reimplementation of AMICA with optional JAX acceleration and MNE-Python integration. ([#1](https://github.com/snesmaeili/jamica/pull/1))

### Authors

- @snesmaeili and @hamzaabdelhedi ([#1](https://github.com/snesmaeili/jamica/pull/1))
