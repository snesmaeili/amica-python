# Changelog

<!-- towncrier release notes start -->

## [0.1.0](https://github.com/snesmaeili/jamica/releases/tag/v0.1.0) (2026-08-06)

### Enhancements

- Multi-model fits no longer materialise the full `(n_models, n_components, n_samples)` source array when computing model posteriors or when likelihood-based sample rejection is enabled. Both paths now honour `chunk_size`, bounding peak memory on long recordings. Results are unchanged.

### API Changes

- `fit_ica` now validates `n_components` instead of silently reinterpreting it. Requesting more components than there are selected channels raises `ValueError` (matching `sklearn.decomposition.PCA` and `mne.preprocessing.ICA`, which previously differed from this function), and requesting more than the estimated numerical rank of the data also raises. With `n_components` unset the estimated rank is used and a `RuntimeWarning` reports the value chosen. Previously the default kept every channel: on average-referenced EEG the trailing PCA direction has near-zero variance, which made the unmixing matrix numerically singular, collapsed `mixing_matrix_`, and caused `ICA.apply()` to return near-zero data with no warning.

## [0.0.1](https://github.com/snesmaeili/jamica/releases/tag/v0.0.1) (2026-05-07)

### Enhancements

- Initial release of amica, a native Python reimplementation of AMICA with optional JAX acceleration and MNE-Python integration. ([#1](https://github.com/snesmaeili/jamica/pull/1))

### Authors

- @snesmaeili and @hamzaabdelhedi ([#1](https://github.com/snesmaeili/jamica/pull/1))
