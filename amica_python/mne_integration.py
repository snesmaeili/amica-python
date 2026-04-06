"""MNE-Python integration for AMICA.

Provides helper functions to use AMICA with MNE-Python's ICA workflow.

Usage
-----
>>> from amica_python import fit_ica
>>> ica = fit_ica(raw, n_components=20, max_iter=2000)
>>> ica.plot_components()
>>> ica.apply(raw)
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _extract_data(inst, picks):
    """Extract (n_channels, n_samples) data array from Raw or Epochs."""
    from mne.io import BaseRaw
    from mne.epochs import BaseEpochs

    if isinstance(inst, BaseRaw):
        return inst.get_data(picks)
    elif isinstance(inst, BaseEpochs):
        # Concatenate all epochs along time axis
        return np.concatenate(inst.get_data()[:, picks, :], axis=-1)
    else:
        raise TypeError(f"inst must be Raw or Epochs, got {type(inst)}")


def _compute_pre_whitener(data, info, picks):
    """Compute MNE-style pre-whitener: per-channel-type std normalization.

    This replicates what MNE does in ICA._pre_whiten when noise_cov=None:
    divide each channel by the std of channels of that type.
    """
    from mne import channel_type

    ch_types = [channel_type(info, idx) for idx in picks]
    unique_types = set(ch_types)

    pre_whitener = np.ones((len(picks), 1), dtype=np.float64)
    for ch_type in unique_types:
        mask = np.array([t == ch_type for t in ch_types])
        if mask.sum() > 0:
            std = np.std(data[mask])
            if std > 0:
                pre_whitener[mask, 0] = std

    return pre_whitener


def _compute_pca(data, n_components):
    """Compute PCA on pre-whitened, centered data.

    Returns pca_components, pca_mean, pca_explained_variance (all for
    the full set of components, not truncated).
    """
    from sklearn.decomposition import PCA

    # PCA expects (n_samples, n_features)
    pca = PCA(whiten=False)
    pca.fit(data.T)

    pca_components = pca.components_  # (n_features, n_features)
    pca_mean = pca.mean_              # (n_features,)
    pca_explained_variance = pca.explained_variance_  # (n_features,)

    return pca_components, pca_mean, pca_explained_variance


def fit_ica(
    inst,
    n_components: Optional[int] = None,
    max_iter: int = 2000,
    num_mix: int = 3,
    random_state: Optional[int] = None,
    picks=None,
    reject=None,
    flat=None,
    decim=None,
    fit_params: Optional[dict] = None,
    verbose=None,
    _use_infomax_shim: bool = False,
):
    """Fit ICA using AMICA on MNE Raw or Epochs data.

    This function replicates MNE's whitening/PCA pipeline, then runs
    AMICA for the unmixing step. The result is a standard MNE ICA object
    that works with all MNE ICA methods (plot_components, apply, etc.).

    Parameters
    ----------
    inst : mne.io.Raw | mne.Epochs
        MNE data object.
    n_components : int | None
        Number of ICA components. If None, equals n_channels.
    max_iter : int
        Maximum AMICA iterations. Default 2000.
    num_mix : int
        Number of generalized Gaussian mixture components. Default 3.
    random_state : int | None
        Random seed.
    picks : str | array-like | None
        Channels to use for ICA.
    reject : dict | None
        Epoch rejection parameters.
    flat : dict | None
        Flat channel rejection parameters.
    decim : int | None
        Decimation factor.
    fit_params : dict | None
        Additional parameters passed to AmicaConfig.
    verbose : bool | None
        Verbosity.
    _use_infomax_shim : bool
        If True, fall back to the old approach of using a 1-iteration
        Infomax fit to get MNE's whitening pipeline. Default False.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object with AMICA decomposition.

    Examples
    --------
    >>> from amica_python import fit_ica
    >>> ica = fit_ica(raw, n_components=20)
    >>> ica.plot_sources(raw)
    >>> ica.apply(raw)
    """
    try:
        from mne.preprocessing import ICA
    except ImportError:
        raise ImportError(
            "MNE-Python is required for fit_ica(). "
            "Install with: pip install mne"
        )

    from amica_python import Amica, AmicaConfig

    # Guard: MNE's ICA assumes a single decomposition matrix.
    _fit_params = fit_params or {}
    _num_models = _fit_params.get("num_models", 1)
    if _num_models > 1:
        raise ValueError(
            f"fit_ica() only supports single-model AMICA (num_models=1), "
            f"got num_models={_num_models}. For multi-model AMICA, use "
            f"Amica(config).fit(data) directly and access AmicaResult."
        )

    if _use_infomax_shim:
        return _fit_ica_infomax_shim(
            inst, n_components=n_components, max_iter=max_iter,
            num_mix=num_mix, random_state=random_state, picks=picks,
            reject=reject, flat=flat, decim=decim, fit_params=fit_params,
            verbose=verbose,
        )

    # ================================================================
    # Direct MNE ICA construction (no throwaway Infomax)
    # ================================================================
    import mne
    from mne.io import BaseRaw

    # Resolve picks
    picks_idx = mne.io.pick._picks_to_idx(inst.info, picks, exclude="bads")

    # Extract data
    raw_data = _extract_data(inst, picks_idx)
    n_channels, n_samples = raw_data.shape

    # Resolve n_components
    if n_components is None:
        n_comp = n_channels
    else:
        n_comp = min(n_components, n_channels)

    # Decimation
    if decim is not None and decim > 1:
        raw_data = raw_data[:, ::decim]
        n_samples = raw_data.shape[1]

    # Step 1: Pre-whiten (per-channel-type std normalization)
    pre_whitener = _compute_pre_whitener(raw_data, inst.info, picks_idx)
    data_pre = raw_data / pre_whitener

    # Step 2: PCA
    pca_components, pca_mean, pca_explained_variance = _compute_pca(
        data_pre, n_comp
    )

    # Step 3: Project to PCA space (truncated to n_components)
    data_centered = data_pre - pca_mean[:, None]
    pca_data = pca_components[:n_comp] @ data_centered
    # pca_data shape: (n_comp, n_samples)

    # Normalize to unit variance per component to stabilize AMICA's gradient.
    comp_stds = np.std(pca_data, axis=1, keepdims=True)
    comp_stds[comp_stds == 0] = 1.0
    data_for_amica = pca_data / comp_stds

    # Step 4: Run AMICA
    cfg_kwargs = dict(
        max_iter=max_iter,
        num_mix_comps=num_mix,
        do_sphere=False,
        do_mean=False,
        lrate=0.02,
    )
    if fit_params:
        cfg_kwargs.update(fit_params)
    config = AmicaConfig(**cfg_kwargs)

    solver = Amica(config, random_state=random_state)
    result = solver.fit(data_for_amica)

    W = result.unmixing_matrix_white_  # (n_comp, n_comp)

    # Step 5: Undo per-component normalization
    W_corrected = W / comp_stds.squeeze()[np.newaxis, :]

    # MNE convention: unmixing /= sqrt(pca_explained_variance_)
    norms = np.sqrt(pca_explained_variance[:n_comp])
    norms[norms == 0] = 1.0

    # Step 6: Construct MNE ICA object with all required attributes
    # MNE validates method at __init__ — use 'infomax' placeholder, override below
    ica = ICA(n_components=n_comp, method="infomax", max_iter=max_iter)

    # Channel info
    ica.info = mne.pick_info(inst.info, picks_idx)
    ica.ch_names = [inst.info["ch_names"][i] for i in picks_idx]

    # Pre-whitening
    ica.pre_whitener_ = pre_whitener

    # PCA
    ica.pca_components_ = pca_components
    ica.pca_mean_ = pca_mean
    ica.pca_explained_variance_ = pca_explained_variance

    # ICA decomposition
    ica.n_components_ = n_comp
    ica.unmixing_matrix_ = W_corrected / norms
    ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

    # Metadata
    ica.n_iter_ = result.n_iter
    ica.n_samples_ = n_samples
    ica.current_fit = "raw" if isinstance(inst, BaseRaw) else "epochs"
    ica.method = "amica"
    ica.labels_ = dict()
    ica.exclude = []

    # Internal naming
    try:
        ica._ica_names = [f"ICA{ii:03d}" for ii in range(n_comp)]
    except Exception:
        pass

    # Attach full AMICA result for viz module
    ica.amica_result_ = result

    return ica


def _fit_ica_infomax_shim(
    inst, n_components=None, max_iter=2000, num_mix=3,
    random_state=None, picks=None, reject=None, flat=None,
    decim=None, fit_params=None, verbose=None,
):
    """Legacy path: use 1-iteration Infomax to get MNE's whitening pipeline.

    Kept as escape hatch via ``_use_infomax_shim=True``. The throwaway
    Infomax fit is discarded — only the whitening/PCA is kept.
    """
    import warnings
    from mne.preprocessing import ICA
    from amica_python import Amica, AmicaConfig

    ica = ICA(
        n_components=n_components,
        method="infomax",
        random_state=random_state,
        max_iter=1,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*n_components.*unstable.*")
        warnings.filterwarnings("ignore", message=".*convergence.*")
        ica.fit(
            inst, picks=picks, reject=reject, flat=flat,
            decim=decim, verbose=verbose,
        )

    from mne.io import BaseRaw
    from mne.epochs import BaseEpochs

    if isinstance(inst, BaseRaw):
        ch_picks = ica._get_picks(inst)
        raw_data = inst.get_data(ch_picks)
    else:
        raw_data = np.concatenate(
            [inst[i].get_data() for i in range(len(inst))], axis=-1
        )
        ch_picks = ica._get_picks(inst)
        raw_data = raw_data[ch_picks]

    data_pre = ica._pre_whiten(raw_data)
    if ica.pca_mean_ is not None:
        data_pre -= ica.pca_mean_[:, None]
    pca_data = np.dot(ica.pca_components_[:ica.n_components_], data_pre)

    comp_stds = np.std(pca_data, axis=1, keepdims=True)
    comp_stds[comp_stds == 0] = 1.0
    data_for_amica = pca_data / comp_stds

    cfg_kwargs = dict(
        max_iter=max_iter,
        num_mix_comps=num_mix,
        do_sphere=False,
        do_mean=False,
        lrate=0.02,
    )
    if fit_params:
        cfg_kwargs.update(fit_params)
    config = AmicaConfig(**cfg_kwargs)

    solver = Amica(config, random_state=random_state)
    result = solver.fit(data_for_amica)

    W = result.unmixing_matrix_white_
    W_corrected = W / comp_stds.squeeze()[np.newaxis, :]

    norms = np.sqrt(ica.pca_explained_variance_[:ica.n_components_])
    norms[norms == 0] = 1.0

    ica.unmixing_matrix_ = W_corrected / norms
    ica.n_iter_ = result.n_iter
    ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)
    ica.method = "amica"
    ica.amica_result_ = result

    return ica
