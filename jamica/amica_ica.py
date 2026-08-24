"""Multi-model AMICA as a parent object owning per-model MNE ICA views.

``mne.preprocessing.ICA`` holds one unmixing matrix, so it cannot represent a
mixture of ``M`` models natively. Rather than change what an ``ICA`` means,
:class:`AmicaICA` keeps the complete AMICA fit as the authoritative object and
exposes each constituent model as an ordinary ``ICA``.

The children need not teach MNE what a mixture is: each has exactly one
unmixing matrix, one PCA state, its own ``exclude`` and its own ``labels_``.
The mixture -- the model priors and the posterior time course -- stays one
level above them, on the parent.
"""

from __future__ import annotations

import contextlib

import numpy as np

from .mne_integration import _prepare_mne_input

__all__ = ["AmicaICA"]


def _fold_center_into_pca_mean(pca_mean, pca_components, comp_stds, c_h, n_comp):
    """Express an AMICA model centre as an offset to MNE's ``pca_mean_``.

    AMICA's sources for model ``h`` are ``W_h @ (x_white - c_h)``, where
    ``x_white`` is what the fit was given: the PCA projection scaled to unit
    variance per component. MNE instead computes
    ``unmixing_matrix_ @ pca_components_[:n] @ (x_prewhitened - pca_mean_)``.

    Writing ``P`` for ``pca_components[:n_comp]`` and ``D`` for
    ``diag(comp_stds)``, matching the two expressions requires
    ``D^-1 P delta = c_h``, i.e. ``P delta = D c_h``. ``P`` has orthonormal
    rows, so ``delta = P.T @ (comp_stds * c_h)``.

    Returns
    -------
    np.ndarray, shape (n_channels,)
        The per-model ``pca_mean_``. Equals ``pca_mean`` when ``c_h`` is zero.
    """
    scaled = np.asarray(comp_stds).ravel()[:n_comp] * np.asarray(c_h).ravel()[:n_comp]
    return np.asarray(pca_mean) + pca_components[:n_comp].T @ scaled


class AmicaICA:
    """Multi-model AMICA fit exposing one ``mne.preprocessing.ICA`` per model.

    Parameters
    ----------
    n_models : int
        Number of AMICA models. ``1`` reduces to an ordinary single-model fit.
    n_components : int | None
        Number of PCA components. ``None`` uses the estimated numerical rank.
    max_iter : int
        Maximum AMICA iterations.
    num_mix : int
        Generalized-Gaussian mixture components per source.
    random_state : int | None
        Seed for the AMICA fit.
    picks : str | array-like | None
        Channels to use, following :func:`~jamica.fit_ica`.
    reject : dict | None
        MNE-style epoch amplitude rejection applied before fitting.
    flat : dict | None
        Flat-channel rejection applied before fitting.
    decim : int | None
        Decimation factor applied before fitting. See Notes on posteriors.
    fit_params : dict | None
        Extra keywords forwarded to :class:`~jamica.config.AmicaConfig`.
    verbose : bool | None
        Verbosity.

    Attributes
    ----------
    models_ : list of mne.preprocessing.ICA
        One child per AMICA model, in model order. Built once at fit time and
        cached, so ``exclude`` / ``labels_`` set on a child by interactive
        review survive repeated access.
    n_models_ : int
        Number of fitted models.
    model_weights_ : np.ndarray, shape (n_models,)
        Model priors (AMICA's ``gm``).
    model_posteriors_ : np.ndarray
        ``p(h | x_t)`` on the ORIGINAL input sampling grid: ``(n_models,
        n_times)`` for Raw and ``(n_models, n_epochs, n_times)`` for Epochs,
        regardless of any decimation used while fitting. Points excluded from
        the fit by epoch rejection are ``NaN``.
    fit_sample_mask_ : np.ndarray of bool
        Which points of that same grid entered the optimisation --
        ``(n_times,)`` for Raw, ``(n_epochs, n_times)`` for Epochs. See
        :meth:`_build_fit_sample_mask` for how decimation is treated.
    amica_result_ : jamica.AmicaResult
        The complete fit, including the density parameters per model.

    Notes
    -----
    ``model_weights_`` and ``model_posteriors_`` are different quantities: the
    former is a single global prior per model, the latter a per-sample
    responsibility that says which model is active when.

    ``model_posteriors_`` is always on the input grid. Because a genuine
    new-data evaluation exists, the fitted model is simply applied to every
    sample afterwards; nothing is interpolated or resampled from the fit-time
    array. Decimation therefore changes which samples drove the optimisation,
    not the shape of the reported posteriors.
    """

    def __init__(
        self,
        n_models: int = 1,
        n_components: int | None = None,
        max_iter: int = 2000,
        num_mix: int = 3,
        random_state: int | None = None,
        picks=None,
        reject=None,
        flat=None,
        decim=None,
        fit_params: dict | None = None,
        verbose=None,
    ):
        if int(n_models) < 1:
            raise ValueError(f"n_models must be >= 1, got {n_models}.")
        self.n_models = int(n_models)
        self.n_components = n_components
        self.max_iter = max_iter
        self.num_mix = num_mix
        self.random_state = random_state
        self.picks = picks
        self.reject = reject
        self.flat = flat
        self.decim = decim
        self.fit_params = fit_params
        self.verbose = verbose
        self._models = None

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, inst):
        """Fit AMICA on Raw or Epochs.

        Parameters
        ----------
        inst : mne.io.Raw | mne.Epochs
            Data to decompose.

        Returns
        -------
        AmicaICA
            The fitted instance.
        """
        from mne.io import BaseRaw

        from .config import AmicaConfig
        from .solver import Amica

        prep = _prepare_mne_input(
            inst,
            n_components=self.n_components,
            picks=self.picks,
            reject=self.reject,
            flat=self.flat,
            decim=self.decim,
            verbose=self.verbose,
        )

        cfg_kwargs = {
            "max_iter": self.max_iter,
            "num_mix_comps": self.num_mix,
            "num_models": self.n_models,
            "do_sphere": False,
            "do_mean": False,
        }
        if self.fit_params:
            cfg_kwargs.update(self.fit_params)
        config = AmicaConfig(**cfg_kwargs)  # type: ignore[arg-type]

        result = Amica(config, random_state=self.random_state).fit(prep.data_for_amica)

        self._prep = prep
        self._inst_kind = "raw" if isinstance(inst, BaseRaw) else "epochs"
        self.amica_result_ = result
        self.n_components_ = prep.n_comp
        self.n_iter_ = result.n_iter
        self.ch_names = [inst.info["ch_names"][i] for i in prep.picks_idx]

        w_all = np.asarray(result.unmixing_matrix_white_)
        if w_all.ndim == 2:  # single model: give it a leading model axis
            w_all = w_all[None]
        self.n_models_ = w_all.shape[0]

        c_all = np.asarray(result.c_)
        if c_all.ndim == 1:
            c_all = c_all[None]
        self._W_all = w_all
        self._c_all = c_all

        gm = np.asarray(result.gm_).ravel()
        self.model_weights_ = gm if gm.size == self.n_models_ else np.ones(self.n_models_)

        # Build every child eagerly and cache: interactive MNE review mutates
        # ICA.exclude, so handing back a fresh object per access would silently
        # discard the user's work.
        self._models = [self._build_child(inst, h, prep) for h in range(self.n_models_)]

        # Posteriors are reported on the ORIGINAL sampling grid, not the grid the
        # optimiser happened to see. They are re-evaluated from the fitted model
        # rather than stretched from the fit-time array, so no value is invented.
        # Computed after the children exist, since the evaluation path is shared
        # with get_model_probabilities().
        self.fit_sample_mask_ = self._build_fit_sample_mask(inst, prep)
        self.model_posteriors_ = self._posteriors_on_input_grid(inst)
        return self

    # ------------------------------------------------------------------
    # children
    # ------------------------------------------------------------------
    def _build_child(self, inst, model_index, prep):
        """Materialise model ``model_index`` as an ordinary ``ICA``.

        Constructed by assignment rather than by calling ``ICA.fit()``: MNE's
        fit sorts components by explained variance afterwards, which would
        break the correspondence between an AMICA component and its density
        parameters (``alpha`` / ``mu`` / ``beta`` / ``rho``) in
        ``amica_result_``.
        """
        import mne
        from mne.preprocessing import ICA

        n_comp = prep.n_comp
        w_h = self._W_all[model_index]
        c_h = self._c_all[model_index] if self._c_all.shape[0] > model_index else None

        # MNE validates ``method`` at __init__; set a known value and override.
        ica = ICA(n_components=n_comp, method="infomax", max_iter=self.max_iter)

        ica.info = mne.pick_info(inst.info, prep.picks_idx)
        ica.ch_names = list(self.ch_names)
        ica.pre_whitener_ = prep.pre_whitener
        ica.pca_components_ = prep.pca_components
        ica.pca_explained_variance_ = prep.pca_explained_variance

        # Per-model centre, folded in. Its own array per child, never shared.
        if c_h is None or not np.any(c_h):
            ica.pca_mean_ = np.array(prep.pca_mean, copy=True)
        else:
            ica.pca_mean_ = _fold_center_into_pca_mean(
                prep.pca_mean, prep.pca_components, prep.comp_stds, c_h, n_comp
            )

        ica.n_components_ = n_comp
        ica.unmixing_matrix_ = w_h / prep.comp_stds.squeeze()[np.newaxis, :]
        ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)

        ica.n_iter_ = self.n_iter_
        ica.n_samples_ = prep.n_samples
        ica.current_fit = self._inst_kind
        ica.method = "amica"
        ica.labels_ = {}
        ica.exclude = []
        ica.reject_ = self.reject
        ica.drop_inds_ = np.array([], dtype=int)
        with contextlib.suppress(Exception):
            ica._ica_names = [f"ICA{ii:03d}" for ii in range(n_comp)]

        # Identify which AMICA model this view came from, and keep the full fit
        # reachable from any child.
        ica._amica_model_index = int(model_index)
        ica.amica_result_ = self.amica_result_
        ica._amica_comp_stds = prep.comp_stds
        return ica

    @property
    def models_(self):
        """List of per-model ``mne.preprocessing.ICA`` views (cached)."""
        if self._models is None:
            raise RuntimeError("AmicaICA is not fitted; call fit() first.")
        return self._models

    # ------------------------------------------------------------------
    # posteriors
    # ------------------------------------------------------------------
    def _build_fit_sample_mask(self, inst, prep):
        """Which points of the input timeline actually entered the optimisation.

        ``True`` where the sample contributed. Epoch rejection (``reject`` /
        ``flat``) sets ``False`` on the dropped epochs and on any trailing
        segment shorter than one epoch.

        Decimation does **not** clear the mask. ``scipy.signal.decimate`` applies
        an anti-aliasing FIR filter before downsampling, so each retained point
        is a weighted combination of its neighbours and every input sample
        contributes to the fit. Marking the non-retained grid points ``False``
        would misdescribe that.
        """
        mask = prep.fit_sample_mask
        if self._inst_kind == "raw":
            out = np.ones(len(inst.times), dtype=bool)
            if mask is not None and mask.shape[0] == out.shape[0]:
                out = np.asarray(mask, dtype=bool).copy()
            return out
        n_epochs, n_times = len(inst), len(inst.times)
        return np.ones((n_epochs, n_times), dtype=bool)

    def _posteriors_on_input_grid(self, inst):
        """``p(h|x_t)`` for every point of the input timeline.

        Samples excluded from the fit by epoch rejection are returned as ``NaN``
        rather than as a value the fit never saw.
        """
        post = self.get_model_probabilities(inst)
        mask = self.fit_sample_mask_
        if mask is not None and not mask.all():
            post = np.array(post, dtype=float, copy=True)
            if post.ndim == 2:
                post[:, ~mask] = np.nan
            else:
                post[:, ~mask] = np.nan
        return post

    def get_model_probabilities(self, inst):
        """Model posteriors ``p(h | x_t)`` evaluated on new data.

        Recomputed from the fitted parameters, not resampled from the fit-time
        posterior array.

        Parameters
        ----------
        inst : mne.io.Raw | mne.Epochs
            Data with the same channels as the fit.

        Returns
        -------
        np.ndarray
            ``(n_models, n_times)`` for Raw, ``(n_models, n_epochs, n_times)``
            for Epochs.
        """
        from mne.io import BaseRaw

        from .mne_integration import _extract_data
        from .multimodel import compute_model_posteriors

        if self._models is None:
            raise RuntimeError("AmicaICA is not fitted; call fit() first.")

        prep = self._prep
        data_white = prep.project(_extract_data(inst, prep.picks_idx))

        res = self.amica_result_
        # ``sbeta_`` stores verbatim the array the solver passes as ``beta`` to
        # compute_model_posteriors, so it is forwarded as-is -- the name is
        # Fortran lineage, not an inversion.
        alpha = np.asarray(res.alpha_)
        mu = np.asarray(res.mu_)
        beta = np.asarray(res.sbeta_)
        rho = np.asarray(res.rho_)
        # Density parameters lose the model axis for a single-model fit.
        if alpha.ndim == 2:
            alpha, mu, beta, rho = alpha[None], mu[None], beta[None], rho[None]

        # log_det_sphere is identical across models, so it cancels in the
        # softmax over h and can be left at zero here.
        post = np.asarray(
            compute_model_posteriors(
                data_white,
                self._W_all,
                self._c_all,
                alpha,
                mu,
                beta,
                rho,
                np.asarray(self.model_weights_),
                log_det_sphere=0.0,
            )
        )

        if not isinstance(inst, BaseRaw):
            n_epochs, n_times = len(inst), len(inst.times)
            if post.shape[1] == n_epochs * n_times:
                return post.reshape(post.shape[0], n_epochs, n_times)
        return post

    # ------------------------------------------------------------------
    # application
    # ------------------------------------------------------------------
    def apply(self, inst, model_idx=None, **kwargs):
        """Remove excluded components using one model's decomposition.

        Parameters
        ----------
        inst : mne.io.Raw | mne.Epochs
            Data to clean, modified in place by MNE.
        model_idx : int | None
            Which model to apply. Required when ``n_models_ > 1``.
        **kwargs
            Passed through to :meth:`mne.preprocessing.ICA.apply`.

        Returns
        -------
        inst
            The cleaned instance.

        Raises
        ------
        ValueError
            When ``n_models_ > 1`` and no model was named. A mixture has no
            single reconstruction until a combination rule is chosen, and
            silently taking the highest-weight model would hide that choice.
        """
        if self._models is None:
            raise RuntimeError("AmicaICA is not fitted; call fit() first.")

        if model_idx is None:
            if self.n_models_ == 1:
                return self._models[0].apply(inst, **kwargs)
            raise ValueError(
                f"This is a {self.n_models_}-model AMICA fit, and no strategy for "
                "reconstructing data from the whole mixture has been selected, so "
                "apply() is ambiguous. Name the model explicitly:\n"
                "    amica.apply(inst, model_idx=h)\n"
                "    amica.models_[h].apply(inst)\n"
                "Model priors are in .model_weights_ and the per-sample "
                "responsibilities in .model_posteriors_."
            )

        idx = int(model_idx)
        if not 0 <= idx < self.n_models_:
            raise IndexError(f"model_idx {idx} out of range [0, {self.n_models_}).")
        return self._models[idx].apply(inst, **kwargs)

    def __repr__(self):
        if self._models is None:
            return f"<AmicaICA (unfitted, n_models={self.n_models})>"
        return (
            f"<AmicaICA | {self.n_models_} models, {self.n_components_} components, "
            f"{self.n_iter_} iterations>"
        )
