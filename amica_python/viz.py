"""AMICA-specific visualization functions.

These plots visualize the mixture model parameters that are unique to AMICA
and cannot be obtained from MNE's standard ICA plotting routines.

All functions return matplotlib Figure objects for easy customization.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


def _check_result(result):
    """Validate that result is an AmicaResult with required attributes."""
    required = ("alpha_", "mu_", "sbeta_", "rho_", "log_likelihood")
    for attr in required:
        if not hasattr(result, attr):
            raise TypeError(
                f"Expected an AmicaResult object, missing attribute '{attr}'"
            )


def plot_convergence(result, ax=None, show=True):
    """Plot log-likelihood convergence curve.

    Shows LL vs iteration with Newton start annotated (if applicable).

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. If None, creates a new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    _check_result(result)
    ll = np.asarray(result.log_likelihood)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.plot(ll, color="C0", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-likelihood (per component per sample)")
    ax.set_title("AMICA Convergence")

    # Annotate elapsed time if available
    if hasattr(result, "elapsed_times") and len(result.elapsed_times) > 0:
        elapsed = result.elapsed_times[-1]
        ax.annotate(
            f"Total: {elapsed:.1f}s",
            xy=(0.98, 0.05),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            color="gray",
        )

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_source_densities(result, picks=None, data=None, n_bins=100,
                          n_cols=4, show=True):
    """Plot fitted generalized Gaussian mixture densities for each IC.

    For each selected component, overlays the fitted mixture PDF on top of
    the empirical source histogram. This is the signature AMICA
    visualization -- no other ICA method has mixture source densities.

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.
    picks : array-like of int | None
        Component indices to plot. If None, plots first 16 components.
    data : np.ndarray, shape (n_channels, n_samples) | None
        Original data to compute sources for histograms. If None, only
        the fitted PDF curves are shown (no histograms).
    n_bins : int
        Number of histogram bins.
    n_cols : int
        Number of columns in the subplot grid.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    from scipy.special import gammaln

    _check_result(result)
    alpha = np.asarray(result.alpha_)
    mu = np.asarray(result.mu_)
    beta = np.asarray(result.sbeta_)
    rho = np.asarray(result.rho_)

    n_mix, n_comp = alpha.shape

    if picks is None:
        picks = list(range(min(n_comp, 16)))
    picks = list(picks)

    # Compute sources if data provided
    sources = None
    if data is not None:
        data = np.asarray(data, dtype=np.float64)
        centered = data - result.mean_[:, None]
        whitened = result.whitener_ @ centered
        sources = result.unmixing_matrix @ whitened

    n_plots = len(picks)
    n_rows = max(1, (n_plots + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, comp_i in enumerate(picks):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        # Determine x range from source data or from mu/beta
        if sources is not None:
            y_i = sources[comp_i]
            x_min, x_max = np.percentile(y_i, [0.5, 99.5])
        else:
            # Estimate range from parameters
            centers = mu[:, comp_i]
            scales = 1.0 / np.maximum(beta[:, comp_i], 1e-6)
            x_min = np.min(centers - 4 * scales)
            x_max = np.max(centers + 4 * scales)
            y_i = None

        x = np.linspace(x_min, x_max, 500)

        # Plot histogram if sources available
        if y_i is not None:
            ax.hist(y_i, bins=n_bins, density=True, alpha=0.3, color="C0",
                    edgecolor="none")

        # Plot each mixture component and the total
        total_pdf = np.zeros_like(x)
        for j in range(n_mix):
            a_j = alpha[j, comp_i]
            mu_j = mu[j, comp_i]
            b_j = beta[j, comp_i]
            r_j = rho[j, comp_i]

            # Generalized Gaussian PDF
            y_scaled = b_j * (x - mu_j)
            log_norm = (np.log(max(b_j, 1e-300))
                        - gammaln(1.0 + 1.0 / r_j) - np.log(2.0))
            log_pdf = log_norm - np.abs(y_scaled) ** r_j
            pdf_j = np.exp(log_pdf)

            component_pdf = a_j * pdf_j
            total_pdf += component_pdf

            if n_mix > 1:
                ax.plot(x, component_pdf, "--", alpha=0.5, linewidth=0.8,
                        label=f"mix {j}" if idx == 0 else None)

        ax.plot(x, total_pdf, color="C1", linewidth=1.5)
        ax.set_title(f"IC {comp_i} (rho={rho[0, comp_i]:.2f})", fontsize=9)
        ax.set_yticks([])

    # Hide unused axes
    for idx in range(n_plots, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("AMICA Source Density Models", fontsize=12)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_model_responsibilities(result, data, ax=None, show=True):
    """Plot model responsibilities v_h(t) over time (multi-model only).

    For multi-model AMICA (num_models > 1), shows which ICA model "owns"
    each time point as a stacked area chart or heatmap.

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result with multiple models.
    data : np.ndarray, shape (n_channels, n_samples)
        Original data to compute model responsibilities.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. If None, creates a new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    _check_result(result)
    gm = np.asarray(result.gm_)
    n_models = gm.shape[0]

    if n_models < 2:
        raise ValueError(
            "plot_model_responsibilities requires num_models >= 2. "
            "This result has a single model."
        )

    # Compute source activations for each model and model log-likelihoods
    # This is a simplified computation for visualization
    data = np.asarray(data, dtype=np.float64)
    centered = data - result.mean_[:, None]
    whitened = result.whitener_ @ centered

    # For multi-model, unmixing_matrix has shape (n_models, n_comp, n_comp)
    W_all = np.asarray(result.unmixing_matrix)
    alpha_all = np.asarray(result.alpha_)
    mu_all = np.asarray(result.mu_)
    beta_all = np.asarray(result.sbeta_)
    rho_all = np.asarray(result.rho_)
    c_all = np.asarray(result.c_)

    from scipy.special import gammaln, logsumexp

    n_samples = whitened.shape[1]
    model_lls = np.zeros((n_models, n_samples))

    for h in range(n_models):
        W_h = W_all[h]
        y_h = W_h @ (whitened - c_all[h][:, None])
        n_comp = y_h.shape[0]

        # log|det(W)|
        _, logdet = np.linalg.slogdet(W_h)

        # Source log-likelihoods
        source_ll = np.zeros(n_samples)
        for i in range(n_comp):
            # Mixture log-likelihood for component i
            log_pdfs = np.zeros((alpha_all.shape[1], n_samples))
            for j in range(alpha_all.shape[1]):
                a = alpha_all[h, j, i]
                m = mu_all[h, j, i]
                b = beta_all[h, j, i]
                r = rho_all[h, j, i]
                y_sc = b * (y_h[i] - m)
                log_norm = (np.log(max(b, 1e-300))
                            - gammaln(1.0 + 1.0 / r) - np.log(2.0))
                log_pdfs[j] = np.log(max(a, 1e-300)) + log_norm - np.abs(y_sc) ** r
            source_ll += logsumexp(log_pdfs, axis=0)

        model_lls[h] = source_ll + logdet

    # Compute responsibilities via softmax
    log_weighted = model_lls + np.log(gm)[:, None]
    log_total = logsumexp(log_weighted, axis=0)
    responsibilities = np.exp(log_weighted - log_total)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    times = np.arange(n_samples)
    ax.stackplot(times, responsibilities,
                 labels=[f"Model {h}" for h in range(n_models)],
                 alpha=0.8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Responsibility")
    ax.set_title("Model Responsibilities Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_mixture_weights(result, ax=None, show=True):
    """Plot mixture weights (alpha) per component as a bar chart.

    Shows how the generalized Gaussian mixture is distributed across
    components. Uniform weights suggest single-mode sources; skewed
    weights indicate multimodal source distributions.

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. If None, creates a new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    _check_result(result)
    alpha = np.asarray(result.alpha_)
    n_mix, n_comp = alpha.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_comp * 0.4), 4))
    else:
        fig = ax.figure

    x = np.arange(n_comp)
    bottom = np.zeros(n_comp)
    colors = plt.cm.Set2(np.linspace(0, 1, n_mix))

    for j in range(n_mix):
        ax.bar(x, alpha[j], bottom=bottom, color=colors[j],
               label=f"Mix {j}", edgecolor="white", linewidth=0.5)
        bottom += alpha[j]

    ax.set_xlabel("Component")
    ax.set_ylabel("Mixture weight (alpha)")
    ax.set_title("AMICA Mixture Weights")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=7)
    if n_mix > 1:
        ax.legend(fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_shape_parameters(result, ax=None, show=True):
    """Plot shape parameters (rho) per component.

    Rho indicates the source distribution shape:
    - rho = 1.0: Laplacian (super-Gaussian, typical for neural sources)
    - rho = 2.0: Gaussian (typical for noise)

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.
    ax : matplotlib.axes.Axes | None
        Axes to plot on. If None, creates a new figure.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    _check_result(result)
    rho = np.asarray(result.rho_)
    n_mix, n_comp = rho.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_comp * 0.4), 4))
    else:
        fig = ax.figure

    x = np.arange(n_comp)

    # Plot each mixture's rho as scatter points
    colors = plt.cm.Set1(np.linspace(0, 0.5, n_mix))
    for j in range(n_mix):
        offset = (j - (n_mix - 1) / 2) * 0.15
        ax.scatter(x + offset, rho[j], color=colors[j], s=40, zorder=3,
                   label=f"Mix {j}" if n_mix > 1 else None,
                   edgecolors="black", linewidth=0.5)

    # Reference lines
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(2.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(n_comp - 0.5, 1.02, "Laplacian", fontsize=8, color="gray",
            ha="right")
    ax.text(n_comp - 0.5, 2.02, "Gaussian", fontsize=8, color="gray",
            ha="right")

    ax.set_xlabel("Component")
    ax.set_ylabel("Shape parameter (rho)")
    ax.set_title("AMICA Source Distribution Shapes")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=7)
    ax.set_ylim(0.8, 2.2)
    if n_mix > 1:
        ax.legend(fontsize=8)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_parameter_summary(result, data=None, show=True):
    """Dashboard combining convergence, shape parameters, and mixture weights.

    Creates a 2x2 figure with:
    - Top-left: Convergence curve
    - Top-right: Shape parameters (rho)
    - Bottom-left: Mixture weights (alpha)
    - Bottom-right: Source density for first 4 components (if data provided)

    Parameters
    ----------
    result : AmicaResult
        Fitted AMICA result.
    data : np.ndarray, shape (n_channels, n_samples) | None
        Original data for source density histograms.
    show : bool
        Whether to call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    from scipy.special import gammaln

    _check_result(result)

    fig = plt.figure(figsize=(14, 10))

    # Top-left: Convergence
    ax1 = fig.add_subplot(2, 2, 1)
    plot_convergence(result, ax=ax1, show=False)

    # Top-right: Shape parameters
    ax2 = fig.add_subplot(2, 2, 2)
    plot_shape_parameters(result, ax=ax2, show=False)

    # Bottom-left: Mixture weights
    ax3 = fig.add_subplot(2, 2, 3)
    plot_mixture_weights(result, ax=ax3, show=False)

    # Bottom-right: Source densities for first 4 components
    alpha = np.asarray(result.alpha_)
    mu = np.asarray(result.mu_)
    beta = np.asarray(result.sbeta_)
    rho = np.asarray(result.rho_)
    n_mix, n_comp = alpha.shape

    ax4 = fig.add_subplot(2, 2, 4)
    n_show = min(4, n_comp)

    # Compute sources if data provided
    sources = None
    if data is not None:
        data = np.asarray(data, dtype=np.float64)
        centered = data - result.mean_[:, None]
        whitened = result.whitener_ @ centered
        sources = result.unmixing_matrix @ whitened

    colors_ic = plt.cm.tab10(np.linspace(0, 0.4, n_show))
    for comp_i in range(n_show):
        if sources is not None:
            x_min, x_max = np.percentile(sources[comp_i], [1, 99])
        else:
            centers = mu[:, comp_i]
            scales = 1.0 / np.maximum(beta[:, comp_i], 1e-6)
            x_min = np.min(centers - 4 * scales)
            x_max = np.max(centers + 4 * scales)

        x = np.linspace(x_min, x_max, 300)
        total_pdf = np.zeros_like(x)
        for j in range(n_mix):
            a_j = alpha[j, comp_i]
            mu_j = mu[j, comp_i]
            b_j = beta[j, comp_i]
            r_j = rho[j, comp_i]
            y_sc = b_j * (x - mu_j)
            log_norm = (np.log(max(b_j, 1e-300))
                        - gammaln(1.0 + 1.0 / r_j) - np.log(2.0))
            total_pdf += a_j * np.exp(log_norm - np.abs(y_sc) ** r_j)

        ax4.plot(x, total_pdf, color=colors_ic[comp_i], linewidth=1.2,
                 label=f"IC {comp_i}")

    ax4.set_title("Source Density Models (first 4 ICs)")
    ax4.set_ylabel("Density")
    ax4.legend(fontsize=8)
    ax4.set_yticks([])

    fig.suptitle("AMICA Parameter Summary", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig
