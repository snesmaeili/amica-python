"""Publication-ready figures for the amica-python paper.

Two flavours of figure functions:

* Single-subject figures using **MNE-native** ICA viz tools
  (`ica.plot_components`, `ica.plot_properties`, `ica.plot_sources`,
  `mne.viz.plot_topomap`). These need raw + ICA objects.

* Cross-subject aggregate figures (boxplots with individual-subject
  strips, paired-line plots, ICLabel stacked bars). These take a
  long-format DataFrame from `aggregate.load_per_subject_results`.

All functions return the matplotlib Figure for further tweaking and
do not call `plt.show()` — caller decides when to save.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Style — consistent across every figure
# ─────────────────────────────────────────────────────────────────────

METHOD_ORDER = ("amica", "picard", "infomax", "fastica")
METHOD_LABELS = {
    "amica": "AMICA", "picard": "Picard",
    "infomax": "Infomax", "fastica": "FastICA",
}
METHOD_COLORS = {
    "amica": "#1f77b4",
    "picard": "#ff7f0e",
    "infomax": "#2ca02c",
    "fastica": "#d62728",
}
ICLABEL_COLORS = {
    "brain": "#2ca02c", "muscle": "#d62728", "muscle artifact": "#d62728",
    "eye": "#1f77b4", "eye blink": "#1f77b4",
    "heart": "#ff7f0e", "heart beat": "#ff7f0e",
    "line_noise": "#9467bd", "line noise": "#9467bd",
    "channel_noise": "#8c564b", "channel noise": "#8c564b",
    "other": "#7f7f7f",
}

PUB_RC = {
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
}


def use_publication_style() -> None:
    """Apply rcParams once per script."""
    plt.rcParams.update(PUB_RC)


def save_figure(fig, out_dir: Path | str, name: str) -> None:
    """Save a figure as both 300 dpi PNG and vector PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")


def _methods_in_order(items: Iterable[str]) -> list[str]:
    items = list(items)
    return [m for m in METHOD_ORDER if m in items]


# ═════════════════════════════════════════════════════════════════════
# CROSS-SUBJECT AGGREGATE FIGURES
# ═════════════════════════════════════════════════════════════════════


def figure_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    *,
    title: str | None = None,
    ylabel: str | None = None,
    hp: float | None = None,
    ax=None,
):
    """Boxplot of one metric across methods with per-subject scatter overlay.

    Each subject contributes one paired observation per method, drawn as
    light grey lines connecting the same subject across methods so paired
    structure stays visible.
    """
    sub = df.dropna(subset=[metric]).copy()
    if hp is not None:
        sub = sub[sub.hp == hp]
    methods = _methods_in_order(sub["method"].unique())
    if not methods:
        raise ValueError(f"No methods left for metric {metric}")

    pivot = sub.pivot_table(index="subject", columns="method", values=metric).dropna()
    data = [pivot[m].to_numpy() for m in methods]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4.0, 3.2))
    else:
        fig = ax.figure

    bp = ax.boxplot(
        data, positions=range(len(methods)), widths=0.55, showfliers=False,
        patch_artist=True, medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(linewidth=0.8), whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS[m])
        patch.set_alpha(0.55)

    # Paired lines + dots
    rng = np.random.default_rng(0)
    jitter = (rng.random(len(pivot)) - 0.5) * 0.18
    for i, subj in enumerate(pivot.index):
        ys = pivot.loc[subj, methods].to_numpy()
        xs = np.arange(len(methods)) + jitter[i]
        ax.plot(xs, ys, color="grey", alpha=0.25, linewidth=0.6, zorder=1)
        ax.scatter(xs, ys, s=10, c=[METHOD_COLORS[m] for m in methods],
                   edgecolor="white", linewidth=0.3, zorder=3)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    return fig


def figure_iclabel_stacked(df: pd.DataFrame, *, hp: float | None = None):
    """Stacked-bar of mean ICLabel class proportions per method.

    Each method's bar is the across-subject mean count for each IC class
    (brain / muscle / eye / heart / line / channel / other) normalized by
    the total number of components.
    """
    sub = df.copy()
    if hp is not None:
        sub = sub[sub.hp == hp]
    methods = _methods_in_order(sub["method"].unique())

    classes = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]
    cols = [f"iclabel_{c}" for c in classes]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    bottoms = np.zeros(len(methods))
    for cls, col in zip(classes, cols):
        if col not in sub.columns:
            continue
        vals = []
        for m in methods:
            mvals = sub[sub.method == m][col].dropna()
            ncomp = sub[sub.method == m]["n_components"].dropna()
            if len(ncomp) and ncomp.iloc[0] > 0:
                vals.append(mvals.mean() / float(ncomp.iloc[0]))
            else:
                vals.append(0.0)
        vals = np.asarray(vals)
        ax.bar(range(len(methods)), vals, bottom=bottoms,
               label=cls.replace("_", " "),
               color=ICLABEL_COLORS.get(cls, "#999"), edgecolor="white",
               linewidth=0.5)
        bottoms += vals

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("Mean fraction of ICs")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=7)
    ax.set_title("ICLabel composition (mean across subjects)")
    return fig


def figure_paired_metric_panel(
    df: pd.DataFrame, metrics: Iterable[tuple[str, str]],
    *, hp: float | None = None, ncols: int = 3,
):
    """Panel of N metric boxplots — one per metric — sharing methods axis."""
    metrics = list(metrics)
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.0, nrows * 2.8))
    axes = np.atleast_2d(axes).ravel()
    for ax, (metric, label) in zip(axes, metrics):
        try:
            figure_metric_boxplot(df, metric, ylabel=label, hp=hp, ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, f"{metric}\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="red")
            ax.set_xticks([])
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def figure_runtime_vs_quality(
    df: pd.DataFrame, *, hp: float | None = None,
    quality_metric: str = "iclabel_brain_50pct",
):
    """Scatter of runtime vs quality with one cluster per method."""
    sub = df.dropna(subset=["time", quality_metric]).copy()
    if hp is not None:
        sub = sub[sub.hp == hp]

    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for m in _methods_in_order(sub.method.unique()):
        g = sub[sub.method == m]
        ax.scatter(g["time"], g[quality_metric], s=22,
                   c=METHOD_COLORS[m], edgecolor="white", linewidth=0.4,
                   label=METHOD_LABELS[m], alpha=0.9)
        # group centroid
        ax.scatter(g["time"].mean(), g[quality_metric].mean(),
                   marker="X", s=80, c=METHOD_COLORS[m],
                   edgecolor="black", linewidth=0.8, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Wall-clock runtime (s, log scale)")
    ax.set_ylabel("Brain ICs (>50% ICLabel)")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_title("Runtime vs quality (paired by subject)")
    return fig


# ═════════════════════════════════════════════════════════════════════
# SINGLE-SUBJECT FIGURES — MNE-native
# ═════════════════════════════════════════════════════════════════════


def figure_topomap_grid(
    ica_dict: Mapping[str, "mne.preprocessing.ICA"],
    raw,
    labels_dict: Mapping[str, dict] | None = None,
    *,
    n_show: int = 20,
):
    """4 × n_show grid of IC topomaps using ``ica.plot_components``.

    Each row = one ICA method, each column = one IC. IC titles and axis
    spines are coloured by ICLabel category if ``labels_dict`` is given.
    """
    methods = _methods_in_order(ica_dict.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(
        n_methods, n_show, figsize=(n_show * 1.1, n_methods * 1.5),
        gridspec_kw=dict(wspace=0.05, hspace=0.25),
    )
    axes = np.atleast_2d(axes)

    for row, method in enumerate(methods):
        ica = ica_dict[method]
        n_ic = min(n_show, ica.n_components_)
        labels_obj = labels_dict.get(method) if labels_dict else None
        if labels_obj is not None:
            label_list = list(labels_obj.get("labels", []))
            probs = list(labels_obj.get("probs",
                          labels_obj.get("y_pred_proba", [])))
        else:
            label_list = ["other"] * n_ic
            probs = [0.0] * n_ic

        # MNE-native plotting on the prepared axes
        try:
            ica.plot_components(
                picks=list(range(n_ic)),
                axes=list(axes[row, :n_ic]),
                show=False, colorbar=False, contours=4,
                sensors=False, outlines="head", title="",
            )
        except Exception as e:
            logger.warning("plot_components failed for %s: %s", method, e)

        for col in range(n_ic):
            ax = axes[row, col]
            cls = label_list[col] if col < len(label_list) else "other"
            p = probs[col] if col < len(probs) else 0.0
            color = ICLABEL_COLORS.get(cls, ICLABEL_COLORS["other"])
            ax.set_title(f"IC{col}\n{p:.2f}", fontsize=6, color=color,
                         fontweight="bold", pad=2)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(color)
                sp.set_linewidth(1.0)

        for col in range(n_ic, n_show):
            axes[row, col].axis("off")

        # Row label
        axes[row, 0].annotate(
            METHOD_LABELS[method],
            xy=(-0.55, 0.5), xycoords="axes fraction",
            fontsize=10, fontweight="bold", color=METHOD_COLORS[method],
            va="center", ha="right", rotation=0,
        )

    fig.suptitle("Independent components (top 20)", fontsize=11, fontweight="bold")
    return fig


def figure_ic_properties(
    ica, raw, picks: list[int], *, psd_args=None, image_args=None,
):
    """Wrap ``ica.plot_properties`` and return its figures.

    Returns the list of Figure objects (one per pick). Handles older MNE
    pickles that lack the ``reject_`` attribute by initialising it to None.
    """
    psd_args = psd_args or {"fmin": 1.0, "fmax": 45.0}
    image_args = image_args or {"sigma": 1.0}
    if not hasattr(ica, "reject_"):
        ica.reject_ = None
    figs = ica.plot_properties(
        raw, picks=picks, psd_args=psd_args, image_args=image_args, show=False,
    )
    return figs


def figure_source_psd_grid(
    ica_dict: Mapping[str, "mne.preprocessing.ICA"],
    raw, *, n_show: int = 5,
):
    """Source PSD grid: rows = methods, cols = first ``n_show`` ICs."""
    from mne.time_frequency import psd_array_welch

    methods = _methods_in_order(ica_dict.keys())
    fig, axes = plt.subplots(
        len(methods), n_show, figsize=(n_show * 1.7, len(methods) * 1.6),
        sharex=True, sharey=False,
    )
    axes = np.atleast_2d(axes)
    sfreq = raw.info["sfreq"]

    for row, method in enumerate(methods):
        sources = ica_dict[method].get_sources(raw).get_data()[:n_show]
        psds, freqs = psd_array_welch(
            sources, sfreq=sfreq, fmin=1, fmax=45, n_fft=int(2 * sfreq),
            verbose=False,
        )
        for col in range(n_show):
            ax = axes[row, col]
            ax.semilogy(freqs, psds[col], color=METHOD_COLORS[method], linewidth=0.9)
            ax.set_title(f"IC{col}", fontsize=7)
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(METHOD_LABELS[method], fontsize=8,
                              color=METHOD_COLORS[method], fontweight="bold")
        axes[-1, n_show // 2].set_xlabel("Frequency (Hz)", fontsize=8)
    fig.suptitle("Source PSDs (first 5 ICs)", fontsize=10)
    fig.tight_layout()
    return fig


def figure_amica_convergence(amica_results, *, ax=None):
    """Multi-subject AMICA log-likelihood convergence overlay.

    ``amica_results`` is a dict {subject: AmicaResult} or a list.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.4, 3.2))
    else:
        fig = ax.figure

    if isinstance(amica_results, dict):
        items = amica_results.items()
    else:
        items = enumerate(amica_results)

    for label, res in items:
        ll = np.asarray(res.log_likelihood)
        ax.plot(np.arange(1, len(ll) + 1), ll, linewidth=0.9, alpha=0.7,
                label=str(label))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("AMICA convergence across subjects")
    if isinstance(amica_results, dict) and len(amica_results) <= 8:
        ax.legend(frameon=False, fontsize=7, ncol=2)
    return fig
