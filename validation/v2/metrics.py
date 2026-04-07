"""Metric library for amica-python validation (v2).

All metric functions take a fitted MNE ICA object and the corresponding raw,
and return a JSON-serializable dict. They are pure: no plotting, no I/O.

The orchestrator `compute_all` runs every metric and tags failures rather
than aborting, so a single broken metric never wastes a multi-hour ICA run.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# ICLabel
# ───────────────────────────────────────────────────────────────────────

ICLABEL_CLASSES = (
    "brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other",
)


def compute_iclabel(ica, raw) -> dict[str, Any]:
    """ICLabel classification with confidence-thresholded brain counts.

    Returns counts per class plus brain counts at >50%, >70%, >80%
    confidence and the per-IC class labels and probabilities (so the
    aggregate stage can recompute things without re-running ICLabel).
    """
    from mne_icalabel import label_components

    out = label_components(raw, ica, method="iclabel")
    pred_labels = list(out["labels"])
    probs = np.asarray(out["y_pred_proba"], dtype=float)

    counts = {cls: int(sum(1 for l in pred_labels if l == cls)) for cls in ICLABEL_CLASSES}
    brain_mask = np.array([l == "brain" for l in pred_labels])
    counts["brain_50pct"] = int(np.sum(brain_mask & (probs > 0.5)))
    counts["brain_70pct"] = int(np.sum(brain_mask & (probs > 0.7)))
    counts["brain_80pct"] = int(np.sum(brain_mask & (probs > 0.8)))

    return {
        "counts": counts,
        "labels": pred_labels,
        "probs": probs.tolist(),
    }


# ───────────────────────────────────────────────────────────────────────
# Kurtosis
# ───────────────────────────────────────────────────────────────────────


def compute_kurtosis(ica, raw) -> dict[str, Any]:
    """Per-IC excess kurtosis and brain-like count.

    Brain ICs are super-Gaussian but moderate (0 < kurt < 10).
    Muscle / line / outlier ICs typically have kurt > 30.
    """
    from scipy.stats import kurtosis

    sources = ica.get_sources(raw).get_data()
    kurt = kurtosis(sources, axis=1, fisher=True)

    return {
        "values": kurt.tolist(),
        "mean": float(np.mean(kurt)),
        "median": float(np.median(kurt)),
        "brain_like": int(np.sum((kurt > 0) & (kurt < 10))),
        "n_components": int(sources.shape[0]),
    }


# ───────────────────────────────────────────────────────────────────────
# PSD-derived: alpha peaks + 1/f slope
# ───────────────────────────────────────────────────────────────────────


def compute_psd_metrics(
    ica, raw, fmin: float = 1.0, fmax: float = 45.0, alpha_band=(8.0, 13.0)
) -> dict[str, Any]:
    """PSD-derived per-IC metrics.

    Returns:
        alpha_peaked_ics : count of ICs with α-band power > 1.5× flank power.
        alpha_ratios     : per-IC α / flank ratio.
        slope_1_over_f   : per-IC 1/f exponent estimated by log–log fit
                           on 2–30 Hz (excluding the α band).
        n_components     : number of components.
    """
    from mne.time_frequency import psd_array_welch

    sources = ica.get_sources(raw).get_data()
    sfreq = raw.info["sfreq"]
    psds, freqs = psd_array_welch(
        sources, sfreq=sfreq, fmin=fmin, fmax=fmax,
        n_fft=int(2 * sfreq), verbose=False,
    )

    a_lo, a_hi = alpha_band
    alpha_mask = (freqs >= a_lo) & (freqs <= a_hi)
    flank_mask = ((freqs >= 2) & (freqs < a_lo)) | ((freqs > a_hi) & (freqs <= 30))

    alpha_ratios = []
    slopes = []
    for i in range(psds.shape[0]):
        flank = psds[i, flank_mask].mean()
        ratio = (psds[i, alpha_mask].mean() / flank) if flank > 1e-12 else 0.0
        alpha_ratios.append(float(ratio))

        # 1/f slope on log–log, excluding alpha band
        fit_mask = (freqs >= 2) & (freqs <= 30) & ~alpha_mask
        if fit_mask.sum() >= 5:
            x = np.log10(freqs[fit_mask])
            y = np.log10(np.maximum(psds[i, fit_mask], 1e-30))
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = float("nan")
        slopes.append(slope)

    n_alpha = int(sum(r > 1.5 for r in alpha_ratios))
    return {
        "alpha_peaked_ics": n_alpha,
        "alpha_ratios": alpha_ratios,
        "slope_1_over_f": slopes,
        "n_components": int(psds.shape[0]),
    }


# ───────────────────────────────────────────────────────────────────────
# Mutual Information Reduction (MIR)
# ───────────────────────────────────────────────────────────────────────


def _entropy_knn_1d(x: np.ndarray, k: int = 5) -> float:
    """Kozachenko–Leonenko 1-D differential entropy estimator."""
    from scipy.special import digamma

    x = np.asarray(x).ravel()
    n = x.size
    if n <= k + 1:
        return float("nan")
    xs = np.sort(x)
    # k-th nearest-neighbour distance for each point: vectorised over sorted x.
    # For sorted 1-D data, the k-NN distance is min(x[i+k]-x[i], x[i]-x[i-k]).
    dists = np.empty(n)
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n - 1, i + k)
        dists[i] = max(xs[i] - xs[lo], xs[hi] - xs[i])
    dists = np.maximum(dists, 1e-300)
    return float(digamma(n) - digamma(k) + np.log(2.0) + np.mean(np.log(dists)))


def compute_mir(
    ica, raw, n_subsample: int = 50_000, k: int = 5, random_state: int = 42
) -> dict[str, Any]:
    """Mutual Information Reduction = Σ H(channels) − Σ H(sources).

    Higher (more negative reduction relative to original ⇒ more independent).
    Uses Kozachenko-Leonenko 1-D entropy with subsampling for tractability.
    """
    sources = ica.get_sources(raw).get_data()
    original = raw.get_data()[: sources.shape[0]]

    n_sub = min(n_subsample, sources.shape[1])
    rng = np.random.default_rng(random_state)
    idx = rng.choice(sources.shape[1], n_sub, replace=False)

    h_orig = sum(_entropy_knn_1d(original[i, idx], k=k) for i in range(sources.shape[0]))
    h_comp = sum(_entropy_knn_1d(sources[i, idx], k=k) for i in range(sources.shape[0]))

    return {
        "mir": float(h_orig - h_comp),
        "h_channels": float(h_orig),
        "h_sources": float(h_comp),
        "n_components": int(sources.shape[0]),
        "n_samples_used": int(n_sub),
    }


# ───────────────────────────────────────────────────────────────────────
# Reconstruction error (sanity check on the MNE bridge)
# ───────────────────────────────────────────────────────────────────────


def compute_reconstruction_error(ica, raw) -> dict[str, Any]:
    """||X − ica.apply(X)|| / ||X|| with no excluded components.

    Should be ~1e-15 if the MNE ICA object is correctly populated.
    """
    data = raw.get_data()
    try:
        recon = ica.apply(raw.copy(), verbose=False).get_data()
        rel = float(np.linalg.norm(data - recon) / np.linalg.norm(data))
    except Exception as e:
        logger.warning("Reconstruction error failed: %s", e)
        rel = float("nan")
    return {"relative_error": rel}


# ───────────────────────────────────────────────────────────────────────
# Dipolarity (residual variance from single-dipole fit per IC)
# ───────────────────────────────────────────────────────────────────────


def compute_dipolarity(
    ica, raw, *, max_components: int | None = None
) -> dict[str, Any]:
    """Per-IC residual variance after single-dipole fit (a.k.a. dipolarity).

    For each IC topography we fit a single dipole in fsaverage source space
    and report the residual variance. ICs with low residual variance (< 15 %)
    are conventionally counted as "near-dipolar" — the gold-standard
    AMICA quality metric (Delorme 2012).

    Notes
    -----
    Requires `mne.datasets.fetch_fsaverage` to have been run once.
    Returns NaN-filled stub if forward / BEM is not available — never raises.
    """
    try:
        import mne
        from mne.datasets import fetch_fsaverage

        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = fs_dir.parent
        bem = subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
        if not bem.exists():
            raise FileNotFoundError(bem)

        # Build a forward solution for the (montaged) raw if not cached.
        # This is expensive — caller may want to cache externally.
        info = ica.info if hasattr(ica, "info") and ica.info is not None else raw.info
        trans = "fsaverage"
        src = mne.setup_source_space("fsaverage", spacing="oct6", subjects_dir=subjects_dir,
                                     add_dist=False, verbose=False)
        fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=str(bem),
                                        eeg=True, meg=False, n_jobs=1, verbose=False)

        # Per-IC topography: columns of mixing matrix (sensor space)
        topos = ica.get_components()  # (n_channels, n_components)
        n_ic = topos.shape[1]
        if max_components is not None:
            n_ic = min(n_ic, max_components)

        residuals = []
        for k in range(n_ic):
            evoked = mne.EvokedArray(topos[:, [k]], info, tmin=0.0, verbose=False)
            try:
                dip, resid = mne.fit_dipole(evoked, cov=None, bem=str(bem), trans=trans,
                                            n_jobs=1, verbose=False)
                # residual variance percentage
                rv = float(1.0 - dip.gof[0] / 100.0)
            except Exception:
                rv = float("nan")
            residuals.append(rv)

        residuals = np.asarray(residuals, dtype=float)
        return {
            "residual_variance": residuals.tolist(),
            "near_dipolar_15pct": int(np.sum(residuals < 0.15)),
            "median_rv": float(np.nanmedian(residuals)),
            "n_components": int(n_ic),
        }
    except Exception as e:
        logger.warning("Dipolarity unavailable: %s", e)
        return {
            "residual_variance": None,
            "near_dipolar_15pct": None,
            "median_rv": None,
            "n_components": int(ica.n_components_),
            "error": str(e),
        }


# ───────────────────────────────────────────────────────────────────────
# Orchestrator
# ───────────────────────────────────────────────────────────────────────


METRICS = {
    "iclabel": compute_iclabel,
    "kurtosis": compute_kurtosis,
    "psd": compute_psd_metrics,
    "mir": compute_mir,
    "reconstruction": compute_reconstruction_error,
    # "dipolarity": compute_dipolarity,  # opt-in, expensive
}


def compute_all(
    ica, raw, *, skip: tuple[str, ...] = (), include: tuple[str, ...] = ()
) -> dict[str, Any]:
    """Run every metric, tagging failures rather than aborting.

    Parameters
    ----------
    skip : metric names to skip
    include : if non-empty, *only* run these metrics (and skip the rest)
    """
    out: dict[str, Any] = {
        "n_iter": int(getattr(ica, "n_iter_", 0)),
        "n_components": int(getattr(ica, "n_components_", 0)),
    }
    for name, fn in METRICS.items():
        if include and name not in include:
            continue
        if name in skip:
            continue
        t0 = time.time()
        try:
            out[name] = fn(ica, raw)
            out[name]["_elapsed"] = float(time.time() - t0)
        except Exception as e:
            logger.exception("metric %s failed", name)
            out[name] = {"error": str(e), "_elapsed": float(time.time() - t0)}
    return out
