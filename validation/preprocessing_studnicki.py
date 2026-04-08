"""
Studnicki-style dual-layer iCanClean preprocessing for ds004505
==============================================================

Replicates the dual-layer EEG cleaning pipeline used by Studnicki et al.
(2022) on the ds004505 Table Tennis dataset, so that amica-python
validation runs consume data that matches the literature expectations
(per-IC excess kurtosis in the 2–6 range, not the 1700–6500 range that
raw mobile EEG produces).

The implementation is intentionally **self-contained** — it does not
depend on mne-denoise — so it can run on any compute cluster where
amica-python is installed. It re-implements the sliding-window
canonical-correlation-analysis (CCA) cleaning step that iCanClean
performs, using only numpy + scipy.

Parameters are copied verbatim from the working reference in
``D:\\mne-denoise-reports\\scripts\\batch_tabletennis.py`` (lines
123–163), which was already validated on the same dataset. Those
parameters in turn follow the Studnicki et al. 2022 dual-layer
protocol and Gonsisko et al. 2023 iCanClean recommendations.

Modes
-----
- ``dual``   : Clean scalp EEG using the dual-layer noise channels as
               reference, 2-second sliding window, correlation threshold
               0.85, max reject fraction 0.5.
- ``pseudo`` : Pseudo-reference cleaning (uses the scalp channels
               themselves as a filtered reference), threshold 0.95.
- ``combo``  : Apply ``dual`` first, then ``pseudo`` on the result at
               threshold 0.90.
- ``none``   : No cleaning (passthrough).

Usage
-----
>>> from validation.preprocessing_studnicki import clean_studnicki
>>> cleaned_scalp = clean_studnicki(scalp, noise, sfreq, mode="dual")

References
----------
- Studnicki, A., Downey, R. J., & Ferris, D. P. (2022). Characterizing
  and removing artifacts using dual-layer EEG during table tennis.
  Sensors, 22(15), 5867. https://doi.org/10.3390/s22155867
- Gonsisko, C. B., Ferris, D. P., & Downey, R. J. (2023). iCanClean
  improves independent component analysis of mobile brain imaging with
  EEG. Sensors, 23(2), 928. https://doi.org/10.3390/s22020928
- Downey, R. J., & Ferris, D. P. (2022). iCanClean removes motion,
  muscle, eye, and line-noise artifacts from phantom EEG. Sensors.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

import numpy as np
from scipy import linalg as la
from scipy.signal import iirnotch, sosfiltfilt, butter, tf2sos

logger = logging.getLogger(__name__)

Mode = Literal["none", "dual", "pseudo", "combo"]


# ─────────────────────────────────────────────────────────────────────
# Core CCA-based sliding-window cleaner (self-contained iCanClean)
# ─────────────────────────────────────────────────────────────────────

def _cca_clean_window(
    X: np.ndarray,
    Y: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Remove components of ``X`` that correlate with ``Y`` above threshold.

    Solves the canonical correlation problem between ``X`` (primary,
    ``n_x × n_samples``) and ``Y`` (reference, ``n_y × n_samples``),
    then projects ``X`` onto the subspace orthogonal to canonical pairs
    with correlation above ``threshold``. This is the per-window step of
    iCanClean (Downey & Ferris 2022; Gonsisko et al. 2023).
    """
    nx, ns = X.shape
    ny = Y.shape[0]

    # Mean-center each window (per iCanClean / Gonsisko 2023)
    X = X - X.mean(axis=1, keepdims=True)
    Y = Y - Y.mean(axis=1, keepdims=True)

    # Whiten X and Y via QR — avoids explicit covariance inversion
    Qx, Rx = la.qr(X.T, mode="economic")
    Qy, _ = la.qr(Y.T, mode="economic")

    # Canonical correlations via SVD of Qx^T Qy
    try:
        U, s, Vt = la.svd(Qx.T @ Qy, full_matrices=False)
    except la.LinAlgError:
        return X  # bail out on numerical failure; keep window unchanged

    # Canonical directions in original X space: columns are canonical
    # components sorted by correlation.
    Wx = la.solve_triangular(Rx, U, lower=False)  # (nx, k)

    # Components to remove: canonical correlations above threshold
    keep_mask = s <= threshold
    if keep_mask.all():
        return X  # nothing to remove
    if not keep_mask.any():
        return np.zeros_like(X)  # everything correlated; blank out window

    # Project X onto the kept canonical components only. Reconstruction:
    #   X_clean = (Wx[:, keep] @ (Wx[:, keep].T @ X))
    # Using the left inverse via the canonical directions.
    Wx_keep = Wx[:, keep_mask]
    # Normalize columns to unit length in X space so projection is stable
    norms = np.linalg.norm(Wx_keep, axis=0)
    norms[norms == 0] = 1.0
    Wx_keep = Wx_keep / norms
    projected = Wx_keep @ (Wx_keep.T @ X)
    return projected


def _sliding_cca_clean(
    primary: np.ndarray,
    reference: np.ndarray,
    sfreq: float,
    segment_len_sec: float,
    threshold: float,
    max_reject_fraction: float,
) -> tuple[np.ndarray, float]:
    """Sliding-window CCA cleaning over the full recording.

    Returns the cleaned primary array (same shape) and the mean number
    of canonical components removed per window (for bookkeeping).
    """
    n_ch, n_samples = primary.shape
    win = int(round(segment_len_sec * sfreq))
    if win <= 0 or win > n_samples:
        logger.warning(
            "Window length %d out of range for %d samples; skipping clean",
            win, n_samples,
        )
        return primary.copy(), 0.0

    out = primary.copy()
    n_removed = []
    n_windows = 0
    for start in range(0, n_samples - win + 1, win):
        stop = start + win
        X = primary[:, start:stop]
        Y = reference[:, start:stop]

        X_clean = _cca_clean_window(X, Y, threshold)

        # Enforce max reject fraction — if cleaning removed more than
        # this fraction of the window energy, we back off and keep the
        # original window. This matches the iCanClean safeguard.
        e_in = float(np.sum(X ** 2))
        e_out = float(np.sum(X_clean ** 2))
        if e_in > 0:
            removed_frac = max(0.0, 1.0 - e_out / e_in)
            if removed_frac > max_reject_fraction:
                X_clean = X  # too aggressive, keep the raw window
                removed_frac = 0.0
        else:
            removed_frac = 0.0

        out[:, start:stop] = X_clean
        n_removed.append(removed_frac)
        n_windows += 1

    mean_removed = float(np.mean(n_removed)) if n_removed else 0.0
    logger.info(
        "Studnicki sliding CCA: %d windows, mean fraction removed = %.3f",
        n_windows, mean_removed,
    )
    return out, mean_removed


# ─────────────────────────────────────────────────────────────────────
# Pseudo-reference construction (for pseudo / combo modes)
# ─────────────────────────────────────────────────────────────────────

def _make_pseudo_reference(
    scalp: np.ndarray,
    sfreq: float,
    notch_low: float = 5.0,
    notch_high: float = 45.0,
) -> np.ndarray:
    """Build a pseudo-reference by band-reject filtering the scalp signal.

    Retains components outside the physiological EEG band [notch_low,
    notch_high] Hz so that ``_cca_clean_window`` treats line-noise,
    high-frequency muscle, and sub-physiological drift as correlated
    artefact references to be removed from the primary. This follows
    the pseudo-reference trick used in the reference
    ``batch_tabletennis.py`` pipeline.
    """
    nyq = sfreq / 2.0
    low = notch_low / nyq
    high = notch_high / nyq
    # Band-stop Butterworth (removes the EEG band, keeps artefact bands)
    sos = tf2sos(*butter(4, [low, high], btype="bandstop", output="ba"))
    return sosfiltfilt(sos, scalp, axis=1)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

#: Studnicki-style iCanClean parameters, copied verbatim from the
#: working reference in ``batch_tabletennis.py``.
PARAMS = {
    "dual": dict(segment_len_sec=2.0, threshold=0.85, max_reject_fraction=0.5),
    "pseudo": dict(segment_len_sec=2.0, threshold=0.95, max_reject_fraction=0.5),
    "combo_dual": dict(segment_len_sec=2.0, threshold=0.85, max_reject_fraction=0.5),
    "combo_pseudo": dict(segment_len_sec=2.0, threshold=0.90, max_reject_fraction=0.5),
}


def clean_studnicki(
    scalp: np.ndarray,
    noise: Optional[np.ndarray],
    sfreq: float,
    mode: Mode = "dual",
) -> tuple[np.ndarray, dict]:
    """Apply Studnicki-style dual-layer iCanClean preprocessing.

    Parameters
    ----------
    scalp : (n_scalp, n_samples) ndarray
        Scalp EEG channels. Must already be high-pass filtered.
    noise : (n_noise, n_samples) ndarray or None
        Dual-layer noise electrodes. Required for modes ``dual`` and
        ``combo``. Ignored for ``pseudo`` and ``none``.
    sfreq : float
        Sampling frequency in Hz.
    mode : {"none", "dual", "pseudo", "combo"}
        Cleaning mode. See module docstring.

    Returns
    -------
    cleaned : (n_scalp, n_samples) ndarray
        Cleaned scalp EEG, same shape as input.
    meta : dict
        Bookkeeping: mode, threshold, mean rejection fraction per
        stage, and any warnings. Suitable for dropping into the
        validation JSON.
    """
    if mode == "none":
        return scalp.copy(), {"mode": "none"}

    if mode in ("dual", "combo") and noise is None:
        raise ValueError(
            f"Mode '{mode}' requires noise channels but none were provided."
        )

    if mode == "dual":
        p = PARAMS["dual"]
        cleaned, frac = _sliding_cca_clean(
            primary=scalp,
            reference=noise,
            sfreq=sfreq,
            **p,
        )
        return cleaned, {
            "mode": "dual",
            "threshold": p["threshold"],
            "segment_len_sec": p["segment_len_sec"],
            "mean_rejected_fraction": frac,
            "reference": "dual-layer noise electrodes",
        }

    if mode == "pseudo":
        p = PARAMS["pseudo"]
        pseudo_ref = _make_pseudo_reference(scalp, sfreq)
        cleaned, frac = _sliding_cca_clean(
            primary=scalp,
            reference=pseudo_ref,
            sfreq=sfreq,
            **p,
        )
        return cleaned, {
            "mode": "pseudo",
            "threshold": p["threshold"],
            "segment_len_sec": p["segment_len_sec"],
            "mean_rejected_fraction": frac,
            "reference": "pseudo-reference (5-45 Hz band-stop of scalp)",
        }

    if mode == "combo":
        # Stage 1: dual
        p1 = PARAMS["combo_dual"]
        stage1, frac1 = _sliding_cca_clean(
            primary=scalp,
            reference=noise,
            sfreq=sfreq,
            **p1,
        )
        # Stage 2: pseudo on the dual-cleaned output
        p2 = PARAMS["combo_pseudo"]
        pseudo_ref = _make_pseudo_reference(stage1, sfreq)
        stage2, frac2 = _sliding_cca_clean(
            primary=stage1,
            reference=pseudo_ref,
            sfreq=sfreq,
            **p2,
        )
        return stage2, {
            "mode": "combo",
            "stage1": {
                "mode": "dual",
                "threshold": p1["threshold"],
                "mean_rejected_fraction": frac1,
            },
            "stage2": {
                "mode": "pseudo",
                "threshold": p2["threshold"],
                "mean_rejected_fraction": frac2,
            },
            "segment_len_sec": p1["segment_len_sec"],
        }

    raise ValueError(f"Unknown mode: {mode!r}")


# ─────────────────────────────────────────────────────────────────────
# Smoke test (run `python -m validation.preprocessing_studnicki`)
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rng = np.random.default_rng(42)
    sfreq = 250.0
    n_samples = int(sfreq * 30)  # 30 s

    # Fake scalp = brain + shared noise; fake noise = shared noise + ref noise
    shared_noise = rng.standard_normal((5, n_samples)) * 10.0
    brain = rng.standard_normal((20, n_samples))
    scalp = brain + shared_noise[0:1] + shared_noise[1:2]
    noise = shared_noise + rng.standard_normal((5, n_samples)) * 0.1

    for mode in ("none", "dual", "pseudo", "combo"):
        cleaned, meta = clean_studnicki(scalp, noise, sfreq, mode=mode)
        rms_in = float(np.sqrt(np.mean(scalp ** 2)))
        rms_out = float(np.sqrt(np.mean(cleaned ** 2)))
        print(
            f"{mode:8s}  in_rms={rms_in:.3f}  out_rms={rms_out:.3f}  "
            f"meta={meta}"
        )
        assert cleaned.shape == scalp.shape, f"shape mismatch for {mode}"
    print("OK")
