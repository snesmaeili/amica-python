"""
High-Density EEG Validation: ds004505 (Dual-Layer Table Tennis)
===============================================================

Validates amica-python against Picard, Infomax, and FastICA on real
mobile EEG from Studnicki & Ferris (2024), Data in Brief.

Preprocessing follows the dataset paper and notebook Paper9:
- Channel classification via EEGLAB metadata or ds004505 naming conventions
- All 120 scalp EEG channels used (not an arbitrary subset)
- Resample to 250 Hz (matches Frank 2025 and reduces compute)
- 1 Hz HP already applied in Merged files (no redundant re-filter)
- Drop channels missing from montage (O9)
- Average reference
- n_components = min(rank - 1, 60)  (data-driven, not hardcoded)
- Kappa check: N_frames / N_channels^2 >= 30 (Frank 2025)

References
----------
- Studnicki & Ferris (2024). Dual-layer EEG during table tennis. Data in Brief.
- Frank et al. (2025). Data requirements for AMICA. arXiv.
- Klug et al. (2024). Optimizing EEG ICA. Scientific Reports.
"""
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import mne

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──
# Update DS_PATH to your local ds004505 location
DS_PATH = Path(os.environ.get("DS_PATH", "/home/sesma/scratch/ds004505"))
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Parameters ──
SUBJECTS = os.environ.get("SUBJECTS", "sub-01").split(",")
SFREQ_TARGET = 250  # Hz — matches Frank 2025, halves compute vs 500 Hz
MIN_KAPPA = 30  # Frank 2025 recommendation
MAX_DURATION_SEC = None  # Use all available data (kappa-driven)
METHODS = ["amica", "picard", "infomax", "fastica"]
MAX_ITER = int(os.environ.get("MAX_ITER", "500"))
AMICA_NUM_MIX = 3  # Frank 2023 recommendation
HP_FILTERS = [float(x) for x in os.environ.get("HP_FILTERS", "1.0,2.0").split(",")]


# ═══════════════════════════════════════════════════════════════════════
# Channel classification — matches Paper9 notebook exactly
# ═══════════════════════════════════════════════════════════════════════

def classify_channel_by_name(name):
    """Classify ds004505 channel by naming convention.

    From Studnicki 2024 and Paper9 notebook:
    - N-*   : dual-layer noise electrodes (120 ch)
    - None* : unused slots (8 ch)
    - ISCM/SSCM/STrap/ITrap : neck EMG (8 ch, L/R × inferior/superior)
    - CGY/CWR/NGY/NWR : built-in LiveAmp accelerometers (12 ch)
    - Imu_*/Emg_* : Cometas wireless IMU/EMG (variable)
    - Everything else : scalp EEG
    """
    if name.startswith("N-"):
        return "noise"
    if name.startswith("None"):
        return "unused"
    if any(m in name for m in ["ISCM", "SSCM", "STrap", "ITrap"]):
        return "emg"
    if any(name.startswith(p) for p in ["CGY", "CWR", "NGY", "NWR"]):
        return "acc"
    if "Imu_" in name or name.startswith("Emg_"):
        return "cometas"
    return "eeg"


def extract_eeglab_channel_types(set_path):
    """Extract (name, type) from EEGLAB chanlocs.type field."""
    try:
        import scipy.io as sio

        mat = sio.loadmat(str(set_path), squeeze_me=True, struct_as_record=False)
        chanlocs = mat["chanlocs"]
        return [
            (str(getattr(ch, "labels", "")), str(getattr(ch, "type", "")))
            for ch in chanlocs
        ]
    except Exception:
        return None


def classify_channels(raw, set_path=None):
    """Classify all channels into groups. Returns dict of name lists."""
    groups = {"eeg": [], "noise": [], "emg": [], "acc": [], "cometas": [], "unused": []}

    # Try EEGLAB metadata first
    ch_meta = extract_eeglab_channel_types(set_path) if set_path else None

    for i, ch_name in enumerate(raw.ch_names):
        if ch_meta and i < len(ch_meta):
            _, eeglab_type = ch_meta[i]
            eeglab_type = eeglab_type.lower().strip()
            if eeglab_type in ("eeg", ""):
                # EEGLAB sometimes leaves type empty for EEG
                ch_type = classify_channel_by_name(ch_name)
            elif eeglab_type == "noise":
                ch_type = "noise"
            elif eeglab_type == "emg":
                ch_type = "emg"
            elif eeglab_type in ("acc", "cometas"):
                ch_type = eeglab_type
            elif eeglab_type == "none":
                ch_type = "unused"
            else:
                ch_type = classify_channel_by_name(ch_name)
        else:
            ch_type = classify_channel_by_name(ch_name)

        groups[ch_type].append(ch_name)

    return groups


# ═══════════════════════════════════════════════════════════════════════
# Preprocessing — following Paper9 notebook and Frank 2025
# ═══════════════════════════════════════════════════════════════════════

def load_and_preprocess(subject, hp_freq=1.0):
    """Load ds004505, classify channels, preprocess for ICA.

    Parameters
    ----------
    subject : str
        Subject ID (e.g. 'sub-01').
    hp_freq : float
        High-pass filter cutoff in Hz. Merged files already have 1 Hz HP,
        so hp_freq >= 1.0 is required. Use 2.0 for Klug 2024 mobile sweep.

    Returns raw (scalp EEG only, ready for ICA) and channel groups dict.
    """
    set_file = DS_PATH / "sourcedata" / "Merged" / subject / f"{subject}_Merged.set"
    if not set_file.exists():
        raise FileNotFoundError(f"Data not found: {set_file}")

    logger.info("Loading %s ...", set_file.name)
    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)
    logger.info(
        "  Loaded: %d channels, %.0f s, sfreq=%.0f Hz",
        raw.info["nchan"],
        raw.n_times / raw.info["sfreq"],
        raw.info["sfreq"],
    )

    # Classify channels using metadata + naming conventions
    ch_groups = classify_channels(raw, set_path=set_file)
    for grp, names in ch_groups.items():
        if names:
            logger.info("  %s: %d channels", grp, len(names))

    # Pick only scalp EEG
    scalp_ch = ch_groups["eeg"]
    if not scalp_ch:
        raise RuntimeError("No scalp EEG channels found")
    raw.pick(scalp_ch)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    # Set montage — drop channels not in standard_1005
    montage = mne.channels.make_standard_montage("standard_1005")
    montage_names = set(montage.ch_names)
    missing = [ch for ch in raw.ch_names if ch not in montage_names]
    if missing:
        logger.info("  Dropping %d channels not in montage: %s", len(missing), missing)
        raw.drop_channels(missing)
    raw.set_montage(montage, on_missing="ignore")

    # Resample to target (Merged files are at 500 Hz)
    if raw.info["sfreq"] != SFREQ_TARGET:
        raw.resample(SFREQ_TARGET, verbose=False)
        logger.info("  Resampled to %d Hz", SFREQ_TARGET)

    # Merged files are already 1 Hz HP filtered (Studnicki 2024).
    # Apply hp_freq (>= 1 Hz) + 100 Hz LP — ICLabel requires 1-100 Hz bandpass.
    raw.filter(hp_freq, 100.0, verbose=False)
    logger.info("  Filtered: %.1f–100 Hz", hp_freq)
    raw.set_eeg_reference("average", verbose=False)

    # Determine how much data we need for kappa >= MIN_KAPPA
    n_ch = raw.info["nchan"]
    min_samples = MIN_KAPPA * n_ch ** 2
    available = raw.n_times

    if available < min_samples:
        logger.warning(
            "  kappa = %.1f < %d (need %d samples, have %d). "
            "Using all available data.",
            available / n_ch**2,
            MIN_KAPPA,
            min_samples,
            available,
        )
    elif MAX_DURATION_SEC is not None:
        max_samples = int(MAX_DURATION_SEC * raw.info["sfreq"])
        if max_samples < available:
            raw.crop(tmax=MAX_DURATION_SEC)

    n_ch = raw.info["nchan"]
    kappa = raw.n_times / n_ch**2
    logger.info(
        "  Final: %d ch, %d samples, %.0f s, sfreq=%.0f, kappa=%.1f",
        n_ch,
        raw.n_times,
        raw.times[-1],
        raw.info["sfreq"],
        kappa,
    )

    return raw, ch_groups


# ═══════════════════════════════════════════════════════════════════════
# ICA — data-driven n_components
# ═══════════════════════════════════════════════════════════════════════

def determine_n_components(raw, max_components=60):
    """Determine n_components from data rank. Follows Paper9 convention."""
    data = raw.get_data()
    cov = np.cov(data)
    eigvals = np.linalg.eigvalsh(cov)
    rank = int(np.sum(eigvals > eigvals.max() * 1e-6))
    n_comp = min(rank - 1, max_components)
    logger.info("  Data rank = %d, using n_components = %d", rank, n_comp)
    return n_comp


def run_ica_method(raw, method, n_components, max_iter=2000):
    """Run ICA with a given method. Returns (ica, elapsed_seconds)."""
    logger.info("  Running %s (n_comp=%d, max_iter=%d) ...", method, n_components, max_iter)

    if method == "amica":
        from amica_python import fit_ica

        t0 = time.time()
        ica = fit_ica(
            raw,
            n_components=n_components,
            max_iter=max_iter,
            num_mix=AMICA_NUM_MIX,
            random_state=42,
        )
        dt = time.time() - t0
    else:
        import warnings

        t0 = time.time()
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=method,
            random_state=42,
            max_iter=max_iter,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*convergence.*")
            ica.fit(raw, verbose=False)
        dt = time.time() - t0

    logger.info("    Done in %.1f s, %d iterations", dt, ica.n_iter_)
    return ica, dt


# ═══════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════

def run_iclabel(ica, raw):
    """ICLabel classification. Returns (counts_dict, raw_labels)."""
    from mne_icalabel import label_components

    labels = label_components(raw, ica, method="iclabel")
    pred_labels = labels["labels"]

    counts = {}
    for cat in ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]:
        counts[cat] = sum(1 for l in pred_labels if l == cat)

    pred_probs = np.array(labels["y_pred_proba"])
    pred_labels_arr = np.array(pred_labels)
    brain_mask = pred_labels_arr == "brain"
    counts["brain_50pct"] = int(np.sum(brain_mask & (pred_probs > 0.5)))
    counts["brain_70pct"] = int(np.sum(brain_mask & (pred_probs > 0.7)))

    logger.info(
        "    ICLabel: brain=%d (>50%%: %d, >70%%: %d), muscle=%d, eye=%d",
        counts["brain"],
        counts["brain_50pct"],
        counts["brain_70pct"],
        counts["muscle"],
        counts["eye"],
    )
    return counts, labels


def compute_kurtosis_quality(ica, raw):
    """Brain-like IC count via kurtosis (Paper9 fallback metric).

    Brain ICs typically have kurtosis in (0, 10) — super-Gaussian but
    not extremely so. Muscle/artifact ICs tend to have very high kurtosis.
    """
    from scipy.stats import kurtosis

    sources = ica.get_sources(raw).get_data()
    kurt = kurtosis(sources, axis=1, fisher=True)

    brain_like = np.sum((kurt > 0) & (kurt < 10))
    return {
        "brain_like_kurtosis": int(brain_like),
        "n_components": int(sources.shape[0]),
        "kurtosis_mean": float(np.mean(kurt)),
        "kurtosis_median": float(np.median(kurt)),
        "kurtosis_values": kurt.tolist(),
    }


def compute_reconstruction_error(ica, raw):
    """Reconstruction error: ||X - apply(X)|| / ||X|| with no components excluded.

    With exclude=[], apply() should reconstruct perfectly (error ~0).
    This tests that the MNE ICA object is correctly populated.
    """
    data = raw.get_data()
    try:
        raw_recon = ica.apply(raw.copy(), verbose=False)
        recon_data = raw_recon.get_data()
        err = np.linalg.norm(data - recon_data) / np.linalg.norm(data)
    except Exception as e:
        logger.warning("    Reconstruction error failed: %s", e)
        err = float("nan")

    return float(err)


def compute_psd_alpha_peaks(ica, raw):
    """Count ICs with alpha-band (8-13 Hz) spectral peaks.

    Brain sources typically show a clear alpha peak above the 1/f
    background. This metric is method-agnostic (no training bias).
    """
    from mne.time_frequency import psd_array_welch

    sources = ica.get_sources(raw).get_data()
    sfreq = raw.info["sfreq"]

    psds, freqs = psd_array_welch(sources, sfreq=sfreq, fmin=1, fmax=45,
                                   n_fft=int(2 * sfreq), verbose=False)

    alpha_mask = (freqs >= 8) & (freqs <= 13)
    flank_mask = ((freqs >= 2) & (freqs < 7)) | ((freqs > 14) & (freqs <= 30))

    n_alpha = 0
    for i in range(psds.shape[0]):
        flank_power = psds[i, flank_mask].mean()
        if flank_power > 1e-10:
            if psds[i, alpha_mask].mean() / flank_power > 1.5:
                n_alpha += 1

    return {"alpha_peaked_ics": n_alpha, "n_components": int(psds.shape[0])}


def compute_mir(ica, raw):
    """Mutual Information Reduction via k-NN entropy estimator.

    MIR = sum H_marginal(original) - sum H_marginal(sources).
    Higher MIR = more independent sources = better ICA.
    Uses Kozachenko-Leonenko estimator. Subsamples for speed.
    """
    from scipy.special import digamma

    def entropy_knn(x, k=5):
        x = np.sort(x.ravel())
        n = len(x)
        dists = np.zeros(n)
        for i in range(n):
            d = np.abs(x - x[i])
            d_sorted = np.sort(d)
            dists[i] = d_sorted[k] if k < n else d_sorted[-1]
        dists = np.maximum(dists, 1e-300)
        return digamma(n) - digamma(k) + np.log(2) + np.mean(np.log(dists))

    sources = ica.get_sources(raw).get_data()
    original = raw.get_data()[:sources.shape[0]]

    # Subsample to 50k points for speed
    n_sub = min(50000, sources.shape[1])
    rng = np.random.RandomState(42)
    idx = rng.choice(sources.shape[1], n_sub, replace=False)

    n_ch = sources.shape[0]
    h_orig = sum(entropy_knn(original[i, idx]) for i in range(n_ch))
    h_comp = sum(entropy_knn(sources[i, idx]) for i in range(n_ch))

    return {"mir": float(h_orig - h_comp), "n_components": n_ch}


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def _run_metrics(ica, raw, method):
    """Compute all metrics for a fitted ICA, returning a result dict."""
    result = {
        "time": 0.0,  # filled by caller
        "n_iter": int(ica.n_iter_),
        "n_components": int(ica.n_components_),
    }

    # ICLabel
    logger.info("    [%s] Computing ICLabel...", time.strftime("%H:%M:%S"))
    try:
        ic_counts, _ = run_iclabel(ica, raw)
        result["iclabel"] = ic_counts
    except Exception as e:
        logger.warning("    ICLabel failed: %s", e)
        result["iclabel"] = {"error": str(e)}

    # Kurtosis
    logger.info("    [%s] Computing kurtosis...", time.strftime("%H:%M:%S"))
    try:
        kurt = compute_kurtosis_quality(ica, raw)
        result["kurtosis"] = kurt
        logger.info("    Kurtosis: %d/%d brain-like ICs (mean=%.1f)",
                     kurt["brain_like_kurtosis"], kurt["n_components"],
                     kurt["kurtosis_mean"])
    except Exception as e:
        logger.warning("    Kurtosis failed: %s", e)

    # Reconstruction error
    logger.info("    [%s] Computing reconstruction error...", time.strftime("%H:%M:%S"))
    try:
        recon_err = compute_reconstruction_error(ica, raw)
        result["reconstruction_error"] = recon_err
        logger.info("    Reconstruction error: %.2e", recon_err)
    except Exception as e:
        logger.warning("    Reconstruction error failed: %s", e)

    # PSD alpha peaks
    logger.info("    [%s] Computing PSD alpha peaks...", time.strftime("%H:%M:%S"))
    try:
        psd = compute_psd_alpha_peaks(ica, raw)
        result["psd_alpha"] = psd
        logger.info("    Alpha-peaked ICs: %d/%d", psd["alpha_peaked_ics"], psd["n_components"])
    except Exception as e:
        logger.warning("    PSD alpha failed: %s", e)

    # MIR
    logger.info("    [%s] Computing MIR...", time.strftime("%H:%M:%S"))
    try:
        mir = compute_mir(ica, raw)
        result["mir"] = mir
        logger.info("    MIR: %.2f", mir["mir"])
    except Exception as e:
        logger.warning("    MIR failed: %s", e)

    return result


def _print_summary(results, methods):
    """Print a summary table for one HP sweep."""
    logger.info("\n" + "=" * 80)
    header = (
        f"{'Method':<10s} {'Time':>7s} {'Iter':>5s} {'nIC':>4s} "
        f"{'Brain':>6s} {'B>50%':>6s} {'B>70%':>6s} "
        f"{'Kurt':>5s} {'Alpha':>6s} {'MIR':>8s} {'Err':>9s}"
    )
    logger.info(header)
    logger.info("-" * 80)
    for method in methods:
        r = results.get(method, {})
        if "error" in r:
            logger.info("%-10s  FAILED: %s", method, str(r["error"])[:50])
            continue
        ic = r.get("iclabel", {})
        kurt = r.get("kurtosis", {})
        psd = r.get("psd_alpha", {})
        mir = r.get("mir", {})
        err = r.get("reconstruction_error", float("nan"))
        logger.info(
            "%-10s %7.1f %5d %4d %6s %6s %6s %5s %6s %8s %9.1e",
            method, r["time"], r["n_iter"], r["n_components"],
            ic.get("brain", "?"), ic.get("brain_50pct", "?"), ic.get("brain_70pct", "?"),
            kurt.get("brain_like_kurtosis", "?"), psd.get("alpha_peaked_ics", "?"),
            f"{mir['mir']:.1f}" if isinstance(mir, dict) and "mir" in mir else "?",
            err,
        )
    logger.info("=" * 80)


def main():
    logger.info("=" * 65)
    logger.info("HIGH-DENSITY EEG BENCHMARK (ds004505)")
    logger.info("  Subjects: %s", SUBJECTS)
    logger.info("  HP filters: %s Hz", HP_FILTERS)
    logger.info("  Methods: %s", METHODS)
    logger.info("  MAX_ITER: %d", MAX_ITER)
    logger.info("=" * 65)

    all_results = {}

    for subject in SUBJECTS:
        logger.info("\n########## %s ##########", subject)
        all_results[subject] = {}

        for hp_freq in HP_FILTERS:
            sweep_key = f"hp{hp_freq}"
            logger.info("\n===== %s | HP = %.1f Hz =====", subject, hp_freq)

            try:
                raw, ch_groups = load_and_preprocess(subject, hp_freq=hp_freq)
            except FileNotFoundError as e:
                logger.warning("Skipping %s: %s", subject, e)
                break

            n_components = determine_n_components(raw)
            sweep_results = {}

            for method in METHODS:
                logger.info("\n--- %s | %s | HP %.1f Hz ---", method.upper(), subject, hp_freq)
                try:
                    ica, dt = run_ica_method(raw, method, n_components, max_iter=MAX_ITER)
                    result = _run_metrics(ica, raw, method)
                    result["time"] = dt
                    sweep_results[method] = result
                except Exception as e:
                    logger.error("    %s FAILED: %s", method, e)
                    import traceback; traceback.print_exc()
                    sweep_results[method] = {"error": str(e)}

            all_results[subject][sweep_key] = sweep_results

            # Print summary for this sweep
            _print_summary(sweep_results, METHODS)

            # Save incrementally (in case job gets killed)
            output_file = RESULTS_DIR / f"benchmark_{subject}_hp{hp_freq}hz.json"
            with open(output_file, "w") as f:
                json.dump(sweep_results, f, indent=2, default=str)
            logger.info("Saved: %s", output_file)

    # Save combined results
    combined_file = RESULTS_DIR / "benchmark_combined.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nAll results saved to: %s", combined_file)


if __name__ == "__main__":
    main()
