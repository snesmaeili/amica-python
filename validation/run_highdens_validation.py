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
import time
from pathlib import Path

import numpy as np
import mne

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──
# Update DS_PATH to your local ds004505 location
DS_PATH = Path(
    r"D:\mne-denoise\_codex_main_revert_20260306\examples\tutorials\data\ds004505"
)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Parameters ──
SUBJECT = "sub-01"
SFREQ_TARGET = 250  # Hz — matches Frank 2025, halves compute vs 500 Hz
MIN_KAPPA = 30  # Frank 2025 recommendation
MAX_DURATION_SEC = None  # Use all available data (kappa-driven)
METHODS = ["amica", "picard", "infomax", "fastica"]
MAX_ITER = 2000  # Frank 2023 recommendation
AMICA_NUM_MIX = 3  # Frank 2023 recommendation


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

def load_and_preprocess(subject):
    """Load ds004505, classify channels, preprocess for ICA.

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
    # No redundant filtering needed. Just apply average reference.
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
            fit_params={"do_newton": True},
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
    pred = labels["labels"]
    probs = labels["y_pred_proba"]

    counts = {}
    for cat in ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]:
        counts[cat] = sum(1 for l in pred if l == cat)

    brain_probs = probs[:, 0]
    counts["brain_50pct"] = int(np.sum(brain_probs > 0.5))
    counts["brain_70pct"] = int(np.sum(brain_probs > 0.7))

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
    """Reconstruction error: ||X - A @ S|| / ||X||."""
    data = raw.get_data()
    sources = ica.get_sources(raw).get_data()
    reconstructed = ica.mixing_matrix_ @ sources

    # MNE applies PCA/whitening, so reconstruct through the full pipeline
    try:
        raw_recon = ica.apply(raw.copy(), verbose=False)
        recon_data = raw_recon.get_data()
        err = np.linalg.norm(data - recon_data) / np.linalg.norm(data)
    except Exception as e:
        logger.warning("    Reconstruction error failed: %s", e)
        err = float("nan")

    return float(err)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 65)
    logger.info("HIGH-DENSITY EEG VALIDATION (ds004505)")
    logger.info("=" * 65)

    # Load and preprocess
    raw, ch_groups = load_and_preprocess(SUBJECT)
    n_components = determine_n_components(raw)

    all_results = {}

    for method in METHODS:
        logger.info("\n--- %s ---", method.upper())
        try:
            ica, dt = run_ica_method(raw, method, n_components, max_iter=MAX_ITER)

            result = {
                "time": dt,
                "n_iter": int(ica.n_iter_),
                "n_components": int(ica.n_components_),
            }

            # ICLabel (needs onnxruntime or pytorch)
            try:
                ic_counts, _ = run_iclabel(ica, raw)
                result["iclabel"] = ic_counts
            except Exception as e:
                logger.warning("    ICLabel failed: %s", e)
                result["iclabel"] = {"error": str(e)}

            # Kurtosis quality (always works, no extra deps)
            try:
                kurt = compute_kurtosis_quality(ica, raw)
                result["kurtosis"] = kurt
                logger.info(
                    "    Kurtosis: %d/%d brain-like ICs",
                    kurt["brain_like_kurtosis"],
                    kurt["n_components"],
                )
            except Exception as e:
                logger.warning("    Kurtosis scoring failed: %s", e)

            # Reconstruction error
            try:
                recon_err = compute_reconstruction_error(ica, raw)
                result["reconstruction_error"] = recon_err
                logger.info("    Reconstruction error: %.2e", recon_err)
            except Exception as e:
                logger.warning("    Reconstruction error failed: %s", e)

            all_results[method] = result

        except Exception as e:
            logger.error("    %s FAILED: %s", method, e)
            all_results[method] = {"error": str(e)}

    # Save results
    output_file = RESULTS_DIR / "highdens_validation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("\nResults saved to: %s", output_file)

    # Summary table
    logger.info("\n" + "=" * 75)
    logger.info("SUMMARY")
    logger.info("=" * 75)
    header = (
        f"{'Method':<10s} {'Time':>7s} {'Iter':>5s} {'nIC':>4s} "
        f"{'Brain':>6s} {'B>50%':>6s} {'B>70%':>6s} "
        f"{'Musc':>5s} {'Eye':>4s} {'Kurt':>5s} {'Err':>9s}"
    )
    logger.info(header)
    logger.info("-" * 75)

    for method in METHODS:
        r = all_results.get(method, {})
        if "error" in r:
            logger.info("%s  FAILED: %s", method, r["error"][:50])
            continue

        ic = r.get("iclabel", {})
        kurt = r.get("kurtosis", {})
        err = r.get("reconstruction_error", float("nan"))

        logger.info(
            "%-10s %7.1f %5d %4d %6s %6s %6s %5s %4s %5s %9.1e",
            method,
            r["time"],
            r["n_iter"],
            r["n_components"],
            ic.get("brain", "?"),
            ic.get("brain_50pct", "?"),
            ic.get("brain_70pct", "?"),
            ic.get("muscle", "?"),
            ic.get("eye", "?"),
            kurt.get("brain_like_kurtosis", "?"),
            err,
        )

    logger.info("=" * 75)


if __name__ == "__main__":
    main()
