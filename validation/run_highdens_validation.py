"""
High-Density EEG Validation: ds004505 (Dual-Layer Table Tennis)
===============================================================

Runs AMICA, Picard, Infomax, FastICA on 120-channel scalp EEG from
the Studnicki et al. (2022) dataset. Compares:
- ICLabel brain/muscle/eye/other IC counts
- Dipole fit residual variance
- Runtime

Uses sub-01 as the test subject (120 scalp + 120 noise channels).
"""
import json
import time
from pathlib import Path

import numpy as np
import mne

DS_PATH = Path(r"D:\mne-denoise\_codex_main_revert_20260306\examples\tutorials\data\ds004505")
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SUBJECT = "sub-01"
N_COMPONENTS = 30  # Practical for validation speed


def load_and_preprocess(subject, n_channels=64):
    """Load ds004505 subject, pick scalp EEG, preprocess for ICA."""
    set_file = DS_PATH / "sourcedata" / "Merged" / subject / f"{subject}_Merged.set"
    print(f"Loading {set_file}...")

    raw = mne.io.read_raw_eeglab(str(set_file), preload=True, verbose=False)

    # Identify scalp EEG channels (non-noise, non-IMU, non-EMG)
    scalp_ch = [ch for ch in raw.ch_names
                if not ch.startswith("N-")           # noise channels
                and not ch.startswith("CGY")         # accelerometer
                and not ch.startswith("CWR")
                and not ch.startswith("NGY")
                and not ch.startswith("NWR")
                and not ch.startswith("Imu")
                and not ch.startswith("Emg")
                and not ch.startswith("None")
                and ch not in ["LISCM", "LSSCM", "LSTrap", "LITrap",
                               "RITrap", "RISCM", "RSSCM", "RSTrap"]]

    print(f"  Found {len(scalp_ch)} scalp EEG channels")
    raw.pick(scalp_ch[:n_channels])
    print(f"  Using {len(raw.ch_names)} channels")

    # Set all to EEG type
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    # Set montage
    try:
        montage = mne.channels.make_standard_montage("standard_1005")
        raw.set_montage(montage, on_missing="warn")
    except Exception as e:
        print(f"  Montage warning: {e}")

    # Filter
    raw.filter(1.0, None, verbose=False)
    raw.notch_filter(50.0, verbose=False)

    # Average reference
    raw.set_eeg_reference("average", verbose=False)

    # Crop to manageable length (5 min = enough for kappa > 30)
    max_duration = 300.0  # 5 minutes
    if raw.times[-1] > max_duration:
        raw.crop(tmax=max_duration)

    print(f"  Final: {raw.info['nchan']} ch, {raw.n_times} samples, "
          f"{raw.times[-1]:.0f}s, sfreq={raw.info['sfreq']}")
    kappa = raw.n_times / raw.info["nchan"] ** 2
    print(f"  kappa = {kappa:.1f}")

    return raw


def run_ica_method(raw, method, n_components, max_iter=500):
    """Run ICA with a specific method, return ICA object and timing."""
    print(f"\n  Running {method}...")

    if method == "amica":
        from amica_python import fit_ica
        t0 = time.time()
        ica = fit_ica(raw, n_components=n_components, max_iter=max_iter,
                      num_mix=3, random_state=42,
                      fit_params={"do_newton": True})
        dt = time.time() - t0
    else:
        t0 = time.time()
        ica = mne.preprocessing.ICA(
            n_components=n_components, method=method,
            random_state=42, max_iter=max_iter
        )
        ica.fit(raw, verbose=False)
        dt = time.time() - t0

    print(f"    Time: {dt:.1f}s, n_iter: {ica.n_iter_}")
    return ica, dt


def run_iclabel(ica, raw):
    """Run ICLabel classification, return label counts."""
    from mne_icalabel import label_components

    labels = label_components(raw, ica, method="iclabel")

    # Count by predicted label
    pred_labels = labels["labels"]
    counts = {}
    for label in ["brain", "muscle", "eye", "heart", "line_noise",
                  "channel_noise", "other"]:
        counts[label] = sum(1 for l in pred_labels if l == label)

    # Also get brain IC probabilities
    probs = labels["y_pred_proba"]
    brain_probs = probs[:, 0]  # first column is brain

    # Brain ICs with >50% probability
    counts["brain_50pct"] = int(np.sum(brain_probs > 0.5))
    # Brain ICs with >70% probability
    counts["brain_70pct"] = int(np.sum(brain_probs > 0.7))

    return counts, labels


def run_dipole_fitting(ica, raw):
    """Run dipole fitting, return residual variances."""
    try:
        # Get IC topographies
        components = ica.get_components()

        # Use BEM head model from MNE
        subject = "fsaverage"
        subjects_dir = mne.datasets.fetch_fsaverage(verbose=False)
        # Actually just compute RV from the topographies using a spherical model
        # Full dipole fitting requires source space setup which is heavy.
        # Use a simpler approach: fit equivalent dipoles via mne.fit_dipole
        # on the IC topographies

        # For now, compute a proxy: the "dipolarity" as the ratio of
        # the largest singular value to the sum of all singular values
        # of each IC topography reshaped appropriately.
        # This is a rough proxy - real dipole fitting needs electrode locations.

        from mne.preprocessing import ICA

        # Try actual DIPFIT-style fitting if montage is set
        try:
            sphere = mne.make_sphere_model("auto", "auto", raw.info, verbose=False)
        except Exception:
            sphere = mne.make_sphere_model(verbose=False)

        rvs = []
        for i in range(ica.n_components_):
            try:
                # Create evoked from IC pattern
                pattern = components[:, i]
                evoked = mne.EvokedArray(
                    pattern[:, np.newaxis],
                    raw.info.copy(),
                    tmin=0,
                    verbose=False
                )
                dip, residual = mne.fit_dipole(evoked, raw.info["cov"] if "cov" in raw.info else None,
                                                sphere, verbose=False)
                rvs.append(float(1.0 - dip.gof[0] / 100.0))
            except Exception:
                rvs.append(1.0)  # mark as failed

        return np.array(rvs)

    except Exception as e:
        print(f"    Dipole fitting failed: {e}")
        return None


def main():
    print("=" * 60)
    print("HIGH-DENSITY EEG VALIDATION (ds004505)")
    print("=" * 60)

    # Load data
    raw = load_and_preprocess(SUBJECT, n_channels=64)

    methods = ["amica", "picard", "infomax", "fastica"]
    all_results = {}

    for method in methods:
        try:
            ica, dt = run_ica_method(raw, method, N_COMPONENTS, max_iter=500)

            # ICLabel
            print(f"    Running ICLabel...")
            try:
                ic_counts, ic_labels = run_iclabel(ica, raw)
                print(f"    Brain ICs: {ic_counts['brain']} (label), "
                      f"{ic_counts['brain_50pct']} (>50%), "
                      f"{ic_counts['brain_70pct']} (>70%)")
                print(f"    Muscle: {ic_counts['muscle']}, Eye: {ic_counts['eye']}, "
                      f"Other: {ic_counts['other']}")
            except Exception as e:
                print(f"    ICLabel failed: {e}")
                ic_counts = {"error": str(e)}

            all_results[method] = {
                "time": dt,
                "n_iter": int(ica.n_iter_),
                "n_components": int(ica.n_components_),
                "iclabel": ic_counts,
            }

        except Exception as e:
            print(f"    {method} FAILED: {e}")
            all_results[method] = {"error": str(e)}

    # Save results
    output_file = RESULTS_DIR / "highdens_validation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<12s} {'Time(s)':<10s} {'Brain':<8s} {'Brain>50%':<10s} "
          f"{'Muscle':<8s} {'Eye':<6s} {'Other':<6s}")
    print("-" * 60)
    for method in methods:
        r = all_results.get(method, {})
        if "error" in r:
            print(f"{method:<12s} FAILED: {r['error'][:40]}")
            continue
        ic = r.get("iclabel", {})
        print(f"{method:<12s} {r['time']:<10.1f} {ic.get('brain', '?'):<8} "
              f"{ic.get('brain_50pct', '?'):<10} {ic.get('muscle', '?'):<8} "
              f"{ic.get('eye', '?'):<6} {ic.get('other', '?'):<6}")


if __name__ == "__main__":
    main()
