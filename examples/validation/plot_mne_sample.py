"""
MNE sample validation
=====================

This advanced example validates jamica on the MNE sample EEG dataset.

It demonstrates that jamica works as a drop-in MNE-compatible ICA method and
produces reproducibility and comparison artifacts:

- AMICA vs. Picard topomap comparison
- same-seed AMICA reproducibility check
- inverse-transform roundtrip check
- JSON manifest with numeric metrics

This example is intended for local validation rather than lightweight
documentation builds.
"""

# %%
# Imports and configuration
# -------------------------

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

N_COMPONENTS = 20
MAX_ITER_AMICA = 3000
MAX_ITER_PICARD = 500
RANDOM_STATE = 42
N_SHOW = 12
HIGHPASS_HZ = 1.0
LOWPASS_HZ = 40.0

OUT_DIR = Path(__file__).resolve().parent / "results"


# %%
# Fitting helpers
# ---------------


def fit_amica(raw, *, n_components: int, max_iter: int, random_state: int):
    """Fit jamica through the standard MNE-compatible entry point."""
    from jamica import fit_ica

    start = time.perf_counter()

    ica = fit_ica(
        raw,
        n_components=int(n_components),
        max_iter=int(max_iter),
        random_state=int(random_state),
    )

    elapsed = time.perf_counter() - start

    return ica, float(elapsed)


def fit_picard(raw, *, n_components: int, max_iter: int, random_state: int):
    """Fit Picard through MNE for a reference ICA comparison."""
    import mne

    ica = mne.preprocessing.ICA(
        n_components=int(n_components),
        method="picard",
        fit_params={
            "ortho": False,
            "extended": True,
            "tol": 1e-6,
        },
        max_iter=int(max_iter),
        random_state=int(random_state),
        verbose="WARNING",
    )

    start = time.perf_counter()
    ica.fit(raw, verbose="WARNING")
    elapsed = time.perf_counter() - start

    return ica, float(elapsed)


# %%
# Metric helpers
# --------------


def reproducibility_diff(ica_a, ica_b) -> dict:
    """Compare two ICA fits using their mixing matrices."""
    mixing_a = np.asarray(ica_a.mixing_matrix_, dtype=np.float64)
    mixing_b = np.asarray(ica_b.mixing_matrix_, dtype=np.float64)

    if mixing_a.shape != mixing_b.shape:
        return {"error": f"shape mismatch: {mixing_a.shape} vs {mixing_b.shape}"}

    frob_diff = float(np.linalg.norm(mixing_a - mixing_b))
    frob_a = float(np.linalg.norm(mixing_a))
    rel_diff = frob_diff / frob_a if frob_a > 0 else float("nan")

    correlations = []

    for idx in range(mixing_a.shape[1]):
        col_a = mixing_a[:, idx] - mixing_a[:, idx].mean()
        col_b = mixing_b[:, idx] - mixing_b[:, idx].mean()

        denom = np.linalg.norm(col_a) * np.linalg.norm(col_b)

        if denom > 0:
            correlations.append(float((col_a @ col_b) / denom))
        else:
            correlations.append(float("nan"))

    abs_correlations = [abs(corr) for corr in correlations]

    return {
        "n_components": int(mixing_a.shape[1]),
        "frob_norm_a": frob_a,
        "frob_norm_diff": frob_diff,
        "frob_rel_diff": rel_diff,
        "per_component_signed_corr": correlations,
        "per_component_abs_corr_median": float(np.median(abs_correlations)),
        "per_component_abs_corr_min": float(min(abs_correlations)),
    }


def inverse_transform_check(raw, ica) -> dict:
    """Check that applying ICA with no excluded components reconstructs data."""
    raw_original = raw.copy()
    raw_applied = ica.apply(raw.copy(), exclude=[], verbose="WARNING")

    data_original = raw_original.get_data()
    data_applied = raw_applied.get_data()

    diff = data_original - data_applied
    denom = max(float(np.linalg.norm(data_original)), 1e-30)

    return {
        "raw_n_channels": int(data_original.shape[0]),
        "raw_n_samples": int(data_original.shape[1]),
        "frob_rel_residual": float(np.linalg.norm(diff)) / denom,
        "max_abs_residual_volts": float(np.max(np.abs(diff))),
    }


# %%
# Plotting helpers
# ----------------


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    """Z-score columns by centering and L2-normalizing."""
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    return centered / norms


def make_topomap_grid(ica_amica, ica_picard, raw, n_show: int, out_dir: Path):
    """Create a side-by-side AMICA/Picard topomap grid."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import mne
    from scipy.optimize import linear_sum_assignment

    amica_components = np.asarray(ica_amica.get_components(), dtype=np.float64)
    picard_components = np.asarray(ica_picard.get_components(), dtype=np.float64)

    n_show = min(n_show, amica_components.shape[1])
    amica_subset = amica_components[:, :n_show]

    amica_norm = _zscore_columns(amica_subset)
    picard_norm = _zscore_columns(picard_components)

    corr = amica_norm.T @ picard_norm
    row_idx, col_idx = linear_sum_assignment(-np.abs(corr))

    matched_corr = np.array([corr[row, col_idx[idx]] for idx, row in enumerate(row_idx)])
    matched_abs = np.abs(matched_corr)
    matched_sign = np.sign(matched_corr)
    matched_sign[matched_sign == 0] = 1.0

    fig, axes = plt.subplots(
        2,
        n_show,
        figsize=(1.6 * n_show, 3.7),
        gridspec_kw={
            "hspace": 0.55,
            "wspace": 0.1,
        },
    )

    if n_show == 1:
        axes = axes.reshape(2, 1)

    info_amica = mne.pick_info(
        raw.info,
        mne.pick_channels(
            raw.info["ch_names"],
            ica_amica.ch_names,
            ordered=True,
        ),
    )
    info_picard = mne.pick_info(
        raw.info,
        mne.pick_channels(
            raw.info["ch_names"],
            ica_picard.ch_names,
            ordered=True,
        ),
    )

    for component_idx in range(n_show):
        ax_amica = axes[0, component_idx]

        mne.viz.plot_topomap(
            amica_subset[:, component_idx],
            info_amica,
            axes=ax_amica,
            show=False,
            contours=4,
            sphere="auto",
        )

        ax_amica.set_title(f"IC {component_idx}", fontsize=8)

        ax_picard = axes[1, component_idx]

        picard_idx = int(col_idx[component_idx])
        picard_topography = (
            matched_sign[component_idx]
            * picard_components[
                :,
                picard_idx,
            ]
        )

        mne.viz.plot_topomap(
            picard_topography,
            info_picard,
            axes=ax_picard,
            show=False,
            contours=4,
            sphere="auto",
        )

        ax_picard.set_title(
            f"IC {picard_idx} |r|={matched_abs[component_idx]:.2f}",
            fontsize=8,
        )

    axes[0, 0].set_ylabel("AMICA", fontsize=11)
    axes[1, 0].set_ylabel("Picard\nmatched", fontsize=11)

    fig.suptitle(
        "MNE sample EEG: AMICA components and matched Picard components",
        fontsize=11,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "fig_mne_sample_topomaps.pdf", bbox_inches="tight")
    fig.savefig(
        out_dir / "fig_mne_sample_topomaps.png",
        bbox_inches="tight",
        dpi=150,
    )

    plt.close(fig)

    return {
        "amica_indices": list(range(n_show)),
        "picard_indices_matched": [int(value) for value in col_idx[:n_show]],
        "matched_abs_corr": [float(value) for value in matched_abs[:n_show]],
        "matched_signed_corr": [float(value) for value in matched_corr[:n_show]],
    }


def make_reproducibility_figure(ica_a, ica_b, raw, n_show: int, out_dir: Path):
    """Plot components from two same-seed AMICA fits."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import mne

    components_a = ica_a.get_components()
    components_b = ica_b.get_components()

    n_show = min(n_show, components_a.shape[1], components_b.shape[1])

    fig, axes = plt.subplots(
        2,
        n_show,
        figsize=(1.6 * n_show, 3.6),
        gridspec_kw={
            "hspace": 0.4,
            "wspace": 0.1,
        },
    )

    if n_show == 1:
        axes = axes.reshape(2, 1)

    info_used = mne.pick_info(
        raw.info,
        mne.pick_channels(
            raw.info["ch_names"],
            ica_a.ch_names,
            ordered=True,
        ),
    )

    for component_idx in range(n_show):
        mne.viz.plot_topomap(
            components_a[:, component_idx],
            info_used,
            axes=axes[0, component_idx],
            show=False,
            contours=4,
            sphere="auto",
        )
        axes[0, component_idx].set_title(f"IC {component_idx}", fontsize=8)

        mne.viz.plot_topomap(
            components_b[:, component_idx],
            info_used,
            axes=axes[1, component_idx],
            show=False,
            contours=4,
            sphere="auto",
        )

    axes[0, 0].set_ylabel("AMICA fit 1\nseed=42", fontsize=10)
    axes[1, 0].set_ylabel("AMICA fit 2\nseed=42", fontsize=10)

    fig.suptitle(
        "AMICA reproducibility on MNE sample EEG: two same-seed fits",
        fontsize=11,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        out_dir / "fig_mne_sample_reproducibility.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out_dir / "fig_mne_sample_reproducibility.png",
        bbox_inches="tight",
        dpi=150,
    )

    plt.close(fig)


# %%
# Load MNE sample EEG
# -------------------


def load_mne_sample_eeg():
    """Load and preprocess the MNE sample EEG recording."""
    import mne

    sample_path = mne.datasets.sample.data_path()
    raw_fname = sample_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"

    raw = mne.io.read_raw_fif(
        raw_fname,
        preload=True,
        verbose="WARNING",
    )

    raw.pick("eeg")
    raw.filter(HIGHPASS_HZ, LOWPASS_HZ, verbose="WARNING")
    raw.set_eeg_reference(
        "average",
        projection=False,
        verbose="WARNING",
    )

    return raw


raw = load_mne_sample_eeg()

print(
    f"Raw data: {len(raw.ch_names)} EEG channels, "
    f"{raw.n_times} samples @ {raw.info['sfreq']:.1f} Hz"
)


# %%
# Fit AMICA twice
# ---------------
#
# Two runs with the same seed are used to check reproducibility.

ica_amica_a, runtime_amica_a = fit_amica(
    raw,
    n_components=N_COMPONENTS,
    max_iter=MAX_ITER_AMICA,
    random_state=RANDOM_STATE,
)

ica_amica_b, runtime_amica_b = fit_amica(
    raw,
    n_components=N_COMPONENTS,
    max_iter=MAX_ITER_AMICA,
    random_state=RANDOM_STATE,
)

print(f"AMICA fit 1: {runtime_amica_a:.1f}s, n_iter={ica_amica_a.n_iter_}")
print(f"AMICA fit 2: {runtime_amica_b:.1f}s, n_iter={ica_amica_b.n_iter_}")


# %%
# Fit Picard reference
# --------------------
#
# Picard is used here as a reference ICA method through MNE.

ica_picard, runtime_picard = fit_picard(
    raw,
    n_components=N_COMPONENTS,
    max_iter=MAX_ITER_PICARD,
    random_state=RANDOM_STATE,
)

print(f"Picard fit: {runtime_picard:.1f}s, n_iter={ica_picard.n_iter_}")


# %%
# Compute validation metrics
# --------------------------

reproducibility = reproducibility_diff(ica_amica_a, ica_amica_b)
roundtrip = inverse_transform_check(raw, ica_amica_a)

print(f"AMICA same-seed relative difference: {reproducibility['frob_rel_diff']:.2e}")
print(f"AMICA inverse-transform residual: {roundtrip['frob_rel_residual']:.2e}")


# %%
# Generate figures
# ----------------

matching = make_topomap_grid(
    ica_amica_a,
    ica_picard,
    raw,
    N_SHOW,
    OUT_DIR,
)

make_reproducibility_figure(
    ica_amica_a,
    ica_amica_b,
    raw,
    N_SHOW,
    OUT_DIR,
)


# %%
# Save validation manifest
# ------------------------

manifest = {
    "_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "_hostname": platform.node(),
    "_python_version": sys.version.split()[0],
    "config": {
        "n_components": int(N_COMPONENTS),
        "max_iter_amica": int(MAX_ITER_AMICA),
        "max_iter_picard": int(MAX_ITER_PICARD),
        "random_state": int(RANDOM_STATE),
        "highpass_hz": float(HIGHPASS_HZ),
        "lowpass_hz": float(LOWPASS_HZ),
        "n_show": int(N_SHOW),
    },
    "raw_info": {
        "dataset": "mne.datasets.sample",
        "filename": "sample_audvis_filt-0-40_raw.fif",
        "modality": "eeg",
        "n_channels": len(raw.ch_names),
        "n_samples": int(raw.n_times),
        "sfreq_hz": float(raw.info["sfreq"]),
    },
    "amica_fit_1": {
        "runtime_s": float(runtime_amica_a),
        "n_iter": int(ica_amica_a.n_iter_),
    },
    "amica_fit_2": {
        "runtime_s": float(runtime_amica_b),
        "n_iter": int(ica_amica_b.n_iter_),
    },
    "picard_fit": {
        "runtime_s": float(runtime_picard),
        "n_iter": int(ica_picard.n_iter_),
    },
    "reproducibility_amica_vs_amica_same_seed": reproducibility,
    "inverse_transform_roundtrip_amica": roundtrip,
    "topomap_amica_picard_matching": matching,
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

(OUT_DIR / "mne_sample_demo.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print(f"Saved validation artifacts to: {OUT_DIR}")
