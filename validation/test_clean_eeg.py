"""Quick validation on clean stationary EEG (MNE sample dataset).

Much cleaner than mobile walking data (ds004505). Downloads automatically
if not cached. Uses 59 EEG channels, ~60s of data.
"""
import time, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

OUT = '/home/sesma/amica-python/validation/results/clean_eeg_test'
os.makedirs(OUT, exist_ok=True)

# Download/load MNE sample dataset
print("Loading MNE sample dataset...")
sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    str(sample_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'),
    preload=True, verbose=False
)
raw.pick_types(eeg=True, exclude='bads')
raw.filter(1, 40, verbose=False)
raw.set_eeg_reference('average', verbose=False)
print(f"Data: {raw.info['nchan']} EEG ch, {raw.n_times} samples, {raw.info['sfreq']} Hz")
print(f"Duration: {raw.n_times / raw.info['sfreq']:.0f}s")

n_comp = min(30, raw.info['nchan'] - 1)
print(f"Using {n_comp} components")

# --- Run AMICA via fit_ica ---
print("\n=== AMICA via fit_ica ===")
from amica_python import fit_ica

t0 = time.time()
ica = fit_ica(raw, n_components=n_comp, max_iter=500, num_mix=3,
              random_state=42, fit_params={'lrate': 0.01})
elapsed = time.time() - t0
print(f"Completed in {elapsed:.1f}s")
print(f"n_iter: {ica.n_iter_}")

# LL trajectory
if hasattr(ica, 'amica_result_') and ica.amica_result_ is not None:
    ar = ica.amica_result_
    ll = np.asarray(ar.log_likelihood)
    rho = np.asarray(ar.rho_)
    n_floor = int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))
    n_inc = int(np.sum(np.diff(ll) > 0))
    print(f"LL: {ll[0]:.4f} -> {ll[-1]:.4f}")
    print(f"LL increasing: {n_inc}/{len(ll)-1} ({100*n_inc/(len(ll)-1):.0f}%)")
    print(f"rho: [{rho.min():.3f}, {rho.max():.3f}], floor={n_floor}/{rho.size}")
    print(f"converged: {ar.converged}")

# Matrix check
eye_err = np.max(np.abs(ica.mixing_matrix_ @ ica.unmixing_matrix_ - np.eye(n_comp)))
print(f"mix @ unmix = I err: {eye_err:.2e}")

# --- Topoplots ---
print("\n=== Generating topoplots ===")
n_plot = min(20, n_comp)

# MNE default style
fig_mne = ica.plot_components(picks=range(n_plot), show=False)
if isinstance(fig_mne, list):
    fig_mne[0].savefig(f'{OUT}/topoplots_mne.png', dpi=150)
    for f in fig_mne: plt.close(f)
else:
    fig_mne.savefig(f'{OUT}/topoplots_mne.png', dpi=150)
    plt.close(fig_mne)
print("  MNE style: OK")

# EEGLAB style
fig_eeg = ica.plot_components(picks=range(n_plot), show=False,
    cmap='jet', contours=0, sensors='k.', res=128, size=1.5)
if isinstance(fig_eeg, list):
    fig_eeg[0].savefig(f'{OUT}/topoplots_eeglab_style.png', dpi=150)
    for f in fig_eeg: plt.close(f)
else:
    fig_eeg.savefig(f'{OUT}/topoplots_eeglab_style.png', dpi=150)
    plt.close(fig_eeg)
print("  EEGLAB style: OK")

# --- ICLabel ---
print("\n=== ICLabel ===")
try:
    from mne_icalabel import label_components
    labels = label_components(raw, ica, method="iclabel")
    brain_count = sum(1 for l in labels['labels'] if l == 'brain')
    print(f"  Brain ICs: {brain_count}/{n_comp}")
    print(f"  Labels: {labels['labels']}")
except Exception as e:
    print(f"  FAILED: {e}")

# --- Source properties ---
print("\n=== Source properties ===")
try:
    figs = ica.plot_properties(raw, picks=[0, 1, 2], show=False)
    for i, f in enumerate(figs):
        f.savefig(f'{OUT}/properties_ic{i}.png', dpi=100)
        plt.close(f)
    print("  Properties: OK")
except Exception as e:
    print(f"  Properties: FAILED ({e})")

print(f"\nAll figures saved to {OUT}/")
print("DONE")
