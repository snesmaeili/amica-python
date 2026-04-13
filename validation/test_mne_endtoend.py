"""Full MNE end-to-end validation: fit_ica() on sub-01 + all MNE features.

Tests: source extraction, topoplots, source plots, properties,
ICLabel, save/load roundtrip, matrix conventions.
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for sbatch
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("mne").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DS_PATH", "/home/sesma/scratch/ds004505")

from validation.run_highdens_validation import load_and_preprocess, determine_n_components

OUT = Path("validation/results/mne_endtoend")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Step 1: Load and preprocess ----
print("=" * 60)
print("1. PREPROCESSING")
print("=" * 60)
raw, _, _ = load_and_preprocess("sub-01", hp_freq=1.0, iclean_mode="none")
n_comp = determine_n_components(raw, max_components=60)
print(f"raw: {raw.info['nchan']} ch, {raw.n_times} samples, n_comp={n_comp}")

# ---- Step 2: Fit ICA with AMICA via fit_ica() ----
print("\n" + "=" * 60)
print("2. FIT ICA (fit_ica path)")
print("=" * 60)
from amica_python import fit_ica
t0 = time.time()
ica = fit_ica(raw, n_components=n_comp, max_iter=500, num_mix=3,
              random_state=42, fit_params={'lrate': 0.01})
elapsed = time.time() - t0
print(f"fit_ica completed in {elapsed:.1f}s")
print(f"  n_components_: {ica.n_components_}")
print(f"  unmixing_matrix_ shape: {ica.unmixing_matrix_.shape}")
print(f"  mixing_matrix_ shape: {ica.mixing_matrix_.shape}")
print(f"  pca_components_ shape: {ica.pca_components_.shape}")

# ---- Step 3: Matrix conventions ----
print("\n" + "=" * 60)
print("3. MATRIX CONVENTIONS")
print("=" * 60)
n = ica.n_components_
eye_check = ica.mixing_matrix_ @ ica.unmixing_matrix_
eye_err = np.max(np.abs(eye_check - np.eye(n)))
print(f"  mixing @ unmixing ≈ I: max err = {eye_err:.2e} {'OK' if eye_err < 1e-6 else 'FAIL'}")

# Source reconstruction
sources = ica.get_sources(raw)
sources_data = sources.get_data()
print(f"  sources shape: {sources_data.shape}")
print(f"  sources std: {sources_data.std():.4f}")

# ---- Step 4: AMICA-specific results ----
print("\n" + "=" * 60)
print("4. AMICA RESULT")
print("=" * 60)
if hasattr(ica, 'amica_result_') and ica.amica_result_ is not None:
    ar = ica.amica_result_
    rho = np.asarray(ar.rho_)
    n_floor = int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))
    print(f"  rho shape: {rho.shape}")
    print(f"  rho range: [{rho.min():.3f}, {rho.max():.3f}]")
    print(f"  rho floor: {n_floor}/{rho.size}")
    print(f"  converged: {ar.converged}")
    print(f"  n_iter: {ar.n_iter}")
    ll = np.asarray(ar.log_likelihood)
    print(f"  LL: first={ll[0]:.4f} final={ll[-1]:.4f}")
else:
    print("  WARNING: ica.amica_result_ not found!")

# ---- Step 5: Plotting ----
print("\n" + "=" * 60)
print("5. PLOTTING")
print("=" * 60)

try:
    fig = ica.plot_components(picks=range(min(20, n)), show=False)
    if isinstance(fig, list):
        for i, f in enumerate(fig):
            f.savefig(OUT / f"topoplots_{i}.png", dpi=100)
            plt.close(f)
    else:
        fig.savefig(OUT / "topoplots.png", dpi=100)
        plt.close(fig)
    print("  plot_components: OK")
except Exception as e:
    print(f"  plot_components: FAILED ({e})")

try:
    fig = ica.plot_sources(raw, start=0, stop=10, show=False)
    fig.savefig(OUT / "sources.png", dpi=100)
    plt.close(fig)
    print("  plot_sources: OK")
except Exception as e:
    print(f"  plot_sources: FAILED ({e})")

try:
    figs = ica.plot_properties(raw, picks=[0, 1, 2], show=False)
    for i, f in enumerate(figs):
        f.savefig(OUT / f"properties_ic{i}.png", dpi=100)
        plt.close(f)
    print("  plot_properties: OK")
except Exception as e:
    print(f"  plot_properties: FAILED ({e})")

# ---- Step 6: ICLabel ----
print("\n" + "=" * 60)
print("6. ICLABEL")
print("=" * 60)
try:
    from mne_icalabel import label_components
    labels = label_components(raw, ica, method="iclabel")
    brain_count = sum(1 for l in labels['labels'] if l == 'brain')
    print(f"  ICLabel: {brain_count} brain components out of {n}")
    print(f"  Labels: {labels['labels'][:10]}...")
except Exception as e:
    print(f"  ICLabel: FAILED ({e})")

# ---- Step 7: Save/Load roundtrip ----
print("\n" + "=" * 60)
print("7. SAVE/LOAD ROUNDTRIP")
print("=" * 60)
try:
    import mne
    save_path = OUT / "test_ica.fif"
    ica.save(save_path, overwrite=True)
    ica_loaded = mne.preprocessing.read_ica(save_path)
    w_err = np.max(np.abs(ica.unmixing_matrix_ - ica_loaded.unmixing_matrix_))
    print(f"  Save/load: OK (unmixing max err = {w_err:.2e})")
except Exception as e:
    print(f"  Save/load: FAILED ({e})")

# ---- Step 8: Also test the amica() Picard API path ----
print("\n" + "=" * 60)
print("8. PICARD-COMPATIBLE API (ICA(method='amica'))")
print("=" * 60)
try:
    from mne.preprocessing import ICA as MNE_ICA
    ica2 = MNE_ICA(n_components=30, method='amica', random_state=42,
                   fit_params={'max_iter': 100, 'num_mix': 3, 'lrate': 0.01})
    ica2.fit(raw)
    src2 = ica2.get_sources(raw).get_data()
    print(f"  ICA(method='amica'): OK")
    print(f"  n_components_: {ica2.n_components_}")
    print(f"  sources shape: {src2.shape}")
except Exception as e:
    print(f"  ICA(method='amica'): FAILED ({e})")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
