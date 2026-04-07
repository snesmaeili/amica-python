"""Quick 200-iter run to visually inspect topoplots before full batch."""
import sys, os
sys.path.insert(0, '/home/sesma/amica-python')

import numpy as np
import mne
import time
import warnings

mne.set_log_level('WARNING')
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

DS_PATH = Path(os.environ.get("DS_PATH", "/home/sesma/scratch/ds004505"))
FIGURES_DIR = Path(__file__).parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'brain': '#2ca02c', 'muscle artifact': '#d62728', 'eye blink': '#1f77b4',
    'heart beat': '#ff7f0e', 'line noise': '#9467bd', 'channel noise': '#8c564b',
    'other': '#7f7f7f'
}

def plot_ica_grid(ica, labels, title, filename, n_show=20):
    n_show = min(n_show, ica.n_components_)
    ncols = 5
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = axes.flatten()
    pred = labels['labels']
    probs = labels['y_pred_proba']
    for idx in range(n_show):
        ax = axes[idx]
        label = pred[idx]
        prob = probs[idx]
        color = COLORS.get(label, '#7f7f7f')
        try:
            ica.plot_components(picks=[idx], axes=ax, show=False,
                               colorbar=False, title="")
        except Exception as e:
            ax.text(0.5, 0.5, f'ERR', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'IC{idx:02d}: {label}\n({prob:.2f})',
                     fontsize=9, color=color, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.5)
    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")

# ── Load ──
from validation.run_highdens_validation import load_and_preprocess
raw, n_components = load_and_preprocess("sub-01")
N = 20  # components for quick run

print(f"\n=== Quick run: {N} components, 200 iters ===\n")

from mne_icalabel import label_components

# 1. AMICA standalone
print("AMICA standalone...")
from amica_python import Amica, AmicaConfig
t0 = time.time()
config = AmicaConfig(max_iter=200, num_mix_comps=3, do_newton=True,
                     do_mean=True, do_sphere=True, pcakeep=N)
result = Amica(config, random_state=42).fit(raw.get_data())
ica_sa = result.to_mne(raw.info)
dt_sa = time.time() - t0
lab_sa = label_components(raw, ica_sa, method='iclabel')
plot_ica_grid(ica_sa, lab_sa, f'AMICA standalone (200 iter) — {dt_sa:.0f}s', 'quick_amica_standalone.png')

# 2. AMICA via MNE
print("AMICA via MNE...")
from amica_python import fit_ica
t0 = time.time()
ica_mne = fit_ica(raw, n_components=N, max_iter=200, num_mix=3, random_state=42,
                  fit_params={"do_newton": True})
dt_mne = time.time() - t0
lab_mne = label_components(raw, ica_mne, method='iclabel')
plot_ica_grid(ica_mne, lab_mne, f'AMICA MNE (200 iter) — {dt_mne:.0f}s', 'quick_amica_mne.png')

# 3. Infomax
print("Infomax...")
t0 = time.time()
ica_info = mne.preprocessing.ICA(n_components=N, method='infomax', random_state=42,
                                  max_iter=200, fit_params=dict(extended=True))
ica_info.fit(raw, verbose=False)
dt_info = time.time() - t0
lab_info = label_components(raw, ica_info, method='iclabel')
plot_ica_grid(ica_info, lab_info, f'Infomax ext (200 iter) — {dt_info:.0f}s', 'quick_infomax.png')

# Summary
print("\n=== Summary ===")
for name, lab in [("AMICA standalone", lab_sa), ("AMICA MNE", lab_mne), ("Infomax", lab_info)]:
    pred = np.array(lab['labels'])
    probs = np.array(lab['y_pred_proba'])
    bm = pred == 'brain'
    print(f"  {name:20s}: Brain={bm.sum():2d} (>50%:{int(np.sum(bm & (probs>0.5)))}), "
          f"Muscle={np.sum(pred=='muscle artifact')}, Eye={np.sum(pred=='eye blink')}, "
          f"Other={np.sum(pred=='other')}")
print("\nDone.")
