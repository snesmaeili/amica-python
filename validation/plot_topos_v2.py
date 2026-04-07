"""Plot AMICA vs Infomax vs Picard topoplots — proper preprocessing."""
import sys, os
sys.path.insert(0, '/home/sesma/amica-python')

import numpy as np
import mne
import time
import json
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

def plot_ica_grid(ica, labels, title, filename, n_show=30):
    n_show = min(n_show, ica.n_components_)
    ncols = 6
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
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
            ax.text(0.5, 0.5, f'ERR\n{e}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=6)
        ax.set_title(f'IC{idx:02d}: {label}\n({prob:.2f})',
                     fontsize=7, color=color, fontweight='bold')
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.5)

    for idx in range(n_show, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {filename}")


# ── Load and preprocess ──
from validation.run_highdens_validation import load_and_preprocess
raw, n_components = load_and_preprocess("sub-01")

# Use 30 components for faster run + clearer topoplots
n_comp = min(n_components, 30)
print(f"\nUsing {n_comp} components (from max {n_components})")

results = {}

# ── 1. AMICA via MNE integration ──
print("\n=== AMICA via fit_ica (500 iter, m=3) ===")
from amica_python import fit_ica
t0 = time.time()
ica_amica = fit_ica(raw, n_components=n_comp, max_iter=500, num_mix=3, random_state=42,
                    fit_params={"do_newton": True})
dt = time.time() - t0
print(f"  Time: {dt:.1f}s, n_iter: {ica_amica.n_iter_}")

# ── 2. AMICA standalone (own whitening) ──
print("\n=== AMICA standalone (500 iter, m=3) ===")
from amica_python import Amica, AmicaConfig
data = raw.get_data()
t0 = time.time()
config = AmicaConfig(max_iter=500, num_mix_comps=3, do_newton=True,
                     do_mean=True, do_sphere=True, pcakeep=n_comp)
solver = Amica(config, random_state=42)
result = solver.fit(data)
dt_standalone = time.time() - t0
ica_standalone = result.to_mne(raw.info)
print(f"  Time: {dt_standalone:.1f}s, n_iter: {result.n_iter}")

# ── 3. Infomax ──
print("\n=== Infomax (extended) ===")
t0 = time.time()
ica_infomax = mne.preprocessing.ICA(n_components=n_comp, method='infomax',
                                     random_state=42, max_iter=500,
                                     fit_params=dict(extended=True))
ica_infomax.fit(raw, verbose=False)
dt_infomax = time.time() - t0
print(f"  Time: {dt_infomax:.1f}s, n_iter: {ica_infomax.n_iter_}")

# ── 4. Picard ──
print("\n=== Picard ===")
t0 = time.time()
ica_picard = mne.preprocessing.ICA(n_components=n_comp, method='picard',
                                    random_state=42, max_iter=500,
                                    fit_params=dict(ortho=False, extended=True))
ica_picard.fit(raw, verbose=False)
dt_picard = time.time() - t0
print(f"  Time: {dt_picard:.1f}s, n_iter: {ica_picard.n_iter_}")

# ── ICLabel all ──
from mne_icalabel import label_components

all_icas = [
    ("AMICA MNE", ica_amica, "amica_mne"),
    ("AMICA standalone", ica_standalone, "amica_standalone"),
    ("Infomax", ica_infomax, "infomax"),
    ("Picard", ica_picard, "picard"),
]

print("\n=== ICLabel + Topoplots ===")
for name, ica, key in all_icas:
    try:
        lab = label_components(raw, ica, method='iclabel')
        pred = lab['labels']
        probs = np.array(lab['y_pred_proba'])
        brain_mask = np.array(pred) == 'brain'

        n_brain = int(brain_mask.sum())
        n_b50 = int(np.sum(brain_mask & (probs > 0.5)))
        n_b70 = int(np.sum(brain_mask & (probs > 0.7)))
        n_muscle = sum(1 for l in pred if l == 'muscle artifact')
        n_eye = sum(1 for l in pred if l == 'eye blink')
        n_other = sum(1 for l in pred if l == 'other')

        print(f"{name:20s}: Brain={n_brain:2d} (>50%:{n_b50}, >70%:{n_b70}), "
              f"Muscle={n_muscle}, Eye={n_eye}, Other={n_other}")

        plot_ica_grid(ica, lab,
                      f'{name} — ds004505 sub-01 ({ica.n_components_} ICs)',
                      f'topos_v2_{key}.png')

    except Exception as e:
        import traceback
        print(f"{name:20s}: ICLabel/plot FAILED: {e}")
        traceback.print_exc()

print("\nDone. Figures in:", FIGURES_DIR)
