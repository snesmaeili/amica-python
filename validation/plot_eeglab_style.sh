#!/bin/bash
#SBATCH --job-name=eeglab-topoplots
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/eeglab-topoplots-%j.out
#SBATCH --error=logs/eeglab-topoplots-%j.err
#SBATCH --chdir=/home/sesma/amica-python
source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu

python -u -c "
import mne, pickle, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load the saved ICA from the end-to-end test
ica = mne.preprocessing.read_ica('validation/results/mne_endtoend/test_ica.fif')
print(f'Loaded ICA: {ica.n_components_} components')

picks = list(range(min(20, ica.n_components_)))

# --- MNE default style (what we have now) ---
fig_mne = ica.plot_components(picks=picks, show=False,
    cmap='RdBu_r',       # default
    contours=6,           # default
    colorbar=False,       # default
    sensors=True,         # default
    outlines='head',      # default
    res=64,               # default
)
if isinstance(fig_mne, list):
    fig_mne[0].savefig('validation/results/mne_endtoend/topoplots_mne_style.png', dpi=150)
    for f in fig_mne: plt.close(f)
else:
    fig_mne.savefig('validation/results/mne_endtoend/topoplots_mne_style.png', dpi=150)
    plt.close(fig_mne)
print('MNE style saved')

# --- EEGLAB style ---
fig_eeg = ica.plot_components(picks=picks, show=False,
    cmap='jet',           # EEGLAB's default colormap
    contours=0,           # EEGLAB: no contour lines (filled only)
    colorbar=False,
    sensors='k.',         # small black dots for electrodes
    outlines='head',
    res=128,              # higher resolution like EEGLAB
    image_interp='cubic',
    size=1.5,             # slightly larger
)
if isinstance(fig_eeg, list):
    fig_eeg[0].savefig('validation/results/mne_endtoend/topoplots_eeglab_style.png', dpi=150)
    for f in fig_eeg: plt.close(f)
else:
    fig_eeg.savefig('validation/results/mne_endtoend/topoplots_eeglab_style.png', dpi=150)
    plt.close(fig_eeg)
print('EEGLAB style saved')

print('Both saved to validation/results/mne_endtoend/')
"
