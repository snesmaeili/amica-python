"""Validate chunked E-step on MNE sample dataset (clean EEG).

Runs AMICA with chunk_size=None and chunk_size=1024 on the SAME data,
SAME random seed, SAME max_iter. Reports:
  - wall time
  - peak RSS
  - LL trajectory per iteration (side by side)
  - final W relative error
  - topoplot agreement
"""
import time, resource, gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne

from amica_python import fit_ica

OUT = 'validation/results/mne_chunking'
import os
os.makedirs(OUT, exist_ok=True)

def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

print("Loading MNE sample dataset (clean stationary EEG)...")
sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    str(sample_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'),
    preload=True, verbose=False,
)
raw.pick_types(eeg=True, exclude='bads')
raw.filter(1, 40, verbose=False)
raw.set_eeg_reference('average', verbose=False)
n_comp = min(30, raw.info['nchan'] - 1)
print(f"Data: {raw.info['nchan']} EEG ch, {raw.n_times} samples, {raw.info['sfreq']} Hz")
print(f"Using {n_comp} components")

MAX_ITER = 100

def run(chunk_size, label):
    gc.collect()
    print(f"\n{'='*60}\n{label} (chunk_size={chunk_size})\n{'='*60}")
    t0 = time.time()
    rss_before = peak_rss_gb()
    ica = fit_ica(
        raw, n_components=n_comp, max_iter=MAX_ITER, num_mix=3,
        random_state=42,
        fit_params={'lrate': 0.01, 'chunk_size': chunk_size},
    )
    wall = time.time() - t0
    rss_peak = peak_rss_gb()
    print(f"  wall: {wall:.1f}s")
    print(f"  peak RSS: {rss_peak:.2f} GB (+{rss_peak - rss_before:.2f})")
    ll = np.asarray(ica.amica_result_.log_likelihood)
    print(f"  LL trajectory: first={ll[0]:.6f} final={ll[-1]:.6f}")
    print(f"  n_iter: {ica.amica_result_.n_iter}")
    print(f"  converged: {ica.amica_result_.converged}")
    rho = np.asarray(ica.amica_result_.rho_)
    print(f"  rho range: [{rho.min():.3f}, {rho.max():.3f}]")
    print(f"  rho floor: {int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))}/{rho.size}")
    return {
        'wall': wall,
        'rss_peak': rss_peak,
        'rss_delta': rss_peak - rss_before,
        'll': ll,
        'ica': ica,
        'rho': rho,
    }

r_full = run(None, 'FULL-BATCH')
r_chunk = run(1024, 'CHUNKED (chunk=1024)')

# Matrix agreement
W_full = r_full['ica'].unmixing_matrix_
W_chunk = r_chunk['ica'].unmixing_matrix_
rel = np.max(np.abs(W_chunk - W_full)) / max(np.max(np.abs(W_full)), 1e-20)

# Construct sensor-space unmixing: W @ pca_components[:n_comp] @ diag(1/pre_whitener)
def sensor_W(ica):
    pw = np.asarray(ica.pre_whitener_).flatten()
    pca = np.asarray(ica.pca_components_[:ica.n_components_])
    return np.asarray(ica.unmixing_matrix_) @ pca / pw[None, :]

Ws_full = sensor_W(r_full['ica'])
Ws_chunk = sensor_W(r_chunk['ica'])
rel_sensor = np.max(np.abs(Ws_chunk - Ws_full)) / max(np.max(np.abs(Ws_full)), 1e-20)

print(f"\n{'='*60}\nCOMPARISON\n{'='*60}")
print(f"max|W_chunk - W_full|      / max|W_full|      = {rel:.3e}")
print(f"max|Ws_chunk - Ws_full|    / max|Ws_full|    = {rel_sensor:.3e}")
print(f"|LL_final chunk - full|                       = {abs(r_chunk['ll'][-1] - r_full['ll'][-1]):.3e}")
print(f"Wall ratio (chunk/full):  {r_chunk['wall']/r_full['wall']:.2f}x")
print(f"RSS ratio (full/chunk):   {r_full['rss_delta']/max(r_chunk['rss_delta'], 0.01):.2f}x")

# LL trajectory side by side
print(f"\nLL trajectory (iter by iter):")
print(f"{'iter':>5s} {'full':>12s} {'chunk':>12s} {'diff':>12s}")
print("-" * 45)
for i in range(min(len(r_full['ll']), len(r_chunk['ll']))):
    d = r_chunk['ll'][i] - r_full['ll'][i]
    print(f"{i:5d} {r_full['ll'][i]:12.6f} {r_chunk['ll'][i]:12.6f} {d:+.3e}")

# LL trajectory plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(r_full['ll'], 'b-', label='full-batch (chunk_size=None)', lw=2)
ax.plot(r_chunk['ll'], 'r--', label='chunked (chunk_size=1024)', lw=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('Log-likelihood')
ax.set_title('Full-batch vs chunked E-step on MNE sample dataset (CPU)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(f'{OUT}/ll_trajectory.png', dpi=120, bbox_inches='tight')
plt.close(fig)

# Topoplot comparison: side by side
fig_full = r_full['ica'].plot_components(picks=range(min(20, n_comp)), show=False)
if isinstance(fig_full, list): fig_full = fig_full[0]
fig_full.savefig(f'{OUT}/topoplots_full.png', dpi=120)
plt.close(fig_full)

fig_chunk = r_chunk['ica'].plot_components(picks=range(min(20, n_comp)), show=False)
if isinstance(fig_chunk, list): fig_chunk = fig_chunk[0]
fig_chunk.savefig(f'{OUT}/topoplots_chunk.png', dpi=120)
plt.close(fig_chunk)

# Summary JSON
import json
summary = {
    'max_iter': MAX_ITER,
    'n_components': int(n_comp),
    'n_samples': int(raw.n_times),
    'n_channels': int(raw.info['nchan']),
    'full_batch': {
        'wall_s': float(r_full['wall']),
        'peak_rss_gb': float(r_full['rss_peak']),
        'rss_delta_gb': float(r_full['rss_delta']),
        'll_first': float(r_full['ll'][0]),
        'll_final': float(r_full['ll'][-1]),
        'n_iter': int(r_full['ica'].amica_result_.n_iter),
    },
    'chunked': {
        'chunk_size': 1024,
        'wall_s': float(r_chunk['wall']),
        'peak_rss_gb': float(r_chunk['rss_peak']),
        'rss_delta_gb': float(r_chunk['rss_delta']),
        'll_first': float(r_chunk['ll'][0]),
        'll_final': float(r_chunk['ll'][-1]),
        'n_iter': int(r_chunk['ica'].amica_result_.n_iter),
    },
    'parity': {
        'W_rel_err': float(rel),
        'W_sensor_rel_err': float(rel_sensor),
        'll_diff_final': float(abs(r_chunk['ll'][-1] - r_full['ll'][-1])),
    },
    'speed': {
        'wall_ratio_chunk_over_full': float(r_chunk['wall']/r_full['wall']),
        'rss_ratio_full_over_chunk': float(r_full['rss_delta']/max(r_chunk['rss_delta'], 0.01)),
    },
}
with open(f'{OUT}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nFiles saved to {OUT}/")
print("  ll_trajectory.png, topoplots_full.png, topoplots_chunk.png, summary.json")
