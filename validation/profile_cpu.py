"""CPU performance + memory profiling for AMICA.

Measures on MNE sample dataset (30 comps × 166800 samples × 100 iters):
  - Peak RSS (kernel-reported, in GB)
  - Per-iteration wall time (breakdown: JIT compile, iter 1, iters 2-100, M-step)
  - Memory before/after each phase

Sweeps chunk_size ∈ {None, 8192, 4096, 2048, 1024, 512}.
Writes JSON + generates bar chart.
"""
import gc, json, os, resource, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["JAX_PLATFORMS"] = "cpu"
import mne
from amica_python import fit_ica

OUT = 'validation/results/mne_chunking'
os.makedirs(OUT, exist_ok=True)


def peak_rss_gb():
    """Max resident set size since process start, in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def current_rss_gb():
    """Current RSS (not peak) — sum of lines in /proc/self/status."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        return 0.0
    return 0.0


print("Loading MNE sample dataset...")
sample_path = mne.datasets.sample.data_path()
raw = mne.io.read_raw_fif(
    str(sample_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'),
    preload=True, verbose=False,
)
raw.pick_types(eeg=True, exclude='bads')
raw.filter(1, 40, verbose=False)
raw.set_eeg_reference('average', verbose=False)
n_comp = 30
print(f"Data: {raw.info['nchan']} ch, {raw.n_times} samples")

MAX_ITER = 100
CHUNK_SIZES = [None, 8192, 4096, 2048, 1024, 512]

results = []

for chunk_size in CHUNK_SIZES:
    gc.collect()
    rss_before = current_rss_gb()
    rss_peak_before = peak_rss_gb()

    label = f"chunk_size={chunk_size}" if chunk_size else "full-batch"
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    t0 = time.time()
    ica = fit_ica(
        raw, n_components=n_comp, max_iter=MAX_ITER, num_mix=3,
        random_state=42,
        fit_params={'lrate': 0.01, 'chunk_size': chunk_size},
    )
    t_total = time.time() - t0

    rss_after = current_rss_gb()
    rss_peak = peak_rss_gb()
    rss_delta = rss_peak - rss_peak_before

    ll = np.asarray(ica.amica_result_.log_likelihood)
    per_iter = np.asarray(ica.amica_result_.iteration_times)
    # iter 0 includes JIT compile; iters 1+ are steady-state
    jit_time = float(per_iter[0]) if len(per_iter) > 0 else 0.0
    steady_per_iter = float(np.mean(per_iter[1:])) if len(per_iter) > 1 else 0.0
    steady_total = float(np.sum(per_iter[1:])) if len(per_iter) > 1 else 0.0

    print(f"  wall total:           {t_total:.2f}s")
    print(f"  iter 0 (JIT+iter):    {jit_time:.2f}s")
    print(f"  iters 1-{len(per_iter)-1} sum:       {steady_total:.2f}s")
    print(f"  per-iter (steady):    {steady_per_iter*1000:.1f} ms")
    print(f"  RSS before:           {rss_before:.3f} GB")
    print(f"  RSS after:            {rss_after:.3f} GB")
    print(f"  peak RSS (this run):  {rss_delta:.3f} GB")
    print(f"  LL final:             {ll[-1]:.6f}")

    results.append({
        'chunk_size': chunk_size,
        'wall_total_s': t_total,
        'jit_compile_s': jit_time,
        'steady_per_iter_ms': steady_per_iter * 1000,
        'steady_total_s': steady_total,
        'rss_before_gb': rss_before,
        'rss_after_gb': rss_after,
        'peak_rss_this_run_gb': rss_delta,
        'll_first': float(ll[0]),
        'll_final': float(ll[-1]),
        'n_iter': int(ica.amica_result_.n_iter),
    })

# Save JSON
def _jsonify(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_jsonify(v) for v in obj]
    return obj

with open(f'{OUT}/cpu_profile.json', 'w') as f:
    json.dump(_jsonify({
        'data_shape': [raw.info['nchan'], raw.n_times],
        'n_components': n_comp,
        'max_iter': MAX_ITER,
        'runs': results,
    }), f, indent=2)

# Summary table
print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
print(f"{'chunk_size':>11s}  {'total(s)':>9s}  {'JIT(s)':>8s}  {'iter(ms)':>9s}  {'peak(GB)':>9s}  {'LL final':>10s}")
print("-" * 75)
for r in results:
    cs = 'None' if r['chunk_size'] is None else str(r['chunk_size'])
    print(f"{cs:>11s}  {r['wall_total_s']:9.1f}  {r['jit_compile_s']:8.1f}  "
          f"{r['steady_per_iter_ms']:9.1f}  {r['peak_rss_this_run_gb']:9.3f}  {r['ll_final']:10.6f}")

# Parity: all LL finals should match full-batch
ll_full = next(r['ll_final'] for r in results if r['chunk_size'] is None)
print(f"\nParity check (|LL_chunk - LL_full|):")
for r in results:
    if r['chunk_size'] is None: continue
    d = abs(r['ll_final'] - ll_full)
    flag = 'OK' if d < 1e-4 else 'DRIFT'
    print(f"  chunk={r['chunk_size']:>5d}:  {d:.2e}  {flag}")

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels = ['full' if r['chunk_size'] is None else f"{r['chunk_size']}" for r in results]

axes[0].bar(labels, [r['wall_total_s'] for r in results], color='steelblue')
axes[0].set_ylabel('Wall time (s)'); axes[0].set_title('Total wall time (JIT + 100 iters)')

axes[1].bar(labels, [r['steady_per_iter_ms'] for r in results], color='forestgreen')
axes[1].set_ylabel('ms/iter'); axes[1].set_title('Steady-state per-iter (post-JIT)')

axes[2].bar(labels, [r['peak_rss_this_run_gb'] for r in results], color='firebrick')
axes[2].set_ylabel('Peak RSS (GB)'); axes[2].set_title('Peak RSS during run')

for ax in axes:
    ax.set_xlabel('chunk_size')
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('AMICA CPU profile: MNE sample (30 comps × 166k samples × 100 iters)')
fig.tight_layout()
fig.savefig(f'{OUT}/cpu_profile.png', dpi=120)
plt.close(fig)

print(f"\nSaved: {OUT}/cpu_profile.json, {OUT}/cpu_profile.png")
