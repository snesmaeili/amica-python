"""Test: run Python AMICA with Fortran's sphering matrix on sub-01.

If Python's LL matches Fortran's trajectory (+5.34 → +8.70), the sphering
is the ONLY issue. If it still crashes, there's a bug in the AMICA step.
"""
import numpy as np
import time
import pickle
from pathlib import Path

import jax
print(f"JAX devices: {jax.devices()}")

from amica_python import Amica, AmicaConfig

FDIR = Path("validation/results/post_f1_audit/fortran_output")
PREPROC = Path("validation/results/post_f1_audit/sub01_preproc.npz")

# Load Fortran's sphering matrix and mean
S_fortran = np.fromfile(FDIR / "S", dtype=np.float64).reshape((118, 118), order='F')
mean_fortran = np.fromfile(FDIR / "mean", dtype=np.float64)
print(f"Fortran S shape: {S_fortran.shape}, mean shape: {mean_fortran.shape}")

# The Fortran S is full 118x118 but only first 60 rows are the kept components.
# Extract the 60x118 sphere (PCA-reduced).
n_comp = 60
S_reduced = S_fortran[:n_comp, :]
print(f"S_reduced shape: {S_reduced.shape} (first {n_comp} rows)")
print(f"S_reduced row norms: min={np.linalg.norm(S_reduced, axis=1).min():.4f}, "
      f"max={np.linalg.norm(S_reduced, axis=1).max():.4f}")

# Load raw data
z = np.load(PREPROC, allow_pickle=True)
data = z['data']  # (118, 1206350)
print(f"data: {data.shape}")

# Run Python with Fortran's sphering
cfg = AmicaConfig(
    num_models=1, num_mix_comps=3, max_iter=20, pcakeep=n_comp,
    dtype="float64",
)
t0 = time.time()
m = Amica(cfg, random_state=42)
res = m.fit(data, init_mean=mean_fortran, init_sphere=S_reduced)
elapsed = time.time() - t0

ll = np.asarray(res.log_likelihood)
rho = np.asarray(res.rho_)
n_floor = int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))

print(f"\n{'='*70}")
print(f"PYTHON with FORTRAN SPHERING ({jax.devices()[0]}) — {len(ll)} iters, {elapsed:.1f}s")
print(f"{'='*70}")
for i in range(len(ll)):
    d = ll[i] - ll[i-1] if i > 0 else 0
    flag = " <<<" if d < -0.01 and i > 0 else ""
    print(f"  {i:3d}: LL={ll[i]:12.6f}  dll={d:+.6e}{flag}")
print(f"  rho_floor: {n_floor}/{rho.size}")

# Also run with Python's OWN sphering for comparison
print(f"\n{'='*70}")
print(f"PYTHON with OWN SPHERING (control) — 20 iters")
print(f"{'='*70}")
t0 = time.time()
m2 = Amica(cfg, random_state=42)
res2 = m2.fit(data)
elapsed2 = time.time() - t0
ll2 = np.asarray(res2.log_likelihood)
for i in range(len(ll2)):
    d = ll2[i] - ll2[i-1] if i > 0 else 0
    flag = " <<<" if d < -0.01 and i > 0 else ""
    print(f"  {i:3d}: LL={ll2[i]:12.6f}  dll={d:+.6e}{flag}")

# Print Fortran reference
print(f"\n{'='*70}")
print("FORTRAN REFERENCE (from log)")
print(f"{'='*70}")
fortran_log = Path("validation/results/post_f1_audit/fortran_20iter.log")
if fortran_log.exists():
    with open(fortran_log) as f:
        for line in f:
            if line.strip().startswith('iter'):
                print(f"  {line.strip()}")
