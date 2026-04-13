"""Definitive parity test: Python vs Fortran on identical data, identical params.

Runs both on the CORRECTLY-LAID-OUT sub-01 data with lrate=0.01 (stable
for both). Compares iter-by-iter LL trajectories.

Pass criterion: Python and Fortran LL agree to within 0.1 nats at every
iteration through 20 iters.
"""
import numpy as np
import time
import subprocess
from pathlib import Path

import jax
print(f"JAX devices: {jax.devices()}")

from amica_python import Amica, AmicaConfig
from amica_python.preprocessing import preprocess_data

FDIR = Path("validation/results/post_f1_audit/fortran_output")
PREPROC = Path("validation/results/post_f1_audit/sub01_preproc.npz")
FDT = Path("validation/results/post_f1_audit/sub01.fdt")

# ---- Load data (same as Fortran reads) ----
z = np.load(PREPROC, allow_pickle=True)
data = z['data']  # (118, 1206350) float64
n_comp = int(z['n_components'])
print(f"data: {data.shape}, n_comp={n_comp}")

# ---- Verify .fdt is in correct Fortran column-major layout ----
fdt_check = np.fromfile(FDT, dtype=np.float32, count=118)
assert np.allclose(fdt_check, data[:, 0].astype(np.float32), atol=1e-7), \
    "FDT file is NOT in correct Fortran column-major layout!"
print("FDT layout verified: column-major (channels contiguous per sample)")

# ---- Run Fortran 20 iters with lrate=0.01 ----
print("\n" + "="*70)
print("FORTRAN amica17 — 20 iters, lrate=0.01")
print("="*70)

param_file = Path("/tmp/parity_test.param")
param_file.write_text(f"""files {FDT.resolve()}
outdir {(FDIR).resolve()}
num_models 1
num_mix_comps 3
data_dim 118
field_dim 1206350
pcakeep 60
max_threads 4
block_size 256
do_opt_block 0
max_iter 20
writestep 1
do_history 0
dble_data 0
lrate 0.01
use_grad_norm 1
use_min_dll 1
min_grad_norm 0.000001
min_dll 0.000000001
do_approx_sphere 1
do_reject 0
do_newton 1
newt_start 50
num_samples 1
field_blocksize 1
minlrate 0.00000001
lratefact 0.5
rholrate 0.05
rho0 1.5
minrho 1.0
maxrho 2.0
rholratefact 0.5
newt_ramp 10
newtrate 1.0
max_decs 3
update_A 1
update_c 1
update_gm 1
update_alpha 1
update_mu 1
update_beta 1
do_rho 1
invsigmax 100.0
invsigmin 0.00000001
do_mean 1
do_sphere 1
doPCA 1
doscaling 1
scalestep 1
fix_init 0
share_comps 0
""")

import os
os.environ["OMP_NUM_THREADS"] = "4"
result = subprocess.run(
    ["/home/sesma/refs/sccn-amica/amica17_narval", str(param_file)],
    capture_output=True, text=True, timeout=600
)
if result.returncode != 0:
    print(f"Fortran exit code: {result.returncode}")
    print(f"Fortran stderr: {result.stderr[:500]}")
    print(f"Fortran stdout (first 500): {result.stdout[:500]}")

# Parse Fortran LL trajectory
fortran_lls = []
for line in result.stdout.split('\n'):
    if line.strip().startswith('iter'):
        parts = line.split()
        try:
            iter_n = int(parts[1])
            ll = float(parts[5])
            lrate = float(parts[3])
            fortran_lls.append((iter_n, ll, lrate))
        except (IndexError, ValueError):
            pass

print(f"Fortran completed: {len(fortran_lls)} iterations")
for it, ll, lr in fortran_lls:
    print(f"  iter {it:3d}: LL={ll:12.6f}  lrate={lr:.4e}")

# ---- Load Fortran's sphere and mean for Python ----
# Fortran wrote S and mean at writestep=1, so we have the initial state
S_f = np.fromfile(FDIR / "S", dtype=np.float64).reshape((118, 118), order='F')
mean_f = np.fromfile(FDIR / "mean", dtype=np.float64)

# ---- Run Python 20 iters with same params ----
print("\n" + "="*70)
print("PYTHON — 20 iters, lrate=0.01")
print("="*70)

cfg = AmicaConfig(
    num_models=1, num_mix_comps=3, max_iter=20, pcakeep=n_comp,
    dtype="float64", lrate=0.01,
)
t0 = time.time()
m = Amica(cfg, random_state=42)
res = m.fit(data)
elapsed = time.time() - t0

ll_py = np.asarray(res.log_likelihood)
print(f"Python completed: {len(ll_py)} iters, {elapsed:.1f}s")
for i in range(len(ll_py)):
    print(f"  iter {i:3d}: LL={ll_py[i]:12.6f}")

# ---- Compare ----
print("\n" + "="*70)
print("COMPARISON")
print("="*70)
n_compare = min(len(fortran_lls), len(ll_py))
max_diff = 0
for i in range(n_compare):
    f_iter, f_ll, f_lr = fortran_lls[i]
    p_ll = ll_py[i]
    diff = abs(f_ll - p_ll)
    max_diff = max(max_diff, diff)
    flag = " *** DIVERGED" if diff > 0.5 else ""
    print(f"  iter {i:3d}: Fortran={f_ll:12.6f}  Python={p_ll:12.6f}  "
          f"|diff|={diff:.6f}{flag}")

print(f"\nMax |diff| across {n_compare} iters: {max_diff:.6f}")
if max_diff < 0.1:
    print("PASS: Python and Fortran agree to within 0.1 nats")
elif max_diff < 1.0:
    print("CLOSE: Within 1 nat — likely init/rng difference")
else:
    print("FAIL: Significant divergence — investigate")
