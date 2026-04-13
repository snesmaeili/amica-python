"""20-iteration head-to-head: Python vs Fortran on sub-01.

Prints iter-by-iter LL, lrate, and ||dA|| for comparison.
Runs Python directly (CPU or GPU depending on available device).
Fortran output is loaded from a separate run.
"""
from __future__ import annotations
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("mne").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DS_PATH", "/home/sesma/scratch/ds004505")

import jax
print(f"JAX devices: {jax.devices()}")

from amica_python import Amica, AmicaConfig

PREPROC = Path("validation/results/post_f1_audit/sub01_preproc.npz")
FORTRAN_20_LOG = Path("validation/results/post_f1_audit/fortran_20iter.log")

# Load data
z = np.load(PREPROC, allow_pickle=True)
data = z['data']
n_comp = int(z['n_components'])
print(f"data: {data.shape}, n_comp={n_comp}")

# Run Python 20 iters
cfg = AmicaConfig(
    num_models=1, num_mix_comps=3, max_iter=20, pcakeep=n_comp,
    dtype="float64",
)
t0 = time.time()
res = Amica(cfg, random_state=42).fit(data)
elapsed = time.time() - t0

ll_py = np.asarray(res.log_likelihood)
rho_py = np.asarray(res.rho_)
n_floor_py = int(np.sum(np.isclose(rho_py, 1.0, atol=1e-6)))

print(f"\n{'='*70}")
print(f"PYTHON ({jax.devices()[0]}) — {len(ll_py)} iters, {elapsed:.1f}s")
print(f"{'='*70}")
for i in range(len(ll_py)):
    d = ll_py[i] - ll_py[i-1] if i > 0 else 0
    flag = " <<<" if d < -0.01 and i > 0 else ""
    print(f"  {i:3d}: LL={ll_py[i]:12.6f}  dll={d:+.6e}{flag}")
print(f"  rho_floor: {n_floor_py}/{rho_py.size}")

# Load Fortran 20-iter log if available
if FORTRAN_20_LOG.exists():
    print(f"\n{'='*70}")
    print(f"FORTRAN (from {FORTRAN_20_LOG})")
    print(f"{'='*70}")
    with open(FORTRAN_20_LOG) as f:
        for line in f:
            if line.strip().startswith('iter'):
                print(f"  {line.strip()}")
else:
    print(f"\n(Fortran 20-iter log not found at {FORTRAN_20_LOG})")
    print(f"Run Fortran with max_iter=20 and save output to {FORTRAN_20_LOG}")

# Save Python result
out = Path("validation/results/post_f1_audit/python_20iter_result.pkl")
with open(out, "wb") as f:
    pickle.dump(res, f)
print(f"\nPython result saved to {out}")
