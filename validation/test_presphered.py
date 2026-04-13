"""Definitive test: feed Python the EXACT whitened data Fortran uses.

1. Load Fortran's S and mean
2. Compute x_sphered = S[:60,:] @ (data - mean)  [exactly what Fortran does]
3. Compute log_det_sphere from Fortran's eigenvalues
4. Feed x_sphered to Amica.fit() with do_sphere=False, do_mean=False
5. Compare LL trajectory to Fortran's

If this matches Fortran, the AMICA algorithm is correct and the ONLY
issue is in Python's sphering. If it still crashes, there's a real
bug in the AMICA step.
"""
import numpy as np
import time
from pathlib import Path

import jax
print(f"JAX devices: {jax.devices()}")

from amica_python import Amica, AmicaConfig

FDIR = Path("validation/results/post_f1_audit/fortran_output")

# Load Fortran's sphering matrix, mean, and eigenvalues
S_full = np.fromfile(FDIR / "S", dtype=np.float64).reshape((118, 118), order='F')
mean_f = np.fromfile(FDIR / "mean", dtype=np.float64)

# Load raw data
z = np.load("validation/results/post_f1_audit/sub01_preproc.npz", allow_pickle=True)
data = z['data']  # (118, 1206350)
n_comp = 60

# Compute the EXACT whitened data Fortran uses
S60 = S_full[:n_comp, :]  # (60, 118)
x_sphered = S60 @ (data - mean_f[:, None])  # (60, 1206350)
print(f"x_sphered: shape={x_sphered.shape}, std={x_sphered.std():.4f}, "
      f"range=[{x_sphered.min():.2f}, {x_sphered.max():.2f}]")

# Compute log_det_sphere the Fortran way: from eigenvalues of covariance
# Fortran: sldet = -0.5 * sum(log(eigv(1:numeigs)))
from scipy.linalg import eigh
cov = (data - mean_f[:, None]) @ (data - mean_f[:, None]).T / data.shape[1]
cov = (cov + cov.T) / 2
eigvals = eigh(cov, eigvals_only=True)[::-1]  # descending
log_det_sphere = float(-0.5 * np.sum(np.log(np.maximum(eigvals[:n_comp], 1e-12))))
print(f"log_det_sphere = {log_det_sphere:.6f}")

# Run Python with do_sphere=False, do_mean=False (data already whitened)
cfg = AmicaConfig(
    num_models=1, num_mix_comps=3, max_iter=20, pcakeep=n_comp,
    dtype="float64", do_sphere=False, do_mean=False,
)
t0 = time.time()
m = Amica(cfg, random_state=42)
# Override log_det_sphere after fit starts (it'll be 0 by default)
# Actually, we need to set it BEFORE fit. Let me check if there's a way...
# The fit() method computes log_det_sphere from eigenvalues.
# With do_sphere=False, eigenvalues are all 1.0, so log_det_sphere = 0.
# We need to inject the correct value.
# HACK: set it on the object after preprocessing but... fit() is one call.
# Better approach: just modify the config or use a wrapper.
# Actually, the cleanest way: just set self.log_det_sphere after fit and
# accept that the LL values will be off by a constant. The TRAJECTORY
# (monotonic vs crashing) is what matters, not the absolute LL value.
res = m.fit(x_sphered)
elapsed = time.time() - t0

ll = np.asarray(res.log_likelihood)
rho = np.asarray(res.rho_)
n_floor = int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))

# The LL will be missing log_det_sphere (= 0 instead of 339.6).
# Correct it: add log_det_sphere / n_comp to each LL value.
ll_corrected = ll + log_det_sphere / n_comp
print(f"\nlog_det_sphere / n_comp = {log_det_sphere / n_comp:.4f}")

print(f"\n{'='*70}")
print(f"PYTHON on FORTRAN-SPHERED DATA ({jax.devices()[0]}) — {len(ll)} iters, {elapsed:.1f}s")
print(f"{'='*70}")
for i in range(len(ll)):
    d = ll_corrected[i] - ll_corrected[i-1] if i > 0 else 0
    flag = " <<<" if d < -0.01 and i > 0 else ""
    print(f"  {i:3d}: LL={ll_corrected[i]:12.6f}  (raw={ll[i]:12.6f})  dll={d:+.6e}{flag}")
print(f"  rho_floor: {n_floor}/{rho.size}")

print(f"\n{'='*70}")
print("FORTRAN REFERENCE")
print(f"{'='*70}")
fortran_log = Path("validation/results/post_f1_audit/fortran_20iter.log")
if fortran_log.exists():
    with open(fortran_log) as f:
        for line in f:
            if line.strip().startswith('iter'):
                print(f"  {line.strip()}")
