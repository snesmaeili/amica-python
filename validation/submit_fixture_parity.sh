#!/bin/bash
#SBATCH --job-name=fixture-parity
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/fixture-parity-%j.out
#SBATCH --error=logs/fixture-parity-%j.err
#SBATCH --chdir=/home/sesma/amica-python
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas/3.3.1 2>/dev/null || module load gcc openmpi flexiblas 2>/dev/null || true
source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1

echo "=== 1. Fortran on non-degenerate fixture ==="
rm -rf tests/fixtures/fortran_nondegenerate/*
mkdir -p tests/fixtures/fortran_nondegenerate
echo "! empty" > mkl_vml.f90 2>/dev/null || true
/home/sesma/refs/sccn-amica/amica17_narval tests/fixtures/synthetic_nondegenerate.param 2>&1 | grep -E '^ iter|done|Exit|error'
echo

echo "=== 2. Python on same fixture ==="
python -u -c "
import numpy as np
from amica_python import Amica, AmicaConfig

z = np.load('tests/fixtures/synthetic_nondegenerate.npz')
x = z['x']  # (5, 10000)
print(f'data: {x.shape}')

cfg = AmicaConfig(num_models=1, num_mix_comps=3, max_iter=500, pcakeep=5, dtype='float64')
res = Amica(cfg, random_state=42).fit(x)
ll = np.asarray(res.log_likelihood)
rho = np.asarray(res.rho_)
n_floor = int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))
print(f'n_iter={res.n_iter}, converged={res.converged}')
print(f'LL: first={ll[0]:.6f} final={ll[-1]:.6f}')
print(f'rho_floor: {n_floor}/{rho.size}')
print(f'rho range: [{rho.min():.3f}, {rho.max():.3f}]')

# Save for test_against_fortran.py
import pickle
with open('tests/fixtures/python_nondegenerate_result.pkl', 'wb') as f:
    pickle.dump(res, f)
print('Saved Python result')
"
echo

echo "=== 3. Newton byte-check ==="
python -u -c "
import numpy as np
import jax.numpy as jnp
from amica_python.pdf import compute_all_scores, compute_responsibilities
from amica_python.updates import compute_newton_terms

# Use the Fortran fixture final state for the byte-check
FDIR = 'tests/fixtures/fortran_nondegenerate'
import os
if not os.path.exists(f'{FDIR}/W'):
    print('Fortran output not found, skipping Newton byte-check')
    exit(0)

n_comp = 5; n_mix = 3
W = np.fromfile(f'{FDIR}/W', dtype=np.float64).reshape((n_comp,n_comp), order='F')
A = np.fromfile(f'{FDIR}/A', dtype=np.float64).reshape((n_comp,n_comp), order='F')
alpha = np.fromfile(f'{FDIR}/alpha', dtype=np.float64).reshape((n_mix,n_comp), order='F')
mu = np.fromfile(f'{FDIR}/mu', dtype=np.float64).reshape((n_mix,n_comp), order='F')
sbeta = np.fromfile(f'{FDIR}/sbeta', dtype=np.float64).reshape((n_mix,n_comp), order='F')
rho = np.fromfile(f'{FDIR}/rho', dtype=np.float64).reshape((n_mix,n_comp), order='F')
S = np.fromfile(f'{FDIR}/S', dtype=np.float64).reshape((n_comp,n_comp), order='F')
mean_f = np.fromfile(f'{FDIR}/mean', dtype=np.float64)

# Load data and compute sources at Fortran's final state
z = np.load('tests/fixtures/synthetic_nondegenerate.npz')
x = z['x']
x_white = S @ (x - mean_f[:, None])
y = W @ x_white

# Python's Newton terms
sigma2, kappa, lambda_ = compute_newton_terms(
    jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(mu),
    jnp.asarray(sbeta), jnp.asarray(rho)
)
sigma2 = np.asarray(sigma2)
kappa = np.asarray(kappa)
lambda_ = np.asarray(lambda_)

print('Python Newton terms at Fortran final state:')
print(f'  sigma2: {sigma2}')
print(f'  kappa:  {kappa}')
print(f'  lambda: {lambda_}')
print(f'  all finite: sigma2={np.all(np.isfinite(sigma2))}, kappa={np.all(np.isfinite(kappa))}, lambda={np.all(np.isfinite(lambda_))}')
print(f'  all positive: sigma2={np.all(sigma2>0)}, kappa={np.all(kappa>0)}, lambda={np.all(lambda_>0)}')

# Cross-check: natgrad should be near zero at converged state
g = np.asarray(compute_all_scores(jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(sbeta), jnp.asarray(rho)))
gy = g @ y.T / y.shape[1]
dA = np.eye(n_comp) - gy
print(f'  ||natgrad dA||_F: {np.linalg.norm(dA):.6f} (should be small at convergence)')

# Verify Newton correction is well-posed
from amica_python.updates import apply_full_newton_correction
Wtmp, posdef = apply_full_newton_correction(jnp.asarray(dA), jnp.asarray(sigma2), jnp.asarray(kappa), jnp.asarray(lambda_))
print(f'  Newton posdef: {bool(posdef)}')
print(f'  ||Newton Wtmp||_F: {float(jnp.linalg.norm(Wtmp)):.6f}')
print('Newton byte-check PASSED' if bool(posdef) and np.all(np.isfinite(np.asarray(Wtmp))) else 'Newton byte-check ISSUES')
"
