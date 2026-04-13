# Fortran Validation Guide for amica-python

How to validate Python AMICA against the original Fortran amica17
after any code change. Written from hard-won experience during the
2026-04-08 → 2026-04-13 source-level audit.

---

## 1. Reference artifacts on disk

| What | Path | Notes |
|---|---|---|
| Fortran source (amica17) | `/home/sesma/refs/sccn-amica/amica17.f90` | 3903 lines |
| Fortran header | `/home/sesma/refs/sccn-amica/amica17_header.f90` | type defs, defaults |
| Math functions (gamln, psifun) | `/home/sesma/refs/sccn-amica/funmod2.f90` | 567 lines |
| Compiled binary (Narval) | `/home/sesma/refs/sccn-amica/amica17_narval` | gfortran + MPI + flexiblas, needs `module load gcc openmpi flexiblas` |
| MKL stub replacements | `/home/sesma/refs/sccn-amica/mkl_stubs.f90` | vdLn/vdExp wrappers |
| Pre-compiled amica15 binary | `/home/sesma/refs/sccn-amica/amica15ub` | statically linked, works on login but has buffer-size limits (~30 ch max) |
| Palmer 2011 tech report | `/home/sesma/refs/papers/palmer_2011_amica.pdf` | canonical equations |
| Palmer 2008 ICASSP | `/home/sesma/refs/papers/palmer_2008_icassp_newton.pdf` | Newton update derivation |
| scott-huberty reference impl | `/home/sesma/refs/huberty-amica/` | third oracle |
| BeMoBIL pipeline | `/home/sesma/refs/bemobil/` | canonical AMICA wrapper |
| Synthetic fixture (3ch) | `tests/fixtures/` | tiny test, both impls run in seconds |
| Sub-01 preprocessed | `validation/results/post_f1_audit/sub01_preproc.npz` | 118ch × 1.2M samples |

## 2. Critical gotchas discovered during the audit

### 2.1 Data layout for Fortran `.fdt` files

**Fortran reads `(data_dim, field_dim)` in column-major order.**
Columns are contiguous in memory = all channels for one sample.

```python
# CORRECT way to write .fdt for Fortran:
data.T.astype(np.float32).tofile('output.fdt')

# WRONG (what we did initially — gave transposed data to Fortran):
data.astype(np.float32).tofile('output.fdt')
```

**Verification:** first `data_dim` values in the file should equal
`data[:, 0]` (all channels, first sample):

```python
check = np.fromfile('output.fdt', dtype=np.float32, count=data_dim)
assert np.allclose(check, data[:, 0].astype(np.float32))
```

This bug caused us to spend hours comparing Python against a Fortran
run on scrambled data that accidentally appeared to work perfectly.

### 2.2 Reading Fortran output files

Fortran writes arrays in **column-major** order. Always use `order='F'`:

```python
W = np.fromfile('W', dtype=np.float64).reshape((nw, nw), order='F')
A = np.fromfile('A', dtype=np.float64).reshape((nw, num_comps), order='F')
S = np.fromfile('S', dtype=np.float64).reshape((nx, nx), order='F')
alpha = np.fromfile('alpha', dtype=np.float64).reshape((num_mix, num_comps), order='F')
```

### 2.3 `amica17.f90` non-MKL path has a known bug

Line 1465 (non-MKL `#else` block): uses `(rho - 0.0)` instead of
`(rho - 1.0)` for the score function exponent. **Always compile with
`-DMKL`** and use the MKL stubs (`mkl_stubs.f90`) to get the correct
`(rho - 1.0)` code path.

### 2.4 `amica17.f90` has an unguarded OMP reduction bug

Lines 1639-1658: thread-local arrays (`dsigma2_*_t`, `dkappa_*_t`,
`drho_*_t`) are reduced unconditionally, but are only allocated when
`do_newton=true` / `dorho=true`. The patched version
(`amica17_patched.f90`) adds `if` guards. Use the patched version.

### 2.5 `pinv` vs `inv` for W = A^{-1}

Fortran uses LU decomposition (`DGETRF + DGETRI`). Python uses
`jnp.linalg.pinv` (SVD pseudoinverse). On well-conditioned A these
agree; on ill-conditioned A they diverge.

**Do NOT change to `jnp.linalg.inv`.** We tested this — `inv` causes
catastrophic numerical explosion on 60-component real EEG because it
propagates singular-value noise without regularization. `pinv` provides
necessary damping. The small numerical difference vs Fortran's LU is
acceptable and does not affect the algorithm's trajectory.

### 2.6 lrate=0.1 is too aggressive for some real EEG

Both Fortran and Python crash at `lrate=0.1` on 60-component sub-01
data. This is not a bug in either implementation — the Fortran default
was tuned for specific pipelines. Use `lrate=0.01` for high-dimensional
real EEG, or implement adaptive lrate reduction.

### 2.7 The `mu` compensation on `c` update

Fortran does NOT adjust `mu` when `c` changes (amica17.f90:1900).
Python's earlier code had `mu -= W @ delta_c` which is a valid
reparameterization but changes the transient trajectory. **Removed to
match Fortran** — the next iteration's gradient adapts naturally.

### 2.8 `rholrate` per-iteration reset

amica15 resets `rholrate = rholrate0` every iteration. amica17 does
NOT (line 1908 is commented out). **We follow amica17** — rholrate
decays and is not reset.

### 2.9 Responsibility floor

Fortran adds `+1e-15` to each mixture responsibility then renormalizes
(amica17.f90:1353-1358). Python now does the same (pdf.py). This
prevents exactly-zero responsibilities from causing 0/0 downstream.

## 3. How to run the parity test

### 3.1 Prepare data

```python
# Load preprocessed data (or preprocess fresh)
z = np.load('validation/results/post_f1_audit/sub01_preproc.npz')
data = z['data']  # (n_channels, n_samples) float64

# Write .fdt for Fortran (CORRECT column-major layout)
data.T.astype(np.float32).tofile('sub01.fdt')
```

### 3.2 Write Fortran param file

Key parameters that must match between Python and Fortran:

```
files sub01.fdt
data_dim 118
field_dim 1206350
pcakeep 60
num_mix_comps 3
num_models 1
lrate 0.01
max_iter 20
dble_data 0
do_approx_sphere 1
do_sphere 1
do_mean 1
do_newton 1
newt_start 50
minrho 1.0
maxrho 2.0
rho0 1.5
rholrate 0.05
invsigmin 0.00000001
max_decs 3
```

### 3.3 Run Fortran

```bash
module load gcc openmpi flexiblas
OMP_NUM_THREADS=4 /home/sesma/refs/sccn-amica/amica17_narval param_file
```

**Always use sbatch on Narval, never on login nodes.**

### 3.4 Run Python

```python
from amica_python import Amica, AmicaConfig
cfg = AmicaConfig(
    num_models=1, num_mix_comps=3, max_iter=20,
    pcakeep=60, dtype="float64", lrate=0.01,
)
res = Amica(cfg, random_state=42).fit(data)
```

### 3.5 Compare

- **Iter 0 LL should match to ~0.01 nats** (different random init
  causes a small gap, but the LL formula is identical)
- **Both should show the same qualitative trajectory** (increasing LL,
  similar oscillation pattern, same magnitude)
- **Neither should crash** — if one does, investigate lrate/data

### 3.6 Compiling amica17 from source

```bash
module load gcc openmpi flexiblas

cd /home/sesma/refs/sccn-amica

# Compile with MKL stubs and bug patches
mpifort -cpp -DMKL -O2 -fopenmp -ffree-line-length-none \
    -fallow-argument-mismatch -I. \
    -c funmod2.f90 -o funmod2.o
mpifort -cpp -DMKL -O2 -fopenmp -ffree-line-length-none \
    -c mkl_stubs.f90 -o mkl_stubs.o
mpifort -cpp -DMKL -O2 -fopenmp -ffree-line-length-none \
    -fallow-argument-mismatch -I. \
    -c amica17_patched.f90 -o amica17_p.o
mpifort -O2 -fopenmp funmod2.o mkl_stubs.o amica17_p.o \
    -lflexiblas -o amica17_narval
```

The patched version fixes the OMP reduction guard bug (§2.4).
The MKL stubs replace Intel's vdLn/vdExp with standard Fortran
log/exp (§2.3).

## 4. What to check after a code change

### 4.1 Quick sanity (< 1 min, login node OK)

```bash
pytest tests/test_against_fortran.py tests/test_amica.py -q
```

Should give: `N passed, 4 skipped` (the 4 skipped are the synthetic-
fixture tests that need a non-degenerate fixture — see §5).

### 4.2 Synthetic fixture (tiny, 3 components)

Run both Python and Fortran on `tests/fixtures/synthetic.fdt` with
`tests/fixtures/synthetic.param` (num_mix=1, 3 channels, 5000 samples).
Both should converge in ~60 iters. Note: the synthetic fixture has a
degenerate orthogonal mixing matrix, so W may converge to different
rotations between Python and Fortran. The LL trajectory shape (not
the absolute W) is the comparison target.

### 4.3 Real EEG parity (sub-01, 60 components)

Run both on `sub01_preproc.npz` (via sbatch, NOT login) with
`lrate=0.01`, `max_iter=20`. Compare iter-by-iter LL.

Use `validation/validate_parity.py` as the template.

**Expected result:** LL agrees to ~0.01 nats at iter 0, diverges
to ~0.5 nats by iter 20 (due to different random init), same
qualitative oscillation pattern, neither crashes.

## 5. Known limitations

- **Synthetic fixture is degenerate**: orthogonal `A_true` + ZCA
  sphering produces a saddle at W=I. Fortran exits early at the
  saddle; Python explores past it. Not a bug — just a bad test
  fixture. Need to build a non-degenerate fixture for proper
  regression testing (TODO).

- **Multi-model AMICA not implemented**: `num_models > 1` is a
  **missing feature**, not a validation gap. Python's `_amica_step`
  only handles a single model — there is no loop over models, no
  model responsibilities `v(h)`, no per-model gradient accumulation,
  no `comp_list` or component sharing. The Fortran multi-model code
  (amica17.f90 lines 1270-1360, 1379-1404, 1845-1856) would need to
  be ported as new code. For most practical use (BeMoBIL, Klug 2024),
  `num_models=1` is the standard setting.

- **Newton terms (kappa/lambda) not byte-checked**: the formulas match
  on paper but haven't been compared element-by-element against
  Fortran on real data.

- **Sphering convention**: Python uses scipy `eigh` + ZCA construction
  via SVD polar factor. Fortran uses LAPACK `DSYEV` + the same ZCA
  construction. The eigenvalues match to machine precision; the
  eigenvector signs may differ but cancel in ZCA. On the synthetic,
  this produces different basins (different W up to rotation); on
  real EEG, the effect is absorbed into the random init noise.

## 6. Algorithm checklist (Python ↔ amica17 line-by-line)

Every formula verified to match:

| Operation | Python | Fortran | Status |
|---|---|---|---|
| GG log-density | pdf.py:58-77 | amica17:1305-1331 | ✅ |
| Score fp | pdf.py:208-211 | amica17:1455-1467 | ✅ |
| Responsibilities | pdf.py:164-172 | amica17:1352-1358 | ✅ (+1e-15 floor) |
| Weighted score g | pdf.py:259 | amica17:1493 | ✅ |
| Natgrad dA | solver.py:103-104 | amica17:1800-1806 | ✅ |
| Newton correction | updates.py:539-551 | amica17:1817-1832 | ✅ |
| A update | solver.py:138-139 | amica17:1909 | ✅ |
| W = pinv(A) | solver.py:153 | amica17:2157 (LU) | ⚠️ pinv vs LU |
| alpha update | updates.py:127 | amica17:1891 | ✅ |
| mu update | updates.py:178-205 | amica17:1527-1544 | ✅ |
| beta update | updates.py:263-289 | amica17:1993-1995 | ✅ |
| rho update | updates.py:353-385 | amica17:2013 | ✅ |
| LL normalization | likelihood.py:104 | amica17:1868 | ✅ |
| log\|det W\| | likelihood.py:26 | amica17:971-978 | ✅ |
| lrate state machine | solver.py:614-641 | amica17:1068+1907 | ✅ |
| A init | solver.py:857-862 | amica17:818-825 | ✅ |
| Sphering (ZCA) | preprocessing.py:148-167 | amica17:486-493 | ✅ |
