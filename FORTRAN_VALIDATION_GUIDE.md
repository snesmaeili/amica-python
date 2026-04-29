# Fortran validation guide for amica-python

How to validate amica-python against the reference Fortran amica17
after any code change. Written 2026-04-13 after a multi-day audit
that uncovered every pitfall the hard way.

---

## 1. Reference artifacts on Narval

| What | Path | Notes |
|---|---|---|
| Fortran source (patched) | `/home/sesma/refs/sccn-amica/amica17_patched.f90` | OMP reduction guard fix applied |
| Compiled binary | `/home/sesma/refs/sccn-amica/amica17_narval` | gfortran + OpenMPI + FlexiBLAS, dynamically linked |
| MKL stubs | `/home/sesma/refs/sccn-amica/mkl_stubs.f90` | Provides `vdLn`/`vdExp` using standard Fortran intrinsics |
| Function module | `/home/sesma/refs/sccn-amica/funmod2.f90` | `gamln`, `psifun`, `matout`, etc. |
| Sub-01 preprocessed | `validation/results/post_f1_audit/sub01_preproc.npz` | 118 ch × 1.2M samples, float64 |
| Sub-01 .fdt (Fortran input) | `validation/results/post_f1_audit/sub01.fdt` | Same data, float32, **column-major** |
| Fortran param file | `validation/results/post_f1_audit/sub01_fortran.param` | All defaults matching Python |
| Non-degenerate synthetic | `tests/fixtures/synthetic_nondegenerate.npz` | 5 sources, non-orthogonal mixing |
| Fortran output on synthetic | `tests/fixtures/fortran_nondegenerate/` | W, A, S, mean, alpha, mu, sbeta, rho, LL |

### Recompiling Fortran

```bash
cd /home/sesma/refs/sccn-amica
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas/3.3.1
# Compile modules
mpifort -cpp -DMKL -O2 -fopenmp -ffree-line-length-none -c funmod2.f90 -o funmod2.o
mpifort -cpp -DMKL -O2 -fopenmp -ffree-line-length-none -c mkl_stubs.f90 -o mkl_stubs.o
# Compile patched amica17 (has OMP reduction guard fix)
mpifort -cpp -DMKL -O2 -fopenmp -ffree-line-length-none -fallow-argument-mismatch -I. \
    -c amica17_patched.f90 -o amica17_p.o
# Link
mpifort -O2 -fopenmp funmod2.o mkl_stubs.o amica17_p.o -lflexiblas -o amica17_narval
```

The `-DMKL` flag is **required** — the non-MKL code path has a bug
at line 1465 where `(rho-0.0)` should be `(rho-1.0)`. The `-DMKL`
path is correct.

Module load may fail on some compute nodes. Use
`module load gcc openmpi flexiblas 2>/dev/null || true` as fallback.

---

## 2. The .fdt data layout trap

**This was the single biggest pitfall in the entire audit.** Fortran
reads `.fdt` files as `(data_dim, field_dim)` in **column-major**
order. This means the first `data_dim` values in the file are ALL
CHANNELS of sample 0, then all channels of sample 1, etc.

Python's `numpy.ndarray.tofile()` writes **row-major** (C order). So
`data.tofile('out.fdt')` writes all SAMPLES of channel 0 first — the
**TRANSPOSED** layout vs what Fortran expects.

### Correct way to write .fdt for Fortran

```python
# data is (n_channels, n_samples), C-contiguous
# Fortran needs (n_channels, n_samples) column-major
data.T.astype(np.float32).tofile('output.fdt')
```

This writes row-by-row of `data.T`, which is `data[:, 0]` (all
channels of sample 0) first — matching Fortran's column-major layout.

### Verifying the layout

```python
check = np.fromfile('output.fdt', dtype=np.float32, count=n_channels)
assert np.allclose(check, data[:, 0].astype(np.float32))
```

If this assertion fails, the data is transposed and **all Fortran
results will be meaningless** — Fortran will happily process the
scrambled data without error, producing results that look superficially
reasonable but are completely wrong.

---

## 3. Running the parity test

### Quick (synthetic, ~2 min)

```bash
cd /home/sesma/amica-python
pytest tests/test_against_fortran.py -v
```

This uses the non-degenerate 5-component fixture and compares Python's
output to pre-captured Fortran output. All 6 tests should pass.

### Full (sub-01, ~20 min on CPU)

```bash
# Fortran (needs module load for dynamic libs)
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas/3.3.1
OMP_NUM_THREADS=4 /home/sesma/refs/sccn-amica/amica17_narval \
    validation/results/post_f1_audit/sub01_fortran.param

# Python (CPU)
JAX_PLATFORMS=cpu python validation/validate_parity.py

# Python (GPU)
module load cuda/12.6
python validation/validate_parity.py
```

### What to compare

| Check | Pass criterion | Why |
|---|---|---|
| Initial LL (iter 0) | `|diff| < 0.01` | Same LL formula, different W init adds ~0.002 |
| LL trajectory shape | Both increasing, same range | Same algorithm on same data |
| No catastrophic crashes | No LL < -10 at any iter | Both stable at the chosen lrate |
| rho not collapsed | `frac_at_floor < 0.5` after 500+ iters | Healthy mixture density fitting |

**LL values will NOT match exactly** after the first few iterations
because Python and Fortran use different random W initialization
(Python: `random_state=42`, Fortran: system-clock seed). This is
expected — ICA trajectories diverge chaotically from different initial
W. The test is that both STAY STABLE and reach similar final LL.

---

## 4. Known numerical differences (acceptable)

These are differences that exist between the implementations but do
NOT affect correctness:

### 4.1 `pinv` vs LU inverse

- **Fortran**: `DGETRF` + `DGETRI` (LU decomposition, exact inverse)
- **Python**: `jnp.linalg.pinv` (SVD pseudoinverse)

On well-conditioned A, both give the same W. On ill-conditioned A,
`pinv` truncates small singular values, providing regularization.
`jnp.linalg.inv` causes catastrophic explosion on real EEG data —
**do NOT switch to inv**.

### 4.2 Accumulation order

Fortran accumulates `g^T @ b` block-by-block across data segments
and MPI processes, then reduces. Python does one matrix multiply
`g @ y.T` over all samples. Mathematically identical, numerically
differs by O(eps × N_samples) ≈ 1e-10 in float64.

### 4.3 Random initialization

Both use `A = I + 0.01*(0.5 - random)` with column normalization.
Different RNG seeds → different initial W → different trajectory
→ different local optimum. NOT a bug.

### 4.4 Responsibility floor

Fortran adds `+1e-15` to each responsibility then re-normalizes
(amica17.f90:1353-1358). Python does the same (pdf.py, after the
audit fix).

---

## 5. What to check after changing Python code

### If you change `solver.py` (the EM loop)

Run the full parity test on sub-01. The most sensitive checks:
- Does lrate ramp correctly? (Print lrate at each iter, compare to
  Fortran's `lrate = ...` output)
- Does LL stay positive and generally increase?
- Does the Newton start at `newt_start=50`?
- Is `numdecs` reset to 0 at Newton start?

### If you change `pdf.py` (density/score/responsibilities)

Run the byte-equivalent rho update test:
```bash
pytest tests/test_against_fortran.py::test_rho_update_step_byte_equivalent_to_fortran -v
```

Then check the score function on controlled inputs:
```python
# rho=1.5: g should be rho * sign(y) * |y|^(rho-1)
# rho=2.0: g should be 2*y
# rho=1.0: g should be sign(y)
```

### If you change `updates.py` (parameter updates)

Run all Fortran tests:
```bash
pytest tests/test_against_fortran.py -v
```

Check that `test_newton_terms_well_posed` and
`test_newton_posdef_at_fortran_state` still pass.

### If you change `preprocessing.py` (sphering)

This is the most dangerous change. The sphering matrix determines the
initial optimization landscape. A different sphering can:
- Change the initial LL by several nats
- Put the optimizer in a different basin
- Make `lrate=0.01` crash where it used to work

After changing sphering:
1. Verify `data_white` has unit variance: `np.std(data_white) ≈ 1.0`
2. Verify covariance is identity: `data_white @ data_white.T / N ≈ I`
3. Run the 20-iter parity test and check initial LL still matches
   Fortran to within 0.01

### If you change `likelihood.py`

Check the LL at iter 0 matches Fortran's to within 0.01 nats on the
sub-01 data. The LL formula is:

```
LL = (log|det W| + log|det S|) / n_comp + mean_t(sum_i(logsumexp)) / n_comp
```

Both Fortran and Python normalize by `N_samples × n_components`.

### If you add a new feature

Check the Fortran validation guide's "parity test" still passes with
the feature enabled AND disabled. Run with `max_iter=20, lrate=0.01`
on sub-01 to verify no regression.

---

## 6. Gotchas and lessons learned

### 6.1 The synthetic fixture was degenerate

The original 3-source synthetic with orthogonal mixing was degenerate:
Fortran converged in 62 iters to W≈I (a saddle point), while Python
found a higher-LL solution. This is NOT a bug — it's a fixture problem.
The non-degenerate 5-component fixture with non-orthogonal mixing and
condition number ~3 is the correct test.

### 6.2 lrate=0.1 is too aggressive for 60-component real EEG

Both Fortran and Python crash at `lrate=0.1` on sub-01 (118ch, 60
components). The default `lrate=0.01` is stable. This is an algorithm
property, not a port bug. The BeMoBIL pipeline uses `lrate=0.1` but
on datasets that happen to be better-conditioned.

### 6.3 The lrate state machine ordering matters

Fortran's order is: **decay → ramp → step** (all in the same
iteration). The ramp and A update are both inside `update_params`
(amica17.f90:1907-1909). The A update uses the RAMPED lrate, not the
pre-ramp value. Placing the ramp after the step (or before the decay)
gives catastrophically different behavior.

### 6.4 The mu compensation was wrong

The original Python code adjusted `mu -= W @ delta_c` when the model
center `c` changed (solver.py, "Palmer tech report reparameterization").
Fortran does NOT do this. The compensation changes the transient
trajectory and can destabilize the optimizer. It was removed.

### 6.5 The rholrate reset was amica15-specific

`amica15.f90` resets `rholrate = rholrate0` every iteration.
`amica17.f90` comments this out (line 1908). amica-python follows
amica17 (no reset).

### 6.6 amica17 has a non-MKL code path bug

`amica17.f90` line 1465 (non-MKL path) uses `(rho-0.0)` instead of
`(rho-1.0)` for the score function exponent. The MKL path (line 1462)
is correct. **Always compile with `-DMKL`** (and provide MKL stubs if
actual MKL is unavailable).

### 6.7 amica17 has an OMP reduction guard bug

The thread-local array reduction section (lines 1639-1658) accesses
`dsigma2_numer_tmp_t`, `dkappa_numer_tmp_t`, `drho_numer_tmp_t` etc.
**unconditionally**, even though they're only allocated when
`do_newton=true` or `dorho=true`. The patched version
(`amica17_patched.f90`) wraps each block in the appropriate `if` guard.

---

## 7. Quick reference: Fortran ↔ Python parameter mapping

| Fortran param file key | Python `AmicaConfig` field | Default |
|---|---|---|
| `lrate` | `lrate` | 0.01 |
| `minlrate` | `minlrate` | 1e-8 |
| `lratefact` | `lratefact` | 0.5 |
| `rholrate` | `rholrate` | 0.05 |
| `rholratefact` | `rholratefact` | 0.5 |
| `minrho` | `minrho` | 1.0 |
| `maxrho` | `maxrho` | 2.0 |
| `rho0` | `rho0` | 1.5 |
| `do_newton` | `do_newton` | True |
| `newt_start` | `newt_start` | 50 |
| `newt_ramp` | `newt_ramp` | 10 |
| `newtrate` | `newtrate` | 1.0 |
| `max_iter` | `max_iter` | 2000 |
| `num_mix_comps` | `num_mix_comps` | 3 |
| `num_models` | `num_models` | 1 |
| `pcakeep` | `pcakeep` | None (use rank) |
| `max_decs` | `max_decs` | 3 |
| `min_dll` | `min_dll` | 1e-9 |
| `max_incs` (internal) | `max_incs` | 10 |
| `invsigmin` | `invsigmin` | 1e-8 |
| `invsigmax` | `invsigmax` | 100.0 |
| `do_approx_sphere` | `do_approx_sphere` | True |
| `do_reject` | `do_reject` | False |
| `block_size` | `block_size` | 128 (unused in Python) |
| `dble_data` | N/A | 0 (float32 in .fdt) |
| `use_grad_norm` | `use_grad_norm` | False (not implemented) |
