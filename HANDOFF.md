# Laptop handoff — picking up amica-python locally

Last sync from Compute Canada Narval: **2026-04-29**.

This doc captures the full project state across our three repos, what the
MNE-Python issue thread is waiting for, and the order of next steps so you
can land on the laptop and keep moving without re-tracing context.

---

## 1. The three repos and how they relate

| Repo | Role | Branch | Latest pushed commit |
|---|---|---|---|
| [amica-python](https://github.com/snesmaeili/amica-python) | **Active algorithm package** (JAX with NumPy fallback). The one we ship, the one MNE will evaluate. | `fortran-audit` | `5c95a37` Add Fortran validation guide |
| [amica-python-benchmark](https://github.com/snesmaeili/amica-python-benchmark) | Benchmarks, parity reports, real-EEG runs, vendored patched Fortran 1.7 | `main` | `cdef3d1` Vendor patched AMICA 1.7 Fortran source with portable build |
| [mne-amica](https://github.com/snesmaeili/mne-amica) | Apr-5 publication snapshot (JOSS-ready). Diverged from amica-python and got debugging fixes for MNE integration on top. **To be archived after consolidation.** | `standalone` | `c4e1797a6` Add expanded validation suite |

Local clones on Narval (for reference / data sync):

- `/home/sesma/amica-python`
- `/home/sesma/amica-python-benchmark` (via remote, dir is `amica-benchmark/`)
- `/home/sesma/mne-amica`
- `/home/sesma/refs/sccn-amica/` — original Fortran reference (now vendored into amica-benchmark/fortran/)

---

## 2. Where validation is at

### Done
- **MATLAB parity**: 24 configs, W_corr ≥ 0.99999. Captured in `mne-amica/validation/results/parity_expanded.json`.
- **Synthetic Amari**: AMICA 0.008, FastICA 0.010, Infomax 0.195, Picard 0.223.
- **MNE integration contract**: 8/8 checks pass (`mne-amica/validation/results/mne_integration_checks.json`).
- **Fortran multi-day audit (Apr 13)**: 7 numerical bugs found and fixed. Documented in `amica-python/FORTRAN_VALIDATION_GUIDE.md`. Patched Fortran source vendored at `amica-python-benchmark/fortran/`.
- **3-way parity (Apr 17)**: amica-python vs pyamica vs Fortran. Initial LL within 6e-5 nats of Fortran (pyamica 1.6e-3). Algorithm correct.

### Open
- **AMICA brain-IC under-count on real EEG**: `mne-amica/validation/results/highdens_validation.json` shows AMICA finding 1 brain IC vs Picard's 7 on ds004505. Same in `dipole_validation.json` (AMICA dipole_rv=0.83, others ~0.59). Likely fixed once the mne-amica MNE-integration bug fixes (per-PCA-component variance normalization + `c`-update gating) are ported into amica-python.
- **`to_mne()` standalone path**: `test_standalone_amica.py` failed Apr 5 with "ICA instance was not fitted" — `to_mne()` doesn't populate `pca_components_`/`pca_mean_`/`pre_whitener_` correctly.
- **Cluster timeout**: the multi-method ds004505 run hit the 4 h SLURM limit. Need to split or extend.

---

## 3. What we are competing for (MNE issue [#13819](https://github.com/mne-tools/mne-python/issues/13819))

@drammock (MNE maintainer) said the selection criteria across the multiple Python AMICA implementations are:

> **benchmark numbers, API, license, long-term support, install footprint**

We win on:
- **API**: `fit_ica(raw, ...)` returns a standard MNE ICA object (Picard-shaped).
- **License**: BSD-3 (compatible with MNE).
- **Install footprint**: JAX is **optional** with NumPy fallback (`AMICA_NO_JAX=1`). Competitors require PyTorch (~700 MB).
- **Numerical correctness**: Fortran audit + 3-way parity report — receipts no competitor has.

We still need to deliver:
- **Multi-subject ds004505 ICLabel benchmark** (we committed to this on the thread).
- **CPU perf vs scott-huberty/amica-python and pyamica** (both PyTorch).
- **One canonical repo** instead of mne-amica + amica-python.

Competitors:
- [scott-huberty/amica-python](https://github.com/scott-huberty/amica-python) — PyTorch, scikit-learn API, BSD-2, v0.1.1 (Apr 7), no MNE wrapper documented. Maintained by an MNE maintainer (relevant COI).
- [DerAndereJohannes/pyamica](https://github.com/DerAndereJohannes/pyamica) — PyTorch, has MNE wrapper `AmicaICA`, v0.3.0 (Mar 15), claims Fortran-validated.

---

## 4. First steps on the laptop (in order)

### Step 0 — Get the repos and Fortran build
```bash
git clone git@github.com:snesmaeili/amica-python.git
git clone git@github.com:snesmaeili/amica-python-benchmark.git
git clone git@github.com:snesmaeili/mne-amica.git

# Make sure you're on the right branches
cd amica-python         && git checkout fortran-audit && cd ..
cd amica-python-benchmark && git checkout main         && cd ..
cd mne-amica            && git checkout standalone   && cd ..
```

For Fortran on Windows, follow `amica-python-benchmark/fortran/README.md`. Two paths:
- **Easiest**: install Docker Desktop, run `docker build -t amica17 .` inside `fortran/`.
- **Smaller**: install WSL2 + Ubuntu, then `sudo apt-get install gfortran libopenmpi-dev libopenblas-dev && BLAS=openblas ./build.sh`.

### Step 1 — Set up the amica-python dev env
```bash
cd amica-python
python -m venv .venv
source .venv/bin/activate    # PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[all]"
pip install mne-icalabel pytest
```

Verify imports and run the cheap test suite:
```bash
JAX_PLATFORMS=cpu pytest tests/test_against_fortran.py -v   # uses pre-captured Fortran fixtures
JAX_PLATFORMS=cpu pytest tests/test_mne_contract.py -v
```

If `test_against_fortran.py` passes, the algorithm is correct on this machine. No Fortran binary needed for that test — fixtures are checked in.

### Step 2 — Consolidate amica-python ← mne-amica (you decided this)

The Apr-5 mne-amica snapshot has bug fixes that did **not** flow back to amica-python. Port them across, then archive mne-amica.

Files to diff (left = canonical / target, right = has the fixes to lift):
```bash
diff amica-python/amica_python/mne_integration.py  mne-amica/mne_amica/mne_integration.py
diff amica-python/amica_python/solver.py           mne-amica/mne_amica/solver.py
```

Fixes to lift into amica-python:
1. **`mne_integration.py`** — replace any usage of MNE's internal
   `_transform_raw` / `_transform_epochs` with the manual pipeline
   (pre-whiten → center → PCA project → **per-PCA-component variance
   normalization**). After fitting, restore the unmixing-matrix scale
   (`W /= comp_stds`) and apply MNE's `unmixing_matrix_ /= sqrt(pca_explained_variance_)` convention.
2. **`solver.py`** — gate the model-center (`c`) update behind `do_mean`.
   When data is pre-centered (`do_mean=False`), don't move `c` — it
   destabilizes the optimizer.
3. **`solver.py::Amica.to_mne()`** — populate `pca_components_`,
   `pca_mean_`, `pre_whitener_` so the returned object passes
   `_check_n_pca_components`.

Verify after porting:
```bash
pytest tests/test_against_fortran.py tests/test_mne_contract.py -v
```

Then archive mne-amica:
```bash
cd mne-amica
git tag v0.1.0-snapshot
git push --tags
# Replace README.md with a one-paragraph redirect to amica-python.
```

### Step 3 — Fix `to_mne()` regression test

Add to `amica-python/tests/test_mne_contract.py`:
```python
def test_to_mne_returns_fitted_ica(small_raw):
    cfg = AmicaConfig(max_iter=20, num_mix_comps=1)
    model = Amica(cfg, random_state=0).fit(small_raw.get_data(picks="eeg"))
    ica = model.to_mne(small_raw.info)
    sources = ica.get_sources(small_raw)   # this currently raises
    assert sources is not None
```

### Step 4 — Comparison harness vs scott-huberty + pyamica (you chose this priority)

Drop into `amica-python-benchmark/scripts/comparison/three_implementation_perf.py`:

```python
# Pseudocode — full impl when you start the file.
data = mne.datasets.sample.data_path()  # 60-ch EEG, ~1 min
X = preprocess(data, n_components=30, seed=0)

results = {}
for name, runner in [
    ("amica_python_jax",    run_amica_python),       # AMICA_NO_JAX=0
    ("amica_python_numpy",  run_amica_python),       # AMICA_NO_JAX=1
    ("scott_huberty_torch", run_scott_huberty),
    ("pyamica_torch",       run_pyamica),
]:
    rss_peak, t, W, ll = with_rss(runner, X, n_components=30, seed=0)
    results[name] = dict(time=t, rss_peak=rss_peak, ll_final=ll, W=W.tolist())

# Pairwise W correlations + LL gaps + relative install size
report = make_table(results)
json.dump(results, open("results/comparison/three_implementation_perf.json", "w"))
```

Each PyTorch competitor goes in its own venv to avoid the JAX/PyTorch
import-order conflict. Measure peak RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss` (Linux/WSL) or `psutil.Process().memory_info().peak_wset` (Windows).

### Step 5 — Multi-subject ds004505 benchmark for the MNE thread

ds004505 lives at `/scratch/sesma/ds004505` on Narval. Pull a few subjects locally for laptop runs (or keep the full sweep on Narval as a 6 h job). After the Step-2 fixes land, the AMICA brain-IC count should normalize. Measure: ICLabel brain/muscle/eye, dipole RV, runtime, peak RSS. Save as `amica-python-benchmark/results/highdens_multi_subject.json`.

### Step 6 — Update Overleaf

Two projects:
- [69d187ee23884ed437b3b21f](https://www.overleaf.com/project/69d187ee23884ed437b3b21f) — last update Apr 14
- [69ba088f80e4568a93fc12e7](https://www.overleaf.com/project/69ba088f80e4568a93fc12e7) — last update Apr 17

I (Claude) couldn't read either project — they're login-walled. Either share the .tex source by pasting key sections, or use the Overleaf MCP server if it's still configured locally. Sections to add:
- 3-way parity table (data in `amica-python-benchmark/results/parity/three_way_parity.json`)
- Fortran audit cross-reference (`FORTRAN_VALIDATION_GUIDE.md`)
- Three-implementation perf table (after Step 4)
- Multi-subject ds004505 results (after Step 5)

### Step 7 — Post benchmark numbers to MNE issue [#13819](https://github.com/mne-tools/mne-python/issues/13819)

Drop the multi-subject table + the three-implementation perf comparison + a link to the consolidated repo + the Overleaf preprint. The thread is waiting; this closes the loop.

---

## 5. Things to NOT forget on the laptop

- **AMICA 1.7 only.** Never touch amica15 sources even though they're in the same `refs/sccn-amica/` directory. The parity contract is specifically against 1.7.
- **The Narval Fortran binary won't run on the laptop** — it's CVMFS-linked. Use `amica-python-benchmark/fortran/` (Docker or WSL2) on the laptop.
- **`AMICA_NO_JAX=1`** forces the NumPy fallback. Use it when benchmarking the install-footprint advantage; without it, JAX is loaded.
- **No Co-Authored-By Claude lines** in commits (per `mne-amica/CLAUDE.md`).
- The `test_against_fortran.py` test uses **pre-captured Fortran outputs** in `amica-python/tests/fixtures/fortran_nondegenerate/` — you do NOT need a built Fortran binary to run it.

---

## 6. Quick reference — what to run when

| Task | Command |
|---|---|
| Algorithm correctness check | `pytest amica-python/tests/test_against_fortran.py -v` |
| MNE API contract | `pytest amica-python/tests/test_mne_contract.py -v` |
| Build Fortran (Docker, Windows) | `cd amica-python-benchmark/fortran && docker build -t amica17 .` |
| Build Fortran (WSL2 Ubuntu) | `cd amica-python-benchmark/fortran && BLAS=openblas ./build.sh` |
| Run Fortran in Docker | `docker run --rm -v ${PWD}:/work -w /work amica17 amica17 myparam.param` |
| 3-way parity refresh | `python amica-python-benchmark/scripts/parity/run_three_way_parity.py` |

---

## 7. Open questions / decisions for next session

- After the Step-2 consolidation lands, do we cut a 0.2.0 release of amica-python and tag it? (PyPI presence will help the MNE evaluation.)
- For the comparison harness in Step 4: do we pin specific commits of scott-huberty/amica-python and pyamica, or test latest releases? (Pinning is more reproducible.)
- The SCCN staff member's AMICA implementation that scott-huberty mentioned in the MNE thread — do we want to find it and add it to the comparison? (Wider competitor coverage strengthens our case.)
