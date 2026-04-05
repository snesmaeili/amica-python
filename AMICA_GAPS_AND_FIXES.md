# AMICA Implementation: Gaps, Missing Features, and Fix Plan

> From the design report audit — everything that is incomplete, underspecified,
> or missing for a bug-resistant, feature-complete AMICA that mirrors Palmer's
> Fortran implementation and integrates cleanly with MNE-Python.

---

## 1. Algorithmic Features — Incomplete or Undocumented

### 1.1 Shared Components Across Models

**Status:** NOT IMPLEMENTED

Palmer's "AMICA with Shared Components" allows a global component index `k`
that appears in multiple models with tied source density parameters but
model-specific activations.

**Missing pieces:**
- Mapping from global component index `k` to per-model row index `i(h,k)`
- Shared updates of `α_kj, μ_kj, β_kj, ρ_kj` pooled over all models containing component `k`
- Config surface for "single-model", "multi-model independent", and "multi-model with shared components"

**Decision needed:** Implement shared components, or explicitly document that we implement the *non-shared* variant only (which is the common use case for EEG).

**Action:**
- [ ] Add `shared_components: bool = False` to AmicaConfig
- [ ] If False (default), document clearly that this is the non-shared variant
- [ ] If True, implement the shared parameter pooling logic

### 1.2 Updates for Model Centers c_h

**Status:** PARTIALLY IMPLEMENTED — `c_h` exists in config but update rule is not documented

The generative model uses `y^(h) = W_h (x - c_h)`, but the update for `c_h`
is not specified. Palmer uses a reparameterization between `c_h` and the
mixture means `μ_hij` to keep sources zero-mean given each model.

**Without an explicit update:**
- `c_h` either stays fixed (suboptimal likelihood)
- Or drifts numerically, breaking the intended invariance

**Action:**
- [ ] Check current code: does `solver.py` update `c_` ?
- [ ] If not: add update rule `c_h = weighted mean of x under model h`
- [ ] Adjust μ accordingly at each iteration
- [ ] Add `update_c: bool = True` to AmicaConfig

### 1.3 Explicit E-Step for Source Mixture Responsibilities

**Status:** IMPLIED but not explicitly documented

The M-step formulas use `u_m(t)` but never write the E-step:

```
u_m(t) = P(z_t = m | y_t) = α_m p_m(y_t) / Σ_{m'} α_{m'} p_{m'}(y_t)
```

**Must be:**
- Computed in log-domain with logsumexp for numerical stability
- Defined per source, per model: `u[h, i, m, t]` (not just global `u_m(t)`)
- Multi-model case: careful indexing to avoid silent bugs

**Action:**
- [ ] Verify `pdf.py` computes responsibilities in log-domain
- [ ] Verify indexing is `[h, i, m, t]` in multi-model case
- [ ] Add explicit comments/docstrings with the formula

### 1.4 Row-Norm / Scale Reparameterization of W_h and β

**Status:** PARTIALLY IMPLEMENTED (`doscaling` flag exists in config)

Palmer normalizes each row of W_h to fixed norm and rescales β to compensate.
Without this, EM and Newton steps can blow up column norms of W while
shrinking β (or vice versa), causing overflow in generalized Gaussian PDFs.

**Action:**
- [ ] Verify `doscaling` logic in `solver.py` matches Palmer's invariance:
  1. Compute column norms of A_h (= row norms of W_h)
  2. Normalize columns to unit norm
  3. Scale μ by the norm, scale β by inverse norm
- [ ] Ensure this runs EVERY iteration, not just optionally
- [ ] Add tests that verify norms stay bounded over 2000 iterations

---

## 2. Numerical Stability and Edge-Case Handling

### 2.1 Log-Domain Responsibilities and Likelihood

**Status:** NEEDS VERIFICATION

All probability computations must use log-domain:
- Model responsibilities `v_h(t)` via logsumexp
- Mixture responsibilities `u_m(t)` via logsumexp
- Data log-likelihood with logsumexp over models and mixtures

**Action:**
- [ ] Audit `likelihood.py` — verify logsumexp is used everywhere
- [ ] Audit `pdf.py` — verify no raw exp() without log protection
- [ ] Test with extreme values: very large β, ρ near boundaries
- [ ] Test with 128-channel EEG data (high dimensional)

### 2.2 Newton Statistics — Precise Definitions

**Status:** IMPLEMENTED in `updates.py` but not documented in report

The Newton correction uses σ²_ij, κ_ij, λ_ij estimated from data.
Need clear sample estimates and regularization for near-singular denominators.

**Action:**
- [ ] Verify `updates.py:compute_newton_terms()` matches Palmer's formulas exactly
- [ ] Add regularization: when `σ²_ii σ²_jj κ_ij κ_ji - 1 ≈ 0`, add epsilon
- [ ] Test Newton on ill-conditioned data (nearly collinear channels)

### 2.3 Convergence Criteria

**Status:** BASIC — only ΔL and lrate checks

Palmer also uses gradient-norm criterion. Current implementation has
`use_grad_norm` and `min_grad_norm` in config but unclear if active.

**Action:**
- [ ] Verify gradient norm is computed and checked each iteration
- [ ] Add per-model convergence in multi-model case
- [ ] Add `max_decs` (max LL decreases before stopping) — already in config?

### 2.4 NaNs, Zero-Variance, and Rank-Deficient Data

**Status:** PARTIALLY HANDLED (mineig threshold exists)

**Action:**
- [ ] Add explicit checks in `preprocessing.py`:
  - Zero-variance channels → raise or auto-drop
  - Rank-deficient covariance → PCA to rank `pcakeep`
- [ ] Add sanity checks per iteration:
  - α below floor → clamp to min_alpha (e.g., 1e-6)
  - β outside [invsigmin, invsigmax] → clamp
  - ρ at boundaries for too many iterations → log warning
  - NaN in W or A → revert to previous iteration, reduce lrate
- [ ] Test with interpolated EEG data (rank < n_channels)

---

## 3. Sample Rejection and Multi-Model Behavior

### 3.1 Sample Rejection Algorithm

**Status:** CONFIG EXISTS but implementation may be incomplete

Palmer's rejection:
1. Compute per-sample LL (normalized by n_channels)
2. Compute mean and std of LL across samples
3. Reject samples with LL < mean - rejsig * std
4. Re-run iteration without rejected samples
5. Start at iteration `rejstart`, recompute every `rejint` iterations
6. Cap maximum rejected fraction to prevent catastrophic loss

**Action:**
- [ ] Verify rejection is implemented in `solver.py` fit loop
- [ ] Add `max_reject_fraction: float = 0.2` to prevent over-rejection
- [ ] Verify: is rejection global (all models) or per-model?
- [ ] Match Klug et al. 2024 recommendations: 5-10 iterations, rejsig=3

### 3.2 Multi-Model Data Requirements

**Status:** NO WARNINGS

**Action:**
- [ ] Add heuristic warning when `n_frames / (n_channels * num_models) < 30`
- [ ] Document minimum data requirements per model in docstring

---

## 4. MNE Integration and API Gaps

### 4.1 Shape Conventions

**Status:** FUNCTIONAL API EXISTS but needs verification

MNE expects:
- `unmixing_matrix_`: shape `(n_components, n_components)` in whitened space
- `mixing_matrix_`: `pinv(unmixing_matrix_)`
- Data passed as `(n_samples, n_components)` — TRANSPOSED

**Action:**
- [ ] Write integration test: `ICA(method='amica').fit(raw)` end-to-end
- [ ] Verify W orientation matches MNE convention
- [ ] Verify no double-whitening (MNE whitens, we pass `whiten=False`)
- [ ] Verify `|det S|` in LL formula uses internal S, not MNE's whitener

### 4.2 Reproducibility and Random State

**Status:** BASIC (`random_state` parameter exists)

**Action:**
- [ ] Verify JAX PRNG keys are handled correctly (no reuse across JIT calls)
- [ ] Verify seeding produces identical results across runs
- [ ] Document seeding scheme for multi-model initialization
- [ ] Test: same seed → same decomposition (bit-exact on CPU)

---

## 5. Validation Coverage

### 5.1 MATLAB Parity — Expand Scope

**Current:** Framework exists but scope unclear.

**Action:**
- [ ] Test matrix: `num_models ∈ {1, 2, 3}` × `num_mix ∈ {1, 3, 5}` × `do_newton ∈ {F, T}` × `do_reject ∈ {F, T}`
- [ ] Datasets: synthetic, MNE sample, ds004505 (table tennis)
- [ ] Metrics: LL curves, MIR, dipolarity, RELICA stability
- [ ] Pass criterion: LL within 1% of Fortran, mixing matrix correlation > 0.95

### 5.2 Stress Tests

**Action:**
- [ ] Very short data: 4 channels × 100 samples
- [ ] Very long data: 64 channels × 1M samples
- [ ] Rank-deficient: 32 channels, rank 20 (after interpolation)
- [ ] Heavy artifacts: ensure rejection + Newton behave
- [ ] Non-monotone LL with Newton → verify damping/fallback
- [ ] Collapsing mixture component → verify re-initialization or pruning

---

## 6. Summary: Concrete Next Steps (Priority Order)

### P0 — Must Fix Before Any Validation
1. Verify/fix `c_h` update rule
2. Verify/fix row-norm + β rescaling (`doscaling`)
3. Verify log-domain computation in all probability functions
4. Add NaN/rank-deficiency guards

### P1 — Must Fix Before MNE Integration
5. Verify W orientation matches MNE convention (integration test)
6. Verify no double-whitening
7. Add gradient-norm convergence criterion
8. Implement sample rejection (if not complete)

### P2 — Should Fix Before Release
9. Add shared components option (or document non-shared)
10. Expand MATLAB parity test matrix
11. Add stress tests and edge-case tests
12. Document all formulas with explicit indexing

### P3 — Nice to Have
13. Multi-model data requirement warnings
14. Component collapse detection and re-initialization
15. Online/streaming mode
16. Complex-valued ICA for MEG

---

## Files to Audit

| File | LOC | What to Check |
|------|-----|---------------|
| `solver.py` | 822 | c_h update, doscaling, rejection loop, convergence |
| `updates.py` | 743 | Newton terms, β/μ rescaling, parameter clamps |
| `pdf.py` | 345 | Log-domain computation, logsumexp, score function |
| `likelihood.py` | 269 | Log-domain LL, det terms, multi-model handling |
| `preprocessing.py` | 333 | Rank detection, zero-variance, mineig |
| `config.py` | 204 | All flags documented, defaults match Fortran |
| `backend.py` | 97 | JAX PRNG handling, x64 precision |
| `__init__.py` | 75 | amica() API shape conventions |
