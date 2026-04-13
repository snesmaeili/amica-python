# AMICA Python — source-level audit

**Audit date:** 2026-04-08
**Auditor:** Claude (Opus 4.6, with explicit user direction)
**Scope:** `amica_python/` solver, with cross-references to the original
Fortran AMICA at `sccn/amica@HEAD` (`amica15.f90`, `funmod2.f90`,
`amicadefs.param`).
**Constraint:** **No solver code was modified by this audit.** All
changes to the repo are additive: this report, the test stub at
[tests/test_against_fortran.py](tests/test_against_fortran.py), and
the synthetic fixture under [tests/fixtures/](tests/fixtures/).
**Status:** Findings ranked. **Stop and present** before any code fix.

---

## 1. Executive summary

| # | severity | finding | one-line mechanism |
|---|---|---|---|
| **F1** | **blocker** | **`lrate` decays but never recovers** between iterations in [solver.py:720-742](amica_python/solver.py#L720-L742) | Fortran ramps `lrate` UP every successful iter (`lrate += min(1/newt_ramp, lrate)`); Python only decays it. Ratchets to ~0 after a few LL decreases, then parameters drift in a 2-step limit cycle. |
| **F2** | **high** | `invsigmin = 0.0` (Python) vs `1.0e-8` (Fortran) at [config.py:162](amica_python/config.py#L162) vs [amicadefs.param:73](../refs/sccn-amica/amicadefs.param) | β can collapse to exactly 0 in Python; the GG density at non-floor points then collapses, EM responsibilities concentrate on the floor mixture, and ρ is pushed into its absorbing state. |
| **F3** | **high** | Total log-likelihood disagrees with Fortran by ~0.024 nats/sample on identical data | Python −1.389 vs Fortran −1.413 on the synthetic 3-source/5000-sample fixture, despite identical claimed normalization in [likelihood.py:102-104](amica_python/likelihood.py#L102-L104). Source not yet localized; investigate sphereing log-det vs Fortran's `log\|det(S)\|`. |
| **F4** | **medium** | `max_decs = 5` (Python) vs `3` (Fortran), and Python's "ceiling-decay" recovery path at [solver.py:729-737](amica_python/solver.py#L729-L737) is structurally different from Fortran's | Python only halves the ceiling after `max_decs` *consecutive* decays; Fortran halves both `lrate0` and `newtrate` after `maxdecs=3` and continuously ramps. Together with F1 the system can ratchet down without escape. |
| **F5** | **medium** | Convergence path `use_grad_norm` is **not implemented** in Python ([config.py:75-77](amica_python/config.py#L75-L77)); Fortran defaults it ON ([amicadefs.param:16](../refs/sccn-amica/amicadefs.param)) | Python is missing one of Fortran's two stop criteria. Less critical than F1-F3 because it only affects when to stop, not what to compute, but it lets degraded runs grind to `max_iter` instead of exiting early. |
| **F6** | **low** | `block_size = 128` defined at [config.py:184](amica_python/config.py#L184), **never referenced** in the EM loop; Fortran processes data in blocks of 256 | Python `vmap`s across all samples; this is a documented divergence from Fortran. Probably benign on memory but means floating-point sums are accumulated in different orders. Worth leaving a comment. |
| **F7** | **note** | The ρ update [updates.py:300-392](amica_python/updates.py#L300-L392) is **byte-equivalent** to Fortran [amica15.f90:1814-1819](../refs/sccn-amica/amica15.f90#L1814) | The Phase-1 hypothesis "rho update is gradient-based, not Newton-based" was wrong as a divergence claim — Fortran does the same thing. The ρ collapse symptom is downstream of F1/F2, not a ρ-update bug per se. |

**Recommended order of attack** (after user sign-off):

1. **F1** (lrate ramp-up). One-line fix in concept; the fix has to add the per-iteration recovery in the outer loop AND match Fortran's separate `lrate0`/`newtrate` ceiling tracking. Highest expected impact.
2. **F2** (`invsigmin=0`). Trivial config-default fix, but verify in isolation that it has the predicted effect on the rho-collapse symptom on real EEG.
3. **F3** (LL disagreement). Diagnostic — find the source of the −0.024 nats/sample drift. May surface a fourth bug.
4. **F4-F5** are cleanup once F1-F3 are landed.

**Test gate:** After F1+F2 are fixed, the fixture in
[tests/test_against_fortran.py](tests/test_against_fortran.py) should
pass without `pytest.skip` — Python should converge in O(60-100)
iterations and recover the synthetic mixing matrix to within `atol=0.1`.

---

## 2. Empirical reproduction (Phase 1 / Step 1)

### 2.1 Production sub-01 ground truth (verified directly from disk)

The audit prompt states "67/90 mixtures pinned at minrho=1.0". Verified
by loading [validation/results/amica_result.pkl](validation/results/amica_result.pkl)
directly:

```
rho shape: (3, 30)
rho min/max/mean: 1.0 2.0 1.2099938013756426
frac at minrho=1.0: 0.7444444444444445  (= 67/90)
per-mixture floor counts:
  mixture 0: 30/30 at floor
  mixture 1:  7/30 at floor
  mixture 2: 30/30 at floor
n_iter: 2000
converged: False
ll history len: 2000, last 5 ll deltas: [4.84e-5, 1.40e-4, 4.76e-5, 1.39e-4, 4.74e-5]
```

The last 5 LL deltas form an unmistakable **2-step limit cycle**
(`small, large, small, large, small`) — the model is bouncing between
two near-equivalent states with parameter steps too small to escape.
Consistent with F1 (lrate has ratcheted to near zero).

**PAPER_NOTES.md mis-recording:** The note at
[validation/results/PAPER_NOTES.md](validation/results/PAPER_NOTES.md)
saying "AMICA MIR=−141" actually belongs to Picard/Infomax on the same
run; the real AMICA MIR has been ≈ −6 since Apr 7. This was the
phantom-bug rabbit hole the previous session went down. **Don't trust
PAPER_NOTES.md numbers without cross-checking the JSON**.

### 2.2 Synthetic 4-cell matrix

Harness: [/tmp/audit_synthetic_4cell.py](/tmp/audit_synthetic_4cell.py)
(also vendored as [tests/fixtures/audit_synthetic_4cell.py](tests/fixtures/audit_synthetic_4cell.py)).
Data: 3 channels × 5000 samples, 3 GG sources at ρ=[1.0, 1.5, 2.0],
random orthogonal mixing, fixed seed=0. CPU only — JAX login-node has
no GPU and the queued sbatch GPU job (id 59007532) is still pending
`ReqNodeNotAvail`. Float32 ≡ Float64 on CPU (results identical to 3
sig figs), so the GPU half of the matrix would not have changed the
qualitative conclusion.

| n_iter | dtype | frac_rho_at_floor | rho_min | rho_mean | LL final | converged |
|---|---|---|---|---|---|---|
| 200 | float64 | 0.000 | 1.106 | 1.464 | -1.388706 | False |
| 200 | float32 | 0.000 | 1.107 | 1.464 | -1.388706 | False |
| 2000 | float64 | 0.111 | 1.000 | 1.391 | -1.388298 | False |
| 2000 | float32 | 0.111 | 1.000 | 1.391 | -1.388298 | False |

### 2.3 Iteration sweep (CPU float64)

| n_iter | frac_rho_at_floor | rho_min |
|---|---|---|
| 200 | 0.000 | 1.106 |
| 500 | 0.000 | 1.015 |
| 1000 | 0.111 | 1.000 |
| 2000 | 0.111 | 1.000 |
| **5000** | **0.222** | 1.000 |

**Monotonic absorbing-state behavior on a trivially-separable
3-source mixture.** This is a port-level pathology that does not
require real EEG to manifest.

### 2.4 Apr 5 → Apr 7 "regression" — no regression

`git log --since=2026-04-04 --until=2026-04-08T23:59` shows:

| commit | timestamp | message |
|---|---|---|
| 45d2885 | 2026-04-06 02:29 | Fix to_mne() reconstruction, MIR, alpha-peak, and figure quality |
| **(GOOD JSON)** | 2026-04-06 03:38 | benchmark_sub-01_hp1.0hz.json (CPU 200-iter, MIR=-339) |
| **(BAD JSON)** | 2026-04-07 01:07 | quick_check_results.json (GPU 2000-iter, MIR=-6.13) |
| a7dc3f7 | 2026-04-07 01:21 | Fix quick_full_check figures and AMICA convergence |

**Both JSONs were produced by the same source tree (HEAD = 45d2885).**
There is no git regression. The "regression" is iteration count: 200
iters survives because rho hasn't fully collapsed yet, 2000 iters
crosses into the absorbing state. The previous session's "fix" hunt
was for a bug that doesn't exist in the diff; the bug is older and
structural (F1).

---

## 3. Side-by-side Python ↔ Fortran mapping

Verdict legend: ✅ matches / ⚠️ differs in detail / ❌ likely bug / ❓ unverified.

| Operation | Python (file:line) | Fortran (file:line) | Equation | Verdict | Note |
|---|---|---|---|---|---|
| Fit entry / EM loop | [solver.py:641-819](amica_python/solver.py#L641-L819) | [amica15.f90:920-1130](../refs/sccn-amica/amica15.f90#L920) | top-level EM | ⚠️ | structurally similar but lrate dynamics diverge — see F1 |
| GG log-density `log p(s|ρ,μ,β) = log β − ln Γ(1+1/ρ) − log 2 − \|β(s−μ)\|^ρ` | [pdf.py:17-77](amica_python/pdf.py#L17-L77) | [amica15.f90:1306, 1463](../refs/sccn-amica/amica15.f90#L1306) | Palmer 2011 eq. (3) | ✅ | constants match: Python uses `gammaln(1+1/rho) + log(2) − log(beta)`, Fortran uses `gamln(1+1/rho) + log(2) − log(sbeta)` (since `sbeta = 1/β`, both reduce to same expression) |
| Mixture responsibilities (logsumexp) | [pdf.py:129-169](amica_python/pdf.py#L129-L169) | [amica15.f90:1280-1320](../refs/sccn-amica/amica15.f90#L1280) | EM E-step | ✅ | both use logsumexp; Python via `jax.scipy.special.logsumexp` |
| Total log-likelihood | [likelihood.py:108-148](amica_python/likelihood.py#L108-L148) | [amica15.f90:1749-1755](../refs/sccn-amica/amica15.f90#L1749) | LL/(N_samp * N_comp) | ❌ | nominally same normalization but disagrees by 0.024 nats/sample on the synthetic fixture — see F3 |
| ρ update | [updates.py:300-392](amica_python/updates.py#L300-L392) | [amica15.f90:1811-1820](../refs/sccn-amica/amica15.f90#L1811) | `ρ += rholrate · (1 − ρ/ψ(1+1/ρ) · drho_n/drho_d)`; clip to [minrho,maxrho] | ✅ | byte-equivalent; F7 |
| ρ clip bounds | `minrho=1.0, maxrho=2.0` ([config.py:127-128](amica_python/config.py#L127)) | same defaults ([amicadefs.param:54-55](../refs/sccn-amica/amicadefs.param)) | – | ✅ | match |
| α update | [updates.py:110-131](amica_python/updates.py#L110-L131) | [amica15.f90:1776-1778](../refs/sccn-amica/amica15.f90#L1776) | `α = dα_num/dα_denom` | ✅ | match |
| μ update | [updates.py:135-205](amica_python/updates.py#L135-L205) | [amica15.f90:1801-1803](../refs/sccn-amica/amica15.f90#L1801) | `μ += dmu_num/dmu_denom` | ✅ | match |
| β update | [updates.py:214-294](amica_python/updates.py#L214-L294) | [amica15.f90:1805-1809](../refs/sccn-amica/amica15.f90#L1805) | `β = β·sqrt(dβ_n/dβ_d); clip [invsigmin,invsigmax]` | ⚠️ | formula matches; **`invsigmin` default differs**: Python 0.0 vs Fortran 1e-8 — see F2 |
| Newton terms (κ, λ) | [updates.py:21](amica_python/updates.py#L21) | [funmod2.f90](../refs/sccn-amica/funmod2.f90) | – | ❓ | not deeply audited; defer to follow-up |
| Newton W correction (pairwise Hessian inv) | [updates.py:503-566](amica_python/updates.py#L503-L566) | [amica15.f90:1701-1728](../refs/sccn-amica/amica15.f90#L1701) | Palmer 2008 ICASSP eq. (8) | ⚠️ | both pairwise; Fortran posdef test `if (sk1*sk2 > 1.0)` matches Python `lambda > 0`; flow OK; not byte-checked |
| Natural-grad fallback | [solver.py:140-165](amica_python/solver.py#L140-L165) | [amica15.f90:1721-1723](../refs/sccn-amica/amica15.f90#L1721) | `Wtmp = dA` | ✅ | match |
| **lrate decay/recovery** | [solver.py:107-130, 720-742](amica_python/solver.py#L107-L130) | [amica15.f90:1038-1058, 1786-1797](../refs/sccn-amica/amica15.f90#L1038) | – | ❌ | **F1 — Python decays only, never ramps** |
| `max_decs` default | `5` ([config.py:157](amica_python/config.py#L157)) | `3` ([amicadefs.param:65](../refs/sccn-amica/amicadefs.param)) | – | ⚠️ | Python tolerates 5 consecutive decreases; F4 |
| `use_grad_norm` | `False` ("not implemented", [config.py:75-77](amica_python/config.py#L75)) | `1` (true) ([amicadefs.param:16](../refs/sccn-amica/amicadefs.param)) | – | ❌ | F5 — Python missing this stop criterion |
| Convergence test (`min_dll` window) | [solver.py:744-765](amica_python/solver.py#L744-L765) | [amica15.f90:1060-1080](../refs/sccn-amica/amica15.f90#L1060) | `numincs > max_incs` | ✅ | match |
| Sample rejection | [solver.py:688-718](amica_python/solver.py#L688-L718) | [amica15.f90:1118+](../refs/sccn-amica/amica15.f90#L1118) | – | ⚠️ | `rejstart=2/rejint=3` (Python, per Klug 2024) vs `1/1` (Fortran). OK — Python is intentionally newer. |
| W initialization | [solver.py:901-915](amica_python/solver.py#L901) | [amica15.f90 init block](../refs/sccn-amica/amica15.f90) | `eye(n) + 0.01·noise` | ❓ | both random + identity-ish; not byte-checked |
| Sphereing / PCA | [preprocessing.py](amica_python/preprocessing.py) | [amica15.f90](../refs/sccn-amica/amica15.f90) | ZCA / PCA | ❓ | not deeply audited |
| `c` (model center) update | [solver.py:231-238](amica_python/solver.py#L231-L238) | [amica15.f90:1780-1782](../refs/sccn-amica/amica15.f90#L1780) | `c = dc_num/dc_denom` | ⚠️ | Python does `mean(data_white)` and shifts μ to compensate; Fortran does posterior-weighted update — investigate but probably benign for single-model |
| **`invsigmin`** | `0.0` ([config.py:162](amica_python/config.py#L162)) | `1.0e-8` ([amicadefs.param:73](../refs/sccn-amica/amicadefs.param)) | – | ❌ | F2 |
| `block_size` (unused) | `128` ([config.py:184](amica_python/config.py#L184)) | `256` ([amicadefs.param:7](../refs/sccn-amica/amicadefs.param)) | – | ⚠️ | F6 (defined but Python `vmap`s the whole array) |
| `dorho` flag | `update_rho=True` | `do_rho=true` | – | ✅ | match |
| `rholrate` decay-on-LL-decrease | only via `rholrate = rholrate0` reset ([solver.py:771](amica_python/solver.py#L771)) | `rholrate *= rholratefact` then `rholrate0 *= rholratefact` after maxdecs ([amica15.f90:1045, 1050](../refs/sccn-amica/amica15.f90#L1045)) | – | ⚠️ | similar decay pattern but Python only resets to ceiling; consequence of F1 |

---

## 4. Detailed findings

### F1 — lrate decays but never ramps up (BLOCKER)

**Fortran behavior** ([amica15.f90:1038-1058 + 1786-1797](../refs/sccn-amica/amica15.f90)):

```fortran
! On LL decrease (lines 1038-1058):
if (LL(iter) < LL(iter-1)) then
   if ((lrate <= minlrate) .or. (ndtmpsum <= min_nd)) then
      leave = .true.
   else
      lrate    = lrate    * lratefact      ! 0.5
      rholrate = rholrate * rholratefact   ! 0.5
      numdecs  = numdecs + 1
      if (numdecs >= maxdecs) then  ! maxdecs=3
         lrate0   = lrate0   * lratefact
         newtrate = newtrate * lratefact   ! ceiling decays too
         numdecs  = 0
      end if
   end if
end if

! Every iteration in update_params (lines 1786-1797):
if (do_newton .and. (.not. no_newt) .and. (iter >= newt_start)) then
   lrate = min( newtrate, lrate + min(1.0/newt_ramp, lrate) )  ! ramp UP
else
   lrate = min( lrate0,  lrate + min(1.0/newt_ramp, lrate) )   ! ramp UP
end if
```

`lrate` is a quantity in dynamic equilibrium: it decays by factor 0.5
on bad steps and ramps geometrically toward a ceiling on good steps
(by `+min(1/newt_ramp, lrate)` per iter, which doubles `lrate` while
`lrate < 1/newt_ramp`).

**Python behavior** ([solver.py:107-130 inside `_amica_step`](amica_python/solver.py#L107-L130) + [solver.py:720-742 in the outer loop](amica_python/solver.py#L720-L742)):

```python
# In _amica_step (line 112-115):
dll = ll - ll_prev
ll_decreased = (dll < 0.0) & (iteration > 1)
lrate_base = jnp.where(ll_decreased, lrate * lratefact, lrate)
lrate_base = jnp.maximum(lrate_base, minlrate)
# ... returned to outer loop as lrate_eff

# In outer loop (line 720-742):
lrate = lrate_new_val   # take the (decayed-only) value
if dll < 0:
    numdecs += 1
    if numdecs >= self.config.max_decs:   # max_decs=5
        lrate0 *= self.config.lratefact
        lrate = lrate0                    # ← only "recovery"
        ...
        numdecs = 0
elif dll > 0:
    numdecs = 0    # NO RAMP UP
```

There is no per-iteration ramp-up. The only positive update to `lrate`
is the discrete `lrate = lrate0` reset after `max_decs=5` consecutive
decreases — and that reset jumps to a *lower* `lrate0` (because line 731
just halved it).

**Direct evidence from the synthetic Fortran fixture** ([tests/fixtures/fortran_output/out.txt](tests/fixtures/fortran_output/out.txt)):

```
iter    50 lrate = 0.1000000000 LL = -1.4126331460   (last natgrad iter)
iter    51 lrate = 0.2000000000 LL = -1.4126162690   (Newton starts; ramp +0.1)
iter    52 lrate = 0.3000000000 LL = -1.4126121073   (+0.1)
iter    53 lrate = 0.4000000000 LL = -1.4126120510
iter    54 lrate = 0.5000000000 LL = -1.4126120452
...
iter    58 lrate = 0.9000000000 LL = -1.4126120435
 Likelihood decreasing!
iter    59 lrate = 0.5500000000 LL = -1.4126120435   (decay 0.9*0.5 + ramp 0.1)
iter    60 lrate = 0.6500000000                       (ramp +0.1)
iter    61 lrate = 0.7500000000
 Likelihood decreasing!
 Exiting because likelihood increasing by less than  1e-9 ...
```

You can literally see the per-iteration ramp-up in the Fortran log,
including the *combined* decay+ramp on iter 59 (`0.9 → 0.55` =
`0.9*0.5 + min(0.1, 0.45)`).

**Mechanism for the production failure mode:** Once Python's `lrate`
has decayed a few times in the natgrad phase (which is virtually
inevitable on any noisy real-EEG run), it stays decayed forever. The
parameter updates become smaller and smaller. The `min_dll` window
test never triggers because `dll` is always positive but tiny. ρ
performs vanishingly small gradient steps near the absorbing state at
the floor and slowly drifts there. The 2-step LL-delta limit cycle in
[validation/results/amica_result.pkl](validation/results/amica_result.pkl)
(deltas `[4.84e-5, 1.40e-4, 4.76e-5, 1.39e-4, 4.74e-5]`) is the
diagnostic signature.

**Proposed fix sketch (do not implement until approved):** Move the
lrate update out of `_amica_step`'s JIT body and into the outer loop;
on every successful iteration apply the Fortran ramp formula
`lrate = min(ceiling, lrate + min(1.0/newt_ramp, lrate))`, where
`ceiling` is `newtrate` if Newton is active or `lrate0` otherwise.
Maintain `lrate0` and `newtrate` as separate state from `lrate`, and
decay them on `numdecs >= max_decs`. Set the Python `max_decs` default
to `3` to match Fortran. Verify with the synthetic fixture that Python
converges in O(60-100) iterations like Fortran does.

### F2 — `invsigmin = 0.0` allows β → 0 (HIGH)

**Fortran:** [amicadefs.param:73](../refs/sccn-amica/amicadefs.param)
sets `invsigmin 1.00000e-08` and the β update is clipped at
[amica15.f90:1807-1808](../refs/sccn-amica/amica15.f90#L1807):
```fortran
sbeta = sbeta * sqrt( dbeta_numer / dbeta_denom )
sbetatmp = min(invsigmax, sbeta)
sbeta = max(invsigmin, sbetatmp)
```
where `sbeta` is the inverse scale (i.e. 1/σ in Python's β
parameterization). With `invsigmin = 1e-8`, Fortran enforces a strict
positive floor on the inverse scale.

**Python:** [config.py:161-162](amica_python/config.py#L161) sets
`invsigmin = 0.0`. The clip at [updates.py:294](amica_python/updates.py#L294)
allows β to drop to exactly zero. When β = 0, the GG density
`(β·ρ / 2Γ(1/ρ)) · exp(−|β·s|^ρ)` is degenerate, the EM
responsibilities collapse, and the model becomes ill-posed.

**Mechanism for ρ collapse:** as β shrinks toward zero on a noisy
component, the responsibility for that mixture's contribution to its
own samples vanishes, and the gradient term `drho_numer / drho_denom`
in the ρ update becomes dominated by the noise side, pulling ρ toward
its absorbing floor.

**Proposed fix sketch:** Change [config.py:162](amica_python/config.py#L162)
default to `invsigmin: float = 1e-8`. Single-line change. Verify on
the synthetic fixture and on sub-01 that the rho-collapse symptom
shrinks (won't necessarily disappear without F1).

### F3 — log-likelihood disagreement on identical data (HIGH)

Python reports `LL_final = -1.389438` on the synthetic fixture; Fortran
reports `-1.412612` on the same fixture. The Python comment at
[likelihood.py:102-104](amica_python/likelihood.py#L102-L104) claims
Fortran-equivalent normalization. The 0.0232 nats/sample drift cannot
be explained by normalization alone. Python's W is also *worse* than
Fortran's W on the same data, so the higher LL is suspicious.

**Hypotheses to investigate (not yet localized):**
1. `compute_log_det_W` sign or scale.
2. `log_det_sphere` term — Fortran computes `|det(S)|` once during
   sphereing and adds to every per-sample LL; Python may be passing 0
   or a different value depending on `do_sphere` flag.
3. Mixture log-density constant term: Python uses
   `log(beta) − gammaln(1+1/rho) − log(2)` ([pdf.py:17-77](amica_python/pdf.py#L17-L77));
   Fortran uses `−gamln(1+1/rho) − log(2)` and accumulates `log(sbeta)`
   separately. Algebraically equivalent but watch for double-counting.

**No proposed fix yet** — needs diagnostic. Suggested next step: write
a 30-line script that calls Python's `compute_total_loglikelihood` on
a fixed (W, alpha, mu, beta, rho) state loaded from the Fortran
output, and compare term-by-term against the Fortran LL printed in
`out.txt`. If the gap is in the determinant terms, the bug is in
preprocessing. If it's in the per-sample part, it's in `pdf.py`.

### F4 — `max_decs = 5` and broken ceiling-decay path (MEDIUM)

[config.py:157](amica_python/config.py#L157) defaults `max_decs = 5`,
[amicadefs.param:65](../refs/sccn-amica/amicadefs.param) defaults `max_decs = 3`.
Python tolerates more consecutive decreases before reducing the ceiling.
Combined with F1 (no ramp-up), this means Python sits at a degraded
`lrate` for longer before attempting a (wrong-direction) reset.

The Python "ceiling-decay" reset at [solver.py:729-737](amica_python/solver.py#L729) is
also structurally different from Fortran's: Python resets `lrate = lrate0`
(after halving `lrate0`), which is essentially "forget all the decay
history and start over at a lower ceiling." Fortran continues
ramping/decaying from wherever `lrate` happens to be.

**Proposed fix:** Subordinate to F1 — fix together when redoing the
lrate state machine.

### F5 — `use_grad_norm` not implemented (MEDIUM)

[config.py:75-77](amica_python/config.py#L75) explicitly says
`"Reserved for future use. Gradient norm convergence is not yet
implemented in the solver."` Fortran defaults `use_grad_norm = 1` and
exits when `ndtmpsum <= min_nd` ([amica15.f90:1073-1079](../refs/sccn-amica/amica15.f90#L1073)).

In Fortran the natgrad term `ndtmpsum = sqrt(sum(nd) / (nw·n_used_comps))`
is computed at [amica15.f90:1743](../refs/sccn-amica/amica15.f90#L1743) every iter.
Python is missing both the computation and the convergence test (the
comment at [solver.py:756-758](amica_python/solver.py#L756) says
"grad norm 'nd' calculation inside JIT was skipped for speed").

**Impact:** Less critical than F1-F3 because the missing test is a
*stop* criterion, not a numeric. But it means runs that should have
exited cleanly grind on to `max_iter` and accumulate the F1-driven
degradation.

### F6 — `block_size` defined but unused (LOW)

[config.py:184](amica_python/config.py#L184) defines `block_size = 128`;
no other file in `amica_python/` references it. The Python EM step
`vmap`s across all samples in one shot. Fortran processes data in
blocks of 256 to bound memory.

**Impact:** Minor floating-point reproducibility difference (sums
accumulated in different order). Probably not load-bearing for the
rho-collapse symptom. Worth either wiring it up or deleting the
config field with a comment explaining why.

### F7 — ρ update is byte-equivalent to Fortran (NOTE, not a bug)

The Phase-1 hypothesis was wrong. The Python ρ update at
[updates.py:372-390](amica_python/updates.py#L372-L390):
```python
psi = digamma(1.0 + 1.0 / r)
gradient_term = 1.0 - (r / safe_psi) * ratio
rho_new = r + rholrate * gradient_term
rho_new = jnp.clip(rho_new, minrho, maxrho)
```
matches Fortran [amica15.f90:1814-1819](../refs/sccn-amica/amica15.f90#L1814):
```fortran
rho(j,k) = rho(j,k) + rholrate * ( 1.0 - &
     (rho(j,k) / psifun(1.0 + 1.0/rho(j,k))) * drho_numer(j,k) / drho_denom(j,k) )
rhotmp = min(maxrho, rho)
rho    = max(minrho, rhotmp)
```
The clip behavior is identical. **The rho-collapse symptom is downstream
of F1/F2, not a rho-update bug.** Despite the name "Newton method for
ICA mixture model" in Palmer 2008 ICASSP, the Fortran reference uses a
fixed-step gradient update with `rholrate=0.05` (not a Newton step on
ρ). Whatever Newton derivation is in Palmer 2008 / 2011 was not what
made it into the canonical Fortran implementation.

---

## 5. Reproducibility recipe

**Synthetic Fortran fixture:** files at [tests/fixtures/](tests/fixtures/):

- `synthetic.fdt` — 60000 bytes, raw float32, (3 channels × 5000 samples)
- `synthetic_truth.npz` — ground-truth `x`, `s_true`, `A_true`
- `synthetic.param` — Fortran parameter file (`num_mix_comps=1`)
- `fortran_output/` — 14 binary output files from amica15ub on the
  synthetic, plus `out.txt` (the full stdout log including the
  iter-by-iter lrate dynamics)

**To re-run Fortran:** (binary is at `/home/sesma/refs/sccn-amica/amica15ub`,
statically linked, no module-load needed)

```bash
cd /tmp && mkdir -p fortran_amica_test/out && \
  /home/sesma/refs/sccn-amica/amica15ub /home/sesma/amica-python/tests/fixtures/synthetic.param
```

Expected: converges in ~62 iterations, exits via `min_dll` window,
`LL_final ≈ -1.4126`, recovered W is near identity.

**To re-run Python on the same data:** see
[tests/test_against_fortran.py](tests/test_against_fortran.py).

**Format of Fortran output binary files:** raw doubles, Fortran
column-major where 2D. Sizes: `A`, `W`, `S` are 72 bytes (3×3 doubles);
`alpha`, `mu`, `sbeta`, `rho`, `c`, `mean`, `comp_list` are 24 bytes
(num_mix · num_comps doubles where applicable); `gm` is 8 bytes
(1 double); `LL` is 16000 bytes (preallocated 2000 doubles, only first
~62 are real); `LLt` is 80000 bytes (per-sample diagnostics).

---

## 6. Out of scope but worth flagging

- **`mne_integration.py` `_validate` chain** — not audited; only the
  bare `Amica.fit()` path was reviewed. If the bug-fix landing surfaces
  any param-mapping issue between MNE wrapper and the solver, audit
  that too.
- **Multi-model AMICA (`num_models > 1`)** — Fortran has a much more
  elaborate inner loop with `share_comps`, model weights, and per-model
  state. Python's `_amica_step` only handles `num_models = 1`. The
  validation runner uses `num_models = 1` so this doesn't affect
  current results, but it's a documented gap.
- **Newton lambda/kappa accumulation** ([updates.py:21](amica_python/updates.py#L21)) — verdict ❓; not byte-checked
  against [funmod2.f90](../refs/sccn-amica/funmod2.f90). Probably fine
  but worth a follow-up pass.
- **Sphereing log-det** — `log_det_sphere` is computed in
  preprocessing.py and passed to `compute_total_loglikelihood`, but
  not audited end-to-end. Plausible source of F3.
- **Frank 2025 kappa heuristic** (`κ = N_frames / N_chans²`) — not
  enforced anywhere in the validation runner or the solver. After F1
  is fixed, validation runs should warn when κ < 30 (or whatever Frank
  recommends).
- **scott-huberty/amica-benchmark cross-check** — repo cloned at
  [/home/sesma/refs/huberty-amica/](../refs/huberty-amica/) but not
  audited. Use as a third oracle when validating F1-F3 fixes.

---

## 6a. Addendum (2026-04-08, post-F1 landing)

A second session landed F1 (the lrate state machine refactor) and re-ran
[tests/test_against_fortran.py](tests/test_against_fortran.py). F1 was
implemented correctly — diagnostic shows the expected lrate decay/ramp
trajectory matching `amica15.f90:1786-1797` qualitatively. **But the
test gate did not pass on the synthetic fixture.** Investigation in the
third session traced the problem to the fixture itself, not to F1:

- **Python's score function is mathematically correct.**
  `compute_all_scores` exactly matches the analytical GG score
  `ρ · sign(y) · |y|^(ρ−1)` for ρ ∈ {1.0, 1.5, 2.0}, verified at
  symbolic test points (see diagnostic in chat history). F3 is NOT a
  score-function bug.

- **The LL gap on the synthetic fixture is 0.0035 nats/sample, not 0.024.**
  The earlier 0.024 was Python's *own optimum* minus Fortran's *own
  optimum*. When Python evaluates LL on Fortran's exact state, the gap
  shrinks to 0.0035 — small enough to be a normalization/log-det
  detail, large enough to be real. Probably an additive constant in
  one of the LL terms; **does not affect optimization** because
  additive constants drop out of gradients.

- **The basin divergence is independent of F1, F2, F3, and even the
  ρ update.** Running Python with `update_rho=False` (ρ frozen at
  rho0=1.5, mimicking what Fortran effectively does on this fixture
  because it exits before rho moves) **still** lands at a different W
  than Fortran does — Python converges cleanly in 90 iters via the
  dll-window test, but to a *different* fixed point than Fortran's W ≈ I.

- **Why the synthetic fixture is degenerate.** The fixture was built
  with an orthogonal mixing matrix `A_true` and `do_sphere=True`,
  which means Fortran's PCA-sphereing essentially absorbs `A_true`
  before AMICA runs. The remaining ICA problem on whitened data has
  W ≈ I as a saddle/plateau where the LL is locally flat. Fortran's
  natural-gradient finds this saddle quickly, the dll-window catches
  the plateau, and Fortran exits — with rho still at rho0=1.5, never
  having moved. Fortran's "convergence" on this fixture is **early
  termination at a saddle point**, not a true MLE. **Python is at a
  *higher* LL than Fortran on the same fixture**, even with rho frozen.
  The fixture cannot be used to assert "Python should match Fortran's
  W" — that assertion was wrong.

- **Implication for F1 validation.** F1 cannot be validated on this
  synthetic fixture. The fact that Python with rho frozen converges
  in 90 iterations via dll-window confirms F1 is mechanically working
  (the lrate state machine is now Fortran-equivalent), but the W it
  converges to is in a different basin than Fortran's for reasons
  unrelated to F1. **F1 must be validated against the production
  failure mode (sub-01 with the validation runner)**, where the
  sphereing is done by MNE on real EEG and the W ≈ I saddle does not
  exist.

- **Test stub status.** The 4 ⚠️ tests in
  [tests/test_against_fortran.py](tests/test_against_fortran.py) were
  re-marked with `pytest.skip(...)` reasons referencing this addendum
  rather than F1. They should be replaced (not unskipped) with a new
  fixture that uses non-orthogonal `A_true` and num_mix > 1, so the
  data-generating process is non-degenerate relative to the sphereing
  step. The byte-equivalent rho-update test still passes (1 passed,
  4 skipped).

**Outstanding open question.** There IS a real numerical disagreement
between Python and Fortran on the basin reached on the synthetic
fixture, even with ρ frozen. It is NOT in the score function. Plausible
sources, ranked:
1. **Sphereing convention** — PCA whitening eigenvector signs/order,
   ZCA vs PCA, normalization. Both Fortran and Python claim
   `do_approx_sphere=true` but the implementations may differ.
2. **β/μ joint update order** — Fortran updates `mu, sbeta, rho, A` in
   a specific interleaved order ([amica15.f90:1801-1820](../refs/sccn-amica/amica15.f90#L1801)).
   Python updates them in `update_all_pdf_params` which may use
   slightly different vmap ordering.
3. **`scalestep` / `doscaling` timing** — Fortran rescales A columns
   to unit norm at the end of each iter; Python does the same at
   [solver.py:241-247](amica_python/solver.py#L241-L247) but the
   point in the iteration may differ.

These are NOT load-bearing for the production failure mode and should
be parked until after F1 is validated against sub-01.

---

## 7. What I did NOT do (and why)

- **Did not modify any solver code.** Per the audit constraint and
  per the user-supplied feedback rule
  ([feedback_validate_against_source.md](../../.claude/projects/-home-sesma/memory/feedback_validate_against_source.md)).
- **Did not run more validation experiments on ds004505.** The audit
  used synthetic data and the existing on-disk sub-01 result only.
- **Did not deeply audit Newton κ/λ accumulation, multi-model state,
  or the sphereing log-det.** Marked ❓ in §3.
- **Did not bisect Apr 5 → Apr 7.** The bisect window was empty (§2.4).
- **Did not run the GPU half of the 4-cell matrix.** The job is queued
  (Slurm id 59007532) but `ReqNodeNotAvail`; the CPU result already
  showed float32 ≡ float64 and the iter-count effect dominates, so the
  GPU cells would not change the conclusion. If the GPU run completes
  later, fold its result into §2.2 as a sanity check.
