"""Run post-F1+F2 AMICA directly on sub-01 preprocessed data.

Bypasses MNE ICA wrapper. Saves both the preprocessed data (for future
A/B tests) and the AMICA result (for comparison to the pre-F1 baseline).

Usage: python -u validation/audit_post_f1_direct.py
"""
from __future__ import annotations
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("mne").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("DS_PATH", "/home/sesma/scratch/ds004505")

from validation.run_highdens_validation import load_and_preprocess, determine_n_components
from amica_python import Amica, AmicaConfig

OUT_DIR = Path("validation/results/post_f1_audit")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREPROC_NPZ = OUT_DIR / "sub01_preproc.npz"
RESULT_PKL  = OUT_DIR / "amica_result.pkl"

# ---- Step 1: preprocess sub-01 (or load cached) ----
if PREPROC_NPZ.exists():
    print(f"Loading cached preprocess: {PREPROC_NPZ}")
    z = np.load(PREPROC_NPZ, allow_pickle=True)
    data = z["data"]
    n_components = int(z["n_components"])
    sfreq = float(z["sfreq"])
else:
    print("Preprocessing sub-01 ...")
    t0 = time.time()
    raw, _, _ = load_and_preprocess("sub-01", hp_freq=1.0, iclean_mode="none")
    n_components = determine_n_components(raw, max_components=60)
    data = raw.get_data(picks="eeg")
    sfreq = float(raw.info["sfreq"])
    np.savez(PREPROC_NPZ, data=data, n_components=n_components,
             sfreq=sfreq, ch_names=np.array(raw.ch_names))
    print(f"Cached to {PREPROC_NPZ} in {time.time()-t0:.1f}s")

n_chan, n_samp = data.shape
print(f"Data: {n_chan} ch x {n_samp} samples, n_components={n_components}, "
      f"sfreq={sfreq:.0f}, kappa={n_samp/n_chan**2:.1f}")

# ---- Step 2: run AMICA (post-F1+F2) ----
cfg = AmicaConfig(
    num_models=1,
    num_mix_comps=3,
    max_iter=2000,
    pcakeep=n_components,
    dtype="float64",
)
print(f"AMICA config: num_mix={cfg.num_mix_comps} max_iter={cfg.max_iter} "
      f"pcakeep={cfg.pcakeep} dtype={cfg.dtype} "
      f"minrho={cfg.minrho} maxrho={cfg.maxrho} "
      f"invsigmin={cfg.invsigmin} max_decs={cfg.max_decs}")

t0 = time.time()
m = Amica(cfg, random_state=42)
res = m.fit(data)
elapsed = time.time() - t0

with open(RESULT_PKL, "wb") as f:
    pickle.dump(res, f)
print(f"\nResult saved to {RESULT_PKL} ({elapsed:.1f}s)")

# ---- Step 3: diagnostic summary ----
rho = np.asarray(res.rho_)
n_at_floor = int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))
ll = np.asarray(res.log_likelihood)

print("\n" + "="*60)
print("POST-F1+F2 SUB-01 RESULT")
print("="*60)
print(f"  n_iter:            {res.n_iter}")
print(f"  converged:         {res.converged}")
print(f"  elapsed:           {elapsed:.1f} s")
print(f"  rho shape:         {rho.shape}")
print(f"  frac at floor=1.0: {n_at_floor}/{rho.size} = {n_at_floor/rho.size:.3f}")
for k in range(rho.shape[0]):
    nf = int(np.sum(np.isclose(rho[k], 1.0, atol=1e-6)))
    print(f"    mix {k}: {nf:3d}/{rho.shape[1]}  "
          f"range [{rho[k].min():.3f}, {rho[k].max():.3f}]  "
          f"mean={rho[k].mean():.3f}")
print(f"  LL: first={ll[0]:.6f}  final={ll[-1]:.6f}  delta={ll[-1]-ll[0]:+.6f}")

if len(ll) >= 11:
    deltas = np.diff(ll[-11:])
    print(f"  last 10 LL deltas:")
    for i, d in enumerate(deltas):
        print(f"    {i:2d}: {d:+.6e}")
    odd, evn = deltas[::2], deltas[1::2]
    if len(odd) > 1 and len(evn) > 1 and abs(np.mean(odd)) > 1e-15:
        ratio = abs(np.mean(odd) - np.mean(evn)) / abs(np.mean(odd))
        label = "2-STEP LIMIT CYCLE" if ratio > 0.4 else "no limit cycle"
        print(f"  odd/even separation: {ratio:.3f} -> {label}")

print(f"\n{'='*60}")
print("BASELINE (pre-F1, validation/results/amica_result.pkl)")
print("="*60)
base_pkl = Path("validation/results/amica_result.pkl")
if base_pkl.exists():
    with open(base_pkl, "rb") as f:
        base = pickle.load(f)
    rho_b = np.asarray(base.rho_)
    nb = int(np.sum(np.isclose(rho_b, 1.0, atol=1e-6)))
    ll_b = np.asarray(base.log_likelihood)
    print(f"  n_iter:            {base.n_iter}")
    print(f"  converged:         {base.converged}")
    print(f"  frac at floor=1.0: {nb}/{rho_b.size} = {nb/rho_b.size:.3f}")
    print(f"  LL: first={ll_b[0]:.6f}  final={ll_b[-1]:.6f}")
    print(f"  last 5 LL deltas:  {np.diff(ll_b[-6:])}")

    print(f"\n{'='*60}")
    print("DELTA (post-F1 minus baseline)")
    print("="*60)
    print(f"  frac_rho_at_floor:  {nb/rho_b.size:.3f} -> {n_at_floor/rho.size:.3f}")
    print(f"  n_iter:             {base.n_iter} -> {res.n_iter}")
    print(f"  converged:          {base.converged} -> {res.converged}")
    print(f"  ll final:           {ll_b[-1]:.6f} -> {ll[-1]:.6f}")
else:
    print(f"  (baseline pkl not found at {base_pkl})")
