"""CPU benchmark: chunked vs full-batch E-step on sub-01.

Measures peak RSS and wall time for max_iter=100 on 118ch × 1.2M samples.
Writes results to validation/results/cpu_chunking_benchmark.json.
"""
import json, os, time, gc, resource
import numpy as np
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("DS_PATH", "/home/sesma/scratch/ds004505")

from amica_python import Amica, AmicaConfig

PREPROC = Path("validation/results/post_f1_audit/sub01_preproc.npz")
OUT = Path("validation/results/cpu_chunking_benchmark.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading sub-01 preprocessed data...")
z = np.load(PREPROC, allow_pickle=True)
data = z["data"]
n_comp = int(z["n_components"])
print(f"data: {data.shape}, n_comp={n_comp}")


def peak_rss_gb():
    """Peak resident set size in GB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def run(chunk_size, max_iter=100, label=""):
    gc.collect()
    print(f"\n=== {label} (chunk_size={chunk_size}) ===")
    cfg = AmicaConfig(
        num_models=1, num_mix_comps=3, max_iter=max_iter,
        pcakeep=n_comp, dtype="float64", lrate=0.01,
        chunk_size=chunk_size,
    )
    t0 = time.time()
    rss_before = peak_rss_gb()
    res = Amica(cfg, random_state=42).fit(data)
    wall = time.time() - t0
    rss_after = peak_rss_gb()
    ll = float(np.asarray(res.log_likelihood)[-1])
    W = np.asarray(res.unmixing_matrix_white_)
    print(f"  wall: {wall:.1f}s, peak RSS: {rss_after:.2f} GB (+{rss_after-rss_before:.2f})")
    print(f"  LL final: {ll:.6f}, n_iter: {res.n_iter}")
    return {
        "chunk_size": chunk_size,
        "wall_s": wall,
        "peak_rss_gb": rss_after,
        "n_iter": int(res.n_iter),
        "final_ll": ll,
        "W": W,
    }


results = []

r_full = run(None, max_iter=100, label="FULL-BATCH")
results.append({k: v for k, v in r_full.items() if k != "W"})

r_chunk = run(1024, max_iter=100, label="CHUNKED chunk=1024")
results.append({k: v for k, v in r_chunk.items() if k != "W"})

# Parity check
rel = np.max(np.abs(r_chunk["W"] - r_full["W"])) / max(np.max(np.abs(r_full["W"])), 1e-20)
ll_diff = abs(r_chunk["final_ll"] - r_full["final_ll"])
print(f"\n=== PARITY ===")
print(f"max|W_chunk - W_full| / max|W_full| = {rel:.2e}")
print(f"|LL_chunk - LL_full|                  = {ll_diff:.2e}")

rss_reduction = r_full["peak_rss_gb"] / max(r_chunk["peak_rss_gb"], 1e-6)
print(f"\n=== SPEED/MEMORY ===")
print(f"RSS ratio (full/chunk): {rss_reduction:.2f}x reduction")
print(f"Wall time ratio (chunk/full): {r_chunk['wall_s']/r_full['wall_s']:.2f}x")

summary = {
    "runs": results,
    "parity": {
        "W_rel_err": float(rel),
        "ll_diff": float(ll_diff),
    },
    "rss_reduction_factor": float(rss_reduction),
    "data_shape": list(data.shape),
    "n_comp": n_comp,
    "max_iter": 100,
}

with open(OUT, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved benchmark to {OUT}")
