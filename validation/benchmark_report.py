"""Unified benchmark: MNE sample + MoBI ds004505, CPU + GPU, multi-seed.

Usage:
  python validation/benchmark_report.py --dataset {mne,mobi} --device {cpu,gpu} --seed N [--chunk_size K]

Writes JSON to validation/results/benchmark_report/<dataset>_<device>_seed<N>_chunk<K>.json
Records: wall total, JIT compile, steady per-iter, peak RSS, peak GPU mem,
LL trajectory, final unmixing matrix hash, n_iter, data shape, n_comp.
"""
import argparse, gc, json, os, resource, time
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--dataset", choices=["mne", "mobi"], required=True)
p.add_argument("--device", choices=["cpu", "gpu"], required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--chunk_size", type=int, default=0, help="0 = full-batch")
p.add_argument("--max_iter", type=int, default=20)
p.add_argument("--n_comp", type=int, default=30)
args = p.parse_args()

os.environ["JAX_PLATFORMS"] = "cuda" if args.device == "gpu" else "cpu"
chunk_size = args.chunk_size if args.chunk_size > 0 else None

OUT = Path("validation/results/benchmark_report")
OUT.mkdir(parents=True, exist_ok=True)
tag = f"{args.dataset}_{args.device}_seed{args.seed}_chunk{chunk_size or 'full'}"
out_file = OUT / f"{tag}.json"

import jax
from amica_python import Amica, AmicaConfig


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def gpu_mem_gb():
    if args.device != "gpu":
        return None
    try:
        stats = jax.devices()[0].memory_stats()
        return stats.get("peak_bytes_in_use", 0) / (1024 ** 3)
    except Exception:
        return None


print(f"[{tag}] loading data...")
if args.dataset == "mne":
    import mne
    sample_path = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(
        str(sample_path / "MEG" / "sample" / "sample_audvis_raw.fif"),
        preload=True, verbose=False,
    )
    raw.pick_types(eeg=True, exclude="bads")
    raw.filter(1, 40, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    data = raw.get_data().astype(np.float64)
    n_ch, n_t = data.shape
    # sphere/pca
    data = data - data.mean(axis=1, keepdims=True)
    U, s, Vt = np.linalg.svd(data, full_matrices=False)
    data = (U[:, :args.n_comp] * (1.0 / s[:args.n_comp])).T @ data * np.sqrt(n_t)
    n_comp = args.n_comp
else:  # mobi
    PREPROC = Path("validation/results/post_f1_audit/sub01_preproc.npz")
    z = np.load(PREPROC, allow_pickle=True)
    data = z["data"]
    n_comp = int(z["n_components"])

print(f"[{tag}] data shape: {data.shape}, n_comp={n_comp}")

cfg = AmicaConfig(
    num_models=1, num_mix_comps=3, max_iter=args.max_iter,
    pcakeep=n_comp, dtype="float64", lrate=0.01,
    chunk_size=chunk_size,
)

gc.collect()
rss_before = peak_rss_gb()
t0 = time.time()
result = Amica(cfg, random_state=args.seed).fit(data)
wall = time.time() - t0
rss_after = peak_rss_gb()
gpu_peak = gpu_mem_gb()

ll = np.asarray(result.log_likelihood).tolist()
per_iter = np.asarray(result.iteration_times).tolist()
W = np.asarray(result.unmixing_matrix_white_)

jit_time = per_iter[0] if per_iter else 0.0
steady = per_iter[1:] if len(per_iter) > 1 else []
steady_mean_ms = float(np.mean(steady) * 1000) if steady else 0.0
steady_total = float(np.sum(steady)) if steady else 0.0

summary = {
    "tag": tag,
    "dataset": args.dataset,
    "device": args.device,
    "seed": args.seed,
    "chunk_size": chunk_size,
    "max_iter": args.max_iter,
    "data_shape": list(data.shape),
    "n_comp": int(n_comp),
    "wall_total_s": float(wall),
    "jit_compile_s": float(jit_time),
    "steady_per_iter_ms": steady_mean_ms,
    "steady_total_s": steady_total,
    "peak_rss_gb": float(rss_after),
    "rss_delta_gb": float(rss_after - rss_before),
    "peak_gpu_mem_gb": gpu_peak,
    "ll_first": float(ll[0]) if ll else None,
    "ll_final": float(ll[-1]) if ll else None,
    "ll_trajectory": ll,
    "n_iter": int(result.n_iter),
    "W_frobenius": float(np.linalg.norm(W)),
    "W_hash": float(np.sum(W * W)),
}

with open(out_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[{tag}] wall={wall:.1f}s jit={jit_time:.1f}s per-iter={steady_mean_ms:.1f}ms "
      f"peak_rss={rss_after:.2f}GB gpu_peak={gpu_peak} LL={ll[-1] if ll else 'NA'}")
print(f"[{tag}] saved: {out_file}")
