"""Aggregate benchmark_report/*.json into a markdown report.

Reads all JSONs in validation/results/benchmark_report/, groups by
(dataset, device), averages over seeds, writes benchmark_report.md
with publication-ready tables.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

IN = Path("validation/results/benchmark_report")
OUT = Path("validation/results/benchmark_report/benchmark_report.md")

runs = []
for f in sorted(IN.glob("*.json")):
    with open(f) as fp:
        runs.append(json.load(fp))

if not runs:
    print(f"No JSONs in {IN}/")
    raise SystemExit(1)

# Group by (dataset, device, chunk_size)
groups = defaultdict(list)
for r in runs:
    key = (r["dataset"], r["device"], r["chunk_size"])
    groups[key].append(r)

def fmt(v, prec=2):
    if v is None: return "—"
    return f"{v:.{prec}f}"

lines = []
lines.append("# AMICA benchmark report\n")
lines.append(f"Runs: {len(runs)}  •  Groups: {len(groups)}  •  max_iter per run: {runs[0]['max_iter']}\n")
lines.append("Averaged over seeds (mean ± std).\n")
lines.append("")
lines.append("## Per-configuration summary\n")
lines.append("| dataset | device | chunk | shape | n_comp | seeds | wall (s) | JIT (s) | steady ms/iter | peak RSS (GB) | peak GPU (GB) | LL final |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

for (ds, dev, chunk), rs in sorted(groups.items()):
    shape = "x".join(str(s) for s in rs[0]["data_shape"])
    n_comp = rs[0]["n_comp"]
    wall = np.array([r["wall_total_s"] for r in rs])
    jit = np.array([r["jit_compile_s"] for r in rs])
    steady = np.array([r["steady_per_iter_ms"] for r in rs])
    rss = np.array([r["peak_rss_gb"] for r in rs])
    gpu_vals = [r["peak_gpu_mem_gb"] for r in rs if r["peak_gpu_mem_gb"] is not None]
    gpu_mean = np.mean(gpu_vals) if gpu_vals else None
    ll_final = np.array([r["ll_final"] for r in rs])
    chunk_str = "full" if chunk is None else str(chunk)
    lines.append(
        f"| {ds} | {dev} | {chunk_str} | {shape} | {n_comp} | {len(rs)} | "
        f"{fmt(wall.mean(),1)} ± {fmt(wall.std(),1)} | "
        f"{fmt(jit.mean(),1)} | "
        f"{fmt(steady.mean(),1)} ± {fmt(steady.std(),1)} | "
        f"{fmt(rss.mean(),2)} | "
        f"{fmt(gpu_mean,2) if gpu_mean is not None else '—'} | "
        f"{fmt(ll_final.mean(),4)}"
        f"{' ± ' + fmt(ll_final.std(), 6) if ll_final.std() > 1e-10 else ''} |"
    )

lines.append("")
lines.append("## Seed-level LL final (reproducibility check)\n")
lines.append("| dataset | device | chunk | seed=0 | seed=1 | seed=2 | max|Δ| |")
lines.append("|---|---|---|---|---|---|---|")
for (ds, dev, chunk), rs in sorted(groups.items()):
    by_seed = {r["seed"]: r["ll_final"] for r in rs}
    vals = [by_seed.get(s) for s in (0, 1, 2)]
    vv = [v for v in vals if v is not None]
    mx = max(vv) - min(vv) if len(vv) > 1 else 0.0
    chunk_str = "full" if chunk is None else str(chunk)
    lines.append(
        f"| {ds} | {dev} | {chunk_str} | "
        + " | ".join(fmt(v, 6) if v is not None else "—" for v in vals)
        + f" | {mx:.2e} |"
    )

# Cross-device speedup
lines.append("")
lines.append("## CPU vs GPU speedup (steady per-iter)\n")
lines.append("| dataset | CPU ms/iter | GPU ms/iter | speedup |")
lines.append("|---|---|---|---|")
by_ds = defaultdict(dict)
for (ds, dev, chunk), rs in groups.items():
    steady_mean = np.mean([r["steady_per_iter_ms"] for r in rs])
    by_ds[ds][dev] = steady_mean
for ds, devs in sorted(by_ds.items()):
    cpu = devs.get("cpu")
    gpu = devs.get("gpu")
    sp = f"{cpu/gpu:.1f}x" if (cpu and gpu) else "—"
    lines.append(f"| {ds} | {fmt(cpu,1) if cpu else '—'} | {fmt(gpu,1) if gpu else '—'} | {sp} |")

OUT.write_text("\n".join(lines) + "\n")
print(f"Wrote {OUT}")
print("\n".join(lines[:40]))
