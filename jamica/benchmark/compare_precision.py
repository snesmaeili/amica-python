"""What does single precision cost, and what does it buy, on a real fit?

The CPU path is limited by transcendental throughput rather than by BLAS or
memory bandwidth (see ``profile_cpu.py``), and single-precision transcendentals
are more than twice as fast, so ``dtype="float32"`` is the largest single lever
available on a machine without a GPU. It also roughly halves the working set,
which on a laptop is often the binding constraint rather than time.

Neither of those is worth anything if the decomposition changes materially, so
this fits the *same* data twice, at both precisions, from the same seed and
configuration, and reports the speed and memory won next to the agreement lost.
Agreement uses the same definition the manuscript's cross-implementation table
uses: worst Hungarian-matched, sign-aligned correlation between unmixing rows.

Usage::

    python -m jamica.benchmark.compare_precision                       # synthetic
    python -m jamica.benchmark.compare_precision --n-components 64 \
        --n-samples 200000 --max-iter 100
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np


def peak_rss_gib() -> float:
    """Process high-water mark, so a transient mid-fit peak is not missed."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):  # Windows
            return info.peak_wset / 1024**3
        return info.rss / 1024**3
    except Exception:
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return ru / (1024**3 if platform.system() == "Darwin" else 1024**2)
        except Exception:
            return float("nan")


def worst_matched_r(a: np.ndarray, b: np.ndarray) -> float:
    """Worst unsigned row correlation after Hungarian matching.

    Deliberately the manuscript's definition -- rows normalised, no mean
    centring -- so a number produced here is comparable with the
    cross-implementation table rather than merely similar to it.
    """
    from scipy.optimize import linear_sum_assignment

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    corr = np.abs(a @ b.T)
    row, col = linear_sum_assignment(-corr)
    return float(np.min(corr[row, col]))


def make_data(n_components: int, n_samples: int, seed: int) -> np.ndarray:
    """Super-Gaussian sources through a random mixing -- an ICA problem with a
    known answer, so a precision change that breaks separation is visible."""
    rng = np.random.default_rng(seed)
    s = rng.laplace(size=(n_components, n_samples))
    mixing = rng.normal(size=(n_components, n_components))
    return mixing @ s


def run_one(data: np.ndarray, dtype: str, max_iter: int, seed: int) -> dict:
    from jamica import Amica, AmicaConfig

    cfg = AmicaConfig(
        num_models=1,
        num_mix_comps=3,
        max_iter=max_iter,
        dtype=dtype,
        # Same starting point in both runs, so any difference in the result is
        # precision and not a different basin.
        fix_init=True,
    )
    model = Amica(cfg)
    t0 = time.perf_counter()
    result = model.fit(data)
    elapsed = time.perf_counter() - t0

    # Compare in whitened space: the sensor-space matrices fold in the whitener,
    # which is computed in float64 in both runs and would mask the difference
    # this is trying to measure.
    W = np.asarray(result.unmixing_matrix_white_, dtype=np.float64)
    if W.ndim == 3:  # (n_components, n_components, n_models)
        W = W[:, :, 0]
    ll = result.log_likelihood
    ll_final = float(np.asarray(ll).ravel()[-1]) if ll is not None else float("nan")
    return {
        "dtype": dtype,
        "fit_time_s": elapsed,
        "peak_rss_gib": peak_rss_gib(),
        "ll_final": ll_final,
        "W": W,
    }


def _worker(args) -> int:
    """Fit one precision and write the result. Runs as its own process.

    Peak RSS is a per-process high-water mark, so fitting both precisions in one
    process makes the second inherit the first's peak and the memory comparison
    -- half the reason to use float32 on a laptop -- silently reports no
    difference. One process per precision is the only way to measure it.
    """
    data = make_data(args.n_components, args.n_samples, args.seed)
    r = run_one(data, args.single, args.max_iter, args.seed)
    np.save(args.w_out, r.pop("W"))
    with open(args.result_out, "w", encoding="utf-8") as f:
        json.dump(r, f)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-components", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=100_000)
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None)
    # Internal: one precision, in a dedicated process, for an honest peak RSS.
    ap.add_argument("--single", choices=["float32", "float64"], default=None)
    ap.add_argument("--w-out", default=None)
    ap.add_argument("--result-out", default=None)
    args = ap.parse_args()

    if args.single:
        return _worker(args)

    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    print(f"platform : {platform.processor() or platform.machine()}")
    print(f"fixture  : C={args.n_components} T={args.n_samples:,} iterations={args.max_iter}")
    print()

    tmp = Path(tempfile.mkdtemp(prefix="amica_prec_"))
    runs = {}
    for dtype in ("float64", "float32"):
        print(f"fitting {dtype} ...", flush=True)
        w_out, r_out = tmp / f"W_{dtype}.npy", tmp / f"r_{dtype}.json"
        cp = subprocess.run(
            [
                sys.executable,
                "-m",
                "jamica.benchmark.compare_precision",
                "--single",
                dtype,
                "--n-components",
                str(args.n_components),
                "--n-samples",
                str(args.n_samples),
                "--max-iter",
                str(args.max_iter),
                "--seed",
                str(args.seed),
                "--w-out",
                str(w_out),
                "--result-out",
                str(r_out),
            ],
            capture_output=True,
            text=True,
            timeout=36000,
        )
        if cp.returncode != 0:
            print(cp.stderr[-2000:])
            raise SystemExit(f"{dtype} fit failed with exit {cp.returncode}")
        r = json.loads(r_out.read_text())
        r["W"] = np.load(w_out)
        runs[dtype] = r
        print(
            f"  {r['fit_time_s']:8.1f} s   peak {r['peak_rss_gib']:5.2f} GiB   "
            f"ll {r['ll_final']:.8f}"
        )

    f64, f32 = runs["float64"], runs["float32"]
    speedup = f64["fit_time_s"] / f32["fit_time_s"] if f32["fit_time_s"] else float("nan")
    agreement = worst_matched_r(f64["W"], f32["W"])
    dll = abs(f64["ll_final"] - f32["ll_final"])

    print()
    print("=== float32 against float64 ===")
    print(f"  speedup                      : {speedup:6.2f}x")
    print(
        f"  peak memory                  : {f32['peak_rss_gib']:.2f} vs "
        f"{f64['peak_rss_gib']:.2f} GiB"
    )
    print(f"  |delta final log-likelihood| : {dll:.3e}")
    print(f"  worst matched row |r|        : {agreement:.4f}")
    print()
    if agreement >= 0.99:
        print("  Same decomposition to within the agreement the cross-implementation")
        print("  table reports between independent implementations.")
    elif agreement >= 0.95:
        print("  Comparable to the spread between separate implementations; usable")
        print("  for exploratory work, not for a parity claim.")
    else:
        print("  Materially different decomposition. The speedup is not free here.")

    if args.json:
        out = {k: {kk: vv for kk, vv in v.items() if kk != "W"} for k, v in runs.items()}
        out["comparison"] = {
            "speedup": speedup,
            "worst_matched_r": agreement,
            "abs_delta_ll": dll,
            "n_components": args.n_components,
            "n_samples": args.n_samples,
            "max_iter": args.max_iter,
            "platform": platform.processor(),
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
