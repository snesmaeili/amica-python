"""How does a fit's per-iteration cost scale with recording length and rank?

This exists to say which axis may be normalised away in a cross-platform
comparison and which may not. Two panels measured on different machines with
different problem sizes cannot be compared unless the scaling in the differing
axis is known, and assuming a law is how a benchmark ends up reporting a
correction it never measured.

Cost per iteration is expected to be

    a * n_comp * n_mix * n_samples      the E-step's generalized-Gaussian work
  + b * n_comp^2 * n_samples            y = W X and g y^T
  + c * n_comp^3                        the Newton solve and pinv

so it should be linear in ``n_samples`` at any rank, and neither linear nor
quadratic in ``n_comp`` -- it crosses over from one to the other as the GEMMs
overtake the transcendentals. That is a prediction, and this measures it.

Timing comes from the fit's own ``iteration_times``, with the first few
iterations dropped: the first pays JIT compilation, which is a fixed cost and
would otherwise be smeared across the per-iteration figure and inflate the small
problems most.

Usage::

    python -m jamica.benchmark.profile_scaling --axis samples --n-components 30
    python -m jamica.benchmark.profile_scaling --axis components --n-samples 200000
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def steady_ms_per_iter(
    n_components: int, n_samples: int, n_mix: int, max_iter: int, chunk_size, seed: int = 0
) -> dict:
    """Median per-iteration wall time after compilation has been paid."""
    from jamica import Amica, AmicaConfig

    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_components, n_components)) @ rng.laplace(
        size=(n_components, n_samples)
    )

    cfg = AmicaConfig(
        num_models=1,
        num_mix_comps=n_mix,
        max_iter=max_iter,
        do_sphere=False,
        do_mean=False,
        chunk_size=chunk_size,
        fix_init=True,
    )
    model = Amica(cfg)
    result = model.fit(data)

    times = np.asarray(result.iteration_times, dtype=float)
    warmup = min(3, max(1, len(times) // 5))
    steady = times[warmup:]
    return {
        "n_components": n_components,
        "n_samples": n_samples,
        "n_mix": n_mix,
        "steady_ms": float(np.median(steady) * 1000),
        "spread_pct": float((steady.max() - steady.min()) / np.median(steady) * 100),
        "first_iter_ms": float(times[0] * 1000),
        "block": model.effective_chunk_size_,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--axis", choices=("samples", "components"), default="samples")
    ap.add_argument("--n-components", type=int, default=30)
    ap.add_argument("--n-samples", type=int, default=200000)
    ap.add_argument("--n-mix", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--chunk-size", default="auto", help="'auto', 'none', or an int")
    ap.add_argument(
        "--values", default=None, help="comma-separated sweep values (defaults per axis)"
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    chunk = (
        None
        if args.chunk_size == "none"
        else ("auto" if args.chunk_size == "auto" else int(args.chunk_size))
    )

    if args.values:
        values = [int(v) for v in args.values.split(",")]
    elif args.axis == "samples":
        values = [50_000, 100_000, 200_000, 400_000, 800_000]
    else:
        values = [16, 24, 32, 48, 64, 96]

    rows = []
    print(
        f"axis={args.axis}  n_mix={args.n_mix}  chunk_size={args.chunk_size}  "
        f"max_iter={args.max_iter}"
    )
    if args.axis == "samples":
        print(f"fixed n_components={args.n_components}\n")
        print(
            f"{'samples':>9} {'ms/iter':>9} {'ns/sample':>11} {'spread':>7} "
            f"{'block':>7} {'1st iter ms':>12}"
        )
    else:
        print(f"fixed n_samples={args.n_samples}\n")
        print(
            f"{'n_comp':>7} {'ms/iter':>9} {'per C':>8} {'per C^2':>9} {'spread':>7} "
            f"{'block':>7} {'1st iter ms':>12}"
        )
    print("-" * 66)

    for v in values:
        C = args.n_components if args.axis == "samples" else v
        T = v if args.axis == "samples" else args.n_samples
        r = steady_ms_per_iter(C, T, args.n_mix, args.max_iter, chunk)
        rows.append(r)
        if args.axis == "samples":
            print(
                f"{T:9d} {r['steady_ms']:9.2f} {r['steady_ms'] * 1e6 / T:11.1f} "
                f"{r['spread_pct']:6.1f}% {r['block']!s:>7} {r['first_iter_ms']:12.0f}"
            )
        else:
            print(
                f"{C:7d} {r['steady_ms']:9.2f} {r['steady_ms'] / C:8.3f} "
                f"{r['steady_ms'] / C**2:9.4f} {r['spread_pct']:6.1f}% "
                f"{r['block']!s:>7} {r['first_iter_ms']:12.0f}"
            )

    x = np.array(
        [r["n_samples"] if args.axis == "samples" else r["n_components"] for r in rows], dtype=float
    )
    y = np.array([r["steady_ms"] for r in rows])

    print()
    if args.axis == "samples":
        # Linearity is the whole question: if cost per sample is flat, dividing
        # by the sample count is a fair way to compare recordings of different
        # length; if it is not, that normalisation invents a correction.
        per = y / x
        print(
            f"cost per sample: {per.min() * 1e6:.0f} to {per.max() * 1e6:.0f} ns "
            f"({(per.max() / per.min() - 1) * 100:.0f}% spread over a "
            f"{x.max() / x.min():.0f}x range)"
        )
        verdict = (
            "linear -- normalising by sample count is fair"
            if (per.max() / per.min() - 1) < 0.15
            else "NOT linear -- do not normalise by sample count"
        )
        print(f"verdict: {verdict}")
    else:
        lin = float(np.sum(x * y) / np.sum(x * x))
        lin_err = float(np.max(np.abs(lin * x - y) / y) * 100)
        quad, *_ = np.linalg.lstsq(np.vstack([x, x**2]).T, y, rcond=None)
        quad_err = float(np.max(np.abs(np.vstack([x, x**2]).T @ quad - y) / y) * 100)
        print(f"pure-linear  ms/iter = {lin:.4f}*C            worst error {lin_err:.0f}%")
        print(
            f"linear+quad  ms/iter = {quad[0]:.4f}*C + {quad[1]:.6f}*C^2  "
            f"worst error {quad_err:.0f}%"
        )
        share = quad[1] * x[-1] ** 2 / (quad[0] * x[-1] + quad[1] * x[-1] ** 2) * 100
        print(f"at C={int(x[-1])} the quadratic (GEMM) term is {share:.0f}% of the cost")
        print(
            "verdict: "
            + (
                "linear enough to normalise by C"
                if lin_err < 15
                else "NOT linear -- match C across platforms, do not divide by it"
            )
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"axis": args.axis, "chunk_size": args.chunk_size, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
