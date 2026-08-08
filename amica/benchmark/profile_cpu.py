"""Where does a CPU iteration actually spend its time?

Written to settle one question before anything is optimised: is the CPU fit
dominated by the dense products that compute the sources, by the elementwise
transcendental work in the density evaluation, or by memory traffic. The three
have different fixes -- a faster BLAS, a fused kernel, and a smaller footprint --
and guessing which applies has a poor track record.

Three independent views, because no single one is trustworthy on its own:

``cost``
    XLA's own cost analysis of the compiled step: floating-point operations and
    bytes accessed. Their ratio is the arithmetic intensity, which says whether
    the step can possibly be compute-bound on this machine.

``hlo``
    A count of transcendental operations (exp/log/pow/tanh/rsqrt) and dense
    products in the optimised HLO, weighted by output size. This is what the
    compiler actually emitted, not what the source suggests, so it survives
    whatever fusion XLA decided to do.

``time``
    Measured wall time for the whole step, and for the density and
    source-projection paths in isolation at the same shapes. Attribution by
    subtraction is crude but it is empirical, and the parts are reported against
    the whole so that an attribution which does not add up is visible rather
    than quietly assumed.

Usage::

    python -m amica.benchmark.profile_cpu                      # default fixture
    python -m amica.benchmark.profile_cpu --n-components 64 --n-samples 785328
    python -m amica.benchmark.profile_cpu --dtype float32      # precision lever
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from typing import Any

import numpy as np


def _fmt(n: float) -> str:
    """Human-readable magnitude; these numbers span 1e3 to 1e12."""
    for unit, scale in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return f"{n:.2f}"


def make_fixture(n_components: int, n_samples: int, n_mix: int, seed: int, dtype):
    """Whitened super-Gaussian sources, the shape a fit sees after projection."""
    rng = np.random.default_rng(seed)
    y = rng.laplace(size=(n_components, n_samples)).astype(dtype)
    alpha = np.full((n_components, n_mix), 1.0 / n_mix, dtype=dtype)
    mu = rng.normal(scale=0.1, size=(n_components, n_mix)).astype(dtype)
    beta = np.ones((n_components, n_mix), dtype=dtype)
    rho = np.full((n_components, n_mix), 1.5, dtype=dtype)
    return y, alpha, mu, beta, rho


def time_call(fn, *args, repeats: int, warmup: int = 2) -> float:
    """Median seconds per call, with compilation and first-touch excluded.

    Median rather than mean: a single scheduling hiccup on a laptop otherwise
    dominates, and this is run on machines that are also doing other things.
    """
    import jax

    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


TRANSCENDENTAL = ("exponential", "log", "power", "tanh", "rsqrt", "logistic")
DENSE = ("dot", "convolution")


def hlo_op_profile(compiled) -> dict[str, Any]:
    """Count transcendental and dense ops in the optimised HLO, by output size.

    Reading the optimised text rather than the source is the point: XLA fuses
    aggressively on CPU, and a count taken from ``pdf.py`` would describe code
    that no longer exists by the time it runs.
    """
    try:
        text = compiled.as_text()
    except Exception as exc:  # pragma: no cover - backend dependent
        return {"available": False, "reason": str(exc)}

    # e.g. "%exponential.3 = f64[64,3,785328]{2,1,0} exponential(...)"
    pattern = re.compile(
        r"=\s*\w+\[([\d,]*)\][^=]*?\s(\w+)\(", re.MULTILINE)
    counts: dict[str, int] = {}
    elements: dict[str, float] = {}
    for dims, op in pattern.findall(text):
        base = op.split(".")[0]
        n = 1
        if dims:
            for d in dims.split(","):
                if d.strip().isdigit():
                    n *= int(d)
        counts[base] = counts.get(base, 0) + 1
        elements[base] = elements.get(base, 0.0) + n

    trans = {k: v for k, v in elements.items()
             if any(t in k for t in TRANSCENDENTAL)}
    dense = {k: v for k, v in elements.items() if any(d in k for d in DENSE)}
    return {
        "available": True,
        "transcendental_element_passes": sum(trans.values()),
        "transcendental_ops": {k: counts[k] for k in trans},
        "dense_output_elements": sum(dense.values()),
        "dense_ops": {k: counts[k] for k in dense},
        "n_hlo_instructions": sum(counts.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-components", type=int, default=32)
    ap.add_argument("--n-samples", type=int, default=100_000)
    ap.add_argument("--n-mix", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--json", type=str, default=None,
                    help="also write the measurements here")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    if args.dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = np.float64 if args.dtype == "float64" else np.float32

    from amica import pdf

    C, T, K = args.n_components, args.n_samples, args.n_mix
    y, alpha, mu, beta, rho = make_fixture(C, T, K, args.seed, dtype)
    yj, aj, mj, bj, rj = (jnp.asarray(v) for v in (y, alpha, mu, beta, rho))
    Wj = jnp.asarray(np.random.default_rng(args.seed).normal(size=(C, C)).astype(dtype))

    print(f"platform      : {platform.processor() or platform.machine()}")
    print(f"jax backend   : {jax.default_backend()}  devices={jax.device_count()}")
    print(f"fixture       : C={C} T={T:,} K={K} dtype={args.dtype}")
    print(f"source array  : {C * T * 8 / 1e9 if dtype is np.float64 else C * T * 4 / 1e9:.2f} GB")
    print()

    # --- the two paths, each over the whole (C, T) array -------------------
    # Responsibilities: the density evaluation, per component and mixture.
    resp = jax.jit(jax.vmap(pdf.compute_responsibilities, in_axes=(0, 0, 0, 0, 0)))
    # Source projection: the dense product the E-step opens with.
    proj = jax.jit(lambda W, Y: W @ Y)

    t_resp = time_call(resp, yj, aj, mj, bj, rj, repeats=args.repeats)
    t_proj = time_call(proj, Wj, yj, repeats=args.repeats)

    print("=== measured, same shapes ===")
    print(f"  responsibilities (density) : {t_resp * 1e3:8.1f} ms")
    print(f"  source projection (W @ Y)  : {t_proj * 1e3:8.1f} ms")
    ratio = t_resp / t_proj if t_proj > 0 else float("inf")
    print(f"  density / dense-product    : {ratio:8.1f}x")
    print()

    # --- what XLA thinks it is doing ---------------------------------------
    compiled = resp.lower(yj, aj, mj, bj, rj).compile()
    cost: dict[str, Any] = {}
    try:
        raw = compiled.cost_analysis()
        cost = raw[0] if isinstance(raw, (list, tuple)) else dict(raw)
    except Exception as exc:  # pragma: no cover - backend dependent
        cost = {"unavailable": str(exc)}

    flops = float(cost.get("flops", 0) or 0)
    bytes_accessed = float(cost.get("bytes accessed", 0) or 0)
    print("=== XLA cost analysis (density step) ===")
    if flops or bytes_accessed:
        print(f"  flops                      : {_fmt(flops)}")
        print(f"  bytes accessed             : {_fmt(bytes_accessed)}")
        if bytes_accessed:
            print(f"  arithmetic intensity       : {flops / bytes_accessed:8.2f} flop/byte")
        print(f"  implied bandwidth at measured time : "
              f"{_fmt(bytes_accessed / t_resp)}B/s")
    else:
        print(f"  unavailable: {cost}")
    print()

    hlo = hlo_op_profile(compiled)
    print("=== optimised HLO (density step) ===")
    if hlo.get("available"):
        passes = hlo["transcendental_element_passes"]
        print(f"  transcendental element-passes : {_fmt(passes)}")
        print(f"  over C*K*T = {_fmt(C * K * T)} -> {passes / (C * K * T):.1f} passes/element")
        print(f"  transcendental ops            : {hlo['transcendental_ops']}")
        print(f"  dense ops                     : {hlo['dense_ops'] or 'none'}")
        print(f"  total HLO instructions        : {hlo['n_hlo_instructions']}")
    else:
        print(f"  unavailable: {hlo.get('reason')}")
    print()

    verdict = ("transcendental-dominated" if ratio > 3
               else "mixed" if ratio > 1
               else "dense-product-dominated")
    print(f"=== verdict: {verdict} ===")
    print("  The density path is the target for fusion or a hand-written kernel"
          if ratio > 3 else
          "  A faster BLAS or better blocking matters more than fusing the density")

    out = {
        "platform": platform.processor(),
        "jax_backend": jax.default_backend(),
        "n_components": C, "n_samples": T, "n_mix": K, "dtype": args.dtype,
        "t_responsibilities_s": t_resp,
        "t_source_projection_s": t_proj,
        "density_over_dense_ratio": ratio,
        "cost_analysis": cost,
        "hlo": hlo,
        "verdict": verdict,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
