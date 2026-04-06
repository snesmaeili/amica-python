"""Stress tests for amica-python edge cases.

These tests verify that AMICA handles degenerate or difficult conditions
gracefully (no crashes, NaN, or hangs). Metric quality is documented
but not strictly asserted — the goal is robustness.

Run: python -m validation.test_stress
"""
import sys
import time
import traceback
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_amica_safe(data, label, **config_kwargs):
    """Run AMICA and return result or error string."""
    from amica_python import Amica, AmicaConfig

    defaults = dict(max_iter=100, num_mix_comps=2, do_newton=False)
    defaults.update(config_kwargs)

    print(f"\n  [{label}] data shape: {data.shape}")
    t0 = time.time()
    try:
        config = AmicaConfig(**defaults)
        model = Amica(config, random_state=42)
        result = model.fit(data)
        elapsed = time.time() - t0

        ll = result.log_likelihood
        has_nan = np.any(np.isnan(ll))
        final_ll = float(ll[-1]) if len(ll) > 0 else float('nan')

        # Reconstruction check
        sources = model.transform(data)
        recon = model.inverse_transform(sources)
        nrmse = float(np.mean((data - recon) ** 2) / (np.mean(data ** 2) + 1e-30))

        print(f"    OK: {result.n_iter} iters, LL={final_ll:.4f}, "
              f"NRMSE={nrmse:.2e}, NaN={has_nan}, time={elapsed:.1f}s")
        return {
            "status": "OK",
            "n_iter": result.n_iter,
            "final_ll": final_ll,
            "nrmse": nrmse,
            "has_nan": has_nan,
            "time": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"status": "FAILED", "error": str(e), "time": elapsed}


def test_very_short_data():
    """4 channels x 100 samples — near the data-adequacy boundary."""
    rng = np.random.RandomState(42)
    data = rng.randn(4, 100)
    return run_amica_safe(data, "very_short")


def test_rank_deficient():
    """32 channels but rank 20 — simulates interpolated channels."""
    rng = np.random.RandomState(42)
    n_ch, rank, n_samp = 32, 20, 5000
    U = rng.randn(n_ch, rank)
    S = rng.laplace(size=(rank, n_samp))
    data = U @ S
    return run_amica_safe(data, "rank_deficient", max_iter=50)


def test_outlier_heavy():
    """Data with 5% extreme outliers — tests rejection robustness."""
    rng = np.random.RandomState(42)
    n_ch, n_samp = 8, 5000
    S = rng.laplace(size=(n_ch, n_samp))
    A = rng.randn(n_ch, n_ch)
    data = A @ S
    # Inject outliers in 5% of samples
    n_outlier = n_samp // 20
    data[:, :n_outlier] *= 50
    return run_amica_safe(
        data, "outlier_heavy",
        do_reject=True, rejstart=2, rejint=3, rejsig=3.0, numrej=5,
    )


def test_outlier_no_rejection():
    """Same outlier data but without rejection — should still not crash."""
    rng = np.random.RandomState(42)
    n_ch, n_samp = 8, 5000
    S = rng.laplace(size=(n_ch, n_samp))
    A = rng.randn(n_ch, n_ch)
    data = A @ S
    data[:, :250] *= 50
    return run_amica_safe(data, "outlier_no_rej", do_reject=False)


def test_near_collinear():
    """Channels that are nearly identical — tests whitening stability."""
    rng = np.random.RandomState(42)
    n_samp = 3000
    base = rng.randn(1, n_samp)
    data = np.vstack([
        base + 1e-6 * rng.randn(1, n_samp),
        base + 1e-6 * rng.randn(1, n_samp),
        rng.laplace(size=(2, n_samp)),
    ])
    return run_amica_safe(data, "near_collinear", max_iter=30)


def test_single_channel():
    """1 channel — degenerate case, should error or return trivially."""
    data = np.random.RandomState(42).randn(1, 1000)
    return run_amica_safe(data, "single_channel", max_iter=10)


def test_high_dimensional():
    """64 channels x 50000 samples — scalability check."""
    rng = np.random.RandomState(42)
    n_ch, n_samp = 64, 50000
    S = rng.laplace(size=(n_ch, n_samp))
    A = rng.randn(n_ch, n_ch)
    data = A @ S
    return run_amica_safe(data, "high_dim_64ch", max_iter=30,
                          do_newton=True, num_mix_comps=3)


def test_newton_with_outliers():
    """Newton + outliers — tests Newton fallback to natural gradient."""
    rng = np.random.RandomState(42)
    n_ch, n_samp = 6, 3000
    S = rng.laplace(size=(n_ch, n_samp))
    A = rng.randn(n_ch, n_ch)
    data = A @ S
    data[:, :100] *= 100
    return run_amica_safe(
        data, "newton_outliers",
        do_newton=True, newt_start=10, max_iter=50,
    )


def main():
    import json

    print("=" * 60)
    print("AMICA STRESS TESTS")
    print("=" * 60)

    tests = [
        ("very_short_data", test_very_short_data),
        ("rank_deficient", test_rank_deficient),
        ("outlier_heavy", test_outlier_heavy),
        ("outlier_no_rejection", test_outlier_no_rejection),
        ("near_collinear", test_near_collinear),
        ("single_channel", test_single_channel),
        ("high_dimensional", test_high_dimensional),
        ("newton_with_outliers", test_newton_with_outliers),
    ]

    results = {}
    for name, test_fn in tests:
        results[name] = test_fn()

    # Summary
    print("\n" + "=" * 60)
    print("STRESS TEST SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        status = res["status"]
        info = ""
        if status == "OK":
            info = f"LL={res['final_ll']:.4f}, NRMSE={res['nrmse']:.2e}"
            if res["has_nan"]:
                info += " [NaN!]"
        else:
            info = res.get("error", "unknown error")
        print(f"  {name:25s} {status:6s}  {info}")

    output_file = RESULTS_DIR / "stress_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")

    n_fail = sum(1 for r in results.values() if r["status"] == "FAILED")
    n_nan = sum(1 for r in results.values()
                if r["status"] == "OK" and r.get("has_nan", False))
    print(f"\n{len(results)} tests: {len(results) - n_fail} OK, "
          f"{n_fail} FAILED, {n_nan} with NaN")


if __name__ == "__main__":
    main()
