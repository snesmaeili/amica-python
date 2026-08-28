"""Check the current solver against the validated pre-optimization baseline.

The default baseline (``450a63cd``) is the last commit before the numerical
optimizations used for the 0.2.0 release. The script checks three execution
paths because they carry different reproducibility guarantees:

* full batch: current fused E-step with ``chunk_size=None``;
* blocked: the current ``chunk_size="auto"`` default;
* classic: the pre-fusion compatibility E-step.

Floating-point regrouping means bit identity is informative but not required.
The command fails if the unmixing matrices, likelihood history, iteration
count, or matched component directions leave the tolerances printed below.

The baseline is checked out into a temporary Git worktree and both revisions
run in isolated subprocesses via ``PYTHONPATH``. No second installation is
needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = "450a63cd"
MAX_REL_W = 1e-4
MAX_REL_LL = 1e-10
MIN_MATCHED_CORR = 0.999999


class FitResult(TypedDict):
    """Numerical outputs retained from one benchmark fit."""

    W: np.ndarray
    log_likelihood: np.ndarray
    n_iter: int


class Comparison(TypedDict):
    """Metrics comparing one current fit to the baseline."""

    bit_identical: bool
    relative_W: float
    relative_ll: float
    matched_correlation: float
    iterations_match: bool


WORKER = r"""
import json
import importlib
import sys

import numpy as np

package = importlib.import_module(sys.argv[5])
Amica = package.Amica
AmicaConfig = package.AmicaConfig

cfg_kw = json.loads(sys.argv[1])
out = sys.argv[2]
n_samples = int(sys.argv[3])
max_iter = int(sys.argv[4])

rng = np.random.default_rng(20260809)
sources = rng.laplace(size=(12, n_samples))
data = rng.normal(size=(12, 12)) @ sources

config = AmicaConfig(
    num_models=1,
    num_mix_comps=3,
    max_iter=max_iter,
    dtype="float64",
    fix_init=True,
    **cfg_kw,
)
result = Amica(config, random_state=42).fit(data)
W = np.asarray(result.unmixing_matrix_white_, dtype=np.float64)
if W.ndim == 3:
    W = W[:, :, 0]
np.savez(
    out,
    W=W,
    log_likelihood=np.asarray(result.log_likelihood, dtype=np.float64),
    n_iter=np.asarray(result.n_iter, dtype=np.int64),
)
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-ref",
        default=DEFAULT_BASELINE,
        help=f"Git revision used as the numerical reference (default: {DEFAULT_BASELINE}).",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=REPO_ROOT,
        help="Current checkout to evaluate (default: repository containing this script).",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable used for each isolated fit.",
    )
    parser.add_argument("--n-samples", type=int, default=40_000)
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument(
        "--backend",
        choices=("cpu", "numpy"),
        default="cpu",
        help="Use JAX on CPU or force the NumPy fallback (default: cpu).",
    )
    return parser.parse_args()


def _run(
    checkout: Path,
    cfg_kw: dict[str, Any],
    tag: str,
    *,
    python: Path,
    scratch: Path,
    n_samples: int,
    max_iter: int,
    backend: str,
) -> FitResult:
    worker = scratch / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    output = scratch / f"result-{tag}.npz"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(checkout)
    env["JAX_ENABLE_X64"] = "1"
    if backend == "numpy":
        env["AMICA_NO_JAX"] = "1"
        env.pop("JAX_PLATFORM_NAME", None)
    else:
        env.pop("AMICA_NO_JAX", None)
        env["JAX_PLATFORM_NAME"] = "cpu"
    if (checkout / "jamica").is_dir():
        import_name = "jamica"
    elif (checkout / "amica").is_dir():
        import_name = "amica"
    else:
        raise RuntimeError(f"no jamica or amica import package found in {checkout}")
    process = subprocess.run(
        [
            str(python),
            str(worker),
            json.dumps(cfg_kw),
            str(output),
            str(n_samples),
            str(max_iter),
            import_name,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(
            f"{tag} failed with exit code {process.returncode}:\n{process.stderr[-4000:]}"
        )
    with np.load(output) as result:
        return {
            "W": result["W"].copy(),
            "log_likelihood": result["log_likelihood"].copy(),
            "n_iter": int(result["n_iter"]),
        }


def _max_relative_error(reference: np.ndarray, actual: np.ndarray) -> float:
    denominator = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(reference - actual) / denominator))


def _worst_matched_correlation(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    actual = actual / np.linalg.norm(actual, axis=1, keepdims=True)
    correlations = np.abs(reference @ actual.T)
    rows, columns = linear_sum_assignment(-correlations)
    return float(np.min(correlations[rows, columns]))


def _compare(reference: FitResult, actual: FitResult) -> Comparison:
    ref_W = np.asarray(reference["W"])
    got_W = np.asarray(actual["W"])
    ref_ll = np.asarray(reference["log_likelihood"])
    got_ll = np.asarray(actual["log_likelihood"])
    if ref_W.shape != got_W.shape:
        raise RuntimeError(f"unmixing shape changed from {ref_W.shape} to {got_W.shape}")
    if ref_ll.shape != got_ll.shape:
        raise RuntimeError(
            f"likelihood-history shape changed from {ref_ll.shape} to {got_ll.shape}"
        )
    if not np.isfinite(got_W).all() or not np.isfinite(got_ll).all():
        raise RuntimeError("current result contains NaN or Inf")
    return {
        "bit_identical": np.array_equal(ref_W, got_W) and np.array_equal(ref_ll, got_ll),
        "relative_W": _max_relative_error(ref_W, got_W),
        "relative_ll": _max_relative_error(ref_ll, got_ll),
        "matched_correlation": _worst_matched_correlation(ref_W, got_W),
        "iterations_match": reference["n_iter"] == actual["n_iter"],
    }


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    process = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return process.stdout.strip()


def main() -> int:
    """Run the regression comparison and return a process exit status."""
    args = _parse_args()
    if args.n_samples < 12:
        raise ValueError("--n-samples must be >= 12")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be >= 1")
    current = args.current.resolve()
    python = args.python.resolve()
    baseline_hash = _git("rev-parse", args.baseline_ref)
    current_hash = _git("-C", str(current), "rev-parse", "HEAD")

    cases: list[tuple[str, dict[str, Any]]] = [
        ("full-batch", {"chunk_size": None}),
        ("blocked", {"chunk_size": "auto"}),
        ("classic", {"chunk_size": None, "estep": "classic"}),
    ]

    with tempfile.TemporaryDirectory(prefix="jamica-regression-") as temp_dir:
        scratch = Path(temp_dir)
        baseline = scratch / "baseline"
        _git("worktree", "add", "--detach", str(baseline), baseline_hash)
        try:
            reference = _run(
                baseline,
                {"chunk_size": None},
                "baseline",
                python=python,
                scratch=scratch,
                n_samples=args.n_samples,
                max_iter=args.max_iter,
                backend=args.backend,
            )
            results = {
                name: _run(
                    current,
                    config,
                    name,
                    python=python,
                    scratch=scratch,
                    n_samples=args.n_samples,
                    max_iter=args.max_iter,
                    backend=args.backend,
                )
                for name, config in cases
            }
        finally:
            _git("worktree", "remove", "--force", str(baseline))

    print(f"baseline: {baseline_hash[:12]} ({args.baseline_ref})")
    print(f"current : {current_hash[:12]} ({current})")
    print(f"backend : {args.backend}; samples={args.n_samples}; iterations={args.max_iter}")
    print()
    print(
        f"{'configuration':16} {'exact':>7} {'rel dW':>12} {'rel dLL':>12} "
        f"{'worst |r|':>12} {'n_iter':>8}"
    )
    print("-" * 75)

    passed = True
    for name, _ in cases:
        comparison = _compare(reference, results[name])
        case_passed = (
            comparison["relative_W"] <= MAX_REL_W
            and comparison["relative_ll"] <= MAX_REL_LL
            and comparison["matched_correlation"] >= MIN_MATCHED_CORR
            and comparison["iterations_match"]
        )
        passed &= case_passed
        print(
            f"{name:16} "
            f"{('yes' if comparison['bit_identical'] else 'no'):>7} "
            f"{comparison['relative_W']:12.2e} "
            f"{comparison['relative_ll']:12.2e} "
            f"{comparison['matched_correlation']:12.10f} "
            f"{('match' if comparison['iterations_match'] else 'DIFF'):>8}"
        )

    print()
    print(
        f"required: rel dW <= {MAX_REL_W:.0e}, rel dLL <= {MAX_REL_LL:.0e}, "
        f"worst |r| >= {MIN_MATCHED_CORR:.6f}, and matching n_iter"
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
