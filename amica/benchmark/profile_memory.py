"""Where does the peak memory of a fit actually go?

``profile_cpu.py`` answered the same question for time and the answer decided
the optimisation (transcendentals, not BLAS). This does it for memory, because
the obvious suspects disagree about the answer:

* Preprocessing materialises the recording several times over -- an input copy,
  a centred copy in the covariance, another centred copy in the sphering.
* The E-step's per-mixture intermediates are ``n_comp x n_mix x B`` each, which
  at full batch is many times the recording.

Both are real, but they scale differently -- the first with the recording, the
second with ``n_comp x n_mix`` -- so which one dominates depends on the shape of
the problem, and fixing the wrong one buys nothing. This samples RSS on a
background thread and splits the trace at the phase boundaries.

Two things are needed to make the split honest:

* JAX dispatches asynchronously, so a phase boundary in Python time is not a
  boundary in allocation time. Every boundary blocks on the arrays crossing it.
* Phases are attributed from the sampled trace, never from ``peak_wset``: a
  high-water mark only ever rises, so it would credit the whole peak to whichever
  phase happened to run last.

The high-water mark is still reported alongside, because the trace has the
opposite weakness -- it cannot see an allocation that lived and died between two
samples. Attribution comes from the trace; the true peak comes from the mark.
A large gap between them is itself the finding.

Usage::

    python -m amica.benchmark.profile_memory                  # user preset
    python -m amica.benchmark.profile_memory --preset benchmark
    python -m amica.benchmark.profile_memory --n-channels 64 --n-samples 400000
"""

from __future__ import annotations

import argparse
import json
import platform
import threading
import time

import numpy as np


class RssSampler:
    """Sample RSS on a background thread and timestamp every reading.

    An instantaneous trace, not a high-water mark: the point is to know *when*
    memory was held, which a high-water mark cannot tell you.
    """

    def __init__(self, interval_s: float = 0.005) -> None:
        import psutil

        self._proc = psutil.Process()
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float]] = []  # (t, gib)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(
                    (time.perf_counter(), self._proc.memory_info().rss / 1024**3)
                )
            except Exception:
                break
            self._stop.wait(self._interval)

    def __enter__(self) -> RssSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def peak_between(self, t0: float, t1: float) -> float:
        vals = [g for t, g in self.samples if t0 <= t <= t1]
        return max(vals) if vals else float("nan")

    def at(self, t: float) -> float:
        """RSS at the sample nearest ``t`` -- the level a phase started from."""
        if not self.samples:
            return float("nan")
        return min(self.samples, key=lambda s: abs(s[0] - t))[1]


def peak_rss_gib() -> float:
    """The OS high-water mark for the process.

    The sampled trace can only see memory that was held across a sample, so a
    short-lived allocation between two samples is invisible to it. This cannot
    attribute a peak to a phase, but it does not miss one -- so the two are
    reported together and a large gap between them means a transient spike.
    """
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


def _settle(*arrays) -> None:
    """Force pending JAX work, so a Python-time boundary is an allocation-time
    boundary. Without this the covariance's buffers can be charged to the fit."""
    for a in arrays:
        try:
            a.block_until_ready()
        except AttributeError:
            pass


class PhaseLog:
    """Records (name, t_enter, t_exit) as the fit crosses each boundary."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, float, float]] = []

    def wrap(self, module, attr: str, name: str, settle_result: bool = True):
        original = getattr(module, attr)

        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            if settle_result:
                _settle(*(out if isinstance(out, tuple) else (out,)))
            self.spans.append((name, t0, time.perf_counter()))
            return out

        setattr(module, attr, wrapper)
        return original


def make_data(n_channels: int, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.laplace(size=(n_channels, n_samples))
    mixing = rng.normal(size=(n_channels, n_channels))
    return mixing @ s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-channels", type=int, default=30)
    ap.add_argument("--n-samples", type=int, default=166800)
    ap.add_argument("--n-mix", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--preset",
        choices=("user", "benchmark"),
        default="user",
        help=(
            "'user' fits raw sensor data (mean removal + sphering on); "
            "'benchmark' mirrors the cross-implementation harness, which "
            "PCA-projects first and so fits with both switched off"
        ),
    )
    ap.add_argument(
        "--chunk-size",
        default="auto",
        help="'auto', 'none' (full batch), or an integer",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import psutil  # noqa: F401  -- fail early and clearly if absent

    from amica import Amica, AmicaConfig
    from amica import preprocessing as prep
    from amica import solver as solver_mod

    if args.chunk_size == "none":
        chunk_size = None
    elif args.chunk_size == "auto":
        chunk_size = "auto"
    else:
        chunk_size = int(args.chunk_size)

    do_sphere = args.preset == "user"
    cfg = AmicaConfig(
        num_models=1,
        num_mix_comps=args.n_mix,
        max_iter=args.max_iter,
        do_sphere=do_sphere,
        do_mean=do_sphere,
        chunk_size=chunk_size,
        fix_init=True,
    )

    phases = PhaseLog()
    phases.wrap(solver_mod, "preprocess_data", "preprocess")
    phases.wrap(prep, "compute_covariance", "  covariance")
    phases.wrap(prep, "apply_sphering", "  sphering")

    recording_gib = args.n_channels * args.n_samples * 8 / 1024**3

    with RssSampler() as sampler:
        time.sleep(0.05)  # a few samples of the pre-allocation floor
        t_base = time.perf_counter()
        baseline = sampler.at(t_base)

        data = make_data(args.n_channels, args.n_samples, args.seed)
        t_loaded = time.perf_counter()

        model = Amica(cfg)
        t_fit0 = time.perf_counter()
        result = model.fit(data)
        _settle(result.unmixing_matrix_white_)
        t_fit1 = time.perf_counter()
        time.sleep(0.05)
    high_water = peak_rss_gib()

    chunk_used = getattr(model, "effective_chunk_size_", None)

    spans = dict((n, (a, b)) for n, a, b in phases.spans)
    pre = spans.get("preprocess", (t_fit0, t_fit0))
    peak_overall = sampler.peak_between(t_base, t_fit1)
    peak_pre = sampler.peak_between(*pre)
    peak_loop = sampler.peak_between(pre[1], t_fit1)

    print(f"\n{args.preset} preset: {args.n_channels} ch x {args.n_samples} samples, "
          f"n_mix={args.n_mix}, {args.max_iter} iters, chunk_size={args.chunk_size}")
    print(f"one float64 copy of the recording : {recording_gib:.3f} GiB")
    print(f"blocking engaged                  : "
          f"{'no (full batch)' if chunk_used is None else f'{chunk_used} samples/block'}")
    print()
    print(f"{'phase':22} {'peak GiB':>9} {'above baseline':>15} {'copies':>8} {'s':>7}")
    print("-" * 66)
    print(f"{'baseline (imports)':22} {baseline:9.3f} {'':>15} {'':>8}")
    print(f"{'data allocated':22} {sampler.at(t_loaded):9.3f} "
          f"{sampler.at(t_loaded) - baseline:15.3f} "
          f"{(sampler.at(t_loaded) - baseline) / recording_gib:8.2f}")
    for name, a, b in phases.spans:
        pk = sampler.peak_between(a, b)
        print(f"{name:22} {pk:9.3f} {pk - baseline:15.3f} "
              f"{(pk - baseline) / recording_gib:8.2f} {b - a:7.2f}")
    print(f"{'EM loop':22} {peak_loop:9.3f} {peak_loop - baseline:15.3f} "
          f"{(peak_loop - baseline) / recording_gib:8.2f} {t_fit1 - pre[1]:7.2f}")
    print("-" * 66)
    print(f"{'whole fit':22} {peak_overall:9.3f} {peak_overall - baseline:15.3f} "
          f"{(peak_overall - baseline) / recording_gib:8.2f} {t_fit1 - t_fit0:7.2f}")
    print(f"{'OS high-water mark':22} {high_water:9.3f}   "
          f"(sampled trace misses spikes shorter than the 5 ms interval)")

    dominant = "preprocessing" if peak_pre >= peak_loop else "the EM loop"
    print(f"\nPeak is set by {dominant}: "
          f"preprocess {peak_pre - baseline:.3f} GiB vs loop {peak_loop - baseline:.3f} GiB "
          f"above baseline.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "preset": args.preset,
                    "n_channels": args.n_channels,
                    "n_samples": args.n_samples,
                    "n_mix": args.n_mix,
                    "max_iter": args.max_iter,
                    "chunk_size": args.chunk_size,
                    "chunk_used": chunk_used,
                    "recording_gib": recording_gib,
                    "baseline_gib": baseline,
                    "peak_overall_gib": peak_overall,
                    "peak_high_water_gib": high_water,
                    "peak_preprocess_gib": peak_pre,
                    "peak_loop_gib": peak_loop,
                    "spans": [
                        {"name": n, "peak_gib": sampler.peak_between(a, b), "seconds": b - a}
                        for n, a, b in phases.spans
                    ],
                    "trace": [{"t": t - t_base, "gib": g} for t, g in sampler.samples],
                },
                f,
                indent=2,
            )
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
