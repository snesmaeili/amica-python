"""YOR19: score a single continuing AMICA trajectory at intermediate iterations.

The question this answers is narrow and prespecified: on real EEG, does the
likelihood plateau long before the decomposition stops changing, and do the
non-likelihood outcomes (MIR, remnant PMI, near-dipolarity, map stability)
keep moving after it does?

The measurement must come from ONE continuing optimisation run, not from
separate fits at different iteration caps -- restarting would reset the
learning-rate ramp and the Newton state and would no longer be the same
trajectory. We therefore let the solver run once to --max-iter and snapshot
its state every --writestep iterations, then score each snapshot afterwards
with exactly the same artifact code used for the published benchmark
(``compute_v3_artifacts``), so the numbers are directly comparable.

This is an exploratory analysis. It makes no prediction that dipolarity must
increase; a flat trajectory is a real and reportable answer.

Run on a compute node only (it imports JAX and fits AMICA).
"""
from __future__ import annotations

import argparse
import inspect
import json
import shutil
import time
from pathlib import Path

import numpy as np

try:  # the cluster venv installs the package under its original name
    import amica_python as amica_pkg
    from amica_python import fit_ica
    from amica_python.solver import Amica
    from amica_python.benchmark.runner import (
        load_data,
        preprocess,
        compute_v3_artifacts,
    )
except ImportError:  # released name
    import amica as amica_pkg
    from amica import fit_ica
    from amica.solver import Amica
    from amica.benchmark.runner import (  # type: ignore[no-redef]
        load_data,
        preprocess,
        compute_v3_artifacts,
    )


def install_numbered_checkpoints(root: Path) -> list[Path]:
    """Make the solver's periodic checkpoint keep every snapshot.

    ``Amica.save`` normally writes to one directory and each checkpoint
    overwrites the last, which preserves only the final state. We redirect it
    to ``root/iter_<n>`` so the whole trajectory survives. The fit itself is
    untouched -- this only changes where bytes land.
    """
    written: list[Path] = []
    original_save = Amica.save

    def save_numbered(self, outdir):  # noqa: ANN001
        n_iter = int(getattr(self.result_, "n_iter", 0) or 0)
        target = root / f"iter_{n_iter:06d}"
        original_save(self, target)
        written.append(target)
        return None

    Amica.save = save_numbered  # type: ignore[method-assign]
    return written


def read_snapshot(d: Path, n_comp: int, n_chan: int) -> dict[str, np.ndarray]:
    """Read the Fortran-order binaries written by ``Amica.save``.

    The leading dimension of ``A`` is inferred from the file rather than
    assumed. When the fit runs on PCA-reduced data, the "sensor" axis of the
    saved mixing matrix is the retained space, not the original channel count,
    and hard-coding the channel count silently mis-shapes the array.
    """

    def rd(name: str, cols: int) -> np.ndarray:
        arr = np.fromfile(d / name, dtype="<f8")
        if arr.size % cols:
            raise ValueError(
                f"{d/name}: {arr.size} values is not divisible by {cols}"
            )
        rows = arr.size // cols
        # save() writes arr.T in C order, i.e. the original in Fortran order.
        return arr.reshape(cols, rows).T

    return {
        "W_white": rd("W", n_comp),
        "A_sensor": rd("A", n_comp),
        "LL": np.fromfile(d / "LL", dtype="<f8"),
    }


def matched_map_corr(A: np.ndarray, B: np.ndarray) -> float:
    """Median Hungarian-matched |r| between two sets of sensor maps."""
    from scipy.optimize import linear_sum_assignment

    Ac = A - A.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    An = Ac / np.linalg.norm(Ac, axis=0, keepdims=True)
    Bn = Bc / np.linalg.norm(Bc, axis=0, keepdims=True)
    C = np.abs(An.T @ Bn)
    r, c = linear_sum_assignment(-C)
    return float(np.median(C[r, c]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ds004505")
    p.add_argument("--subject", type=int, required=True)
    p.add_argument("--n-components", type=int, default=64)
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument("--writestep", type=int, default=250,
                   help="snapshot interval, in solver iterations")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--num-mix", type=int, default=3)
    p.add_argument("--input-level", default="bids")
    p.add_argument("--line-freq", type=float, default=60.0)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--keep-snapshots", action="store_true",
                   help="retain the raw checkpoint binaries (large)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snap_root = args.out_dir / "snapshots"
    snap_root.mkdir(exist_ok=True)

    print(f"[trajectory] package={getattr(amica_pkg, '__file__', '?')}")
    print(f"[trajectory] dataset={args.dataset} subject={args.subject} "
          f"max_iter={args.max_iter} writestep={args.writestep}")

    raw = load_data(args.dataset, args.subject, input_level=args.input_level)
    # Older checkouts of the benchmark runner hardcode the notch frequency and
    # take no line_freq argument. Which one a given venv resolves to is not
    # obvious from the outside -- the amica-python venv's editable install
    # points at a different tree than the capsule's -- so adapt rather than
    # assume, and say which one was used.
    if "line_freq" in inspect.signature(preprocess).parameters:
        raw = preprocess(raw, line_freq=args.line_freq)
        print(f"[trajectory] preprocess(line_freq={args.line_freq})")
    else:
        raw = preprocess(raw)
        print("[trajectory] preprocess() takes no line_freq; using its built-in notch")
    print(f"[trajectory] raw: {len(raw.ch_names)} ch, {raw.n_times} samples, "
          f"{raw.info['sfreq']} Hz")

    written = install_numbered_checkpoints(snap_root)

    t0 = time.perf_counter()
    ica = fit_ica(
        raw,
        n_components=args.n_components,
        max_iter=args.max_iter,
        num_mix=args.num_mix,
        random_state=args.random_state,
        fit_params={"writestep": args.writestep, "outdir": str(snap_root)},
    )
    fit_s = time.perf_counter() - t0
    print(f"[trajectory] fit done in {fit_s:.1f}s; "
          f"{len(written)} snapshots written")

    res = ica.amica_result_
    ll_full = np.asarray(res.log_likelihood, dtype=float)
    n_chan = ica.pca_components_.shape[1]

    # score every snapshot with the published artifact code
    snaps = sorted({d for d in written if d.exists()}, key=lambda d: d.name)
    base_unmix = ica.unmixing_matrix_.copy()
    base_mix = ica.mixing_matrix_.copy()

    # Diagnostic for the negative complete-MIR seen in the first smoke test.
    # The fitted object and the final checkpoint describe the same state, so
    # scoring both answers whether the swap itself is what breaks the metric.
    # If they differ only by a per-row scale, the checkpoint holds the in-loop
    # unmixing while the metric path expects the post-fit normalised one, and
    # MIR's scale invariance is lost because the sources are not rescaled with
    # it. Recorded rather than asserted: a mismatch is the finding, not a crash.
    # Per-PCA-component scaling that fit_ica applied to the input before
    # handing it to AMICA. fit_ica stores it precisely so downstream code can
    # map an in-loop matrix back to the fitted object's domain.
    comp_stds = np.asarray(ica._amica_comp_stds).squeeze()[np.newaxis, :]

    diag: dict = {}
    if snaps:
        art_direct = compute_v3_artifacts(ica, raw)
        diag["mir_direct_from_fitted_ica"] = (
            (art_direct.get("complete_mir") or {}).get("kbits_per_sec"))
        diag["nd_10_direct_from_fitted_ica"] = (
            (art_direct.get("dipolarity") or {}).get("nd_10_percent"))
        diag["dipolarity_method"] = (
            (art_direct.get("dipolarity") or {}).get("method"))
        w_final = read_snapshot(snaps[-1], args.n_components, n_chan)["W_white"]
        if w_final.shape == base_unmix.shape:
            # The last checkpoint and the fitted object describe the same
            # state, so after the comp_stds correction they must agree. This
            # is the check that the correction is right, not merely plausible.
            w_corrected = w_final / comp_stds
            diag["final_snapshot_vs_fitted"] = {
                "max_abs_difference_uncorrected": float(np.max(np.abs(w_final - base_unmix))),
                "max_abs_difference_corrected": float(np.max(np.abs(w_corrected - base_unmix))),
                "relative_frobenius_corrected": float(
                    np.linalg.norm(w_corrected - base_unmix) / np.linalg.norm(base_unmix)),
                "log2_abs_det_snapshot_raw": float(np.linalg.slogdet(w_final)[1] / np.log(2)),
                "log2_abs_det_snapshot_corrected": float(
                    np.linalg.slogdet(w_corrected)[1] / np.log(2)),
                "log2_abs_det_fitted": float(np.linalg.slogdet(base_unmix)[1] / np.log(2)),
            }
        else:
            diag["final_snapshot_vs_fitted"] = {
                "shape_mismatch": [list(w_final.shape), list(base_unmix.shape)]}
        print(f"[trajectory] diagnostic: {json.dumps(diag, default=float)}")

    rows = []
    prev_A = None
    final_A = None
    for d in snaps:
        n_iter = int(d.name.split("_")[1])
        snap = read_snapshot(d, args.n_components, n_chan)
        # Swap the snapshot's unmixing into the fitted ICA. The sphering and
        # PCA basis are estimated once before the EM loop and do not change,
        # so everything else in the object stays valid -- but the checkpoint
        # is NOT in the same domain as ica.unmixing_matrix_. AMICA is fed
        # pca_data / comp_stds, so the in-loop matrix acts on the normalised
        # input, while the fitted object's matrix acts on raw pca_data.
        # fit_ica reconciles them at mne_integration.py:318 by dividing each
        # column by comp_stds, and get_model_ica does the same; skipping it
        # leaves log2|det W| short by sum(log2 comp_stds) -- about 50 bits per
        # sample here, which is what drove complete MIR negative.
        ica.unmixing_matrix_ = snap["W_white"] / comp_stds
        ica.mixing_matrix_ = np.linalg.pinv(ica.unmixing_matrix_)
        art = compute_v3_artifacts(ica, raw)

        A_sensor = snap["A_sensor"]
        row = {
            "iteration": n_iter,
            "log_likelihood": float(snap["LL"][-1]) if snap["LL"].size else None,
            "complete_mir_kbits_s": (art.get("complete_mir") or {}).get("kbits_per_sec"),
            "remnant_pmi_percent": (art.get("pmi") or {}).get("remnant_PMI_percent"),
            "nd_5_percent": (art.get("dipolarity") or {}).get("nd_5_percent"),
            "nd_10_percent": (art.get("dipolarity") or {}).get("nd_10_percent"),
            "map_corr_vs_previous_checkpoint": (
                matched_map_corr(prev_A, A_sensor) if prev_A is not None else None
            ),
        }
        rows.append(row)
        prev_A = A_sensor
        final_A = A_sensor
        print(f"  iter {n_iter:6d}  LL {row['log_likelihood']}  "
              f"MIR {row['complete_mir_kbits_s']}  "
              f"ND10 {row['nd_10_percent']}  "
              f"dmap {row['map_corr_vs_previous_checkpoint']}")

    # distance of every checkpoint from the final solution
    if final_A is not None:
        for row, d in zip(rows, snaps):
            A_sensor = read_snapshot(d, args.n_components, n_chan)["A_sensor"]
            row["map_corr_vs_final"] = matched_map_corr(A_sensor, final_A)

    ica.unmixing_matrix_ = base_unmix
    ica.mixing_matrix_ = base_mix

    out = {
        "_meta": {
            "analysis": "YOR19 single-trajectory iteration audit (exploratory)",
            "dataset": args.dataset,
            "subject": args.subject,
            "n_components": args.n_components,
            "max_iter": args.max_iter,
            "writestep": args.writestep,
            "random_state": args.random_state,
            "num_mix": args.num_mix,
            "fit_seconds": fit_s,
            "n_snapshots": len(snaps),
            "single_continuing_trajectory": True,
            "diagnostic": diag,
            "note": (
                "Checkpoints come from one uninterrupted fit; the solver was "
                "never restarted, so the learning-rate ramp and Newton state "
                "carry across checkpoints."
            ),
        },
        "full_log_likelihood": ll_full.tolist(),
        "checkpoints": rows,
    }
    dest = args.out_dir / f"trajectory_{args.dataset}_sub-{args.subject:02d}.json"
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(f"[trajectory] wrote {dest}")

    if not args.keep_snapshots:
        shutil.rmtree(snap_root, ignore_errors=True)
        print("[trajectory] removed raw snapshot binaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
