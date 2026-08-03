"""YOR19 on the controlled fixture: one continuing trajectory vs planted maps.

``run_iteration_trajectory.py`` answers YOR19 on real EEG, where there is no
ground truth and the outcomes are MIR, remnant PMI and near-dipolarity. The
numbers YOR19 actually disputes -- median matched sensor-map correlation of
0.842 after 3,000 iterations against 0.987 after 10,000 -- come from this
fixture instead, the controlled clean known-topography audit behind main
Figure 2. This script scores that fixture the same way, but from ONE
uninterrupted fit rather than two.

The two published configurations are matched in every input: same simulated
recording, same external whitener, same initial unmixing matrix, same seed,
both stopped on the iteration cap without converging. They differ only in
``max_iter``, and ``max_iter`` reaches the solver in exactly two places -- the
loop bound and a log message. Nothing in the annealing, learning-rate or
Newton schedule reads it. The published 3,000-iteration fit should therefore
already BE the state at iteration 3,000 of the 10,000-iteration fit, which
turns the two published medians into a pass/fail gate on this run rather than
a loose comparison.

That gate is the point. If checkpoint 3,000 and checkpoint 10,000 reproduce
the published medians, every checkpoint between and before them is trustworthy
on the same footing, and the trajectory shows where the maps stop improving
relative to where the likelihood stops rising. If they do not reproduce, the
solver has ``max_iter``-dependent behaviour that the source does not show, and
that is the finding -- it is recorded, not worked around.

This is exploratory. A trajectory whose maps keep climbing long after the
likelihood flattens supports YOR19; one that flattens with the likelihood
narrows it. Both are reportable.

Runs anywhere with the MNE sample dataset: no cluster data is involved. It
does fit AMICA, so it must not be run on a login node.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
CAPSULE_ROOT = THIS_DIR.parent.parent
CC_BENCHMARK = CAPSULE_ROOT / "benchmark" / "cc_benchmark"
for _p in (str(CAPSULE_ROOT), str(THIS_DIR), str(CC_BENCHMARK)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from figure2_topography_analysis import match_topographies  # noqa: E402
from generate_synthetic_raw import generate, load_config  # noqa: E402
from run_figure2_topography import (  # noqa: E402
    DATA_SEED,
    FIT_SEED,
    N_COMPONENTS,
    _sensor_matrices,
    _sha256_array,
    common_initial_weights,
    common_whitening,
)
from run_iteration_trajectory import (  # noqa: E402
    install_numbered_checkpoints,
    read_snapshot,
)

# From figure2_topography_manifest.json (shared_input). Kept as recorded
# diagnostics, NOT as a gate: sha256 over float64 bytes cannot absorb the
# eigenvector sign conventions and ~1e-14 arithmetic drift that a different
# BLAS produces, and it flags those as loudly as a genuinely different fixture.
# The gate below compares the fixture numerically instead.
PUBLISHED_INPUT_HASHES = {
    "input_data_hash": "87c5f05dd994600bb4c746834cecef7e3f1c3c5591251f88b0b777fc1510c2f6",
    "whitened_data_hash": "988ee9dbf8107e60620c55ed795d17dee5ba6f6baa6b02821be524dc0ea1966d",
    "whitener_hash": "b7b886405a341635f5773e98a162e43d32aac5a29c419a2100583c4e2e649196",
    "initial_weights_hash": "42abbbf82c854ee72973f4c99c25020149aefbce5eb2273a3c64e061fe747e5d",
}

# The archived fit outputs. A_true, whitener and dewhitener are taken from here
# rather than from the rebuild so the trajectory runs in the published sign
# convention and stays directly comparable to the published medians.
ARCHIVE_NPZ = (
    CAPSULE_ROOT / "results" / "figure2_topography" / "figure2_topography_fit_outputs.npz"
)

# topography_recovery_summary.csv, median_abs_r column.
PUBLISHED_MEDIAN_ABS_R = {3000: 0.8421372304520969, 10000: 0.9867382962489333}


def score_maps(a_true: np.ndarray, a_sensor: np.ndarray) -> dict:
    """Median / IQR of Hungarian-matched |r| against the planted topographies.

    Same estimator as the published summary: ``match_topographies`` for the
    assignment, then the quantiles ``_summary_rows`` reports.
    """
    matched = match_topographies(a_true, a_sensor)
    abs_r = np.asarray(matched["abs_r"], dtype=float)
    q1, median, q3 = (float(v) for v in np.quantile(abs_r, [0.25, 0.5, 0.75]))
    return {
        "median_abs_r": median,
        "q1_abs_r": q1,
        "q3_abs_r": q3,
        "minimum_abs_r": float(abs_r.min()),
        "maximum_abs_r": float(abs_r.max()),
        "mean_abs_r": float(abs_r.mean()),
        "fraction_abs_r_ge_0_90": float(np.mean(abs_r >= 0.90)),
        "fraction_abs_r_ge_0_95": float(np.mean(abs_r >= 0.95)),
    }


def build_fixture(config_path: Path, cache_dir: Path, *, force: bool,
                  archive_path: Path = ARCHIVE_NPZ) -> dict:
    """Rebuild the Figure 2 fixture and prove it is the published one.

    The simulation is regenerated (the whitened data is far too large to have
    been archived), but ``A_true``, the whitener and the dewhitener are taken
    from the archive. Two reasons. The planted maps are what every median is
    scored against, so they should come from the published artefact rather than
    a rebuild. And ``scipy.linalg.eigh`` picks eigenvector signs per BLAS
    build, so a rebuilt whitener flips signs of the whitened data and sends the
    optimiser down a different path for no scientific reason.

    ``initial_weights`` is NOT taken from the archive. The archived array is not
    orthogonal, so it cannot be the output of ``common_initial_weights``; it was
    overwritten in place after the fits ran. It is regenerated here instead.
    """
    if not archive_path.exists():
        raise SystemExit(f"archived fit outputs not found at {archive_path}")
    archive = np.load(archive_path)

    config = load_config(config_path)
    bundle = generate(
        config,
        condition_id="clean",
        seed=DATA_SEED,
        cache_dir=cache_dir,
        force=force,
        verbose=True,
    )
    raw = bundle["raw"]
    x_sensor = np.asarray(raw.get_data(), dtype=np.float64)
    a_true_rebuilt = np.asarray(bundle["A_true"], dtype=np.float64)
    if x_sensor.shape != (59, 300000):
        raise AssertionError(f"unexpected sensor matrix shape {x_sensor.shape}")
    if a_true_rebuilt.shape != (59, N_COMPONENTS):
        raise AssertionError(f"unexpected planted mixing shape {a_true_rebuilt.shape}")

    a_true = np.asarray(archive["A_true"], dtype=np.float64)
    whitener = np.asarray(archive["whitener"], dtype=np.float64)
    dewhitener = np.asarray(archive["dewhitener"], dtype=np.float64)

    # Numerical fixture gate. The planted maps must be bit-identical: they are
    # the scoring reference, and any drift there makes every median a different
    # measurement rather than a reproduction.
    a_true_delta = float(np.abs(a_true_rebuilt - a_true).max())
    if not np.array_equal(a_true_rebuilt, a_true):
        raise SystemExit(
            "The regenerated planted mixing matrix differs from the archived "
            f"one (max |delta| = {a_true_delta:.3e}). Scoring against a "
            "different ground truth would not reproduce anything, so this run "
            "is aborted rather than reported."
        )

    mean = x_sensor.mean(axis=1)
    x_white = whitener @ (x_sensor - mean[:, None])
    whitened_covariance_error = float(
        np.linalg.norm(
            x_white @ x_white.T / x_white.shape[1] - np.eye(N_COMPONENTS), ord="fro"
        )
    )
    if whitened_covariance_error > 1e-8 * N_COMPONENTS:
        raise SystemExit(
            "The archived whitener does not whiten the regenerated recording "
            f"(Frobenius error {whitened_covariance_error:.3e}); the fixture "
            "and the archive are not the same experiment."
        )

    w0 = common_initial_weights(N_COMPONENTS, FIT_SEED)
    orthogonality_error = float(
        np.abs(w0 @ w0.T - np.eye(N_COMPONENTS)).max()
    )
    if orthogonality_error > 1e-10:
        raise SystemExit(
            f"initial weights are not orthogonal (max deviation "
            f"{orthogonality_error:.3e})"
        )

    observed_hashes = {
        "input_data_hash": _sha256_array(x_sensor),
        "whitened_data_hash": _sha256_array(x_white),
        "whitener_hash": _sha256_array(whitener),
        "initial_weights_hash": _sha256_array(w0),
    }
    hash_report = {
        k: {
            "published": v,
            "observed": observed_hashes[k],
            "identical": observed_hashes[k] == v,
        }
        for k, v in PUBLISHED_INPUT_HASHES.items()
    }
    for k, pair in hash_report.items():
        state = "identical" if pair["identical"] else "differs (see numerical gate)"
        print(f"[fixture] {k}: {state}")
    print(f"[fixture] planted maps bit-identical to the archive; "
          f"whitened covariance error {whitened_covariance_error:.3e}; "
          f"init orthogonality {orthogonality_error:.3e}")

    return {
        "a_true": a_true,
        "x_white": x_white,
        "whitener": whitener,
        "dewhitener": dewhitener,
        "mean": mean,
        "w0": w0,
        "hash_report": hash_report,
        "a_true_max_abs_delta": a_true_delta,
        "whitened_covariance_frobenius_error": whitened_covariance_error,
        "initial_weights_orthogonality_error": orthogonality_error,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=THIS_DIR / "configs" / "benchmark_v1.json")
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument("--writestep", type=int, default=250,
                   help="snapshot interval, in solver iterations")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="fixture cache; defaults to <out-dir>/cache")
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--archive", type=Path, default=ARCHIVE_NPZ,
                   help="figure2_topography_fit_outputs.npz; on the cluster it "
                        "lives under the job results dir, not in the repo")
    p.add_argument("--reproduction-tolerance", type=float, default=0.02,
                   help="absolute median |r| tolerance when comparing "
                        "checkpoints 3,000 and 10,000 to the published medians")
    p.add_argument("--keep-snapshots", action="store_true",
                   help="retain the raw checkpoint binaries")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / "cache")
    snap_root = args.out_dir / "snapshots"
    snap_root.mkdir(exist_ok=True)

    from amica_python import Amica, AmicaConfig
    import amica_python as amica_pkg

    print(f"[trajectory] package={getattr(amica_pkg, '__file__', '?')}")
    print(f"[trajectory] max_iter={args.max_iter} writestep={args.writestep}")

    fixture = build_fixture(args.config, cache_dir, force=args.force_regenerate,
                            archive_path=args.archive)
    a_true = fixture["a_true"]
    x_white = fixture["x_white"]
    whitener = fixture["whitener"]
    dewhitener = fixture["dewhitener"]

    # Identical to _run_amica in run_figure2_topography.py, plus the checkpoint
    # fields. do_sphere=False and pcakeep=None matter: the whitener is external,
    # so the solver sees already-whitened data and the saved W needs no rescale
    # before _sensor_matrices. That is the opposite of the fit_ica path, which
    # normalises each PCA component to unit variance and must be undone.
    config = AmicaConfig(
        max_iter=args.max_iter,
        num_models=1,
        num_mix_comps=3,
        dtype="float64",
        do_mean=False,
        do_sphere=False,
        pcakeep=None,
        do_reject=False,
        writestep=args.writestep,
        outdir=snap_root,
    )
    written = install_numbered_checkpoints(snap_root)

    solver = Amica(config, random_state=FIT_SEED)
    started = time.perf_counter()
    result = solver.fit(x_white, init_weights=fixture["w0"])
    fit_seconds = time.perf_counter() - started
    print(f"[trajectory] fit done in {fit_seconds:.1f}s over "
          f"{int(result.n_iter)} iterations; {len(written)} snapshots")

    w_final = np.asarray(result.unmixing_matrix_white_, dtype=np.float64)
    a_final_sensor, _, _ = _sensor_matrices(w_final, whitener, dewhitener, x_white)
    fitted_score = score_maps(a_true, a_final_sensor)
    _, logdet_fitted = np.linalg.slogdet(w_final)
    logdet_fitted = float(logdet_fitted / np.log(2.0))

    ll_full = np.asarray(result.log_likelihood, dtype=float)

    rows = []
    snaps = sorted({d for d in written if d.exists()}, key=lambda d: d.name)
    for d in snaps:
        snap = read_snapshot(d, N_COMPONENTS, a_true.shape[0])
        w_white = np.asarray(snap["W_white"], dtype=np.float64)
        a_sensor, _, residual = _sensor_matrices(
            w_white, whitener, dewhitener, x_white
        )
        ll = np.asarray(snap["LL"], dtype=float)
        n_iter = int(ll.size)
        _, logdet = np.linalg.slogdet(w_white)
        row = {
            # The solver checkpoints at (iteration + 1) % writestep == 0 and
            # stores n_iter = len(LL), so these two agree here. The ds004505
            # run recorded an off-by-one between them; recorded rather than
            # assumed away.
            "iteration": n_iter,
            "iteration_dir": d.name,
            "log_likelihood": float(ll[-1]) if ll.size else float("nan"),
            "log2_abs_det_w_white": float(logdet / np.log(2.0)),
            "sensor_reconstruction_relative_residual": residual,
        }
        row.update(score_maps(a_true, a_sensor))
        # Solution stability, the same quantity the real-EEG run reports, so the
        # two trajectories can be read side by side.
        row["map_corr_vs_final"] = float(
            np.median(np.abs(match_topographies(a_final_sensor, a_sensor)["abs_r"]))
        )
        rows.append(row)
        print(f"  iter {row['iteration']:>6}  LL {row['log_likelihood']:.6f}  "
              f"median|r| {row['median_abs_r']:.6f}  "
              f"stability {row['map_corr_vs_final']:.6f}")

    by_iter = {r["iteration"]: r for r in rows}
    gates = []
    for iteration, published in PUBLISHED_MEDIAN_ABS_R.items():
        row = by_iter.get(iteration)
        if row is None:
            gates.append({
                "gate": f"checkpoint_{iteration}_matches_published",
                "status": "not_evaluated",
                "reason": f"no checkpoint at iteration {iteration} "
                          f"(writestep={args.writestep}, max_iter={args.max_iter})",
                "published_median_abs_r": published,
            })
            continue
        delta = row["median_abs_r"] - published
        # Not bit-exactness. The published run's initial weights survive only as
        # a hash, and that hash does not describe the array the archive stored,
        # so the init used here cannot be byte-verified against it. What the
        # comparison can establish is whether the same iteration budget lands in
        # the same place, which is the claim YOR19 disputes.
        if abs(delta) < 1e-9:
            status = "exact"
        elif abs(delta) < args.reproduction_tolerance:
            status = "pass"
        else:
            status = "fail"
        gates.append({
            "gate": f"checkpoint_{iteration}_matches_published",
            "status": status,
            "published_median_abs_r": published,
            "observed_median_abs_r": row["median_abs_r"],
            "absolute_difference": abs(delta),
            "tolerance": args.reproduction_tolerance,
        })

    if rows:
        final_delta = rows[-1]["median_abs_r"] - fitted_score["median_abs_r"]
        gates.append({
            "gate": "final_checkpoint_equals_fitted_object",
            "status": "pass" if abs(final_delta) < 1e-12 else "fail",
            "checkpoint_median_abs_r": rows[-1]["median_abs_r"],
            "fitted_median_abs_r": fitted_score["median_abs_r"],
            "absolute_difference": abs(final_delta),
            "log2_abs_det_final_checkpoint": rows[-1]["log2_abs_det_w_white"],
            "log2_abs_det_fitted": logdet_fitted,
            "note": "A per-row scale mismatch here is the fit_ica unit-variance "
                    "rescale leaking in; this fixture whitens externally and "
                    "must not need it.",
        })

    for g in gates:
        print(f"[gate] {g['gate']}: {g['status'].upper()}")

    payload = {
        "_meta": {
            "analysis": "YOR19 continuing trajectory on the Figure 2 "
                        "known-topography fixture (exploratory)",
            "fixture": "mne.datasets.sample forward, condition=clean",
            "data_seed": DATA_SEED,
            "fit_seed": FIT_SEED,
            "n_components": N_COMPONENTS,
            "max_iter": args.max_iter,
            "writestep": args.writestep,
            "actual_n_iter": int(result.n_iter),
            "converged_flag": bool(result.converged),
            "hit_iteration_cap": bool(
                int(result.n_iter) >= args.max_iter and not result.converged
            ),
            "fit_seconds": fit_seconds,
            "n_snapshots": len(rows),
            "single_continuing_trajectory": True,
            "fixture_provenance": {
                "planted_maps_source": "archived figure2_topography_fit_outputs.npz",
                "whitener_source": "archived figure2_topography_fit_outputs.npz",
                "initial_weights_source": "regenerated via common_initial_weights"
                                          "(32, 42); the archived array is not "
                                          "orthogonal and was overwritten in place",
                "planted_maps_max_abs_delta_vs_rebuild": fixture["a_true_max_abs_delta"],
                "whitened_covariance_frobenius_error":
                    fixture["whitened_covariance_frobenius_error"],
                "initial_weights_orthogonality_error":
                    fixture["initial_weights_orthogonality_error"],
                "byte_hashes": fixture["hash_report"],
            },
            "fitted_object_score": fitted_score,
            "log2_abs_det_fitted": logdet_fitted,
            "note": "Checkpoints come from one uninterrupted fit; the solver was "
                    "never restarted, so the learning-rate ramp and Newton state "
                    "carry across checkpoints.",
        },
        "acceptance_gates": gates,
        "full_log_likelihood": ll_full.tolist(),
        "checkpoints": rows,
    }

    out_json = args.out_dir / (
        f"trajectory_topography_w{args.writestep}_i{args.max_iter}.json"
    )
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[trajectory] wrote {out_json}")

    if not args.keep_snapshots:
        import shutil
        shutil.rmtree(snap_root, ignore_errors=True)
        print("[trajectory] snapshots discarded (--keep-snapshots to retain)")

    return 0 if all(g["status"] != "fail" for g in gates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
