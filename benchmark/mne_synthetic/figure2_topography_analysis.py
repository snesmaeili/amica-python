"""Analysis utilities for the controlled Figure 2 topography audit.

The fitting job writes one compact NPZ archive plus a JSON manifest.  This
module performs the sensor-space Hungarian matching, exports the provenance
tables used by the manuscript, and selects the three common difficult planted
maps with the prespecified minimax rule.

No model fitting is performed here.  Figure generation and tests can therefore
be run from the archived outputs alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PRIMARY_METHODS = (
    "amica_3000",
    "amica_10000",
    "picard",
    "extended_infomax",
    "fastica",
)

STRICT_PAIRS = (
    ("amica_3000", "amica_10000"),
    ("picard", "picard_strict"),
    ("extended_infomax", "extended_infomax_strict"),
    ("fastica", "fastica_strict"),
)


def _as_finite_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def centre_normalise_topographies(matrix: np.ndarray) -> np.ndarray:
    """Centre and L2-normalise each sensor-space topography column."""
    matrix = _as_finite_matrix(matrix, name="topography matrix")
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centred, axis=0, keepdims=True)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("topography matrix contains a constant or zero column")
    return centred / norms


def match_topographies(a_true: np.ndarray, a_est: np.ndarray) -> dict[str, np.ndarray]:
    """Hungarian-match estimated maps to planted maps using absolute correlation.

    Both inputs must be in the same original sensor channel order.  The returned
    aligned estimated matrix has one column per planted source and uses the same
    assignment later shown in both Figure 2C and Figure 2D.
    """
    a_true = _as_finite_matrix(a_true, name="A_true")
    a_est = _as_finite_matrix(a_est, name="A_est")
    if a_true.shape[0] != a_est.shape[0]:
        raise ValueError(
            "A_true and A_est must have the same sensor dimension; "
            f"got {a_true.shape[0]} and {a_est.shape[0]}"
        )
    if a_true.shape[1] != a_est.shape[1]:
        raise ValueError(
            "The controlled audit requires equal planted and recovered ranks; "
            f"got {a_true.shape[1]} and {a_est.shape[1]}"
        )

    true_z = centre_normalise_topographies(a_true)
    est_z = centre_normalise_topographies(a_est)
    correlations = true_z.T @ est_z
    row_ind, col_ind = linear_sum_assignment(-np.abs(correlations))
    order = np.argsort(row_ind)
    row_ind = row_ind[order]
    col_ind = col_ind[order]
    expected = np.arange(a_true.shape[1])
    if not np.array_equal(row_ind, expected):
        raise AssertionError("Hungarian assignment did not cover every planted source")

    signed_r = correlations[row_ind, col_ind]
    signs = np.where(signed_r < 0.0, -1.0, 1.0)
    aligned = a_est[:, col_ind] * signs[np.newaxis, :]
    abs_r = np.abs(signed_r)
    if not np.all(np.isfinite(abs_r)) or np.any((abs_r < 0.0) | (abs_r > 1.0 + 1e-12)):
        raise AssertionError("matched correlations are not finite values in [0, 1]")

    return {
        "planted_source_index": row_ind.astype(int),
        "estimated_component_index": col_ind.astype(int),
        "signed_r": signed_r.astype(float),
        "abs_r": np.clip(abs_r, 0.0, 1.0).astype(float),
        "sign_flip": (signs < 0).astype(bool),
        "aligned_estimated": aligned,
        "correlation_matrix": correlations,
    }


def select_minimax_sources(
    matched: pd.DataFrame,
    *,
    methods: Iterable[str] = PRIMARY_METHODS,
    n_select: int = 3,
) -> pd.DataFrame:
    """Select common difficult sources by ``argsort(max_method_r)[:n]``."""
    methods = tuple(methods)
    subset = matched[matched["method"].isin(methods)].copy()
    counts = subset.groupby("planted_source_index")["method"].nunique()
    if len(counts) == 0 or not np.all(counts.to_numpy() == len(methods)):
        raise ValueError("each planted source must have one match from every primary method")

    wide = subset.pivot(
        index="planted_source_index", columns="method", values="abs_r"
    ).loc[:, list(methods)]
    q = wide.max(axis=1)
    selected_indices = q.sort_values(kind="mergesort").index[:n_select]
    rows = []
    for rank, source_index in enumerate(selected_indices, start=1):
        source_values = wide.loc[source_index]
        best_method = str(source_values.idxmax())
        first = subset[subset.planted_source_index == source_index].iloc[0]
        rows.append(
            {
                "rank": rank,
                "planted_source_index": int(source_index),
                "q_best_method": float(q.loc[source_index]),
                "best_method": best_method,
                "amica_3000_r": float(source_values["amica_3000"]),
                "amica_10000_r": float(source_values["amica_10000"]),
                "picard_r": float(source_values["picard"]),
                "extended_infomax_r": float(source_values["extended_infomax"]),
                "fastica_r": float(source_values["fastica"]),
                "source_vertex": first.get("source_vertex", ""),
                "source_hemi": first.get("source_hemi", ""),
                "source_label": first.get("source_label", ""),
            }
        )
    return pd.DataFrame(rows)


def _configuration_rows(manifest: dict) -> pd.DataFrame:
    rows = []
    for method in PRIMARY_METHODS:
        record = manifest["fits"][method]
        config = record["configuration"]
        outcome = record["outcome"]
        rows.append(
            {
                "method": method,
                "display_name": record["display_name"],
                "software_package": record["software_package"],
                "software_version": record["software_version"],
                "git_commit": manifest["provenance"].get("git_commit", "not archived"),
                "random_seed": config["random_seed"],
                "initialization_id": manifest["shared_input"]["initialization_id"],
                "n_components": manifest["shared_input"]["n_components"],
                "whitening_strategy": manifest["shared_input"]["whitening_strategy"],
                "internal_whitening_enabled": config["internal_whitening_enabled"],
                "max_iter": config["max_iter"],
                "actual_n_iter": outcome["actual_n_iter"],
                "stopping_parameter_name": config["stopping_parameter_name"],
                "stopping_parameter_value": config["stopping_parameter_value"],
                "secondary_stopping_parameters": json.dumps(
                    config.get("secondary_stopping_parameters", {}), sort_keys=True
                ),
                "stopping_reason": outcome["stopping_reason"],
                "hit_iteration_cap": outcome["hit_iteration_cap"],
                "converged_flag": outcome["converged_flag"],
                "runtime_seconds": outcome["runtime_seconds"],
                "input_data_hash": manifest["shared_input"]["input_data_hash"],
                "whitener_hash": manifest["shared_input"]["whitener_hash"],
                "initial_weights_hash": manifest["shared_input"]["initial_weights_hash"],
                "result_file": record["result_file"],
                "notes": record.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def _summary_rows(matched: pd.DataFrame, configurations: pd.DataFrame) -> pd.DataFrame:
    config_lookup = configurations.set_index("method")
    rows = []
    for method in PRIMARY_METHODS:
        values = matched.loc[matched.method == method, "abs_r"].to_numpy(float)
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        rows.append(
            {
                "method": method,
                "display_name": config_lookup.loc[method, "display_name"],
                "n_planted_maps": int(values.size),
                "median_abs_r": float(median),
                "q1_abs_r": float(q1),
                "q3_abs_r": float(q3),
                "iqr_abs_r": float(q3 - q1),
                "minimum_abs_r": float(values.min()),
                "maximum_abs_r": float(values.max()),
                "mean_abs_r": float(values.mean()),
                "n_abs_r_ge_0_90": int(np.sum(values >= 0.90)),
                "fraction_abs_r_ge_0_90": float(np.mean(values >= 0.90)),
                "n_abs_r_ge_0_95": int(np.sum(values >= 0.95)),
                "fraction_abs_r_ge_0_95": float(np.mean(values >= 0.95)),
                "actual_n_iter": int(config_lookup.loc[method, "actual_n_iter"]),
                "stopping_reason": config_lookup.loc[method, "stopping_reason"],
            }
        )
    return pd.DataFrame(rows)


def _sensitivity_rows(
    arrays: np.lib.npyio.NpzFile,
    matched: pd.DataFrame,
    manifest: dict,
) -> pd.DataFrame:
    rows = []
    ground_truth_medians = matched.groupby("method")["abs_r"].median()
    ground_truth_minima = matched.groupby("method")["abs_r"].min()
    for primary, strict in STRICT_PAIRS:
        stability = match_topographies(arrays[f"A_est_{primary}"], arrays[f"A_est_{strict}"])
        primary_outcome = manifest["fits"][primary]["outcome"]
        strict_outcome = manifest["fits"][strict]["outcome"]
        strict_gt = match_topographies(arrays["A_true"], arrays[f"A_est_{strict}"])["abs_r"]
        rows.append(
            {
                "primary_method": primary,
                "strict_method": strict,
                "primary_stopping_rule": manifest["fits"][primary]["configuration"]["stopping_parameter_name"],
                "strict_stopping_rule": manifest["fits"][strict]["configuration"]["stopping_parameter_name"],
                "primary_stopping_value": manifest["fits"][primary]["configuration"]["stopping_parameter_value"],
                "strict_stopping_value": manifest["fits"][strict]["configuration"]["stopping_parameter_value"],
                "primary_actual_n_iter": primary_outcome["actual_n_iter"],
                "strict_actual_n_iter": strict_outcome["actual_n_iter"],
                "primary_hit_iteration_cap": primary_outcome["hit_iteration_cap"],
                "strict_hit_iteration_cap": strict_outcome["hit_iteration_cap"],
                "median_solution_stability_abs_r": float(np.median(stability["abs_r"])),
                "minimum_solution_stability_abs_r": float(np.min(stability["abs_r"])),
                "primary_median_ground_truth_abs_r": float(ground_truth_medians[primary]),
                "strict_median_ground_truth_abs_r": float(np.median(strict_gt)),
                "change_median_ground_truth_abs_r": float(np.median(strict_gt) - ground_truth_medians[primary]),
                "primary_minimum_ground_truth_abs_r": float(ground_truth_minima[primary]),
                "strict_minimum_ground_truth_abs_r": float(np.min(strict_gt)),
                "change_minimum_ground_truth_abs_r": float(np.min(strict_gt) - ground_truth_minima[primary]),
                "primary_stopping_reason": primary_outcome["stopping_reason"],
                "strict_stopping_reason": strict_outcome["stopping_reason"],
            }
        )
    return pd.DataFrame(rows)


def analyse_archive(archive: Path, manifest_path: Path, output_dir: Path) -> dict:
    """Create all Figure 2 audit CSV/NPZ files from one archived fit bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = np.load(archive, allow_pickle=False)
    a_true = _as_finite_matrix(arrays["A_true"], name="A_true")
    ch_names = arrays["ch_names"].astype(str).tolist()
    expected_channels = int(manifest["simulation"]["n_channels"])
    expected_sources = int(manifest["simulation"]["n_true_sources"])
    if a_true.shape != (expected_channels, expected_sources):
        raise ValueError(
            "A_true is not in the archived original sensor space; expected "
            f"{(expected_channels, expected_sources)}, got {a_true.shape}"
        )
    if len(ch_names) != a_true.shape[0] or len(set(ch_names)) != len(ch_names):
        raise ValueError("archived channel names do not uniquely match A_true rows")

    vertex_records = manifest["simulation"]["vertex_records"]
    if len(vertex_records) != a_true.shape[1]:
        raise ValueError("vertex provenance does not match the planted source count")

    matched_rows = []
    aligned_payload = {
        "A_true": a_true,
        "sensor_positions": arrays["sensor_positions"],
        "ch_names": arrays["ch_names"],
    }
    for method in PRIMARY_METHODS:
        a_est = _as_finite_matrix(arrays[f"A_est_{method}"], name=f"A_est_{method}")
        result = match_topographies(a_true, a_est)
        aligned_payload[f"A_est_aligned_{method}"] = result["aligned_estimated"]
        for source_index, component_index, signed_r, abs_r, sign_flip in zip(
            result["planted_source_index"],
            result["estimated_component_index"],
            result["signed_r"],
            result["abs_r"],
            result["sign_flip"],
        ):
            vertex = vertex_records[int(source_index)]
            matched_rows.append(
                {
                    "method": method,
                    "display_name": manifest["fits"][method]["display_name"],
                    "planted_source_index": int(source_index),
                    "estimated_component_index": int(component_index),
                    "signed_r": float(signed_r),
                    "abs_r": float(abs_r),
                    "sign_flip": bool(sign_flip),
                    "source_vertex": vertex["vertex_id"],
                    "source_hemi": vertex["hemi"],
                    "source_label": vertex["label_name"],
                }
            )

    matched = pd.DataFrame(matched_rows)
    selected = select_minimax_sources(matched)
    score_lookup = selected.set_index("planted_source_index")["q_best_method"]
    selected_set = set(selected.planted_source_index.astype(int))
    matched["selection_score"] = matched.planted_source_index.map(score_lookup)
    matched["selected_for_panel_d"] = matched.planted_source_index.isin(selected_set)

    configurations = _configuration_rows(manifest)
    summary = _summary_rows(matched, configurations)
    sensitivity = _sensitivity_rows(arrays, matched, manifest)

    configurations.to_csv(output_dir / "method_configuration.csv", index=False)
    matched.to_csv(output_dir / "matched_topographies.csv", index=False)
    summary.to_csv(output_dir / "topography_recovery_summary.csv", index=False)
    selected.to_csv(output_dir / "panel_d_selected_sources.csv", index=False)
    sensitivity.to_csv(output_dir / "convergence_sensitivity.csv", index=False)
    np.savez_compressed(output_dir / "figure2_topography_maps.npz", **aligned_payload)

    audit = {
        "archive": str(archive.resolve()),
        "manifest": str(manifest_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "n_channels": int(a_true.shape[0]),
        "n_sources": int(a_true.shape[1]),
        "methods": list(PRIMARY_METHODS),
        "selected_sources": selected.planted_source_index.astype(int).tolist(),
    }
    (output_dir / "analysis_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = analyse_archive(args.archive, args.manifest, args.output_dir)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
