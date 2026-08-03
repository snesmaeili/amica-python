from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from figure2_topography_analysis import (  # noqa: E402
    PRIMARY_METHODS,
    analyse_archive,
    match_topographies,
    select_minimax_sources,
)
from run_figure2_topography import common_whitening  # noqa: E402


def _maps(seed: int = 7, n_channels: int = 8, n_sources: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_channels, n_sources))
    matrix -= matrix.mean(axis=0, keepdims=True)
    return matrix


def test_common_whitening_uses_scale_aware_rank_for_eeg_volts():
    rng = np.random.default_rng(101)
    sources = 1e-6 * rng.standard_normal((4, 4000))
    mixing = rng.standard_normal((7, 4))
    sensor_data = mixing @ sources
    _, whitener, dewhitener, whitened, eigenvalues = common_whitening(
        sensor_data, 4
    )
    assert whitener.shape == (4, 7)
    assert dewhitener.shape == (7, 4)
    assert np.all(eigenvalues > 0.0)
    assert np.allclose(
        whitened @ whitened.T / whitened.shape[1],
        np.eye(4),
        atol=1e-9,
        rtol=1e-9,
    )


def test_common_whitening_rejects_requested_rank_above_latent_rank():
    rng = np.random.default_rng(102)
    sources = 1e-6 * rng.standard_normal((3, 4000))
    sensor_data = rng.standard_normal((7, 3)) @ sources
    with pytest.raises(ValueError, match="scale-aware numerical data rank"):
        common_whitening(sensor_data, 4)


def test_matching_recovers_known_permutation_and_signs():
    a_true = _maps()
    permutation = np.array([2, 0, 3, 1])
    signs = np.array([-1.0, 1.0, -1.0, 1.0])
    a_est = a_true[:, permutation] * signs
    result = match_topographies(a_true, a_est)
    assert np.allclose(result["abs_r"], 1.0)
    assert np.allclose(result["aligned_estimated"], a_true)
    inverse = np.argsort(permutation)
    assert np.array_equal(result["estimated_component_index"], inverse)


def test_absolute_matching_is_invariant_to_component_permutation_and_sign():
    a_true = _maps()
    base = match_topographies(a_true, a_true)
    changed = match_topographies(a_true, -a_true[:, [3, 1, 0, 2]])
    assert np.allclose(base["abs_r"], changed["abs_r"])


def test_correlations_are_finite_and_bounded():
    result = match_topographies(_maps(), _maps(seed=11))
    assert np.all(np.isfinite(result["abs_r"]))
    assert np.all((result["abs_r"] >= 0.0) & (result["abs_r"] <= 1.0))


def test_minimax_selection_matches_prespecified_formula_and_is_reproducible():
    rows = []
    for source in range(5):
        for method_index, method in enumerate(PRIMARY_METHODS):
            rows.append(
                {
                    "method": method,
                    "planted_source_index": source,
                    "abs_r": 0.50 + 0.05 * source + 0.01 * method_index,
                    "source_vertex": source,
                    "source_hemi": "lh",
                    "source_label": f"source-{source}",
                }
            )
    matched = pd.DataFrame(rows)
    first = select_minimax_sources(matched)
    second = select_minimax_sources(matched.sample(frac=1.0, random_state=3))
    expected = (
        matched.pivot(index="planted_source_index", columns="method", values="abs_r")
        .loc[:, list(PRIMARY_METHODS)]
        .max(axis=1)
        .sort_values(kind="mergesort")
        .index[:3]
        .tolist()
    )
    assert first.planted_source_index.tolist() == expected
    assert second.planted_source_index.tolist() == expected


def _dummy_manifest(n_channels: int, n_sources: int) -> dict:
    fits = {}
    method_keys = list(PRIMARY_METHODS) + [
        "picard_strict",
        "extended_infomax_strict",
        "fastica_strict",
    ]
    for method in method_keys:
        fits[method] = {
            "display_name": method,
            "software_package": "test",
            "software_version": "1",
            "configuration": {
                "random_seed": 42,
                "max_iter": 100,
                "internal_whitening_enabled": False,
                "stopping_parameter_name": "native test rule",
                "stopping_parameter_value": 1e-6,
                "secondary_stopping_parameters": {"test": True},
            },
            "outcome": {
                "actual_n_iter": 50,
                "runtime_seconds": 1.0,
                "stopping_reason": "native test rule",
                "hit_iteration_cap": False,
                "converged_flag": True,
            },
            "result_file": "dummy.npz",
            "notes": "test",
        }
    return {
        "provenance": {"git_commit": "abc"},
        "simulation": {
            "n_channels": n_channels,
            "n_true_sources": n_sources,
            "vertex_records": [
                {"vertex_id": index, "hemi": "lh", "label_name": f"label-{index}"}
                for index in range(n_sources)
            ],
        },
        "shared_input": {
            "initialization_id": "w0",
            "n_components": n_sources,
            "whitening_strategy": "test common whitening",
            "input_data_hash": "input",
            "whitener_hash": "white",
            "initial_weights_hash": "w0hash",
        },
        "fits": fits,
    }


def _write_dummy_archive(tmp_path: Path, *, wrong_sensor_space: bool = False):
    n_channels, n_sources = 8, 4
    a_true = _maps(n_channels=n_channels, n_sources=n_sources)
    if wrong_sensor_space:
        a_true = a_true[:n_sources]
    arrays = {
        "A_true": a_true,
        "sensor_positions": np.zeros((a_true.shape[0], 2)),
        "ch_names": np.asarray([f"EEG{index:03d}" for index in range(a_true.shape[0])]),
    }
    rng = np.random.default_rng(12)
    for method in list(PRIMARY_METHODS) + [
        "picard_strict",
        "extended_infomax_strict",
        "fastica_strict",
    ]:
        permutation = rng.permutation(n_sources)
        signs = rng.choice([-1.0, 1.0], size=n_sources)
        arrays[f"A_est_{method}"] = a_true[:, permutation] * signs
    archive = tmp_path / "fits.npz"
    np.savez_compressed(archive, **arrays)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(_dummy_manifest(n_channels, n_sources)), encoding="utf-8"
    )
    return archive, manifest


def test_archive_analysis_reuses_matches_and_writes_all_required_outputs(tmp_path):
    archive, manifest = _write_dummy_archive(tmp_path)
    output = tmp_path / "results"
    audit = analyse_archive(archive, manifest, output)
    assert audit["n_sources"] == 4
    required = {
        "method_configuration.csv",
        "matched_topographies.csv",
        "topography_recovery_summary.csv",
        "panel_d_selected_sources.csv",
        "convergence_sensitivity.csv",
        "figure2_topography_maps.npz",
        "analysis_audit.json",
    }
    assert required == {path.name for path in output.iterdir()}

    matched = pd.read_csv(output / "matched_topographies.csv")
    assert matched.groupby("method").size().eq(4).all()
    assert matched.selected_for_panel_d.sum() == 3 * len(PRIMARY_METHODS)
    selected = pd.read_csv(output / "panel_d_selected_sources.csv")
    assert selected.planted_source_index.tolist() == audit["selected_sources"]

    maps = np.load(output / "figure2_topography_maps.npz")
    for method in PRIMARY_METHODS:
        aligned = maps[f"A_est_aligned_{method}"]
        method_rows = matched[matched.method == method].sort_values("planted_source_index")
        assert aligned.shape == maps["A_true"].shape
        assert method_rows.estimated_component_index.nunique() == 4


def test_archive_analysis_rejects_pca_space_mistaken_for_sensor_space(tmp_path):
    archive, manifest = _write_dummy_archive(tmp_path, wrong_sensor_space=True)
    with pytest.raises(ValueError, match="not in the archived original sensor space"):
        analyse_archive(archive, manifest, tmp_path / "results")


def test_method_configuration_contains_stopping_and_initialisation_provenance(tmp_path):
    archive, manifest = _write_dummy_archive(tmp_path)
    output = tmp_path / "results"
    analyse_archive(archive, manifest, output)
    configuration = pd.read_csv(output / "method_configuration.csv")
    required_columns = {
        "method",
        "actual_n_iter",
        "stopping_parameter_name",
        "stopping_parameter_value",
        "stopping_reason",
        "hit_iteration_cap",
        "input_data_hash",
        "whitener_hash",
        "initial_weights_hash",
    }
    assert required_columns.issubset(configuration.columns)
    assert set(configuration.method) == set(PRIMARY_METHODS)


def test_sensor_space_mixing_unmixing_reconstruction():
    rng = np.random.default_rng(9)
    whitener = rng.normal(size=(4, 7))
    while np.linalg.matrix_rank(whitener) < 4:
        whitener = rng.normal(size=(4, 7))
    dewhitener = np.linalg.pinv(whitener)
    w_white = rng.normal(size=(4, 4))
    while np.linalg.matrix_rank(w_white) < 4:
        w_white = rng.normal(size=(4, 4))
    x_white = rng.normal(size=(4, 100))
    a_sensor = dewhitener @ np.linalg.inv(w_white)
    w_sensor = w_white @ whitener
    retained_sensor = dewhitener @ x_white
    reconstructed = a_sensor @ (w_sensor @ retained_sensor)
    assert np.allclose(reconstructed, retained_sensor, atol=1e-10, rtol=1e-10)
