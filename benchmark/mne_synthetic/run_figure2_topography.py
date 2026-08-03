"""Run the controlled clean known-topography audit for main Figure 2.

This is a deliberately narrow, one-simulation benchmark.  All algorithms
receive the same centred, PCA-whitened 32-dimensional data and the same
archived initial unmixing matrix.  Each algorithm retains its method-native
stopping rule; no numerical tolerance is described as equivalent across
methods.

The two AMICA configurations are separate matched-initialisation fits with
different iteration maxima.  They are not checkpoints from one trajectory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.linalg


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from figure2_topography_analysis import analyse_archive  # noqa: E402
from generate_synthetic_raw import generate, load_config  # noqa: E402


DATA_SEED = 101
FIT_SEED = 42
N_COMPONENTS = 32


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _collect_logger(name: str):
    logger = logging.getLogger(name)
    handler = _RecordCollector()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "not archived"


def common_whitening(
    sensor_data: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, whitener, dewhitener, whitened data, and eigenvalues."""
    sensor_data = np.asarray(sensor_data, dtype=np.float64)
    mean = sensor_data.mean(axis=1)
    centred = sensor_data - mean[:, None]
    covariance = centred @ centred.T / centred.shape[1]
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = scipy.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # EEG is stored in volts, so covariance eigenvalues are commonly far below
    # one in absolute units.  Rank must therefore be assessed relative to the
    # covariance spectrum, not with an absolute threshold such as 1e-12.
    largest = float(eigenvalues[0])
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("sensor covariance has no positive finite eigenvalue")
    rank_tolerance = (
        max(covariance.shape) * np.finfo(np.float64).eps * largest
    )
    numerical_rank = int(np.sum(eigenvalues > rank_tolerance))
    if n_components > numerical_rank:
        raise ValueError(
            "requested component count exceeds the scale-aware numerical data "
            f"rank ({n_components} requested, rank={numerical_rank}, "
            f"tolerance={rank_tolerance:.3e}, largest={largest:.3e})"
        )
    kept_values = eigenvalues[:n_components]
    kept_vectors = eigenvectors[:, :n_components]
    whitener = np.diag(1.0 / np.sqrt(kept_values)) @ kept_vectors.T
    dewhitener = kept_vectors @ np.diag(np.sqrt(kept_values))
    whitened = whitener @ centred
    whitened_covariance = whitened @ whitened.T / whitened.shape[1]
    error = np.linalg.norm(whitened_covariance - np.eye(n_components), ord="fro")
    if error > 1e-8 * n_components:
        raise AssertionError(f"common whitening failed (Frobenius error={error:.3e})")
    return mean, whitener, dewhitener, whitened, kept_values


def common_initial_weights(n_components: int, seed: int) -> np.ndarray:
    """Generate a deterministic orthogonal initial unmixing matrix."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n_components, n_components)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q @ np.diag(signs)


def _sensor_matrices(
    w_white: np.ndarray, whitener: np.ndarray, dewhitener: np.ndarray, x_white: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    w_white = np.asarray(w_white, dtype=np.float64)
    if w_white.shape != (N_COMPONENTS, N_COMPONENTS):
        raise ValueError(f"unexpected whitened unmixing shape {w_white.shape}")
    a_white = np.linalg.pinv(w_white)
    a_sensor = dewhitener @ a_white
    w_sensor = w_white @ whitener
    retained = dewhitener @ x_white
    reconstructed = a_sensor @ (w_white @ x_white)
    residual = np.linalg.norm(reconstructed - retained) / np.linalg.norm(retained)
    if not np.isfinite(residual) or residual > 1e-8:
        raise AssertionError(f"sensor-space reconstruction residual is {residual:.3e}")
    return a_sensor, w_sensor, float(residual)


def _base_fit_record(
    *,
    display_name: str,
    software_package: str,
    software_version: str,
    configuration: dict,
    outcome: dict,
    result_file: str,
    notes: str = "",
) -> dict:
    return {
        "display_name": display_name,
        "software_package": software_package,
        "software_version": software_version,
        "configuration": configuration,
        "outcome": outcome,
        "result_file": result_file,
        "notes": notes,
    }


def _run_amica(
    x_white: np.ndarray,
    w0: np.ndarray,
    whitener: np.ndarray,
    dewhitener: np.ndarray,
    *,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(
        max_iter=max_iter,
        num_models=1,
        num_mix_comps=3,
        dtype="float64",
        do_mean=False,
        do_sphere=False,
        pcakeep=None,
        do_reject=False,
    )
    solver = Amica(config, random_state=FIT_SEED)
    with _collect_logger("amica_python.solver") as log_messages:
        started = time.perf_counter()
        result = solver.fit(x_white, init_weights=w0)
        runtime = time.perf_counter() - started

    convergence_messages = [message for message in log_messages if "Converged at iteration" in message]
    if convergence_messages:
        stopping_reason = convergence_messages[-1]
    elif int(result.n_iter) >= max_iter:
        stopping_reason = "iteration cap"
    elif bool(result.converged):
        stopping_reason = "method-native AMICA convergence criterion"
    else:
        stopping_reason = "terminated without a recorded convergence message"

    w_white = np.asarray(result.unmixing_matrix_white_, dtype=np.float64)
    a_sensor, w_sensor, reconstruction = _sensor_matrices(
        w_white, whitener, dewhitener, x_white
    )
    likelihood = np.asarray(result.log_likelihood, dtype=np.float64)
    final_increment = float(likelihood[-1] - likelihood[-2]) if likelihood.size >= 2 else np.nan
    outcome = {
        "actual_n_iter": int(result.n_iter),
        "runtime_seconds": float(runtime),
        "stopping_reason": stopping_reason,
        "hit_iteration_cap": bool(int(result.n_iter) >= max_iter and not result.converged),
        "converged_flag": bool(result.converged),
        "final_likelihood": float(likelihood[-1]) if likelihood.size else np.nan,
        "final_likelihood_increment": final_increment,
        "final_learning_rate": "not retained by AmicaResult",
        "sensor_reconstruction_relative_residual": reconstruction,
    }
    configuration = {
        "random_seed": FIT_SEED,
        "max_iter": max_iter,
        "internal_whitening_enabled": False,
        "stopping_parameter_name": "min_dll with consecutive-increment rule",
        "stopping_parameter_value": config.min_dll,
        "secondary_stopping_parameters": {
            "use_min_dll": config.use_min_dll,
            "max_incs": config.max_incs,
            "minlrate": config.minlrate,
            "lrate": config.lrate,
            "lratefact": config.lratefact,
            "rholrate": config.rholrate,
            "rholratefact": config.rholratefact,
            "do_newton": config.do_newton,
            "newt_start": config.newt_start,
            "newt_ramp": config.newt_ramp,
            "do_mean": config.do_mean,
            "do_sphere": config.do_sphere,
            "do_reject": config.do_reject,
            "density_initialisation": "AmicaConfig defaults with random_state=42",
        },
    }
    return a_sensor, w_sensor, configuration, outcome


def _run_picard(
    x_white: np.ndarray,
    w0: np.ndarray,
    whitener: np.ndarray,
    dewhitener: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    from picard import picard

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        _, w_white, _, n_iter_zero_based = picard(
            x_white,
            fun="tanh",
            n_components=N_COMPONENTS,
            ortho=False,
            extended=True,
            whiten=False,
            centering=False,
            return_n_iter=True,
            max_iter=max_iter,
            tol=tol,
            m=7,
            ls_tries=10,
            lambda_min=0.01,
            check_fun=True,
            w_init=w0,
            fastica_it=None,
            random_state=FIT_SEED,
            verbose=False,
        )
        runtime = time.perf_counter() - started
    warning_text = " | ".join(str(item.message) for item in caught)
    did_not_converge = "did not converge" in warning_text.lower()
    actual_n_iter = int(n_iter_zero_based) + 1
    stopping_reason = "iteration cap" if did_not_converge else "Picard gradient tolerance"
    a_sensor, w_sensor, reconstruction = _sensor_matrices(
        w_white, whitener, dewhitener, x_white
    )
    configuration = {
        "random_seed": FIT_SEED,
        "max_iter": max_iter,
        "internal_whitening_enabled": False,
        "stopping_parameter_name": "Picard gradient tolerance",
        "stopping_parameter_value": tol,
        "secondary_stopping_parameters": {
            "ortho": False,
            "extended": True,
            "fun": "tanh",
            "m": 7,
            "ls_tries": 10,
            "lambda_min": 0.01,
            "fastica_it": None,
            "centering": False,
            "whiten": False,
        },
    }
    outcome = {
        "actual_n_iter": actual_n_iter,
        "runtime_seconds": float(runtime),
        "stopping_reason": stopping_reason,
        "hit_iteration_cap": bool(did_not_converge),
        "converged_flag": not did_not_converge,
        "warning": warning_text,
        "sensor_reconstruction_relative_residual": reconstruction,
    }
    return a_sensor, w_sensor, configuration, outcome


_INFOMAX_STEP = re.compile(
    r"step\s+(?P<step>\d+)\s+-\s+lrate\s+(?P<lrate>[0-9.eE+-]+),\s+"
    r"wchange\s+(?P<change>[0-9.eE+-]+),\s+angledelta\s+(?P<angle>[0-9.eE+-]+)"
)


def _infomax_stop_from_log(
    messages: list[str], *, max_iter: int, w_change: float, anneal_deg: float, n_small_angle: int
) -> tuple[int, str, float | None, float | None]:
    records = []
    for message in messages:
        match = _INFOMAX_STEP.search(message)
        if match:
            records.append(
                (
                    int(match.group("step")),
                    float(match.group("change")),
                    float(match.group("angle")),
                )
            )
    if not records:
        return max_iter, "stopping cause not recoverable from log", None, None
    actual = records[-1][0]
    final_change = records[-1][1]
    final_angle = records[-1][2]
    trailing_small = 0
    for _, _, angle in reversed(records):
        if angle <= anneal_deg:
            trailing_small += 1
        else:
            break
    if actual > 2 and final_change < w_change:
        reason = "extended Infomax weight-change criterion"
    elif trailing_small > n_small_angle:
        reason = "extended Infomax consecutive small-angle criterion"
    elif actual >= max_iter:
        reason = "iteration cap"
    else:
        reason = "method-native extended Infomax criterion (unresolved subtype)"
    return actual, reason, final_change, final_angle


def _run_infomax(
    x_white: np.ndarray,
    w0: np.ndarray,
    whitener: np.ndarray,
    dewhitener: np.ndarray,
    *,
    max_iter: int,
    w_change: float,
    n_small_angle: int,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    from mne.preprocessing import infomax

    l_rate = 0.01 / math.log(N_COMPONENTS**2.0)
    block = int(math.floor(math.sqrt(x_white.shape[1] / 3.0)))
    anneal_deg = 60.0
    anneal_step = 0.9
    with _collect_logger("mne") as log_messages:
        started = time.perf_counter()
        w_white, returned_n_iter = infomax(
            x_white.T,
            weights=w0,
            l_rate=l_rate,
            block=block,
            w_change=w_change,
            anneal_deg=anneal_deg,
            anneal_step=anneal_step,
            extended=True,
            n_subgauss=1,
            kurt_size=6000,
            ext_blocks=1,
            max_iter=max_iter,
            random_state=FIT_SEED,
            blowup=10000.0,
            blowup_fac=0.5,
            n_small_angle=n_small_angle,
            use_bias=True,
            verbose=True,
            return_n_iter=True,
        )
        runtime = time.perf_counter() - started
    actual_n_iter, stopping_reason, final_change, final_angle = _infomax_stop_from_log(
        log_messages,
        max_iter=max_iter,
        w_change=w_change,
        anneal_deg=anneal_deg,
        n_small_angle=n_small_angle,
    )
    hit_cap = stopping_reason == "iteration cap"
    a_sensor, w_sensor, reconstruction = _sensor_matrices(
        w_white, whitener, dewhitener, x_white
    )
    configuration = {
        "random_seed": FIT_SEED,
        "max_iter": max_iter,
        "internal_whitening_enabled": False,
        "stopping_parameter_name": "weight change and consecutive small-angle criteria",
        "stopping_parameter_value": w_change,
        "secondary_stopping_parameters": {
            "extended": True,
            "w_change": w_change,
            "n_small_angle": n_small_angle,
            "anneal_deg": anneal_deg,
            "anneal_step": anneal_step,
            "l_rate": l_rate,
            "block": block,
            "n_subgauss": 1,
            "kurt_size": 6000,
            "ext_blocks": 1,
            "use_bias": True,
        },
    }
    outcome = {
        "actual_n_iter": int(actual_n_iter),
        "library_returned_n_iter": int(returned_n_iter),
        "runtime_seconds": float(runtime),
        "stopping_reason": stopping_reason,
        "hit_iteration_cap": hit_cap,
        "converged_flag": not hit_cap and "unresolved" not in stopping_reason,
        "final_weight_change": final_change,
        "final_angle_degrees": final_angle,
        "sensor_reconstruction_relative_residual": reconstruction,
    }
    return a_sensor, w_sensor, configuration, outcome


def _run_fastica(
    x_white: np.ndarray,
    w0: np.ndarray,
    whitener: np.ndarray,
    dewhitener: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning

    estimator = FastICA(
        n_components=N_COMPONENTS,
        algorithm="parallel",
        whiten=False,
        fun="logcosh",
        fun_args=None,
        max_iter=max_iter,
        tol=tol,
        w_init=w0,
        whiten_solver="svd",
        random_state=FIT_SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        estimator.fit(x_white.T)
        runtime = time.perf_counter() - started
    convergence_warnings = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    actual_n_iter = int(estimator.n_iter_)
    hit_cap = bool(convergence_warnings or actual_n_iter >= max_iter)
    stopping_reason = "iteration cap" if hit_cap else "FastICA unmixing convergence tolerance"
    w_white = np.asarray(estimator.components_, dtype=np.float64)
    a_sensor, w_sensor, reconstruction = _sensor_matrices(
        w_white, whitener, dewhitener, x_white
    )
    configuration = {
        "random_seed": FIT_SEED,
        "max_iter": max_iter,
        "internal_whitening_enabled": False,
        "stopping_parameter_name": "FastICA unmixing convergence tolerance",
        "stopping_parameter_value": tol,
        "secondary_stopping_parameters": {
            "algorithm": "parallel",
            "fun": "logcosh",
            "fun_args": None,
            "whiten": False,
            "whiten_solver": "svd",
        },
    }
    outcome = {
        "actual_n_iter": actual_n_iter,
        "runtime_seconds": float(runtime),
        "stopping_reason": stopping_reason,
        "hit_iteration_cap": hit_cap,
        "converged_flag": not hit_cap,
        "warning": " | ".join(str(item.message) for item in caught),
        "sensor_reconstruction_relative_residual": reconstruction,
    }
    return a_sensor, w_sensor, configuration, outcome


def _sensor_positions(raw) -> np.ndarray:
    import mne
    from mne.channels.layout import _find_topomap_coords

    picks = mne.pick_types(raw.info, meg=False, eeg=True, exclude=[])
    positions = _find_topomap_coords(raw.info, picks=picks, ignore_overlap=True)
    if positions.shape != (len(raw.ch_names), 2) or not np.all(np.isfinite(positions)):
        raise AssertionError("failed to archive valid two-dimensional EEG sensor positions")
    return positions


def run(config_path: Path, output_dir: Path, *, force_regenerate: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    config = load_config(config_path)
    bundle = generate(
        config,
        condition_id="clean",
        seed=DATA_SEED,
        cache_dir=cache_dir,
        force=force_regenerate,
        verbose=True,
    )
    raw = bundle["raw"]
    x_sensor = np.asarray(raw.get_data(), dtype=np.float64)
    a_true = np.asarray(bundle["A_true"], dtype=np.float64)
    if x_sensor.shape != (59, 300000):
        raise AssertionError(f"unexpected controlled sensor matrix shape {x_sensor.shape}")
    if a_true.shape != (59, N_COMPONENTS):
        raise AssertionError(f"unexpected planted mixing shape {a_true.shape}")
    if list(raw.ch_names) != list(bundle["ch_names"]):
        raise AssertionError("ground-truth and Raw channel orders differ")

    mean, whitener, dewhitener, x_white, eigenvalues = common_whitening(
        x_sensor, N_COMPONENTS
    )
    w0 = common_initial_weights(N_COMPONENTS, FIT_SEED)
    archive_path = output_dir / "figure2_topography_fit_outputs.npz"
    manifest_path = output_dir / "figure2_topography_manifest.json"
    arrays: dict[str, np.ndarray] = {
        "A_true": a_true,
        "S_true": np.asarray(bundle["S_true"], dtype=np.float64),
        "sensor_positions": _sensor_positions(raw),
        "ch_names": np.asarray(raw.ch_names, dtype="U32"),
        "sensor_mean": mean,
        "whitener": whitener,
        "dewhitener": dewhitener,
        "whitening_eigenvalues": eigenvalues,
        "initial_weights": w0,
    }

    repo_root = THIS_DIR.parent.parent
    manifest = {
        "schema_version": "figure2-topography-v1",
        "provenance": {
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "git_commit": _git_commit(repo_root),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "software_versions": {
                "amica-python": _version("amica-python"),
                "mne": _version("mne"),
                "numpy": _version("numpy"),
                "scipy": _version("scipy"),
                "scikit-learn": _version("scikit-learn"),
                "python-picard": _version("python-picard"),
                "jax": _version("jax"),
                "jaxlib": _version("jaxlib"),
            },
        },
        "simulation": {
            "dataset": "mne.datasets.sample",
            "forward_solution": config["forward"]["filename"],
            "forward_orientation": config["forward"]["orientation"],
            "condition": "clean",
            "source_waveform": config["sources"]["waveform"],
            "data_seed": DATA_SEED,
            "n_true_sources": N_COMPONENTS,
            "n_channels": len(raw.ch_names),
            "channel_names": list(raw.ch_names),
            "sampling_rate_hz": float(raw.info["sfreq"]),
            "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
            "n_samples": int(raw.n_times),
            "reference": "average, applied directly",
            "highpass_hz": config["preprocess"]["highpass_hz"],
            "ssp_projectors_active": bool(any(projector.get("active", False) for projector in raw.info["projs"])),
            "vertex_records": bundle["vertex_records"],
        },
        "shared_input": {
            "n_components": N_COMPONENTS,
            "whitening_strategy": "one external covariance-eigendecomposition PCA whitener shared by every fit",
            "initialization_id": "orthogonal_qr_seed_42",
            "input_data_hash": _sha256_array(x_sensor),
            "whitened_data_hash": _sha256_array(x_white),
            "whitener_hash": _sha256_array(whitener),
            "initial_weights_hash": _sha256_array(w0),
            "whitened_covariance_frobenius_error": float(
                np.linalg.norm(x_white @ x_white.T / x_white.shape[1] - np.eye(N_COMPONENTS))
            ),
        },
        "fits": {},
    }

    run_specs = (
        ("amica_3000", "AMICA 3,000", _run_amica, {"max_iter": 3000}),
        ("amica_10000", "AMICA 10,000", _run_amica, {"max_iter": 10000}),
        ("picard", "Picard", _run_picard, {"max_iter": 5000, "tol": 1e-6}),
        (
            "extended_infomax",
            "Ext. Infomax",
            _run_infomax,
            {"max_iter": 5000, "w_change": 1e-7, "n_small_angle": 20},
        ),
        ("fastica", "FastICA", _run_fastica, {"max_iter": 5000, "tol": 1e-6}),
        ("picard_strict", "Picard strict", _run_picard, {"max_iter": 10000, "tol": 1e-8}),
        (
            "extended_infomax_strict",
            "Ext. Infomax strict",
            _run_infomax,
            {"max_iter": 10000, "w_change": 1e-9, "n_small_angle": 40},
        ),
        ("fastica_strict", "FastICA strict", _run_fastica, {"max_iter": 10000, "tol": 1e-8}),
    )

    for method, display_name, runner, kwargs in run_specs:
        print(f"[figure2-topography] fitting {display_name}", flush=True)
        # Every fit gets its own copy. ``mne.preprocessing.infomax`` updates its
        # ``weights`` argument in place, so passing the shared ``w0`` meant the
        # four fits scheduled after extended Infomax started from an Infomax
        # solution rather than the orthogonal QR initialisation this audit
        # claims they all share. Only ``fastica`` of those four reaches the
        # manuscript, and re-running it from the correct initialisation moves
        # its median matched correlation by 3.2e-06, so no published comparison
        # changes -- but the archived ``initial_weights`` array was the mutated
        # matrix, not the initialisation, and the claim was untrue as written.
        a_est, w_est, fit_configuration, outcome = runner(
            x_white, w0.copy(), whitener, dewhitener, **kwargs
        )
        arrays[f"A_est_{method}"] = a_est
        arrays[f"W_est_{method}"] = w_est
        if method.startswith("amica"):
            package = "amica-python"
            version = manifest["provenance"]["software_versions"]["amica-python"]
            notes = (
                "Separate matched-initialisation fit; the 3,000 and 10,000 "
                "configurations are not checkpoints from one trajectory."
            )
        elif method.startswith("picard"):
            package = "python-picard"
            version = manifest["provenance"]["software_versions"]["python-picard"]
            notes = "Non-orthogonal Picard (ortho=False, extended=True), not Picard-O."
        elif method.startswith("extended_infomax"):
            package = "MNE-Python"
            version = manifest["provenance"]["software_versions"]["mne"]
            notes = "Native weight-change and consecutive small-angle stopping logic retained."
        else:
            package = "scikit-learn"
            version = manifest["provenance"]["software_versions"]["scikit-learn"]
            notes = "Parallel FastICA on the shared externally whitened input."
        manifest["fits"][method] = _base_fit_record(
            display_name=display_name,
            software_package=package,
            software_version=version,
            configuration=fit_configuration,
            outcome=outcome,
            result_file=str(archive_path),
            notes=notes,
        )
        print(
            f"  n_iter={outcome['actual_n_iter']} reason={outcome['stopping_reason']} "
            f"runtime={outcome['runtime_seconds']:.2f}s",
            flush=True,
        )

    np.savez_compressed(archive_path, **arrays)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit = analyse_archive(archive_path, manifest_path, output_dir)
    return {"manifest": str(manifest_path), "archive": str(archive_path), "analysis": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=THIS_DIR / "configs" / "benchmark_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output_dir, force_regenerate=args.force_regenerate)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
