"""
AMICA Sample Rejection Ablation Study
======================================

Benchmarks AMICA decomposition quality across different rejection settings,
following Klug et al. (2024) recommendations.

Tests:
- Rejection passes: 0, 3, 5, 10
- Rejection thresholds: 2.0, 3.0, 4.0 SD

Metrics per configuration:
- Final log-likelihood
- Fraction of samples rejected
- Kurtosis distribution of sources
- Reconstruction error
- Runtime

References
----------
- Klug et al. (2024). Optimizing EEG ICA decomposition with data cleaning
  in stationary and mobile experiments. Scientific Reports.
"""
import json
import itertools
import logging
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def generate_eeg_like_data(n_channels=16, n_samples=20000, seed=42):
    """Generate synthetic EEG-like data with known sources.

    Creates a mix of super-Gaussian (brain-like), Gaussian (noise-like),
    and impulsive (artifact-like) sources with outlier contamination.
    """
    rng = np.random.RandomState(seed)

    n_brain = n_channels // 2
    n_noise = n_channels // 4
    n_artifact = n_channels - n_brain - n_noise

    sources = np.vstack([
        rng.laplace(size=(n_brain, n_samples)),         # brain-like
        rng.randn(n_noise, n_samples),                  # Gaussian noise
        rng.standard_t(df=3, size=(n_artifact, n_samples)),  # impulsive artifacts
    ])

    # Add outlier contamination (2% of samples)
    n_outlier = n_samples // 50
    outlier_idx = rng.choice(n_samples, n_outlier, replace=False)
    sources[:, outlier_idx] *= 10

    A = rng.randn(n_channels, n_channels)
    data = A @ sources

    return data, A, sources


def run_amica_config(data, numrej, rejsig, max_iter=200, seed=42):
    """Run AMICA with specific rejection settings."""
    from amica_python import Amica, AmicaConfig

    do_reject = numrej > 0
    config = AmicaConfig(
        max_iter=max_iter,
        num_mix_comps=3,
        do_newton=True,
        do_reject=do_reject,
        rejstart=2,
        rejint=3,
        rejsig=rejsig,
        numrej=numrej,
    )

    t0 = time.time()
    model = Amica(config, random_state=seed)
    result = model.fit(data)
    elapsed = time.time() - t0

    # Compute source kurtosis
    from scipy.stats import kurtosis
    sources = model.transform(data)
    kurt = kurtosis(sources, axis=1, fisher=True)

    # Reconstruction error
    recon = model.inverse_transform(sources)
    nrmse = float(np.linalg.norm(data - recon) / np.linalg.norm(data))

    return {
        "numrej": numrej,
        "rejsig": rejsig,
        "do_reject": do_reject,
        "n_iter": result.n_iter,
        "final_ll": float(result.log_likelihood[-1]),
        "converged": result.converged,
        "time": elapsed,
        "reconstruction_error": nrmse,
        "kurtosis_mean": float(np.mean(kurt)),
        "kurtosis_median": float(np.median(kurt)),
        "kurtosis_std": float(np.std(kurt)),
    }


def main():
    logger.info("=" * 65)
    logger.info("AMICA REJECTION ABLATION (Klug et al. 2024)")
    logger.info("=" * 65)

    data, A_true, sources_true = generate_eeg_like_data()
    n_channels, n_samples = data.shape
    logger.info("Data: %d channels x %d samples", n_channels, n_samples)

    # Ablation grid
    numrej_values = [0, 3, 5, 10]
    rejsig_values = [2.0, 3.0, 4.0]

    results = []

    for numrej, rejsig in itertools.product(numrej_values, rejsig_values):
        if numrej == 0 and rejsig != 3.0:
            continue  # No rejection — threshold is irrelevant

        label = f"rej={numrej}_sig={rejsig}"
        logger.info("\n--- %s ---", label)

        res = run_amica_config(data, numrej=numrej, rejsig=rejsig)
        res["label"] = label
        results.append(res)

        logger.info(
            "  LL=%.4f  NRMSE=%.2e  Kurt=%.2f±%.2f  Time=%.1fs",
            res["final_ll"], res["reconstruction_error"],
            res["kurtosis_mean"], res["kurtosis_std"], res["time"],
        )

    # Summary table
    logger.info("\n" + "=" * 85)
    logger.info("REJECTION ABLATION SUMMARY")
    logger.info("=" * 85)
    header = (
        f"{'Config':<18s} {'LL':>10s} {'NRMSE':>10s} "
        f"{'Kurt mean':>10s} {'Kurt std':>10s} {'Time':>7s} {'Iters':>6s}"
    )
    logger.info(header)
    logger.info("-" * 85)

    for r in results:
        logger.info(
            "%-18s %10.4f %10.2e %10.2f %10.2f %7.1f %6d",
            r["label"], r["final_ll"], r["reconstruction_error"],
            r["kurtosis_mean"], r["kurtosis_std"], r["time"], r["n_iter"],
        )

    # Save
    output_file = RESULTS_DIR / "rejection_ablation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("\nResults saved to: %s", output_file)

    # Recommendation
    logger.info("\n--- RECOMMENDATION (Klug et al. 2024) ---")
    logger.info("Default: rejstart=2, rejint=3, rejsig=3.0, numrej=5")
    best = min(results, key=lambda r: -r["final_ll"])
    logger.info("Best LL config: %s (LL=%.4f)", best["label"], best["final_ll"])


if __name__ == "__main__":
    main()
