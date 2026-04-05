"""
amica-python Validation Suite
==========================

Runs all validation benchmarks:
1. Synthetic source separation (Amari index)
2. ICA algorithm comparison on MNE sample data (MIR, dipolarity proxy)
3. Convergence curves
4. Parameter sensitivity (iterations, mix components, data quantity)

Results are saved to validation/results/ as JSON + figures.
"""
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════

def amari_index(W_est, A_true):
    """Compute Amari index (lower = better separation, 0 = perfect).

    Parameters
    ----------
    W_est : ndarray (n, n)
        Estimated unmixing matrix (full pipeline: W @ whitener).
    A_true : ndarray (n, n)
        True mixing matrix.
    """
    C = W_est @ A_true
    C = np.abs(C)
    n = C.shape[0]
    row_max = C.max(axis=1, keepdims=True)
    col_max = C.max(axis=0, keepdims=True)
    row_max[row_max == 0] = 1
    col_max[col_max == 0] = 1
    row_ratios = np.sum(C / row_max, axis=1) - 1
    col_ratios = np.sum(C / col_max, axis=0) - 1
    return (np.mean(row_ratios) + np.mean(col_ratios)) / (2 * (n - 1))


def compute_mir(X_orig, X_components):
    """Compute mutual information reduction (k-NN entropy estimator).

    MIR = sum H_marginal(x_i) - sum H_marginal(y_i)

    Uses the Kozachenko-Leonenko k-NN entropy estimator for each
    marginal, which is consistent and works well on real EEG data.
    The difference of marginal entropies captures how much more
    independent the components are vs the original channels.
    """
    from scipy.special import digamma

    def entropy_knn(x, k=5):
        """Kozachenko-Leonenko k-NN entropy estimator for 1D data."""
        x = np.sort(x.ravel())
        n = len(x)
        # k-th nearest neighbor distances
        dists = np.zeros(n)
        for i in range(n):
            # Distances to all other points
            d = np.abs(x - x[i])
            d_sorted = np.sort(d)
            # k-th neighbor (skip self at index 0)
            dists[i] = d_sorted[k] if k < n else d_sorted[-1]
        dists = np.maximum(dists, 1e-300)
        # KL estimator: H = digamma(n) - digamma(k) + log(2) + mean(log(eps_i))
        return digamma(n) - digamma(k) + np.log(2) + np.mean(np.log(dists))

    n_ch = min(X_orig.shape[0], X_components.shape[0])
    h_orig = sum(entropy_knn(X_orig[i]) for i in range(n_ch))
    h_comp = sum(entropy_knn(X_components[i]) for i in range(n_ch))
    return h_orig - h_comp


# ══════════════════════════════════════════════════════════════
# 1. Synthetic Source Separation
# ══════════════════════════════════════════════════════════════

def run_synthetic_benchmark():
    """Compare ICA algorithms on synthetic data with known ground truth."""
    print("\n" + "=" * 60)
    print("1. SYNTHETIC SOURCE SEPARATION BENCHMARK")
    print("=" * 60)

    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_sources = 6
    n_samples = 10000

    # Generate diverse independent sources
    sources = np.zeros((n_sources, n_samples))
    sources[0] = rng.laplace(size=n_samples)            # super-Gaussian
    sources[1] = rng.standard_t(df=3, size=n_samples)   # heavy-tailed
    sources[2] = np.sign(rng.randn(n_samples)) * rng.exponential(size=n_samples)  # asymmetric
    sources[3] = rng.uniform(-1, 1, size=n_samples)     # sub-Gaussian
    t = np.linspace(0, 20 * np.pi, n_samples)
    sources[4] = np.sin(t) + 0.1 * rng.randn(n_samples)  # quasi-periodic
    sources[5] = rng.laplace(size=n_samples) * 0.5       # scaled Laplacian

    # Random mixing
    A_true = rng.randn(n_sources, n_sources)
    X = A_true @ sources

    results = {}
    algorithms = {}

    # --- AMICA ---
    print("\n  Running AMICA...")
    t0 = time.time()
    config = AmicaConfig(max_iter=500, num_mix_comps=3, do_newton=True)
    model = Amica(config=config, random_state=42)
    result = model.fit(X)
    t_amica = time.time() - t0
    W_full = result.unmixing_matrix @ result.whitener_
    amari_amica = amari_index(W_full, A_true)
    algorithms["AMICA"] = {"amari": amari_amica, "time": t_amica, "n_iter": result.n_iter}
    print(f"    Amari index: {amari_amica:.4f}, Time: {t_amica:.1f}s, Iters: {result.n_iter}")

    # --- Shared whitening for non-AMICA methods ---
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_sources, whiten=True, random_state=42)
    X_white = pca.fit_transform(X.T).T  # (n_sources, n_samples), whitened

    # --- Picard ---
    print("  Running Picard...")
    try:
        from picard import picard
        t0 = time.time()
        _, W_picard, _, n_iter_picard = picard(
            X_white, whiten=False, return_n_iter=True, random_state=42, max_iter=500
        )
        t_picard = time.time() - t0
        W_picard_full = W_picard @ pca.components_
        amari_picard = amari_index(W_picard_full, A_true)
        algorithms["Picard"] = {"amari": amari_picard, "time": t_picard, "n_iter": n_iter_picard}
        print(f"    Amari index: {amari_picard:.4f}, Time: {t_picard:.1f}s, Iters: {n_iter_picard}")
    except Exception as e:
        print(f"    Picard failed: {e}")

    # --- Infomax ---
    print("  Running Infomax...")
    try:
        from mne.preprocessing.infomax_ import infomax
        t0 = time.time()
        W_info, n_iter_info = infomax(
            X_white.T, random_state=42, return_n_iter=True, max_iter=500
        )
        t_info = time.time() - t0
        W_info_full = W_info @ pca.components_
        amari_info = amari_index(W_info_full, A_true)
        algorithms["Infomax"] = {"amari": amari_info, "time": t_info, "n_iter": n_iter_info}
        print(f"    Amari index: {amari_info:.4f}, Time: {t_info:.1f}s, Iters: {n_iter_info}")
    except Exception as e:
        print(f"    Infomax failed: {e}")

    # --- FastICA ---
    print("  Running FastICA...")
    try:
        from sklearn.decomposition import FastICA
        t0 = time.time()
        fica = FastICA(n_components=n_sources, random_state=42, max_iter=500, whiten="unit-variance")
        S_fica = fica.fit_transform(X.T)
        t_fica = time.time() - t0
        W_fica_full = fica.components_
        amari_fica = amari_index(W_fica_full, A_true)
        algorithms["FastICA"] = {"amari": amari_fica, "time": t_fica, "n_iter": fica.n_iter_}
        print(f"    Amari index: {amari_fica:.4f}, Time: {t_fica:.1f}s, Iters: {fica.n_iter_}")
    except Exception as e:
        print(f"    FastICA failed: {e}")

    # Save results
    results["synthetic"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in algorithms.items()}

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(algorithms.keys())
    amaris = [algorithms[n]["amari"] for n in names]
    times = [algorithms[n]["time"] for n in names]
    colors = ["#2196F3" if n == "AMICA" else "#90CAF9" for n in names]
    bars = ax.bar(names, amaris, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Amari Index (lower = better)")
    ax.set_title("Source Separation Quality: Synthetic Data (6 sources, 10k samples)")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(amaris) * 1.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "synthetic_amari.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {RESULTS_DIR / 'synthetic_amari.png'}")

    return results


# ══════════════════════════════════════════════════════════════
# 2. MNE Sample Dataset Comparison
# ══════════════════════════════════════════════════════════════

def run_mne_sample_benchmark():
    """Compare ICA algorithms on MNE sample EEG data."""
    print("\n" + "=" * 60)
    print("2. MNE SAMPLE DATASET BENCHMARK")
    print("=" * 60)

    import mne

    # Load sample data
    sample_path = mne.datasets.sample.data_path()
    raw_fname = sample_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

    # Pick EEG only, filter
    raw.pick("eeg")
    raw.filter(1.0, None, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    n_components = 15
    results = {}

    # --- AMICA via fit_ica ---
    print("\n  Running AMICA on MNE sample EEG...")
    from amica_python import fit_ica
    t0 = time.time()
    ica_amica = fit_ica(raw, n_components=n_components, max_iter=200,
                        num_mix=3, random_state=42,
                        fit_params={"do_newton": True})
    t_amica = time.time() - t0
    sources_amica = ica_amica.get_sources(raw).get_data()
    mir_amica = compute_mir(raw.get_data()[:n_components], sources_amica)
    results["AMICA"] = {"mir": float(mir_amica), "time": t_amica,
                        "n_iter": int(ica_amica.n_iter_), "method": "amica"}
    print(f"    MIR: {mir_amica:.2f}, Time: {t_amica:.1f}s, Iters: {ica_amica.n_iter_}")

    # --- Picard ---
    print("  Running Picard...")
    t0 = time.time()
    ica_picard = mne.preprocessing.ICA(n_components=n_components, method="picard",
                                        random_state=42, max_iter=200)
    ica_picard.fit(raw, verbose=False)
    t_picard = time.time() - t0
    sources_picard = ica_picard.get_sources(raw).get_data()
    mir_picard = compute_mir(raw.get_data()[:n_components], sources_picard)
    results["Picard"] = {"mir": float(mir_picard), "time": t_picard,
                         "n_iter": int(ica_picard.n_iter_), "method": "picard"}
    print(f"    MIR: {mir_picard:.2f}, Time: {t_picard:.1f}s, Iters: {ica_picard.n_iter_}")

    # --- Infomax ---
    print("  Running Infomax...")
    t0 = time.time()
    ica_infomax = mne.preprocessing.ICA(n_components=n_components, method="infomax",
                                         random_state=42, max_iter=200)
    ica_infomax.fit(raw, verbose=False)
    t_infomax = time.time() - t0
    sources_infomax = ica_infomax.get_sources(raw).get_data()
    mir_infomax = compute_mir(raw.get_data()[:n_components], sources_infomax)
    results["Infomax"] = {"mir": float(mir_infomax), "time": t_infomax,
                          "n_iter": int(ica_infomax.n_iter_), "method": "infomax"}
    print(f"    MIR: {mir_infomax:.2f}, Time: {t_infomax:.1f}s, Iters: {ica_infomax.n_iter_}")

    # --- FastICA ---
    print("  Running FastICA...")
    t0 = time.time()
    ica_fastica = mne.preprocessing.ICA(n_components=n_components, method="fastica",
                                         random_state=42, max_iter=200)
    ica_fastica.fit(raw, verbose=False)
    t_fastica = time.time() - t0
    sources_fastica = ica_fastica.get_sources(raw).get_data()
    mir_fastica = compute_mir(raw.get_data()[:n_components], sources_fastica)
    results["FastICA"] = {"mir": float(mir_fastica), "time": t_fastica,
                          "n_iter": int(ica_fastica.n_iter_), "method": "fastica"}
    print(f"    MIR: {mir_fastica:.2f}, Time: {t_fastica:.1f}s, Iters: {ica_fastica.n_iter_}")

    # Plot MIR comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = list(results.keys())
    mirs = [results[n]["mir"] for n in names]
    times = [results[n]["time"] for n in names]
    colors = ["#2196F3" if n == "AMICA" else "#90CAF9" for n in names]

    ax1.bar(names, mirs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Mutual Information Reduction (bits)")
    ax1.set_title("MNE Sample EEG: MIR Comparison")

    ax2.bar(names, times, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("MNE Sample EEG: Runtime Comparison")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "mne_sample_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {RESULTS_DIR / 'mne_sample_comparison.png'}")

    return {"mne_sample": results}


# ══════════════════════════════════════════════════════════════
# 3. Convergence Curves
# ══════════════════════════════════════════════════════════════

def run_convergence_analysis():
    """Analyze AMICA convergence: LL vs iteration, Newton vs natural gradient."""
    print("\n" + "=" * 60)
    print("3. CONVERGENCE ANALYSIS")
    print("=" * 60)

    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_sources, n_samples = 6, 5000
    S = rng.laplace(size=(n_sources, n_samples))
    A = rng.randn(n_sources, n_sources)
    X = A @ S

    results = {}

    # Newton ON
    print("\n  Running with Newton (quadratic convergence)...")
    config_newton = AmicaConfig(max_iter=500, num_mix_comps=3, do_newton=True)
    model_newton = Amica(config_newton, random_state=42)
    res_newton = model_newton.fit(X)
    results["newton"] = {"ll": res_newton.log_likelihood.tolist(),
                         "n_iter": res_newton.n_iter}
    print(f"    Final LL: {res_newton.log_likelihood[-1]:.6f}, Iters: {res_newton.n_iter}")

    # Newton OFF (natural gradient only)
    print("  Running with natural gradient only (linear convergence)...")
    config_natgrad = AmicaConfig(max_iter=500, num_mix_comps=3, do_newton=False)
    model_natgrad = Amica(config_natgrad, random_state=42)
    res_natgrad = model_natgrad.fit(X)
    results["natgrad"] = {"ll": res_natgrad.log_likelihood.tolist(),
                          "n_iter": res_natgrad.n_iter}
    print(f"    Final LL: {res_natgrad.log_likelihood[-1]:.6f}, Iters: {res_natgrad.n_iter}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ll_newton = res_newton.log_likelihood
    ll_natgrad = res_natgrad.log_likelihood
    ax.plot(ll_newton, label=f"Newton (final LL={ll_newton[-1]:.4f})", linewidth=2)
    ax.plot(ll_natgrad, label=f"Natural Gradient (final LL={ll_natgrad[-1]:.4f})",
            linewidth=2, linestyle="--")
    ax.axvline(x=50, color="gray", linestyle=":", alpha=0.5, label="Newton start (iter 50)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-Likelihood (per component per sample)")
    ax.set_title("AMICA Convergence: Newton vs Natural Gradient")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence_curves.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {RESULTS_DIR / 'convergence_curves.png'}")

    return {"convergence": results}


# ══════════════════════════════════════════════════════════════
# 4. Parameter Sensitivity
# ══════════════════════════════════════════════════════════════

def run_parameter_sensitivity():
    """Test sensitivity to key parameters: num_mix, kappa, max_iter."""
    print("\n" + "=" * 60)
    print("4. PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)

    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(42)
    n_sources = 6
    S = rng.laplace(size=(n_sources, 10000))
    A = rng.randn(n_sources, n_sources)
    X_full = A @ S

    results = {}

    # 4a. Number of mixture components
    print("\n  4a. Mixture components (num_mix_comps)...")
    mix_results = {}
    for n_mix in [1, 2, 3, 5]:
        print(f"    num_mix={n_mix}...", end=" ", flush=True)
        t0 = time.time()
        config = AmicaConfig(max_iter=300, num_mix_comps=n_mix, do_newton=True)
        model = Amica(config, random_state=42)
        res = model.fit(X_full)
        dt = time.time() - t0
        W_full = res.unmixing_matrix @ res.whitener_
        ai = amari_index(W_full, A)
        mix_results[n_mix] = {"amari": float(ai), "time": dt,
                              "final_ll": float(res.log_likelihood[-1])}
        print(f"Amari={ai:.4f}, LL={res.log_likelihood[-1]:.4f}, Time={dt:.1f}s")
    results["mix_components"] = mix_results

    # 4b. Data quantity (kappa)
    print("\n  4b. Data quantity (kappa = N / n_ch^2)...")
    kappa_results = {}
    for kappa in [5, 10, 20, 30, 50]:
        n_samples = int(kappa * n_sources ** 2)
        X_sub = X_full[:, :n_samples]
        print(f"    kappa={kappa} (N={n_samples})...", end=" ", flush=True)
        t0 = time.time()
        config = AmicaConfig(max_iter=300, num_mix_comps=3, do_newton=True)
        model = Amica(config, random_state=42)
        res = model.fit(X_sub)
        dt = time.time() - t0
        W_full = res.unmixing_matrix @ res.whitener_
        ai = amari_index(W_full, A)
        kappa_results[kappa] = {"amari": float(ai), "time": dt, "n_samples": n_samples,
                                "final_ll": float(res.log_likelihood[-1])}
        print(f"Amari={ai:.4f}, Time={dt:.1f}s")
    results["kappa"] = kappa_results

    # 4c. Seed sensitivity
    print("\n  4c. Seed sensitivity (5 seeds)...")
    seed_results = {}
    for seed in [0, 42, 123, 456, 999]:
        config = AmicaConfig(max_iter=300, num_mix_comps=3, do_newton=True)
        model = Amica(config, random_state=seed)
        res = model.fit(X_full)
        W_full = res.unmixing_matrix @ res.whitener_
        ai = amari_index(W_full, A)
        seed_results[seed] = {"amari": float(ai), "final_ll": float(res.log_likelihood[-1])}
    amaris = [v["amari"] for v in seed_results.values()]
    lls = [v["final_ll"] for v in seed_results.values()]
    print(f"    Amari: {np.mean(amaris):.4f} +/- {np.std(amaris):.4f}")
    print(f"    LL:    {np.mean(lls):.4f} +/- {np.std(lls):.6f}")
    results["seed_sensitivity"] = seed_results

    # Plot parameter sensitivity
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Mix components
    ax = axes[0]
    mixes = sorted(mix_results.keys())
    ax.plot(mixes, [mix_results[m]["amari"] for m in mixes], "o-", linewidth=2)
    ax.set_xlabel("Number of Mixture Components")
    ax.set_ylabel("Amari Index")
    ax.set_title("Effect of num_mix_comps")
    ax.grid(True, alpha=0.3)

    # Kappa
    ax = axes[1]
    kappas = sorted(kappa_results.keys())
    ax.plot(kappas, [kappa_results[k]["amari"] for k in kappas], "o-", linewidth=2, color="#E91E63")
    ax.set_xlabel(r"$\kappa$ = N / n$^2$")
    ax.set_ylabel("Amari Index")
    ax.set_title("Effect of Data Quantity")
    ax.grid(True, alpha=0.3)

    # Seed
    ax = axes[2]
    seeds = sorted(seed_results.keys())
    ax.bar(range(len(seeds)), [seed_results[s]["amari"] for s in seeds],
           color="#4CAF50", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Amari Index")
    ax.set_title("Seed Sensitivity")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "parameter_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {RESULTS_DIR / 'parameter_sensitivity.png'}")

    return {"parameters": results}


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("amica-python VALIDATION SUITE")
    print("=" * 60)

    all_results = {}

    # 1. Synthetic
    all_results.update(run_synthetic_benchmark())

    # 2. MNE sample
    all_results.update(run_mne_sample_benchmark())

    # 3. Convergence
    all_results.update(run_convergence_analysis())

    # 4. Parameter sensitivity
    all_results.update(run_parameter_sensitivity())

    # Save all results
    with open(RESULTS_DIR / "validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)
