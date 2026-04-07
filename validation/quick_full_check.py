"""Quick full validation check — 200 iter AMICA + Picard + Infomax.

Produces ALL metrics + ALL figures + saves ICA objects.
Run on GPU node for speed: sbatch submit_quick_check.sh
"""
import sys, os, time, json, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import mne

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── Config ──
DS_PATH = Path(os.environ.get("DS_PATH", "/home/sesma/scratch/ds004505"))
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MAX_ITER = 200
AMICA_MAX_ITER = 2000  # AMICA converges much slower than Picard/Infomax in iter count
N_COMP = 30  # fewer for quick check
METHODS = ["amica", "picard", "infomax"]

# ── Colors ──
METHOD_NAMES = {"amica": "AMICA", "picard": "Picard", "infomax": "Infomax", "fastica": "FastICA"}
METHOD_COLORS = {"amica": "#2196F3", "picard": "#FF9800", "infomax": "#4CAF50", "fastica": "#F44336"}
IC_COLORS = {
    "brain": "#2ca02c", "muscle artifact": "#d62728", "eye blink": "#1f77b4",
    "heart beat": "#ff7f0e", "line noise": "#9467bd", "channel noise": "#8c564b",
    "other": "#7f7f7f",
}

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})


# ═══════════════════════════════════════════════════════════════
# 1. PREPROCESSING
# ═══════════════════════════════════════════════════════════════
def load_and_preprocess(subject="sub-01"):
    from validation.run_highdens_validation import (
        load_and_preprocess as _load,
        determine_n_components,
    )
    raw, ch_groups = _load(subject)
    n_comp = determine_n_components(raw, max_components=N_COMP)
    return raw, n_comp


# ═══════════════════════════════════════════════════════════════
# 2. RUN ICA METHODS
# ═══════════════════════════════════════════════════════════════
def run_all_methods(raw, n_comp):
    ica_dict = {}
    times = {}
    amica_result = None

    for method in METHODS:
        method_iter = AMICA_MAX_ITER if method == "amica" else MAX_ITER
        print(f"\n{'='*50}")
        print(f"  {method.upper()} ({n_comp} comp, {method_iter} iter)")
        print(f"{'='*50}")

        if method == "amica":
            from amica_python import Amica, AmicaConfig
            data = raw.get_data()
            t0 = time.time()
            config = AmicaConfig(
                max_iter=method_iter, num_mix_comps=3, do_newton=True,
                do_mean=True, do_sphere=True, pcakeep=n_comp,
            )
            solver = Amica(config, random_state=42)
            amica_result = solver.fit(data)
            dt = time.time() - t0
            ica = amica_result.to_mne(raw.info)
        else:
            fit_params = {}
            if method == "infomax":
                fit_params = dict(extended=True)
            elif method == "picard":
                fit_params = dict(ortho=False, extended=True)

            t0 = time.time()
            ica = mne.preprocessing.ICA(
                n_components=n_comp, method=method, random_state=42,
                max_iter=method_iter, fit_params=fit_params,
            )
            ica.fit(raw, verbose=False)
            dt = time.time() - t0

        print(f"  Done: {dt:.1f}s, {ica.n_iter_} iter")
        ica_dict[method] = ica
        times[method] = dt

    return ica_dict, times, amica_result


# ═══════════════════════════════════════════════════════════════
# 3. ALL METRICS
# ═══════════════════════════════════════════════════════════════
def compute_all_metrics(ica_dict, raw, times):
    from mne_icalabel import label_components
    from scipy.stats import kurtosis
    from mne.time_frequency import psd_array_welch
    from scipy.special import digamma

    def entropy_knn(x, k=5):
        x = np.sort(x.ravel())
        n = len(x)
        d = np.abs(x[:, None] - x[None, :])
        d_sorted = np.sort(d, axis=1)
        dists = d_sorted[:, k]
        dists = np.maximum(dists, 1e-300)
        return digamma(n) - digamma(k) + np.log(2) + np.mean(np.log(dists))

    results = {}
    labels_dict = {}

    for method, ica in ica_dict.items():
        print(f"\n  Evaluating {method}...")
        r = {"time": times[method], "n_iter": int(ica.n_iter_), "n_components": int(ica.n_components_)}

        # ICLabel
        try:
            lab = label_components(raw, ica, method="iclabel")
            pred = lab["labels"]
            probs = np.array(lab["y_pred_proba"])
            brain_mask = np.array(pred) == "brain"
            r["iclabel"] = {
                "brain": int(brain_mask.sum()),
                "brain_50pct": int(np.sum(brain_mask & (probs > 0.5))),
                "brain_70pct": int(np.sum(brain_mask & (probs > 0.7))),
                "muscle": sum(1 for l in pred if l == "muscle artifact"),
                "eye": sum(1 for l in pred if l == "eye blink"),
                "other": sum(1 for l in pred if l == "other"),
            }
            labels_dict[method] = lab
            print(f"    ICLabel: brain={r['iclabel']['brain']} (>50%:{r['iclabel']['brain_50pct']})")
        except Exception as e:
            print(f"    ICLabel failed: {e}")
            r["iclabel"] = {"error": str(e)}
            labels_dict[method] = None

        # Kurtosis
        try:
            sources = ica.get_sources(raw).get_data()
            kurt = kurtosis(sources, axis=1, fisher=True)
            brain_like = int(np.sum((kurt > 0) & (kurt < 10)))
            r["kurtosis"] = {"brain_like": brain_like, "mean": float(np.mean(kurt)),
                             "median": float(np.median(kurt)), "values": kurt.tolist()}
            print(f"    Kurtosis: {brain_like} brain-like")
        except Exception as e:
            print(f"    Kurtosis failed: {e}")
            sources = None

        # PSD alpha peaks
        try:
            if sources is not None:
                sfreq = raw.info["sfreq"]
                psds, freqs = psd_array_welch(sources, sfreq=sfreq, fmin=1, fmax=45,
                                               n_fft=int(2 * sfreq), verbose=False)
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                flank_mask = ((freqs >= 2) & (freqs < 7)) | ((freqs > 14) & (freqs <= 30))
                n_alpha = 0
                for i in range(psds.shape[0]):
                    flank_power = psds[i, flank_mask].mean()
                    if flank_power > 1e-10:
                        if psds[i, alpha_mask].mean() / flank_power > 1.5:
                            n_alpha += 1
                r["psd_alpha"] = {"alpha_peaked_ics": n_alpha}
                print(f"    Alpha ICs: {n_alpha}")
        except Exception as e:
            print(f"    PSD failed: {e}")

        # MIR (subsample for speed)
        try:
            if sources is not None:
                original = raw.get_data()[:sources.shape[0]]
                n_sub = min(20000, sources.shape[1])
                idx = np.random.RandomState(42).choice(sources.shape[1], n_sub, replace=False)
                n_ch = min(original.shape[0], sources.shape[0])
                h_orig = sum(entropy_knn(original[i, idx]) for i in range(n_ch))
                h_comp = sum(entropy_knn(sources[i, idx]) for i in range(n_ch))
                r["mir"] = {"mir": float(h_orig - h_comp)}
                print(f"    MIR: {r['mir']['mir']:.1f}")
        except Exception as e:
            print(f"    MIR failed: {e}")

        # Reconstruction error
        try:
            raw_copy = raw.copy()
            raw_recon = ica.apply(raw_copy, verbose=False)
            err = np.linalg.norm(raw.get_data() - raw_recon.get_data()) / np.linalg.norm(raw.get_data())
            r["reconstruction_error"] = float(err)
            print(f"    Recon error: {err:.2e}")
        except Exception as e:
            print(f"    Recon error failed: {e}")
            r["reconstruction_error"] = float("nan")

        results[method] = r

    return results, labels_dict


# ═══════════════════════════════════════════════════════════════
# 4. ALL FIGURES
# ═══════════════════════════════════════════════════════════════

def fig_topoplots(ica_dict, labels_dict, raw):
    """Fig: IC topoplots per method with ICLabel coloring."""
    methods = [m for m in METHODS if m in ica_dict]
    n_show = min(15, N_COMP)
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, n_show, figsize=(n_show * 1.8, n_methods * 2.5))
    if n_methods == 1:
        axes = axes[np.newaxis, :]

    for row, method in enumerate(methods):
        ica = ica_dict[method]
        lab = labels_dict.get(method)
        pred = lab["labels"] if lab else ["other"] * n_show
        probs = lab["y_pred_proba"] if lab else [0] * n_show

        for col in range(min(n_show, ica.n_components_)):
            ax = axes[row, col]
            label = pred[col] if col < len(pred) else "other"
            prob = probs[col] if col < len(probs) else 0
            color = IC_COLORS.get(label, IC_COLORS["other"])

            try:
                ica.plot_components(picks=[col], axes=ax, show=False,
                                   colorbar=False, title="")
            except Exception:
                pass

            ax.set_title(f"IC{col}\n{prob:.2f}", fontsize=7, color=color, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(2)

        for col in range(min(n_show, ica.n_components_), n_show):
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(METHOD_NAMES[method], fontsize=10,
                                fontweight="bold", rotation=0, labelpad=45, va="center")

    legend_elements = [Patch(facecolor=IC_COLORS[c], label=c.replace("artifact", "").strip().capitalize())
                       for c in ["brain", "muscle artifact", "eye blink", "other"]]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"IC Topoplots — ds004505 sub-01 ({N_COMP} comp, {MAX_ITER} iter)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.06, 0.03, 1, 0.95])
    fig.savefig(FIGURES_DIR / "check_topoplots.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "check_topoplots.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved check_topoplots")


def fig_multimetric(results):
    """Fig: Multi-metric comparison bar chart."""
    methods = [m for m in METHODS if m in results and "error" not in results[m]]

    metrics = [
        ("ICLabel Brain ICs", lambda r: r.get("iclabel", {}).get("brain", 0)),
        ("Kurtosis Brain-like", lambda r: r.get("kurtosis", {}).get("brain_like", 0)),
        ("Alpha-peaked ICs", lambda r: r.get("psd_alpha", {}).get("alpha_peaked_ics", 0)),
        ("MIR (higher=better)", lambda r: r.get("mir", {}).get("mir", 0)),
        ("Recon Error", lambda r: r.get("reconstruction_error", 0)),
        ("Runtime (s)", lambda r: r.get("time", 0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (title, extractor) in zip(axes.flatten(), metrics):
        vals = [extractor(results[m]) for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        labels = [METHOD_NAMES[m] for m in methods]
        # Use log scale for recon error (values span many orders of magnitude)
        if title == "Recon Error":
            plot_vals = [max(v, 1e-16) for v in vals]  # avoid log(0)
            bars = ax.bar(labels, plot_vals, color=colors, edgecolor="white", linewidth=0.5)
            ax.set_yscale("log")
        else:
            bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontweight="bold")
        for bar, val in zip(bars, vals):
            fmt = f"{val:.1e}" if (abs(val) < 0.01 or abs(val) > 1000) and val != 0 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"Multi-Metric Comparison — ds004505 sub-01 ({MAX_ITER} iter)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "check_multimetric.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "check_multimetric.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved check_multimetric")


def fig_amica_specific(amica_result, raw):
    """Fig: AMICA-specific visualizations (convergence, source densities, etc.)."""
    from amica_python.viz import plot_parameter_summary

    # plot_parameter_summary expects raw channel data; it whitens and unmixes internally.
    data = raw.get_data()
    fig = plot_parameter_summary(amica_result, data=data[:, :50000], show=False)
    fig.suptitle("AMICA Parameter Summary — ds004505 sub-01", fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(FIGURES_DIR / "check_amica_params.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "check_amica_params.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved check_amica_params")


def fig_convergence(amica_result):
    """Fig: AMICA convergence curve."""
    from amica_python.viz import plot_convergence

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_convergence(amica_result, ax=ax, show=False)
    ax.set_title("AMICA Convergence — ds004505 sub-01", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "check_convergence.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "check_convergence.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved check_convergence")


def fig_source_psd(ica_dict, raw):
    """Fig: Source PSD for top brain-like ICs per method."""
    from mne.time_frequency import psd_array_welch
    from scipy.stats import kurtosis as scipy_kurtosis

    methods = [m for m in METHODS if m in ica_dict]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        ica = ica_dict[method]
        sources = ica.get_sources(raw).get_data()
        sfreq = raw.info["sfreq"]

        psds, freqs = psd_array_welch(sources, sfreq=sfreq, fmin=1, fmax=45,
                                       n_fft=int(2 * sfreq), verbose=False)

        # Pick top 5 ICs by lowest kurtosis (most brain-like)
        kurt = scipy_kurtosis(sources, axis=1, fisher=True)
        top_idx = np.argsort(kurt)[:5]

        for i, idx in enumerate(top_idx):
            ax.semilogy(freqs, psds[idx], label=f"IC{idx} (k={kurt[idx]:.0f})", alpha=0.8)

        ax.axvspan(8, 13, alpha=0.1, color="green", label="Alpha")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(METHOD_NAMES[method], fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

    axes[0].set_ylabel("PSD (V²/Hz)")
    fig.suptitle("Source PSD — Top 5 Brain-like ICs (lowest kurtosis)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "check_source_psd.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "check_source_psd.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved check_source_psd")


def fig_summary_table(results):
    """Fig: Summary table."""
    methods = [m for m in METHODS if m in results and "error" not in results[m]]
    columns = ["Method", "Time(s)", "Iter", "nIC", "Brain\n(ICL)", "B>50%",
               "Kurt\nbrain", "Alpha", "MIR", "Recon\nErr"]
    rows = []
    for m in methods:
        r = results[m]
        ic = r.get("iclabel", {})
        k = r.get("kurtosis", {})
        p = r.get("psd_alpha", {})
        mir = r.get("mir", {})
        rows.append([
            METHOD_NAMES[m], f"{r['time']:.0f}", str(r["n_iter"]), str(r["n_components"]),
            str(ic.get("brain", "?")), str(ic.get("brain_50pct", "?")),
            str(k.get("brain_like", "?")), str(p.get("alpha_peaked_ics", "?")),
            f"{mir.get('mir', 0):.1f}" if "mir" in mir else "?",
            f"{r.get('reconstruction_error', 0):.1e}",
        ])

    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.4 * len(methods)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(len(columns)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")
    fig.suptitle(f"Validation Summary — ds004505 sub-01 ({MAX_ITER} iter)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "check_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved check_summary")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    import jax
    print(f"JAX backend: {jax.default_backend()}, devices: {jax.devices()}")

    # 1. Load data
    print("\n[1/5] Loading and preprocessing...")
    raw, n_comp = load_and_preprocess()

    # 2. Run all methods
    print("\n[2/5] Running ICA methods...")
    ica_dict, times, amica_result = run_all_methods(raw, n_comp)

    # 3. Save ICA objects
    print("\n[3/5] Saving ICA objects...")
    for method, ica in ica_dict.items():
        path = RESULTS_DIR / f"ica_{method}.pkl"
        with open(path, "wb") as f:
            pickle.dump(ica, f)
        print(f"  Saved {path.name}")
    if amica_result is not None:
        with open(RESULTS_DIR / "amica_result.pkl", "wb") as f:
            pickle.dump(amica_result, f)
        print("  Saved amica_result.pkl")

    # 4. Compute all metrics
    print("\n[4/5] Computing metrics...")
    results, labels_dict = compute_all_metrics(ica_dict, raw, times)

    # Save JSON
    with open(RESULTS_DIR / "quick_check_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("  Saved quick_check_results.json")

    # 5. Generate ALL figures
    print("\n[5/5] Generating figures...")

    try:
        fig_topoplots(ica_dict, labels_dict, raw)
    except Exception as e:
        print(f"  Topoplots failed: {e}")
        import traceback; traceback.print_exc()

    try:
        fig_multimetric(results)
    except Exception as e:
        print(f"  Multimetric failed: {e}")

    try:
        fig_summary_table(results)
    except Exception as e:
        print(f"  Summary table failed: {e}")

    try:
        fig_amica_specific(amica_result, raw)
    except Exception as e:
        print(f"  AMICA params failed: {e}")
        import traceback; traceback.print_exc()

    try:
        fig_convergence(amica_result)
    except Exception as e:
        print(f"  Convergence failed: {e}")

    try:
        fig_source_psd(ica_dict, raw)
    except Exception as e:
        print(f"  Source PSD failed: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for m in METHODS:
        r = results.get(m, {})
        if "error" in r:
            print(f"  {METHOD_NAMES.get(m, m):10s}: FAILED")
            continue
        ic = r.get("iclabel", {})
        print(f"  {METHOD_NAMES.get(m, m):10s}: Brain={ic.get('brain', '?'):>2}, "
              f"Time={r['time']:.0f}s, Iter={r['n_iter']}")

    print(f"\nFigures: {FIGURES_DIR}")
    print(f"Results: {RESULTS_DIR / 'quick_check_results.json'}")
    print(f"ICA objects: {RESULTS_DIR}/ica_*.pkl")
    print("\nDone.")


if __name__ == "__main__":
    main()
