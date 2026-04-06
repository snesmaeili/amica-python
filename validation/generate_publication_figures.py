"""Generate publication-quality figures from validation results.

Reads highdens_validation.json and generates:
  Fig 1: Multi-metric comparison bar chart (6 panels)
  Fig 2: IC topoplot grid (4 methods × 20 ICs)
  Fig 3: Convergence curves
  Fig 4: Source PSD comparison (top brain-like ICs)

Run after run_highdens_validation.py has completed.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Consistent styling
METHOD_NAMES = {"amica": "AMICA", "picard": "Picard", "infomax": "Infomax", "fastica": "FastICA"}
METHOD_COLORS = {"amica": "#2196F3", "picard": "#FF9800", "infomax": "#4CAF50", "fastica": "#F44336"}
IC_COLORS = {
    "brain": "#2ca02c", "muscle artifact": "#d62728", "eye blink": "#1f77b4",
    "heart beat": "#ff7f0e", "line noise": "#9467bd", "channel noise": "#8c564b",
    "other": "#7f7f7f",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def load_results():
    with open(RESULTS_DIR / "highdens_validation.json") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# Figure 1: Multi-metric comparison
# ═══════════════════════════════════════════════════════════════
def fig_multimetric(results):
    methods = [m for m in ["amica", "picard", "infomax", "fastica"] if m in results and "error" not in results[m]]

    metrics = [
        ("ICLabel Brain ICs", lambda r: r.get("iclabel", {}).get("brain", 0)),
        ("Kurtosis Brain-like", lambda r: r.get("kurtosis", {}).get("brain_like_kurtosis", 0)),
        ("Alpha-peaked ICs", lambda r: r.get("psd_alpha", {}).get("alpha_peaked_ics", 0)),
        ("MIR (higher=better)", lambda r: r.get("mir", {}).get("mir", 0)),
        ("Recon Error", lambda r: r.get("reconstruction_error", 0)),
        ("Runtime (s)", lambda r: r.get("time", 0)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for ax, (title, extractor) in zip(axes, metrics):
        vals = [extractor(results[m]) for m in methods]
        colors = [METHOD_COLORS[m] for m in methods]
        labels = [METHOD_NAMES[m] for m in methods]
        bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontweight="bold")
        for bar, val in zip(bars, vals):
            fmt = f"{val:.1e}" if abs(val) < 0.01 or abs(val) > 1000 else f"{val:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("")

    fig.suptitle("High-Density EEG Validation — ds004505 sub-01", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGURES_DIR / "fig1_multimetric.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_multimetric.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_multimetric")


# ═══════════════════════════════════════════════════════════════
# Figure 2: IC Topoplot Grid
# ═══════════════════════════════════════════════════════════════
def fig_topoplots(raw, ica_dict, labels_dict):
    """Plot top 20 ICs for each method in a grid."""
    methods = [m for m in ["amica", "picard", "infomax", "fastica"] if m in ica_dict]
    n_show = 20
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, n_show, figsize=(n_show * 1.3, n_methods * 1.8))
    if n_methods == 1:
        axes = axes[np.newaxis, :]

    for row, method in enumerate(methods):
        ica = ica_dict[method]
        labels = labels_dict.get(method)
        pred = labels["labels"] if labels else ["other"] * n_show
        probs = labels["y_pred_proba"] if labels else [0] * n_show

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

            ax.set_title(f"{col}", fontsize=5, color=color, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(1.5)

        for col in range(min(n_show, ica.n_components_), n_show):
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(METHOD_NAMES[method], fontsize=10,
                                fontweight="bold", rotation=0, labelpad=45, va="center")

    legend_elements = [Patch(facecolor=IC_COLORS[c], label=c.replace("artifact", "").strip().capitalize())
                       for c in ["brain", "muscle artifact", "eye blink", "other"]]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("IC Topoplots — ds004505 sub-01", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.06, 0.03, 1, 0.95])
    fig.savefig(FIGURES_DIR / "fig2_topoplots.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_topoplots.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_topoplots")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Summary table
# ═══════════════════════════════════════════════════════════════
def fig_summary_table(results):
    methods = [m for m in ["amica", "picard", "infomax", "fastica"] if m in results and "error" not in results[m]]

    columns = ["Method", "Time (s)", "Iter", "n_IC",
               "Brain\n(ICLabel)", "Brain\n(Kurt)", "Alpha\nICs", "MIR", "Recon\nError"]
    cell_data = []
    for m in methods:
        r = results[m]
        ic = r.get("iclabel", {})
        kurt = r.get("kurtosis", {})
        psd = r.get("psd_alpha", {})
        mir = r.get("mir", {})
        cell_data.append([
            METHOD_NAMES[m],
            f"{r['time']:.0f}",
            str(r["n_iter"]),
            str(r["n_components"]),
            str(ic.get("brain", "?")),
            str(kurt.get("brain_like_kurtosis", "?")),
            str(psd.get("alpha_peaked_ics", "?")),
            f"{mir.get('mir', 0):.1f}" if "mir" in mir else "?",
            f"{r.get('reconstruction_error', 0):.1e}",
        ])

    fig, ax = plt.subplots(figsize=(9, 1.5 + 0.4 * len(methods)))
    ax.axis("off")
    colors = [[METHOD_COLORS.get(m, "#999")] + ["white"] * (len(columns) - 1) for m in methods]
    table = ax.table(cellText=cell_data, colLabels=columns, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(columns)):
        table[0, j].set_facecolor("#1565C0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.suptitle("High-Density EEG Validation Summary — ds004505 sub-01",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_summary_table.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig3_summary_table.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_summary_table")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    print("Loading results...")
    results = load_results()

    print("Generating figures...")

    # Fig 1: Multi-metric bars
    try:
        fig_multimetric(results)
    except Exception as e:
        print(f"  fig_multimetric failed: {e}")

    # Fig 3: Summary table
    try:
        fig_summary_table(results)
    except Exception as e:
        print(f"  fig_summary_table failed: {e}")

    # Fig 2: Topoplots — needs ICA objects, so re-run if needed
    # This is called separately when ICA objects are available
    print("\nNote: fig2_topoplots requires ICA objects. Run with --topoplots flag")
    print(f"  or call fig_topoplots(raw, ica_dict, labels_dict) from your script.")

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
