"""Generate the full paper figure + table set from validation outputs.

Inputs
------
* validation/results/benchmark_sub-*_hp*hz.json   (per-subject metric JSONs)
* validation/results/ica_*.pkl                    (optional, for topo + props
                                                   figures from sub-01)
* validation/results/amica_result.pkl             (optional, for convergence)

Outputs
-------
* validation/results/figures_paper/*.{png,pdf}
* validation/results/paper_summary_table.csv
* validation/results/paper_stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running both as `python -m validation.v2.run_paper_pipeline`
# and as `python validation/v2/run_paper_pipeline.py`.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation.v2 import aggregate, viz  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("paper-pipeline")


METRICS_PANEL = [
    ("iclabel_brain_50pct", "Brain ICs (>50%)"),
    ("iclabel_brain_70pct", "Brain ICs (>70%)"),
    ("kurt_brain_like",     "Kurtosis brain-like ICs"),
    ("alpha_peaked",        "Alpha-peaked ICs"),
    ("mir",                 "MIR (nats, ↑ better)"),
    ("time",                "Runtime (s, ↓ better)"),
]

SUMMARY_METRICS = [
    "time", "n_iter", "iclabel_brain", "iclabel_brain_50pct",
    "iclabel_brain_70pct", "iclabel_brain_80pct", "kurt_brain_like",
    "alpha_peaked", "mir", "recon_error",
]


def _maybe_load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="validation/results")
    parser.add_argument("--out-dir", default="validation/results/figures_paper")
    parser.add_argument("--hp", type=float, default=None,
                        help="Restrict to one HP filter (e.g. 1.0). Default: all.")
    parser.add_argument("--example-subject", default="sub-01",
                        help="Subject for single-subject figures (topo grid, "
                             "IC properties, source PSD)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    viz.use_publication_style()

    # ── 1. Load and aggregate ────────────────────────────────────────
    logger.info("Loading per-subject results from %s", results_dir)
    df = aggregate.load_per_subject_results(results_dir)
    logger.info("  → %d rows, %d subjects, %d methods, hp=%s",
                len(df), df["subject"].nunique(), df["method"].nunique(),
                sorted(df["hp"].unique()))

    if args.hp is not None:
        df = df[df.hp == args.hp]
        logger.info("  Filtered to hp=%.1f → %d rows", args.hp, len(df))

    df.to_csv(out_dir / "long_dataframe.csv", index=False)

    # ── 2. Summary table ────────────────────────────────────────────
    logger.info("Building summary table")
    summary = aggregate.summary_table(df, SUMMARY_METRICS, hp=args.hp)
    summary.to_csv(out_dir / "summary_table.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))

    # ── 3. Statistics: Friedman + pairwise Wilcoxon ─────────────────
    logger.info("Running paired statistics")
    stats = {}
    for metric, _ in METRICS_PANEL:
        if metric not in df.columns:
            continue
        stats[metric] = {
            "friedman": aggregate.friedman_across_methods(df, metric, hp=args.hp),
            "pairwise": {},
        }
        methods = aggregate.METHOD_ORDER
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                key = f"{m1}_vs_{m2}"
                stats[metric]["pairwise"][key] = aggregate.paired_wilcoxon(
                    df, metric, m1, m2, hp=args.hp,
                )
    with open(out_dir / "paper_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("  Saved paper_stats.json")

    # ── 4. Cross-subject figures ────────────────────────────────────
    logger.info("Generating cross-subject figures")

    fig = viz.figure_paired_metric_panel(df, METRICS_PANEL, hp=args.hp, ncols=3)
    fig.suptitle("Per-subject metric comparison across ICA methods",
                 y=1.02, fontsize=11, fontweight="bold")
    viz.save_figure(fig, out_dir, "fig1_metric_panel")
    plt.close(fig)
    logger.info("  fig1_metric_panel ✓")

    try:
        fig = viz.figure_iclabel_stacked(df, hp=args.hp)
        viz.save_figure(fig, out_dir, "fig2_iclabel_stacked")
        plt.close(fig)
        logger.info("  fig2_iclabel_stacked ✓")
    except Exception as e:
        logger.warning("  fig2_iclabel_stacked failed: %s", e)

    try:
        fig = viz.figure_runtime_vs_quality(df, hp=args.hp)
        viz.save_figure(fig, out_dir, "fig3_runtime_vs_quality")
        plt.close(fig)
        logger.info("  fig3_runtime_vs_quality ✓")
    except Exception as e:
        logger.warning("  fig3_runtime_vs_quality failed: %s", e)

    # ── 5. Single-subject MNE-native figures (sub-01) ───────────────
    logger.info("Generating single-subject figures (%s)", args.example_subject)

    ica_amica = _maybe_load_pickle(results_dir / "ica_amica.pkl")
    ica_picard = _maybe_load_pickle(results_dir / "ica_picard.pkl")
    ica_infomax = _maybe_load_pickle(results_dir / "ica_infomax.pkl")
    amica_result = _maybe_load_pickle(results_dir / "amica_result.pkl")

    have_icas = all(x is not None for x in (ica_amica, ica_picard, ica_infomax))
    if not have_icas:
        logger.warning("Skipping single-subject figs — saved ICAs not found in %s",
                       results_dir)
    else:
        # Need raw to drive the topomap grid + IC properties
        try:
            from validation.run_highdens_validation import load_and_preprocess
            raw, _ = load_and_preprocess(args.example_subject, hp_freq=1.0)

            ica_dict = {"amica": ica_amica, "picard": ica_picard, "infomax": ica_infomax}

            # Topomap grid (uses ica.plot_components under the hood)
            try:
                fig = viz.figure_topomap_grid(ica_dict, raw, n_show=20)
                viz.save_figure(fig, out_dir, "fig4_topomap_grid")
                plt.close(fig)
                logger.info("  fig4_topomap_grid ✓")
            except Exception as e:
                logger.warning("  fig4_topomap_grid failed: %s", e)

            # Source PSD grid
            try:
                fig = viz.figure_source_psd_grid(ica_dict, raw, n_show=5)
                viz.save_figure(fig, out_dir, "fig5_source_psd")
                plt.close(fig)
                logger.info("  fig5_source_psd ✓")
            except Exception as e:
                logger.warning("  fig5_source_psd failed: %s", e)

            # IC properties for AMICA top-5
            try:
                figs = viz.figure_ic_properties(ica_amica, raw, picks=list(range(5)))
                for k, f in enumerate(figs):
                    viz.save_figure(f, out_dir, f"fig6_amica_ic{k}_properties")
                    plt.close(f)
                logger.info("  fig6_amica_ic*_properties ✓ (%d ICs)", len(figs))
            except Exception as e:
                logger.warning("  fig6_amica_ic_properties failed: %s", e)

        except Exception as e:
            logger.warning("Could not load %s: %s", args.example_subject, e)

    if amica_result is not None:
        try:
            fig = viz.figure_amica_convergence({args.example_subject: amica_result})
            viz.save_figure(fig, out_dir, "fig7_amica_convergence")
            plt.close(fig)
            logger.info("  fig7_amica_convergence ✓")
        except Exception as e:
            logger.warning("  fig7_amica_convergence failed: %s", e)

    logger.info("Done. All outputs in %s", out_dir.resolve())


if __name__ == "__main__":
    main()
