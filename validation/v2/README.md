# validation.v2 — paper-quality metrics, viz, and aggregation

A clean, modular replacement for the per-script ad-hoc figure code that
preceded it. Designed to turn the per-subject JSONs produced by
`run_highdens_validation.py` into the full set of paper figures and
tables in one shot.

## Modules

| Module | Purpose |
|--------|---------|
| `metrics.py` | Per-(method, subject) metric library: ICLabel, kurtosis, PSD (alpha-peak + 1/f slope), MIR, reconstruction error, optional dipolarity. Each metric is pure (no I/O), JSON-serialisable, and fault-tolerant via the `compute_all` orchestrator. |
| `aggregate.py` | Loads `benchmark_sub-*_hp*hz.json` files into a long-format DataFrame, supports paired Wilcoxon, Friedman χ², and Cliff's δ effect sizes. |
| `viz.py` | Publication-ready figures (300 dpi PNG + vector PDF). Uses MNE-native ICA viz tools (`plot_components`, `plot_properties`, `plot_sources`) for single-subject panels, plus cross-subject paired boxplots, ICLabel stacked bars, and a runtime-vs-quality scatter. |
| `run_paper_pipeline.py` | One-shot orchestrator. Reads JSONs (+ optional pickled ICAs), generates every figure, writes the summary table and `paper_stats.json`. |

## Usage

```bash
# After the validation job array has finished:
python validation/v2/run_paper_pipeline.py \
    --results-dir validation/results \
    --out-dir validation/results/figures_paper \
    --hp 1.0 \
    --example-subject sub-01
```

Outputs in `validation/results/figures_paper/`:

- `fig1_metric_panel.{png,pdf}` — six paired boxplots (brain ICs >50%, >70%,
  kurt-brain-like, alpha-peaked, MIR, runtime), one dot per subject
- `fig2_iclabel_stacked.{png,pdf}` — mean ICLabel composition per method
- `fig3_runtime_vs_quality.{png,pdf}` — runtime vs brain-IC scatter, one
  cluster per method, group centroids in bold
- `fig4_topomap_grid.{png,pdf}` — 4 × 20 grid of MNE topomaps coloured by
  ICLabel category (single example subject)
- `fig5_source_psd.{png,pdf}` — first 5 source PSDs per method
- `fig6_amica_ic*_properties.{png,pdf}` — MNE 4-panel `plot_properties`
  for the top AMICA ICs (single example subject)
- `fig7_amica_convergence.{png,pdf}` — log-likelihood curve(s)
- `summary_table.csv` — mean ± SD across subjects per method per metric
- `paper_stats.json` — Friedman + pairwise Wilcoxon + Cliff's δ
- `long_dataframe.csv` — full long-format DataFrame for ad hoc analysis

## Hooking into a fresh run

`metrics.compute_all(ica, raw)` is a drop-in replacement for the metric
block in `run_highdens_validation.py`. To use the v2 metric schema in a
new run, replace the `_run_metrics` body with:

```python
from validation.v2 import metrics
result = metrics.compute_all(ica, raw)
result["time"] = dt
```

The aggregator is schema-tolerant — it reads both v1 (`iclabel: {brain: 4,
…}`) and v2 (`iclabel: {counts: {brain: 4, …}, labels: [...], probs: [...]}`)
shapes.

## Dipolarity

`metrics.compute_dipolarity` is implemented but **opt-in** (not in
`METRICS` by default) because building a forward solution per subject is
expensive. Enable it with:

```python
metrics.compute_all(ica, raw, include=("iclabel", "kurtosis", "mir", "dipolarity"))
```

It returns NaN-filled metadata if `fsaverage` BEM/forward isn't
available, so it never breaks a long ICA run.
