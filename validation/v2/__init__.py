"""validation.v2 — modular metrics, viz, and aggregation for the amica-python paper.

Modules
-------
metrics      : Per-(method, subject) metric library (ICLabel, kurtosis, MIR,
               PSD, reconstruction error, dipolarity).
viz          : Publication-ready figures using MNE-native ICA tools and
               cross-subject aggregate plots.
aggregate    : Load per-subject JSONs, build long-format DataFrame, run
               paired Wilcoxon / Friedman / effect-size statistics.
run_paper_pipeline : Orchestrator that turns benchmark JSONs + (optional) ICA
                     pickles into the full set of paper figures and tables.
"""
