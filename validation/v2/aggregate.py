"""Cross-subject aggregation, paired statistics, and DataFrame helpers.

The validation pipeline writes one JSON per (subject, hp_filter) at
``validation/results/benchmark_<subject>_hp<hp>hz.json``. This module
loads them, melts to long format, and exposes paired statistical tests
across methods.
"""
from __future__ import annotations

import glob
import json
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METHOD_ORDER = ["amica", "picard", "infomax", "fastica"]


def _flatten_metrics(method_result: dict) -> dict:
    """Flatten one method's nested metric dict into scalar columns."""
    flat = {
        "time": method_result.get("time", float("nan")),
        "n_iter": method_result.get("n_iter", float("nan")),
        "n_components": method_result.get("n_components", float("nan")),
        "recon_error": (
            method_result.get("reconstruction", {}).get("relative_error")
            if isinstance(method_result.get("reconstruction"), dict)
            else method_result.get("reconstruction_error", float("nan"))
        ),
        "mir": (
            method_result.get("mir", {}).get("mir")
            if isinstance(method_result.get("mir"), dict)
            else float("nan")
        ),
        "alpha_peaked": (
            method_result.get("psd", {}).get("alpha_peaked_ics")
            if "psd" in method_result
            else method_result.get("psd_alpha", {}).get("alpha_peaked_ics", float("nan"))
        ),
    }
    # ICLabel — supports both v1 (flat dict) and v2 (counts subkey)
    ic = method_result.get("iclabel", {})
    if isinstance(ic, dict):
        counts = ic.get("counts", ic)
        for k in ("brain", "muscle", "eye", "heart", "line_noise",
                  "channel_noise", "other", "brain_50pct", "brain_70pct",
                  "brain_80pct"):
            flat[f"iclabel_{k}"] = counts.get(k, float("nan"))
    # Kurtosis
    kurt = method_result.get("kurtosis", {})
    if isinstance(kurt, dict):
        flat["kurt_brain_like"] = kurt.get("brain_like", kurt.get("brain_like_kurtosis", float("nan")))
        flat["kurt_mean"] = kurt.get("mean", kurt.get("kurtosis_mean", float("nan")))
        flat["kurt_median"] = kurt.get("median", kurt.get("kurtosis_median", float("nan")))
    # Dipolarity (optional)
    dip = method_result.get("dipolarity", {})
    if isinstance(dip, dict):
        flat["dipolar_15pct"] = dip.get("near_dipolar_15pct", float("nan"))
        flat["rv_median"] = dip.get("median_rv", float("nan"))
    return flat


def load_per_subject_results(
    results_dir: Path | str = "validation/results",
    pattern: str = "benchmark_sub-*_hp*hz.json",
) -> pd.DataFrame:
    """Load per-(subject, hp) JSONs into one long-format DataFrame.

    Each row is one (subject, hp_filter, method) combination.
    """
    results_dir = Path(results_dir)
    rows = []
    files = sorted(results_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {results_dir}")

    fname_re = re.compile(r"benchmark_(?P<subject>sub-\d+)_hp(?P<hp>[\d.]+)hz\.json")
    for fp in files:
        m = fname_re.match(fp.name)
        if not m:
            logger.warning("Skipping unrecognised filename %s", fp.name)
            continue
        subject = m["subject"]
        hp = float(m["hp"])
        with open(fp) as f:
            sweep = json.load(f)
        for method, res in sweep.items():
            if not isinstance(res, dict) or "error" in res:
                rows.append({
                    "subject": subject,
                    "hp": hp,
                    "method": method,
                    "error": res if isinstance(res, str) else res.get("error", "unknown"),
                })
                continue
            flat = _flatten_metrics(res)
            flat.update(subject=subject, hp=hp, method=method)
            rows.append(flat)

    df = pd.DataFrame(rows)
    # Stable column order
    front = ["subject", "hp", "method"]
    other = [c for c in df.columns if c not in front]
    df = df[front + sorted(other)]
    return df


# ───────────────────────────────────────────────────────────────────────
# Statistics
# ───────────────────────────────────────────────────────────────────────


def paired_wilcoxon(
    df: pd.DataFrame, metric: str, m1: str, m2: str, hp: float | None = None
) -> dict:
    """Wilcoxon signed-rank test paired by subject between two methods.

    Returns the W statistic, p-value, n, median diff, and Cliff's delta.
    """
    from scipy.stats import wilcoxon

    sub = df.dropna(subset=[metric])
    if hp is not None:
        sub = sub[sub.hp == hp]
    pivot = sub.pivot_table(index="subject", columns="method", values=metric)
    pivot = pivot.dropna(subset=[m1, m2])
    a = pivot[m1].to_numpy()
    b = pivot[m2].to_numpy()
    if len(a) < 3:
        return {"n": int(len(a)), "p": float("nan"), "stat": float("nan")}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    diff = a - b
    return {
        "n": int(len(a)),
        "stat": float(stat),
        "p": float(p),
        "median_diff": float(np.median(diff)),
        "mean_diff": float(np.mean(diff)),
        "cliffs_delta": _cliffs_delta(a, b),
    }


def friedman_across_methods(
    df: pd.DataFrame, metric: str, methods: Iterable[str] | None = None,
    hp: float | None = None,
) -> dict:
    """Friedman χ² across multiple methods, paired by subject."""
    from scipy.stats import friedmanchisquare

    sub = df.dropna(subset=[metric])
    if hp is not None:
        sub = sub[sub.hp == hp]
    methods = list(methods) if methods is not None else METHOD_ORDER
    pivot = sub.pivot_table(index="subject", columns="method", values=metric)
    pivot = pivot.dropna(subset=methods)
    arrays = [pivot[m].to_numpy() for m in methods]
    if any(len(a) < 3 for a in arrays):
        return {"n": int(len(pivot)), "p": float("nan")}
    stat, p = friedmanchisquare(*arrays)
    return {
        "n": int(len(pivot)),
        "stat": float(stat),
        "p": float(p),
        "methods": methods,
    }


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta — non-parametric effect size in [-1, 1]."""
    a = np.asarray(a); b = np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (n_a * n_b))


def summary_table(
    df: pd.DataFrame, metrics: Iterable[str], hp: float | None = None
) -> pd.DataFrame:
    """Wide summary: one row per method × metric, mean ± SD across subjects."""
    sub = df.copy()
    if hp is not None:
        sub = sub[sub.hp == hp]
    rows = []
    for method, g in sub.groupby("method"):
        row = {"method": method, "n_subjects": int(g["subject"].nunique())}
        for m in metrics:
            if m not in g.columns:
                continue
            vals = g[m].dropna().to_numpy()
            if len(vals) == 0:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_sd"] = float("nan")
            else:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_order"] = out["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)
