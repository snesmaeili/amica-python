"""
Generate LaTeX Tables from Validation Results
==============================================

Reads JSON result files and produces LaTeX tables suitable for
inclusion in an Overleaf manuscript.

Generates:
1. Table 1: MATLAB Parity — LL difference and W correlation
2. Table 2: Method Comparison — synthetic Amari index and runtime
3. Table 3: Real EEG (ds004505) — ICLabel, MIR, kurtosis, alpha peaks
4. Table 4: Rejection Ablation — LL, kurtosis, runtime vs settings
5. Table 5: Stress Test Summary — edge case robustness
"""
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR = Path(__file__).parent / "tables"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_json(name):
    """Load a JSON results file."""
    path = RESULTS_DIR / name
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return None
    with open(path) as f:
        return json.load(f)


def table_matlab_parity():
    """Table 1: MATLAB parity results."""
    data = load_json("matlab_parity_results.json")
    if data is None:
        return

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{MATLAB AMICA 1.7 parity. LL = log-likelihood, $\rho_W$ = mean unmixing matrix correlation (permutation-matched).}",
        r"\label{tab:matlab-parity}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Configuration & LL diff (\%) & $\rho_W$ (mean) & $\rho_W$ (min) \\",
        r"\midrule",
    ]

    for name, res in data.items():
        if "error" in res:
            lines.append(f"  {name} & \\multicolumn{{3}}{{c}}{{FAILED}} \\\\")
            continue
        ll_pct = res.get("ll_diff_pct", float("nan"))
        w_mean = res.get("W_mean_corr", float("nan"))
        w_min = res.get("W_min_corr", float("nan"))
        lines.append(f"  {name} & {ll_pct:.4f} & {w_mean:.6f} & {w_min:.6f} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = OUTPUT_DIR / "table_matlab_parity.tex"
    out.write_text("\n".join(lines))
    print(f"  Wrote {out}")


def table_method_comparison():
    """Table 2: Synthetic source separation comparison."""
    data = load_json("validation_results.json")
    if data is None:
        return

    synth = data.get("synthetic_benchmark", {}).get("algorithms", {})
    if not synth:
        print("  WARNING: No synthetic_benchmark in validation_results.json")
        return

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Synthetic source separation: 6 sources, 10\,000 samples. Lower Amari index = better separation.}",
        r"\label{tab:synthetic}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Amari index & Time (s) & Iterations \\",
        r"\midrule",
    ]

    for method, res in synth.items():
        amari = res.get("amari", float("nan"))
        t = res.get("time", float("nan"))
        n_iter = res.get("n_iter", "---")
        lines.append(f"  {method} & {amari:.4f} & {t:.2f} & {n_iter} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = OUTPUT_DIR / "table_synthetic.tex"
    out.write_text("\n".join(lines))
    print(f"  Wrote {out}")


def table_highdens():
    """Table 3: High-density EEG validation on ds004505."""
    data = load_json("highdens_validation.json")
    if data is None:
        return

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Real EEG validation on ds004505 (120-channel mobile EEG, sub-01). Brain = ICLabel brain class, $\alpha$ = ICs with alpha-band spectral peak.}",
        r"\label{tab:highdens}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Time (s) & Iter & Brain ICs & $\alpha$ ICs & MIR & Recon err \\",
        r"\midrule",
    ]

    for method in ["amica", "picard", "infomax", "fastica"]:
        r = data.get(method, {})
        if "error" in r:
            lines.append(f"  {method} & \\multicolumn{{6}}{{c}}{{FAILED}} \\\\")
            continue

        ic = r.get("iclabel", {})
        psd = r.get("psd_alpha", {})
        mir = r.get("mir", {})
        err = r.get("reconstruction_error", float("nan"))

        brain = ic.get("brain", "?")
        alpha = psd.get("alpha_peaked_ics", "?")
        mir_val = f"{mir['mir']:.1f}" if isinstance(mir, dict) and "mir" in mir else "?"

        lines.append(
            f"  {method} & {r.get('time', 0):.0f} & {r.get('n_iter', 0)} "
            f"& {brain} & {alpha} & {mir_val} & {err:.1e} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = OUTPUT_DIR / "table_highdens.tex"
    out.write_text("\n".join(lines))
    print(f"  Wrote {out}")


def table_rejection_ablation():
    """Table 4: Rejection ablation results."""
    data = load_json("rejection_ablation_results.json")
    if data is None:
        return

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Sample rejection ablation (Klug et al.\ 2024). numrej = rejection passes, $\sigma$ = threshold in SD.}",
        r"\label{tab:rejection}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Config & Final LL & Kurt mean & Recon err & Time (s) \\",
        r"\midrule",
    ]

    for r in data:
        lines.append(
            f"  {r['label']} & {r['final_ll']:.4f} "
            f"& {r['kurtosis_mean']:.2f} "
            f"& {r['reconstruction_error']:.1e} "
            f"& {r['time']:.1f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = OUTPUT_DIR / "table_rejection.tex"
    out.write_text("\n".join(lines))
    print(f"  Wrote {out}")


def table_stress():
    """Table 5: Stress test results."""
    data = load_json("stress_test_results.json")
    if data is None:
        return

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Solver robustness under edge conditions. All tests should produce finite results without crashes.}",
        r"\label{tab:stress}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Test & Status & Final LL & NRMSE \\",
        r"\midrule",
    ]

    for name, r in data.items():
        status = r["status"]
        if status == "OK":
            ll = f"{r['final_ll']:.4f}" if not r.get("has_nan") else "NaN"
            nrmse = f"{r['nrmse']:.1e}"
        else:
            ll = "---"
            nrmse = "---"
        lines.append(f"  {name.replace('_', ' ')} & {status} & {ll} & {nrmse} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    out = OUTPUT_DIR / "table_stress.tex"
    out.write_text("\n".join(lines))
    print(f"  Wrote {out}")


def main():
    print("Generating LaTeX tables from validation results...\n")

    table_matlab_parity()
    table_method_comparison()
    table_highdens()
    table_rejection_ablation()
    table_stress()

    print(f"\nAll tables written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
