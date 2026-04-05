"""
MATLAB Parity Test for amica-python
================================

Runs AMICA on synthetic data using both:
1. amica-python (Python/JAX)
2. MATLAB AMICA 1.7 (Fortran reference via MATLAB Engine)

Compares: log-likelihood curves, unmixing matrices, source density params.

Requirements:
- Python 3.11 venv with matlab.engine installed
- EEGLAB with AMICA 1.7 plugin at E:/PhD/eeglab2022.1
"""
import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_python_amica(data, max_iter=200, num_mix=3, do_newton=True, seed=42):
    """Run amica-python on data."""
    from amica_python import Amica, AmicaConfig

    config = AmicaConfig(
        max_iter=max_iter,
        num_mix_comps=num_mix,
        do_newton=do_newton,
        do_reject=False,
        rho0=1.5,
        minrho=1.0,
        maxrho=2.0,
    )
    model = Amica(config, random_state=seed)
    result = model.fit(data)
    return result


def run_matlab_amica(data, max_iter=200, num_mix=3, do_newton=1, seed=42):
    """Run MATLAB AMICA 1.7 on data via MATLAB Engine."""
    import matlab.engine

    eng = matlab.engine.start_matlab()

    # Add EEGLAB and AMICA to path
    eng.addpath(r'E:\PhD\eeglab2022.1', nargout=0)
    eng.eval("eeglab nogui;", nargout=0)

    tmpdir = tempfile.mkdtemp(prefix="amica_parity_")
    outdir = os.path.join(tmpdir, "amicaout")

    n_channels, n_samples = data.shape

    # Pass data as MATLAB matrix directly (channels x samples)
    import matlab
    data_matlab = matlab.double(data.tolist())

    eng.workspace['data_mat'] = data_matlab
    eng.workspace['outdir'] = outdir
    eng.workspace['max_iter'] = float(max_iter)
    eng.workspace['num_mix'] = float(num_mix)
    eng.workspace['do_newton'] = float(do_newton)

    matlab_cmd = """
    runamica15(data_mat, ...
        'outdir', outdir, ...
        'max_iter', max_iter, ...
        'num_mix_comps', num_mix, ...
        'do_newton', do_newton, ...
        'do_reject', 0, ...
        'minrho', 1.0, ...
        'maxrho', 2.0, ...
        'rho0', 1.5, ...
        'do_sphere', 1, ...
        'doscaling', 1, ...
        'max_threads', 4);
    """

    print("  Running MATLAB AMICA...")
    try:
        eng.eval(matlab_cmd, nargout=0)
    except Exception as e:
        print(f"  MATLAB AMICA failed: {e}")
        eng.quit()
        return None

    # Read results
    result = {}

    # Read W (unmixing matrix)
    W_file = os.path.join(outdir, "W")
    if os.path.exists(W_file):
        W = np.fromfile(W_file, dtype='<f8').reshape(n_channels, n_channels, order='F')
        result['W'] = W

    # Read A (mixing matrix)
    A_file = os.path.join(outdir, "A")
    if os.path.exists(A_file):
        A = np.fromfile(A_file, dtype='<f8').reshape(n_channels, n_channels, order='F')
        result['A'] = A

    # Read S (sphere)
    S_file = os.path.join(outdir, "S")
    if os.path.exists(S_file):
        S = np.fromfile(S_file, dtype='<f8').reshape(n_channels, n_channels, order='F')
        result['S'] = S

    # Read LL
    LL_file = os.path.join(outdir, "LL")
    if os.path.exists(LL_file):
        LL = np.fromfile(LL_file, dtype='<f8')
        result['LL'] = LL

    # Read mu, alpha, sbeta, rho
    for param in ['mu', 'alpha', 'sbeta', 'rho']:
        f = os.path.join(outdir, param)
        if os.path.exists(f):
            result[param] = np.fromfile(f, dtype='<f8').reshape(num_mix, n_channels, order='F')

    # Read c (model center)
    c_file = os.path.join(outdir, "c")
    if os.path.exists(c_file):
        result['c'] = np.fromfile(c_file, dtype='<f8')

    result['outdir'] = outdir
    eng.quit()
    return result


def compare_results(py_result, mat_result, n_channels):
    """Compare Python and MATLAB AMICA results."""
    comparison = {}

    # 1. Log-likelihood comparison
    py_ll = py_result.log_likelihood
    mat_ll = mat_result.get('LL', np.array([]))

    if len(mat_ll) > 0:
        # MATLAB LL may have different normalization
        # Compare final values and relative difference
        py_final = py_ll[-1]
        mat_final = mat_ll[-1] if len(mat_ll) > 0 else float('nan')
        comparison['ll_python_final'] = float(py_final)
        comparison['ll_matlab_final'] = float(mat_final)
        comparison['ll_diff_pct'] = float(abs(py_final - mat_final) / abs(mat_final) * 100) if mat_final != 0 else float('nan')
        print(f"    LL Python: {py_final:.6f}")
        print(f"    LL MATLAB: {mat_final:.6f}")
        print(f"    LL diff:   {comparison['ll_diff_pct']:.2f}%")

    # 2. Unmixing matrix comparison (correlation-based, permutation-invariant)
    W_py = py_result.unmixing_matrix  # (n, n) in whitened space
    W_mat = mat_result.get('W', None)

    if W_mat is not None:
        # Compute full unmixing: W @ sphere
        W_py_full = W_py @ py_result.whitener_
        S_mat = mat_result.get('S', np.eye(n_channels))
        W_mat_full = W_mat @ S_mat

        # Correlation matrix between rows (sources)
        corr = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                c = np.corrcoef(W_py_full[i], W_mat_full[j])[0, 1]
                corr[i, j] = abs(c)

        # Best matching (greedy)
        matched_corrs = []
        used_py = set()
        used_mat = set()
        for _ in range(n_channels):
            best_val = -1
            best_i, best_j = 0, 0
            for i in range(n_channels):
                if i in used_py:
                    continue
                for j in range(n_channels):
                    if j in used_mat:
                        continue
                    if corr[i, j] > best_val:
                        best_val = corr[i, j]
                        best_i, best_j = i, j
            matched_corrs.append(best_val)
            used_py.add(best_i)
            used_mat.add(best_j)

        comparison['W_mean_corr'] = float(np.mean(matched_corrs))
        comparison['W_min_corr'] = float(np.min(matched_corrs))
        comparison['W_matched_corrs'] = [float(c) for c in matched_corrs]
        print(f"    W correlation (mean): {comparison['W_mean_corr']:.4f}")
        print(f"    W correlation (min):  {comparison['W_min_corr']:.4f}")

    # 3. Source density parameters
    for param in ['rho', 'alpha']:
        py_val = getattr(py_result, f'{param}_', None)
        mat_val = mat_result.get(param, None)
        if py_val is not None and mat_val is not None:
            # Mean absolute difference (shape may differ in ordering)
            diff = float(np.mean(np.abs(np.sort(py_val.ravel()) - np.sort(mat_val.ravel()))))
            comparison[f'{param}_mean_diff'] = diff
            print(f"    {param} mean diff: {diff:.4f}")

    return comparison


def main():
    print("=" * 60)
    print("MATLAB PARITY TEST")
    print("=" * 60)

    rng = np.random.RandomState(42)

    # Test configurations
    configs = [
        {"name": "M1_m3_newton", "max_iter": 200, "num_mix": 3, "do_newton": True},
        {"name": "M1_m3_natgrad", "max_iter": 200, "num_mix": 3, "do_newton": False},
        {"name": "M1_m1_newton", "max_iter": 200, "num_mix": 1, "do_newton": True},
    ]

    # Generate synthetic data
    n_channels, n_samples = 6, 5000
    S = rng.laplace(size=(n_channels, n_samples))
    A_true = rng.randn(n_channels, n_channels)
    data = A_true @ S

    all_results = {}

    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'-' * 60}")
        print(f"Config: {name}")
        print(f"  max_iter={cfg['max_iter']}, num_mix={cfg['num_mix']}, newton={cfg['do_newton']}")
        print(f"{'-' * 60}")

        # Python
        print("\n  [Python] Running amica-python...")
        py_result = run_python_amica(
            data, max_iter=cfg['max_iter'], num_mix=cfg['num_mix'],
            do_newton=cfg['do_newton']
        )
        print(f"  [Python] Final LL: {py_result.log_likelihood[-1]:.6f}")

        # MATLAB
        mat_result = run_matlab_amica(
            data, max_iter=cfg['max_iter'], num_mix=cfg['num_mix'],
            do_newton=1 if cfg['do_newton'] else 0
        )

        if mat_result is not None:
            print("\n  Comparing...")
            comparison = compare_results(py_result, mat_result, n_channels)
            all_results[name] = comparison
        else:
            print("  MATLAB run failed, skipping comparison.")
            all_results[name] = {"error": "MATLAB AMICA failed"}

    # Save results
    output_file = RESULTS_DIR / "matlab_parity_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("PARITY SUMMARY")
    print("=" * 60)
    for name, res in all_results.items():
        if "error" in res:
            print(f"  {name}: FAILED ({res['error']})")
        else:
            ll_pct = res.get('ll_diff_pct', float('nan'))
            w_corr = res.get('W_mean_corr', float('nan'))
            status = "PASS" if ll_pct < 5.0 and w_corr > 0.9 else "CHECK"
            print(f"  {name}: LL diff={ll_pct:.2f}%, W corr={w_corr:.4f} [{status}]")


if __name__ == "__main__":
    main()
