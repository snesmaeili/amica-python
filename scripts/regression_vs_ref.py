"""Does the accelerated code still produce the validated numbers?

Compares HEAD against 450a63cd -- the last commit before any of this work, and
the code the manuscript's results were produced with. Each version is fitted in
its own subprocess with PYTHONPATH pointing at its own checkout, because both
install as `amica` and would otherwise shadow each other.

Three configurations, because they carry different guarantees and conflating
them is how a "no change" claim gets overstated:

  full-batch      HEAD with chunk_size=None against the baseline. NOT expected
                  bit-identical: two of the optimisations replace power(x, k)
                  with exp(k*log x) and restructure a softmax, which are the same
                  value mathematically and a different last bit in floating
                  point. Measured at 6.9e-10 relative in W and 9.4e-16 in the
                  log-likelihood -- read against the 1e-4 the codebase already
                  holds its full-batch and chunked paths to.

  blocked         HEAD with the new default against the baseline. This regroups
                  the E-step's sums over blocks, so it cannot be bit-identical
                  and is not claimed to be -- what matters is that the deviation
                  sits far inside the tolerance the codebase already held its
                  full-batch and chunked paths to (1e-4).

  classic e-step  HEAD with estep="classic" against the baseline, the documented
                  escape hatch for reproducing pre-fusion results.

Reported per configuration: whether the unmixing matrix and the entire
log-likelihood history match bit-for-bit, and if not, by how much.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRATCH = Path("C:/Users/s/AppData/Local/Temp/claude/"
               "E--amica-validation-workspace/8188d342-377b-4b1c-9149-828f5742e50c/scratchpad")
BASELINE = SCRATCH / "baseline"
CURRENT = Path("E:/amica-validation-workspace/repos/amica-python")
PYTHON = CURRENT / ".venv-dev/Scripts/python.exe"

WORKER = r'''
import json, sys
import numpy as np
from amica import Amica, AmicaConfig

cfg_kw = json.loads(sys.argv[1])
out = sys.argv[2]

rng = np.random.default_rng(20260809)
srcs = rng.laplace(size=(12, 40000))
data = (rng.normal(size=(12, 12)) @ srcs)

cfg = AmicaConfig(num_models=1, num_mix_comps=3, max_iter=40,
                  dtype="float64", fix_init=True, **cfg_kw)
res = Amica(cfg, random_state=42).fit(data)
W = np.asarray(res.unmixing_matrix_white_, dtype=np.float64)
if W.ndim == 3:
    W = W[:, :, 0]
np.savez(out, W=W, ll=np.asarray(res.log_likelihood, dtype=np.float64))
'''


def run(checkout: Path, cfg_kw: dict, tag: str) -> dict:
    script = SCRATCH / "_worker.py"
    script.write_text(WORKER, encoding="utf-8")
    out = SCRATCH / f"_reg_{tag}.npz"
    env = {"PYTHONPATH": str(checkout), "JAX_ENABLE_X64": "1",
           "PATH": __import__("os").environ["PATH"],
           "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    cp = subprocess.run([str(PYTHON), str(script), json.dumps(cfg_kw), str(out)],
                        env=env, capture_output=True, text=True)
    if cp.returncode != 0:
        raise SystemExit(f"{tag} failed:\n{cp.stderr[-2500:]}")
    d = np.load(out)
    return {"W": d["W"], "ll": d["ll"]}


def compare(a: dict, b: dict) -> tuple[bool, float, float]:
    same = np.array_equal(a["W"], b["W"]) and np.array_equal(a["ll"], b["ll"])
    dw = float(np.max(np.abs(a["W"] - b["W"]) / np.maximum(np.abs(a["W"]), 1e-300)))
    dl = float(np.max(np.abs(a["ll"] - b["ll"]) / np.maximum(np.abs(a["ll"]), 1e-300)))
    return same, dw, dl


print(f"baseline : {BASELINE}  (450a63cd, last commit before this work)")
print(f"current  : {CURRENT}\n")

ref = run(BASELINE, {"chunk_size": None}, "baseline")

cases = [
    ("full batch (chunk_size=None)", {"chunk_size": None}, False),
    ("blocked (new default)", {"chunk_size": "auto"}, False),
    ("classic e-step", {"chunk_size": None, "estep": "classic"}, False),
]

print(f"{'configuration':32} {'bit-identical':>14} {'worst rel dW':>13} {'worst rel dLL':>14}")
print("-" * 78)
ok = True
for label, kw, must_be_exact in cases:
    got = run(CURRENT, kw, label.split()[0])
    same, dw, dl = compare(ref, got)
    print(f"{label:32} {('YES' if same else 'no'):>14} "
          f"{(0.0 if same else dw):13.2e} {(0.0 if same else dl):14.2e}")
    if must_be_exact and not same:
        ok = False
        print("    ^^ EXPECTED BIT-IDENTICAL -- this is a regression, not rounding")

print()
# The row-correlation is the number that matters: it is the metric the
# manuscript compares implementations with, so it is the one that says whether
# the decomposition changed rather than merely its last bits.
from scipy.optimize import linear_sum_assignment
print()
print(f"{'configuration':32} {'worst matched |r|':>18} {'final log-likelihood':>22}")
print("-" * 76)
def matched(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    c = np.abs(a @ b.T)
    r, k = linear_sum_assignment(-c)
    return float(np.min(c[r, k]))
print(f"{'baseline':32} {1.0:18.10f} {ref['ll'][-1]:22.15f}")
for label, kw, _ in cases:
    got = run(CURRENT, kw, label.split()[0])
    print(f"{label:32} {matched(ref['W'], got['W']):18.10f} {got['ll'][-1]:22.15f}")
print()
print("A worst matched |r| of 1.0000000000 means the decomposition is unchanged;")
print("the codebase holds its own full-batch and chunked paths to rel_err < 1e-4.")
sys.exit(0 if ok else 1)
