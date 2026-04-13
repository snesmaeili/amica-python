"""Quick test: fix_init=True vs random init on sub-01 (10 iters each)."""
import numpy as np, time, pickle
from amica_python import Amica, AmicaConfig

z = np.load('validation/results/post_f1_audit/sub01_preproc.npz', allow_pickle=True)
data = z['data']; n_comp = int(z['n_components'])

for fix in [True, False]:
    for seed in [42, 0, 7, 123]:
        cfg = AmicaConfig(num_models=1, num_mix_comps=3, max_iter=20, pcakeep=n_comp,
                          dtype="float64", fix_init=fix)
        t0 = time.time()
        res = Amica(cfg, random_state=seed).fit(data)
        ll = np.asarray(res.log_likelihood)
        rho = np.asarray(res.rho_)
        n_floor = int(np.sum(np.isclose(rho, 1.0, atol=1e-6)))
        n_dec = int(np.sum(np.diff(ll) < 0))
        print(f"fix_init={fix!s:5s} seed={seed:3d}  LL[0]={ll[0]:8.3f}  LL[-1]={ll[-1]:8.3f}  "
              f"n_dec={n_dec:2d}/{len(ll)-1}  rho_floor={n_floor}/{rho.size}  ({time.time()-t0:.1f}s)")
        for i in range(min(10, len(ll))):
            d = ll[i]-ll[i-1] if i > 0 else 0
            flag = " <<<" if d < -0.1 and i > 0 else ""
            print(f"    {i:2d}: LL={ll[i]:10.4f}  dll={d:+.4e}{flag}")
        print()
