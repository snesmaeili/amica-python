# AMICA benchmark report

Runs: 12  •  Groups: 4  •  max_iter per run: 20

Averaged over seeds (mean ± std).


## Per-configuration summary

| dataset | device | chunk | shape | n_comp | seeds | wall (s) | JIT (s) | steady ms/iter | peak RSS (GB) | peak GPU (GB) | LL final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mne | cpu | full | 30x166800 | 30 | 3 | 18.9 ± 0.6 | 1.9 | 839.4 ± 18.6 | 1.53 | — | -1.4075 ± 0.000027 |
| mne | gpu | full | 30x166800 | 30 | 3 | 3.9 ± 2.4 | 1.5 | 6.9 ± 0.0 | 2.03 | 0.71 | -1.4075 ± 0.000027 |
| mobi | cpu | 1024 | 118x1206350 | 60 | 3 | 269.8 ± 3.3 | 16.0 | 13240.1 ± 184.8 | 4.13 | — | 4.5750 ± 3.094345 |
| mobi | gpu | full | 118x1206350 | 60 | 3 | 4.4 ± 1.6 | 1.5 | 47.3 ± 0.3 | 3.56 | 7.03 | 5.0291 ± 2.363577 |

## Seed-level LL final (reproducibility check)

| dataset | device | chunk | seed=0 | seed=1 | seed=2 | max|Δ| |
|---|---|---|---|---|---|---|
| mne | cpu | full | -1.407494 | -1.407456 | -1.407521 | 6.49e-05 |
| mne | gpu | full | -1.407494 | -1.407456 | -1.407521 | 6.50e-05 |
| mobi | cpu | 1024 | 0.198971 | 6.762609 | 6.763529 | 6.56e+00 |
| mobi | gpu | full | 1.686586 | 6.684232 | 6.716591 | 5.03e+00 |

## CPU vs GPU speedup (steady per-iter)

| dataset | CPU ms/iter | GPU ms/iter | speedup |
|---|---|---|---|
| mne | 839.4 | 6.9 | 120.8x |
| mobi | 13240.1 | 47.3 | 280.0x |
