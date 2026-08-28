`jamica.amica` is now a stable, single-model solver boundary for ICA frameworks
that already preprocess their data. With `whiten=False` it disables JAMICA
centering, sphering, and PCA, composes any internal scalar rescaling into the
returned unmixing matrix, and guarantees `Y == W @ X`. The function no longer
accepts arbitrary configuration keywords, so callers cannot enable multiple
models or hidden preprocessing through adapter parameters. It accepts both
NumPy random-state APIs, reports attempted iterations accurately, and warns
with `JamicaConvergenceWarning` when a finite last iterate has not converged.
