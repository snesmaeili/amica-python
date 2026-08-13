"""
Pure-JAX AMICA on a NumPy array
===============================

This example shows how to configure :class:`jamica.AmicaConfig` and fit AMICA
directly on a NumPy array.

Use this interface when your data is already represented as an array, for
example in a custom pipeline, simulation, or non-MNE workflow.

The expected input shape is ``(n_channels, n_samples)``.
"""

# %%
# Imports
# -------

from __future__ import annotations

import numpy as np

from jamica import Amica, AmicaConfig

# %%
# Generate synthetic data
# -----------------------
#
# We create a simple ICA problem by mixing independent Laplacian sources with a
# random linear mixing matrix.


def make_synthetic_mixture(
    n_sources: int = 6,
    n_samples: int = 20_000,
    seed: int = 0,
) -> np.ndarray:
    """Create a random linear mixture of independent sources."""
    rng = np.random.default_rng(seed)

    sources = rng.laplace(size=(n_sources, n_samples))
    mixing = rng.standard_normal((n_sources, n_sources))

    return (mixing @ sources).astype(np.float64)


X = make_synthetic_mixture()

print(f"Data shape: {X.shape}  (channels, samples)")


# %%
# Configure AMICA
# ---------------
#
# ``num_mix_comps`` controls the number of mixture components per source.
# ``do_newton=True`` enables Newton optimization after the initial natural
# gradient phase.

config = AmicaConfig(
    max_iter=500,
    num_mix_comps=3,
    do_newton=True,
)

model = Amica(
    config,
    random_state=42,
)


# %%
# Fit the model
# -------------
#
# JAX will use an available GPU automatically when the JAX GPU package is
# installed; otherwise it runs on CPU.

result = model.fit(X)

final_ll = float(np.asarray(result.log_likelihood)[-1])

print(f"Converged in {int(result.n_iter)} iterations; final log-likelihood = {final_ll:.4f}")


# %%
# Recover sources
# ---------------
#
# The fitted model can transform the observed mixtures back into source
# activations.

estimated_sources = model.transform(X)

print(f"Recovered sources shape: {estimated_sources.shape}")


# %%
# Inspect outputs
# ---------------
#
# The ``result`` object also exposes AMICA parameters such as unmixing matrices,
# mixing matrices, source-density parameters, and, for multi-model AMICA,
# per-model weights and posterior probabilities.

print(type(result))
