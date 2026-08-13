"""NumPy-based fallback for JAX functions.

This module provides NumPy implementations when JAX is not available.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

# Allow forcing NumPy fallback via env var
use_jax_env = os.environ.get("AMICA_NO_JAX", "0") != "1"
HAS_JAX = False

# Enable XLA's Triton GEMM emitter (GPU only; harmless/no-op on CPU). On a GPU
# this can speed up the per-iteration matmuls (W @ data, g @ y.T) by ~5-15%.
# Must be set BEFORE jax/XLA initialize. User-overridable: if XLA_FLAGS already
# mentions triton_gemm we leave it alone.
if use_jax_env:
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    if "triton_gemm" not in _xla_flags:
        os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_triton_gemm_any=True").strip()

# Optional GPU scheduling flag (Stage 3F, opt-in until benchmark-proven on the
# target stack). Set AMICA_XLA_SCHED=1 to enable XLA's latency-hiding scheduler,
# which overlaps compute with memory transfers on the per-iteration dense flows.
# GPU-only (no-op on CPU), scheduling-only (does not change emitted arithmetic),
# and override-safe (skipped if already present in XLA_FLAGS). Default OFF so the
# reference behavior is unchanged; flip to default-on only after the H100 number.
if use_jax_env and os.environ.get("AMICA_XLA_SCHED", "0") == "1":
    _xla_flags = os.environ.get("XLA_FLAGS", "")
    if "latency_hiding_scheduler" not in _xla_flags:
        os.environ["XLA_FLAGS"] = (
            _xla_flags + " --xla_gpu_enable_latency_hiding_scheduler=true"
        ).strip()

if use_jax_env:
    try:
        import jax

        # Enable 64-bit precision by default for scientific accuracy.
        # WARNING: This sets the global JAX context. If `jamica` is used
        # alongside other JAX packages expecting float32, this may cause conflicts.
        jax.config.update("jax_enable_x64", True)

        # Configure persistent compilation cache when the target is writable.
        _cache_dir_env = os.environ.get("JAX_COMPILATION_CACHE_DIR")
        _cache_dir: str | None = (
            _cache_dir_env
            if _cache_dir_env
            else os.path.join(os.path.expanduser("~"), ".cache", "jamica", "jax_cache")
        )
        try:
            assert _cache_dir is not None
            os.makedirs(_cache_dir, exist_ok=True)
        except OSError:
            _cache_dir = None
        if _cache_dir is not None:
            jax.config.update("jax_compilation_cache_dir", _cache_dir)
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)

        import jax.numpy as jnp

        HAS_JAX = True
    except ImportError:
        pass

if not HAS_JAX:
    # Use numpy as fallback
    jnp = np

    # Stub for jax.jit - just return the function unchanged
    class _JaxStub:
        """Dummy JAX module mirroring the subset of the JAX API used by AMICA.

        This allows the rest of the codebase to call `jax.vmap`, `jax.random`, etc.,
        without needing constant `if HAS_JAX:` checks everywhere.
        """

        @staticmethod
        def jit(func: Callable | None = None, **kwargs) -> Callable:
            """Stub for jax.jit. Returns the function uncompiled."""
            # handle @jax.jit() or @jax.jit(static_argnames=...)
            if func is None:

                def wrapper(f):
                    return f

                return wrapper
            return func

        @staticmethod
        def vmap(func: Callable, *args, **kwargs) -> Callable:
            """Vectorize using numpy - matches JAX vmap behavior for tuple returns.

            WARNING: This stub uses a pure Python loop over the first dimension.
            Unlike true `jax.vmap`, this does not actually vectorize operations
            in C/C++ and will be significantly slower for large batch dimensions.
            """
            in_axes = kwargs.get("in_axes", 0)

            def vmapped(*arrays):
                axes = (in_axes,) * len(arrays) if isinstance(in_axes, int) else in_axes

                # Find length of mapped axis
                n = 0
                for a, ax in zip(arrays, axes, strict=True):
                    if ax is not None:
                        n = a.shape[ax]
                        break

                results = []
                for i in range(n):
                    args_i = []
                    for a, ax in zip(arrays, axes, strict=True):
                        if ax is None:
                            args_i.append(a)
                        else:
                            args_i.append(np.take(a, i, axis=ax))
                    results.append(func(*args_i))

                def stack_tree(items):
                    first = items[0]
                    if isinstance(first, tuple):
                        return tuple(
                            stack_tree([item[j] for item in items]) for j in range(len(first))
                        )
                    if isinstance(first, list):
                        return [stack_tree([item[j] for item in items]) for j in range(len(first))]
                    return np.array(items)

                return stack_tree(results)

            return vmapped

        class random:
            """Stub for jax.random using numpy.random."""

            @staticmethod
            def PRNGKey(seed: int):
                """Create a numpy RandomState acting as a JAX PRNGKey."""
                return np.random.RandomState(seed)

            @staticmethod
            def split(key, num: int = 2):
                if hasattr(key, "randint"):
                    seeds = [key.randint(0, 2**31) for _ in range(num)]
                    return [np.random.RandomState(s) for s in seeds]
                return [np.random.RandomState(i) for i in range(num)]

            @staticmethod
            def normal(key, shape):
                if hasattr(key, "randn"):
                    return key.randn(*shape)
                return np.random.randn(*shape)

        class scipy:
            """Stub for jax.scipy."""

            class special:
                @staticmethod
                def logsumexp(a, axis=None):
                    """Stub for jax.scipy.special.logsumexp using scipy.special."""
                    from scipy.special import logsumexp

                    return logsumexp(a, axis=axis)

        class lax:
            @staticmethod
            def cond(pred, true_fun, false_fun, *operands):
                """Stub for jax.lax.cond.

                WARNING: Evaluates `pred` as a Python bool. In actual JAX, if `pred`
                is a traced array, calling `bool()` will force synchronization and
                defeat JIT compilation. Safe here only because this is the non-JIT fallback.
                """
                if pred:
                    return true_fun(*operands)
                return false_fun(*operands)

    jax = _JaxStub()

# Export
__all__ = ["HAS_JAX", "jax", "jnp"]


def get_array_module():
    """Get the appropriate array module (jax.numpy or numpy).

    Returns
    -------
    module : module
        The `jnp` or `np` module depending on JAX availability.
    """
    return jnp


def ensure_numpy(x):
    """Convert array to numpy if it's a JAX array.

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    numpy_array : np.ndarray
        NumPy version of the input array.
    """
    if HAS_JAX and hasattr(x, "device"):
        return np.asarray(x)
    return np.asarray(x)


def optional_jit(func: Callable) -> Callable:
    """Decorator that applies jax.jit only if JAX is available.

    Parameters
    ----------
    func : Callable
        Function to potentially JIT compile.

    Returns
    -------
    wrapped : Callable
        JIT-compiled function if JAX is available, otherwise the original function.
    """
    if HAS_JAX:
        return jax.jit(func)
    return func
