"""NumPy-based fallback for JAX functions.

This module provides NumPy implementations when JAX is not available.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

import os

# Allow forcing NumPy fallback via env var
use_jax_env = os.environ.get("AMICA_NO_JAX", "0") != "1"
HAS_JAX = False

if use_jax_env:
    try:
        import jax
        # Enable 64-bit precision by default for scientific accuracy
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        HAS_JAX = True
    except ImportError:
        pass

if not HAS_JAX:
    # Use numpy as fallback
    jnp = np
    
    # Stub for jax.jit - just return the function unchanged
    class _JaxStub:
        @staticmethod
        def jit(func: Callable = None, **kwargs) -> Callable:
            # handle @jax.jit() or @jax.jit(static_argnames=...)
            if func is None:
                def wrapper(f):
                    return f
                return wrapper
            return func
        
        @staticmethod
        def vmap(func: Callable, *args, **kwargs) -> Callable:
            """Vectorize using numpy - matches JAX vmap behavior for tuple returns."""
            def vmapped(*arrays):
                results = [func(*[a[i] for a in arrays]) for i in range(len(arrays[0]))]
                # Handle tuple returns like JAX: return tuple of stacked arrays
                if results and isinstance(results[0], tuple):
                    n_outputs = len(results[0])
                    return tuple(np.array([r[j] for r in results]) for j in range(n_outputs))
                return np.array(results)
            return vmapped
        
        class random:
            @staticmethod
            def PRNGKey(seed: int):
                return np.random.RandomState(seed)
            
            @staticmethod
            def split(key, num: int = 2):
                if hasattr(key, 'randint'):
                    seeds = [key.randint(0, 2**31) for _ in range(num)]
                    return [np.random.RandomState(s) for s in seeds]
                return [np.random.RandomState(i) for i in range(num)]
            
            @staticmethod
            def normal(key, shape):
                if hasattr(key, 'randn'):
                    return key.randn(*shape)
                return np.random.randn(*shape)
        
        class scipy:
            class special:
                @staticmethod
                def logsumexp(a, axis=None):
                    from scipy.special import logsumexp
                    return logsumexp(a, axis=axis)
        
        class lax:
            @staticmethod
            def cond(pred, true_fun, false_fun, *operands):
                if pred:
                    return true_fun(*operands)
                else:
                    return false_fun(*operands)
    
    jax = _JaxStub()

# Export
__all__ = ["jax", "jnp", "HAS_JAX"]


def get_array_module():
    """Get the appropriate array module (jax.numpy or numpy)."""
    return jnp


def ensure_numpy(x):
    """Convert array to numpy if it's a JAX array."""
    if HAS_JAX and hasattr(x, 'device'):
        return np.asarray(x)
    return np.asarray(x)


def optional_jit(func: Callable) -> Callable:
    """Decorator that applies jax.jit only if JAX is available."""
    if HAS_JAX:
        return jax.jit(func)
    return func
