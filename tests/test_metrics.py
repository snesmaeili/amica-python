"""Tests for AMICA component metrics."""
import unittest
import numpy as np


class TestMetrics(unittest.TestCase):
    """Test per-component metrics from AMICA mixture model."""

    @classmethod
    def setUpClass(cls):
        """Fit a small AMICA model once for all tests."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(42)
        n_channels, n_samples = 4, 2000
        S = rng.laplace(size=(n_channels, n_samples))
        A = rng.randn(n_channels, n_channels)
        cls.data = A @ S

        config = AmicaConfig(max_iter=50, num_mix_comps=3, do_newton=False)
        model = Amica(config=config, random_state=42)
        cls.result = model.fit(cls.data)

    def test_rho_mean_shape_and_range(self):
        from amica_python.metrics import rho_mean
        rm = rho_mean(self.result)
        self.assertEqual(rm.shape, (4,))
        # rho should be between minrho (1.0) and maxrho (2.0)
        self.assertTrue(np.all(rm >= 0.9))
        self.assertTrue(np.all(rm <= 2.1))

    def test_rho_range_shape(self):
        from amica_python.metrics import rho_range
        rr = rho_range(self.result)
        self.assertEqual(rr.shape, (4,))
        self.assertTrue(np.all(rr >= 0))

    def test_mixture_entropy_shape_and_bounds(self):
        from amica_python.metrics import mixture_entropy
        ent = mixture_entropy(self.result)
        self.assertEqual(ent.shape, (4,))
        # Entropy >= 0
        self.assertTrue(np.all(ent >= 0))
        # Entropy <= log(n_mix) = log(3)
        self.assertTrue(np.all(ent <= np.log(3) + 1e-10))

    def test_multimodality_flag_shape(self):
        from amica_python.metrics import multimodality_flag
        flags = multimodality_flag(self.result)
        self.assertEqual(flags.shape, (4,))
        self.assertEqual(flags.dtype, bool)

    def test_source_kurtosis_shape(self):
        from amica_python.metrics import source_kurtosis
        kurt = source_kurtosis(self.result, self.data)
        self.assertEqual(kurt.shape, (4,))
        # Laplacian sources have positive excess kurtosis
        self.assertTrue(np.any(kurt > 0))

    def test_laplacian_rho_near_one(self):
        """Laplacian sources should yield rho close to 1.0."""
        from amica_python.metrics import rho_mean
        rm = rho_mean(self.result)
        # At least one component should have rho near 1 (Laplacian)
        self.assertTrue(np.any(rm < 1.5),
                        f"Expected some rho < 1.5 for Laplacian sources, got {rm}")


if __name__ == "__main__":
    unittest.main()
