"""Tests for amica-python package."""
import unittest
import numpy as np


class TestAmicaBasic(unittest.TestCase):
    """Basic smoke tests for AMICA solver."""

    def test_fit_random_data(self):
        """Test basic fitting on random data."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(42)
        n_channels, n_samples = 4, 500
        data = rng.randn(n_channels, n_samples)

        config = AmicaConfig(
            max_iter=20,
            num_mix_comps=2,
            do_newton=False,
        )
        model = Amica(config=config, random_state=42)
        result = model.fit(data)

        self.assertEqual(result.unmixing_matrix.shape, (n_channels, n_channels))
        self.assertEqual(result.mixing_matrix.shape, (n_channels, n_channels))
        self.assertGreater(len(result.log_likelihood), 0)

    def test_transform_inverse(self):
        """Test that transform + inverse_transform reconstructs data."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(42)
        n_channels, n_samples = 4, 500
        data = rng.randn(n_channels, n_samples)

        config = AmicaConfig(max_iter=20, num_mix_comps=2, do_newton=False)
        model = Amica(config=config, random_state=42)
        model.fit(data)

        sources = model.transform(data)
        self.assertEqual(sources.shape, (n_channels, n_samples))

        recon = model.inverse_transform(sources)
        self.assertEqual(recon.shape, (n_channels, n_samples))

        err = np.mean((data - recon) ** 2) / np.mean(data ** 2)
        self.assertLess(err, 1e-6, f"Reconstruction NRMSE too high: {err:.2e}")

    def test_ll_increases(self):
        """Test that log-likelihood generally increases over iterations."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(123)
        n_channels, n_samples = 4, 1000
        # Create data with some structure (mixed sources)
        S = rng.laplace(size=(n_channels, n_samples))
        A = rng.randn(n_channels, n_channels)
        data = A @ S

        config = AmicaConfig(max_iter=100, num_mix_comps=3, do_newton=True)
        model = Amica(config=config, random_state=42)
        result = model.fit(data)

        ll = result.log_likelihood
        self.assertGreater(len(ll), 10)
        # LL at end should be higher than at start (allowing some tolerance)
        self.assertGreater(ll[-1], ll[5],
                           "Log-likelihood should increase over training")


class TestAmicaFunctionalAPI(unittest.TestCase):
    """Test the Picard-compatible functional API."""

    def test_amica_function(self):
        """Test amica() functional API returns correct shapes."""
        from amica_python import amica

        rng = np.random.RandomState(42)
        n_samples, n_components = 500, 4
        # MNE convention: (n_samples, n_components)
        X = rng.randn(n_samples, n_components)

        W = amica(X, max_iter=20, num_mix=2)
        self.assertEqual(W.shape, (n_components, n_components))

    def test_amica_return_n_iter(self):
        """Test return_n_iter flag."""
        from amica_python import amica

        rng = np.random.RandomState(42)
        X = rng.randn(500, 4)

        W, n_iter = amica(X, max_iter=20, num_mix=2, return_n_iter=True)
        self.assertEqual(W.shape, (4, 4))
        self.assertIsInstance(n_iter, int)
        self.assertGreater(n_iter, 0)


class TestAmicaSourceSeparation(unittest.TestCase):
    """Test actual source separation quality."""

    def test_separate_laplacian_sources(self):
        """Test separation of known Laplacian sources."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(0)
        n_sources, n_samples = 3, 5000

        # Generate independent Laplacian sources
        S = rng.laplace(size=(n_sources, n_samples))

        # Random mixing
        A_true = rng.randn(n_sources, n_sources)
        X = A_true @ S

        config = AmicaConfig(max_iter=500, num_mix_comps=3, do_newton=True)
        model = Amica(config=config, random_state=42)
        result = model.fit(X)

        # Recover sources
        S_hat = model.transform(X)

        # Check Amari index (permutation-invariant separation quality)
        # Perfect separation: each row/col of W @ A_true has one dominant entry
        C = result.unmixing_matrix @ result.whitener_ @ A_true
        # Normalize rows and columns
        C = C / np.max(np.abs(C), axis=1, keepdims=True)

        # Amari index: sum of (sum/max - 1) for rows and cols
        row_ratios = np.sum(np.abs(C), axis=1) / np.max(np.abs(C), axis=1) - 1
        col_ratios = np.sum(np.abs(C), axis=0) / np.max(np.abs(C), axis=0) - 1
        amari = (np.mean(row_ratios) + np.mean(col_ratios)) / 2

        self.assertLess(amari, 0.3,
                        f"Amari index too high ({amari:.3f}), poor separation")


class TestAmicaConfig(unittest.TestCase):
    """Test configuration validation."""

    def test_defaults(self):
        """Test default config values match literature recommendations."""
        from amica_python import AmicaConfig
        cfg = AmicaConfig()
        self.assertEqual(cfg.max_iter, 2000)
        self.assertEqual(cfg.num_mix_comps, 3)
        self.assertTrue(cfg.do_newton)
        self.assertEqual(cfg.newt_start, 50)
        self.assertEqual(cfg.rejstart, 2)
        self.assertEqual(cfg.rejint, 3)
        self.assertEqual(cfg.rejsig, 3.0)

    def test_invalid_config(self):
        """Test that invalid config raises errors."""
        from amica_python import AmicaConfig
        with self.assertRaises(ValueError):
            AmicaConfig(num_models=0)
        with self.assertRaises(ValueError):
            AmicaConfig(minrho=0.5)
        with self.assertRaises(ValueError):
            AmicaConfig(maxrho=3.0)


class TestAmicaRejection(unittest.TestCase):
    """Test sample rejection feature."""

    def test_rejection_enabled(self):
        """Test that rejection runs without errors when enabled."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(42)
        n_channels, n_samples = 4, 1000
        data = rng.randn(n_channels, n_samples)
        # Add some outliers
        data[:, :10] *= 100

        config = AmicaConfig(
            max_iter=30,
            num_mix_comps=2,
            do_reject=True,
            rejstart=2,
            rejint=3,
            rejsig=3.0,
            numrej=3,
            do_newton=False,
        )
        model = Amica(config=config, random_state=42)
        result = model.fit(data)

        self.assertIsNotNone(result)
        self.assertGreater(len(result.log_likelihood), 0)


class TestMNEIntegration(unittest.TestCase):
    """Test MNE-Python integration."""

    def test_fit_ica_on_raw(self):
        """Test fit_ica produces a working MNE ICA object."""
        try:
            import mne
        except ImportError:
            self.skipTest("MNE-Python not installed")

        from amica_python import fit_ica

        # Create synthetic Raw object
        sfreq = 256
        n_channels = 8
        n_samples = 2000
        info = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(n_channels)],
            sfreq=sfreq,
            ch_types="eeg",
        )
        rng = np.random.RandomState(42)
        data = rng.randn(n_channels, n_samples) * 1e-6  # Volts
        raw = mne.io.RawArray(data, info)

        ica = fit_ica(raw, n_components=4, max_iter=20,
                      num_mix=2, random_state=42,
                      fit_params={"do_newton": False})

        self.assertEqual(ica.n_components_, 4)
        self.assertEqual(ica.method, "amica")
        self.assertIsNotNone(ica.unmixing_matrix_)

        # Test that standard MNE methods work
        sources = ica.get_sources(raw)
        self.assertEqual(sources.get_data().shape[0], 4)


if __name__ == "__main__":
    unittest.main()
