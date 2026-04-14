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

        self.assertEqual(result.unmixing_matrix_white_.shape, (n_channels, n_channels))
        self.assertEqual(result.mixing_matrix_white_.shape, (n_channels, n_channels))
        self.assertEqual(result.unmixing_matrix_sensor_.shape, (n_channels, n_channels))
        self.assertEqual(result.mixing_matrix_sensor_.shape, (n_channels, n_channels))
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
        C = result.unmixing_matrix_white_ @ result.whitener_ @ A_true
        # Normalize rows and columns
        C = C / np.max(np.abs(C), axis=1, keepdims=True)

        # Amari index: sum of (sum/max - 1) for rows and cols
        row_ratios = np.sum(np.abs(C), axis=1) / np.max(np.abs(C), axis=1) - 1
        col_ratios = np.sum(np.abs(C), axis=0) / np.max(np.abs(C), axis=0) - 1
        amari = (np.mean(row_ratios) + np.mean(col_ratios)) / 2

        self.assertLess(amari, 0.3,
                        f"Amari index too high ({amari:.3f}), poor separation")


class TestMatrixConventions(unittest.TestCase):
    """Test explicit matrix naming and consistency."""

    def test_matrix_shapes_and_consistency(self):
        """Test all four matrices have correct shapes and are consistent."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(42)
        n_channels, n_samples = 6, 2000
        S = rng.laplace(size=(n_channels, n_samples))
        A_true = rng.randn(n_channels, n_channels)
        data = A_true @ S

        config = AmicaConfig(max_iter=50, num_mix_comps=2, do_newton=False)
        model = Amica(config=config, random_state=42)
        result = model.fit(data)

        # White-space matrices are square (n_comp x n_comp)
        self.assertEqual(result.unmixing_matrix_white_.shape,
                         (n_channels, n_channels))
        self.assertEqual(result.mixing_matrix_white_.shape,
                         (n_channels, n_channels))

        # Sensor-space matrices bridge channels and components
        self.assertEqual(result.unmixing_matrix_sensor_.shape,
                         (n_channels, n_channels))
        self.assertEqual(result.mixing_matrix_sensor_.shape,
                         (n_channels, n_channels))

        # W_white @ A_white ≈ I
        WA = result.unmixing_matrix_white_ @ result.mixing_matrix_white_
        np.testing.assert_allclose(WA, np.eye(n_channels), atol=1e-6)

        # sensor unmixing = W_white @ sphere
        expected_sensor = result.unmixing_matrix_white_ @ result.whitener_
        np.testing.assert_allclose(
            result.unmixing_matrix_sensor_, expected_sensor, atol=1e-10)

        # sensor mixing = desphere @ A_white
        expected_mix = result.dewhitener_ @ result.mixing_matrix_white_
        np.testing.assert_allclose(
            result.mixing_matrix_sensor_, expected_mix, atol=1e-10)

    def test_sensor_roundtrip(self):
        """Test mixing_sensor @ unmixing_sensor ≈ I for full-rank data."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(42)
        n_channels, n_samples = 4, 1000
        data = rng.randn(n_channels, n_samples)

        config = AmicaConfig(max_iter=20, num_mix_comps=2, do_newton=False)
        model = Amica(config=config, random_state=42)
        result = model.fit(data)

        # mixing_sensor @ unmixing_sensor should be close to identity
        product = result.mixing_matrix_sensor_ @ result.unmixing_matrix_sensor_
        np.testing.assert_allclose(product, np.eye(n_channels), atol=1e-6)

    def test_deprecated_properties_warn(self):
        """Test that old property names emit DeprecationWarning."""
        from amica_python import Amica, AmicaConfig
        import warnings

        rng = np.random.RandomState(42)
        data = rng.randn(4, 500)

        config = AmicaConfig(max_iter=10, num_mix_comps=2, do_newton=False)
        model = Amica(config=config, random_state=42)
        result = model.fit(data)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = result.unmixing_matrix
            self.assertTrue(any(issubclass(x.category, DeprecationWarning)
                                for x in w))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = result.mixing_matrix
            self.assertTrue(any(issubclass(x.category, DeprecationWarning)
                                for x in w))

    def test_deprecated_properties_return_correct_values(self):
        """Test deprecated properties return the right matrices."""
        from amica_python import Amica, AmicaConfig
        import warnings

        rng = np.random.RandomState(42)
        data = rng.randn(4, 500)

        config = AmicaConfig(max_iter=10, num_mix_comps=2, do_newton=False)
        model = Amica(config=config, random_state=42)
        result = model.fit(data)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            np.testing.assert_array_equal(
                result.unmixing_matrix, result.unmixing_matrix_white_)
            np.testing.assert_array_equal(
                result.mixing_matrix, result.mixing_matrix_sensor_)


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


class TestMNEDirectVsShim(unittest.TestCase):
    """Test that direct path matches old Infomax shim path."""

    def test_direct_vs_shim_sources_correlated(self):
        """Both paths should produce correlated source activations."""
        try:
            import mne
        except ImportError:
            self.skipTest("MNE-Python not installed")

        from amica_python import fit_ica

        info = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(6)],
            sfreq=256, ch_types="eeg",
        )
        rng = np.random.RandomState(42)
        data = rng.randn(6, 3000) * 1e-6
        raw = mne.io.RawArray(data, info)

        common_params = dict(
            n_components=3, max_iter=30, num_mix=2, random_state=42,
            fit_params={"do_newton": False},
        )

        ica_direct = fit_ica(raw.copy(), _use_infomax_shim=False, **common_params)
        ica_shim = fit_ica(raw.copy(), _use_infomax_shim=True, **common_params)

        src_direct = ica_direct.get_sources(raw).get_data()
        src_shim = ica_shim.get_sources(raw).get_data()

        # Sources should be highly correlated (up to sign/permutation)
        corr = np.abs(np.corrcoef(src_direct, src_shim)[:3, 3:])
        # Each direct source should match some shim source
        max_corrs = np.max(corr, axis=1)
        self.assertTrue(
            np.all(max_corrs > 0.9),
            f"Source correlation too low: {max_corrs}"
        )


class TestMNEIntegrationGuards(unittest.TestCase):
    """Test MNE integration guards and metadata."""

    def test_multi_model_raises(self):
        """fit_ica() with num_models > 1 should raise ValueError."""
        try:
            import mne
        except ImportError:
            self.skipTest("MNE-Python not installed")

        from amica_python import fit_ica

        sfreq = 256
        info = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(4)],
            sfreq=sfreq, ch_types="eeg",
        )
        rng = np.random.RandomState(42)
        raw = mne.io.RawArray(rng.randn(4, 1000) * 1e-6, info)

        with self.assertRaises(ValueError, msg="num_models > 1"):
            fit_ica(raw, n_components=2, max_iter=10,
                    fit_params={"num_models": 2, "do_newton": False})

    def test_amica_result_attached(self):
        """fit_ica() should attach amica_result_ to the ICA object."""
        try:
            import mne
        except ImportError:
            self.skipTest("MNE-Python not installed")

        from amica_python import fit_ica

        info = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(4)],
            sfreq=256, ch_types="eeg",
        )
        rng = np.random.RandomState(42)
        raw = mne.io.RawArray(rng.randn(4, 1000) * 1e-6, info)

        ica = fit_ica(raw, n_components=2, max_iter=10,
                      fit_params={"do_newton": False})
        self.assertTrue(hasattr(ica, "amica_result_"))
        self.assertIsNotNone(ica.amica_result_)

    def test_apply_preserves_shape(self):
        """ica.apply() should preserve data shape."""
        try:
            import mne
        except ImportError:
            self.skipTest("MNE-Python not installed")

        from amica_python import fit_ica

        info = mne.create_info(
            ch_names=[f"EEG{i:03d}" for i in range(4)],
            sfreq=256, ch_types="eeg",
        )
        rng = np.random.RandomState(42)
        data = rng.randn(4, 1000) * 1e-6
        raw = mne.io.RawArray(data, info)

        ica = fit_ica(raw, n_components=2, max_iter=10,
                      fit_params={"do_newton": False})
        raw_clean = ica.apply(raw.copy())
        self.assertEqual(raw_clean.get_data().shape, data.shape)


class TestChunkedAccumulator(unittest.TestCase):
    """Chunked E-step should match full-batch within float64 rounding."""

    def test_chunked_loglik_additivity(self):
        """sum(compute_loglik_chunk) across halves == compute_total_loglikelihood."""
        import jax.numpy as jnp
        from amica_python.likelihood import (
            compute_total_loglikelihood, compute_loglik_chunk,
        )
        rng = np.random.RandomState(0)
        n_comp, n_mix, n_samp = 4, 3, 10000
        y = rng.randn(n_comp, n_samp)
        W = np.eye(n_comp) + 0.01 * rng.randn(n_comp, n_comp)
        alpha = np.ones((n_mix, n_comp)) / n_mix
        mu = rng.randn(n_mix, n_comp) * 0.1
        beta = np.ones((n_mix, n_comp)) + 0.05 * rng.randn(n_mix, n_comp)
        rho = np.full((n_mix, n_comp), 1.5)

        ll_full = float(compute_total_loglikelihood(
            jnp.asarray(y), jnp.asarray(W), jnp.asarray(alpha),
            jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho),
            log_det_sphere=0.3,
        ))
        ll_h1, n1 = compute_loglik_chunk(
            jnp.asarray(y[:, :5000]), jnp.asarray(W), jnp.asarray(alpha),
            jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho),
            log_det_sphere=0.3,
        )
        ll_h2, n2 = compute_loglik_chunk(
            jnp.asarray(y[:, 5000:]), jnp.asarray(W), jnp.asarray(alpha),
            jnp.asarray(mu), jnp.asarray(beta), jnp.asarray(rho),
            log_det_sphere=0.3,
        )
        ll_merged = float((ll_h1 + ll_h2) / (n1 + n2) / n_comp)
        self.assertLess(abs(ll_full - ll_merged) / max(abs(ll_full), 1e-20), 1e-12)

    def test_chunked_matches_fullbatch_synthetic(self):
        """Chunked vs full-batch: W and LL agree within rounding after 50 iters."""
        from amica_python import Amica, AmicaConfig

        rng = np.random.RandomState(42)
        n_src, n_samp = 4, 5000
        srcs = np.stack([
            rng.laplace(size=n_samp),
            rng.standard_t(df=3, size=n_samp),
            rng.laplace(size=n_samp) * 1.5,
            np.sign(rng.randn(n_samp)) * rng.exponential(size=n_samp),
        ])[:n_src]
        srcs = srcs / srcs.std(axis=1, keepdims=True)
        A_true = rng.randn(n_src, n_src)
        A_true = A_true / np.linalg.norm(A_true, axis=0, keepdims=True)
        x = A_true @ srcs

        cfg_kw = dict(num_models=1, num_mix_comps=3, max_iter=50,
                      dtype="float64", pcakeep=n_src)
        res_full = Amica(AmicaConfig(**cfg_kw, chunk_size=None),
                         random_state=42).fit(x)
        res_chunk = Amica(AmicaConfig(**cfg_kw, chunk_size=1024),
                          random_state=42).fit(x)

        W_full = np.asarray(res_full.unmixing_matrix_white_)
        W_chunk = np.asarray(res_chunk.unmixing_matrix_white_)
        rel_err = np.max(np.abs(W_chunk - W_full)) / np.max(np.abs(W_full))
        self.assertLess(rel_err, 1e-4,
            f"Chunked W diverged from full-batch: rel_err={rel_err:.2e}")

        ll_full = float(np.asarray(res_full.log_likelihood)[-1])
        ll_chunk = float(np.asarray(res_chunk.log_likelihood)[-1])
        self.assertLess(abs(ll_full - ll_chunk), 1e-5,
            f"Final LL diverged: full={ll_full:.8f} chunk={ll_chunk:.8f}")


if __name__ == "__main__":
    unittest.main()
