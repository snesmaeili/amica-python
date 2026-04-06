"""MNE ICA contract tests for amica-python.

Verifies that fit_ica() produces an ICA object that behaves correctly
with all standard MNE ICA operations: plotting, artifact scoring,
save/load, Epochs, rank-deficient data, and ICLabel (if installed).
"""
import os
import tempfile
import unittest

import numpy as np


def _make_raw_with_montage(n_channels=32, n_samples=5000, sfreq=256, seed=42):
    """Create a synthetic Raw with a real EEG montage."""
    import mne

    # Use standard 10-20 channel names for montage compatibility
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_names_avail = montage.ch_names
    ch_names = ch_names_avail[:n_channels]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    rng = np.random.RandomState(seed)

    # Create sources: some brain-like, one EOG-like, one ECG-like
    n_brain = n_channels - 2
    S = np.vstack([
        rng.laplace(size=(n_brain, n_samples)),
        # EOG: slow blink pattern
        np.sin(2 * np.pi * 0.3 * np.arange(n_samples) / sfreq)[None, :] * 5,
        # ECG: periodic spikes
        (np.mod(np.arange(n_samples), int(sfreq * 0.8)) < 5).astype(float)[None, :] * 10,
    ])
    A = rng.randn(n_channels, n_channels)
    data = A @ S * 1e-6  # in Volts

    raw = mne.io.RawArray(data, info)
    raw.set_montage(montage)

    # Add EOG and ECG channels for artifact scoring
    eog_data = S[-2:, :] * 1e-6
    ecg_data = S[-1:, :] * 1e-6
    eog_info = mne.create_info(["EOG001", "EOG002"], sfreq, ["eog", "eog"])
    ecg_info = mne.create_info(["ECG001"], sfreq, ["ecg"])
    raw.add_channels([mne.io.RawArray(eog_data, eog_info)], force_update_info=True)
    raw.add_channels([mne.io.RawArray(ecg_data, ecg_info)], force_update_info=True)

    return raw


class TestPlotComponents(unittest.TestCase):
    """Test that plot_components works with a real montage."""

    def test_plot_components_no_error(self):
        try:
            import mne
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
        except ImportError:
            self.skipTest("MNE or matplotlib not installed")

        from amica_python import fit_ica

        raw = _make_raw_with_montage(n_channels=16, n_samples=3000)
        ica = fit_ica(raw, n_components=5, max_iter=20,
                      fit_params={"do_newton": False}, random_state=42)

        # Should not raise
        import matplotlib.pyplot as plt
        figs = ica.plot_components(picks=range(5), show=False)
        plt.close("all")
        self.assertIsNotNone(figs)


class TestArtifactScoring(unittest.TestCase):
    """Test find_bads_eog and find_bads_ecg."""

    def test_find_bads_eog(self):
        try:
            import mne
        except ImportError:
            self.skipTest("MNE not installed")

        from amica_python import fit_ica

        raw = _make_raw_with_montage(n_channels=16, n_samples=5000)
        ica = fit_ica(raw, n_components=8, max_iter=30,
                      fit_params={"do_newton": False}, random_state=42)

        eog_idx, eog_scores = ica.find_bads_eog(raw, verbose=False)
        self.assertIsInstance(eog_idx, list)
        # eog_scores is ndarray (1 EOG ch) or list of ndarrays (multiple)
        if isinstance(eog_scores, list):
            self.assertTrue(all(isinstance(s, np.ndarray) for s in eog_scores))
        else:
            self.assertIsInstance(eog_scores, np.ndarray)

    def test_find_bads_ecg(self):
        try:
            import mne
        except ImportError:
            self.skipTest("MNE not installed")

        from amica_python import fit_ica

        raw = _make_raw_with_montage(n_channels=16, n_samples=5000)
        ica = fit_ica(raw, n_components=8, max_iter=30,
                      fit_params={"do_newton": False}, random_state=42)

        ecg_idx, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
        self.assertIsInstance(ecg_idx, list)
        self.assertIsInstance(ecg_scores, np.ndarray)


class TestSaveLoad(unittest.TestCase):
    """Test save/read_ica roundtrip."""

    def test_save_load_roundtrip(self):
        try:
            import mne
            from mne.preprocessing import read_ica
        except ImportError:
            self.skipTest("MNE not installed")

        from amica_python import fit_ica

        raw = _make_raw_with_montage(n_channels=8, n_samples=2000)
        ica = fit_ica(raw, n_components=4, max_iter=15,
                      fit_params={"do_newton": False}, random_state=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "test-ica.fif")
            ica.save(fname, overwrite=True)
            ica_loaded = read_ica(fname)

        # Core matrices should match
        np.testing.assert_allclose(
            ica_loaded.unmixing_matrix_, ica.unmixing_matrix_, atol=1e-10)
        np.testing.assert_allclose(
            ica_loaded.mixing_matrix_, ica.mixing_matrix_, atol=1e-10)
        self.assertEqual(ica_loaded.n_components_, ica.n_components_)

        # Should still produce sources
        sources = ica_loaded.get_sources(raw)
        self.assertEqual(sources.get_data().shape[0], 4)


class TestEpochs(unittest.TestCase):
    """Test fit_ica with Epochs input."""

    def test_fit_on_epochs(self):
        try:
            import mne
        except ImportError:
            self.skipTest("MNE not installed")

        from amica_python import fit_ica

        raw = _make_raw_with_montage(n_channels=8, n_samples=5000)
        events = mne.make_fixed_length_events(raw, duration=1.0)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=0.99,
                            picks="eeg", baseline=None, preload=True,
                            verbose=False)

        ica = fit_ica(epochs, n_components=4, max_iter=15,
                      fit_params={"do_newton": False}, random_state=42)

        self.assertEqual(ica.n_components_, 4)
        self.assertEqual(ica.current_fit, "epochs")

        # Should work with raw too
        sources = ica.get_sources(raw)
        self.assertEqual(sources.get_data().shape[0], 4)


class TestRankDeficient(unittest.TestCase):
    """Test with average-referenced / rank-deficient data."""

    def test_average_reference(self):
        try:
            import mne
        except ImportError:
            self.skipTest("MNE not installed")

        from amica_python import fit_ica

        raw = _make_raw_with_montage(n_channels=16, n_samples=3000)
        raw.pick("eeg")
        raw.set_eeg_reference("average", projection=False, verbose=False)

        # Rank is now n_channels - 1
        rank = np.linalg.matrix_rank(raw.get_data())
        n_comp = min(rank, 10)

        ica = fit_ica(raw, n_components=n_comp, max_iter=20,
                      fit_params={"do_newton": False}, random_state=42)

        self.assertEqual(ica.n_components_, n_comp)

        # apply should work
        raw_clean = ica.apply(raw.copy())
        self.assertEqual(raw_clean.get_data().shape, raw.get_data().shape)


class TestICLabel(unittest.TestCase):
    """Test mne-icalabel compatibility (skipped if not installed)."""

    def test_icalabel(self):
        try:
            import mne
            from mne_icalabel import label_components
        except ImportError:
            self.skipTest("mne-icalabel not installed")

        from amica_python import fit_ica

        raw = _make_raw_with_montage(n_channels=32, n_samples=10000)
        raw.pick("eeg")
        ica = fit_ica(raw, n_components=10, max_iter=30,
                      fit_params={"do_newton": False}, random_state=42)

        labels = label_components(raw, ica, method="iclabel")
        self.assertIn("y_pred_proba", labels)
        self.assertEqual(labels["y_pred_proba"].shape[0], 10)


if __name__ == "__main__":
    unittest.main()
