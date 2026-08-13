"""Tests for jamica.viz module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from jamica import viz

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockAmicaResult:
    """Mock result object containing attributes needed for visualization."""

    def __init__(self, n_comp=4, n_mix=3, n_models=1):
        rng = np.random.RandomState(42)

        self.alpha_ = np.ones((n_mix, n_comp)) / n_mix
        self.mu_ = rng.randn(n_mix, n_comp) * 0.1
        self.sbeta_ = np.ones((n_mix, n_comp)) + 0.5
        self.rho_ = np.full((n_mix, n_comp), 1.5)
        self.log_likelihood = np.linspace(-10, -5, 50).tolist()
        self.elapsed_times = np.linspace(0.1, 5.0, 50).tolist()

        self.mean_ = np.zeros(n_comp)
        self.whitener_ = np.eye(n_comp)
        self.unmixing_matrix_white_ = np.eye(n_comp)
        self.data_scale = 1.0

        # Multi-model specific
        self.gm_ = np.ones(n_models) / n_models
        if n_models > 1:
            self.alpha_ = np.ones((n_models, n_mix, n_comp)) / n_mix
            self.mu_ = rng.randn(n_models, n_mix, n_comp) * 0.1
            self.sbeta_ = np.ones((n_models, n_mix, n_comp)) + 0.5
            self.rho_ = np.full((n_models, n_mix, n_comp), 1.5)
            self.unmixing_matrix_white_ = np.array([np.eye(n_comp) for _ in range(n_models)])
            self.c_ = rng.randn(n_models, n_comp)


def test_check_result():
    res = MockAmicaResult()
    viz._check_result(res)

    class BadResult:
        pass

    with pytest.raises(TypeError, match="missing attribute 'alpha_'"):
        viz._check_result(BadResult())


def test_plot_convergence(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    res = MockAmicaResult()
    fig = viz.plot_convergence(res, show=True)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)

    # Test without elapsed_times, and providing an ax
    del res.elapsed_times
    fig2, ax = plt.subplots()
    fig_ret = viz.plot_convergence(res, ax=ax, show=False)
    assert fig_ret is fig2
    plt.close(fig2)


def test_plot_source_densities(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    res = MockAmicaResult(n_comp=4, n_mix=3)

    # Test without data
    fig = viz.plot_source_densities(res, show=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test with data
    rng = np.random.RandomState(42)
    data = rng.randn(4, 1000)
    fig = viz.plot_source_densities(res, data=data, picks=[0, 2], show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_model_responsibilities(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    # Should raise error if only 1 model
    res1 = MockAmicaResult(n_models=1)
    with pytest.raises(ValueError, match="requires num_models >= 2"):
        viz.plot_model_responsibilities(res1, data=np.zeros((4, 100)), show=False)

    # Should work for >1 model
    res2 = MockAmicaResult(n_models=2)
    rng = np.random.RandomState(42)
    data = rng.randn(4, 100)
    # With ax
    _fig2, ax = plt.subplots()
    fig = viz.plot_model_responsibilities(res2, data=data, ax=ax, show=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    # Without ax
    fig_noax = viz.plot_model_responsibilities(res2, data=data, show=False)
    assert isinstance(fig_noax, plt.Figure)
    plt.close(fig_noax)


def test_plot_mixture_weights(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    res = MockAmicaResult()
    # With ax
    _fig2, ax = plt.subplots()
    fig = viz.plot_mixture_weights(res, ax=ax, show=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    # Without ax
    fig_noax = viz.plot_mixture_weights(res, show=False)
    assert isinstance(fig_noax, plt.Figure)
    plt.close(fig_noax)


def test_plot_shape_parameters(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    res = MockAmicaResult()
    # With ax
    _fig2, ax = plt.subplots()
    fig = viz.plot_shape_parameters(res, ax=ax, show=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    # Without ax
    fig_noax = viz.plot_shape_parameters(res, show=False)
    assert isinstance(fig_noax, plt.Figure)
    plt.close(fig_noax)


def test_plot_parameter_summary(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    res = MockAmicaResult()
    # without data
    fig = viz.plot_parameter_summary(res, show=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # with data
    rng = np.random.RandomState(42)
    data = rng.randn(4, 1000)
    fig = viz.plot_parameter_summary(res, data=data, show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_component_metrics(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    res = MockAmicaResult(n_comp=2, n_mix=2)
    # Mocking metrics required attributes
    res.alpha_ = np.array([[0.1, 0.9], [0.9, 0.1]])  # 2 mix, 2 comp
    res.rho_ = np.array([[1.5, 1.8], [1.1, 1.9]])
    res.unmixing_matrix_white_ = np.eye(2)

    # without data
    fig = viz.plot_component_metrics(res, show=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # with data
    rng = np.random.RandomState(42)
    data = rng.randn(2, 1000)
    fig = viz.plot_component_metrics(res, data=data, picks=[0], show=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
