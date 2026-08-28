"""Contract tests for the public single-model ``amica`` function."""

from importlib.metadata import version
from types import SimpleNamespace

import numpy as np
import pytest

from jamica import Amica, AmicaConfig, JamicaConvergenceWarning, __version__, amica


def test_public_version_matches_distribution():
    """Expose the installed distribution version for dependency checks."""
    assert __version__ == version("jamica")


def _fake_result(n_features, *, data_scale=1.0, converged=True):
    """Create the result fields consumed by the functional wrapper."""
    return SimpleNamespace(
        unmixing_matrix_white_=np.eye(n_features),
        whitener_=np.eye(n_features),
        mean_=np.zeros(n_features),
        data_scale=data_scale,
        converged=converged,
        n_iter=3,
    )


def test_preprocessed_single_model_contract(monkeypatch):
    """The MNE path must disable every internal affine preprocessing step."""
    captured = {}

    def fake_fit(self, X):
        captured["config"] = self.config
        captured["X"] = X.copy()
        return _fake_result(X.shape[0])

    monkeypatch.setattr(Amica, "fit", fake_fit)
    X = np.arange(60, dtype=np.float32).reshape(3, 20) + 7.0

    K, W, Y, n_iter = amica(
        X,
        whiten=False,
        return_n_iter=True,
        random_state=42,
        max_iter=10,
        num_mix=2,
        num_models=1,
    )

    config = captured["config"]
    assert config.num_models == 1
    assert config.do_mean is False
    assert config.do_sphere is False
    assert config.do_pca is False
    assert config.pcakeep is None
    assert config.update_c is False
    assert config.dtype == "float64"
    assert captured["X"].dtype == np.float64
    assert K is None
    np.testing.assert_array_equal(W, np.eye(3))
    np.testing.assert_allclose(Y, W @ X)
    assert n_iter == 3


def test_preprocessed_operator_includes_internal_scale(monkeypatch):
    """Returned W must act on the caller input, not hidden scaled data."""
    scale = 2.5e-4

    def fake_fit(self, X):
        return _fake_result(X.shape[0], data_scale=scale)

    monkeypatch.setattr(Amica, "fit", fake_fit)
    X = 1e4 * np.random.default_rng(0).standard_normal((3, 30))

    K, W, Y = amica(X, whiten=False, max_iter=1)

    assert K is None
    np.testing.assert_array_equal(W, scale * np.eye(3))
    np.testing.assert_allclose(Y, W @ X)


def test_scaled_operator_matches_solver_transform():
    """The real wrapper and estimator agree when emergency scaling triggers."""
    X = 1e4 * np.random.default_rng(0).standard_normal((3, 40))
    config = AmicaConfig(
        max_iter=1,
        num_mix_comps=2,
        num_models=1,
        dtype="float64",
        do_sphere=False,
        do_mean=False,
        do_pca=False,
        pcakeep=None,
        update_c=False,
        do_newton=False,
        chunk_size=None,
    )
    estimator = Amica(config, random_state=0)
    estimator.fit(X)
    expected = estimator.transform(X)

    with pytest.warns(JamicaConvergenceWarning, match="did not converge"):
        K, W, Y = amica(
            X,
            whiten=False,
            max_iter=1,
            num_mix=2,
            do_newton=False,
            chunk_size=None,
            random_state=0,
        )

    assert K is None
    assert estimator.result_.data_scale != 1.0
    np.testing.assert_allclose(Y, expected)
    np.testing.assert_allclose(Y, W @ X)


@pytest.mark.parametrize(
    "rng_factory",
    [lambda: np.random.RandomState(42), lambda: np.random.default_rng(42)],
    ids=["RandomState", "Generator"],
)
def test_numpy_random_state_objects_are_supported(rng_factory):
    """The public MNE-facing function accepts both NumPy RNG APIs."""
    X = np.random.default_rng(0).standard_normal((3, 40))

    outputs = []
    for _ in range(2):
        with pytest.warns(JamicaConvergenceWarning, match="did not converge"):
            outputs.append(
                amica(
                    X,
                    max_iter=1,
                    do_newton=False,
                    chunk_size=None,
                    random_state=rng_factory(),
                )
            )

    for left, right in zip(outputs[0], outputs[1], strict=True):
        if left is None:
            assert right is None
        else:
            np.testing.assert_allclose(left, right)


def test_whitened_operator_includes_internal_scale(monkeypatch):
    """The optional internally whitened path must use caller coordinates."""
    scale = 0.25
    mean_scaled = np.array([1.0, -2.0, 3.0])

    def fake_fit(self, X):
        result = _fake_result(X.shape[0], data_scale=scale)
        result.mean_ = mean_scaled
        result.whitener_ = 2.0 * np.eye(X.shape[0])
        return result

    monkeypatch.setattr(Amica, "fit", fake_fit)
    X = np.random.default_rng(0).standard_normal((3, 30))

    K, W, Y = amica(X, n_components=3, whiten=True, max_iter=1)

    expected_mean = mean_scaled[:, None] / scale
    np.testing.assert_array_equal(K, 2.0 * scale * np.eye(3))
    np.testing.assert_allclose(Y, W @ K @ (X - expected_mean))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_models": 2},
        {"do_mean": True},
        {"do_sphere": True},
        {"do_pca": True},
        {"pcakeep": 2},
        {"dtype": "float32"},
        {"update_c": True},
        {"sphere_type": "pca"},
        {"do_approx_sphere": True},
        {"mineig": 1e-10},
        {"backend": "jax"},
    ],
)
def test_protected_options_are_not_public(kwargs):
    """The functional boundary must not expose broad AmicaConfig options."""
    X = np.random.default_rng(0).standard_normal((3, 30))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        amica(X, max_iter=1, **kwargs)


def test_multi_model_request_has_actionable_error():
    """MNE fit_params must propagate guidance to the native multi-model API."""
    X = np.random.default_rng(0).standard_normal((3, 30))
    message = (
        "jamica.amica() supports a single AMICA model. For multi-model AMICA, use jamica.AmicaICA."
    )

    with pytest.raises(ValueError) as error:
        amica(X, num_models=2)
    assert str(error.value) == message

    fit_params = {"num_models": 2}
    with pytest.raises(ValueError) as error:
        amica(X, whiten=False, return_n_iter=True, **fit_params)
    assert str(error.value) == message


@pytest.mark.parametrize(
    "X, error, match",
    [
        (np.ones(20), ValueError, "2D"),
        (np.ones((1, 20)), ValueError, "at least 2 features"),
        (np.ones((4, 3)), ValueError, "at least as many samples"),
        (np.full((3, 20), np.nan), ValueError, "non-finite"),
        (np.ones((3, 20), dtype=complex), TypeError, "real numeric"),
    ],
)
def test_invalid_input(X, error, match):
    """Invalid arrays fail before constructing a solver."""
    with pytest.raises(error, match=match):
        amica(X, max_iter=1)


def test_preprocessed_n_components_must_match_input():
    """Dimension reduction belongs to the caller on the MNE path."""
    X = np.ones((3, 20))
    with pytest.raises(ValueError, match=r"must equal X.shape\[0\]"):
        amica(X, n_components=2, whiten=False, max_iter=1)


def test_nonconvergence_warns_and_returns_last_operator(monkeypatch):
    """A finite last iterate remains usable but non-convergence is visible."""

    def fake_fit(self, X):
        return _fake_result(X.shape[0], converged=False)

    monkeypatch.setattr(Amica, "fit", fake_fit)
    X = np.random.default_rng(0).standard_normal((3, 30))

    with pytest.warns(JamicaConvergenceWarning, match="did not converge after 3"):
        K, W, Y, n_iter = amica(X, return_n_iter=True, max_iter=3)

    assert K is None
    assert n_iter == 3
    np.testing.assert_allclose(Y, W @ X)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("unmixing_matrix_white_", np.ones((3, 2)), "unmixing shape"),
        ("unmixing_matrix_white_", np.full((3, 3), np.nan), "non-finite"),
        ("unmixing_matrix_white_", np.zeros((3, 3)), "singular"),
        ("data_scale", 0.0, "data scale"),
    ],
)
def test_invalid_solver_output_is_rejected(monkeypatch, field, value, match):
    """MNE must never receive a malformed linear operator."""

    def fake_fit(self, X):
        result = _fake_result(X.shape[0])
        setattr(result, field, value)
        return result

    monkeypatch.setattr(Amica, "fit", fake_fit)
    X = np.random.default_rng(0).standard_normal((3, 30))
    with pytest.raises(RuntimeError, match=match):
        amica(X, max_iter=1)
