import numpy as np

from wmdp_linearized.attack import _predict, _ridge_fit


def test_ridge_reduces_constructed_squared_error():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(40, 4, 6)).astype(np.float32)
    base = np.zeros((40, 4), dtype=np.float32)
    labels = rng.integers(0, 4, size=40)
    ids = np.arange(30)
    weight = _ridge_fit(
        features, base, labels, ids, 1e-4
    )
    pred = _predict(features, base, weight)
    target = np.eye(4)[labels] - 0.25
    assert np.mean((pred[ids] - target[ids]) ** 2) < np.mean(
        target[ids] ** 2
    )

