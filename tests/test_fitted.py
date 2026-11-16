"""Checks for the fitted detectors, on synthetic vectors.

The real test is tests/bench_separation.py.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from trustrag_guard.layers.fitted import (  # noqa: E402
    fit_detectors,
    isolation_score,
    load,
    mahalanobis_score,
    save,
)


def part_a():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(600, 20))
    far = rng.normal(size=(30, 20)) + 6.0
    fresh = rng.normal(size=(30, 20))
    fitted = fit_detectors(clean, trees=50, dims=10)

    for name, score in (("Isolation Forest", isolation_score), ("Mahalanobis", mahalanobis_score)):
        assert score(fitted, far).min() > np.median(score(fitted, fresh)), name
    print("[ok] both scores rate far points above fresh clean points")

    assert np.array_equal(isolation_score(fit_detectors(clean, trees=50, dims=10), fresh), isolation_score(fitted, fresh))
    print("[ok] the same seed gives the same scores")

    z = fitted.pca.transform(fresh[:1]) - fitted.mean
    assert np.isclose(mahalanobis_score(fitted, fresh[:1])[0], (z @ fitted.precision @ z.T).item())
    print("[ok] the Mahalanobis score matches the formula")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.joblib")
        save(fitted, path)
        back = load(path)
        assert np.array_equal(mahalanobis_score(back, fresh), mahalanobis_score(fitted, fresh))
    print("[ok] fitted models round-trip through a file")


if __name__ == "__main__":
    part_a()
    print("\nStep 5.2 checks finished.")
