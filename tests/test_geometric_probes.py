from __future__ import annotations

import numpy as np

from geoprobe.probes import (
    ClassMahalanobisProbe,
    MahalanobisProbe,
    MeanCentroidProbe,
    TangentSubspaceProbe,
    build_probe,
    known_probes,
)


def test_centroid_predicts_nearest():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0], [1.1, 1.0]])
    y = np.array([0, 1, 0, 1])
    probe = MeanCentroidProbe().fit(x, y)
    preds = probe.predict(np.array([[0.0, 0.1], [1.0, 0.9]]))
    assert preds.tolist() == [0, 1]


def test_mahalanobis_predicts_nearest():
    x = np.array([[0.0, 0.0], [3.0, 0.0], [0.1, 0.0], [3.1, 0.0]])
    y = np.array([0, 1, 0, 1])
    probe = MahalanobisProbe().fit(x, y)
    preds = probe.predict(np.array([[0.2, 0.0], [2.9, 0.0]]))
    assert preds.tolist() == [0, 1]


def test_class_mahalanobis_predicts_nearest():
    x = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [3.0, 0.0],
            [3.1, 0.0],
            [3.0, 0.1],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    probe = ClassMahalanobisProbe().fit(x, y)
    preds = probe.predict(np.array([[0.2, 0.0], [2.9, 0.0]]))
    assert preds.tolist() == [0, 1]


def test_tangent_subspace_prefers_lower_reconstruction_error():
    x0 = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    x1 = np.array([[0.0, 3.0], [1.0, 3.0], [-1.0, 3.0]])
    x = np.concatenate([x0, x1], axis=0)
    y = np.array([0, 0, 0, 1, 1, 1])
    probe = TangentSubspaceProbe(n_components=1).fit(x, y)
    preds = probe.predict(np.array([[0.2, 0.1], [0.2, 2.9]]))
    assert preds.tolist() == [0, 1]


def test_registry_builds_all():
    # every registered probe builds
    for name in known_probes():
        assert build_probe(name) is not None


def test_registry_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        build_probe("not_a_probe")
