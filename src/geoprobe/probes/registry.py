"""Single source of truth for probe construction.

Every experiment builds probes ONLY through `build_probe()`, so a fix or bug
lives in one place (this module) instead of being copy-pasted across scripts.

Each probe is tagged with an HONEST geometry family:
  euclidean  : flat-space methods with a linear or piecewise boundary
               (linear, mlp, pca+linear, centroid, knn)
  metric     : learned *linear* metric — Mahalanobis == LDA, NOT curved geometry
  manifold   : geodesic distance on a kNN graph (Isomap-style); a manifold, but
               still embedded in Euclidean space
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from geoprobe.probes.sklearn_probes import (
    ClassMahalanobisProbe,
    GraphGeodesicProbe,
    MahalanobisProbe,
    MeanCentroidProbe,
    TangentSubspaceProbe,
)

# family tag per probe
FAMILY = {
    "linear": "euclidean",
    "mlp": "euclidean",
    "pca8": "euclidean",
    "pca50": "euclidean",
    "centroid": "euclidean",
    "knn": "euclidean",
    "mahalanobis": "metric",
    "class_mahalanobis": "metric",
    "tangent_subspace": "manifold",
    "graph_geodesic": "manifold",
}

# probes that must pass synthetic gates before their activation numbers count
GATED = set()


def _logreg():
    return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)


def build_probe(name: str):
    """Return a fresh sklearn-compatible estimator (pipeline) for `name`."""
    if name == "linear":
        return make_pipeline(StandardScaler(), _logreg())
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0))
    if name == "pca8":
        return make_pipeline(StandardScaler(), PCA(8, random_state=0), _logreg())
    if name == "pca50":
        return make_pipeline(StandardScaler(), PCA(50, random_state=0), _logreg())
    if name == "centroid":
        return make_pipeline(StandardScaler(), MeanCentroidProbe())
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
    if name == "mahalanobis":
        return make_pipeline(StandardScaler(), MahalanobisProbe(shrinkage=0.2))
    if name == "class_mahalanobis":
        return make_pipeline(StandardScaler(), ClassMahalanobisProbe(shrinkage=0.2))
    if name == "tangent_subspace":
        return make_pipeline(StandardScaler(), TangentSubspaceProbe(n_components=8))
    if name == "graph_geodesic":
        return make_pipeline(StandardScaler(), GraphGeodesicProbe())
    raise ValueError(f"unknown probe: {name!r} (known: {sorted(FAMILY)})")


def known_probes() -> list[str]:
    return list(FAMILY)
