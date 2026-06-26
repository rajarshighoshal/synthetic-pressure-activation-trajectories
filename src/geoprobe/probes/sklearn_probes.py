from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class MeanCentroidProbe(ClassifierMixin, BaseEstimator):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "MeanCentroidProbe":
        self.classes_ = np.array(sorted(set(y.tolist())))
        self.centroids_ = np.stack([x[y == cls].mean(axis=0) for cls in self.classes_])
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(x[:, None, :] - self.centroids_[None, :, :], axis=-1)
        scores = -distances
        return scores[:, self.classes_.tolist().index(1)] - scores[:, self.classes_.tolist().index(0)]

    def predict(self, x: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(x[:, None, :] - self.centroids_[None, :, :], axis=-1)
        return self.classes_[np.argmin(distances, axis=1)]


class MahalanobisProbe(ClassifierMixin, BaseEstimator):
    """Nearest class centroid under the shared (covariance-whitened) metric.

    Distinct from MeanCentroidProbe (plain Euclidean): distances are measured
    with the inverse pooled covariance, so correlated/scaled directions are
    accounted for. This is a learned *metric* (a linear reweighting of space),
    NOT a curved/non-Euclidean manifold — keep that distinction when reporting.
    """

    def __init__(self, shrinkage: float = 0.1, max_full_features: int = 2048):
        self.shrinkage = shrinkage
        self.max_full_features = max_full_features

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MahalanobisProbe":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        self.centroids_ = np.stack([x[y == c].mean(axis=0) for c in self.classes_])
        if x.shape[1] > self.max_full_features:
            var = x.var(axis=0) + 1e-6
            self.precision_ = 1.0 / var
            self.diagonal_ = True
            return self

        cov = np.cov(x, rowvar=False)
        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]])
        diag = np.diag(np.diag(cov))
        cov = (1 - self.shrinkage) * cov + self.shrinkage * diag
        cov += 1e-6 * np.eye(cov.shape[0])
        self.precision_ = np.linalg.pinv(cov)
        self.diagonal_ = False
        return self

    def _maha_sq(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        diffs = x[:, None, :] - self.centroids_[None, :, :]  # (n, classes, d)
        if self.diagonal_:
            return np.sum((diffs**2) * self.precision_[None, None, :], axis=-1)
        return np.einsum("ncd,df,ncf->nc", diffs, self.precision_, diffs)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        d2 = self._maha_sq(x)
        cl = self.classes_.tolist()
        return d2[:, cl.index(0)] - d2[:, cl.index(1)]  # larger => closer to class 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmin(self._maha_sq(x), axis=1)]


class ClassMahalanobisProbe(ClassifierMixin, BaseEstimator):
    def __init__(self, shrinkage: float = 0.2, max_full_features: int = 512):
        self.shrinkage = shrinkage
        self.max_full_features = max_full_features

    def fit(self, x: np.ndarray, y: np.ndarray) -> "ClassMahalanobisProbe":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        self.centroids_ = []
        self.precisions_ = []
        self.logdets_ = []
        self.diagonal_ = x.shape[1] > self.max_full_features
        for cls in self.classes_:
            x_cls = x[y == cls]
            self.centroids_.append(x_cls.mean(axis=0))
            if self.diagonal_ or len(x_cls) < 3:
                var = x_cls.var(axis=0) + 1e-6
                self.precisions_.append(1.0 / var)
                self.logdets_.append(float(np.log(var).sum()))
                continue
            cov = np.cov(x_cls, rowvar=False)
            if cov.ndim == 0:
                cov = np.asarray([[float(cov)]])
            diag = np.diag(np.diag(cov))
            cov = (1 - self.shrinkage) * cov + self.shrinkage * diag
            cov += 1e-6 * np.eye(cov.shape[0])
            sign, logdet = np.linalg.slogdet(cov)
            self.precisions_.append(np.linalg.pinv(cov))
            self.logdets_.append(float(logdet if sign > 0 else 0.0))
        self.centroids_ = np.stack(self.centroids_)
        return self

    def _negative_energy(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        scores = []
        for class_idx in range(len(self.classes_)):
            diff = x - self.centroids_[class_idx]
            precision = self.precisions_[class_idx]
            if self.diagonal_:
                energy = np.sum((diff**2) * precision[None, :], axis=-1)
            else:
                energy = np.einsum("nd,df,nf->n", diff, precision, diff)
            scores.append(-0.5 * (energy + self.logdets_[class_idx]))
        return np.stack(scores, axis=1)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        scores = self._negative_energy(x)
        class_list = self.classes_.tolist()
        return scores[:, class_list.index(1)] - scores[:, class_list.index(0)]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._negative_energy(x), axis=1)]


class TangentSubspaceProbe(ClassifierMixin, BaseEstimator):
    def __init__(self, n_components: int = 8):
        self.n_components = n_components

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TangentSubspaceProbe":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.array(sorted(set(y.tolist())))
        self.centers_ = []
        self.bases_ = []
        for cls in self.classes_:
            x_cls = x[y == cls]
            center = x_cls.mean(axis=0)
            centered = x_cls - center
            rank = min(self.n_components, max(0, len(x_cls) - 1), x.shape[1])
            if rank:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                basis = vt[:rank]
            else:
                basis = np.zeros((0, x.shape[1]), dtype=np.float64)
            self.centers_.append(center)
            self.bases_.append(basis)
        self.centers_ = np.stack(self.centers_)
        return self

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        residuals = []
        for center, basis in zip(self.centers_, self.bases_):
            diff = x - center
            if len(basis):
                projected = diff @ basis.T @ basis
                diff = diff - projected
            residuals.append(np.sum(diff**2, axis=1))
        return np.stack(residuals, axis=1)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        residuals = self._residuals(x)
        class_list = self.classes_.tolist()
        return residuals[:, class_list.index(0)] - residuals[:, class_list.index(1)]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmin(self._residuals(x), axis=1)]


class GraphGeodesicProbe(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_neighbors: int = 15,
        bandwidth: float | None = None,
        max_train: int = 4000,
        random_state: int = 0,
    ):
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.max_train = max_train
        self.random_state = random_state

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GraphGeodesicProbe":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y)
        x, y = self._subsample(x, y)
        self.x_train_ = x
        self.y_train_ = y
        self.classes_ = np.array(sorted(set(y.tolist())))

        n_samples = len(x)
        n_neighbors = min(self.n_neighbors + 1, n_samples)
        self.neighbor_model_ = NearestNeighbors(n_neighbors=n_neighbors)
        self.neighbor_model_.fit(x)
        distances, indices = self.neighbor_model_.kneighbors(x)

        rows = np.repeat(np.arange(n_samples), n_neighbors - 1)
        cols = indices[:, 1:].reshape(-1)
        data = distances[:, 1:].reshape(-1)
        graph = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        graph = graph.minimum(graph.T) + graph.maximum(graph.T)

        train_geodesic = shortest_path(graph, directed=False, unweighted=False)
        self.train_geodesic_ = train_geodesic.astype(np.float32)
        finite = self.train_geodesic_[np.isfinite(self.train_geodesic_)]
        finite = finite[finite > 0]
        self.bandwidth_ = float(self.bandwidth or (np.median(finite) if len(finite) else 1.0))
        return self

    def _subsample(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.max_train is None or len(x) <= self.max_train:
            return x, y
        rng = np.random.default_rng(self.random_state)
        selected = []
        per_class = max(1, self.max_train // len(set(y.tolist())))
        for cls in sorted(set(y.tolist())):
            cls_idx = np.flatnonzero(y == cls)
            take = min(per_class, len(cls_idx))
            selected.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
        remaining = self.max_train - len(selected)
        if remaining > 0:
            rest = np.setdiff1d(np.arange(len(x)), np.array(selected), assume_unique=False)
            selected.extend(rng.choice(rest, size=remaining, replace=False).tolist())
        selected = np.array(sorted(selected))
        return x[selected], y[selected]

    def _scores(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n_neighbors = min(self.n_neighbors, len(self.x_train_))
        edge_distances, edge_indices = self.neighbor_model_.kneighbors(x, n_neighbors=n_neighbors)
        scores = np.zeros((len(x), len(self.classes_)), dtype=np.float64)
        for row_idx, (dist_row, index_row) in enumerate(zip(edge_distances, edge_indices)):
            geo = np.min(dist_row[:, None] + self.train_geodesic_[index_row], axis=0)
            weights = np.exp(-geo / max(self.bandwidth_, 1e-8))
            weights[~np.isfinite(weights)] = 0.0
            for class_idx, cls in enumerate(self.classes_):
                mask = self.y_train_ == cls
                scores[row_idx, class_idx] = float(weights[mask].mean())
        return scores

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        scores = self._scores(x)
        class_list = self.classes_.tolist()
        return np.log(scores[:, class_list.index(1)] + 1e-12) - np.log(
            scores[:, class_list.index(0)] + 1e-12
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self._scores(x)
        scores = scores / np.maximum(scores.sum(axis=1, keepdims=True), 1e-12)
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self._scores(x), axis=1)]
