from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.bootstrap_ci import clustered_bootstrap


def test_point_estimate_equals_pooled_auroc():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    oof = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.05, 0.6])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    point, lo, hi, n_ok = clustered_bootstrap(y, oof, groups, n_boot=200, seed=0)
    assert abs(point - roc_auc_score(y, oof)) < 1e-12
    assert 0.0 <= lo <= point <= hi <= 1.0
    assert n_ok > 0


def test_perfect_separation_has_point_auroc_one():
    y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    oof = np.array([0.1, 0.2, 0.8, 0.9, 0.15, 0.25, 0.85, 0.95])
    groups = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    point, lo, hi, _ = clustered_bootstrap(y, oof, groups, n_boot=200, seed=0)
    assert point == 1.0
    assert hi == 1.0


def test_nan_out_of_fold_scores_are_dropped():
    y = np.array([0, 1, 0, 1])
    oof = np.array([0.2, 0.8, np.nan, 0.6])
    groups = np.array(["a", "a", "b", "b"])
    point, _, _, _ = clustered_bootstrap(y, oof, groups, n_boot=50, seed=0)
    mask = ~np.isnan(oof)
    assert abs(point - roc_auc_score(y[mask], oof[mask])) < 1e-12
