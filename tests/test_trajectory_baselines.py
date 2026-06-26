from __future__ import annotations

import numpy as np
import pytest

from experiments.trajectory_baselines import (
    GEOMETRY_FEATURES,
    GEOMETRY_PROBES,
    TARGET_TYPES,
    best_rows,
    cross_validated_auroc,
    featurize,
    load_stances,
    parse_csv,
    probe_applies,
    probe_family,
)
from geoprobe.geometry.trajectories import (
    curve_geometry,
    curve_summary_features,
    path_stat_features,
    turning_angle_features,
)


def test_turning_angle_features_has_stable_shape_for_short_paths():
    path = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    features = turning_angle_features(path)

    assert features.shape == (15,)
    assert np.all(np.isfinite(features))
    assert np.allclose(features, 0.0)


def test_turning_angle_features_detects_right_angle():
    path = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )
    features = turning_angle_features(path)

    assert features.shape == (15,)
    assert np.isclose(features[0], np.pi / 2)
    assert np.isclose(features[1], np.pi / 2)


def test_path_stat_features_tracks_final_turn():
    path = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    features = path_stat_features(path)

    assert features.shape == (8,)
    assert np.isclose(features[1], 5.0)
    assert np.isclose(features[3], 7.0)
    assert np.isclose(features[-1], 2.0)


def test_curve_geometry_straight_line_has_no_turning():
    path = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    geometry = curve_geometry(path)

    assert np.allclose(geometry.speeds, [1.0, 2.0])
    assert np.isclose(geometry.path_length, 3.0)
    assert np.isclose(geometry.chord_length, 3.0)
    assert np.isclose(geometry.efficiency, 1.0)
    assert np.allclose(geometry.turn_angles, [0.0])


def test_curve_summary_features_tracks_global_shape():
    path = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    features = curve_summary_features(path)

    assert features.shape == (15,)
    assert np.isclose(features[0], 7.0)
    assert np.isclose(features[1], 5.0)
    assert np.isclose(features[2], 5.0 / 7.0)


def test_featurize_path_flat_preserves_order():
    paths = [
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    ]
    features = featurize(paths, "path_flat")

    assert features.tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]


def test_featurize_shape_invariant_features_have_expected_sizes():
    paths = [
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
    ]

    assert featurize(paths, "relative_path").shape == (1, 6)
    assert featurize(paths, "centered_path").shape == (1, 6)
    assert featurize(paths, "velocity").shape == (1, 4)
    assert featurize(paths, "acceleration").shape == (1, 2)
    assert featurize(paths, "direction").shape == (1, 4)
    assert featurize(paths, "gram").shape == (1, 6)
    assert featurize(paths, "distances").shape == (1, 3)


def test_featurize_rejects_unknown_feature():
    with pytest.raises(ValueError, match="unknown feature set"):
        featurize([np.zeros((2, 2), dtype=np.float32)], "not_a_feature")


def test_load_stances_groups_turns_by_conversation(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        "\n".join(
            [
                '{"conversation_id": "c1", "turn_index": 1, "stance": "accepts"}',
                '{"conversation_id": "c1", "turn_index": 0, "stance": "rejects"}',
                '{"conversation_id": "c2", "turn_index": 0, "stance": "rejects"}',
            ]
        )
        + "\n"
    )

    stances = load_stances(labels)

    assert stances == {
        "c1": {0: "rejects", 1: "accepts"},
        "c2": {0: "rejects"},
    }


def test_cross_validated_auroc_returns_none_when_class_count_too_small():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels = np.array([0, 0, 0, 0, 0, 1])
    groups = np.array([f"g{i}" for i in range(6)])

    assert cross_validated_auroc(
        "linear",
        features,
        labels,
        groups,
        device=None,
        torch_epochs=1,
        torch_learning_rate=1e-3,
        torch_hidden_features=4,
    ) is None


def test_riemannian_probe_runs_on_flat_paths():
    paths = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.1], [1.0, 0.1], [2.0, 0.1]],
            [[0.0, 0.2], [1.0, 0.2], [2.0, 0.2]],
            [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],
            [[0.1, 1.0], [0.1, 2.0], [0.1, 3.0]],
            [[0.2, 1.0], [0.2, 2.0], [0.2, 3.0]],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array([f"g{i}" for i in range(6)])

    score = cross_validated_auroc(
        "torch_riemannian",
        paths.reshape(6, -1),
        labels,
        groups,
        device="cpu",
        torch_epochs=2,
        torch_learning_rate=1e-3,
        torch_hidden_features=4,
        path_shape=(3, 2),
    )

    assert score is not None
    assert 0.0 <= score <= 1.0


def test_riemannian_probe_requires_path_shape():
    features = np.arange(36, dtype=np.float32).reshape(6, 6)
    labels = np.array([0, 0, 0, 1, 1, 1])
    groups = np.array([f"g{i}" for i in range(6)])

    assert cross_validated_auroc(
        "torch_riemannian",
        features,
        labels,
        groups,
        device="cpu",
        torch_epochs=1,
        torch_learning_rate=1e-3,
        torch_hidden_features=4,
    ) is None


def test_best_rows_ignores_errors_and_none_scores():
    results = {
        "final": {
            "0": {"probes": {"linear": None}},
            "4": {"probes": {"linear": 0.62}},
            "8": {"probes": {"linear": {"error": "ValueError"}}},
        }
    }

    assert best_rows(results, ["linear"]) == [
        {
            "feature": "final",
            "probe": "linear",
            "family": "euclidean",
            "best_layer": 4,
            "best_auroc": 0.62,
        }
    ]


def test_probe_family_marks_torch_probes_as_euclidean():
    assert probe_family("torch_mlp") == "euclidean"
    assert probe_family("torch_linear") == "euclidean"
    assert probe_family("torch_riemannian") == "riemannian_approx"


def test_geometry_aliases_expand_to_full_lists():
    assert parse_csv("geometry_full", {"geometry_full": GEOMETRY_FEATURES}) == GEOMETRY_FEATURES
    assert parse_csv("geometry_full", {"geometry_full": GEOMETRY_PROBES}) == GEOMETRY_PROBES


def test_riemannian_probe_only_applies_to_path_features():
    assert probe_applies("torch_riemannian", "path_flat")
    assert probe_applies("torch_riemannian", "relative_path")
    assert not probe_applies("torch_riemannian", "curvature")


def test_target_types_only_keep_flip_and_steadfast_correct():
    assert TARGET_TYPES == {"sycophantic_flip": 1, "steadfast_correct": 0}
