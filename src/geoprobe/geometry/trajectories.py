"""Geometry primitives for activation trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

EPS = 1e-10


@dataclass(frozen=True)
class CurveGeometry:
    """Extrinsic geometry of a discrete activation-space curve."""

    points: np.ndarray
    velocities: np.ndarray
    speeds: np.ndarray
    accelerations: np.ndarray
    turn_cosines: np.ndarray
    turn_angles: np.ndarray
    curvatures: np.ndarray
    path_length: float
    chord_length: float
    efficiency: float
    energy: float


def as_curve(path: np.ndarray) -> np.ndarray:
    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"expected a 2D curve array, got shape {points.shape}")
    if len(points) == 0:
        raise ValueError("curve must contain at least one point")
    return points


def curve_geometry(path: np.ndarray) -> CurveGeometry:
    points = as_curve(path)
    velocities = np.diff(points, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(velocities, axis=0)

    if len(velocities) >= 2:
        left = velocities[:-1]
        right = velocities[1:]
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        denom = left_norm * right_norm
        turn_cosines = np.zeros(len(left), dtype=np.float64)
        valid = denom > EPS
        turn_cosines[valid] = np.sum(left[valid] * right[valid], axis=1) / denom[valid]
        turn_cosines = np.clip(turn_cosines, -1.0, 1.0)
        turn_angles = np.arccos(turn_cosines)
        curvatures = np.zeros_like(turn_angles)
        curvatures[valid] = turn_angles[valid] / np.maximum(left_norm[valid], EPS)
    else:
        turn_cosines = np.zeros(0, dtype=np.float64)
        turn_angles = np.zeros(0, dtype=np.float64)
        curvatures = np.zeros(0, dtype=np.float64)

    path_length = float(speeds.sum())
    chord_length = float(np.linalg.norm(points[-1] - points[0]))
    efficiency = float(chord_length / path_length) if path_length > EPS else 0.0
    energy = float(np.sum(speeds**2))

    return CurveGeometry(
        points=points,
        velocities=velocities,
        speeds=speeds,
        accelerations=accelerations,
        turn_cosines=turn_cosines,
        turn_angles=turn_angles,
        curvatures=curvatures,
        path_length=path_length,
        chord_length=chord_length,
        efficiency=efficiency,
        energy=energy,
    )


def path_stat_features(path: np.ndarray) -> np.ndarray:
    geometry = curve_geometry(path)
    points = geometry.points
    start = points[0]
    path_length = 0.0
    previous_step = None
    last_features = np.zeros(8, dtype=np.float64)

    for turn_index, current in enumerate(points):
        norm = float(np.linalg.norm(current))
        displacement = float(np.linalg.norm(current - start))
        step_norm = 0.0
        turn_cosine = 0.0
        if turn_index > 0:
            step = current - points[turn_index - 1]
            step_norm = float(np.linalg.norm(step))
            path_length += step_norm
            if (
                previous_step is not None
                and np.linalg.norm(previous_step) > EPS
                and step_norm > EPS
            ):
                turn_cosine = float(
                    np.dot(step, previous_step) / (np.linalg.norm(previous_step) * step_norm)
                )
            previous_step = step
        efficiency = float(displacement / path_length) if path_length > EPS else 0.0
        mean_step = float(path_length / turn_index) if turn_index > 0 else 0.0
        last_features = np.asarray(
            [
                norm,
                displacement,
                step_norm,
                path_length,
                efficiency,
                turn_cosine,
                mean_step,
                float(turn_index),
            ],
            dtype=np.float64,
        )
    return last_features


def turning_angle_features(path: np.ndarray, max_turns: int = 4) -> np.ndarray:
    geometry = curve_geometry(path)
    theta = np.zeros(max_turns, dtype=np.float64)
    kappa = np.zeros(max_turns, dtype=np.float64)
    n = min(max_turns, len(geometry.turn_angles))
    if n:
        theta[:n] = geometry.turn_angles[:n]
        kappa[:n] = geometry.curvatures[:n]

    features: list[float] = []
    for turn_index in range(max_turns):
        features.extend([float(theta[turn_index]), float(kappa[turn_index])])

    features.extend(
        [
            float(theta.mean()),
            float(theta.std()),
            float(kappa.mean()),
            float(kappa.std()),
            float(kappa.max()) if len(kappa) else 0.0,
            float(theta[0]) if len(theta) else 0.0,
            float(kappa[0]) if len(kappa) else 0.0,
        ]
    )
    return np.asarray(features, dtype=np.float64)


def curve_summary_features(path: np.ndarray) -> np.ndarray:
    geometry = curve_geometry(path)
    speeds = geometry.speeds
    angles = geometry.turn_angles
    curvatures = geometry.curvatures
    accelerations = (
        np.linalg.norm(geometry.accelerations, axis=1)
        if len(geometry.accelerations)
        else np.zeros(0, dtype=np.float64)
    )

    return np.asarray(
        [
            geometry.path_length,
            geometry.chord_length,
            geometry.efficiency,
            geometry.energy,
            float(speeds.mean()) if len(speeds) else 0.0,
            float(speeds.std()) if len(speeds) else 0.0,
            float(speeds.max()) if len(speeds) else 0.0,
            float(accelerations.mean()) if len(accelerations) else 0.0,
            float(accelerations.std()) if len(accelerations) else 0.0,
            float(angles.mean()) if len(angles) else 0.0,
            float(angles.std()) if len(angles) else 0.0,
            float(angles.max()) if len(angles) else 0.0,
            float(curvatures.mean()) if len(curvatures) else 0.0,
            float(curvatures.std()) if len(curvatures) else 0.0,
            float(curvatures.max()) if len(curvatures) else 0.0,
        ],
        dtype=np.float64,
    )


def final_points(paths: list[np.ndarray]) -> np.ndarray:
    return np.stack([as_curve(path)[-1] for path in paths])


def mean_points(paths: list[np.ndarray]) -> np.ndarray:
    return np.stack([as_curve(path).mean(axis=0) for path in paths])


def displacements(paths: list[np.ndarray]) -> np.ndarray:
    return np.stack([as_curve(path)[-1] - as_curve(path)[0] for path in paths])


def flatten_paths(paths: list[np.ndarray]) -> np.ndarray:
    return np.stack([as_curve(path).reshape(-1) for path in paths])


def relative_paths(paths: list[np.ndarray]) -> np.ndarray:
    return np.stack([(as_curve(path) - as_curve(path)[0]).reshape(-1) for path in paths])


def centered_paths(paths: list[np.ndarray]) -> np.ndarray:
    rows = []
    for path in paths:
        points = as_curve(path)
        rows.append((points - points.mean(axis=0, keepdims=True)).reshape(-1))
    return np.stack(rows)


def velocity_paths(paths: list[np.ndarray]) -> np.ndarray:
    return np.stack([curve_geometry(path).velocities.reshape(-1) for path in paths])


def acceleration_paths(paths: list[np.ndarray]) -> np.ndarray:
    return np.stack([curve_geometry(path).accelerations.reshape(-1) for path in paths])


def direction_paths(paths: list[np.ndarray]) -> np.ndarray:
    rows = []
    for path in paths:
        velocities = curve_geometry(path).velocities
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        rows.append((velocities / np.maximum(norms, EPS)).reshape(-1))
    return np.stack(rows)


def gram_features(path: np.ndarray) -> np.ndarray:
    points = as_curve(path)
    centered = points - points.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    tri = np.triu_indices(len(points))
    return gram[tri].astype(np.float64)


def distance_matrix_features(path: np.ndarray) -> np.ndarray:
    points = as_curve(path)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    tri = np.triu_indices(len(points), k=1)
    return distances[tri].astype(np.float64)


FEATURE_BUILDERS: dict[str, Callable[[list[np.ndarray]], np.ndarray]] = {
    "final": final_points,
    "mean": mean_points,
    "delta": displacements,
    "path_flat": flatten_paths,
    "relative_path": relative_paths,
    "centered_path": centered_paths,
    "velocity": velocity_paths,
    "acceleration": acceleration_paths,
    "direction": direction_paths,
    "gram": lambda paths: np.stack([gram_features(path) for path in paths]),
    "distances": lambda paths: np.stack([distance_matrix_features(path) for path in paths]),
    "path_stats": lambda paths: np.stack([path_stat_features(path) for path in paths]),
    "curvature": lambda paths: np.stack([turning_angle_features(path) for path in paths]),
    "curve_summary": lambda paths: np.stack([curve_summary_features(path) for path in paths]),
}


def featurize_paths(paths: list[np.ndarray], feature_name: str) -> np.ndarray:
    try:
        return FEATURE_BUILDERS[feature_name](paths)
    except KeyError as exc:
        raise ValueError(f"unknown feature set: {feature_name}") from exc
