"""Trajectory-level probe baselines for SYCON."""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.label_sycon_flips import trajectory_type
from geoprobe.geometry.trajectories import featurize_paths
from geoprobe.probes.registry import FAMILY, build_probe

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def git_provenance(extra_paths: list[str | Path] | None = None) -> dict:
    """Best-effort commit hash + dirty flag + optional file hashes for run provenance."""
    import hashlib
    import subprocess

    try:
        repo = str(Path(__file__).resolve().parents[1])
        commit = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", repo, "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        file_sha256 = {}
        for path in extra_paths or []:
            p = Path(path)
            file_sha256[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return {"git_hash": commit or None, "git_dirty": dirty, "file_sha256": file_sha256}
    except Exception:
        return {"git_hash": None, "git_dirty": None, "file_sha256": {}}


TARGET_TYPES = {"sycophantic_flip": 1, "steadfast_correct": 0}
GEOMETRY_FEATURES = [
    "final",
    "mean",
    "delta",
    "path_flat",
    "relative_path",
    "centered_path",
    "velocity",
    "acceleration",
    "direction",
    "gram",
    "distances",
    "path_stats",
    "curvature",
    "curve_summary",
]
GEOMETRY_PROBES = [
    "linear",
    "torch_mlp",
    "mahalanobis",
    "class_mahalanobis",
    "centroid",
    "knn",
    "tangent_subspace",
    "graph_geodesic",
    "torch_riemannian",
]
DEFAULT_FEATURES = "geometry_full"
DEFAULT_PROBES = "geometry_full"
TORCH_PROBES = {"torch_linear", "torch_mlp", "torch_riemannian"}
PATH_FEATURES = {"path_flat", "relative_path", "centered_path"}


@dataclass(frozen=True)
class TrajectoryDataset:
    turns: dict
    labels: np.ndarray
    groups: np.ndarray
    conversation_ids: list[str]
    conversation_rows: dict[str, list[tuple[int, int]]]

    @property
    def n_sycophantic_flip(self) -> int:
        return int(self.labels.sum())

    @property
    def n_steadfast_correct(self) -> int:
        return int((1 - self.labels).sum())


def parse_csv(value: str, aliases: dict[str, list[str]] | None = None) -> list[str]:
    if aliases and value in aliases:
        return aliases[value]
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_stances(labels_path: Path) -> dict[str, dict[int, str]]:
    stances: dict[str, dict[int, str]] = defaultdict(dict)
    for line in labels_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        stances[row["conversation_id"]][int(row["turn_index"])] = row["stance"]
    return dict(stances)


def group_activation_rows(turns: dict) -> dict[str, list[tuple[int, int]]]:
    rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    turn_indices = turns["turn_index"].tolist()
    for row_idx, (conversation_id, turn_index) in enumerate(
        zip(turns["conversation_id"], turn_indices)
    ):
        rows[conversation_id].append((int(turn_index), row_idx))
    return {cid: sorted(items) for cid, items in rows.items()}


def paired_group_id(conversation_id: str) -> str:
    """Keep neutral/pressured arms of one synthetic scenario in one fold."""
    stem, sep, suffix = conversation_id.rpartition("_")
    if sep and suffix in {"n", "p"}:
        return stem
    return conversation_id


def load_dataset(config: dict, labels_path: Path) -> TrajectoryDataset:
    turns_path = Path(config["activations"]["output_dir"]) / "turns.pt"
    turns = torch.load(turns_path, map_location="cpu", weights_only=False)
    stances = load_stances(labels_path)
    conversation_rows = group_activation_rows(turns)

    labels: list[int] = []
    groups: list[str] = []
    conversation_ids: list[str] = []
    for conversation_id, turnmap in stances.items():
        if conversation_id not in conversation_rows:
            continue
        ordered_stances = [turnmap[t] for t in sorted(turnmap)]
        label = TARGET_TYPES.get(trajectory_type(ordered_stances))
        if label is None:
            continue
        conversation_ids.append(conversation_id)
        labels.append(label)
        groups.append(paired_group_id(conversation_id))

    return TrajectoryDataset(
        turns=turns,
        labels=np.asarray(labels, dtype=int),
        groups=np.asarray(groups),
        conversation_ids=conversation_ids,
        conversation_rows=conversation_rows,
    )


def layer_paths(dataset: TrajectoryDataset, layer: int) -> list[np.ndarray]:
    activations = dataset.turns["activations"][layer].numpy().astype(np.float32)
    paths = []
    for conversation_id in dataset.conversation_ids:
        rows = dataset.conversation_rows[conversation_id]
        paths.append(np.stack([activations[row_idx] for _, row_idx in rows]))
    return paths


def featurize(paths: list[np.ndarray], feature_name: str) -> np.ndarray:
    return featurize_paths(paths, feature_name)


def positive_scores(probe, features: np.ndarray) -> np.ndarray:
    if hasattr(probe, "decision_function"):
        return probe.decision_function(features)
    classes = probe.steps[-1][1].classes_ if hasattr(probe, "steps") else probe.classes_
    return probe.predict_proba(features)[:, list(classes).index(1)]


class TorchBinaryProbe(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None):
        super().__init__()
        if hidden_features is None:
            self.net = nn.Linear(in_features, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, 1),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class TorchDiagonalRiemannianPathProbe(nn.Module):
    def __init__(self, n_turns: int, activation_dim: int, hidden_features: int):
        super().__init__()
        self.n_turns = n_turns
        self.activation_dim = activation_dim
        self.metric_net = nn.Sequential(
            nn.Linear(activation_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, activation_dim),
        )
        n_steps = n_turns - 1
        geometry_dim = n_steps * 4 + 4
        self.classifier = nn.Sequential(
            nn.Linear(geometry_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        path = features.reshape(-1, self.n_turns, self.activation_dim)
        velocities = path[:, 1:] - path[:, :-1]
        midpoints = 0.5 * (path[:, 1:] + path[:, :-1])
        metric_diag = F.softplus(self.metric_net(midpoints)) + 1e-6
        metric_energy = (metric_diag * velocities.square()).sum(dim=-1)
        metric_length = torch.sqrt(metric_energy + 1e-6)
        euclidean_speed = velocities.norm(dim=-1)

        if velocities.shape[1] > 1:
            left = velocities[:, :-1]
            right = velocities[:, 1:]
            denom = left.norm(dim=-1) * right.norm(dim=-1) + 1e-6
            turn_cosine = ((left * right).sum(dim=-1) / denom).clamp(-1.0, 1.0)
            turn_cosine = F.pad(turn_cosine, (0, 1))
        else:
            turn_cosine = torch.zeros_like(metric_energy)

        displacement = path[:, -1] - path[:, 0]
        meanpoint = path.mean(dim=1)
        displacement_metric = F.softplus(self.metric_net(meanpoint)) + 1e-6
        displacement_energy = (displacement_metric * displacement.square()).sum(dim=-1, keepdim=True)
        euclidean_displacement = displacement.norm(dim=-1, keepdim=True)

        geometry = torch.cat(
            [
                metric_energy,
                metric_length,
                euclidean_speed,
                turn_cosine,
                metric_energy.sum(dim=-1, keepdim=True),
                metric_length.sum(dim=-1, keepdim=True),
                displacement_energy,
                euclidean_displacement,
            ],
            dim=-1,
        )
        return self.classifier(geometry).squeeze(-1)


def torch_train_scores(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight = (len(y_train) - float(y_train.sum())) / max(float(y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    x_train_t = torch.from_numpy(x_train).to(device)
    y_train_t = torch.from_numpy(y_train.astype(np.float32)).to(device)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_test).to(device)).detach().cpu().numpy()


def torch_cross_validated_auroc(
    probe_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    hidden_features: int,
    path_shape: tuple[int, int] | None,
) -> float | None:
    class_counts = np.bincount(labels, minlength=2)
    n_splits = min(5, int(class_counts.min()), len(set(groups.tolist())))
    if n_splits < 2:
        return None
    if probe_name == "torch_riemannian" and path_shape is None:
        return None

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    out_of_fold = np.full(len(labels), np.nan)
    hidden = None if probe_name == "torch_linear" else hidden_features

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(features, labels, groups)):
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[test_idx].tolist())) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx]).astype(np.float32)
        x_test = scaler.transform(features[test_idx]).astype(np.float32)
        y_train = labels[train_idx].astype(np.float32)

        if probe_name == "torch_riemannian":
            assert path_shape is not None
            model = TorchDiagonalRiemannianPathProbe(
                n_turns=path_shape[0],
                activation_dim=path_shape[1],
                hidden_features=hidden_features,
            )
        else:
            model = TorchBinaryProbe(x_train.shape[1], hidden)

        out_of_fold[test_idx] = torch_train_scores(
            model,
            x_train,
            y_train,
            x_test,
            device,
            epochs,
            learning_rate,
            seed=fold_idx,
        )

    valid = ~np.isnan(out_of_fold)
    if valid.sum() == 0 or len(set(labels[valid].tolist())) < 2:
        return None
    return float(roc_auc_score(labels[valid], out_of_fold[valid]))


def cross_validated_auroc(
    probe_name: str,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    device: torch.device,
    torch_epochs: int,
    torch_learning_rate: float,
    torch_hidden_features: int,
    path_shape: tuple[int, int] | None = None,
) -> float | None:
    if probe_name in TORCH_PROBES:
        return torch_cross_validated_auroc(
            probe_name,
            features,
            labels,
            groups,
            device,
            torch_epochs,
            torch_learning_rate,
            torch_hidden_features,
            path_shape,
        )

    class_counts = np.bincount(labels, minlength=2)
    n_splits = min(5, int(class_counts.min()), len(set(groups.tolist())))
    if n_splits < 2:
        return None

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    out_of_fold = np.full(len(labels), np.nan)
    for train_idx, test_idx in splitter.split(features, labels, groups):
        if len(set(labels[train_idx].tolist())) < 2 or len(set(labels[test_idx].tolist())) < 2:
            continue
        try:
            probe = build_probe(probe_name)
            probe.fit(features[train_idx], labels[train_idx])
            out_of_fold[test_idx] = positive_scores(probe, features[test_idx])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue

    valid = ~np.isnan(out_of_fold)
    if valid.sum() == 0 or len(set(labels[valid].tolist())) < 2:
        return None
    return float(roc_auc_score(labels[valid], out_of_fold[valid]))


def probe_family(probe_name: str) -> str | None:
    if probe_name == "torch_riemannian":
        return "riemannian_approx"
    if probe_name in TORCH_PROBES:
        return "euclidean"
    return FAMILY.get(probe_name)


def probe_applies(probe_name: str, feature_name: str) -> bool:
    if probe_name == "torch_riemannian":
        return feature_name in PATH_FEATURES
    return True


def best_rows(results: dict, probes: list[str]) -> list[dict]:
    rows = []
    for feature_name, by_layer in results.items():
        for probe_name in probes:
            candidates = []
            for layer, layer_result in by_layer.items():
                score = layer_result["probes"].get(probe_name)
                if isinstance(score, float):
                    candidates.append((int(layer), score))
            best_layer, best_auroc = max(candidates, key=lambda item: item[1], default=(None, None))
            rows.append(
                {
                    "feature": feature_name,
                    "probe": probe_name,
                    "family": probe_family(probe_name),
                    "best_layer": best_layer,
                    "best_auroc": best_auroc,
                }
            )
    return rows


def infer_scope(labels_path: Path) -> str:
    return "kcfalse" if "kcfalse" in labels_path.name else "all"


def run(
    config: dict,
    labels_path: Path,
    feature_names: list[str],
    probe_names: list[str],
    device: torch.device,
    torch_epochs: int,
    torch_learning_rate: float,
    torch_hidden_features: int,
) -> dict:
    dataset = load_dataset(config, labels_path)
    by_feature: dict[str, dict] = {name: {} for name in feature_names}
    total_jobs = sum(
        1
        for _ in dataset.turns["layers"]
        for feature_name in feature_names
        for probe_name in probe_names
        if probe_applies(probe_name, feature_name)
    )
    completed_jobs = 0
    started_at = time.perf_counter()

    for layer in dataset.turns["layers"]:
        paths = layer_paths(dataset, int(layer))
        for feature_name in feature_names:
            features = featurize(paths, feature_name)
            path_shape = paths[0].shape if feature_name in PATH_FEATURES else None
            layer_result = {
                "n_features": int(features.shape[1]),
                "path_shape": list(path_shape) if path_shape is not None else None,
                "probes": {},
            }
            for probe_name in probe_names:
                if not probe_applies(probe_name, feature_name):
                    continue
                completed_jobs += 1
                job_started_at = time.perf_counter()
                print(
                    f"[{completed_jobs:03d}/{total_jobs:03d}] "
                    f"L{int(layer):02d} {feature_name}:{probe_name} "
                    f"(n={len(dataset.labels)}, d={features.shape[1]})",
                    flush=True,
                )
                try:
                    auroc = cross_validated_auroc(
                        probe_name,
                        features,
                        dataset.labels,
                        dataset.groups,
                        device,
                        torch_epochs,
                        torch_learning_rate,
                        torch_hidden_features,
                        path_shape,
                    )
                    layer_result["probes"][probe_name] = (
                        round(auroc, 4) if auroc is not None else None
                    )
                    elapsed = time.perf_counter() - job_started_at
                    score = f"{auroc:.4f}" if auroc is not None else "n/a"
                    print(f"    -> {score} in {elapsed:.1f}s", flush=True)
                except Exception as exc:
                    layer_result["probes"][probe_name] = {
                        "error": type(exc).__name__,
                        "message": str(exc)[:300],
                    }
                    elapsed = time.perf_counter() - job_started_at
                    print(f"    -> ERROR {type(exc).__name__} in {elapsed:.1f}s", flush=True)
            by_feature[feature_name][str(layer)] = layer_result

    print(f"finished {total_jobs} jobs in {time.perf_counter() - started_at:.1f}s", flush=True)

    return {
        "model_name": dataset.turns.get("model_name"),
        "torch_device": str(device),
        "n": int(len(dataset.labels)),
        "n_sf": dataset.n_sycophantic_flip,
        "n_sc": dataset.n_steadfast_correct,
        "by_feature": by_feature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--scope")
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--probes", default=DEFAULT_PROBES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-epochs", type=int, default=250)
    parser.add_argument("--torch-lr", type=float, default=1e-3)
    parser.add_argument("--torch-hidden", type=int, default=64)
    args = parser.parse_args()

    config_path = Path(args.config)
    labels_path = Path(args.labels)
    config = yaml.safe_load(config_path.read_text())
    feature_names = parse_csv(args.features, {"geometry_full": GEOMETRY_FEATURES})
    probe_names = parse_csv(args.probes, {"geometry_full": GEOMETRY_PROBES})
    device = resolve_device(args.device)

    result = run(
        config,
        labels_path,
        feature_names,
        probe_names,
        device,
        args.torch_epochs,
        args.torch_lr,
        args.torch_hidden,
    )
    result.update(
        {
            "config": config["name"],
            "labels_path": str(labels_path),
            "scope": args.scope or infer_scope(labels_path),
            "task": "sycophantic_flip_vs_steadfast_correct",
            "grouping": "paired_scenario_id",
            "features": feature_names,
            "probes": probe_names,
        }
    )
    result["summary"] = best_rows(result["by_feature"], probe_names)

    output_dir = Path(config["eval"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"trajectory_baselines_{result['scope']}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))

    print(f"\n{config['name']} {result['scope']}: n={result['n']} sf={result['n_sf']} sc={result['n_sc']}")
    top_rows = sorted(
        result["summary"],
        key=lambda row: (row["best_auroc"] is None, -(row["best_auroc"] or -1.0)),
    )[:12]
    for row in top_rows:
        score = row["best_auroc"] if row["best_auroc"] is not None else "n/a"
        print(f"{row['feature']:10s} {row['probe']:14s} {score} @ L{row['best_layer']}")
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
