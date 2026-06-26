"""Point-cloud control frontier for the graded deception-pressure ramp.

Fits a within-level deception direction from train families only,

    d_p = mean(deceptive | scenario, level) - mean(honest | scenario, level),

then regenerates held-out final reports under raw linear and local-tangent steering.  This is the
control counterpart to ``within_level_dp_rotation.py``: the question is not AUROC, but whether a
geometric/tangent nudge improves the fix-vs-harm/coherence frontier at matched injected norm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.control_deception_intent_transition import (  # noqa: E402
    injection_stats,
    prompt_without_final_answer,
    reply_coherence,
)
from experiments.eval_deception_intent_controlled import clean_matrix  # noqa: E402
from experiments.rollout_deception_intent import parse_status  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.models import ResidualSteeringSpec, cleanup, load_activation_pipeline  # noqa: E402


METHOD_ALIASES = {
    "baseline": "baseline",
    "linear": "dp_linear",
    "dp": "dp_linear",
    "dp_linear": "dp_linear",
    "tangent": "dp_tangent",
    "dp_tangent": "dp_tangent",
    "off_tangent": "dp_off_tangent",
    "dp_off_tangent": "dp_off_tangent",
}
STEER_METHODS = {"dp_linear", "dp_tangent", "dp_off_tangent"}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def canonical_methods(value: str) -> list[str]:
    out: list[str] = []
    for item in parse_csv(value):
        if item not in METHOD_ALIASES:
            raise ValueError(f"unknown method {item!r}; expected one of {sorted(METHOD_ALIASES)}")
        method = METHOD_ALIASES[item]
        if method not in out:
            out.append(method)
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def to_jsonable(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def read_jsonl_paths(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = str(row.get("conversation_id", ""))
            if cid and cid in seen:
                continue
            if cid:
                seen.add(cid)
            rows.append(row)
    return rows


def clean_id_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def unit(vec: np.ndarray) -> np.ndarray:
    vec = clean_matrix(np.asarray(vec, dtype=np.float64))
    scale = float(np.max(np.abs(vec))) if vec.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(vec)
    scaled = vec / scale
    norm = float(np.linalg.norm(scaled))
    return scaled / norm if np.isfinite(norm) and norm > 1e-12 else np.zeros_like(vec)


def vector_stats(vec: np.ndarray | torch.Tensor) -> dict:
    arr = vec.detach().float().cpu().numpy() if torch.is_tensor(vec) else np.asarray(vec, dtype=np.float64)
    arr = clean_matrix(arr)
    return {
        "norm": float(np.linalg.norm(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "max_abs": float(np.abs(arr).max()),
    }


def config_defaults(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    model_cfg = cfg.get("model", {})
    gen_cfg = cfg.get("generation", {})
    act_cfg = cfg.get("activations", {})
    return {
        "model_key": model_cfg.get("name", "llama31_8b_instruct"),
        "backend": model_cfg.get("backend", "auto"),
        "dtype": model_cfg.get("dtype", "float16"),
        "mlx_model": model_cfg.get("mlx_model"),
        "max_length": int(act_cfg.get("max_length", model_cfg.get("max_length", 2048))),
        "max_new_tokens": int(gen_cfg.get("final_max_new_tokens", gen_cfg.get("max_new_tokens", 120))),
        "max_generation_seconds": float(gen_cfg.get("max_generation_seconds", 60.0)),
        "temperature": 0.0,
        "top_p": float(gen_cfg.get("top_p", 1.0)),
    }


def arr(values) -> np.ndarray:
    return values.numpy() if torch.is_tensor(values) else np.asarray(values)


def load_activation_points(
    path: Path,
    *,
    layer: int,
    turns: set[int] | None = None,
    phases: set[str] | None = None,
    levels: set[str] | None = None,
) -> tuple[list[dict], dict]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if layer not in data["activations"]:
        raise ValueError(f"layer {layer} not in activation file; available={sorted(data['activations'])}")
    cids = np.asarray(data["conversation_id"]).astype(str)
    scenarios = np.asarray(data["scenario_id"]).astype(str)
    families = np.asarray(data["family"]).astype(str)
    arms = np.asarray(data["arm"]).astype(str)
    phases_all = np.asarray(data["phase"]).astype(str)
    true_status = np.asarray(data["true_status"]).astype(str)
    desired_status = np.asarray(data["desired_status"]).astype(str)
    sample_seed = arr(data["sample_seed"]).astype(int)
    turn_index = arr(data["turn_index"]).astype(int)
    labels = arr(data["deceptive"]).astype(int)
    x = data["activations"][layer]
    x_np = x.float().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)
    points: list[dict] = []
    for idx in range(len(cids)):
        if turns is not None and int(turn_index[idx]) not in turns:
            continue
        if phases is not None and phases_all[idx] not in phases:
            continue
        if levels is not None and arms[idx] not in levels:
            continue
        vec = np.asarray(x_np[idx], dtype=np.float64)
        if not np.isfinite(vec).all():
            continue
        points.append({
            "conversation_id": cids[idx],
            "scenario_id": scenarios[idx],
            "family": families[idx],
            "arm": arms[idx],
            "sample_seed": int(sample_seed[idx]),
            "turn_index": int(turn_index[idx]),
            "phase": phases_all[idx],
            "true_status": true_status[idx],
            "desired_status": desired_status[idx],
            "label": int(labels[idx]),
            "x": vec,
        })
    meta = {
        "model_name": data.get("model_name"),
        "backend": data.get("backend"),
        "device": data.get("device"),
        "layers": data.get("layers"),
        "capture": data.get("capture"),
    }
    return points, meta


def point_index(points: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    duplicates: list[str] = []
    for row in points:
        cid = row["conversation_id"]
        if cid in out:
            duplicates.append(cid)
        out[cid] = row
    if duplicates:
        raise ValueError(f"duplicate activation points for requested turn/phase: {duplicates[:5]}")
    return out


def fit_dp_direction(
    direction_points: list[dict],
    *,
    heldout_family: str,
    direction_levels: set[str],
    min_mixed_scenarios: int,
    min_levels: int,
) -> dict | None:
    train = [
        row for row in direction_points
        if row["family"] != heldout_family and row["arm"] in direction_levels
    ]
    by_level_scenario: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in train:
        by_level_scenario[(row["arm"], row["scenario_id"])].append(row)

    level_diffs: dict[str, list[np.ndarray]] = defaultdict(list)
    scenario_counts: dict[str, list[dict]] = defaultdict(list)
    for (level, scenario_id), group in sorted(by_level_scenario.items()):
        y = np.asarray([row["label"] for row in group], dtype=int)
        if int(y.sum()) < 1 or int(len(y) - y.sum()) < 1:
            continue
        x = clean_matrix(np.vstack([row["x"] for row in group]))
        diff = x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)
        level_diffs[level].append(diff)
        scenario_counts[level].append({
            "scenario_id": scenario_id,
            "n_deceptive": int(y.sum()),
            "n_honest": int(len(y) - y.sum()),
        })

    usable_levels = [level for level in sorted(level_diffs) if len(level_diffs[level]) >= min_mixed_scenarios]
    if len(usable_levels) < min_levels:
        return None

    level_means = [clean_matrix(np.vstack(level_diffs[level])).mean(axis=0) for level in usable_levels]
    dp = unit(clean_matrix(np.vstack(level_means)).mean(axis=0))
    if not np.isfinite(dp).all() or np.linalg.norm(dp) <= 1e-8:
        return None

    direction = -dp
    return {
        "heldout_family": heldout_family,
        "direction_levels": usable_levels,
        "n_train_points": int(len(train)),
        "n_mixed_scenario_level_pairs": int(sum(len(level_diffs[level]) for level in usable_levels)),
        "mixed_scenarios_by_level": {
            level: scenario_counts[level][:100]
            for level in usable_levels
        },
        "dp_stats": vector_stats(dp),
        "direction_convention": "direction = honest - deceptive = -d_p",
        "direction_stats": vector_stats(direction),
        "direction": torch.from_numpy(direction.astype(np.float32)),
        "_direction_np": direction,
    }


def fit_tangent_cloud(
    tangent_points: list[dict],
    *,
    heldout_family: str | None = None,
    heldout_scenario_ids: set[str] | None = None,
) -> dict | None:
    train = [
        row for row in tangent_points
        if (heldout_family is None or row["family"] != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    if len(train) < 4:
        return None
    x = clean_matrix(np.vstack([row["x"] for row in train]))
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    return {
        "heldout_family": heldout_family,
        "n_train": int(len(train)),
        "_train_x": x,
        "_train_scaled": (x - mean) / std,
        "_scale_mean": mean,
        "_scale_std": std,
    }


def project_to_local_tangent(
    raw_direction: np.ndarray,
    tangent_info: dict,
    query: np.ndarray,
    *,
    tangent_neighbors: int,
    tangent_dim: int,
) -> tuple[torch.Tensor | None, dict]:
    train_x = tangent_info["_train_x"]
    train_scaled = tangent_info["_train_scaled"]
    q = clean_matrix(np.asarray(query, dtype=np.float64))
    q_scaled = (q - tangent_info["_scale_mean"]) / tangent_info["_scale_std"]
    k = min(int(tangent_neighbors), len(train_x))
    if k < 2:
        return None, {"reason": "too_few_neighbors", "neighbors": int(k)}
    distances = np.linalg.norm(train_scaled - q_scaled[None, :], axis=1)
    idx = np.argsort(distances)[:k]
    local_scaled = train_scaled[idx]
    local = clean_matrix(local_scaled - local_scaled.mean(axis=0, keepdims=True))
    max_dim = min(int(tangent_dim), local.shape[0] - 1, local.shape[1])
    if max_dim < 1:
        return None, {"reason": "no_tangent_dim", "neighbors": int(k)}
    try:
        _, singular_values, vh = np.linalg.svd(local, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, {"reason": "svd_failed", "neighbors": int(k), "tangent_dim": int(max_dim)}
    basis = clean_matrix(vh[:max_dim])
    raw_unit = unit(raw_direction)
    raw_scaled = clean_matrix(raw_unit / tangent_info["_scale_std"])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected_scaled = clean_matrix(basis.T @ (basis @ raw_scaled))
        projected = clean_matrix(projected_scaled * tangent_info["_scale_std"])
    projected_norm = float(np.linalg.norm(projected))
    raw_norm = float(np.linalg.norm(raw_unit))
    if not np.isfinite(projected_norm) or projected_norm < 1e-8:
        return None, {
            "reason": "zero_projection",
            "neighbors": int(k),
            "tangent_dim": int(max_dim),
            "raw_norm": raw_norm,
            "projected_norm": projected_norm,
        }
    projected_unit = projected / projected_norm
    return torch.from_numpy(projected_unit.astype(np.float32)), {
        "neighbors": int(k),
        "tangent_dim": int(max_dim),
        "raw_norm": raw_norm,
        "projected_norm": projected_norm,
        "projection_fraction": float(projected_norm / max(raw_norm, 1e-12)),
        "cos_to_raw": float(np.dot(projected_unit, raw_unit / max(raw_norm, 1e-12))),
        "mean_neighbor_distance": float(distances[idx].mean()),
        "singular_values": [float(x) for x in singular_values[:max_dim]],
    }


def off_tangent_direction(raw_direction: np.ndarray, tangent_direction: torch.Tensor | None) -> torch.Tensor | None:
    if tangent_direction is None:
        return None
    tangent = tangent_direction.detach().float().cpu().numpy().astype(np.float64)
    residual = clean_matrix(unit(raw_direction) - tangent * float(np.dot(unit(raw_direction), tangent)))
    norm = float(np.linalg.norm(residual))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    return torch.from_numpy((residual / norm).astype(np.float32))


def public_direction_stats(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    return {k: v for k, v in payload.items() if not k.startswith("_") and k != "direction"}


def select_eval_rows(rows: list[dict], *, limit: int | None, strategy: str, seed: int) -> list[dict]:
    rows = sorted(rows, key=lambda row: str(row["conversation_id"]))
    if limit is None or len(rows) <= limit:
        return rows
    if strategy == "first":
        return rows[:limit]
    if strategy == "shuffle":
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(rows), size=limit, replace=False)
        return [rows[int(i)] for i in sorted(idx)]
    if strategy != "family_round_robin":
        raise ValueError(f"unknown limit strategy {strategy!r}")
    by_family_arm: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_family_arm[str(row["family"])][str(row.get("arm", ""))].append(row)
    arm_offsets = {family: 0 for family in by_family_arm}
    chosen: list[dict] = []
    while len(chosen) < limit and any(any(queue for queue in arms.values()) for arms in by_family_arm.values()):
        for family in sorted(by_family_arm):
            arms = by_family_arm[family]
            arm_names = sorted(arms)
            if not arm_names:
                continue
            for offset in range(len(arm_names)):
                arm = arm_names[(arm_offsets[family] + offset) % len(arm_names)]
                if arms[arm]:
                    chosen.append(arms[arm].pop(0))
                    arm_offsets[family] = (arm_names.index(arm) + 1) % len(arm_names)
                    break
            if len(chosen) >= limit:
                break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=28)
    parser.add_argument("--direction-turn", type=int, default=2)
    parser.add_argument("--direction-phase", default="pre_response")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--eval-levels", default="p3,p4,p5")
    parser.add_argument("--direction-levels", default="p3,p4,p5,p6")
    parser.add_argument("--tangent-levels", default="p0,p1,p2,p3,p4,p5,p6")
    parser.add_argument("--tangent-turns", default="0,1,2,3")
    parser.add_argument("--tangent-phases", default="pre_response,post_response")
    parser.add_argument("--min-mixed-scenarios", type=int, default=8)
    parser.add_argument("--min-direction-levels", type=int, default=2)
    parser.add_argument("--methods", default="baseline,dp_linear,dp_tangent")
    parser.add_argument("--alphas", default="2,4,8,16")
    parser.add_argument("--tangent-neighbors", type=int, default=32)
    parser.add_argument("--tangent-dim", type=int, default=8)
    parser.add_argument("--conversation-ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-strategy",
        choices=["first", "shuffle", "family_round_robin"],
        default="family_round_robin",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", choices=["auto", "hf", "mlx"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    transcript_paths = [Path(path) for path in args.transcripts]
    activation_path = Path(args.activations)
    config_path = Path(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    defaults = config_defaults(config_path)
    model_key = args.model or defaults["model_key"]
    backend = args.backend or defaults["backend"]
    dtype = args.dtype or defaults["dtype"]
    mlx_model = args.mlx_model or defaults["mlx_model"]
    max_length = args.max_length or defaults["max_length"]
    max_new_tokens = args.max_new_tokens or defaults["max_new_tokens"]
    max_generation_seconds = args.max_generation_seconds or defaults["max_generation_seconds"]
    temperature = defaults["temperature"] if args.temperature is None else args.temperature
    top_p = defaults["top_p"] if args.top_p is None else args.top_p
    methods = canonical_methods(args.methods)
    alphas = [float(item) for item in parse_csv(args.alphas)]
    eval_levels = set(parse_csv(args.eval_levels))
    direction_levels = set(parse_csv(args.direction_levels))
    tangent_levels = set(parse_csv(args.tangent_levels))
    tangent_turns = {int(item) for item in parse_csv(args.tangent_turns)}
    tangent_phases = set(parse_csv(args.tangent_phases))

    if args.tangent_neighbors < 2:
        raise ValueError("--tangent-neighbors must be at least 2")
    if args.tangent_dim < 1:
        raise ValueError("--tangent-dim must be at least 1")

    direction_points, activation_meta = load_activation_points(
        activation_path,
        layer=args.layer,
        turns={args.direction_turn},
        phases={args.direction_phase},
    )
    query_points, _ = load_activation_points(
        activation_path,
        layer=args.layer,
        turns={args.query_turn},
        phases={args.query_phase},
    )
    tangent_points, _ = load_activation_points(
        activation_path,
        layer=args.layer,
        turns=tangent_turns,
        phases=tangent_phases,
        levels=tangent_levels,
    )
    direction_by_cid = point_index(direction_points)
    query_by_cid = point_index(query_points)

    allowed = clean_id_lines(Path(args.conversation_ids_file)) if args.conversation_ids_file else None
    rows = []
    for row in read_jsonl_paths(transcript_paths):
        cid = str(row.get("conversation_id", ""))
        if allowed is not None and cid not in allowed:
            continue
        if not row.get("valid_outcome"):
            continue
        if row.get("arm") not in eval_levels:
            continue
        if cid not in direction_by_cid or cid not in query_by_cid:
            continue
        rows.append(row)
    rows = select_eval_rows(rows, limit=args.limit, strategy=args.limit_strategy, seed=args.seed)
    if not rows:
        raise ValueError("no eval rows matched transcripts + activations")

    existing_results: list[dict] = []
    completed: set[tuple[str, str, float]] = set()
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        if not existing.get("blocked") and isinstance(existing.get("results"), list):
            existing_results = existing["results"]
            completed = {
                (str(row["conversation_id"]), str(row["method"]), float(row["alpha"]))
                for row in existing_results
            }

    direction_cache: dict[str, dict | None] = {}
    tangent_cache: dict[str, dict | None] = {}
    projection_cache: dict[str, tuple[torch.Tensor | None, dict]] = {}

    def get_direction(family: str) -> dict | None:
        if family not in direction_cache:
            direction_cache[family] = fit_dp_direction(
                direction_points,
                heldout_family=family,
                direction_levels=direction_levels,
                min_mixed_scenarios=args.min_mixed_scenarios,
                min_levels=args.min_direction_levels,
            )
        return direction_cache[family]

    def get_tangent(family: str) -> dict | None:
        if family not in tangent_cache:
            tangent_cache[family] = fit_tangent_cloud(tangent_points, heldout_family=family)
        return tangent_cache[family]

    def tangent_projection(row: dict) -> tuple[torch.Tensor | None, dict]:
        cid = str(row["conversation_id"])
        if cid in projection_cache:
            return projection_cache[cid]
        direction_info = get_direction(str(row["family"]))
        tangent_info = get_tangent(str(row["family"]))
        if direction_info is None:
            out = None, {"reason": "no_direction"}
        elif tangent_info is None:
            out = None, {"reason": "no_tangent_cloud"}
        else:
            out = project_to_local_tangent(
                direction_info["_direction_np"],
                tangent_info,
                query_by_cid[cid]["x"],
                tangent_neighbors=args.tangent_neighbors,
                tangent_dim=args.tangent_dim,
            )
        projection_cache[cid] = out
        return out

    results: list[dict] = list(existing_results)
    planned_skips: Counter = Counter()
    jobs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        cid = str(row["conversation_id"])
        for method in methods:
            active_alphas = [0.0] if method == "baseline" else [alpha for alpha in alphas if alpha != 0]
            if not active_alphas:
                active_alphas = [0.0]
            for alpha in active_alphas:
                key = (cid, method, float(alpha))
                if key in completed:
                    planned_skips["already_completed"] += 1
                    continue
                direction = None
                direction_info = None
                projection = None
                if method in STEER_METHODS and alpha != 0:
                    direction_info = get_direction(str(row["family"]))
                    if direction_info is None:
                        planned_skips[f"no_direction::{row['family']}"] += 1
                        continue
                    if method == "dp_linear":
                        direction = direction_info["direction"]
                    else:
                        tangent_direction, projection = tangent_projection(row)
                        if tangent_direction is None:
                            reason = projection.get("reason", "projection_failed")
                            planned_skips[f"no_tangent::{row['family']}::{reason}"] += 1
                            continue
                        if method == "dp_tangent":
                            direction = tangent_direction
                        else:
                            direction = off_tangent_direction(direction_info["_direction_np"], tangent_direction)
                            if direction is None:
                                planned_skips[f"no_off_tangent::{row['family']}"] += 1
                                continue
                jobs.append({
                    "row_index": row_index,
                    "row": row,
                    "method": method,
                    "alpha": float(alpha),
                    "direction": direction,
                    "direction_info": direction_info,
                    "projection": projection,
                })

    def write_results(pipeline_meta: dict | None, *, validate_only: bool = False) -> None:
        out = {
            "schema_version": 1,
            "argv": sys.argv,
            "blocked": False,
            "validate_only": validate_only,
            "model": pipeline_meta,
            "requested_model_key": model_key,
            "backend": backend,
            "dtype": dtype,
            "mlx_model": mlx_model,
            "layer": args.layer,
            "direction_turn": args.direction_turn,
            "direction_phase": args.direction_phase,
            "query_turn": args.query_turn,
            "query_phase": args.query_phase,
            "eval_levels": sorted(eval_levels),
            "direction_levels": sorted(direction_levels),
            "tangent_levels": sorted(tangent_levels),
            "tangent_turns": sorted(tangent_turns),
            "tangent_phases": sorted(tangent_phases),
            "methods": methods,
            "alphas": alphas,
            "tangent_neighbors": args.tangent_neighbors,
            "tangent_dim": args.tangent_dim,
            "min_mixed_scenarios": args.min_mixed_scenarios,
            "min_direction_levels": args.min_direction_levels,
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "max_generation_seconds": max_generation_seconds,
            "transcripts": [str(path.resolve()) for path in transcript_paths],
            "transcripts_sha256": {str(path): file_sha256(path) for path in transcript_paths},
            "activations": str(activation_path.resolve()),
            "activations_sha256": file_sha256(activation_path),
            "activation_meta": activation_meta,
            "config": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "provenance": git_provenance([Path(__file__), config_path, activation_path, *transcript_paths]),
            "eval_rows": len(rows),
            "limit_strategy": args.limit_strategy,
            "direction_points": len(direction_points),
            "query_points": len(query_points),
            "tangent_points": len(tangent_points),
            "existing_results_loaded": len(existing_results),
            "planned_generations": len(jobs),
            "planned_skips": dict(planned_skips),
            "eval_family_balance": dict(Counter(str(row["family"]) for row in rows)),
            "eval_arm_balance": dict(Counter(str(row["arm"]) for row in rows)),
            "eval_original_label_balance": dict(Counter(str(bool(row["deceptive"])) for row in rows)),
            "direction_stats": {
                family: public_direction_stats(payload)
                for family, payload in sorted(direction_cache.items())
            },
            "tangent_cloud_stats": {
                family: {"heldout_family": family, "n_train": payload["n_train"]}
                for family, payload in sorted(tangent_cache.items())
                if payload is not None
            },
            "results": results,
            "note": (
                "Directions are fit leave-one-family-out from scenario-paired d_p on the graded "
                "pressure point cloud. dp_linear injects honest-deceptive; dp_tangent projects "
                "that same unit direction onto a local train-family tangent at the query state. "
                "Use summarize_deception_control.py for fix/harm/coherence frontiers."
            ),
        }
        atomic_text(out_path, json.dumps(to_jsonable(out), indent=2, sort_keys=True))

    if args.validate_only:
        write_results(None, validate_only=True)
        print(f"validated -> {out_path} ({len(jobs)} planned generations; model not loaded)", flush=True)
        return

    pipeline = load_activation_pipeline(
        model_key,
        backend=backend,
        device=args.device,
        dtype=dtype,
        max_length=max_length,
        mlx_model=mlx_model,
    )
    pipeline_meta = asdict(pipeline.meta)

    for job in tqdm(jobs, desc="control"):
        row = job["row"]
        row_index = int(job["row_index"])
        method = str(job["method"])
        alpha = float(job["alpha"])
        spec = None
        if method != "baseline" and alpha != 0:
            spec = ResidualSteeringSpec(layer=args.layer, direction=job["direction"], alpha=alpha)
        reply = pipeline.generate(
            prompt_without_final_answer(row),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=args.seed + row_index,
            max_generation_seconds=max_generation_seconds,
            steering=spec,
        )
        true_status = str(row["true_status"])
        reported = parse_status(reply, "report")
        coherence = reply_coherence(reply, reported)
        results.append({
            "conversation_id": row["conversation_id"],
            "scenario_id": row["scenario_id"],
            "sample_seed": row.get("sample_seed"),
            "family": row["family"],
            "arm": row["arm"],
            "true_status": true_status,
            "desired_status": row.get("desired_status"),
            "original_deceptive": bool(row.get("deceptive")),
            "method": method,
            "base_representation": "graded_dp",
            "alpha": alpha,
            "heldout_family": row["family"],
            "direction_projection": job["projection"],
            "injection": injection_stats(spec.direction, alpha) if spec is not None else None,
            "reported_status": reported,
            "honest": reported == true_status,
            "coherence": coherence,
            "reply": reply,
        })
        write_results(pipeline_meta)

    write_results(pipeline_meta)
    print(f"saved -> {out_path} ({len(results)} generations)", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
