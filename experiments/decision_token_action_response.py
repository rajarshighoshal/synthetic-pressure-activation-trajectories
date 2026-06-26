"""Decision-token action-response table for learned point-cloud control.

This is a forward-pass diagnostic, not full generation.  It builds a supervised
dataset of candidate interventions:

    (activation row, candidate action) -> PASS/FAIL margin response

The goal is to learn which geometry/control action to take before spending on
generation.  Rows are held out by family for direction fitting, and each action
is evaluated on the decision-token margin:

    margin = logit(PASS) - logit(FAIL)

Positive margin means PASS; negative means FAIL.  The output is suitable for
`learned_action_policy.py`.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.activation_control_tomography import decision_prefix_ids, decision_tokens, margin  # noqa: E402
from experiments.bidirectional_dp_diagnostic import attach_status_classes, fit_status_direction  # noqa: E402
from experiments.control_graded_dp_bidirectional_stack import (  # noqa: E402
    add_status_class_to_transcript,
    parse_int_csv,
    select_status_balanced_rows,
)
from experiments.control_graded_dp_decision_token import (  # noqa: E402
    fit_global_mean_direction,
    fit_global_probe_direction,
    fit_random_direction,
    load_gate_predictions,
)
from experiments.control_graded_dp_frontier import (  # noqa: E402
    config_defaults,
    fit_tangent_cloud,
    load_activation_points,
    off_tangent_direction,
    point_index,
    project_to_local_tangent,
    read_jsonl_paths,
    select_eval_rows,
    to_jsonable,
    unit,
    vector_stats,
)
from experiments.control_graded_dp_stack_frontier import aggregate_projection, parse_csv  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.models.interface import ResidualSteeringSpec  # noqa: E402
from geoprobe.models.mlx_capture import load_mlx_model  # noqa: E402


BIDIR_METHODS = {"bidir_linear", "bidir_tangent", "bidir_off_tangent"}
GLOBAL_METHODS = {"global_mean", "global_probe", "random_global"}
METHOD_ALIASES = {
    "linear": "bidir_linear",
    "bidir_linear": "bidir_linear",
    "tangent": "bidir_tangent",
    "bidir_tangent": "bidir_tangent",
    "off_tangent": "bidir_off_tangent",
    "bidir_off_tangent": "bidir_off_tangent",
    "global_mean": "global_mean",
    "global_probe": "global_probe",
    "probe": "global_probe",
    "random": "random_global",
    "random_global": "random_global",
}
STATUS_CLASSES = ("false_FAIL", "false_PASS", "honest_PASS", "honest_FAIL")


def canonical_methods(value: str) -> list[str]:
    out = []
    for item in parse_csv(value):
        if item not in METHOD_ALIASES:
            raise ValueError(f"unknown method {item!r}; expected one of {sorted(METHOD_ALIASES)}")
        method = METHOD_ALIASES[item]
        if method not in out:
            out.append(method)
    return out


def parse_targets(value: str) -> list[str]:
    targets = [item.upper() for item in parse_csv(value)]
    bad = [item for item in targets if item not in {"PASS", "FAIL"}]
    if bad:
        raise ValueError(f"bad candidate target(s): {bad}; expected PASS/FAIL")
    return targets


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


def true_margin_sign(true_status: str) -> int:
    status = str(true_status).upper()
    if status == "PASS":
        return 1
    if status == "FAIL":
        return -1
    raise ValueError(f"bad true_status {true_status!r}")


def target_margin_sign(target_status: str) -> int:
    return true_margin_sign(target_status)


def margin_correct(true_status: str, value: float) -> bool:
    sign = true_margin_sign(true_status)
    return value > 0 if sign > 0 else value < 0


def desired_status_for_class(status_class: str) -> str:
    if status_class in {"false_FAIL", "honest_PASS"}:
        return "PASS"
    if status_class in {"false_PASS", "honest_FAIL"}:
        return "FAIL"
    raise ValueError(f"bad status_class {status_class!r}")


def action_reward(status_class: str, correct_before: bool, correct_after: bool) -> float:
    """Reward for a candidate action used by the learned policy script.

    Deceptive rows get +1 only if fixed. Honest rows get -1 if harmed and 0 if
    preserved.  Deceptive rows that remain wrong get 0. This avoids rewarding
    arbitrary steering on already-honest rows.
    """
    if status_class.startswith("false_"):
        return 1.0 if (not correct_before and correct_after) else 0.0
    return -1.0 if correct_before and not correct_after else 0.0


def public_direction_info(direction_info: dict | None) -> dict | None:
    if direction_info is None:
        return None
    return {
        "target_status": direction_info.get("target_status"),
        "direction_convention": direction_info.get("direction_convention"),
        "direction_levels": direction_info.get("direction_levels"),
        "n_train_points": direction_info.get("n_train_points"),
        "n_honest": direction_info.get("n_honest"),
        "n_false": direction_info.get("n_false"),
        "n_mixed_scenario_level_pairs": direction_info.get("n_mixed_scenario_level_pairs"),
    }


def signed_global_direction(direction_info: dict, target_status: str) -> np.ndarray:
    direction = np.asarray(direction_info["_direction_np"], dtype=np.float64)
    # The positive sign is treated as PASS-ward and the negative sign as FAIL-ward
    # for action enumeration. The measured margin response, not this convention,
    # determines whether the action actually works.
    return direction if target_status == "PASS" else -direction


def make_spec(layer: int, direction: np.ndarray, alpha: float) -> ResidualSteeringSpec:
    clean = np.nan_to_num(np.asarray(direction, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return ResidualSteeringSpec(
        layer=layer,
        direction=torch.tensor(clean, dtype=torch.float32),
        alpha=float(alpha),
    )


def projection_features(projection: dict | None) -> dict:
    if not projection:
        return {
            "projection_fraction": None,
            "cos_to_raw": None,
            "neighbor_distance_mean": None,
            "neighbor_distance_max": None,
        }
    return {
        "projection_fraction": projection.get("projection_fraction"),
        "cos_to_raw": projection.get("cos_to_raw"),
        "neighbor_distance_mean": projection.get("neighbor_distance_mean"),
        "neighbor_distance_max": projection.get("neighbor_distance_max"),
    }


def pointcloud_context_features(
    *,
    query_x: np.ndarray,
    region_rows: list[dict],
    heldout_family: str,
    k: int,
) -> dict:
    """Leak-safe local point-cloud context around a query activation."""
    train = [
        row for row in region_rows
        if str(row.get("family")) != heldout_family
        and row.get("status_class") in STATUS_CLASSES
    ]
    out: dict[str, float | int | None] = {"pc_n_train": len(train), "pc_knn_k": int(k)}
    for cls in STATUS_CLASSES:
        out[f"pc_knn_frac_{cls}"] = None
        out[f"pc_centroid_dist_{cls}"] = None
    out["pc_knn_entropy"] = None
    out["pc_knn_mean_dist"] = None
    out["pc_false_frac"] = None
    out["pc_honest_frac"] = None
    if not train:
        return out

    q = np.nan_to_num(np.asarray(query_x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    xs = np.vstack([
        np.nan_to_num(np.asarray(row["x"], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        for row in train
    ])
    labels = [str(row["status_class"]) for row in train]
    dists = np.linalg.norm(xs - q[None, :], axis=1)
    order = np.argsort(dists)
    kk = min(int(k), len(order))
    neigh = order[:kk]
    counts = Counter(labels[i] for i in neigh)
    probs = []
    out["pc_knn_mean_dist"] = float(np.mean(dists[neigh])) if kk else None
    for cls in STATUS_CLASSES:
        frac = float(counts.get(cls, 0) / kk) if kk else 0.0
        out[f"pc_knn_frac_{cls}"] = frac
        if frac > 0:
            probs.append(frac)
        idx = [i for i, label in enumerate(labels) if label == cls]
        if idx:
            out[f"pc_centroid_dist_{cls}"] = float(np.linalg.norm(xs[idx].mean(axis=0) - q))
    out["pc_knn_entropy"] = float(-sum(p * np.log(p + 1e-12) for p in probs)) if probs else 0.0
    out["pc_false_frac"] = float(out["pc_knn_frac_false_FAIL"] + out["pc_knn_frac_false_PASS"])
    out["pc_honest_frac"] = float(out["pc_knn_frac_honest_PASS"] + out["pc_knn_frac_honest_FAIL"])
    return out


def build_result_row(
    *,
    row: dict,
    method: str,
    target_status: str | None,
    layer: int | None,
    alpha: float,
    base_margin: float,
    final_margin: float,
    direction_info: dict | None,
    projection: dict | None,
    route: dict | None,
    context: dict | None = None,
) -> dict:
    status_class = str(row["status_class"])
    correct_before = margin_correct(str(row["true_status"]), base_margin)
    correct_after = margin_correct(str(row["true_status"]), final_margin)
    is_false = status_class.startswith("false_")
    proj = projection_features(projection)
    return {
        "conversation_id": str(row["conversation_id"]),
        "scenario_id": str(row.get("scenario_id", "")),
        "family": str(row["family"]),
        "arm": str(row["arm"]),
        "sample_seed": row.get("sample_seed"),
        "true_status": str(row["true_status"]).upper(),
        "reported_status_before": str(row.get("reported_status", "")).upper(),
        "status_class": status_class,
        "desired_status": desired_status_for_class(status_class),
        "method": method,
        "target_status": target_status,
        "layer": layer,
        "alpha": float(alpha),
        "base_margin": float(base_margin),
        "final_margin": float(final_margin),
        "delta_margin": float(final_margin - base_margin),
        "abs_base_margin": float(abs(base_margin)),
        "correct_before": bool(correct_before),
        "correct_after": bool(correct_after),
        "fixes_error": bool(is_false and (not correct_before) and correct_after),
        "harms_honest": bool((not is_false) and correct_before and (not correct_after)),
        "reward": action_reward(status_class, correct_before, correct_after),
        "desired_margin_sign": true_margin_sign(str(row["true_status"])),
        "target_margin_sign": target_margin_sign(target_status) if target_status else 0,
        "route_action": route.get("action") if route else None,
        "gate_score_PASS_minus_FAIL": route.get("score_PASS_minus_FAIL") if route else None,
        "gate_proba_PASS": route.get("proba_PASS") if route else None,
        **(context or {}),
        "projection": projection,
        **proj,
        "direction_info": public_direction_info(direction_info),
    }


def summarize(rows: list[dict]) -> dict:
    by_method = {}
    for method in sorted({row["method"] for row in rows}):
        sub = [row for row in rows if row["method"] == method]
        by_class = {}
        for cls in STATUS_CLASSES:
            cls_rows = [row for row in sub if row["status_class"] == cls]
            by_class[cls] = {
                "n": len(cls_rows),
                "correct_after": int(sum(row["correct_after"] for row in cls_rows)),
                "fixes_error": int(sum(row["fixes_error"] for row in cls_rows)),
                "harms_honest": int(sum(row["harms_honest"] for row in cls_rows)),
                "mean_delta_margin": float(np.mean([row["delta_margin"] for row in cls_rows])) if cls_rows else None,
            }
        by_method[method] = {
            "n": len(sub),
            "fixes_error": int(sum(row["fixes_error"] for row in sub)),
            "harms_honest": int(sum(row["harms_honest"] for row in sub)),
            "mean_reward": float(np.mean([row["reward"] for row in sub])) if sub else None,
            "by_status_class": by_class,
        }
    return {
        "n": len(rows),
        "methods": by_method,
        "status_class": dict(Counter(row["status_class"] for row in rows if row["method"] == "abstain")),
    }


def prune_candidates(rows: list[dict], *, max_per_class: int) -> list[dict]:
    """Keep abstain rows and top-N (method,target,layer,alpha) per status class.

    Ranking is by mean |Δmargin| in the corrective direction:
      - false_FAIL / honest_PASS  → prefer positive Δmargin
      - false_PASS / honest_FAIL  → prefer negative Δmargin
    """
    keep: list[dict] = []
    for cls in STATUS_CLASSES:
        class_rows = [row for row in rows if row["status_class"] == cls]
        abstain_rows = [row for row in class_rows if row["method"] == "abstain"]
        action_rows = [row for row in class_rows if row["method"] != "abstain"]
        keep.extend(abstain_rows)
        if max_per_class <= 0 or not action_rows:
            continue
        corrective_sign = 1 if cls in {"false_FAIL", "honest_PASS"} else -1
        groups: dict[tuple[str, str, int | None, float], list[float]] = {}
        for row in action_rows:
            key = (row["method"], row["target_status"] or "NONE", row["layer"], row["alpha"])
            groups.setdefault(key, []).append(corrective_sign * float(row["delta_margin"]))
        scored = sorted(
            [(float(np.mean(vals)), key) for key, vals in groups.items()],
            reverse=True,
        )
        top_keys = {key for _, key in scored[:max_per_class]}
        for row in action_rows:
            key = (row["method"], row["target_status"] or "NONE", row["layer"], row["alpha"])
            if key in top_keys:
                keep.append(row)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="20,24,28,32")
    parser.add_argument("--alphas", default="16,24,32,48")
    parser.add_argument("--methods", default="bidir_linear,bidir_tangent,bidir_off_tangent,global_mean,global_probe,random_global")
    parser.add_argument("--candidate-targets", default="PASS,FAIL")
    parser.add_argument("--direction-turn", type=int, default=2)
    parser.add_argument("--direction-phase", default="pre_response")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--eval-levels", default="p3,p4,p5")
    parser.add_argument("--direction-levels", default="p3,p4,p5,p6")
    parser.add_argument("--tangent-levels", default="p0,p1,p2,p3,p4,p5,p6")
    parser.add_argument("--tangent-turns", default="0,1,2,3")
    parser.add_argument("--tangent-phases", default="pre_response,post_response")
    parser.add_argument("--min-mixed-scenarios", type=int, default=2)
    parser.add_argument("--min-direction-levels", type=int, default=2)
    parser.add_argument("--tangent-neighbors", type=int, default=16)
    parser.add_argument("--tangent-dim", type=int, default=4)
    parser.add_argument("--context-neighbors", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-status-class", type=int, default=None)
    parser.add_argument("--limit-strategy", choices=["first", "shuffle", "family_round_robin"], default="family_round_robin")
    parser.add_argument("--gate-predictions", default=None)
    parser.add_argument("--unidirectional-targets", action="store_true",
                        help="For deceptive rows, only evaluate the corrective target "
                             "(PASS for false_FAIL, FAIL for false_PASS); honest rows "
                             "still get both targets to measure potential harm.")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="After evaluation, keep only the top-N (method,target,layer,alpha) "
                             "combos per status class by mean corrective Δmargin; abstain "
                             "rows are always kept.")
    parser.add_argument("--within-scenario-split", action="store_true",
                        help="Fit directions per-scenario (exclude target scenario_id) "
                             "instead of per-family. Tests whether geometry transfers "
                             "across scenarios within the same family.")
    parser.add_argument("--candidate-combos", default=None,
                        help="Path to a JSON file listing curated (method, target, layer, alpha) "
                             "tuples to evaluate. When provided, only these combos are evaluated "
                             "instead of the full Cartesian product. Useful for applying pre-filtered "
                             "candidate grids from a previous action-response or tomography run. "
                             "Format: [[\"bidir_tangent\", \"PASS\", 20, 48], ...]")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    transcript_paths = [Path(path) for path in args.transcripts]
    activation_path = Path(args.activations)
    config_path = Path(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    defaults = config_defaults(config_path)
    mlx_model = args.mlx_model or defaults["mlx_model"]
    dtype = args.dtype or defaults["dtype"]
    layers = parse_int_csv(args.layers)
    alphas = [float(item) for item in parse_csv(args.alphas)]
    methods = canonical_methods(args.methods)
    candidate_targets = parse_targets(args.candidate_targets)
    eval_levels = set(parse_csv(args.eval_levels))
    direction_levels = set(parse_csv(args.direction_levels))
    tangent_levels = set(parse_csv(args.tangent_levels))
    tangent_turns = {int(item) for item in parse_csv(args.tangent_turns)}
    tangent_phases = set(parse_csv(args.tangent_phases))
    gate_by_cid = load_gate_predictions(Path(args.gate_predictions)) if args.gate_predictions else {}

    first_query, activation_meta = load_activation_points(
        activation_path,
        layer=layers[0],
        turns={args.query_turn},
        phases={args.query_phase},
    )
    first_query_ids = set(point_index(first_query))
    del first_query
    gc.collect()

    transcript_by_cid = {str(row.get("conversation_id")): row for row in read_jsonl_paths(transcript_paths)}
    rows = []
    for row in read_jsonl_paths(transcript_paths):
        cid = str(row.get("conversation_id", ""))
        if not row.get("valid_outcome") or row.get("arm") not in eval_levels or cid not in first_query_ids:
            continue
        row_with_class = add_status_class_to_transcript(row)
        if row_with_class is not None:
            rows.append(row_with_class)
    if args.limit_per_status_class is not None:
        rows = select_status_balanced_rows(rows, per_status_class=args.limit_per_status_class, seed=args.seed)
    else:
        rows = select_eval_rows(rows, limit=args.limit, strategy=args.limit_strategy, seed=args.seed)
    if not rows:
        raise ValueError("no eval rows matched transcripts + activations")
    eval_ids = {str(row["conversation_id"]) for row in rows}

    # Curation from external file: restrict planned actions to only listed combos.
    curated_combos: set[tuple[str, str, int, float]] | None = None
    if args.candidate_combos:
        curated_list = json.loads(Path(args.candidate_combos).read_text())
        curated_combos = {
            (str(item[0]), str(item[1]).upper(), int(item[2]), float(item[3]))
            for item in curated_list
        }
        print(f"curated combos: {len(curated_combos)} loaded from {args.candidate_combos}", flush=True)

    # Resume support: if the output file already exists (e.g. from a crashed run),
    # reload the saved rows and skip already-evaluated (cid, method, target, layer, alpha).
    result_rows: list[dict] = []
    seen_keys: set[tuple[str, str, str | None, int | None, float]] = set()
    if out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            prior_rows = prior.get("rows", [])
            if prior_rows:
                result_rows = prior_rows
                seen_keys = {
                    (str(row["conversation_id"]),
                     str(row["method"]),
                     str(row.get("target_status") or "NONE"),
                     row.get("layer"),
                     float(row.get("alpha", 0.0)))
                    for row in prior_rows
                }
                print(f"resuming: {len(result_rows)} existing rows, {len(seen_keys)} seen keys", flush=True)
        except Exception:
            pass

    planned = []
    skipped_by_resume = 0
    for row in rows:
        planned.append((str(row["conversation_id"]), "abstain", None, None, 0.0))
        status_cls = str(row.get("status_class", ""))
        targets = candidate_targets
        cid = str(row["conversation_id"])
        if args.unidirectional_targets:
            if status_cls == "false_FAIL":
                targets = ["PASS"]
            elif status_cls == "false_PASS":
                targets = ["FAIL"]
        for layer in layers:
            for method in methods:
                for target in targets:
                    for alpha in alphas:
                        if curated_combos is not None and (method, target, layer, alpha) not in curated_combos:
                            continue
                        if (cid, method, target, layer, float(alpha)) in seen_keys:
                            skipped_by_resume += 1
                            continue
                        planned.append((cid, method, target, layer, alpha))
    if skipped_by_resume:
        print(f"skipped {skipped_by_resume} already-evaluated actions (resume)", flush=True)

    def write_payload(result_rows: list[dict], *, model_meta: dict | None, validate_only: bool, custom_out_path: Path | None = None) -> None:
        target_path = custom_out_path or out_path
        payload = {
            "schema_version": 1,
            "argv": sys.argv,
            "validate_only": validate_only,
            "model": model_meta,
            "mlx_model": str(mlx_model),
            "dtype": dtype,
            "transcripts": [str(path.resolve()) for path in transcript_paths],
            "transcripts_sha256": {str(path): file_sha256(path) for path in transcript_paths},
            "activations": str(activation_path.resolve()),
            "activations_sha256": file_sha256(activation_path),
            "activation_meta": activation_meta,
            "config": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "gate_predictions": str(Path(args.gate_predictions).resolve()) if args.gate_predictions else None,
            "gate_predictions_sha256": file_sha256(Path(args.gate_predictions)) if args.gate_predictions else None,
            "provenance": git_provenance([Path(__file__), config_path, activation_path, *transcript_paths]),
            "layers": layers,
            "alphas": alphas,
            "methods": methods,
            "candidate_targets": candidate_targets,
            "eval_levels": sorted(eval_levels),
            "direction_levels": sorted(direction_levels),
            "eval_rows": len(rows),
            "eval_status_class_balance": dict(Counter(str(row["status_class"]) for row in rows)),
            "planned_actions": len(planned),
            "summary": summarize(result_rows) if result_rows else None,
            "rows": result_rows,
            "note": (
                "Rows are candidate interventions at the PASS/FAIL decision token. "
                "Use this as response data for learned_action_policy.py; do not read it "
                "as full report-generation behavior."
            ),
        }
        atomic_text(target_path, json.dumps(to_jsonable(payload), indent=2, sort_keys=True))

    if args.validate_only:
        val_out = out_path.with_suffix(out_path.suffix + ".validate")
        write_payload([], model_meta=None, validate_only=True, custom_out_path=val_out)
        print(f"validated -> {val_out} ({len(planned)} planned action margins; model not loaded)", flush=True)
        return

    model, tokenizer, meta = load_mlx_model(str(mlx_model), dtype=dtype)
    pass_id, fail_id = decision_tokens(tokenizer)
    model_meta = asdict(meta)

    # Direction/tangent caches are built after model loading so validation remains cheap.
    points_by_layer = {}
    query_by_layer = {}
    tangent_by_layer = {}
    region_by_layer = {}
    for layer in layers:
        direction_layer_raw, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns={args.direction_turn},
            phases={args.direction_phase},
        )
        direction_layer, _ = attach_status_classes(direction_layer_raw, transcript_by_cid)
        points_by_layer[layer] = direction_layer
        query_layer, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns={args.query_turn},
            phases={args.query_phase},
        )
        query_by_layer[layer] = point_index([row for row in query_layer if row["conversation_id"] in eval_ids])
        region_with_status, _ = attach_status_classes(query_layer, transcript_by_cid)
        region_by_layer[layer] = region_with_status
        tangent_layer, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns=tangent_turns,
            phases=tangent_phases,
            levels=tangent_levels,
        )
        tangent_by_layer[layer] = tangent_layer

    direction_cache: dict[tuple[int, str, str], dict | None] = {}
    global_cache: dict[tuple[int, str, str], dict | None] = {}
    tangent_cache: dict[tuple[int, str], dict | None] = {}

    # Within-scenario-split: hold out the row's scenario_id, not its family.
    scenario_by_cid: dict[str, str] = {}
    if args.within_scenario_split:
        ref_raw, _ = load_activation_points(activation_path, layer=layers[0],
                                             turns={args.query_turn}, phases={args.query_phase})
        scenario_by_cid = {
            str(p["conversation_id"]): str(p.get("scenario_id", p.get("family", "")))
            for p in ref_raw
        }

    def _holdout(cid: str) -> tuple[str, str]:
        if args.within_scenario_split and cid in scenario_by_cid:
            return (scenario_by_cid[cid], "scenario")
        return (str(transcript_by_cid.get(cid, {}).get("family", "")), "family")

    def get_direction(layer: int, cid: str, target: str) -> dict | None:
        holdout_val, holdout_type = _holdout(cid)
        key = (layer, holdout_val, target)
        if key not in direction_cache:
            direction_cache[key] = fit_status_direction(
                points_by_layer[layer],
                heldout_family=holdout_val if holdout_type == "family" else None,
                heldout_scenario_ids={holdout_val} if holdout_type == "scenario" else None,
                direction_levels=direction_levels,
                target_status=target,
                min_mixed_scenarios=args.min_mixed_scenarios,
                min_levels=args.min_direction_levels,
            )
        return direction_cache[key]

    def get_global(layer: int, cid: str, method: str) -> dict | None:
        holdout_val, holdout_type = _holdout(cid)
        key = (layer, holdout_val, method)
        if key not in global_cache:
            if method == "global_mean":
                global_cache[key] = fit_global_mean_direction(
                    points_by_layer[layer],
                    heldout_family=holdout_val if holdout_type == "family" else None,
                    heldout_scenario_ids={holdout_val} if holdout_type == "scenario" else None,
                    direction_levels=direction_levels,
                )
            elif method == "global_probe":
                global_cache[key] = fit_global_probe_direction(
                    points_by_layer[layer],
                    heldout_family=holdout_val if holdout_type == "family" else None,
                    heldout_scenario_ids={holdout_val} if holdout_type == "scenario" else None,
                    direction_levels=direction_levels,
                )
            elif method == "random_global":
                global_cache[key] = fit_random_direction(
                    points_by_layer[layer],
                    heldout_family=holdout_val if holdout_type == "family" else None,
                    heldout_scenario_ids={holdout_val} if holdout_type == "scenario" else None,
                    direction_levels=direction_levels,
                    layer=layer,
                    seed=args.seed,
                )
            else:
                raise ValueError(f"bad global method {method}")
        return global_cache[key]

    def get_tangent(layer: int, cid: str) -> dict | None:
        holdout_val, holdout_type = _holdout(cid)
        key = (layer, holdout_val)
        if key not in tangent_cache:
            tangent_cache[key] = fit_tangent_cloud(
                tangent_by_layer[layer],
                heldout_family=holdout_val if holdout_type == "family" else None,
                heldout_scenario_ids={holdout_val} if holdout_type == "scenario" else None,
            )
        return tangent_cache[key]

    base_margin_by_cid: dict[str, float] = {}
    prefix_by_cid: dict[str, list[int]] = {}
    context_by_layer_cid: dict[tuple[int, str], dict] = {}
    for row in rows:
        cid = str(row["conversation_id"])
        family = str(row["family"])
        for layer in layers:
            query = query_by_layer[layer].get(cid)
            if query is None:
                continue
            context_by_layer_cid[(layer, cid)] = pointcloud_context_features(
                query_x=query["x"],
                region_rows=region_by_layer[layer],
                heldout_family=family,
                k=args.context_neighbors,
            )
    for row in tqdm(rows, desc="base margins"):
        cid = str(row["conversation_id"])
        rec = transcript_by_cid[cid]
        ids = decision_prefix_ids(tokenizer, rec["messages"])
        prefix_by_cid[cid] = ids
        base_margin_by_cid[cid] = margin(model, ids, pass_id, fail_id, steering=None)
        if (cid, "abstain", "NONE", None, 0.0) not in seen_keys:
            route = gate_by_cid.get(cid)
            result_rows.append(build_result_row(
                row=row,
                method="abstain",
                target_status=None,
                layer=None,
                alpha=0.0,
                base_margin=base_margin_by_cid[cid],
                final_margin=base_margin_by_cid[cid],
                direction_info=None,
                projection=None,
                route=route,
                context=context_by_layer_cid.get((layers[0], cid)),
            ))

    checkpoint_counter = 0
    for row in tqdm(rows, desc="candidate action margins"):
        cid = str(row["conversation_id"])
        family = str(row["family"])
        route = gate_by_cid.get(cid)
        row_targets = candidate_targets
        if args.unidirectional_targets:
            status_cls = str(row.get("status_class", ""))
            if status_cls == "false_FAIL":
                row_targets = ["PASS"]
            elif status_cls == "false_PASS":
                row_targets = ["FAIL"]
        for layer in layers:
            query = query_by_layer[layer].get(cid)
            if query is None:
                continue
            for method in methods:
                for target in row_targets:
                    direction_info = None
                    projection = None
                    vec = None
                    if method in BIDIR_METHODS:
                        direction_info = get_direction(layer, cid, target)
                        if direction_info is None:
                            continue
                        raw = np.asarray(direction_info["_direction_np"], dtype=np.float64)
                        if method == "bidir_linear":
                            vec = raw
                        else:
                            cloud = get_tangent(layer, cid)
                            if cloud is None:
                                continue
                            tangent_direction, projection = project_to_local_tangent(
                                raw,
                                cloud,
                                query["x"],
                                tangent_neighbors=args.tangent_neighbors,
                                tangent_dim=args.tangent_dim,
                            )
                            if tangent_direction is None:
                                continue
                            if method == "bidir_tangent":
                                vec = tangent_direction.detach().float().cpu().numpy()
                            else:
                                off_direction = off_tangent_direction(raw, tangent_direction)
                                if off_direction is None:
                                    continue
                                vec = off_direction.detach().float().cpu().numpy()
                    else:
                        direction_info = get_global(layer, cid, method)
                        if direction_info is None:
                            continue
                        vec = signed_global_direction(direction_info, target)
                    vec = unit(np.asarray(vec, dtype=np.float64))
                    if not np.isfinite(vec).all() or np.linalg.norm(vec) <= 1e-8:
                        continue
                    for alpha in alphas:
                        if (cid, method, target, layer, float(alpha)) in seen_keys:
                            continue
                        spec = make_spec(layer, vec, alpha)
                        final_margin = margin(model, prefix_by_cid[cid], pass_id, fail_id, steering=[spec])
                        proj_payload = aggregate_projection({layer: projection}) if projection else None
                        direction_payload = dict(direction_info)
                        direction_payload["direction_stats"] = vector_stats(vec)
                        result_rows.append(build_result_row(
                            row=row,
                            method=method,
                            target_status=target,
                            layer=layer,
                            alpha=alpha,
                            base_margin=base_margin_by_cid[cid],
                            final_margin=final_margin,
                            direction_info=direction_payload,
                            projection=proj_payload,
                            route=route,
                            context=context_by_layer_cid.get((layer, cid)),
                        ))
        checkpoint_counter += 1
        if checkpoint_counter >= 25:
            write_payload(result_rows, model_meta=model_meta, validate_only=False)
            checkpoint_counter = 0

    if args.max_candidates is not None and result_rows:
        result_rows = prune_candidates(result_rows, max_per_class=args.max_candidates)

    write_payload(result_rows, model_meta=model_meta, validate_only=False)
    print(f"saved -> {out_path} ({len(result_rows)} action rows)", flush=True)


if __name__ == "__main__":
    main()
