"""Decision-token-only bidirectional control for the graded deception ramp.

The high-alpha tomography result showed that correction directions can cross the PASS/FAIL logit
margin, but persistent steering throughout generation destroys the report. This runner applies the
steering only for the first report-status decision token, then turns steering off and lets the model
complete the rest of the JSON normally.

By default this is still an oracle feasibility test: bidirectional methods use the known true status
to choose to_PASS vs to_FAIL. With ``--routing gate_file``, a held-out-family gate prediction file
chooses abstain / steer_to_PASS / steer_to_FAIL, giving a deployable-style control test without using
the answer at intervention time.
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
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.activation_control_tomography import REPORT_PREFIX, decision_prefix_ids, decision_tokens, margin  # noqa: E402
from experiments.bidirectional_dp_diagnostic import attach_status_classes, fit_status_direction, status_error_class  # noqa: E402
from experiments.control_deception_intent_transition import prompt_without_final_answer, reply_coherence  # noqa: E402
from experiments.control_graded_dp_bidirectional_stack import (  # noqa: E402
    add_status_class_to_transcript,
    parse_int_csv,
    select_status_balanced_rows,
    target_direction_name,
    target_status_for_row,
)
from experiments.control_graded_dp_frontier import (  # noqa: E402
    clean_id_lines,
    clean_matrix,
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
from experiments.control_graded_dp_stack_frontier import aggregate_projection, parse_csv, stack_injection_stats  # noqa: E402
from experiments.rollout_deception_intent import parse_status  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.models.interface import ResidualSteeringSpec  # noqa: E402
from geoprobe.models.mlx_capture import generate_greedy_with_steering, load_mlx_model  # noqa: E402


METHOD_ALIASES = {
    "baseline": "baseline",
    "bidir_linear": "bidir_linear",
    "linear": "bidir_linear",
    "bidir_tangent": "bidir_tangent",
    "tangent": "bidir_tangent",
    "bidir_off_tangent": "bidir_off_tangent",
    "off_tangent": "bidir_off_tangent",
    "global_mean": "global_mean",
    "pooled_global_mean": "global_mean",
    "global_probe": "global_probe",
    "logistic_probe": "global_probe",
    "random_global": "random_global",
    "random": "random_global",
    "global_mean_gated": "global_mean_gated",
    "global_probe_gated": "global_probe_gated",
    "random_gated": "random_gated",
}
BIDIR_METHODS = {"bidir_linear", "bidir_tangent", "bidir_off_tangent"}
GLOBAL_METHODS = {"global_mean", "global_probe", "random_global"}
GATED_GLOBAL_METHODS = {"global_mean_gated", "global_probe_gated", "random_gated"}
STEER_METHODS = BIDIR_METHODS | GLOBAL_METHODS | GATED_GLOBAL_METHODS


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


def load_gate_predictions(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for row in payload.get("predictions", []):
        cid = str(row.get("conversation_id", ""))
        if cid:
            out[cid] = row
    if not out:
        raise ValueError(f"no gate predictions found in {path}")
    return out


def load_policy_choices(path: Path | None, policy_name: str) -> dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if policy_name not in payload:
        raise ValueError(f"policy {policy_name!r} not found in {path}; keys={sorted(payload)}")
    choices = payload[policy_name].get("choices", [])
    out = {}
    for choice in choices:
        cid = str(choice.get("conversation_id", ""))
        if cid:
            out[cid] = choice
    if not out:
        raise ValueError(f"no choices found for policy {policy_name!r} in {path}")
    return out


def route_for_policy_choice(row: dict, choice: dict | None, *, gate_by_cid: dict[str, dict]) -> dict:
    cid = str(row["conversation_id"])
    gate = gate_by_cid.get(cid)
    if choice is None or choice.get("method") in {None, "abstain"}:
        target_status = None
        action = "abstain"
    else:
        target_status = choice.get("target_status")
        if target_status not in {"PASS", "FAIL"}:
            raise ValueError(f"bad policy target for {cid}: {target_status!r}")
        action = f"steer_to_{target_status}"
    return {
        "routing": "policy_file",
        "action": action,
        "target_status": target_status,
        "predicted_true_status": target_status,
        "target_status_correct": target_status == str(row.get("true_status")).upper() if target_status else None,
        "routing_correct": target_status == str(row.get("true_status")).upper() if target_status else None,
        "policy_choice": choice,
        "gate": None if gate is None else {
            "source_conversation_id": gate.get("conversation_id"),
            "reported_status": gate.get("reported_status"),
            "true_status": gate.get("true_status"),
            "status_class": gate.get("status_class"),
            "action": gate.get("action"),
            "proba_PASS": gate.get("proba_PASS"),
            "score_PASS_minus_FAIL": gate.get("score_PASS_minus_FAIL"),
        },
    }


def route_for_row(row: dict, *, routing: str, gate_by_cid: dict[str, dict]) -> dict:
    cid = str(row["conversation_id"])
    if routing == "oracle_true_status":
        target_status = target_status_for_row(row)
        return {
            "routing": routing,
            "action": f"steer_to_{target_status}",
            "target_status": target_status,
            "predicted_true_status": target_status,
            "target_status_correct": True,
            "routing_correct": True,
            "gate": None,
        }
    if routing != "gate_file":
        raise ValueError(f"unknown routing mode {routing!r}")
    gate = gate_by_cid.get(cid)
    if gate is None:
        return {
            "routing": routing,
            "action": "missing_gate_prediction",
            "target_status": None,
            "predicted_true_status": None,
            "target_status_correct": None,
            "routing_correct": None,
            "gate": None,
        }
    action = str(gate.get("action", ""))
    target_status = None
    if action == "steer_to_PASS":
        target_status = "PASS"
    elif action == "steer_to_FAIL":
        target_status = "FAIL"
    elif action != "abstain":
        raise ValueError(f"bad gate action for {cid}: {action!r}")
    return {
        "routing": routing,
        "action": action,
        "target_status": target_status,
        "predicted_true_status": gate.get("predicted_true_status"),
        "target_status_correct": gate.get("target_status_correct"),
        "routing_correct": gate.get("routing_correct"),
        "score_PASS_minus_FAIL": gate.get("score_PASS_minus_FAIL"),
        "proba_PASS": gate.get("proba_PASS"),
        "gate": {
            "source_conversation_id": gate.get("conversation_id"),
            "reported_status": gate.get("reported_status"),
            "true_status": gate.get("true_status"),
            "status_class": gate.get("status_class"),
        },
    }


def target_direction_name_from_route(route: dict) -> str | None:
    target = route.get("target_status")
    return f"to_{target}" if target in {"PASS", "FAIL"} else None


def method_requires_gate_action(method: str) -> bool:
    return method in BIDIR_METHODS or method in GATED_GLOBAL_METHODS


def method_uses_tangent(method: str) -> bool:
    return method in {"bidir_tangent", "bidir_off_tangent"}


def fit_global_mean_direction(rows: list[dict], *, heldout_family: str | None = None, heldout_scenario_ids: set[str] | None = None, direction_levels: set[str]) -> dict | None:
    train = [
        row for row in rows
        if row["arm"] in direction_levels
        and (heldout_family is None or str(row["family"]) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    honest = [row for row in train if str(row["status_class"]).startswith("honest_")]
    false = [row for row in train if str(row["status_class"]).startswith("false_")]
    if not honest or not false:
        return None
    x_h = clean_matrix(np.vstack([row["x"] for row in honest]))
    x_f = clean_matrix(np.vstack([row["x"] for row in false]))
    direction = unit(x_h.mean(axis=0) - x_f.mean(axis=0))
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-8:
        return None
    return {
        "heldout_family": heldout_family,
        "target_status": "global_honesty",
        "direction_convention": "direction = mean(honest_PASS+honest_FAIL) - mean(false_FAIL+false_PASS)",
        "direction_levels": sorted(direction_levels),
        "n_train_points": int(len(train)),
        "n_honest": int(len(honest)),
        "n_false": int(len(false)),
        "direction_stats": vector_stats(direction),
        "_direction_np": direction,
    }


def fit_global_probe_direction(rows: list[dict], *, heldout_family: str | None = None, heldout_scenario_ids: set[str] | None = None, direction_levels: set[str]) -> dict | None:
    train = [
        row for row in rows
        if row["arm"] in direction_levels
        and (heldout_family is None or str(row["family"]) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    if not train:
        return None
    y = np.asarray([1 if str(row["status_class"]).startswith("honest_") else 0 for row in train], dtype=int)
    if len(set(y.tolist())) < 2:
        return None
    x = clean_matrix(np.vstack([row["x"] for row in train])).astype(np.float64)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=0)
    clf.fit(xs, y)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale[scale == 0] = 1.0
    direction = unit(np.asarray(clf.coef_[0], dtype=np.float64) / scale)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-8:
        return None
    return {
        "heldout_family": heldout_family,
        "target_status": "global_honesty_probe",
        "direction_convention": "direction = raw-space normal of logistic probe predicting honest vs false",
        "direction_levels": sorted(direction_levels),
        "n_train_points": int(len(train)),
        "n_honest": int(y.sum()),
        "n_false": int((1 - y).sum()),
        "direction_stats": vector_stats(direction),
        "_direction_np": direction,
    }


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def fit_random_direction(rows: list[dict], *, heldout_family: str | None = None, heldout_scenario_ids: set[str] | None = None, direction_levels: set[str], layer: int, seed: int) -> dict | None:
    train = [
        row for row in rows
        if row["arm"] in direction_levels
        and (heldout_family is None or str(row["family"]) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    if not train:
        return None
    dim = int(np.asarray(train[0]["x"]).shape[0])
    rng = np.random.default_rng(stable_seed("random_global", heldout_family, layer, seed))
    direction = unit(rng.normal(size=dim))
    return {
        "heldout_family": heldout_family,
        "target_status": "random_global",
        "direction_convention": "direction = deterministic random unit vector, norm-matched by alpha",
        "direction_levels": sorted(direction_levels),
        "n_train_points": int(len(train)),
        "direction_stats": vector_stats(direction),
        "_direction_np": direction,
    }


def public_direction_info(direction_info: dict) -> dict:
    return {
        "target_status": direction_info.get("target_status"),
        "n_mixed_scenario_level_pairs": direction_info.get("n_mixed_scenario_level_pairs"),
        "n_train_points": direction_info.get("n_train_points"),
        "n_honest": direction_info.get("n_honest"),
        "n_false": direction_info.get("n_false"),
        "direction_levels": direction_info.get("direction_levels"),
        "direction_convention": direction_info.get("direction_convention"),
    }


def target_direction_for_method(method: str, route: dict) -> str | None:
    if method in BIDIR_METHODS:
        return target_direction_name_from_route(route)
    if method in GLOBAL_METHODS:
        return method
    if method in GATED_GLOBAL_METHODS:
        return method if route.get("target_status") in {"PASS", "FAIL"} else None
    return None


def choose_status_then_complete(
    model,
    tokenizer,
    messages: list[dict],
    *,
    pass_id: int,
    fail_id: int,
    steering,
    max_new_tokens: int,
    max_generation_seconds: float | None,
) -> tuple[str, dict]:
    ids = decision_prefix_ids(tokenizer, messages)
    steered_margin = margin(model, ids, pass_id, fail_id, steering=steering)
    status = "PASS" if steered_margin > 0 else "FAIL"
    status_id = pass_id if status == "PASS" else fail_id
    continuation = generate_greedy_with_steering(
        model,
        tokenizer,
        ids + [status_id],
        max_new_tokens=max_new_tokens,
        steering=None,
        max_generation_seconds=max_generation_seconds,
    )
    reply = REPORT_PREFIX + status + continuation
    return reply, {
        "pass_token": int(pass_id),
        "fail_token": int(fail_id),
        "margin": float(steered_margin),
        "forced_status": status,
        "scope": "decision_token_only_binary_status",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="20")
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
    parser.add_argument("--methods", default="baseline,bidir_linear,bidir_tangent")
    parser.add_argument("--alphas", default="48")
    parser.add_argument("--alpha-mode", choices=["total", "per_layer"], default="total")
    parser.add_argument("--routing", choices=["oracle_true_status", "gate_file", "policy_file"], default="oracle_true_status")
    parser.add_argument("--gate-predictions", default=None)
    parser.add_argument("--policy-choices", default=None)
    parser.add_argument("--policy-name", default="selective_route_policy")
    parser.add_argument("--tangent-neighbors", type=int, default=16)
    parser.add_argument("--tangent-dim", type=int, default=4)
    parser.add_argument("--conversation-ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-status-class", type=int, default=None)
    parser.add_argument("--limit-strategy", choices=["first", "shuffle", "family_round_robin"], default="family_round_robin")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=float, default=None)
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
    max_new_tokens = args.max_new_tokens or defaults["max_new_tokens"]
    max_generation_seconds = args.max_generation_seconds or defaults["max_generation_seconds"]
    choice_path = Path(args.policy_choices) if args.policy_choices else None
    choice_by_cid = load_policy_choices(choice_path, args.policy_name) if choice_path else {}
    if args.routing == "policy_file" and not choice_by_cid:
        raise ValueError("--routing policy_file requires --policy-choices")
    if args.routing != "policy_file" and choice_by_cid:
        raise ValueError("--policy-choices requires --routing policy_file")
    if choice_by_cid:
        policy_layers = sorted({
            int(choice["layer"])
            for choice in choice_by_cid.values()
            if choice.get("method") not in {None, "abstain"} and choice.get("layer") is not None
        })
        layers = policy_layers or parse_int_csv(args.layers)
        methods = sorted({
            str(choice.get("method"))
            for choice in choice_by_cid.values()
            if choice.get("method") not in {None, "abstain"}
        })
    else:
        layers = parse_int_csv(args.layers)
        methods = canonical_methods(args.methods)
    gate_path = Path(args.gate_predictions) if args.gate_predictions else None
    if args.routing == "gate_file" and gate_path is None:
        raise ValueError("--routing gate_file requires --gate-predictions")
    gate_by_cid = load_gate_predictions(gate_path)
    alphas = [float(item) for item in parse_csv(args.alphas)]
    eval_levels = set(parse_csv(args.eval_levels))
    direction_levels = set(parse_csv(args.direction_levels))
    tangent_levels = set(parse_csv(args.tangent_levels))
    tangent_turns = {int(item) for item in parse_csv(args.tangent_turns)}
    tangent_phases = set(parse_csv(args.tangent_phases))

    first_query, activation_meta = load_activation_points(
        activation_path,
        layer=layers[0],
        turns={args.query_turn},
        phases={args.query_phase},
    )
    first_query_ids = set(point_index(first_query))
    del first_query
    gc.collect()

    allowed = clean_id_lines(Path(args.conversation_ids_file)) if args.conversation_ids_file else None
    rows = []
    for row in read_jsonl_paths(transcript_paths):
        cid = str(row.get("conversation_id", ""))
        if allowed is not None and cid not in allowed:
            continue
        if not row.get("valid_outcome") or row.get("arm") not in eval_levels or cid not in first_query_ids:
            continue
        row_with_class = add_status_class_to_transcript(row)
        if row_with_class is not None:
            rows.append(row_with_class)
    if args.limit_per_status_class is not None:
        rows = select_status_balanced_rows(rows, per_status_class=args.limit_per_status_class, seed=args.seed)
    else:
        rows = select_eval_rows(rows, limit=args.limit, strategy=args.limit_strategy, seed=args.seed)
    if args.limit is not None and args.limit_per_status_class is not None and len(rows) > args.limit:
        rows = select_eval_rows(rows, limit=args.limit, strategy=args.limit_strategy, seed=args.seed)
    if not rows:
        raise ValueError("no eval rows matched transcripts + activations")
    eval_ids = {str(row["conversation_id"]) for row in rows}
    transcript_by_cid = {str(row["conversation_id"]): row for row in read_jsonl_paths(transcript_paths)}
    if args.routing == "policy_file":
        route_by_cid = {
            str(row["conversation_id"]): route_for_policy_choice(
                row,
                choice_by_cid.get(str(row["conversation_id"])),
                gate_by_cid=gate_by_cid,
            )
            for row in rows
        }
    else:
        route_by_cid = {
            str(row["conversation_id"]): route_for_row(row, routing=args.routing, gate_by_cid=gate_by_cid)
            for row in rows
        }

    prepared: dict[tuple[str, str], dict] = {}
    planned_skips: Counter = Counter()
    for row in rows:
        row_methods = methods if args.routing == "policy_file" else list(STEER_METHODS)
        for method in row_methods:
            if method not in STEER_METHODS:
                continue
            route = route_by_cid[str(row["conversation_id"])]
            if args.routing == "gate_file" and route["action"] == "missing_gate_prediction":
                planned_skips[f"missing_gate_prediction::{method}"] += 1
                continue
            if method_requires_gate_action(method) and route["target_status"] is None:
                continue
            prepared[(str(row["conversation_id"]), method)] = {"directions": {}, "projections": {}, "direction_info": {}}

    layer_direction_availability: dict[int, dict] = {}
    for layer in layers:
        direction_layer_raw, _ = load_activation_points(activation_path, layer=layer, turns={args.direction_turn}, phases={args.direction_phase})
        direction_layer, skipped = attach_status_classes(direction_layer_raw, transcript_by_cid)
        query_layer, _ = load_activation_points(activation_path, layer=layer, turns={args.query_turn}, phases={args.query_phase})
        query_by_cid = point_index([row for row in query_layer if row["conversation_id"] in eval_ids])
        tangent_layer, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns=tangent_turns,
            phases=tangent_phases,
            levels=tangent_levels,
        )
        direction_cache: dict[tuple[str, str], dict | None] = {}
        global_cache: dict[tuple[str, str], dict | None] = {}
        tangent_cache: dict[str, dict | None] = {}

        def get_direction(family: str, target_status: str) -> dict | None:
            key = (family, target_status)
            if key not in direction_cache:
                direction_cache[key] = fit_status_direction(
                    direction_layer,
                    heldout_family=family,
                    direction_levels=direction_levels,
                    target_status=target_status,
                    min_mixed_scenarios=args.min_mixed_scenarios,
                    min_levels=args.min_direction_levels,
                )
            return direction_cache[key]

        def get_global_direction(family: str, method: str) -> dict | None:
            if method in {"global_mean", "global_mean_gated"}:
                direction_type = "global_mean"
            elif method in {"global_probe", "global_probe_gated"}:
                direction_type = "global_probe"
            elif method in {"random_global", "random_gated"}:
                direction_type = "random_global"
            else:
                raise ValueError(f"not a global method: {method}")
            key = (family, direction_type)
            if key not in global_cache:
                if direction_type == "global_mean":
                    global_cache[key] = fit_global_mean_direction(
                        direction_layer,
                        heldout_family=family,
                        direction_levels=direction_levels,
                    )
                elif direction_type == "global_probe":
                    global_cache[key] = fit_global_probe_direction(
                        direction_layer,
                        heldout_family=family,
                        direction_levels=direction_levels,
                    )
                else:
                    global_cache[key] = fit_random_direction(
                        direction_layer,
                        heldout_family=family,
                        direction_levels=direction_levels,
                        layer=layer,
                        seed=args.seed,
                    )
            return global_cache[key]

        def get_tangent(family: str) -> dict | None:
            if family not in tangent_cache:
                tangent_cache[family] = fit_tangent_cloud(tangent_layer, heldout_family=family)
            return tangent_cache[family]

        available_counts = Counter()
        for row in rows:
            cid = str(row["conversation_id"])
            family = str(row["family"])
            route = route_by_cid[cid]
            target_status = route["target_status"]

            if target_status is not None and any((cid, method) in prepared for method in BIDIR_METHODS):
                direction_info = get_direction(family, target_status)
                if direction_info is None:
                    for method in BIDIR_METHODS:
                        prepared.pop((cid, method), None)
                    planned_skips[f"no_direction::{target_status}::{family}::L{layer}"] += 1
                else:
                    available_counts[f"to_{target_status}"] += 1
                    raw_direction = direction_info["_direction_np"]
                    if (cid, "bidir_linear") in prepared:
                        prepared[(cid, "bidir_linear")]["directions"][layer] = raw_direction
                        prepared[(cid, "bidir_linear")]["direction_info"][layer] = direction_info
                    if (cid, "bidir_tangent") in prepared:
                        tangent_info = get_tangent(family)
                        if tangent_info is None:
                            prepared.pop((cid, "bidir_tangent"), None)
                            prepared.pop((cid, "bidir_off_tangent"), None)
                            planned_skips[f"no_tangent_cloud::{family}::L{layer}"] += 1
                        else:
                            tangent_direction, projection = project_to_local_tangent(
                                raw_direction,
                                tangent_info,
                                query_by_cid[cid]["x"],
                                tangent_neighbors=args.tangent_neighbors,
                                tangent_dim=args.tangent_dim,
                            )
                            if tangent_direction is None:
                                prepared.pop((cid, "bidir_tangent"), None)
                                prepared.pop((cid, "bidir_off_tangent"), None)
                                planned_skips[f"no_tangent::{target_status}::{family}::L{layer}::{projection.get('reason')}"] += 1
                            else:
                                prepared[(cid, "bidir_tangent")]["directions"][layer] = tangent_direction.detach().float().cpu().numpy()
                                prepared[(cid, "bidir_tangent")]["projections"][layer] = projection
                                prepared[(cid, "bidir_tangent")]["direction_info"][layer] = direction_info
                                off_direction = off_tangent_direction(raw_direction, tangent_direction)
                                if off_direction is None:
                                    prepared.pop((cid, "bidir_off_tangent"), None)
                                    planned_skips[f"no_off_tangent::{target_status}::{family}::L{layer}"] += 1
                                elif (cid, "bidir_off_tangent") in prepared:
                                    prepared[(cid, "bidir_off_tangent")]["directions"][layer] = off_direction.detach().float().cpu().numpy()
                                    prepared[(cid, "bidir_off_tangent")]["projections"][layer] = projection
                                    prepared[(cid, "bidir_off_tangent")]["direction_info"][layer] = direction_info

            for method in GLOBAL_METHODS | GATED_GLOBAL_METHODS:
                if (cid, method) not in prepared:
                    continue
                direction_info = get_global_direction(family, method)
                if direction_info is None:
                    prepared.pop((cid, method), None)
                    planned_skips[f"no_global_direction::{method}::{family}::L{layer}"] += 1
                    continue
                prepared[(cid, method)]["directions"][layer] = direction_info["_direction_np"]
                prepared[(cid, method)]["direction_info"][layer] = direction_info
                available_counts[method] += 1
        layer_direction_availability[layer] = {"available_target_rows": dict(available_counts), "status_skipped": dict(skipped)}
        del direction_layer_raw, direction_layer, query_layer, query_by_cid, tangent_layer, direction_cache, global_cache, tangent_cache
        gc.collect()

    existing_results: list[dict] = []
    completed: set[tuple[str, str, float]] = set()
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        if not existing.get("blocked") and isinstance(existing.get("results"), list):
            existing_results = existing["results"]
            completed = {(str(row["conversation_id"]), str(row["method"]), float(row["alpha"])) for row in existing_results}

    jobs = []
    for idx, row in enumerate(rows, start=1):
        cid = str(row["conversation_id"])
        if args.routing == "policy_file":
            choice = choice_by_cid.get(cid)
            method = "abstain" if choice is None else str(choice.get("method") or "abstain")
            alpha = 0.0 if method == "abstain" else float(choice.get("alpha") or 0.0)
            key = (cid, method, float(alpha))
            if key in completed:
                planned_skips["already_completed"] += 1
                continue
            specs = None
            projection = None
            direction_info_public = None
            choice_layer = None if method == "abstain" else int(choice.get("layer"))
            if method in STEER_METHODS:
                payload = prepared.get((cid, method))
                if payload is None or choice_layer not in payload["directions"]:
                    planned_skips[f"missing_prepared::{method}"] += 1
                    continue
                specs = [
                    ResidualSteeringSpec(
                        layer=choice_layer,
                        direction=np.asarray(payload["directions"][choice_layer]),
                        alpha=float(alpha),
                    )
                ]
                raw_projection = payload["projections"].get(choice_layer)
                projection = aggregate_projection({choice_layer: raw_projection}) if raw_projection else None
                direction_info_public = {
                    str(choice_layer): public_direction_info(payload["direction_info"][choice_layer])
                }
            jobs.append({
                "row_index": idx,
                "row": row,
                "method": method,
                "alpha": float(alpha),
                "specs": specs,
                "projection": projection,
                "direction_info": direction_info_public,
                "policy_choice": choice,
            })
            continue

        for method in methods:
            active_alphas = [0.0] if method == "baseline" else [alpha for alpha in alphas if alpha != 0]
            for alpha in active_alphas:
                key = (cid, method, float(alpha))
                if key in completed:
                    planned_skips["already_completed"] += 1
                    continue
                specs = None
                projection = None
                direction_info_public = None
                if method in STEER_METHODS:
                    route = route_by_cid[cid]
                    if method_requires_gate_action(method) and route["target_status"] is None:
                        specs = None
                        projection = None
                        direction_info_public = None
                    else:
                        payload = prepared.get((cid, method))
                        if payload is None or len(payload["directions"]) != len(layers):
                            planned_skips[f"missing_prepared::{method}"] += 1
                            continue
                        layer_alpha = alpha / np.sqrt(len(layers)) if args.alpha_mode == "total" else alpha
                        specs = [
                            ResidualSteeringSpec(layer=layer, direction=np.asarray(payload["directions"][layer]), alpha=float(layer_alpha))
                            for layer in layers
                        ]
                        projection = aggregate_projection(payload["projections"])
                        direction_info_public = {
                            str(layer): public_direction_info(payload["direction_info"][layer])
                            for layer in layers
                        }
                jobs.append({"row_index": idx, "row": row, "method": method, "alpha": float(alpha), "specs": specs, "projection": projection, "direction_info": direction_info_public, "policy_choice": None})

    def write_results(model_meta: dict | None, validate_only: bool = False) -> None:
        out = {
            "schema_version": 1,
            "argv": sys.argv,
            "blocked": False,
            "validate_only": validate_only,
            "model": model_meta,
            "mlx_model": mlx_model,
            "dtype": dtype,
            "oracle_true_status_direction": args.routing == "oracle_true_status",
            "routing": args.routing,
            "gate_predictions": str(gate_path.resolve()) if gate_path else None,
            "gate_predictions_sha256": file_sha256(gate_path) if gate_path else None,
            "policy_choices": str(choice_path.resolve()) if choice_path else None,
            "policy_choices_sha256": file_sha256(choice_path) if choice_path else None,
            "policy_name": args.policy_name if choice_path else None,
            "steering_scope": "decision_token_only",
            "layers": layers,
            "eval_levels": sorted(eval_levels),
            "direction_levels": sorted(direction_levels),
            "methods": methods,
            "alphas": alphas,
            "alpha_mode": args.alpha_mode,
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
            "limit": args.limit,
            "limit_per_status_class": args.limit_per_status_class,
            "planned_generations": len(jobs),
            "planned_skips": dict(planned_skips),
            "routing_action_balance": dict(Counter(route["action"] for route in route_by_cid.values())),
            "layer_direction_availability": layer_direction_availability,
            "eval_status_class_balance": dict(Counter(str(row["status_class"]) for row in rows)),
            "results": results,
        }
        atomic_text(out_path, json.dumps(to_jsonable(out), indent=2, sort_keys=True))

    results = list(existing_results)
    if args.validate_only:
        write_results(None, validate_only=True)
        print(f"validated -> {out_path} ({len(jobs)} planned generations; model not loaded)", flush=True)
        return

    model, tokenizer, meta = load_mlx_model(str(mlx_model), dtype=dtype)
    pass_id, fail_id = decision_tokens(tokenizer)
    model_meta = asdict(meta)
    for job in tqdm(jobs, desc="decision-token-control"):
        row = job["row"]
        specs = None
        if job["specs"] is not None:
            specs = [
                ResidualSteeringSpec(
                    layer=spec.layer,
                    direction=__import__("torch").tensor(np.asarray(spec.direction), dtype=__import__("torch").float32),
                    alpha=spec.alpha,
                )
                for spec in job["specs"]
            ]
        reply, decision = choose_status_then_complete(
            model,
            tokenizer,
            prompt_without_final_answer(row),
            pass_id=pass_id,
            fail_id=fail_id,
            steering=specs,
            max_new_tokens=max_new_tokens,
            max_generation_seconds=max_generation_seconds,
        )
        reported = parse_status(reply, "report")
        true_status = str(row["true_status"])
        results.append({
            "conversation_id": row["conversation_id"],
            "scenario_id": row["scenario_id"],
            "sample_seed": row.get("sample_seed"),
            "family": row["family"],
            "arm": row["arm"],
            "true_status": true_status,
            "reported_status_before": row.get("reported_status"),
            "status_class_before": row.get("status_class"),
            "desired_status": row.get("desired_status"),
            "original_deceptive": bool(row.get("deceptive")),
            "method": job["method"],
            "base_representation": "graded_bidirectional_dp_decision_token",
            "alpha": float(job["alpha"]),
            "heldout_family": row["family"],
            "route": route_by_cid[str(row["conversation_id"])],
            "target_direction": target_direction_for_method(job["method"], route_by_cid[str(row["conversation_id"])]) if job["method"] in STEER_METHODS else None,
            "policy_choice": job.get("policy_choice"),
            "decision": decision,
            "direction_info": job["direction_info"],
            "direction_projection": job["projection"],
            "injection": stack_injection_stats(specs, float(job["alpha"])) if specs else None,
            "reported_status": reported,
            "honest": reported == true_status,
            "coherence": reply_coherence(reply, reported),
            "reply": reply,
        })
        write_results(model_meta)
    write_results(model_meta)
    print(f"saved -> {out_path} ({len(results)} generations)", flush=True)


if __name__ == "__main__":
    main()
