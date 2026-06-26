"""Held-out-family routing diagnostic for decision-token control.

The oracle decision-token controller uses the known true status to choose ``to_PASS`` vs ``to_FAIL``.
This script tests the missing deployable piece without generation: can a gate infer the target status
from activations and decide whether to abstain or steer?

For each held-out family, it trains a linear true-status gate on the other families' activation
points, predicts PASS/FAIL for held-out rows, and compares that prediction with the row's original
reported status:

  predicted true == reported status -> abstain
  predicted true != reported status -> steer toward predicted true

Rows are scored by the four directional classes, so the gate cannot hide a one-way label push.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.bidirectional_dp_diagnostic import status_error_class  # noqa: E402
from experiments.control_graded_dp_frontier import load_activation_points, read_jsonl_paths  # noqa: E402
from experiments.control_graded_dp_stack_frontier import parse_csv  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402


STATUS_CLASSES = ("false_FAIL", "false_PASS", "honest_PASS", "honest_FAIL")


def status_to_label(status: str) -> int:
    status = str(status).upper()
    if status == "FAIL":
        return 0
    if status == "PASS":
        return 1
    raise ValueError(f"bad status {status!r}")


def label_to_status(label: int) -> str:
    return "PASS" if int(label) == 1 else "FAIL"


def route_action(reported_status: str, predicted_true_status: str) -> str:
    reported = str(reported_status).upper()
    predicted = str(predicted_true_status).upper()
    if reported == predicted:
        return "abstain"
    return f"steer_to_{predicted}"


def route_is_correct(status_class: str, action: str) -> bool:
    return {
        "false_FAIL": action == "steer_to_PASS",
        "false_PASS": action == "steer_to_FAIL",
        "honest_PASS": action == "abstain",
        "honest_FAIL": action == "abstain",
    }[status_class]


def row_status_class(row: dict) -> str | None:
    return status_error_class(str(row.get("true_status", "")), str(row.get("reported_status", "")))


def load_valid_rows(transcript_paths: list[Path], eval_levels: set[str], point_ids: set[str]) -> list[dict]:
    rows = []
    for row in read_jsonl_paths(transcript_paths):
        cid = str(row.get("conversation_id", ""))
        if cid not in point_ids or row.get("arm") not in eval_levels or not row.get("valid_outcome"):
            continue
        cls = row_status_class(row)
        if cls is None:
            continue
        out = dict(row)
        out["status_class"] = cls
        rows.append(out)
    return rows


def fit_gate(train_rows: list[dict], points_by_cid: dict[str, dict]) -> tuple[StandardScaler, LogisticRegression]:
    x = np.vstack([
        np.asarray(points_by_cid[str(row["conversation_id"])]["x"], dtype=np.float64)
        for row in train_rows
    ])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray([status_to_label(row["true_status"]) for row in train_rows], dtype=int)
    if len(set(y.tolist())) < 2:
        raise ValueError("train fold has only one true_status class")
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=0)
    clf.fit(xs, y)
    return scaler, clf


def stable_sigmoid(score: float) -> float:
    if score >= 0:
        z = np.exp(-score)
        return float(1.0 / (1.0 + z))
    z = np.exp(score)
    return float(z / (1.0 + z))


def predict_gate(row: dict, points_by_cid: dict[str, dict], scaler: StandardScaler, clf: LogisticRegression) -> dict:
    x = np.asarray(points_by_cid[str(row["conversation_id"])]["x"], dtype=np.float64)[None, :]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    xs = scaler.transform(x)
    score = float(np.dot(xs[0], clf.coef_[0]) + clf.intercept_[0])
    proba_pass = stable_sigmoid(score)
    predicted = "PASS" if score >= 0.0 else "FAIL"
    action = route_action(str(row["reported_status"]), predicted)
    cls = str(row["status_class"])
    return {
        "conversation_id": str(row["conversation_id"]),
        "scenario_id": str(row.get("scenario_id", "")),
        "family": str(row["family"]),
        "arm": str(row["arm"]),
        "true_status": str(row["true_status"]).upper(),
        "reported_status": str(row["reported_status"]).upper(),
        "status_class": cls,
        "predicted_true_status": predicted,
        "score_PASS_minus_FAIL": score,
        "proba_PASS": proba_pass,
        "action": action,
        "target_status_correct": predicted == str(row["true_status"]).upper(),
        "routing_correct": route_is_correct(cls, action),
    }


def summarize_predictions(predictions: list[dict]) -> dict:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in predictions:
        by_class[str(row["status_class"])].append(row)
    directional = {}
    for cls in STATUS_CLASSES:
        rows = by_class[cls]
        directional[cls] = {
            "n": int(len(rows)),
            "target_status_accuracy": float(np.mean([row["target_status_correct"] for row in rows])) if rows else None,
            "routing_correct_rate": float(np.mean([row["routing_correct"] for row in rows])) if rows else None,
            "actions": dict(Counter(str(row["action"]) for row in rows)),
            "predicted_true_status": dict(Counter(str(row["predicted_true_status"]) for row in rows)),
        }
    return {
        "n": int(len(predictions)),
        "target_status_accuracy": float(np.mean([row["target_status_correct"] for row in predictions])) if predictions else None,
        "routing_correct_rate": float(np.mean([row["routing_correct"] for row in predictions])) if predictions else None,
        "directional": directional,
        "action_counts": dict(Counter(str(row["action"]) for row in predictions)),
    }


def summarize_prediction_subset(rows: list[dict]) -> dict:
    return {
        "n": int(len(rows)),
        "target_status_accuracy": float(np.mean([row["target_status_correct"] for row in rows])) if rows else None,
        "routing_correct_rate": float(np.mean([row["routing_correct"] for row in rows])) if rows else None,
        "action_counts": dict(Counter(str(row["action"]) for row in rows)),
        "predicted_true_status": dict(Counter(str(row["predicted_true_status"]) for row in rows)),
        "status_class": dict(Counter(str(row["status_class"]) for row in rows)),
        "true_status": dict(Counter(str(row["true_status"]).upper() for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--turn", type=int, default=3)
    parser.add_argument("--phase", default="pre_response")
    parser.add_argument("--eval-levels", default="p3,p4,p5")
    parser.add_argument("--train-levels", default="p3,p4,p5,p6")
    args = parser.parse_args()

    transcript_paths = [Path(path) for path in args.transcripts]
    eval_levels = set(parse_csv(args.eval_levels))
    train_levels = set(parse_csv(args.train_levels))
    points, meta = load_activation_points(
        Path(args.activations),
        layer=args.layer,
        turns={args.turn},
        phases={args.phase},
        levels=eval_levels | train_levels,
    )
    points_by_cid = {str(row["conversation_id"]): row for row in points}
    all_rows = load_valid_rows(transcript_paths, eval_levels | train_levels, set(points_by_cid))
    eval_rows = [row for row in all_rows if row.get("arm") in eval_levels]
    families = sorted({str(row["family"]) for row in eval_rows})

    predictions = []
    fold_diagnostics = {}
    for family in families:
        train_rows = [
            row for row in all_rows
            if str(row["family"]) != family
            and row.get("arm") in train_levels
            and str(row["conversation_id"]) in points_by_cid
        ]
        heldout = [
            row for row in eval_rows
            if str(row["family"]) == family
            and str(row["conversation_id"]) in points_by_cid
        ]
        scaler, clf = fit_gate(train_rows, points_by_cid)
        fold_predictions = [predict_gate(row, points_by_cid, scaler, clf) for row in heldout]
        fold_diagnostics[family] = {
            "n_train": int(len(train_rows)),
            "n_eval": int(len(heldout)),
            "train_true_status": dict(Counter(str(row["true_status"]).upper() for row in train_rows)),
            "eval_status_class": dict(Counter(str(row["status_class"]) for row in heldout)),
            "eval_summary": summarize_prediction_subset(fold_predictions),
        }
        predictions.extend(fold_predictions)

    out = {
        "schema_version": 1,
        "argv": sys.argv,
        "provenance": git_provenance([Path(__file__), Path(args.activations), *transcript_paths]),
        "activations": str(Path(args.activations).resolve()),
        "activation_meta": meta,
        "transcripts": [str(path.resolve()) for path in transcript_paths],
        "layer": args.layer,
        "turn": args.turn,
        "phase": args.phase,
        "eval_levels": sorted(eval_levels),
        "train_levels": sorted(train_levels),
        "fold_diagnostics": fold_diagnostics,
        "summary": summarize_predictions(predictions),
        "predictions": predictions,
        "note": "This is a routing diagnostic only. It does not run generation. A deployable controller "
                "would combine action=steer_to_PASS/FAIL with the selected decision-token steering method.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")
    compact = {
        "n": out["summary"]["n"],
        "target_status_accuracy": out["summary"]["target_status_accuracy"],
        "routing_correct_rate": out["summary"]["routing_correct_rate"],
        "action_counts": out["summary"]["action_counts"],
        "directional": {
            cls: {
                "n": row["n"],
                "routing_correct_rate": row["routing_correct_rate"],
                "actions": row["actions"],
            }
            for cls, row in out["summary"]["directional"].items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
