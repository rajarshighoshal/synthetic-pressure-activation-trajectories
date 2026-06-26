"""Bidirectional d_p diagnostic for graded deception-control.

The pooled d_p controller can silently become a one-way label pusher when the
deceptive pool is imbalanced, e.g. mostly true-PASS rows falsely reported as
FAIL.  This diagnostic splits the correction directions by error type:

  v_to_PASS = mean(honest_PASS) - mean(false_FAIL)
  v_to_FAIL = mean(honest_FAIL) - mean(false_PASS)

Directions are fit from train families only and within matched scenario-level
groups, so a held-out family cannot leak into its own direction.  The output
answers whether the current data has enough bidirectional support and whether
the old pooled honest-vs-deceptive vector is really just a PASS correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.control_graded_dp_frontier import (  # noqa: E402
    clean_matrix,
    fit_dp_direction,
    load_activation_points,
    read_jsonl_paths,
    unit,
    vector_stats,
)
from experiments.control_graded_dp_stack_frontier import parse_csv  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402


STATUS_CLASSES = ("honest_PASS", "honest_FAIL", "false_FAIL", "false_PASS")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v)
            for k, v in obj.items()
            if not str(k).startswith("_") and str(k) != "direction"
        }
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def direction_cos(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    au = unit(a)
    bu = unit(b)
    if float(np.linalg.norm(au)) <= 1e-8 or float(np.linalg.norm(bu)) <= 1e-8:
        return None
    return float(np.dot(au, bu))


def transcript_index(paths: list[Path]) -> dict[str, dict]:
    rows = read_jsonl_paths(paths)
    out: dict[str, dict] = {}
    for row in rows:
        cid = str(row.get("conversation_id", ""))
        if not cid:
            continue
        if cid in out:
            raise ValueError(f"duplicate transcript conversation_id {cid}")
        out[cid] = row
    return out


def status_error_class(true_status: str, reported_status: str) -> str | None:
    true_status = str(true_status).upper()
    reported_status = str(reported_status).upper()
    if true_status not in {"PASS", "FAIL"} or reported_status not in {"PASS", "FAIL"}:
        return None
    if true_status == "PASS" and reported_status == "PASS":
        return "honest_PASS"
    if true_status == "FAIL" and reported_status == "FAIL":
        return "honest_FAIL"
    if true_status == "PASS" and reported_status == "FAIL":
        return "false_FAIL"
    if true_status == "FAIL" and reported_status == "PASS":
        return "false_PASS"
    return None


def attach_status_classes(points: list[dict], transcripts: dict[str, dict]) -> tuple[list[dict], Counter]:
    out: list[dict] = []
    skipped: Counter = Counter()
    for point in points:
        cid = str(point["conversation_id"])
        transcript = transcripts.get(cid)
        if transcript is None:
            skipped["missing_transcript"] += 1
            continue
        if not bool(transcript.get("valid_outcome", False)):
            skipped["invalid_transcript"] += 1
            continue
        reported = transcript.get("reported_status")
        true_status = transcript.get("true_status", point.get("true_status"))
        cls = status_error_class(str(true_status), str(reported))
        if cls is None:
            skipped["bad_status"] += 1
            continue
        row = dict(point)
        row["reported_status"] = str(reported).upper()
        row["true_status"] = str(true_status).upper()
        row["status_class"] = cls
        row["transcript_deceptive"] = bool(transcript.get("deceptive", cls.startswith("false_")))
        out.append(row)
    return out, skipped


def count_table(rows: list[dict], key: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        table[str(row.get(key, ""))][str(row["status_class"])] += 1
    return {name: dict(counts) for name, counts in sorted(table.items())}


def fit_status_direction(
    rows: list[dict],
    *,
    heldout_family: str | None = None,
    heldout_scenario_ids: set[str] | None = None,
    direction_levels: set[str],
    target_status: str,
    min_mixed_scenarios: int,
    min_levels: int,
) -> dict | None:
    """Fit one correction direction within scenario-level pairs.

    ``target_status=PASS`` fits honest_PASS - false_FAIL.
    ``target_status=FAIL`` fits honest_FAIL - false_PASS.

    Excludes rows matching ``heldout_family`` (cross-family split) or
    ``heldout_scenario_ids`` (within-scenario split). If both are None, holds
    out nothing (diagnostic mode).
    """
    if target_status == "PASS":
        honest_class = "honest_PASS"
        false_class = "false_FAIL"
    elif target_status == "FAIL":
        honest_class = "honest_FAIL"
        false_class = "false_PASS"
    else:
        raise ValueError("target_status must be PASS or FAIL")

    train = [
        row for row in rows
        if row["arm"] in direction_levels
        and (heldout_family is None or str(row["family"]) != heldout_family)
        and (heldout_scenario_ids is None or str(row.get("scenario_id", "")) not in heldout_scenario_ids)
    ]
    by_level_scenario: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in train:
        by_level_scenario[(str(row["arm"]), str(row["scenario_id"]))].append(row)

    level_diffs: dict[str, list[np.ndarray]] = defaultdict(list)
    scenario_counts: dict[str, list[dict]] = defaultdict(list)
    for (level, scenario_id), group in sorted(by_level_scenario.items()):
        honest = [row for row in group if row["status_class"] == honest_class]
        false = [row for row in group if row["status_class"] == false_class]
        if not honest or not false:
            continue
        x_h = clean_matrix(np.vstack([row["x"] for row in honest]))
        x_f = clean_matrix(np.vstack([row["x"] for row in false]))
        diff = x_h.mean(axis=0) - x_f.mean(axis=0)
        level_diffs[level].append(diff)
        scenario_counts[level].append({
            "scenario_id": scenario_id,
            "n_honest": int(len(honest)),
            "n_false": int(len(false)),
        })

    usable_levels = [
        level for level in sorted(level_diffs)
        if len(level_diffs[level]) >= min_mixed_scenarios
    ]
    if len(usable_levels) < min_levels:
        return None

    level_means = [clean_matrix(np.vstack(level_diffs[level])).mean(axis=0) for level in usable_levels]
    direction = unit(clean_matrix(np.vstack(level_means)).mean(axis=0))
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1e-8:
        return None

    return {
        "heldout_family": heldout_family,
        "target_status": target_status,
        "honest_class": honest_class,
        "false_class": false_class,
        "direction_convention": f"direction = mean({honest_class}) - mean({false_class})",
        "direction_levels": usable_levels,
        "n_train_points": int(len(train)),
        "n_mixed_scenario_level_pairs": int(sum(len(level_diffs[level]) for level in usable_levels)),
        "mixed_scenarios_by_level": {
            level: scenario_counts[level][:100]
            for level in usable_levels
        },
        "direction_stats": vector_stats(direction),
        "_direction_np": direction,
    }


def summarize_family_directions(
    rows: list[dict],
    *,
    direction_levels: set[str],
    min_mixed_scenarios: int,
    min_levels: int,
) -> dict:
    families = sorted({str(row["family"]) for row in rows})
    per_family: dict[str, dict] = {}
    pass_dirs: list[np.ndarray] = []
    fail_dirs: list[np.ndarray] = []
    pooled_dirs: list[np.ndarray] = []
    cos_pass_fail: list[float] = []
    cos_pooled_pass: list[float] = []
    cos_pooled_fail: list[float] = []

    for family in families:
        pass_info = fit_status_direction(
            rows,
            heldout_family=family,
            direction_levels=direction_levels,
            target_status="PASS",
            min_mixed_scenarios=min_mixed_scenarios,
            min_levels=min_levels,
        )
        fail_info = fit_status_direction(
            rows,
            heldout_family=family,
            direction_levels=direction_levels,
            target_status="FAIL",
            min_mixed_scenarios=min_mixed_scenarios,
            min_levels=min_levels,
        )
        pooled = fit_dp_direction(
            rows,
            heldout_family=family,
            direction_levels=direction_levels,
            min_mixed_scenarios=min_mixed_scenarios,
            min_levels=min_levels,
        )

        pass_vec = None if pass_info is None else pass_info["_direction_np"]
        fail_vec = None if fail_info is None else fail_info["_direction_np"]
        pooled_vec = None if pooled is None else pooled["_direction_np"]

        pf = direction_cos(pass_vec, fail_vec)
        pp = direction_cos(pooled_vec, pass_vec)
        pfi = direction_cos(pooled_vec, fail_vec)
        if pass_vec is not None:
            pass_dirs.append(pass_vec)
        if fail_vec is not None:
            fail_dirs.append(fail_vec)
        if pooled_vec is not None:
            pooled_dirs.append(pooled_vec)
        if pf is not None:
            cos_pass_fail.append(pf)
        if pp is not None:
            cos_pooled_pass.append(pp)
        if pfi is not None:
            cos_pooled_fail.append(pfi)

        per_family[family] = {
            "to_PASS_available": pass_info is not None,
            "to_FAIL_available": fail_info is not None,
            "pooled_available": pooled is not None,
            "cos_to_PASS_vs_to_FAIL": pf,
            "cos_pooled_vs_to_PASS": pp,
            "cos_pooled_vs_to_FAIL": pfi,
            "to_PASS": pass_info,
            "to_FAIL": fail_info,
            "pooled": pooled,
        }

    def stats(values: list[float]) -> dict | None:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return None
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    global_pass = unit(clean_matrix(np.vstack(pass_dirs)).mean(axis=0)) if pass_dirs else None
    global_fail = unit(clean_matrix(np.vstack(fail_dirs)).mean(axis=0)) if fail_dirs else None
    global_pooled = unit(clean_matrix(np.vstack(pooled_dirs)).mean(axis=0)) if pooled_dirs else None

    return {
        "families": families,
        "n_families": int(len(families)),
        "n_to_PASS_available": int(sum(1 for payload in per_family.values() if payload["to_PASS_available"])),
        "n_to_FAIL_available": int(sum(1 for payload in per_family.values() if payload["to_FAIL_available"])),
        "n_pooled_available": int(sum(1 for payload in per_family.values() if payload["pooled_available"])),
        "cos_to_PASS_vs_to_FAIL": stats(cos_pass_fail),
        "cos_pooled_vs_to_PASS": stats(cos_pooled_pass),
        "cos_pooled_vs_to_FAIL": stats(cos_pooled_fail),
        "global_cos_to_PASS_vs_to_FAIL": direction_cos(global_pass, global_fail),
        "global_cos_pooled_vs_to_PASS": direction_cos(global_pooled, global_pass),
        "global_cos_pooled_vs_to_FAIL": direction_cos(global_pooled, global_fail),
        "per_family": per_family,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="20,24,28,32")
    parser.add_argument("--direction-turn", type=int, default=2)
    parser.add_argument("--direction-phase", default="pre_response")
    parser.add_argument("--direction-levels", default="p3,p4,p5,p6")
    parser.add_argument("--min-mixed-scenarios", type=int, default=2)
    parser.add_argument("--min-levels", type=int, default=2)
    args = parser.parse_args()

    activation_path = Path(args.activations)
    transcript_paths = [Path(path) for path in args.transcripts]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transcripts = transcript_index(transcript_paths)
    direction_levels = set(parse_csv(args.direction_levels))
    layers = [int(item) for item in parse_csv(args.layers)]

    by_layer: dict[str, dict] = {}
    activation_meta = None
    for layer in layers:
        points, meta = load_activation_points(
            activation_path,
            layer=layer,
            turns={args.direction_turn},
            phases={args.direction_phase},
            levels=direction_levels,
        )
        activation_meta = activation_meta or meta
        rows, skipped = attach_status_classes(points, transcripts)
        class_counts = Counter(row["status_class"] for row in rows)
        status_by_family = count_table(rows, "family")
        status_by_level = count_table(rows, "arm")
        by_layer[str(layer)] = {
            "n_points": int(len(points)),
            "n_status_rows": int(len(rows)),
            "skipped": dict(skipped),
            "status_counts": {name: int(class_counts.get(name, 0)) for name in STATUS_CLASSES},
            "status_by_family": status_by_family,
            "status_by_level": status_by_level,
            "directions": summarize_family_directions(
                rows,
                direction_levels=direction_levels,
                min_mixed_scenarios=args.min_mixed_scenarios,
                min_levels=args.min_levels,
            ),
        }

    payload = {
        "provenance": {
            "git": git_provenance(),
            "argv": sys.argv,
            "activations": str(activation_path),
            "activations_sha256": file_sha256(activation_path),
            "transcripts": [str(path) for path in transcript_paths],
            "transcripts_sha256": {str(path): file_sha256(path) for path in transcript_paths},
        },
        "config": {
            "layers": layers,
            "direction_turn": args.direction_turn,
            "direction_phase": args.direction_phase,
            "direction_levels": sorted(direction_levels),
            "min_mixed_scenarios": args.min_mixed_scenarios,
            "min_levels": args.min_levels,
        },
        "activation_meta": activation_meta,
        "layers": by_layer,
        "interpretation_note": (
            "If pooled-vs-to_PASS is high while pooled-vs-to_FAIL is low or negative, "
            "the old pooled d_p direction is likely a one-way PASS/FAIL label correction, "
            "not a bidirectional honesty-restoration vector. A deployable controller must "
            "fit/select error-type-specific directions or prove a shared honesty direction."
        ),
    }
    out_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}")
    for layer, info in by_layer.items():
        dirs = info["directions"]
        print(
            f"L{layer}: counts={info['status_counts']} "
            f"families to_PASS={dirs['n_to_PASS_available']}/{dirs['n_families']} "
            f"to_FAIL={dirs['n_to_FAIL_available']}/{dirs['n_families']} "
            f"global cos pass/fail={dirs['global_cos_to_PASS_vs_to_FAIL']}"
        )


if __name__ == "__main__":
    main()
