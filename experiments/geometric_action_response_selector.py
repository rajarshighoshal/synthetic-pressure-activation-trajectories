"""Geometry-native selectors over audited decision-token action responses.

This is the low-capacity version of the point-cloud controller: candidate
case-action rows are embedded as a typed action-response point cloud, then a
held-out-family kNN / metric selector predicts which action should work from
local neighborhoods.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.learned_generation_action_policy import (  # noqa: E402
    baseline_row,
    build_policies,
    file_sha256,
    grouped_by_conversation,
    merge_rows,
    paired_gap,
    safe_float,
    slim_choice,
    summarize_choices,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402


def geom_features(row: dict, *, include_response_margin: bool) -> dict[str, str | float]:
    route = str(row.get("route_action") or "unknown")
    target = str(row.get("target_status") or "NONE")
    target_sign = 1.0 if target == "PASS" else -1.0 if target == "FAIL" else 0.0
    decision_margin = safe_float(row.get("decision_margin"))
    out: dict[str, str | float] = {
        "method": str(row.get("method")),
        "route_action": route,
        "target_status": target,
        "reported_status_before": str(row.get("reported_status_before") or "UNKNOWN"),
        "arm": str(row.get("arm") or ""),
        "layer": safe_float(row.get("layer"), -1.0),
        "alpha": safe_float(row.get("alpha")),
        "gate_score": safe_float(row.get("gate_score_PASS_minus_FAIL")),
        "abs_gate_score": abs(safe_float(row.get("gate_score_PASS_minus_FAIL"))),
        "gate_proba_PASS": safe_float(row.get("gate_proba_PASS"), 0.5),
        "projection_fraction": safe_float(row.get("projection_fraction"), -1.0),
        "cos_to_raw": safe_float(row.get("cos_to_raw"), -1.0),
        "mean_neighbor_distance": safe_float(row.get("mean_neighbor_distance"), -1.0),
        "projected_norm": safe_float(row.get("projected_norm"), -1.0),
        "tangent_dim": safe_float(row.get("tangent_dim"), -1.0),
        "injected_norm": safe_float(row.get("injected_norm")),
        "per_layer_alpha": safe_float(row.get("per_layer_alpha")),
    }
    if include_response_margin:
        out.update({
            "decision_margin": decision_margin,
            "abs_decision_margin": abs(decision_margin),
            "target_aligned_decision_margin": target_sign * decision_margin,
            "decision_forced_status": str(row.get("decision_forced_status") or "UNKNOWN"),
        })
    return out


class GeometricKnnSelector:
    def __init__(
        self,
        *,
        k: int,
        metric: str,
        objective: str,
        include_response_margin: bool,
        shrinkage: float,
    ) -> None:
        self.k = int(k)
        self.metric = metric
        self.objective = objective
        self.include_response_margin = include_response_margin
        self.shrinkage = float(shrinkage)

    def fit(self, rows: list[dict]) -> "GeometricKnnSelector":
        self.rows_ = list(rows)
        self.y_ = np.asarray([safe_float(row[self.objective]) for row in rows], dtype=np.float64)
        self.vectorizer_ = DictVectorizer(sparse=False)
        raw = self.vectorizer_.fit_transform([
            geom_features(row, include_response_margin=self.include_response_margin)
            for row in rows
        ])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler()
        x = self.scaler_.fit_transform(raw) if self.metric != "mahalanobis" else raw
        self.x_ = self._transform_metric(x, raw=raw, fit=True)
        self.raw_x_ = raw
        return self

    def _transform_metric(self, x: np.ndarray, *, raw: np.ndarray | None = None, fit: bool) -> np.ndarray:
        if self.metric == "euclidean":
            return x
        if self.metric == "supervised_diag":
            if fit:
                pos = self.y_ > 0
                non = ~pos
                if pos.any() and non.any():
                    delta = x[pos].mean(axis=0) - x[non].mean(axis=0)
                    weights = delta * delta
                else:
                    weights = np.ones(x.shape[1], dtype=np.float64)
                weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                if float(weights.mean()) <= 1e-12:
                    weights = np.ones_like(weights)
                weights = weights / max(float(weights.mean()), 1e-12)
                self.weights_ = np.clip(weights, 0.05, 20.0)
            return x * np.sqrt(self.weights_[None, :])
        if self.metric == "mahalanobis":
            if fit:
                if raw is None:
                    raw = x
                var = np.var(raw, axis=0)
                var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
                scale = float(np.mean(var))
                if not np.isfinite(scale) or scale <= 1e-12:
                    scale = 1.0
                denom = np.clip((1.0 - self.shrinkage) * var + self.shrinkage * scale, 1e-6, None)
                self.mahalanobis_weights_ = 1.0 / denom
            return x * np.sqrt(self.mahalanobis_weights_[None, :])
        raise ValueError(f"unknown metric {self.metric!r}")

    def transform(self, rows: list[dict]) -> np.ndarray:
        raw = self.vectorizer_.transform([
            geom_features(row, include_response_margin=self.include_response_margin)
            for row in rows
        ])
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self.metric == "mahalanobis":
            x = raw
        else:
            x = self.scaler_.transform(raw)
        return self._transform_metric(x, raw=raw, fit=False)

    def score(self, rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float64)
        q = self.transform(rows)
        scores = []
        for query in q:
            d = np.linalg.norm(self.x_ - query[None, :], axis=1)
            kk = min(self.k, len(d))
            order = np.argsort(d)[:kk]
            local_d = d[order]
            if np.all(local_d <= 1e-12):
                weights = np.ones_like(local_d)
            else:
                scale = float(np.median(local_d) + 1e-6)
                weights = np.exp(-local_d / scale)
            score = float(np.sum(weights * self.y_[order]) / max(float(np.sum(weights)), 1e-12))
            scores.append(score)
        return np.asarray(scores, dtype=np.float64)


def choose_geometric(candidates: list[dict], selector: GeometricKnnSelector, *, threshold: float) -> dict:
    if str(candidates[0].get("route_action")) == "abstain":
        return baseline_row(candidates)
    scores = selector.score(candidates)
    best = int(np.argmax(scores))
    if float(scores[best]) <= threshold:
        chosen = dict(baseline_row(candidates))
        chosen["policy_score"] = float(scores[best])
        chosen["policy_abstained_by_threshold"] = True
        return chosen
    chosen = dict(candidates[best])
    chosen["policy_score"] = float(scores[best])
    chosen["policy_abstained_by_threshold"] = False
    return chosen


def evaluate_geometric(
    rows: list[dict],
    *,
    k: int,
    metric: str,
    objective: str,
    include_response_margin: bool,
    threshold: float,
    shrinkage: float,
) -> dict:
    grouped = grouped_by_conversation(rows)
    families = sorted({str(row["family"]) for row in rows})
    choices = []
    folds = {}
    for family in families:
        train = [row for row in rows if str(row["family"]) != family]
        selector = GeometricKnnSelector(
            k=k,
            metric=metric,
            objective=objective,
            include_response_margin=include_response_margin,
            shrinkage=shrinkage,
        ).fit(train)
        fold_choices = []
        for candidates in grouped.values():
            if str(candidates[0]["family"]) != family:
                continue
            fold_choices.append(choose_geometric(candidates, selector, threshold=threshold))
        choices.extend(fold_choices)
        folds[family] = {"summary": summarize_choices(fold_choices)}
    return {"summary": summarize_choices(choices), "folds": folds, "choices": choices}


def build_geometric_policies(
    rows: list[dict],
    *,
    k_values: list[int],
    threshold: float,
    shrinkage: float,
) -> dict[str, dict]:
    out = {}
    for include_response_margin, mode_name in [(False, "context"), (True, "response")]:
        for metric in ("euclidean", "supervised_diag", "mahalanobis"):
            for k in k_values:
                name = f"geom_{mode_name}_{metric}_k{k}_strict"
                out[name] = evaluate_geometric(
                    rows,
                    k=k,
                    metric=metric,
                    objective="strict_reward",
                    include_response_margin=include_response_margin,
                    threshold=threshold,
                    shrinkage=shrinkage,
                )
    return out


def best_policy_name(policies: dict[str, dict]) -> str:
    def key(item):
        name, value = item
        s = value["summary"]
        return (
            s["deceptive_strict_fixes"],
            s["deceptive_status_fixes"],
            -s["honest_status_harms"],
            s["parse_success"],
            name,
        )
    return max(policies.items(), key=key)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--action-response", default=None,
                        help="Optional action-response JSON to propagate point-cloud "
                             "context features into merged rows.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--k-values", default="5,11,21")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--shrinkage", type=float, default=0.2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results_path = Path(args.results)
    audit_path = Path(args.audit)
    ar_rows = None
    if args.action_response:
        ar_path = Path(args.action_response)
        ar_rows = json.loads(ar_path.read_text()).get("rows", [])
    rows = merge_rows(
        json.loads(results_path.read_text()),
        json.loads(audit_path.read_text()),
        action_response_rows=ar_rows,
    )
    k_values = [int(item) for item in args.k_values.split(",") if item.strip()]
    baselines = build_policies(rows, threshold=args.threshold)
    geometric = build_geometric_policies(
        rows,
        k_values=k_values,
        threshold=args.threshold,
        shrinkage=args.shrinkage,
    )
    policies = {**baselines, **geometric}
    best_geom = best_policy_name(geometric)
    references = [
        "fixed_bidir_tangent",
        "route_hybrid_mean_probe",
        "learned_response_rf_strict",
        "fixed_random_gated",
        "fixed_global_probe_gated",
    ]
    gaps = {}
    for name, policy in policies.items():
        gaps[name] = {}
        for ref in references:
            if name == ref or ref not in policies:
                continue
            gaps[name][ref] = {
                "status_fix": paired_gap(policy["choices"], policies[ref]["choices"], "status_fix", seed=args.seed, bootstrap=args.bootstrap),
                "strict_fix": paired_gap(policy["choices"], policies[ref]["choices"], "strict_fix", seed=args.seed, bootstrap=args.bootstrap),
                "honest_status_harm": paired_gap(policy["choices"], policies[ref]["choices"], "honest_status_harm", seed=args.seed, bootstrap=args.bootstrap),
            }
    out = {
        "schema_version": 1,
        "argv": sys.argv,
        "results": str(results_path.resolve()),
        "results_sha256": file_sha256(results_path),
        "audit": str(audit_path.resolve()),
        "audit_sha256": file_sha256(audit_path),
        "provenance": git_provenance([Path(__file__), results_path, audit_path]),
        "k_values": k_values,
        "threshold": args.threshold,
        "shrinkage": args.shrinkage,
        "n_candidate_rows": len(rows),
        "n_conversations": len(grouped_by_conversation(rows)),
        "family_balance": dict(Counter(row["family"] for row in rows if row["method"] == "baseline")),
        "status_class_balance": dict(Counter(row["status_class"] for row in rows if row["method"] == "baseline")),
        "best_geometric_policy": best_geom,
        "policies": {
            name: {
                "summary": policy["summary"],
                "folds": policy.get("folds"),
                "choices": [slim_choice(row) for row in policy["choices"]],
            }
            for name, policy in policies.items()
        },
        "paired_gaps": gaps,
        "note": (
            "Geometry selectors are held-out-family kNN/metric policies over typed "
            "case-action response point clouds. Response variants use candidate "
            "decision-token margin features and therefore assume a cheap action-response "
            "pass before full generation."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"saved -> {args.out}")
    for name in sorted(geometric):
        s = geometric[name]["summary"]
        marker = "*" if name == best_geom else " "
        print(
            f"{marker} {name:42s} status={s['deceptive_status_fixes']:2d}/{s['deceptive_n']} "
            f"strict={s['deceptive_strict_fixes']:2d}/{s['deceptive_n']} "
            f"harm={s['honest_status_harms']:2d}/{s['honest_n']} "
            f"methods={s['chosen_methods']}"
        )


if __name__ == "__main__":
    main()
