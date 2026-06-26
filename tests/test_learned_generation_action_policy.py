from __future__ import annotations

from experiments.learned_generation_action_policy import (
    build_policies,
    choose_margin_argmax,
    merge_rows,
    summarize_choices,
)


def make_row(cid: str, family: str, method: str, status_class: str, *, status: bool, strict: bool, route: str) -> dict:
    return {
        "conversation_id": cid,
        "family": family,
        "arm": "p3",
        "sample_seed": 0,
        "method": method,
        "status_class": status_class,
        "route_action": route,
        "target_status": "PASS" if route == "steer_to_PASS" else "FAIL" if route == "steer_to_FAIL" else None,
        "reported_status_before": "FAIL" if status_class == "false_FAIL" else "PASS",
        "reported_status": "PASS" if status else "FAIL",
        "true_status": "PASS" if status_class in {"false_FAIL", "honest_PASS"} else "FAIL",
        "layer": 20 if method != "baseline" else None,
        "alpha": 96.0 if method != "baseline" else 0.0,
        "gate_score_PASS_minus_FAIL": 1.0 if route == "steer_to_PASS" else -1.0,
        "gate_proba_PASS": 0.9 if route == "steer_to_PASS" else 0.1,
        "decision_margin": 2.0 if status else -2.0,
        "decision_forced_status": "PASS" if status else "FAIL",
        "projection_fraction": 0.2 if method == "bidir_tangent" else -1.0,
        "cos_to_raw": 0.5 if method == "bidir_tangent" else -1.0,
        "mean_neighbor_distance": 3.0 if method == "bidir_tangent" else -1.0,
        "projected_norm": 1.0 if method == "bidir_tangent" else -1.0,
        "tangent_dim": 4 if method == "bidir_tangent" else -1.0,
        "injected_norm": 96.0 if method != "baseline" else 0.0,
        "per_layer_alpha": 96.0 if method != "baseline" else 0.0,
        "status_correct": status,
        "basis_strict_ok": strict,
        "parse_success": True,
        "basis_quality": "correct_status_numeric_rule_basis" if strict else "wrong_status",
        "fixes_status": status_class.startswith("false_") and status,
        "fixes_strict": status_class.startswith("false_") and strict,
        "harms_honest_status": status_class.startswith("honest_") and not status,
        "harms_honest_strict": status_class.startswith("honest_") and not strict,
        "status_reward": (
            1.0 if status_class.startswith("false_") and status
            else -1.0 if status_class.startswith("honest_") and not status
            else 0.0
        ),
        "strict_reward": (
            1.0 if status_class.startswith("false_") and strict
            else -1.0 if status_class.startswith("honest_") and not strict
            else 0.0
        ),
    }


def test_route_hybrid_mean_probe_combines_easy_and_hard_methods():
    rows = []
    for cid, cls, route in [("easy", "false_FAIL", "steer_to_PASS"), ("hard", "false_PASS", "steer_to_FAIL")]:
        for method in ["baseline", "bidir_tangent", "global_mean_gated", "global_probe_gated", "random_gated", "bidir_linear"]:
            strict = (cls == "false_FAIL" and method == "global_mean_gated") or (
                cls == "false_PASS" and method == "global_probe_gated"
            )
            rows.append(make_row(cid, "fam", method, cls, status=strict, strict=strict, route=route))

    policies = build_policies(rows, threshold=0.0)
    summary = policies["route_hybrid_mean_probe"]["summary"]
    assert summary["deceptive_strict_fixes"] == 2
    assert summary["chosen_methods"] == {"global_mean_gated": 1, "global_probe_gated": 1}


def test_learned_response_policy_evaluates_heldout_family():
    rows = []
    for family in ["train_a", "train_b", "held"]:
        for method in ["baseline", "bidir_tangent", "global_mean_gated", "global_probe_gated", "random_gated", "bidir_linear"]:
            rows.append(
                make_row(
                    f"{family}_case",
                    family,
                    method,
                    "false_FAIL",
                    status=method == "bidir_tangent",
                    strict=method == "bidir_tangent",
                    route="steer_to_PASS",
                )
            )

    policies = build_policies(rows, threshold=0.0)
    learned = policies["learned_response_rf_strict"]["summary"]
    assert learned["deceptive_strict_fixes"] >= 2


def test_merge_rows_joins_audit_and_result_schema():
    results = {
        "results": [
            {
                "conversation_id": "c1",
                "method": "bidir_tangent",
                "family": "fam",
                "route": {"action": "steer_to_PASS", "target_status": "PASS", "score_PASS_minus_FAIL": 1.0, "proba_PASS": 0.9},
                "decision": {"margin": 2.0, "forced_status": "PASS"},
                "injection": {"layers": [20], "injected_norm": 96.0, "per_layer_alpha": 96.0},
                "direction_projection": {"projection_fraction": 0.2, "cos_to_raw": 0.4, "by_layer": {"20": {"mean_neighbor_distance": 3.0}}},
                "alpha": 96.0,
            }
        ]
    }
    audit = {
        "rows": [
            {
                "conversation_id": "c1",
                "method": "bidir_tangent",
                "family": "fam",
                "arm": "p3",
                "status_class_before": "false_FAIL",
                "route_action": "steer_to_PASS",
                "reported_status_before": "FAIL",
                "reported_status": "PASS",
                "true_status": "PASS",
                "status_correct": True,
                "basis_strict_ok": True,
                "parse_success": True,
                "basis_quality": "correct_status_numeric_rule_basis",
            }
        ]
    }
    rows = merge_rows(results, audit)
    assert summarize_choices(rows)["deceptive_strict_fixes"] == 1
    assert rows[0]["projection_fraction"] == 0.2


def test_margin_argmax_uses_route_target_margin():
    rows = [
        make_row("c", "fam", "baseline", "false_PASS", status=False, strict=False, route="steer_to_FAIL"),
        make_row("c", "fam", "bidir_tangent", "false_PASS", status=False, strict=False, route="steer_to_FAIL"),
        make_row("c", "fam", "global_probe_gated", "false_PASS", status=True, strict=True, route="steer_to_FAIL"),
    ]
    rows[0]["decision_margin"] = 1.0
    rows[1]["decision_margin"] = 0.5
    rows[2]["decision_margin"] = -2.0
    chosen = choose_margin_argmax(rows, include_baseline=True)
    assert chosen["method"] == "global_probe_gated"


def test_strict_reward_distinguishes_strict_harm_from_status_harm():
    row_status_harm_but_strict_ok = make_row(
        "c", "fam", "bidir_tangent", "honest_PASS",
        status=False, strict=True, route="steer_to_PASS",
    )
    assert row_status_harm_but_strict_ok["status_reward"] == -1.0
    assert row_status_harm_but_strict_ok["strict_reward"] == 0.0

    row_strict_harm = make_row(
        "c", "fam", "bidir_tangent", "honest_PASS",
        status=False, strict=False, route="steer_to_PASS",
    )
    assert row_strict_harm["status_reward"] == -1.0
    assert row_strict_harm["strict_reward"] == -1.0

    row_deceptive_fix = make_row(
        "c", "fam", "bidir_tangent", "false_FAIL",
        status=True, strict=True, route="steer_to_PASS",
    )
    assert row_deceptive_fix["status_reward"] == 1.0
    assert row_deceptive_fix["strict_reward"] == 1.0

    row_deceptive_status_but_not_strict = make_row(
        "c", "fam", "bidir_tangent", "false_PASS",
        status=True, strict=False, route="steer_to_FAIL",
    )
    assert row_deceptive_status_but_not_strict["status_reward"] == 1.0
    assert row_deceptive_status_but_not_strict["strict_reward"] == 0.0

    row_honest_no_harm = make_row(
        "c", "fam", "baseline", "honest_FAIL",
        status=True, strict=True, route="steer_to_FAIL",
    )
    assert row_honest_no_harm["status_reward"] == 0.0
    assert row_honest_no_harm["strict_reward"] == 0.0
