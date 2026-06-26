from __future__ import annotations

from experiments.geometric_action_response_selector import (
    build_geometric_policies,
    evaluate_geometric,
)
from tests.test_learned_generation_action_policy import make_row


def test_geometric_response_knn_selects_action_by_local_response():
    rows = []
    for family in ["a", "b", "held"]:
        for method in ["baseline", "bidir_tangent", "global_mean_gated", "global_probe_gated", "random_gated", "bidir_linear"]:
            strict = method == "global_probe_gated"
            row = make_row(
                f"{family}_case",
                family,
                method,
                "false_PASS",
                status=strict,
                strict=strict,
                route="steer_to_FAIL",
            )
            row["decision_margin"] = -4.0 if strict else 4.0
            row["decision_forced_status"] = "FAIL" if strict else "PASS"
            rows.append(row)

    result = evaluate_geometric(
        rows,
        k=1,
        metric="euclidean",
        objective="strict_reward",
        include_response_margin=True,
        threshold=0.0,
        shrinkage=0.2,
    )
    assert result["summary"]["deceptive_strict_fixes"] == 3
    assert result["summary"]["chosen_methods"] == {"global_probe_gated": 3}


def test_geometric_context_without_response_can_fail_when_response_needed():
    rows = []
    for family in ["a", "b", "held"]:
        for method in ["baseline", "bidir_tangent", "global_mean_gated", "global_probe_gated", "random_gated", "bidir_linear"]:
            strict = method == "global_probe_gated"
            row = make_row(
                f"{family}_case",
                family,
                method,
                "false_PASS",
                status=strict,
                strict=strict,
                route="steer_to_FAIL",
            )
            row["decision_margin"] = -4.0 if strict else 4.0
            row["decision_forced_status"] = "FAIL" if strict else "PASS"
            rows.append(row)

    policies = build_geometric_policies(rows, k_values=[1], threshold=0.0, shrinkage=0.2)
    response = policies["geom_response_euclidean_k1_strict"]["summary"]["deceptive_strict_fixes"]
    context = policies["geom_context_euclidean_k1_strict"]["summary"]["deceptive_strict_fixes"]
    assert response >= context
    assert response == 3
