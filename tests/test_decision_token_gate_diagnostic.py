from __future__ import annotations

from experiments.control_graded_dp_decision_token import (
    canonical_methods,
    method_requires_gate_action,
    route_for_row,
    target_direction_for_method,
    target_direction_name_from_route,
)
from experiments.decision_token_gate_diagnostic import route_action, route_is_correct, status_to_label


def test_status_to_label():
    assert status_to_label("FAIL") == 0
    assert status_to_label("PASS") == 1


def test_route_action_abstains_when_prediction_matches_report():
    assert route_action("PASS", "PASS") == "abstain"
    assert route_action("FAIL", "FAIL") == "abstain"
    assert route_action("PASS", "FAIL") == "steer_to_FAIL"
    assert route_action("FAIL", "PASS") == "steer_to_PASS"


def test_route_correctness_is_directional():
    assert route_is_correct("false_FAIL", "steer_to_PASS")
    assert not route_is_correct("false_FAIL", "abstain")
    assert route_is_correct("false_PASS", "steer_to_FAIL")
    assert not route_is_correct("false_PASS", "steer_to_PASS")
    assert route_is_correct("honest_PASS", "abstain")
    assert not route_is_correct("honest_PASS", "steer_to_FAIL")
    assert route_is_correct("honest_FAIL", "abstain")
    assert not route_is_correct("honest_FAIL", "steer_to_PASS")


def test_gate_file_route_maps_action_to_target_status():
    row = {
        "conversation_id": "c1",
        "true_status": "FAIL",
        "reported_status": "PASS",
    }
    gate_by_cid = {
        "c1": {
            "action": "steer_to_FAIL",
            "predicted_true_status": "FAIL",
            "target_status_correct": True,
            "routing_correct": True,
        }
    }
    route = route_for_row(row, routing="gate_file", gate_by_cid=gate_by_cid)
    assert route["target_status"] == "FAIL"
    assert target_direction_name_from_route(route) == "to_FAIL"


def test_gate_file_route_abstain_has_no_target_direction():
    row = {
        "conversation_id": "c2",
        "true_status": "PASS",
        "reported_status": "PASS",
    }
    gate_by_cid = {"c2": {"action": "abstain", "predicted_true_status": "PASS"}}
    route = route_for_row(row, routing="gate_file", gate_by_cid=gate_by_cid)
    assert route["target_status"] is None
    assert target_direction_name_from_route(route) is None


def test_global_method_aliases_distinguish_ungated_from_gated():
    assert canonical_methods("global_mean,global_mean_gated,random")[0] == "global_mean"
    assert canonical_methods("global_mean,global_mean_gated,random")[1] == "global_mean_gated"
    assert canonical_methods("global_mean,global_mean_gated,random")[2] == "random_global"

    assert not method_requires_gate_action("global_mean")
    assert not method_requires_gate_action("global_probe")
    assert not method_requires_gate_action("random_global")
    assert method_requires_gate_action("global_mean_gated")
    assert method_requires_gate_action("bidir_linear")


def test_ungated_global_keeps_direction_on_abstain_route():
    route = {"target_status": None}
    assert target_direction_for_method("global_mean", route) == "global_mean"
    assert target_direction_for_method("global_probe", route) == "global_probe"
    assert target_direction_for_method("random_global", route) == "random_global"
    assert target_direction_for_method("global_mean_gated", route) is None
