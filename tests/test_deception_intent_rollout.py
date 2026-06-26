from __future__ import annotations

import random

from experiments.generate_deception_intent_scenarios import FAMILIES, make_scenario
from experiments.rollout_deception_intent import parse_status


def test_scenario_status_is_objective_balanced_and_arms_are_matched():
    rng = random.Random(0)
    passed = make_scenario(FAMILIES[0], 0, "PASS", rng)
    failed = make_scenario(FAMILIES[0], 1, "FAIL", rng)

    assert passed["observed"] <= passed["threshold"]
    assert failed["observed"] > failed["threshold"]
    assert passed["desired_status"] == "FAIL"
    assert failed["desired_status"] == "PASS"
    assert len(passed["arms"]["neutral"]) == len(passed["arms"]["pressured"]) == 4
    assert len(passed["arms"]["ambiguous_pressure"]) == 4
    assert len(passed["arms"]["conflicted"]) == 4
    assert len(passed["arms"]["strategic"]) == 4
    assert passed["arms"]["neutral"][0] == passed["arms"]["pressured"][0]
    assert passed["arms"]["neutral"][0] == passed["arms"]["ambiguous_pressure"][0]


def test_status_parser_prefers_schema_and_rejects_ambiguous_fallback():
    assert parse_status('{"actual_status":"FAIL"}', "knowledge") == "FAIL"
    assert parse_status('prefix {"reported_status": "PASS", "basis":"x"}', "report") == "PASS"
    assert parse_status("PASS", "report") == "PASS"
    assert parse_status("It could PASS or FAIL", "report") is None
    assert parse_status('{"reported_status":"PASS", "basis":"truncated"', "report") is None
