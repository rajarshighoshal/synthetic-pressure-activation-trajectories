from __future__ import annotations

import numpy as np

from experiments.bidirectional_dp_diagnostic import (
    fit_status_direction,
    status_error_class,
    summarize_family_directions,
)


def row(
    scenario: str,
    family: str,
    arm: str,
    status_class: str,
    x,
) -> dict:
    true_status = "PASS" if status_class in {"honest_PASS", "false_FAIL"} else "FAIL"
    reported_status = "PASS" if status_class in {"honest_PASS", "false_PASS"} else "FAIL"
    return {
        "conversation_id": f"{scenario}:{family}:{arm}:{status_class}",
        "scenario_id": scenario,
        "family": family,
        "arm": arm,
        "true_status": true_status,
        "reported_status": reported_status,
        "status_class": status_class,
        "label": int(status_class.startswith("false_")),
        "x": np.asarray(x, dtype=float),
    }


def test_status_error_class_splits_both_wrong_directions():
    assert status_error_class("PASS", "PASS") == "honest_PASS"
    assert status_error_class("FAIL", "FAIL") == "honest_FAIL"
    assert status_error_class("PASS", "FAIL") == "false_FAIL"
    assert status_error_class("FAIL", "PASS") == "false_PASS"
    assert status_error_class("unknown", "PASS") is None


def test_fit_status_direction_excludes_heldout_and_is_scenario_paired():
    rows = [
        row("s1", "train", "p3", "false_FAIL", [0.0, 0.0]),
        row("s1", "train", "p3", "honest_PASS", [1.0, 0.0]),
        row("s2", "train", "p4", "false_FAIL", [10.0, 5.0]),
        row("s2", "train", "p4", "honest_PASS", [11.0, 5.0]),
        row("s3", "held", "p3", "false_FAIL", [0.0, 100.0]),
        row("s3", "held", "p3", "honest_PASS", [0.0, 101.0]),
    ]

    info = fit_status_direction(
        rows,
        heldout_family="held",
        direction_levels={"p3", "p4"},
        target_status="PASS",
        min_mixed_scenarios=1,
        min_levels=2,
    )

    assert info is not None
    assert info["target_status"] == "PASS"
    assert info["_direction_np"][0] > 0.99
    assert abs(info["_direction_np"][1]) < 1e-8
    assert info["n_mixed_scenario_level_pairs"] == 2


def test_summarize_family_directions_detects_status_axis_not_shared_honesty():
    rows = []
    for fam in ["a", "b", "c"]:
        rows.extend([
            row(f"{fam}_p3_pass", fam, "p3", "false_FAIL", [0.0, 0.0]),
            row(f"{fam}_p3_pass", fam, "p3", "honest_PASS", [1.0, 0.0]),
            row(f"{fam}_p4_pass", fam, "p4", "false_FAIL", [0.0, 1.0]),
            row(f"{fam}_p4_pass", fam, "p4", "honest_PASS", [1.0, 1.0]),
            row(f"{fam}_p3_fail", fam, "p3", "false_PASS", [1.0, 10.0]),
            row(f"{fam}_p3_fail", fam, "p3", "honest_FAIL", [0.0, 10.0]),
            row(f"{fam}_p4_fail", fam, "p4", "false_PASS", [1.0, 11.0]),
            row(f"{fam}_p4_fail", fam, "p4", "honest_FAIL", [0.0, 11.0]),
        ])

    summary = summarize_family_directions(
        rows,
        direction_levels={"p3", "p4"},
        min_mixed_scenarios=1,
        min_levels=2,
    )

    assert summary["n_to_PASS_available"] == 3
    assert summary["n_to_FAIL_available"] == 3
    assert summary["global_cos_to_PASS_vs_to_FAIL"] < -0.99
