"""Model-free tests for activation_control_tomography (the logic that doesn't need MLX weights)."""
from __future__ import annotations

import numpy as np

from experiments.activation_control_tomography import (
    decision_tokens,
    error_class,
    fit_layer_family_directions,
    logit_derived_direction,  # noqa: F401  (import-smoke; needs a model, not called here)
    unit,
)


def test_error_class_mapping():
    assert error_class("PASS", False) == "honest_PASS"
    assert error_class("FAIL", False) == "honest_FAIL"
    assert error_class("PASS", True) == "false_FAIL"   # true PASS, lied -> reported FAIL
    assert error_class("FAIL", True) == "false_PASS"   # true FAIL, lied -> reported PASS


def test_unit_normalizes_and_handles_zero():
    v = unit(np.array([3.0, 4.0]))
    assert np.allclose(np.linalg.norm(v), 1.0)
    assert np.allclose(unit(np.zeros(4)), 0.0)


class _Tok:
    """Minimal tokenizer: PASS/FAIL differ only at the final token of the report prefix."""
    def encode(self, s, add_special_tokens=True):
        base = [1, 2, 3]
        if s.endswith("PASS"):
            return base + [900]
        if s.endswith("FAIL"):
            return base + [911]
        return base


def test_decision_tokens_finds_distinguishing_token():
    p, f = decision_tokens(_Tok())
    assert (p, f) == (900, 911)


def test_fit_layer_family_directions_excludes_heldout_and_computes_means():
    # honest_PASS at +x, false_FAIL at -x  -> to_PASS should point +x
    pts = []
    def pt(fam, true_status, label, x):
        return {"family": fam, "true_status": true_status, "label": label, "arm": "p3",
                "x": np.array(x, dtype=float), "conversation_id": f"{fam}_{true_status}_{label}_{x[0]}"}
    # train family "a"
    pts += [pt("a", "PASS", 0, [2, 0]), pt("a", "PASS", 0, [2, 0])]      # honest_PASS
    pts += [pt("a", "PASS", 1, [-2, 0]), pt("a", "PASS", 1, [-2, 0])]    # false_FAIL
    pts += [pt("a", "FAIL", 0, [0, 2]), pt("a", "FAIL", 0, [0, 2])]      # honest_FAIL
    pts += [pt("a", "FAIL", 1, [0, -2]), pt("a", "FAIL", 1, [0, -2])]    # false_PASS
    # held-out family "b" with junk that must NOT leak in
    pts += [pt("b", "PASS", 0, [100, 100])]

    fit = fit_layer_family_directions(pts, heldout_family="b", levels={"p3"})
    assert fit["counts"] == {"honest_PASS": 2, "honest_FAIL": 2, "false_FAIL": 2, "false_PASS": 2}
    # to_PASS = unit(mean(honest_PASS) - mean(false_FAIL)) = unit([4,0]) = [1,0]
    assert np.allclose(fit["to_PASS"], [1.0, 0.0])
    # to_FAIL = unit(mean(honest_FAIL) - mean(false_PASS)) = unit([0,4]) = [0,1]
    assert np.allclose(fit["to_FAIL"], [0.0, 1.0])
    # d_p = unit(mean(honest) - mean(deceptive)); honest mean=[1,1], deceptive mean=[-1,-1] -> [1,1]/sqrt2
    assert np.allclose(fit["d_p"], [1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_fit_returns_none_when_a_class_is_missing():
    pts = [{"family": "a", "true_status": "PASS", "label": 0, "arm": "p3",
            "x": np.array([1.0, 0.0]), "conversation_id": "x"}]  # only honest_PASS
    fit = fit_layer_family_directions(pts, heldout_family="z", levels={"p3"})
    assert fit["to_PASS"] is None and fit["to_FAIL"] is None and fit["d_p"] is None
