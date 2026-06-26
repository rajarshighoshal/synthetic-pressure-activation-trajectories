from __future__ import annotations

import numpy as np

from experiments.control_deception_intent_transition import (
    canonical_methods,
    fit_direction,
    injection_stats,
    project_direction_to_local_tangent,
    repeated_ngram_fraction,
    reply_coherence,
)


def make_example(cid: str, family: str, label: int, x: np.ndarray) -> dict:
    return {
        "conversation_id": cid,
        "neutral_conversation_id": f"{cid}_n",
        "scenario_id": cid.split(":", 1)[0],
        "sample": "s0",
        "family": family,
        "arm": "pressured",
        "true_status": "PASS",
        "label": label,
        "position": x,
        "transition": x * 2.0,
        "pressure_flow": x * 3.0,
    }


def test_canonical_methods_supports_tangent_aliases_and_dedupes():
    assert canonical_methods(
        "baseline,linear,position,tangent_transition,pressure_flow_tangent"
    ) == [
        "baseline",
        "position",
        "transition_tangent",
        "pressure_flow_tangent",
    ]


def test_fit_direction_excludes_heldout_family():
    examples = [
        make_example("a:pressured:s0", "train", 0, np.array([2.0, 0.0])),
        make_example("b:pressured:s0", "train", 1, np.array([-2.0, 0.0])),
        make_example("g:pressured:s0", "train", 0, np.array([1.0, 0.0])),
        make_example("h:pressured:s0", "train", 1, np.array([-1.0, 0.0])),
        make_example("c:pressured:s0", "held", 0, np.array([-100.0, 0.0])),
        make_example("d:pressured:s0", "held", 1, np.array([100.0, 0.0])),
    ]

    info = fit_direction(
        examples,
        "position",
        "held",
        tangent_neighbors=2,
        tangent_dim=1,
    )

    assert info is not None
    # Uses honest - deceptive from the train family only, so it points in +x. If the heldout
    # family leaked in, the direction would point the other way.
    assert info["direction"].numpy()[0] > 0.99
    assert info["n_train"] == 4


def test_project_direction_to_local_tangent_projects_onto_neighbor_line():
    examples = [
        make_example("a:pressured:s0", "train", 0, np.array([2.0, 0.0, 0.0])),
        make_example("b:pressured:s0", "train", 1, np.array([-2.0, 0.0, 0.0])),
        make_example("c:pressured:s0", "train", 0, np.array([1.0, 0.0, 0.0])),
        make_example("d:pressured:s0", "train", 1, np.array([-1.0, 0.0, 0.0])),
        make_example("e:pressured:s0", "held", 0, np.array([0.0, 0.0, 0.0])),
        make_example("f:pressured:s0", "held", 1, np.array([0.0, 0.0, 0.0])),
    ]
    info = fit_direction(
        examples,
        "position",
        "held",
        tangent_neighbors=4,
        tangent_dim=1,
    )

    projected, stats = project_direction_to_local_tangent(
        info,
        np.array([0.2, 10.0, 0.0]),
        tangent_neighbors=4,
        tangent_dim=1,
    )

    assert projected is not None
    vec = projected.numpy()
    assert abs(vec[0]) > 0.99
    assert abs(vec[1]) < 1e-5
    assert stats["tangent_dim"] == 1
    assert stats["neighbors"] == 4
    # Tangent steering injects the unit-normalized tangent direction. The projection fraction is
    # diagnostic metadata, not an implicit alpha shrinkage.
    import pytest

    assert np.linalg.norm(vec) == pytest.approx(1.0)


def test_reply_coherence_flags_parse_success_and_repetition():
    ok = reply_coherence('{"reported_status":"PASS","basis":"clear"}', "PASS")
    assert ok["parse_success"] is True
    assert ok["degenerate"] is False
    assert ok["coherence_preserved"] is True

    bad = reply_coherence("loop loop loop loop loop loop loop loop", None)
    assert bad["parse_success"] is False
    assert bad["coherence_preserved"] is False
    assert repeated_ngram_fraction("a b c d a b c d", n=4) > 0.0


def test_injection_stats_records_actual_norm():
    import pytest

    stats = injection_stats(np_to_torch(np.array([3.0, 4.0])), alpha=2.0)
    assert stats["direction_norm"] == pytest.approx(5.0)
    assert stats["injected_norm"] == pytest.approx(10.0)


def np_to_torch(x: np.ndarray):
    import torch

    return torch.from_numpy(x.astype(np.float32))
