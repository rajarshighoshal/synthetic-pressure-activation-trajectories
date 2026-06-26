from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from experiments import matched_pressure_flow_diagnostic as mpf


def write_turns(
    path: Path,
    include_neutral: bool = True,
    layer: int = 8,
    n_turns: int = 3,
    backend: str = "hf",
    arms_to_include: tuple[str, ...] | None = None,
) -> None:
    """Synthetic matched deception-intent capture.

    Four scenarios across two families; each family has one positive and one negative pressure arm.
    Position is weakly noisy, while matched pressure_flow at turn 2 carries the label.
    """
    rng = np.random.default_rng(7)
    rows = []
    cids, scenarios, families, arms, seeds, turns, phases, labels, statuses, desired = (
        [], [], [], [], [], [], [], [], [], []
    )
    d = 6
    scenario_rows = [
        ("s0", "famA", 0, "PASS"),
        ("s1", "famA", 1, "FAIL"),
        ("s2", "famB", 0, "PASS"),
        ("s3", "famB", 1, "FAIL"),
    ]
    arm_list = arms_to_include or tuple(["neutral", "pressured"] if include_neutral else ["pressured"])
    for si, (sid, family, label, status) in enumerate(scenario_rows):
        for arm in arm_list:
            cid = f"{sid}:{arm}:s0"
            for t in range(n_turns):
                base = rng.normal(scale=0.02, size=d) + t * 0.01
                neutral_step = np.full(d, 0.03 * (t + 1))
                extra = np.full(d, 0.7 if label and arm == "pressured" and t == 2 else 0.0)
                pre = base + rng.normal(scale=0.01, size=d)
                post = pre + neutral_step + extra + rng.normal(scale=0.01, size=d)
                for phase, vec in (("pre_response", pre), ("post_response", post)):
                    cids.append(cid)
                    scenarios.append(sid)
                    families.append(family)
                    arms.append(arm)
                    seeds.append(1000 + si)
                    turns.append(t)
                    phases.append(phase)
                    labels.append(label)
                    statuses.append(status)
                    desired.append("FAIL" if status == "PASS" else "PASS")
                    rows.append(vec)
    acts = torch.tensor(np.vstack(rows), dtype=torch.float32)
    torch.save({
        "activations": {0: acts.clone(), layer: acts},
        "conversation_id": cids,
        "scenario_id": scenarios,
        "family": families,
        "arm": arms,
        "sample_seed": torch.tensor(seeds, dtype=torch.long),
        "turn_index": torch.tensor(turns, dtype=torch.long),
        "phase": phases,
        "deceptive": torch.tensor(labels, dtype=torch.long),
        "true_status": statuses,
        "desired_status": desired,
        "layers": [0, layer],
        "model_name": "synthetic-test-model",
        "backend": backend,
        "device": "test-device",
        "capture": {"capture_method": "synthetic"},
    }, path)


def test_matched_pressure_flow_diagnostic_runs_on_matched_neutral(tmp_path, monkeypatch, capsys):
    turns = tmp_path / "turns.pt"
    out = tmp_path / "mpf.json"
    write_turns(turns, include_neutral=True)

    monkeypatch.setattr(mpf.sys, "argv", [
        "matched_pressure_flow_diagnostic.py",
        "--activations", str(turns),
        "--out", str(out),
        "--layer", "8",
        "--turn", "2",
        "--pca-dim", "3",
        "--bootstrap", "100",
        "--permutations", "20",
        "--seed", "0",
    ])
    mpf.main()
    capsys.readouterr()

    res = json.loads(out.read_text())
    assert res["blocked"] is False
    assert res["n"] == 4 and res["n_pos"] == 2 and res["n_neg"] == 2
    assert res["matched_pair_count"] == 4
    assert res["missing_neutral_count"] == 0
    assert set(res["representations"]) == {"position", "transition", "pressure_flow"}
    assert res["representations"]["pressure_flow"]["direction_alignment"]["n"] == 2
    assert "pressure_flow_minus_position_logistic" in res["paired_gaps"]
    assert 1 <= res["representations"]["pressure_flow"]["label_permutation_null"]["n"] <= 20


def test_matched_pressure_flow_blocks_and_writes_missing_neutral_ids(tmp_path, monkeypatch, capsys):
    turns = tmp_path / "turns.pt"
    out = tmp_path / "blocked.json"
    missing = tmp_path / "missing_neutral_ids.txt"
    write_turns(turns, include_neutral=False)

    monkeypatch.setattr(mpf.sys, "argv", [
        "matched_pressure_flow_diagnostic.py",
        "--activations", str(turns),
        "--out", str(out),
        "--layer", "8",
        "--turn", "2",
        "--missing-neutral-ids-out", str(missing),
    ])
    mpf.main()
    capsys.readouterr()

    res = json.loads(out.read_text())
    assert res["blocked"] is True
    assert res["matched_examples"] == 0
    assert res["missing_neutral_count"] == 4
    assert sorted(missing.read_text().splitlines()) == [
        "s0:neutral:s0",
        "s1:neutral:s0",
        "s2:neutral:s0",
        "s3:neutral:s0",
    ]
    assert "rollout_and_capture_command" in res["minimal_neutral_capture"]


def test_matched_pressure_flow_blocks_mixed_backend_before_matching(tmp_path, monkeypatch, capsys):
    pressure = tmp_path / "pressure.pt"
    neutral = tmp_path / "neutral.pt"
    out = tmp_path / "mixed.json"
    write_turns(pressure, include_neutral=False, backend="hf")
    write_turns(neutral, backend="mlx", arms_to_include=("neutral",))

    monkeypatch.setattr(mpf.sys, "argv", [
        "matched_pressure_flow_diagnostic.py",
        "--activations", str(pressure), str(neutral),
        "--out", str(out),
        "--layer", "8",
        "--turn", "2",
    ])
    mpf.main()
    capsys.readouterr()

    res = json.loads(out.read_text())
    assert res["blocked"] is True
    assert res["matched_examples"] is None
    assert res["reason"].startswith("activation inputs have mixed")
    assert res["mixed_provenance"]["backends"] == ["hf", "mlx"]
