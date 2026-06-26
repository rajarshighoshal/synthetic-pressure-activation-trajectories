"""One runner for every probe experiment. Load -> fit -> score -> AUROC -> JSON.

Two eval modes:
  - "split"      : fixed train.pt / eval.pt (single-turn datasets)
  - "groupkfold" : turns.pt + labels.jsonl, GroupKFold by scenario/pair id (SYCON)

Gated probes only get activation results if their synthetic gates pass in the
SAME run; otherwise the result is recorded as gate-suppressed (no probe in this
release is gated). Every number in the output JSON came from a roc_auc_score call
in this process.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from geoprobe.eval.gates import gates_for
from geoprobe.probes.registry import FAMILY, GATED, build_probe

warnings.filterwarnings("ignore")


def _score(probe, x: np.ndarray) -> np.ndarray:
    if hasattr(probe, "decision_function"):
        return probe.decision_function(x)
    classes = probe.steps[-1][1].classes_ if hasattr(probe, "steps") else probe.classes_
    return probe.predict_proba(x)[:, list(classes).index(1)]


def _auroc_split(probe_name, Xtr, ytr, Xte, yte) -> float | None:
    if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
        return None
    p = build_probe(probe_name).fit(Xtr, ytr)
    return float(roc_auc_score(yte, _score(p, Xte)))


def _auroc_groupkfold(probe_name, X, y, groups, n_splits=5) -> float | None:
    n_splits = min(n_splits, len(set(groups.tolist())))
    if n_splits < 2 or len(set(y.tolist())) < 2:
        return None
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        try:
            p = build_probe(probe_name).fit(X[tr], y[tr])
            oof[te] = _score(p, X[te])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
    m = ~np.isnan(oof)
    if m.sum() == 0 or len(set(y[m].tolist())) < 2:
        return None
    return float(roc_auc_score(y[m], oof[m]))


def _paired_group_id(conversation_id: str) -> str:
    """Keep paired neutral/pressured arms of one synthetic scenario in one fold."""
    stem, sep, suffix = conversation_id.rpartition("_")
    if sep and suffix in {"n", "p"}:
        return stem
    return conversation_id


def _check_gates(probe_names: list[str]) -> dict:
    """Run gates once per gated probe present; return {probe: gate_result}."""
    results = {}
    for p in probe_names:
        if p in GATED:
            fn = gates_for(p)
            results[p] = fn() if fn else {"ALL_PASS": False, "note": "no gate function"}
    return results


def run(config: dict, probe_names: list[str], mode: str,
        labels_path: str | None = None, n_splits: int = 5) -> dict:
    adir = Path(config["activations"]["output_dir"])
    gate_results = _check_gates(probe_names)

    out = {"config": config["name"], "mode": mode, "gates": gate_results, "by_probe": {}}

    if mode == "split":
        tr = torch.load(adir / "train.pt", map_location="cpu", weights_only=False)
        ev = torch.load(adir / "eval.pt", map_location="cpu", weights_only=False)
        ytr, yev = tr["labels"].numpy(), ev["labels"].numpy()
        out["eval_balance"] = {int(v): int((yev == v).sum()) for v in np.unique(yev)}
        out["model_name"] = ev.get("model_name")
        layers = list(tr["activations"].keys())
        for p in probe_names:
            if p in GATED and not gate_results.get(p, {}).get("ALL_PASS"):
                out["by_probe"][p] = {"family": FAMILY[p], "status": "GATE_FAILED", "best": None}
                continue
            per_layer = {int(L): _auroc_split(p, tr["activations"][L].numpy(), ytr,
                                              ev["activations"][L].numpy(), yev) for L in layers}
            best = max(((L, a) for L, a in per_layer.items() if a is not None),
                       key=lambda kv: kv[1], default=(None, None))
            out["by_probe"][p] = {"family": FAMILY[p], "status": "ok",
                                  "best": {"layer": best[0], "auroc": round(best[1], 4) if best[1] else None},
                                  "by_layer": {k: (round(v, 4) if v is not None else None) for k, v in per_layer.items()}}

    elif mode == "groupkfold":
        if not labels_path:
            raise ValueError("groupkfold mode needs labels_path")
        turns = torch.load(adir / "turns.pt", map_location="cpu", weights_only=False)
        lab = {}
        for line in Path(labels_path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                lab[(r["conversation_id"], int(r["turn_index"]))] = int(r["label"])
        conv, tix = turns["conversation_id"], turns["turn_index"].tolist()
        keep = [i for i, (c, t) in enumerate(zip(conv, tix)) if (c, int(t)) in lab]
        y = np.array([lab[(conv[i], int(tix[i]))] for i in keep])
        groups = np.array([_paired_group_id(str(conv[i])) for i in keep])
        out["n"] = len(y)
        out["pos"] = int(y.sum())
        out["model_name"] = turns.get("model_name")
        out["grouping"] = "paired_scenario_id"
        for p in probe_names:
            if p in GATED and not gate_results.get(p, {}).get("ALL_PASS"):
                out["by_probe"][p] = {"family": FAMILY[p], "status": "GATE_FAILED", "best": None}
                continue
            per_layer = {int(L): _auroc_groupkfold(p, turns["activations"][L].numpy()[keep], y, groups, n_splits)
                         for L in turns["layers"]}
            best = max(((L, a) for L, a in per_layer.items() if a is not None),
                       key=lambda kv: kv[1], default=(None, None))
            out["by_probe"][p] = {"family": FAMILY[p], "status": "ok",
                                  "best": {"layer": best[0], "auroc": round(best[1], 4) if best[1] else None},
                                  "by_layer": {k: (round(v, 4) if v is not None else None) for k, v in per_layer.items()}}
    else:
        raise ValueError(f"unknown mode: {mode}")

    return out
