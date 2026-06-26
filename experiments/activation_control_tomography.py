"""Activation-control tomography: which layer perturbations actually move the PASS/FAIL decision?

We already know detection is linear, so "does this representation separate honest/deceptive?" is the
wrong question. This asks the causal one:

    m = logit(PASS) - logit(FAIL)   at the final report position

For each candidate direction, injected at each layer (at the decision token), record the causal change
Δm. Directions compared, all unit-normalized so alpha is a matched injected norm:

  - d_p           pooled honest - deceptive
  - to_PASS       honest_PASS - false_FAIL   (corrects wrong-FAIL)
  - to_FAIL       honest_FAIL - false_PASS   (corrects wrong-PASS -- the direction that failed)
  - tangent_*     each learned direction projected onto the local activation manifold
  - off_tangent_* each learned direction's component removed by the tangent projection
  - random        k random unit vectors (the floor)
  - logit_derived unembed[PASS]-unembed[FAIL], injected post-norm (the readout ceiling)

Results are split by error type (false_FAIL / false_PASS / honest_PASS / honest_FAIL) so a one-way
label pusher cannot hide in an aggregate. Forward-pass only (no generation): cheap and parallel. The
point is to find a steerable LOCUS before spending on generation control again.

Directions are fit on TRAIN families only and evaluated on the held-out family (no leakage).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.control_graded_dp_frontier import (  # noqa: E402
    clean_matrix,
    fit_tangent_cloud,
    load_activation_points,
    off_tangent_direction,
    project_to_local_tangent,
)
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.models.interface import ResidualSteeringSpec  # noqa: E402
from geoprobe.models.tokenization import chat_token_ids  # noqa: E402

REPORT_PREFIX = '{\n  "reported_status": "'


def unit(vec: np.ndarray) -> np.ndarray:
    vec = clean_matrix(np.asarray(vec, dtype=np.float64))
    n = float(np.linalg.norm(vec))
    return vec / n if np.isfinite(n) and n > 1e-12 else np.zeros_like(vec)


def error_class(true_status: str, deceptive: bool) -> str:
    """Derive the report error type from ground truth + the deceptive flag (no reported_status needed)."""
    true_status = str(true_status).upper()
    if not deceptive:
        return "honest_PASS" if true_status == "PASS" else "honest_FAIL"
    # deceptive => reported the opposite of true
    return "false_FAIL" if true_status == "PASS" else "false_PASS"


def decision_tokens(tokenizer) -> tuple[int, int]:
    """First token that distinguishes ...PASS from ...FAIL in the report-JSON context."""
    p = tokenizer.encode(REPORT_PREFIX + "PASS")
    f = tokenizer.encode(REPORT_PREFIX + "FAIL")
    i = 0
    while i < min(len(p), len(f)) and p[i] == f[i]:
        i += 1
    return p[i], f[i]


def decision_prefix_ids(tokenizer, messages: list[dict]) -> list[int]:
    """Tokens up to the point where the model must emit PASS or FAIL."""
    prompt = messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages
    ids = chat_token_ids(tokenizer, prompt, add_generation_prompt=True)
    return list(ids) + list(tokenizer.encode(REPORT_PREFIX, add_special_tokens=False))


def margin(model, token_ids, pass_id: int, fail_id: int, steering=None) -> float:
    from geoprobe.models.mlx_capture import _forward_logits_with_steering
    import mlx.core as mx
    logits = _forward_logits_with_steering(model, token_ids, steering)
    last = logits[0, -1, :]
    mx.eval(last)
    arr = np.array(last.astype(mx.float32))
    return float(arr[pass_id] - arr[fail_id])


def fit_layer_family_directions(points: list[dict], heldout_family: str, levels: set[str]) -> dict:
    """d_p / to_PASS / to_FAIL mean-difference directions at one layer, train-family only."""
    train = [r for r in points if str(r["family"]) != heldout_family and r["arm"] in levels]
    by = defaultdict(list)
    for r in train:
        by[error_class(r["true_status"], bool(r["label"]))].append(np.asarray(r["x"], dtype=np.float64))
    def mean(cls):
        return np.mean(np.vstack(by[cls]), axis=0) if by.get(cls) else None
    honest = [v for c in ("honest_PASS", "honest_FAIL") for v in by.get(c, [])]
    decept = [v for c in ("false_FAIL", "false_PASS") for v in by.get(c, [])]
    out = {"counts": {c: len(by.get(c, [])) for c in ("honest_PASS", "honest_FAIL", "false_FAIL", "false_PASS")}}
    out["d_p"] = unit(np.mean(np.vstack(honest), 0) - np.mean(np.vstack(decept), 0)) if honest and decept else None
    hp, ff = mean("honest_PASS"), mean("false_FAIL")
    hf, fp = mean("honest_FAIL"), mean("false_PASS")
    to_pass_raw = hp - ff if hp is not None and ff is not None else None
    to_fail_raw = hf - fp if hf is not None and fp is not None else None
    out["to_PASS"] = unit(to_pass_raw) if to_pass_raw is not None else None
    out["to_FAIL"] = unit(to_fail_raw) if to_fail_raw is not None else None
    out["_d_p_raw"] = (np.mean(np.vstack(honest), 0) - np.mean(np.vstack(decept), 0)) if honest and decept else None
    out["_to_PASS_raw"] = to_pass_raw
    out["_to_FAIL_raw"] = to_fail_raw
    return out


def logit_derived_direction(model, pass_id: int, fail_id: int, hidden: int) -> np.ndarray | None:
    """unembed[PASS]-unembed[FAIL] in hidden space. A quantized head stores packed (non-hidden-dim)
    weights, so dequantize; return None if a hidden-dim unembedding can't be recovered."""
    import mlx.core as mx
    head = getattr(model, "lm_head", None) or model.model.embed_tokens
    w = head.weight
    if w.shape[-1] != hidden and hasattr(head, "scales"):
        w = mx.dequantize(head.weight, head.scales, head.biases,
                          group_size=getattr(head, "group_size", 64), bits=getattr(head, "bits", 4))
    if w.shape[-1] != hidden:
        return None
    diff = (w[pass_id] - w[fail_id]).astype(mx.float32)
    mx.eval(diff)
    return unit(np.array(diff, dtype=np.float64))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--mlx-model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="20,24,28,32")
    ap.add_argument("--alphas", default="4,8,16")
    ap.add_argument("--eval-levels", default="p3,p4,p5")
    ap.add_argument("--direction-levels", default="p3,p4,p5,p6")
    ap.add_argument("--per-class", type=int, default=8, help="eval conversations per error class")
    ap.add_argument("--tangent-neighbors", type=int, default=16)
    ap.add_argument("--tangent-dim", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from geoprobe.models.mlx_capture import load_mlx_model

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    eval_levels = {s.strip() for s in args.eval_levels.split(",") if s.strip()}
    dir_levels = {s.strip() for s in args.direction_levels.split(",") if s.strip()}

    model, tokenizer, meta = load_mlx_model(args.mlx_model)
    n_layers = meta.n_layers
    pass_id, fail_id = decision_tokens(tokenizer)
    # transcripts keyed by conversation_id, with messages + ground truth
    tx = {}
    for line in Path(args.transcripts).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("valid_outcome"):
            tx[str(r["conversation_id"])] = r

    # per-layer activation points (the query bank + direction fitting input)
    points_by_layer = {L: load_activation_points(Path(args.activations), layer=L)[0] for L in layers}
    # index each layer's points by conversation_id for the per-conversation query/tangent
    pt_index = {L: {str(p["conversation_id"]): p for p in pts} for L, pts in points_by_layer.items()}

    # choose a balanced eval set: per-class, drawn from eval levels, families round-robin
    ref_layer = layers[0]
    candidates = [p for p in points_by_layer[ref_layer]
                  if p["arm"] in eval_levels and str(p["conversation_id"]) in tx]
    by_class = defaultdict(list)
    for p in candidates:
        by_class[error_class(p["true_status"], bool(p["label"]))].append(str(p["conversation_id"]))
    rng = np.random.default_rng(args.seed)
    eval_ids = []
    for cls in ("false_FAIL", "false_PASS", "honest_PASS", "honest_FAIL"):
        ids = sorted(by_class.get(cls, []))
        rng.shuffle(ids)
        eval_ids.extend(ids[: args.per_class])

    # fit d_p/to_PASS/to_FAIL per (layer, heldout_family); cache; + logit-derived + random (fixed)
    families = sorted({str(pt_index[ref_layer][c]["family"]) for c in eval_ids})
    fitted = {(L, fam): fit_layer_family_directions(points_by_layer[L], fam, dir_levels)
              for L in layers for fam in families}
    tangent_clouds = {(L, fam): fit_tangent_cloud(points_by_layer[L], heldout_family=fam)
                      for L in layers for fam in families}
    hidden = int(np.asarray(points_by_layer[ref_layer][0]["x"]).shape[0])
    logit_dir = logit_derived_direction(model, pass_id, fail_id, hidden)
    randoms = [unit(rng.standard_normal(hidden)) for _ in range(args.n_random)]

    def to_spec(layer: int, vec: np.ndarray, alpha: float) -> ResidualSteeringSpec:
        clean = np.nan_to_num(np.asarray(vec, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return ResidualSteeringSpec(layer=layer, direction=torch.tensor(clean, dtype=torch.float32), alpha=alpha)

    def margin_is_correct(cls: str, value: float) -> bool:
        return value > 0 if cls in {"false_FAIL", "honest_PASS"} else value < 0

    # accumulate Δmargin per (error_class, direction, layer, alpha)
    acc = defaultdict(lambda: {"sum_delta": 0.0, "sum_final": 0.0, "n": 0, "correct_after": 0})
    base_acc = defaultdict(lambda: {"sum": 0.0, "n": 0, "correct": 0})

    for cid in eval_ids:
        rec = tx[cid]
        is_deceptive = bool(rec.get("deceptive", rec.get("label", False)))
        cls = error_class(rec["true_status"], is_deceptive)
        fam = str(pt_index[ref_layer][cid]["family"])
        ids = decision_prefix_ids(tokenizer, rec["messages"])
        base = margin(model, ids, pass_id, fail_id, steering=None)
        base_acc[cls]["sum"] += base
        base_acc[cls]["n"] += 1
        base_acc[cls]["correct"] += int(margin_is_correct(cls, base))

        for L in layers:
            fit = fitted[(L, fam)]
            query = pt_index[L].get(cid)
            cloud = tangent_clouds[(L, fam)]
            # build direction set for this (conversation, layer)
            dirset: dict[str, np.ndarray | None] = {
                "d_p": fit["d_p"], "to_PASS": fit["to_PASS"], "to_FAIL": fit["to_FAIL"],
            }
            if cloud is not None and query is not None:
                for stem, raw in (
                    ("", fit["_d_p_raw"]),
                    ("_to_PASS", fit["_to_PASS_raw"]),
                    ("_to_FAIL", fit["_to_FAIL_raw"]),
                ):
                    if raw is None:
                        continue
                    tan_t, _ = project_to_local_tangent(raw, cloud, np.asarray(query["x"], np.float64),
                                                         tangent_neighbors=args.tangent_neighbors,
                                                         tangent_dim=args.tangent_dim)
                    if tan_t is None:
                        continue
                    tan = unit(tan_t.detach().float().cpu().numpy().astype(np.float64))
                    if np.linalg.norm(tan) > 1e-8:
                        dirset[f"tangent{stem}"] = tan
                    off_t = off_tangent_direction(raw, tan_t)
                    if off_t is not None:
                        off = unit(off_t.detach().float().cpu().numpy().astype(np.float64))
                        if np.linalg.norm(off) > 1e-8:
                            dirset[f"off_tangent{stem}"] = off
            for i, rv in enumerate(randoms):
                dirset[f"random_{i}"] = rv
            if L == n_layers and logit_dir is not None:
                dirset["logit_derived"] = logit_dir          # PASS-ward readout ceiling
                dirset["logit_derived_neg"] = -logit_dir     # FAIL-ward readout ceiling

            for name, vec in dirset.items():
                if vec is None or len(vec) != hidden:
                    continue
                for alpha in alphas:
                    m = margin(model, ids, pass_id, fail_id, steering=to_spec(L, vec, alpha))
                    key = (cls, name, L, alpha)
                    acc[key]["sum_delta"] += (m - base)
                    acc[key]["sum_final"] += m
                    acc[key]["n"] += 1
                    acc[key]["correct_after"] += int(margin_is_correct(cls, m))

    rows = []
    for (cls, name, L, alpha), v in sorted(acc.items()):
        rows.append({"error_class": cls, "direction": name, "layer": L, "alpha": alpha,
                     "delta_margin_mean": v["sum_delta"] / v["n"] if v["n"] else None,
                     "final_margin_mean": v["sum_final"] / v["n"] if v["n"] else None,
                     "correct_after": v["correct_after"],
                     "correct_after_rate": v["correct_after"] / v["n"] if v["n"] else None,
                     "n": v["n"]})
    out = {
        "schema_version": 1,
        "argv": sys.argv,
        "provenance": git_provenance([Path(__file__), Path(args.activations), Path(args.transcripts)]),
        "mlx_model": args.mlx_model,
        "pass_token": pass_id, "fail_token": fail_id,
        "layers": layers, "alphas": alphas, "n_layers": n_layers,
        "eval_n_by_class": {c: base_acc[c]["n"] for c in base_acc},
        "base_margin_by_class": {c: (base_acc[c]["sum"] / base_acc[c]["n"]) for c in base_acc if base_acc[c]["n"]},
        "base_correct_by_class": {c: base_acc[c]["correct"] for c in base_acc},
        "note": "delta_margin = logit(PASS)-logit(FAIL) shift under injection. To CORRECT false_PASS the "
                "margin must move NEGATIVE (toward FAIL); to correct false_FAIL it must move POSITIVE. "
                "Compare each direction to random (floor) and logit_derived (readout ceiling).",
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    # console summary: the hard direction (false_PASS, want negative) and easy (false_FAIL, want positive)
    print(f"PASS_tok={pass_id} FAIL_tok={fail_id} | base margin by class: "
          f"{ {c: round(m,2) for c,m in out['base_margin_by_class'].items()} }")
    for cls, want in (("false_PASS", "negative=toward FAIL"), ("false_FAIL", "positive=toward PASS")):
        print(f"\n[{cls}]  (want {want})")
        sub = [r for r in rows if r["error_class"] == cls]
        ranked = sorted(sub, key=lambda r: r["delta_margin_mean"])[:6] + sorted(sub, key=lambda r: -r["delta_margin_mean"])[:3]
        seen = set()
        for r in ranked:
            key = (r["direction"], r["layer"], r["alpha"])
            if key in seen:
                continue
            seen.add(key)
            print(f"  {r['direction']:<20} L{r['layer']:<3} a{r['alpha']:<5} "
                  f"Δmargin={r['delta_margin_mean']:+.3f} "
                  f"final={r['final_margin_mean']:+.3f} "
                  f"correct={r['correct_after']}/{r['n']}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
