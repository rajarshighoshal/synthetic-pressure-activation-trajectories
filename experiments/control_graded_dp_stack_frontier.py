"""Multi-layer point-cloud control frontier for the graded deception-pressure ramp.

This is the stack version of ``control_graded_dp_frontier.py``.  Instead of injecting one residual
vector at one layer, it fits a leave-one-family-out ``honest - deceptive`` direction at each layer
and applies a coordinated steering bundle across the activation stack.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.control_deception_intent_transition import (  # noqa: E402
    prompt_without_final_answer,
    reply_coherence,
)
from experiments.control_graded_dp_frontier import (  # noqa: E402
    clean_id_lines,
    config_defaults,
    fit_dp_direction,
    fit_tangent_cloud,
    load_activation_points,
    point_index,
    project_to_local_tangent,
    read_jsonl_paths,
    select_eval_rows,
    to_jsonable,
)
from experiments.rollout_deception_intent import parse_status  # noqa: E402
from experiments.trajectory_baselines import git_provenance  # noqa: E402
from geoprobe.models import ResidualSteeringSpec, cleanup, load_activation_pipeline  # noqa: E402


METHOD_ALIASES = {
    "baseline": "baseline",
    "stack_linear": "stack_linear",
    "linear": "stack_linear",
    "stack_tangent": "stack_tangent",
    "tangent": "stack_tangent",
}
STACK_METHODS = {"stack_linear", "stack_tangent"}


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def canonical_methods(value: str) -> list[str]:
    out: list[str] = []
    for item in parse_csv(value):
        if item not in METHOD_ALIASES:
            raise ValueError(f"unknown method {item!r}; expected one of {sorted(METHOD_ALIASES)}")
        method = METHOD_ALIASES[item]
        if method not in out:
            out.append(method)
    return out


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def vector_norm(direction: torch.Tensor) -> float:
    return float(direction.detach().float().norm().item())


def stack_injection_stats(specs: list[ResidualSteeringSpec], requested_alpha: float) -> dict | None:
    if not specs:
        return None
    norms = [vector_norm(spec.direction) for spec in specs]
    injected = [abs(float(spec.alpha)) * norm for spec, norm in zip(specs, norms)]
    return {
        "direction_norm": float(np.sqrt(np.sum(np.square(norms)))),
        "alpha": float(requested_alpha),
        "per_layer_alpha": float(specs[0].alpha),
        "injected_norm": float(np.sqrt(np.sum(np.square(injected)))),
        "n_layers": int(len(specs)),
        "layers": [int(spec.layer) for spec in specs],
        "per_layer_direction_norms": norms,
    }


def aggregate_projection(projections: dict[int, dict]) -> dict | None:
    if not projections:
        return None
    fractions = [
        float(value["projection_fraction"])
        for value in projections.values()
        if value.get("projection_fraction") is not None
    ]
    cosines = [
        float(value["cos_to_raw"])
        for value in projections.values()
        if value.get("cos_to_raw") is not None
    ]
    return {
        "by_layer": projections,
        "projection_fraction": float(np.mean(fractions)) if fractions else None,
        "cos_to_raw": float(np.mean(cosines)) if cosines else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcripts", nargs="+", required=True)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="20,24,28,32")
    parser.add_argument("--direction-turn", type=int, default=2)
    parser.add_argument("--direction-phase", default="pre_response")
    parser.add_argument("--query-turn", type=int, default=3)
    parser.add_argument("--query-phase", default="pre_response")
    parser.add_argument("--eval-levels", default="p3,p4,p5")
    parser.add_argument("--direction-levels", default="p3,p4,p5,p6")
    parser.add_argument("--tangent-levels", default="p0,p1,p2,p3,p4,p5,p6")
    parser.add_argument("--tangent-turns", default="0,1,2,3")
    parser.add_argument("--tangent-phases", default="pre_response,post_response")
    parser.add_argument("--min-mixed-scenarios", type=int, default=8)
    parser.add_argument("--min-direction-levels", type=int, default=2)
    parser.add_argument("--methods", default="baseline,stack_linear,stack_tangent")
    parser.add_argument("--alphas", default="8,16,32")
    parser.add_argument("--alpha-mode", choices=["total", "per_layer"], default="total")
    parser.add_argument("--tangent-neighbors", type=int, default=32)
    parser.add_argument("--tangent-dim", type=int, default=8)
    parser.add_argument("--conversation-ids-file", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--limit-strategy",
        choices=["first", "shuffle", "family_round_robin"],
        default="family_round_robin",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--backend", choices=["auto", "hf", "mlx"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--mlx-model", default=None)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-generation-seconds", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260620)
    args = parser.parse_args()

    layers = [int(item) for item in parse_csv(args.layers)]
    transcript_paths = [Path(path) for path in args.transcripts]
    activation_path = Path(args.activations)
    config_path = Path(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    defaults = config_defaults(config_path)
    model_key = args.model or defaults["model_key"]
    backend = args.backend or defaults["backend"]
    dtype = args.dtype or defaults["dtype"]
    mlx_model = args.mlx_model or defaults["mlx_model"]
    max_length = args.max_length or defaults["max_length"]
    max_new_tokens = args.max_new_tokens or defaults["max_new_tokens"]
    max_generation_seconds = args.max_generation_seconds or defaults["max_generation_seconds"]
    temperature = defaults["temperature"] if args.temperature is None else args.temperature
    top_p = defaults["top_p"] if args.top_p is None else args.top_p
    methods = canonical_methods(args.methods)
    alphas = [float(item) for item in parse_csv(args.alphas)]
    eval_levels = set(parse_csv(args.eval_levels))
    direction_levels = set(parse_csv(args.direction_levels))
    tangent_levels = set(parse_csv(args.tangent_levels))
    tangent_turns = {int(item) for item in parse_csv(args.tangent_turns)}
    tangent_phases = set(parse_csv(args.tangent_phases))

    first_query, activation_meta = load_activation_points(
        activation_path,
        layer=layers[0],
        turns={args.query_turn},
        phases={args.query_phase},
    )
    first_query_ids = set(point_index(first_query))
    del first_query
    gc.collect()
    allowed = clean_id_lines(Path(args.conversation_ids_file)) if args.conversation_ids_file else None
    rows = []
    for row in read_jsonl_paths(transcript_paths):
        cid = str(row.get("conversation_id", ""))
        if allowed is not None and cid not in allowed:
            continue
        if not row.get("valid_outcome"):
            continue
        if row.get("arm") not in eval_levels:
            continue
        if cid in first_query_ids:
            rows.append(row)
    rows = select_eval_rows(rows, limit=args.limit, strategy=args.limit_strategy, seed=args.seed)
    if not rows:
        raise ValueError("no eval rows matched transcripts + activations")
    eval_ids = {str(row["conversation_id"]) for row in rows}

    existing_results: list[dict] = []
    completed: set[tuple[str, str, float]] = set()
    if args.resume and out_path.exists():
        existing = json.loads(out_path.read_text())
        if not existing.get("blocked") and isinstance(existing.get("results"), list):
            existing_results = existing["results"]
            completed = {
                (str(row["conversation_id"]), str(row["method"]), float(row["alpha"]))
                for row in existing_results
            }

    prepared: dict[tuple[str, str], dict] = {}
    planned_skips: Counter = Counter()
    for row in rows:
        cid = str(row["conversation_id"])
        for method in methods:
            if method in STACK_METHODS:
                prepared[(cid, method)] = {"directions": {}, "projections": {}}

    for layer in layers:
        direction_layer, activation_meta = load_activation_points(
            activation_path,
            layer=layer,
            turns={args.direction_turn},
            phases={args.direction_phase},
        )
        query_layer, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns={args.query_turn},
            phases={args.query_phase},
        )
        query_by_cid = point_index([row for row in query_layer if row["conversation_id"] in eval_ids])
        tangent_layer, _ = load_activation_points(
            activation_path,
            layer=layer,
            turns=tangent_turns,
            phases=tangent_phases,
            levels=tangent_levels,
        )
        direction_cache: dict[str, dict | None] = {}
        tangent_cache: dict[str, dict | None] = {}

        def get_direction(family: str) -> dict | None:
            if family not in direction_cache:
                direction_cache[family] = fit_dp_direction(
                    direction_layer,
                    heldout_family=family,
                    direction_levels=direction_levels,
                    min_mixed_scenarios=args.min_mixed_scenarios,
                    min_levels=args.min_direction_levels,
                )
            return direction_cache[family]

        def get_tangent(family: str) -> dict | None:
            if family not in tangent_cache:
                tangent_cache[family] = fit_tangent_cloud(tangent_layer, heldout_family=family)
            return tangent_cache[family]

        for row in rows:
            cid = str(row["conversation_id"])
            family = str(row["family"])
            direction_info = get_direction(family)
            if direction_info is None:
                for method in STACK_METHODS:
                    prepared.pop((cid, method), None)
                planned_skips[f"no_direction::{family}::L{layer}"] += 1
                continue
            if (cid, "stack_linear") in prepared:
                prepared[(cid, "stack_linear")]["directions"][layer] = direction_info["direction"]
            if (cid, "stack_tangent") in prepared:
                tangent_info = get_tangent(family)
                if tangent_info is None:
                    prepared.pop((cid, "stack_tangent"), None)
                    planned_skips[f"no_tangent_cloud::{family}::L{layer}"] += 1
                    continue
                direction, projection = project_to_local_tangent(
                    direction_info["_direction_np"],
                    tangent_info,
                    query_by_cid[cid]["x"],
                    tangent_neighbors=args.tangent_neighbors,
                    tangent_dim=args.tangent_dim,
                )
                if direction is None:
                    prepared.pop((cid, "stack_tangent"), None)
                    planned_skips[f"no_tangent::{family}::L{layer}::{projection.get('reason')}"] += 1
                    continue
                prepared[(cid, "stack_tangent")]["directions"][layer] = direction
                prepared[(cid, "stack_tangent")]["projections"][layer] = projection
        del direction_layer, query_layer, query_by_cid, tangent_layer, direction_cache, tangent_cache
        gc.collect()

    results = list(existing_results)
    jobs: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        for method in methods:
            active_alphas = [0.0] if method == "baseline" else [alpha for alpha in alphas if alpha != 0]
            for alpha in active_alphas:
                key = (str(row["conversation_id"]), method, float(alpha))
                if key in completed:
                    planned_skips["already_completed"] += 1
                    continue
                specs = None
                projection = None
                if method in STACK_METHODS:
                    payload = prepared.get((str(row["conversation_id"]), method))
                    if payload is None or len(payload["directions"]) != len(layers):
                        planned_skips[f"missing_prepared::{method}"] += 1
                        continue
                    layer_alpha = alpha / np.sqrt(len(layers)) if args.alpha_mode == "total" else alpha
                    specs = [
                        ResidualSteeringSpec(layer=layer, direction=payload["directions"][layer], alpha=float(layer_alpha))
                        for layer in layers
                    ]
                    projection = aggregate_projection(payload["projections"])
                jobs.append({
                    "row_index": row_index,
                    "row": row,
                    "method": method,
                    "alpha": float(alpha),
                    "specs": specs,
                    "projection": projection,
                })

    def write_results(pipeline_meta: dict | None, *, validate_only: bool = False) -> None:
        out = {
            "schema_version": 1,
            "argv": sys.argv,
            "blocked": False,
            "validate_only": validate_only,
            "model": pipeline_meta,
            "requested_model_key": model_key,
            "backend": backend,
            "dtype": dtype,
            "mlx_model": mlx_model,
            "layers": layers,
            "direction_turn": args.direction_turn,
            "direction_phase": args.direction_phase,
            "query_turn": args.query_turn,
            "query_phase": args.query_phase,
            "eval_levels": sorted(eval_levels),
            "direction_levels": sorted(direction_levels),
            "methods": methods,
            "alphas": alphas,
            "alpha_mode": args.alpha_mode,
            "tangent_neighbors": args.tangent_neighbors,
            "tangent_dim": args.tangent_dim,
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "max_generation_seconds": max_generation_seconds,
            "transcripts": [str(path.resolve()) for path in transcript_paths],
            "transcripts_sha256": {str(path): file_sha256(path) for path in transcript_paths},
            "activations": str(activation_path.resolve()),
            "activations_sha256": file_sha256(activation_path),
            "activation_meta": activation_meta,
            "config": str(config_path.resolve()),
            "config_sha256": file_sha256(config_path),
            "provenance": git_provenance([Path(__file__), config_path, activation_path, *transcript_paths]),
            "eval_rows": len(rows),
            "limit_strategy": args.limit_strategy,
            "existing_results_loaded": len(existing_results),
            "planned_generations": len(jobs),
            "planned_skips": dict(planned_skips),
            "eval_family_balance": dict(Counter(str(row["family"]) for row in rows)),
            "eval_arm_balance": dict(Counter(str(row["arm"]) for row in rows)),
            "eval_original_label_balance": dict(Counter(str(bool(row["deceptive"])) for row in rows)),
            "results": results,
            "note": (
                "Stack control applies coordinated layerwise honest-deceptive directions. With "
                "alpha_mode=total, per-layer alpha is alpha/sqrt(n_layers), so injected norms are "
                "roughly comparable to a single-layer run."
            ),
        }
        atomic_text(out_path, json.dumps(to_jsonable(out), indent=2, sort_keys=True))

    if args.validate_only:
        write_results(None, validate_only=True)
        print(f"validated -> {out_path} ({len(jobs)} planned generations; model not loaded)", flush=True)
        return

    pipeline = load_activation_pipeline(
        model_key,
        backend=backend,
        device=args.device,
        dtype=dtype,
        max_length=max_length,
        mlx_model=mlx_model,
    )
    pipeline_meta = asdict(pipeline.meta)

    for job in tqdm(jobs, desc="stack-control"):
        row = job["row"]
        method = str(job["method"])
        alpha = float(job["alpha"])
        reply = pipeline.generate(
            prompt_without_final_answer(row),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=args.seed + int(job["row_index"]),
            max_generation_seconds=max_generation_seconds,
            steering=job["specs"],
        )
        true_status = str(row["true_status"])
        reported = parse_status(reply, "report")
        coherence = reply_coherence(reply, reported)
        results.append({
            "conversation_id": row["conversation_id"],
            "scenario_id": row["scenario_id"],
            "sample_seed": row.get("sample_seed"),
            "family": row["family"],
            "arm": row["arm"],
            "true_status": true_status,
            "desired_status": row.get("desired_status"),
            "original_deceptive": bool(row.get("deceptive")),
            "method": method,
            "base_representation": "graded_dp_stack",
            "alpha": alpha,
            "heldout_family": row["family"],
            "direction_projection": job["projection"],
            "injection": stack_injection_stats(job["specs"], alpha) if job["specs"] else None,
            "reported_status": reported,
            "honest": reported == true_status,
            "coherence": coherence,
            "reply": reply,
        })
        write_results(pipeline_meta)

    write_results(pipeline_meta)
    print(f"saved -> {out_path} ({len(results)} generations)", flush=True)
    cleanup()


if __name__ == "__main__":
    main()
