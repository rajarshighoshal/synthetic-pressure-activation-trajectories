"""Matched pressure-flow diagnostic for deception-intent activations.

This is the first "real geometry object" test for deception intent:

    transition_arm(t, L) = post_response_arm(t, L) - pre_response_arm(t, L)
    pressure_flow(t, L) = transition_pressurelike(t, L) - transition_neutral(t, L)

The diagnostic is intentionally narrow. It compares raw pressurelike position, raw pressurelike
transition, and matched pressure-flow at one fixed layer/turn under family-held-out evaluation.
If matched neutral activations are unavailable, it writes a blocker JSON and the exact neutral
conversation IDs needed for a minimal capture instead of falling back to an unpaired test.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.eval_deception_intent_controlled import clean_matrix, paired_gap_ci
from experiments.trajectory_baselines import git_provenance


def arr(values) -> np.ndarray:
    return values.numpy() if torch.is_tensor(values) else np.asarray(values)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def sample_key(conversation_id: str) -> str:
    parts = str(conversation_id).rsplit(":", 2)
    return parts[-1] if len(parts) == 3 else "sample0"


def scenario_key(conversation_id: str) -> str:
    return str(conversation_id).split(":", 1)[0]


def clean_id_lines(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def load_activation_conversations(paths: Iterable[Path], allowed_ids: set[str] | None = None) -> dict:
    conversations: dict[str, dict] = {}
    layers_seen: list[list[int]] = []
    model_names: set[str] = set()
    backends: set[str] = set()
    devices: set[str] = set()
    captures = []

    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        layers = [int(layer) for layer in data["layers"]]
        layers_seen.append(layers)
        model_name = str(data.get("model_name", "unknown"))
        backend = str(data.get("backend", "unknown"))
        device = str(data.get("device", "unknown"))
        model_names.add(model_name)
        backends.add(backend)
        devices.add(device)
        captures.append({
            "path": str(path),
            "model_name": model_name,
            "backend": backend,
            "device": device,
            "capture": data.get("capture", {}),
        })

        cids = np.asarray(data["conversation_id"]).astype(str)
        phases = np.asarray(data["phase"]).astype(str)
        scenarios = np.asarray(data["scenario_id"]).astype(str)
        families = np.asarray(data["family"]).astype(str)
        arms = np.asarray(data["arm"]).astype(str)
        true_status = np.asarray(data["true_status"]).astype(str)
        desired_status = np.asarray(data["desired_status"]).astype(str)
        sample_seed = arr(data["sample_seed"]).astype(int)
        turns = arr(data["turn_index"]).astype(int)
        labels = arr(data["deceptive"]).astype(int)

        for cid in sorted(set(cids.tolist())):
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            if cid in conversations:
                raise ValueError(f"duplicate conversation_id across activation inputs: {cid}")
            mask = cids == cid
            first = int(np.where(mask)[0][0])
            label_values = set(labels[mask].tolist())
            if len(label_values) != 1:
                raise ValueError(f"mixed labels inside {cid}")
            conv = {
                "conversation_id": cid,
                "scenario_id": str(scenarios[first]),
                "sample": sample_key(cid),
                "family": str(families[first]),
                "arm": str(arms[first]),
                "true_status": str(true_status[first]),
                "desired_status": str(desired_status[first]),
                "sample_seed": int(sample_seed[first]),
                "deceptive": int(labels[first]),
                "pre": {},
                "post": {},
            }
            for layer in layers:
                conv["pre"][layer] = {}
                conv["post"][layer] = {}
                for phase, key in (("pre_response", "pre"), ("post_response", "post")):
                    idx = np.where(mask & (phases == phase))[0]
                    idx = idx[np.argsort(turns[idx])]
                    conv[key][layer] = data["activations"][layer][idx].numpy().astype(np.float64)
            conversations[cid] = conv

    if not conversations:
        raise ValueError("no conversations loaded from activation inputs")
    first_layers = layers_seen[0]
    if any(layers != first_layers for layers in layers_seen):
        raise ValueError(f"activation inputs have mismatched layers: {layers_seen}")
    return {
        "conversations": conversations,
        "layers": first_layers,
        "model_names": sorted(model_names),
        "backends": sorted(backends),
        "devices": sorted(devices),
        "captures": captures,
    }


def build_examples(
    conversations: dict[str, dict],
    pressure_arms: set[str],
    layer: int,
    turn: int,
) -> tuple[list[dict], list[str], Counter]:
    by_scenario_sample_arm: dict[tuple[str, str, str], dict] = {}
    for conv in conversations.values():
        by_scenario_sample_arm[(conv["scenario_id"], conv["sample"], conv["arm"])] = conv

    examples = []
    missing_neutral: set[str] = set()
    drops: Counter = Counter()
    for conv in sorted(conversations.values(), key=lambda row: row["conversation_id"]):
        if conv["arm"] not in pressure_arms:
            continue
        neutral = by_scenario_sample_arm.get((conv["scenario_id"], conv["sample"], "neutral"))
        if neutral is None:
            missing_neutral.add(f"{conv['scenario_id']}:neutral:{conv['sample']}")
            drops["missing_neutral"] += 1
            continue
        try:
            pre_p = conv["pre"][layer][turn]
            post_p = conv["post"][layer][turn]
            pre_n = neutral["pre"][layer][turn]
            post_n = neutral["post"][layer][turn]
        except (KeyError, IndexError):
            drops["missing_layer_or_turn"] += 1
            continue
        if pre_p.shape != pre_n.shape or post_p.shape != post_n.shape:
            raise ValueError(
                f"activation shape mismatch for {conv['conversation_id']} vs {neutral['conversation_id']}: "
                f"pre {pre_p.shape}/{pre_n.shape}, post {post_p.shape}/{post_n.shape}"
            )
        if conv["family"] != neutral["family"] or conv["true_status"] != neutral["true_status"]:
            drops["metadata_mismatch"] += 1
            continue
        transition = post_p - pre_p
        neutral_transition = post_n - pre_n
        examples.append({
            "conversation_id": conv["conversation_id"],
            "neutral_conversation_id": neutral["conversation_id"],
            "scenario_id": conv["scenario_id"],
            "sample": conv["sample"],
            "family": conv["family"],
            "arm": conv["arm"],
            "true_status": conv["true_status"],
            "label": int(conv["deceptive"]),
            "position": pre_p,
            "transition": transition,
            "pressure_flow": transition - neutral_transition,
        })
    return examples, sorted(missing_neutral), drops


class ActivationPreprocessor:
    def __init__(self, pca_dim: int):
        self.pca_dim = pca_dim
        self.scale1 = StandardScaler()
        self.components_: np.ndarray | None = None
        self.pca_mean_: np.ndarray | None = None
        self.scale2: StandardScaler | None = None

    @staticmethod
    def project(z: np.ndarray, components: np.ndarray) -> np.ndarray:
        return np.einsum("ij,kj->ik", z, components, optimize=True)

    def fit(self, x: np.ndarray):
        x = clean_matrix(x)
        z = self.scale1.fit_transform(x)
        k = min(self.pca_dim, z.shape[1], len(z) - 1)
        if self.pca_dim > 0 and k >= 1 and z.shape[1] > k:
            self.pca_mean_ = z.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(z - self.pca_mean_, full_matrices=False)
            components = vh[:k].copy()
            pivots = np.argmax(np.abs(components), axis=1)
            signs = np.sign(components[np.arange(k), pivots])
            signs[signs == 0] = 1.0
            self.components_ = components * signs[:, None]
            self.scale2 = StandardScaler().fit(self.project(z - self.pca_mean_, self.components_))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.scale1.transform(clean_matrix(x))
        if self.components_ is not None:
            z = self.project(z - self.pca_mean_, self.components_)
            z = self.scale2.transform(z)
        return clean_matrix(z)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


def preprocess_fit(x_train: np.ndarray, pca_dim: int):
    return ActivationPreprocessor(pca_dim)


def unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-12)


def class_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    if len(set(y.tolist())) < 2:
        return None
    return unit(x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0))


def bootstrap_ci(values: list[float], n_boot: int, seed: int) -> dict | None:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(vals) == 0:
        return None
    if len(vals) == 1:
        return {"mean": float(vals[0]), "ci95": [float(vals[0]), float(vals[0])], "n": 1}
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(vals, size=len(vals), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(vals.mean()), "ci95": [float(lo), float(hi)], "n": int(len(vals))}


def family_folds(families: np.ndarray, y: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    folds = []
    for family in sorted(set(families.tolist())):
        te = np.where(families == family)[0]
        tr = np.where(families != family)[0]
        if len(set(y[tr].tolist())) < 2:
            continue
        folds.append((family, tr, te))
    return folds


def oof_for_representation(
    x: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    pca_dim: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    oof_lr = np.full(len(y), np.nan)
    oof_projection = np.full(len(y), np.nan)
    fold_rows = []
    for family, tr, te in family_folds(families, y):
        pre = preprocess_fit(x[tr], pca_dim)
        xtr = pre.fit_transform(x[tr])
        xte = pre.transform(x[te])
        direction_train = class_direction(xtr, y[tr])
        direction_test = class_direction(xte, y[te])
        alignment = None
        if direction_train is not None and direction_test is not None:
            alignment = float(np.dot(direction_train, direction_test))
        if direction_train is not None:
            oof_projection[te] = xte @ direction_train
        if len(set(y[te].tolist())) >= 2:
            model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
            model.fit(xtr, y[tr])
            oof_lr[te] = model.decision_function(xte)
        fold_rows.append({
            "family": family,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_test_pos": int(y[te].sum()),
            "n_test_neg": int(len(te) - y[te].sum()),
            "direction_alignment": alignment,
        })
    return oof_lr, oof_projection, fold_rows


def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(scores)
    if int(ok.sum()) < 4 or len(set(y[ok].tolist())) < 2:
        return None
    return float(roc_auc_score(y[ok], scores[ok]))


def summarize_representation(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    families: np.ndarray,
    pca_dim: int,
    n_boot: int,
    seed: int,
) -> dict:
    oof_lr, oof_projection, folds = oof_for_representation(x, y, families, pca_dim)
    alignments = [
        float(row["direction_alignment"])
        for row in folds
        if row["direction_alignment"] is not None and np.isfinite(row["direction_alignment"])
    ]
    return {
        "name": name,
        "family_folds": folds,
        "direction_alignment": bootstrap_ci(alignments, n_boot, seed),
        "logistic_family_oof_auroc": auroc(oof_lr, y),
        "projection_family_oof_auroc": auroc(oof_projection, y),
        "_scores": {"logistic": oof_lr, "projection": oof_projection},
    }


def label_permutation_null(
    matrices: dict[str, np.ndarray],
    y: np.ndarray,
    families: np.ndarray,
    pca_dim: int,
    n_perm: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    null = {name: [] for name in matrices}
    for _ in range(n_perm):
        yp = rng.permutation(y)
        for name, x in matrices.items():
            _, _, folds = oof_for_representation(x, yp, families, pca_dim)
            vals = [
                float(row["direction_alignment"])
                for row in folds
                if row["direction_alignment"] is not None and np.isfinite(row["direction_alignment"])
            ]
            if vals:
                null[name].append(float(np.mean(vals)))
    return null


def write_missing_ids(path: Path | None, ids: list[str]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{cid}\n" for cid in ids))


def minimal_neutral_commands(args, missing_ids: list[str]) -> dict:
    ids_path = args.missing_neutral_ids_out or "data/raw/deception_intent/missing_neutral_ids_for_matched_pressure_flow.txt"
    neutral_transcript = "data/raw/deception_intent/rollout_llama8b_matched_neutral_transcript.jsonl"
    neutral_activations = "results/activations/deception_intent_matched_neutral/turns.pt"
    seed_arg = f"--base-seed {args.neutral_base_seed} " if args.neutral_base_seed is not None else ""
    return {
        "missing_neutral_ids_file": str(ids_path),
        "write_ids_note": "The diagnostic writes this file when --missing-neutral-ids-out is provided.",
        "neutral_base_seed": args.neutral_base_seed,
        "neutral_base_seed_note": (
            "Set --neutral-base-seed to the original pressurelike rollout base seed when known; "
            "otherwise rollout_deception_intent.py will use its default seed."
        ),
        "rollout_and_capture_command": (
            ".venv/bin/python experiments/rollout_deception_intent.py "
            "--scenarios data/raw/deception_intent/scenarios.jsonl "
            "--config configs/synthetic_pressure_v2_llama8b.yaml "
            f"--transcript-out {neutral_transcript} "
            f"--activations-out {neutral_activations} "
            "--arms neutral --samples-per-arm 1 "
            f"--conversation-ids-file {ids_path} "
            f"{seed_arg}--checkpoint-every 20 --capture-invalid --resume --device mps"
        ),
        "rerun_diagnostic_command": (
            ".venv/bin/python experiments/matched_pressure_flow_diagnostic.py "
            "--activations <pressurelike_turns.pt> "
            f"{neutral_activations} "
            "--out results/eval/deception_intent_matched_pressure_flow_L8_t2.json "
            "--layer 8 --turn 2 --pca-dim 8"
        ),
        "n_missing_neutral_ids": len(missing_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--turn", type=int, default=2)
    parser.add_argument("--pressure-arms", default="pressured,ambiguous_pressure,conflicted,strategic")
    parser.add_argument("--conversation-ids-file", default=None)
    parser.add_argument("--pca-dim", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--missing-neutral-ids-out", default=None)
    parser.add_argument(
        "--allow-mixed-provenance",
        action="store_true",
        help="allow mixed model/backend/device activation inputs; default blocks to avoid subtracting incompatible spaces",
    )
    parser.add_argument(
        "--neutral-base-seed",
        type=int,
        default=None,
        help="optional base seed for the suggested neutral rollout command; defaults to min pressurelike sample_seed",
    )
    args = parser.parse_args()

    activation_paths = [Path(path) for path in args.activations]
    allowed = clean_id_lines(Path(args.conversation_ids_file)) if args.conversation_ids_file else None
    payload = load_activation_conversations(activation_paths, allowed)
    pressure_arms = set(parse_csv(args.pressure_arms))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mixed_fields = {
        "model_names": payload["model_names"],
        "backends": payload["backends"],
        "devices": payload["devices"],
    }
    mixed_fields = {name: values for name, values in mixed_fields.items() if len(values) > 1}
    if mixed_fields and not args.allow_mixed_provenance:
        blocked = {
            "schema_version": 1,
            "argv": sys.argv,
            "activations": [str(path) for path in activation_paths],
            "provenance": git_provenance([Path(__file__), *activation_paths]),
            "model_names": payload["model_names"],
            "backends": payload["backends"],
            "devices": payload["devices"],
            "captures": payload["captures"],
            "layer": args.layer,
            "turn": args.turn,
            "pressure_arms": sorted(pressure_arms),
            "loaded_conversations": len(payload["conversations"]),
            "matched_examples": None,
            "missing_neutral_count": None,
            "blocked": True,
            "reason": (
                "activation inputs have mixed model/backend/device provenance; refusing to subtract "
                "vectors from potentially incompatible representation spaces"
            ),
            "mixed_provenance": mixed_fields,
            "override": "Pass --allow-mixed-provenance only for explicit debugging, not for reported numbers.",
        }
        out_path.write_text(json.dumps(to_jsonable(blocked), indent=2, sort_keys=True))
        print(f"BLOCKED: mixed activation provenance; saved -> {out_path}", flush=True)
        return

    examples, missing_neutral, drops = build_examples(
        payload["conversations"], pressure_arms, args.layer, args.turn
    )
    write_missing_ids(Path(args.missing_neutral_ids_out) if args.missing_neutral_ids_out else None, missing_neutral)

    base_out = {
        "schema_version": 1,
        "argv": sys.argv,
        "activations": [str(path) for path in activation_paths],
        "provenance": git_provenance([Path(__file__), *activation_paths]),
        "model_names": payload["model_names"],
        "backends": payload["backends"],
        "devices": payload["devices"],
        "captures": payload["captures"],
        "layer": args.layer,
        "turn": args.turn,
        "pressure_arms": sorted(pressure_arms),
        "loaded_conversations": len(payload["conversations"]),
        "matched_examples": len(examples),
        "missing_neutral_count": len(missing_neutral),
        "missing_neutral_conversation_ids": missing_neutral[:200],
        "drops": dict(drops),
    }

    if not examples:
        base_out.update({
            "blocked": True,
            "reason": "no matched pressurelike + neutral examples at the requested layer/turn",
            "minimal_neutral_capture": minimal_neutral_commands(args, missing_neutral),
        })
        out_path.write_text(json.dumps(to_jsonable(base_out), indent=2, sort_keys=True))
        print(f"BLOCKED: no matched examples; saved -> {out_path}", flush=True)
        return

    y = np.asarray([row["label"] for row in examples], dtype=int)
    families = np.asarray([row["family"] for row in examples])
    if len(set(y.tolist())) < 2:
        base_out.update({
            "blocked": True,
            "reason": "matched examples contain only one label class",
            "n_pos": int(y.sum()),
            "n_neg": int(len(y) - y.sum()),
        })
        out_path.write_text(json.dumps(to_jsonable(base_out), indent=2, sort_keys=True))
        print(f"BLOCKED: one label class; saved -> {out_path}", flush=True)
        return

    matrices = {
        "position": clean_matrix(np.vstack([row["position"] for row in examples])),
        "transition": clean_matrix(np.vstack([row["transition"] for row in examples])),
        "pressure_flow": clean_matrix(np.vstack([row["pressure_flow"] for row in examples])),
    }
    summaries = {
        name: summarize_representation(
            name, x, y, families, args.pca_dim, args.bootstrap, args.seed + i
        )
        for i, (name, x) in enumerate(matrices.items())
    }
    null = label_permutation_null(matrices, y, families, args.pca_dim, args.permutations, args.seed + 17)
    for name, vals in null.items():
        real = summaries[name]["direction_alignment"]
        null_arr = np.asarray(vals, dtype=np.float64)
        summaries[name]["label_permutation_null"] = {
            "n": int(len(null_arr)),
            "mean": float(null_arr.mean()) if len(null_arr) else None,
            "ci95": (
                [float(x) for x in np.percentile(null_arr, [2.5, 97.5])]
                if len(null_arr)
                else None
            ),
            "p_ge_real": (
                float((1 + np.sum(null_arr >= real["mean"])) / (len(null_arr) + 1))
                if real is not None and len(null_arr)
                else None
            ),
        }

    scores = {name: summaries[name].pop("_scores") for name in summaries}
    gaps = {
        "pressure_flow_minus_position_logistic": paired_gap_ci(
            scores["pressure_flow"]["logistic"], scores["position"]["logistic"], y, args.bootstrap, args.seed + 31
        ),
        "pressure_flow_minus_transition_logistic": paired_gap_ci(
            scores["pressure_flow"]["logistic"], scores["transition"]["logistic"], y, args.bootstrap, args.seed + 32
        ),
        "pressure_flow_minus_position_projection": paired_gap_ci(
            scores["pressure_flow"]["projection"], scores["position"]["projection"], y, args.bootstrap, args.seed + 33
        ),
        "pressure_flow_minus_transition_projection": paired_gap_ci(
            scores["pressure_flow"]["projection"], scores["transition"]["projection"], y, args.bootstrap, args.seed + 34
        ),
    }

    out = {
        **base_out,
        "blocked": False,
        "pca_dim": args.pca_dim,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "family_balance": dict(Counter(row["family"] for row in examples)),
        "arm_balance": dict(Counter(row["arm"] for row in examples)),
        "status_balance": dict(Counter(row["true_status"] for row in examples)),
        "matched_pair_count": len({(row["scenario_id"], row["sample"]) for row in examples}),
        "representations": summaries,
        "paired_gaps": gaps,
        "note": (
            "Primary object is matched pressure_flow=(post-pre)_pressurelike-(post-pre)_neutral. "
            "Family-held-out folds fit preprocessing and probes on train families only."
        ),
    }
    out_path.write_text(json.dumps(to_jsonable(out), indent=2, sort_keys=True))
    pf = summaries["pressure_flow"]
    print(
        "saved -> "
        f"{out_path} | n={len(y)} pressure_flow align={pf['direction_alignment']} "
        f"auc={pf['logistic_family_oof_auroc']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
