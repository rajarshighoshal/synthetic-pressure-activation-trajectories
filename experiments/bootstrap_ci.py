"""Bootstrap confidence intervals for the headline per-turn and trajectory AUROCs.

The point estimates in `final_summary.json` are single cross-validated AUROCs and say
nothing about sampling variability. This script attaches a 95% confidence interval to
each headline number using a **clustered percentile bootstrap**: it reproduces the exact
out-of-fold predictions from `run.py` (per-turn) and `trajectory_baselines.py`
(trajectory), then resamples *paired scenarios* (not individual turns/conversations) with
replacement `--n-boot` times and recomputes AUROC each time. Resampling at the pair level
respects the same grouping the cross-validation uses, so the interval is not inflated by
treating the two arms of a scenario as independent.

The point AUROC printed here matches the committed summary exactly; the interval reflects
evaluation-set variability across scenarios, holding the trained probes fixed.

Activations (`turns.pt`) are not shipped; pass `--activations DIR` or regenerate them with
`rollout_sycon.py` first. Example:

    python experiments/bootstrap_ci.py --config configs/synthetic_pressure_llama8b.yaml \
        --activations results/activations/synthetic_pressure_llama8b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.trajectory_baselines import (
    TARGET_TYPES,
    featurize,
    layer_paths,
    load_dataset,
    paired_group_id,
)
from geoprobe.probes.registry import build_probe

# the six numbers reported in the README / first-draft headline tables
JUDGE_LABELS = {
    "Opus 4.8": "data/raw/synthetic_pressure/labels_opus_4_8/labels.jsonl",
    "DeepSeek v4 Pro": "data/raw/synthetic_pressure/labels_deepseek_v4_pro_max/labels.jsonl",
}
HEADLINE = [
    {"level": "per_turn", "judge": "Opus 4.8", "probe": "mlp", "layer": 16},
    {"level": "per_turn", "judge": "DeepSeek v4 Pro", "probe": "mlp", "layer": 16},
    {"level": "trajectory", "judge": "Opus 4.8", "feature": "delta", "probe": "linear", "layer": 16},
    {"level": "trajectory", "judge": "DeepSeek v4 Pro", "feature": "mean", "probe": "linear", "layer": 16},
    {"level": "trajectory", "judge": "Opus 4.8", "feature": "final", "probe": "tangent_subspace", "layer": 19},
    {"level": "trajectory", "judge": "DeepSeek v4 Pro", "feature": "mean", "probe": "tangent_subspace", "layer": 16},
]


def _positive_scores(probe, x: np.ndarray) -> np.ndarray:
    if hasattr(probe, "decision_function"):
        return probe.decision_function(x)
    classes = probe.steps[-1][1].classes_ if hasattr(probe, "steps") else probe.classes_
    return probe.predict_proba(x)[:, list(classes).index(1)]


def per_turn_oof(turns: dict, labels_path: Path, layer: int, probe: str):
    """Out-of-fold per-turn scores, matching geoprobe.eval.runner groupkfold.

    The runner's fit try/except and n_splits clamp are dropped: every headline config has
    both classes in each fold and groups >> 5, so they never fire (the point AUROC matches).
    """
    lab = {}
    for line in labels_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            lab[(r["conversation_id"], int(r["turn_index"]))] = int(r["label"])
    conv, tix = turns["conversation_id"], turns["turn_index"].tolist()
    keep = [i for i, (c, t) in enumerate(zip(conv, tix)) if (c, int(t)) in lab]
    y = np.array([lab[(conv[i], int(tix[i]))] for i in keep])
    groups = np.array([paired_group_id(str(conv[i])) for i in keep])
    x = turns["activations"][layer].numpy()[keep]
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(x, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        p = build_probe(probe).fit(x[tr], y[tr])
        oof[te] = _positive_scores(p, x[te])
    return y, oof, groups


def trajectory_oof(dataset, layer: int, feature: str, probe: str):
    """Out-of-fold trajectory scores, matching trajectory_baselines.cross_validated_auroc."""
    features = featurize(layer_paths(dataset, layer), feature)
    y, groups = dataset.labels, dataset.groups
    n_splits = min(5, int(np.bincount(y, minlength=2).min()), len(set(groups.tolist())))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in splitter.split(features, y, groups):
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            continue
        scaler = StandardScaler()
        p = build_probe(probe).fit(scaler.fit_transform(features[tr]), y[tr])
        oof[te] = _positive_scores(p, scaler.transform(features[te]))
    return y, oof, groups


def clustered_bootstrap(y, oof, groups, n_boot: int, seed: int = 0):
    """Resample paired scenarios with replacement; recompute AUROC each draw."""
    mask = ~np.isnan(oof)
    y, oof, groups = y[mask], oof[mask], groups[mask]
    point = float(roc_auc_score(y, oof))
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in drawn])
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(roc_auc_score(y[idx], oof[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi), len(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/synthetic_pressure_llama8b.yaml")
    ap.add_argument("--activations", default=None, help="dir holding turns.pt (default: config activations dir)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if args.activations:
        config["activations"]["output_dir"] = args.activations
    adir = Path(config["activations"]["output_dir"])
    turns = torch.load(adir / "turns.pt", map_location="cpu", weights_only=False)
    traj_datasets = {j: load_dataset(config, Path(p)) for j, p in JUDGE_LABELS.items()}

    rows = []
    for h in HEADLINE:
        labels_path = Path(JUDGE_LABELS[h["judge"]])
        if h["level"] == "per_turn":
            y, oof, groups = per_turn_oof(turns, labels_path, h["layer"], h["probe"])
            name = f"per-turn {h['probe']}"
        else:
            y, oof, groups = trajectory_oof(traj_datasets[h["judge"]], h["layer"], h["feature"], h["probe"])
            name = f"{h['feature']} + {h['probe']}"
        point, lo, hi, n_ok = clustered_bootstrap(y, oof, groups, args.n_boot, args.seed)
        rows.append({**h, "name": name, "auroc": round(point, 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "n_boot_valid": n_ok})
        print(f"{h['level']:10s} {h['judge']:16s} {name:28s} "
              f"AUROC {point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    result = {
        "method": "clustered percentile bootstrap over paired scenarios on cross-validated "
                  "out-of-fold predictions",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "rows": rows,
    }
    out = Path(args.out) if args.out else Path(config["eval"]["output_dir"]) / "bootstrap_ci.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
