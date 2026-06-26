"""Controlled evaluator for on-policy deception-intent activations.

Compares static position and trajectory/transition geometry while explicitly controlling for
true_status and family. This is a design gate before any 70B run, not a paper result by itself.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoprobe.geometry.trajectories import curve_summary_features, path_stat_features
from experiments.trajectory_baselines import git_provenance


def arr(values) -> np.ndarray:
    return values.numpy() if torch.is_tensor(values) else np.asarray(values)


def clean_matrix(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def load_conversations(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    cids = np.asarray(data["conversation_id"])
    phases = np.asarray(data["phase"])
    turns = arr(data["turn_index"]).astype(int)
    labels = arr(data["deceptive"]).astype(int)
    families = np.asarray(data["family"])
    true_status = np.asarray(data["true_status"])
    layers = [int(x) for x in data["layers"]]
    out = {
        "layers": layers,
        "labels": {},
        "family": {},
        "true_status": {},
        "pre": {layer: {} for layer in layers},
        "post": {layer: {} for layer in layers},
        "model_name": data.get("model_name"),
        "capture": data.get("capture", {}),
    }
    for cid in sorted(set(cids.tolist())):
        mask = cids == cid
        ys = labels[mask]
        if len(set(ys.tolist())) != 1:
            raise ValueError(f"mixed labels inside {cid}")
        out["labels"][cid] = int(ys[0])
        out["family"][cid] = str(families[np.where(mask)[0][0]])
        out["true_status"][cid] = str(true_status[np.where(mask)[0][0]])
        for phase, store in (("pre_response", "pre"), ("post_response", "post")):
            idx = np.where(mask & (phases == phase))[0]
            idx = idx[np.argsort(turns[idx])]
            for layer in layers:
                out[store][layer][cid] = data["activations"][layer][idx].numpy().astype(np.float64)
    return out


def one_hot(values: list[str], vocab: list[str]) -> np.ndarray:
    ix = {v: i for i, v in enumerate(vocab)}
    out = np.zeros((len(values), len(vocab)), dtype=np.float64)
    for row, value in enumerate(values):
        out[row, ix[value]] = 1.0
    return out


def feature(path: np.ndarray, kind: str, horizon: int) -> np.ndarray:
    p = path[: horizon + 1]
    if kind == "position":
        return p[-1]
    if kind == "mean":
        return p.mean(axis=0)
    if kind == "delta":
        return p[-1] - p[0]
    if kind == "path_stats":
        return path_stat_features(p)
    if kind == "curve_summary":
        return curve_summary_features(p)
    raise ValueError(kind)


def preprocess_fit(x_train: np.ndarray, pca_dim: int):
    x_train = clean_matrix(x_train)
    steps = [StandardScaler()]
    k = min(pca_dim, x_train.shape[1], len(x_train) - 1)
    if k >= 1 and x_train.shape[1] > k:
        steps.append(PCA(n_components=k, random_state=0))
        steps.append(StandardScaler())
    return make_pipeline(*steps)


def fit_scores(x: np.ndarray, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]], pca_dim: int) -> np.ndarray:
    x = clean_matrix(x)
    scores = np.full(len(y), np.nan)
    for tr, te in splits:
        if len(set(y[tr].tolist())) < 2:
            continue
        pre = preprocess_fit(x[tr], pca_dim)
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xtr = pre.fit_transform(x[tr])
            xte = pre.transform(x[te])
            model.fit(xtr, y[tr])
            scores[te] = model.decision_function(xte)
    return scores


def fit_scores_with_controls(
    x: np.ndarray,
    controls: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    pca_dim: int,
) -> np.ndarray:
    """Score controls plus a feature block, reducing only the feature block."""
    x = clean_matrix(x)
    controls = clean_matrix(controls)
    scores = np.full(len(y), np.nan)
    for tr, te in splits:
        if len(set(y[tr].tolist())) < 2:
            continue
        x_pre = preprocess_fit(x[tr], pca_dim)
        c_pre = StandardScaler()
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            xtr = x_pre.fit_transform(x[tr])
            xte = x_pre.transform(x[te])
            ctr = c_pre.fit_transform(controls[tr])
            cte = c_pre.transform(controls[te])
            model.fit(np.hstack([ctr, xtr]), y[tr])
            scores[te] = model.decision_function(np.hstack([cte, xte]))
    return scores


def residualized_scores(
    x: np.ndarray,
    controls: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    pca_dim: int,
) -> np.ndarray:
    x = clean_matrix(x)
    controls = clean_matrix(controls)
    scores = np.full(len(y), np.nan)
    for tr, te in splits:
        if len(set(y[tr].tolist())) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            x_scaler = StandardScaler()
            c_scaler = StandardScaler()
            x_train = x_scaler.fit_transform(x[tr])
            x_test = x_scaler.transform(x[te])
            c_train = c_scaler.fit_transform(controls[tr])
            c_test = c_scaler.transform(controls[te])
            reg = Ridge(alpha=1.0)
            reg.fit(c_train, x_train)
            xtr = x_train - reg.predict(c_train)
            xte = x_test - reg.predict(c_test)
            xtr = clean_matrix(xtr)
            xte = clean_matrix(xte)
            pre = preprocess_fit(xtr, pca_dim)
            model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
            model.fit(pre.fit_transform(xtr), y[tr])
            scores[te] = model.decision_function(pre.transform(xte))
    return scores


def metric(scores: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(scores)
    if ok.sum() < 4 or len(set(y[ok].tolist())) < 2:
        return None
    return float(roc_auc_score(y[ok], scores[ok]))


def paired_gap_ci(a: np.ndarray, b: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> dict | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4 or len(set(y[ok].tolist())) < 2:
        return None
    a, b, yy = a[ok], b[ok], y[ok]
    point = roc_auc_score(yy, a) - roc_auc_score(yy, b)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yy), len(yy))
        if len(set(yy[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(yy[idx], a[idx]) - roc_auc_score(yy[idx], b[idx]))
    if not vals:
        return None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "ci95": [float(lo), float(hi)], "n_joint": int(ok.sum())}


def make_splits(y: np.ndarray, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n_splits = min(5, int(np.bincount(y, minlength=2).min()))
    if n_splits < 2:
        return []
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--horizons", default="0,1,2")
    parser.add_argument("--pca-dim", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data = load_conversations(Path(args.activations))
    cids = sorted(data["labels"])
    y = np.asarray([data["labels"][cid] for cid in cids], dtype=int)
    family = [data["family"][cid] for cid in cids]
    status = [data["true_status"][cid] for cid in cids]
    family_vocab = sorted(set(family))
    status_vocab = sorted(set(status))
    controls = np.hstack([one_hot(status, status_vocab), one_hot(family, family_vocab)])
    splits = make_splits(y, args.seed)
    if not splits:
        raise ValueError("not enough examples per class for controlled CV")

    control_scores = fit_scores(controls, y, splits, pca_dim=0)
    selected_layers = [int(x) for x in args.layers.split(",")] if args.layers else data["layers"]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    rows = []
    for layer in selected_layers:
        max_h = min(len(data["pre"][layer][cid]) for cid in cids) - 1
        transitions = {
            cid: data["post"][layer][cid][:max_h + 1] - data["pre"][layer][cid][:max_h + 1]
            for cid in cids
        }
        for horizon in horizons:
            if horizon > max_h:
                continue
            feats = {
                "position": np.vstack([feature(data["pre"][layer][cid], "position", horizon) for cid in cids]),
                "mean": np.vstack([feature(data["pre"][layer][cid], "mean", horizon) for cid in cids]),
                "delta": np.vstack([feature(data["pre"][layer][cid], "delta", horizon) for cid in cids]),
                "path_stats": np.vstack([feature(data["pre"][layer][cid], "path_stats", horizon) for cid in cids]),
                "curve_summary": np.vstack([feature(data["pre"][layer][cid], "curve_summary", horizon) for cid in cids]),
                "transition_final": np.vstack([feature(transitions[cid], "position", horizon) for cid in cids]),
                "transition_stats": np.vstack([feature(transitions[cid], "path_stats", horizon) for cid in cids]),
            }
            feats["geometry_combo"] = np.hstack([feats["path_stats"], feats["curve_summary"], feats["transition_stats"]])
            position_plus = None
            for name, x in feats.items():
                x_scores = fit_scores(x, y, splits, args.pca_dim)
                plus_scores = fit_scores_with_controls(x, controls, y, splits, args.pca_dim)
                resid_scores = residualized_scores(x, controls, y, splits, args.pca_dim)
                if name == "position":
                    position_plus = plus_scores
                rows.append({
                    "layer": layer,
                    "horizon": horizon,
                    "feature": name,
                    "n": int(len(y)),
                    "n_pos": int(y.sum()),
                    "n_neg": int(len(y) - y.sum()),
                    "controls_auroc": metric(control_scores, y),
                    "feature_auroc": metric(x_scores, y),
                    "feature_plus_controls_auroc": metric(plus_scores, y),
                    "residualized_feature_auroc": metric(resid_scores, y),
                    "feature_plus_minus_controls": paired_gap_ci(
                        plus_scores, control_scores, y, args.bootstrap, args.seed
                    ),
                    "residualized_minus_controls": paired_gap_ci(
                        resid_scores, control_scores, y, args.bootstrap, args.seed + 1
                    ),
                    "feature_plus_minus_position_plus": None if position_plus is None else paired_gap_ci(
                        plus_scores, position_plus, y, args.bootstrap, args.seed + 2
                    ),
                })
            if not args.quiet:
                summary = {row["feature"]: row["feature_plus_controls_auroc"] for row in rows[-len(feats):]}
                print(f"L{layer} h={horizon} controls={metric(control_scores, y)} plus={summary}", flush=True)

    out = {
        "schema_version": 1,
        "provenance": git_provenance([Path(__file__), Path(args.activations)]),
        "argv": sys.argv,
        "activations": args.activations,
        "model_name": data["model_name"],
        "capture": data["capture"],
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int(len(y) - y.sum()),
        "status_balance": {v: status.count(v) for v in status_vocab},
        "family_balance": {v: family.count(v) for v in family_vocab},
        "label_by_status": {
            v: {"n": int(sum(s == v for s in status)),
                "pos": int(sum((s == v) and bool(label) for s, label in zip(status, y)))}
            for v in status_vocab
        },
        "note": "Controls are true_status + family one-hot. Use feature_plus_minus_controls and residualized_minus_controls; do not cite point AUROCs alone.",
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
