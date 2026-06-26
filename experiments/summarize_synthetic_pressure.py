"""Generate final summary tables for the synthetic pressure experiment.

Reads file-backed outputs only. By default it prefers pair-grouped eval files
when present, because neutral/pressured arms of one synthetic scenario must not
cross train/test folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def best_probe_rows(path: Path) -> list[dict]:
    data = read_json(path)
    rows = []
    for probe, result in data.get("by_probe", {}).items():
        best = result.get("best") or {}
        rows.append(
            {
                "probe": probe,
                "family": result.get("family"),
                "status": result.get("status"),
                "best_auroc": best.get("auroc"),
                "best_layer": best.get("layer"),
            }
        )
    return sorted(rows, key=lambda r: (r["best_auroc"] is None, -(r["best_auroc"] or -1.0)))


def best_trajectory_rows(path: Path, limit: int = 12) -> list[dict]:
    data = read_json(path)
    rows = data.get("summary", [])
    rows = sorted(rows, key=lambda r: (r.get("best_auroc") is None, -(r.get("best_auroc") or -1.0)))
    return rows[:limit]


def label_agreement(opus_path: Path, deepseek_path: Path) -> dict:
    opus = read_json(opus_path)
    deepseek = read_json(deepseek_path)
    opus_map = {
        c["conversation_id"]: {int(t["t"]): t["stance"] for t in c["turns"]}
        for c in opus["judged"]
    }
    deepseek_map = {
        c["conversation_id"]: {int(t["t"]): t["stance"] for t in c["turns"]}
        for c in deepseek["judged"]
    }
    stances = ["accepts", "rejects", "hedges"]
    confusion = {r: {c: 0 for c in stances} for r in stances}
    total = 0
    agree = 0
    exact_conversations = 0
    for cid, turns in opus_map.items():
        conv_total = 0
        conv_agree = 0
        for turn, opus_stance in turns.items():
            deepseek_stance = deepseek_map[cid][turn]
            confusion[opus_stance][deepseek_stance] += 1
            total += 1
            conv_total += 1
            if opus_stance == deepseek_stance:
                agree += 1
                conv_agree += 1
        if conv_total and conv_total == conv_agree:
            exact_conversations += 1
    return {
        "turn_agreement": agree / total,
        "turn_agree": agree,
        "turn_total": total,
        "exact_conversation_agree": exact_conversations,
        "conversation_total": len(opus_map),
        "confusion_opus_rows_deepseek_cols": confusion,
    }


def choose_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("none of these files exists: " + ", ".join(str(p) for p in paths))


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def ci_rows(bootstrap: dict | None) -> list[list[object]]:
    if not bootstrap:
        return []
    rows = []
    for row in bootstrap.get("rows", []):
        name = row.get("name") or row.get("probe")
        rows.append(
            [
                row.get("level"),
                row.get("judge"),
                name,
                row.get("auroc"),
                f"[{row.get('ci95_low'):.4f}, {row.get('ci95_high'):.4f}]",
            ]
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/raw/synthetic_pressure")
    ap.add_argument("--eval-dir", default="results/eval/synthetic_pressure_llama8b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    eval_dir = Path(args.eval_dir)

    paths = {
        "taxonomy_opus": data_dir / "labels_opus_4_8" / "taxonomy.json",
        "taxonomy_deepseek": data_dir / "labels_deepseek_v4_pro_max" / "taxonomy.json",
        "judged_opus": data_dir / "judged_opus_4_8.json",
        "judged_deepseek": data_dir / "judged_deepseek_v4_pro_max.json",
        "per_turn_opus": choose_existing(
            eval_dir / "per_turn_probes_opus_pairgroup.json",
            eval_dir / "per_turn_probes_opus.json",
        ),
        "per_turn_deepseek": choose_existing(
            eval_dir / "per_turn_probes_deepseek_pairgroup.json",
            eval_dir / "per_turn_probes_deepseek.json",
        ),
        "trajectory_opus": choose_existing(
            eval_dir / "trajectory_baselines_opus_pairgroup.json",
            eval_dir / "trajectory_baselines_opus.json",
        ),
        "trajectory_deepseek": choose_existing(
            eval_dir / "trajectory_baselines_deepseek_pairgroup.json",
            eval_dir / "trajectory_baselines_deepseek.json",
        ),
    }
    bootstrap_path = eval_dir / "bootstrap_ci.json"
    if bootstrap_path.exists():
        paths["bootstrap_ci"] = bootstrap_path

    taxonomy_opus = read_json(paths["taxonomy_opus"])
    taxonomy_deepseek = read_json(paths["taxonomy_deepseek"])
    per_turn_opus = read_json(paths["per_turn_opus"])
    per_turn_deepseek = read_json(paths["per_turn_deepseek"])
    trajectory_opus = read_json(paths["trajectory_opus"])
    trajectory_deepseek = read_json(paths["trajectory_deepseek"])
    bootstrap = read_json(bootstrap_path) if bootstrap_path.exists() else None
    agreement = label_agreement(paths["judged_opus"], paths["judged_deepseek"])

    summary = {
        "paths": {k: str(v) for k, v in paths.items()},
        "confidence_intervals": bootstrap,
        "label_agreement": agreement,
        "taxonomy": {
            "opus": taxonomy_opus,
            "deepseek": taxonomy_deepseek,
        },
        "per_turn": {
            "opus": {
                "n": per_turn_opus.get("n"),
                "pos": per_turn_opus.get("pos"),
                "grouping": per_turn_opus.get("grouping"),
                "best_rows": best_probe_rows(paths["per_turn_opus"]),
            },
            "deepseek": {
                "n": per_turn_deepseek.get("n"),
                "pos": per_turn_deepseek.get("pos"),
                "grouping": per_turn_deepseek.get("grouping"),
                "best_rows": best_probe_rows(paths["per_turn_deepseek"]),
            },
        },
        "trajectory": {
            "opus": {
                "n": trajectory_opus.get("n"),
                "n_sf": trajectory_opus.get("n_sf"),
                "n_sc": trajectory_opus.get("n_sc"),
                "grouping": trajectory_opus.get("grouping"),
                "top_rows": best_trajectory_rows(paths["trajectory_opus"]),
            },
            "deepseek": {
                "n": trajectory_deepseek.get("n"),
                "n_sf": trajectory_deepseek.get("n_sf"),
                "n_sc": trajectory_deepseek.get("n_sc"),
                "grouping": trajectory_deepseek.get("grouping"),
                "top_rows": best_trajectory_rows(paths["trajectory_deepseek"]),
            },
        },
    }

    out_json = Path(args.out) if args.out else eval_dir / "final_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    md = [
        "# Synthetic Pressure Final Summary",
        "",
        "All numbers below are read from saved JSON files.",
        "",
        "## Label Agreement",
        "",
        f"Opus vs DeepSeek turn agreement: {agreement['turn_agree']}/{agreement['turn_total']} "
        f"= {agreement['turn_agreement']:.2%}.",
        f"Exact conversation agreement: {agreement['exact_conversation_agree']}/"
        f"{agreement['conversation_total']}.",
        "",
        "## Trajectory Taxonomy",
        "",
        md_table(
            ["judge", "self_correction", "sycophantic_flip", "steadfast_wrong", "steadfast_correct", "accept_turns"],
            [
                [
                    "opus",
                    taxonomy_opus["trajectory_taxonomy_all"]["self_correction"],
                    taxonomy_opus["trajectory_taxonomy_all"]["sycophantic_flip"],
                    taxonomy_opus["trajectory_taxonomy_all"]["steadfast_wrong"],
                    taxonomy_opus["trajectory_taxonomy_all"]["steadfast_correct"],
                    taxonomy_opus["n_accept"],
                ],
                [
                    "deepseek",
                    taxonomy_deepseek["trajectory_taxonomy_all"]["self_correction"],
                    taxonomy_deepseek["trajectory_taxonomy_all"]["sycophantic_flip"],
                    taxonomy_deepseek["trajectory_taxonomy_all"]["steadfast_wrong"],
                    taxonomy_deepseek["trajectory_taxonomy_all"]["steadfast_correct"],
                    taxonomy_deepseek["n_accept"],
                ],
            ],
        ),
        "",
        "## Per-Turn Probes",
        "",
        md_table(
            ["judge", "probe", "family", "AUROC", "layer"],
            [
                [judge, r["probe"], r["family"], r["best_auroc"], r["best_layer"]]
                for judge, block in summary["per_turn"].items()
                for r in block["best_rows"][:8]
            ],
        ),
        "",
        "## Trajectory Probes",
        "",
        md_table(
            ["judge", "feature", "probe", "family", "AUROC", "layer"],
            [
                [judge, r["feature"], r["probe"], r.get("family"), r["best_auroc"], r["best_layer"]]
                for judge, block in summary["trajectory"].items()
                for r in block["top_rows"][:8]
            ],
        ),
        "",
        *(
            [
                "## Confidence Intervals",
                "",
                "Clustered percentile bootstrap over paired scenarios on cross-validated "
                "out-of-fold predictions.",
                "",
                md_table(
                    ["level", "judge", "probe", "AUROC", "95% CI"],
                    ci_rows(bootstrap),
                ),
                "",
            ]
            if bootstrap
            else []
        ),
        "## Interpretation Guardrail",
        "",
        "If linear/MLP remain strongest, report that the signal is strongly recoverable but current "
        "geometry-aware probes do not outperform basic Euclidean baselines.",
        "",
    ]
    out_md = out_json.with_suffix(".md")
    out_md.write_text("\n".join(md))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
