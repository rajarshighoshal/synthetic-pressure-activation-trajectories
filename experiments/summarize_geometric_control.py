"""Summarize geometric-action-response selector results.

Reads the JSON output of ``geometric_action_response_selector.py`` (which itself
reads ``learned_generation_action_policy.py`` output) and produces a readable
policy-comparison report: fixed baselines, context-only geometry, response-aware
geometry, learned selectors, route hybrids, and oracle ceilings.

The critical diagnostic is the **context-only geometric floor**: if context-only
kNN / Mahalanobis clears the route-hybrid baseline, then local activation geometry
alone carries the signal. If it does not, then response margin dominates and the
honest claim is ``probe-guided action search'', not ``geometry wins.''

The report includes per-policy counts, per-status-class breakdowns, paired-gap
bootstrap CIs, and a claim-level verdict block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.trajectory_baselines import git_provenance  # noqa: E402


STATUS_CLASSES = ("false_FAIL", "false_PASS", "honest_PASS", "honest_FAIL")

POLICY_CATEGORIES: dict[str, str] = {}


def _categorise(name: str) -> str:
    if name == "baseline":
        return "floor"
    if name.startswith("fixed_bidir_"):
        return "fixed single-method"
    if name.startswith("fixed_global_") or name.startswith("fixed_random_"):
        return "fixed single-method"
    if name.startswith("route_hybrid_"):
        return "route hybrid"
    if name.startswith("train_best_"):
        return "train-best route"
    if name.startswith("learned_context_"):
        return "learned (context-only)"
    if name.startswith("learned_response_"):
        return "learned (response-aware)"
    if name.startswith("geom_context_"):
        return "geometry kNN (context-only)"
    if name.startswith("geom_response_"):
        return "geometry kNN (response-aware)"
    if name.startswith("margin_argmax_"):
        return "margin argmax"
    return "other"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fmt_ci(point: float | None, ci: list | None) -> str:
    if point is None:
        return "—"
    if ci and len(ci) == 2 and ci[0] is not None and ci[1] is not None:
        return f"{point:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    return f"{point:+.3f}"


def _gap_table(
    policies: dict[str, dict],
    gap_data: dict[str, dict],
    metric: str,
    *,
    reference_names: list[str],
    policy_names: list[str],
) -> list[str]:
    lines: list[str] = []
    header = f"{'':45s}"
    for ref in reference_names:
        header += f"  vs {ref:28s}"
    lines.append(header)
    lines.append("-" * len(header))
    for name in policy_names:
        if name not in policies:
            continue
        row = f"{name:45s}"
        gaps = gap_data.get(name, {})
        for ref in reference_names:
            gap = gaps.get(ref, {}).get(metric, {})
            ci_str = _fmt_ci(gap.get("point"), gap.get("ci95"))
            row += f"  {ci_str:34s}"
        lines.append(row)
    return lines


def render_report(data: dict) -> str:
    lines: list[str] = []
    policies: dict[str, dict] = data.get("policies", {})
    gaps: dict[str, dict] = data.get("paired_gaps", {})

    lines.append("=" * 76)
    lines.append("  Geometric Action-Response Selector — Policy Comparison")
    lines.append("=" * 76)
    lines.append("")

    threshold = data.get("threshold")
    n_rows = data.get("n_candidate_rows")
    n_conv = data.get("n_conversations")
    lines.append(f"candidate rows: {n_rows}  conversations: {n_conv}")
    lines.append(f"threshold: {threshold}")
    lines.append("")

    categories: dict[str, list[str]] = defaultdict(list)
    for name in sorted(policies):
        categories[_categorise(name)].append(name)

    cat_order = [
        "floor",
        "fixed single-method",
        "route hybrid",
        "train-best route",
        "geometry kNN (context-only)",
        "learned (context-only)",
        "margin argmax",
        "geometry kNN (response-aware)",
        "learned (response-aware)",
        "other",
    ]

    for cat in cat_order:
        names = categories.get(cat, [])
        if not names:
            continue
        lines.append(f"── {cat} ──")
        for name in names:
            s = policies[name].get("summary", {})
            if "deceptive_status_fixes" not in s:
                continue
            lines.append(
                f"  {name:45s} status={s['deceptive_status_fixes']:3d}/{s['deceptive_n']}"
                f"  strict={s['deceptive_strict_fixes']:3d}/{s['deceptive_n']}"
                f"  harm={s['honest_status_harms']:2d}/{s['honest_n']}"
                f"  parse={s['parse_success']}/{s['n']}"
            )
            by_cls = s.get("by_status_class") or {}
            cls_strs = []
            for cls in STATUS_CLASSES:
                cd = by_cls.get(cls, {})
                corr = cd.get("status_correct", cd.get("correct_after", "?"))
                n = cd.get("n", 0)
                cls_strs.append(f"{cls}={corr}/{n}")
            if cls_strs:
                lines.append(f"  {'':45s}  ({'  '.join(cls_strs)})")
        lines.append("")

    lines.append("── top policy by strict fixes ──")
    best = data.get("best_geometric_policy")
    if best and best in policies:
        bs = policies[best].get("summary", {})
        lines.append(f"  {best}")
        lines.append(
            f"    strict={bs.get('deceptive_strict_fixes')}/{bs.get('deceptive_n')}"
            f"  status={bs.get('deceptive_status_fixes')}/{bs.get('deceptive_n')}"
            f"  harm={bs.get('honest_status_harms')}/{bs.get('honest_n')}"
        )
    lines.append("")

    lines.append("── paired gaps (bootstrap CIs) vs key references ──")
    references = [
        "fixed_bidir_tangent",
        "route_hybrid_mean_probe",
        "fixed_global_probe_gated",
        "fixed_random_gated",
        "learned_response_rf_strict",
    ]
    key_policies = []
    for name in sorted(policies):
        if name.startswith("geom_") or name.startswith("learned_") or name == "margin_argmax_all":
            key_policies.append(name)
    for metric, mlab in [("strict_fix", "strict fix gap"), ("status_fix", "status fix gap"), ("honest_status_harm", "honest harm gap")]:
        lines.append(f"\n  {mlab}:")
        lines.extend(
            _gap_table(policies, gaps, metric, reference_names=references, policy_names=key_policies)
        )

    lines.append("")
    lines.append("── claim-level verdict ──")

    context_best_strict = 0
    context_best_name = None
    for name in (categories.get("geometry kNN (context-only)", [])
                 + categories.get("learned (context-only)", [])):
        s = policies.get(name, {}).get("summary", {})
        if s.get("deceptive_strict_fixes", 0) > context_best_strict:
            context_best_strict = s["deceptive_strict_fixes"]
            context_best_name = name

    route_hybrid_strict = 0
    for name in categories.get("route hybrid", []):
        s = policies.get(name, {}).get("summary", {})
        route_hybrid_strict = max(route_hybrid_strict, s.get("deceptive_strict_fixes", 0))

    best_strict = 0
    best_name = None
    for name, pol in policies.items():
        s = pol.get("summary", {})
        if s.get("deceptive_strict_fixes", 0) > best_strict:
            best_strict = s["deceptive_strict_fixes"]
            best_name = name

    if context_best_name:
        cleared = context_best_strict > route_hybrid_strict
        lines.append(
            f"  context-only geometric floor: {context_best_name} strict {context_best_strict}"
            f" vs route hybrid {route_hybrid_strict}"
        )
        if cleared:
            lines.append("  → context-only geometry CLEARS the hybrid floor.")
        else:
            lines.append("  → context-only geometry does NOT clear the hybrid floor.")
    else:
        lines.append("  context-only geometric floor: no results available.")

    if best_name and best_name != context_best_name:
        lines.append(
            f"  best overall: {best_name} strict {best_strict}"
        )
        lines.append("  → response margin dominates. Honest claim: probe-guided action search,")
        lines.append("    not pure activation geometry, drives the selector win.")

    if "learned_response_rf_strict" in policies and best_name and best_name.startswith("geom_"):
        geom_strict = policies[best_name]["summary"]["deceptive_strict_fixes"]
        rf_strict = policies["learned_response_rf_strict"]["summary"]["deceptive_strict_fixes"]
        if geom_strict >= rf_strict:
            lines.append(
                f"  geometric >= learned RF: {geom_strict} vs {rf_strict} strict fixes"
            )

    lines.append("")
    for name in sorted(key_policies):
        s = policies[name].get("summary", {})
        chosen = s.get("chosen_methods", {})
        if not chosen:
            continue
        methods_str = ", ".join(f"{m}:{c}" for m, c in sorted(chosen.items()))
        lines.append(f"  {name:45s} chosen = {{{methods_str}}}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="geometric_action_response_selector.py output JSON")
    parser.add_argument("--out", required=True, help="output path for the JSON summary")
    parser.add_argument("--text-out", default=None, help="optional text report path")
    args = parser.parse_args()

    results_path = Path(args.results)
    data = json.loads(results_path.read_text())

    report = render_report(data)
    print(report)

    if args.text_out:
        Path(args.text_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.text_out).write_text(report)

    summary = {
        "schema_version": 1,
        "argv": sys.argv,
        "results": str(results_path.resolve()),
        "results_sha256": file_sha256(results_path),
        "provenance": git_provenance([Path(__file__), results_path]),
        "best_geometric_policy": data.get("best_geometric_policy"),
        "threshold": data.get("threshold"),
        "note": (
            "If context-only geometry clears the route-hybrid floor, then local activation "
            "geometry carries signal. If response-aware dominates, then probe-guided action "
            "search is the load-bearing mechanism."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
