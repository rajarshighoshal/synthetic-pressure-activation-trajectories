"""Thin CLI over geoprobe.eval — no logic here, just argument plumbing.

Examples:
  # per-turn probes on a config's activations:
  python experiments/run.py probe --config configs/synthetic_pressure_llama8b.yaml \
      --probes linear,mlp,pca50,tangent_subspace

  # multi-turn, GroupKFold by scenario/pair id:
  python experiments/run.py probe --config configs/synthetic_pressure_llama8b.yaml --mode groupkfold \
      --labels data/raw/synthetic_pressure/labels.jsonl --probes linear,mlp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geoprobe.eval.runner import run
from geoprobe.probes.registry import known_probes


def cmd_probe(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    probes = args.probes.split(",") if args.probes else known_probes()
    result = run(cfg, probes, mode=args.mode, labels_path=args.labels, n_splits=args.n_splits)
    out = args.out or f"results/eval/run_{cfg['name']}_{args.mode}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=2))

    # human summary (also fully reproducible from the JSON)
    print(f"\n=== {cfg['name']} [{args.mode}] ===")
    if "eval_balance" in result:
        print(f"eval balance: {result['eval_balance']}")
    if "n" in result:
        print(f"n={result['n']} pos={result['pos']}")
    for p in probes:
        r = result["by_probe"][p]
        if r["status"] != "ok":
            print(f"  {p:15s} [{r['family']}] {r['status']}")
        else:
            b = r["best"]
            print(f"  {p:15s} [{r['family']:9s}] AUROC {b['auroc']} @L{b['layer']}")
    print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="run probes on a config's activations")
    p.add_argument("--config", required=True)
    p.add_argument("--probes", default=None, help="comma list; default = all known")
    p.add_argument("--mode", choices=["split", "groupkfold"], default="split")
    p.add_argument("--labels", default=None, help="labels.jsonl (groupkfold mode)")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
