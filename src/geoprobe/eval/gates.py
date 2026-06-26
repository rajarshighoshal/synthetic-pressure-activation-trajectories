"""Synthetic correctness gates — run BEFORE trusting a gated probe on real activations.

A gated probe gets its activation numbers suppressed by the runner unless its
known-answer gate passes in the same run. No probe in this release is gated
(GATED is empty in geoprobe.probes.registry), so this module is just the hook the
runner calls; gate functions for future geometry-aware probes register here.
"""
from __future__ import annotations

# registry of gate functions per gated probe
GATE_FUNCS: dict[str, object] = {}


def gates_for(probe_name: str):
    return GATE_FUNCS.get(probe_name)
