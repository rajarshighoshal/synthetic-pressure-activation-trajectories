"""The backend-agnostic activation-pipeline contract.

Experiment scripts depend only on this interface, never on a concrete backend. On Apple Silicon the
factory hands back an MLX pipeline; on a CUDA server / RunPod it hands back a PyTorch pipeline. Both
implement the same ``capture`` / ``generate`` surface and return a uniform ``{layer: tensor}`` map,
so generation, capture, and downstream probe training are written once and run anywhere.

This module is intentionally dependency-light (only ``torch`` + typing) so both backend modules can
import the contract without pulling in transformers or mlx.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import torch

# Canonical precision names accepted across the codebase, mapped to torch dtypes. fp16 is the
# default everywhere because it matches Apple-Silicon/GPU inference and keeps MLX<->PyTorch
# activations numerically comparable.
TORCH_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_torch_dtype(name: str) -> torch.dtype:
    if name not in TORCH_DTYPES:
        raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(TORCH_DTYPES)}")
    return TORCH_DTYPES[name]


@dataclass(frozen=True)
class PipelineMeta:
    """Describes a loaded model + how it is being run, for provenance and downstream checks."""

    name: str
    backend: str  # "hf" | "mlx"
    device: str  # "cuda" | "mps" | "cpu" | "mlx"
    dtype: str  # effective compute precision, e.g. "float16"
    n_layers: int | None
    hidden_size: int | None


@dataclass(frozen=True)
class ResidualSteeringSpec:
    """A residual-stream nudge applied at a captured layer during generation."""

    layer: int
    direction: torch.Tensor
    alpha: float


SteeringSpec: TypeAlias = ResidualSteeringSpec | list[ResidualSteeringSpec] | tuple[ResidualSteeringSpec, ...]


@runtime_checkable
class ActivationPipeline(Protocol):
    """Common surface every backend provides.

    Layer indexing is identical across backends and matches HF ``output_hidden_states``:
    layer ``0`` is the embedding output, layer ``k`` (``1..n_layers-1``) is the residual stream after
    block ``k-1``, and layer ``n_layers`` is the post-final-norm state.
    """

    meta: PipelineMeta

    def capture(
        self,
        messages: list[dict],
        layers: list[int],
        *,
        add_generation_prompt: bool,
    ) -> dict[int, torch.Tensor]:
        """Last-token residual at each requested layer, as CPU float32 tensors of shape ``(1, hidden)``."""
        ...

    def generate(
        self,
        messages: list[dict],
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
        max_generation_seconds: float | None = None,
        steering: SteeringSpec | None = None,
    ) -> str:
        ...
