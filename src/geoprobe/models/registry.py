from __future__ import annotations

import sys


MODEL_REGISTRY = {
    "llama31_8b_instruct": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama",
        "params": "8B",
        "instruct": True,
        "gated": True,
        "sae_layer": 19,  # Goodfire open SAE target layer (huggingface.co/Goodfire)
    },
}


def resolve_model_name(model_name_or_key: str) -> str:
    return MODEL_REGISTRY.get(model_name_or_key, {}).get("name", model_name_or_key)


def resolve_mlx_model_name(model_name_or_key: str) -> str | None:
    if model_name_or_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_key].get("mlx_name")
    return None


def resolve_backend(model_name_or_key: str, requested: str = "auto") -> str:
    if requested not in {"auto", "hf", "mlx"}:
        raise ValueError(f"unknown backend: {requested!r}")
    if requested != "auto":
        return requested
    if sys.platform == "darwin" and resolve_mlx_model_name(model_name_or_key):
        return "mlx"
    return "hf"


def get_model_meta(model_name_or_key: str) -> dict:
    if model_name_or_key in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_key] | {"key": model_name_or_key}
    return {"key": model_name_or_key, "name": model_name_or_key}
